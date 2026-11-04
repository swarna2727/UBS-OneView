"""OneView evaluation scorecard (deck slides 26-31).

  python -m src.evaluation.scorecard [--out reports/oneview_evaluation.md] [--json reports/oneview_evaluation.json]

Runs the system at the deck's four levels (schema, deterministic, competency, trajectory) against each use case, with
the eight stress tests of slide 29 (plus two further controls), and the performance, cost and control report of
slides 4 and 31. Exit code 1 when any rule-based check fails (judge scores never gate).
"""

import argparse
import json
import os
import tempfile
from typing import Any, Callable, Dict, List

os.environ.setdefault("RUN_LOG_DIR", tempfile.mkdtemp(prefix="ubs_eval_runlog_"))

from src.evaluation import metrics as m
from src.evaluation.components import component_level, output_structure
from src.evaluation.harness import perturb, run_comparison
from src.evaluation.judge import HeuristicJudge, Judge, LLMJudge, mean_score
from src.evaluation.performance import operational_summary, performance_report
from src.evaluation.quality import quality_level
from src.evaluation.runner import RunResult, drive_session
from src.evaluation.stress import EXTRA_CASES, SAMPLES, run_stress_suite
from src.evaluation.trajectory import observed, workflow_level
from src.llm.base import LLMClientProtocol
from src.llm.fake import FakeAdvisoryLLMClient

# Slide 26: the question each level answers, and who evaluates it.
LEVELS = {
    "schema": ("Did the system produce what it was supposed to?", "programmatic"),
    "deterministic": ("Are calculable facts correct?", "code / rules"),
    "competency": ("Is the reasoning good?", "rule proxies here; rubric + LLM-as-judge below"),
    "trajectory": ("Did the workflow behave correctly?", "execution trace"),
}
COMPETENCY_CHECKS = {"quality:no-unsupported-claims", "quality:explainability-complete", "uc1:source-and-date-on-each-point",
                     "uc1:no-unsupported-claims", "uc2:each-idea-has-a-why", "uc3:summary-matches-transcript"}
TRAJECTORY_CHECKS = {"uc2:blocked-products-not-shown", "uc3:no-change-without-rm-approval", "uc3:every-change-checked-by-critic"}
SCHEMA_CHECKS = {"uc2:two-lanes-labelled", "uc2:disclosures-attached", "uc1:limits-summary-present"}

def level_of(check_name: str) -> str:
    if check_name.startswith("schema:") or check_name in SCHEMA_CHECKS:
        return "schema"
    if check_name.startswith("workflow:") or check_name in TRAJECTORY_CHECKS:
        return "trajectory"
    if check_name in COMPETENCY_CHECKS:
        return "competency"
    return "deterministic"  # calculations, sources and freshness, eligibility, rule re-runs

def four_levels(run: RunResult) -> Dict[str, List[Dict[str, Any]]]:
    levels: Dict[str, List[Dict[str, Any]]] = {name: [] for name in LEVELS}
    for c in component_level(run) + output_structure(run) + quality_level(run) + workflow_level(run):
        levels[level_of(c["check"])].append(c)
    return levels
USE_CASE_LABEL = {"pre_meeting": "UC1 Prepare", "adhoc_request": "UC2 Discover", "post_meeting": "UC3 Record"}

def _clients() -> Dict[str, Any]:
    return json.load(open(os.path.join(SAMPLES, "client_profiles.json"), encoding="utf-8"))

def _transcripts() -> Dict[str, str]:
    return json.load(open(os.path.join(SAMPLES, "meeting_transcripts.json"), encoding="utf-8"))

def scenarios(client: str = "demo_sarah") -> Dict[str, Dict[str, Any]]:
    raw = _clients()[client]
    return {
        "pre_meeting": {**raw, "use_case": "pre_meeting", "meeting": {"purpose": "Quarterly review", "date": "2026-10-22"}},
        "adhoc_request": {**raw, "use_case": "adhoc_request", "new_capital_amount": 120000.0},
        "post_meeting": {**raw, "use_case": "post_meeting", "transcript": _transcripts()["demo_sarah"],
                         "meeting": {"purpose": "Quarterly review", "date": "2026-10-22"}},
    }

def judge_run(run: RunResult, judge: Judge) -> Dict[str, Any]:
    """LLM-as-judge: persuasive and grounded ideas (UC2), a faithful record (UC3)."""
    s = run.state
    if run.use_case == "adhoc_request" and s.get("candidate_ideas") and s.get("profile"):
        p = s["profile"]
        ctx = {"facts": [p.calculated_risk_tier, p.goals[0], p.base_currency, "years"], "profile": p.model_dump(include={"goals", "calculated_risk_tier", "time_horizon_months"})}
        verdicts = [judge.judge("idea", i.model_dump(include={"ticker", "thesis", "client_fit_rationale", "key_risks", "cited_evidence_ids"}), ctx)
                    for i in s["candidate_ideas"].ideas]
    elif run.use_case == "post_meeting" and s.get("meeting_record"):
        r = s["meeting_record"]
        lines = [l.model_dump() for l in r.client_said + r.decisions]
        ctx = {"transcript": run.raw_input.get("transcript", ""), "facts": [c.label for c in r.profile_changes] + [a.owner or "" for a in r.actions]}
        verdicts = [judge.judge("record", {"lines": lines, "actions": [a.description for a in r.actions], "email": r.follow_up_email.body}, ctx)]
    else:
        return {}
    return {"judge": judge.name, "mean": mean_score(verdicts), "scores": [v.scores for v in verdicts]}

def evaluate_run(run: RunResult, judge: Judge) -> Dict[str, Any]:
    levels = four_levels(run)
    return {"use_case": run.use_case, "label": USE_CASE_LABEL[run.use_case], "completed": run.completed, "blocked_at": run.blocked_at,
            "levels": levels, "judge": judge_run(run, judge), "performance": performance_report(run), "trajectory": observed(run.log)}

def personalization(llm_factory: Callable[[], LLMClientProtocol], raw: Dict[str, Any]) -> Dict[str, Any]:
    """Do the ideas respond when the client's inputs change? (slide 24, personalisation)"""
    a = drive_session(raw, llm_factory(), "eval_pers_base")
    b = drive_session(perturb(raw), llm_factory(), "eval_pers_changed")
    if not a.completed:
        return {"distance": None, "note": "the base run was blocked: no ideas to compare"}
    if not b.completed:
        # The controls refused to recommend for the changed client (e.g. an idea set that breaches the new risk band).
        # That is a safe, personalised response, but there is no second idea set to measure a distance against.
        return {"distance": None, "responded": f"blocked at {b.blocked_at}",
                "note": "the changed client was stopped by the controls rather than given a new idea set"}
    return {"distance": m.personalization_sensitivity(a.state["candidate_ideas"].ideas, b.state["candidate_ideas"].ideas)}

def run_scorecard(llm_factory: Callable[[], LLMClientProtocol] = FakeAdvisoryLLMClient, judge: Judge = None,
                  include_baseline: bool = True, client: str = "demo_sarah") -> Dict[str, Any]:
    judge = judge or HeuristicJudge()
    raw_runs = [drive_session(raw, llm_factory(), f"eval_{uc}") for uc, raw in scenarios(client).items()]
    runs = [evaluate_run(r, judge) for r in raw_runs]
    pers = personalization(llm_factory, scenarios(client)["adhoc_request"])
    result = {"runs": runs, "personalization": pers, "stress": run_stress_suite(), "extra_stress": run_stress_suite(EXTRA_CASES),
              "operational": operational_summary(raw_runs)}
    if include_baseline:
        result["baseline"] = run_comparison(_clients()["balanced_client"], llm_factory)
    result["failures"] = failures(result)
    return result

def failures(result: Dict[str, Any]) -> List[str]:
    out = []
    for r in result["runs"]:
        if not r["completed"]:
            out.append(f"{r['label']}: scripted RM could not complete the run (blocked at {r['blocked_at']})")
        for level, checks in r["levels"].items():
            out += [f"{r['label']} / {level} / {c['check']}: {c['detail']}" for c in checks if not c["passed"]]
    out += [f"Stress / {s['case']}: expected {s['expected']}, got {s['observed']} ({s['detail']})"
            for s in result["stress"] + result.get("extra_stress", []) if not s["passed"]]
    if result.get("operational", {}).get("unapproved_actions"):
        out.append(f"Control: {result['operational']['unapproved_actions']} consequential action(s) without RM approval")
    return out

def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"

def render_markdown(result: Dict[str, Any], offline: bool) -> str:
    L = ["# UBS OneView evaluation scorecard", "",
         "Four levels (schema, deterministic, competency, trajectory), run against each use case, plus stress tests, "
         "performance, cost and control. Evaluators: programmatic, code / rules, rubric + LLM-as-judge, execution trace.", ""]
    if offline:
        L += ["> Offline run: the scripted model returns fixed outputs, so deterministic checks verify the **system's controls** and the judge "
              "scores come from a text-feature heuristic. Set `OPENROUTER_API_KEY` for a real model and a real judge.", ""]
    L += [f"**Result: {'ALL DETERMINISTIC CHECKS PASS' if not result['failures'] else str(len(result['failures'])) + ' FAILURE(S)'}**", ""]
    for r in result["runs"]:
        L += [f"## {r['label']}", "", f"Run completed: {_mark(r['completed'])}" + (f" (blocked at {r['blocked_at']})" if r["blocked_at"] else ""), ""]
        for level, checks in r["levels"].items():
            question, evaluator = LEVELS[level]
            L += [f"### {level.title()} level", "", f"{question} Evaluator: {evaluator}.", "", "| Check | Result | Detail |", "|---|---|---|"]
            L += [f"| {c['check']} | {_mark(c['passed'])} | {str(c['detail'])[:110].replace('|', '/')} |" for c in checks]
            L.append("")
            if level == "competency" and r["judge"]:
                L += [f"Rubric + LLM-as-judge ({r['judge']['judge']}): mean {r['judge']['mean']} / 5 (reported, does not gate)", ""]
    L += ["## Personalisation", "", f"Idea-set distance after changing the client's inputs (shorter horizon, Conservative): {result['personalization'].get('distance')}"
          + (f" ({result['personalization']['note']})" if result["personalization"].get("note") else ""), "",
          "## Stress tests", "", "We deliberately test the cases where the obvious answer is wrong.", "",
          "| Case | Required outcome | Observed | Result |", "|---|---|---|---|"]
    L += [f"| {s['case']} | {s['expected']} | {s['observed']} | {_mark(s['passed'])} |" for s in result["stress"]]
    if result.get("extra_stress"):
        L += ["", "Further controls:", "", "| Case | Required outcome | Observed | Result |", "|---|---|---|---|"]
        L += [f"| {s['case']} | {s['expected']} | {s['observed']} | {_mark(s['passed'])} |" for s in result["extra_stress"]]
    L += ["", "## Performance and cost", "",
          "| Use case | End-to-end (ms) | LLM calls | Cost (US$) | Research | Ideas | Checks | Record | Regen rounds | Idea generations saved |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["runs"]:
        p = r["performance"]; b = p["latency_by_bucket_ms"]
        L.append(f"| {r['label']} | {p['end_to_end_ms']} | {p['llm_calls']} | {p['cost_usd'] if p['cost_usd'] is not None else 'n/a (no token usage)'} | "
                 f"{b['Research']} | {b['Ideas']} | {b['Checks']} | {b['Record']} | {p['regeneration_rounds']} | {p['idea_generations_saved']} |")
    L += ["", "Selective regeneration avoids rerunning successful stages: only the failed idea is regenerated and the whole set is re-checked.", ""]
    op = result.get("operational")
    if op:
        L += ["## Control and recovery", "",
              "| Measure | Value |", "|---|---|",
              f"| Runs without manual recovery (intended RM checkpoints excluded) | {op['runs_without_manual_recovery']} of {op['runs']} "
              f"({op['runs_without_manual_recovery_pct']}%) |",
              f"| Unapproved action rate (PDFs and profile updates without an RM approval just before) | {op['unapproved_actions']} of "
              f"{op['consequential_actions']} ({op['unapproved_action_rate_pct']}%) |",
              "", "RM time saved per case needs a timed manual-preparation baseline; it is not measured by this harness.", ""]
    if "baseline" in result:
        from src.evaluation.harness import render_markdown as baseline_md
        L += [baseline_md({"balanced_client": result["baseline"]}, offline).replace("# Baseline vs proposed architecture", "## Hypothesis: specialists vs one financial-advisor LLM", 1), ""]
    if result["failures"]:
        L += ["## Failures", ""] + [f"- {f}" for f in result["failures"]] + [""]
    return "\n".join(L)

def main() -> None:
    from src.llm.openrouter import default_llm_client
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/oneview_evaluation.md")
    parser.add_argument("--json", default=None)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()
    offline = isinstance(default_llm_client(), FakeAdvisoryLLMClient)
    factory = FakeAdvisoryLLMClient if offline else default_llm_client
    judge: Judge = HeuristicJudge() if offline else LLMJudge(default_llm_client())
    result = run_scorecard(factory, judge, include_baseline=not args.no_baseline)
    report = render_markdown(result, offline)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(report)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({k: v for k, v in result.items() if k != "baseline"}, open(args.json, "w"), indent=2, default=str)
    print(report)
    raise SystemExit(1 if result["failures"] else 0)

if __name__ == "__main__":
    main()
