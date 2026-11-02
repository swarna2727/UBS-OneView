"""Level 3, workflow (deck slides 24-25): did the system behave correctly?

Correct routing, selective regeneration, escalation and checkpoints, read from the append-only run log
(the 'workflow trajectory' method)."""

from typing import Any, Dict, List

from src.evaluation.components import check
from src.evaluation.runner import RunResult

# The agent / checkpoint order each use case must follow (as a subsequence of the observed trajectory).
EXPECTED = {
    "pre_meeting": ["agent:client_profile", "cp:client_profile", "agent:client_limits", "cp:client_limits",
                    "agent:market_intelligence", "cp:market_intelligence", "agent:meeting_brief", "cp:brief_approval", "pdf"],
    "adhoc_request": ["agent:client_profile", "cp:client_profile", "agent:client_limits", "cp:client_limits",
                      "agent:market_intelligence", "cp:market_intelligence", "agent:investment_ideas", "agent:suitability_critic",
                      "cp:suitability_review", "agent:memo_formatter", "cp:advisor_approval", "pdf"],
    "post_meeting": ["agent:client_profile", "agent:client_limits", "agent:meeting_record", "agent:profile_change_check",
                     "cp:record_approval", "pdf"],
}
# Stages a use case must never run (correct routing).
FORBIDDEN = {
    "pre_meeting": {"investment_ideas", "suitability_critic", "memo_formatter", "meeting_record"},
    "adhoc_request": {"meeting_brief", "meeting_record"},
    "post_meeting": {"market_intelligence", "investment_ideas", "suitability_critic", "meeting_brief", "memo_formatter"},
}
# Stages that must run exactly once: selective regeneration never reruns completed upstream work.
UPSTREAM_ONCE = ("client_profile", "client_limits", "market_intelligence", "meeting_record")
FINAL_CHECKPOINTS = {"brief_approval", "advisor_approval", "record_approval"}

def observed(log: List[Dict[str, Any]]) -> List[str]:
    tokens = []
    for e in log:
        ev = e["event"]
        if ev == "agent_output":
            tokens.append(f"agent:{e['stage']}")
        elif ev == "advisor_decision":
            tokens.append(f"cp:{e['stage']}")
        elif ev == "auto_regeneration":
            tokens.append("regen:" + ",".join(str(s) for s in e["slots"]))
        elif ev == "pdf_generated":
            tokens.append("pdf")
        elif ev == "run_ended":
            tokens.append(f"end:{e['outcome']}")
    return tokens

def is_subsequence(expected: List[str], tokens: List[str]) -> bool:
    it = iter(tokens)
    return all(any(t == want for t in it) for want in expected)

def routing(run: RunResult) -> List[Dict[str, Any]]:
    tokens = observed(run.log)
    stages = {t.split(":", 1)[1] for t in tokens if t.startswith("agent:")}
    include_ideas = run.state.get("include_ideas")
    forbidden = set(FORBIDDEN[run.use_case]) - ({"investment_ideas", "suitability_critic"} if include_ideas else set())
    ran_forbidden = sorted(stages & forbidden)
    return [check("workflow:expected-order", is_subsequence(EXPECTED[run.use_case], tokens) if run.completed else True,
                  " > ".join(tokens)),
            check("workflow:no-wrong-agents", not ran_forbidden, f"ran {ran_forbidden}")]

def selective_regeneration(run: RunResult) -> List[Dict[str, Any]]:
    """Only the failed idea is regenerated; passing ideas are untouched; completed upstream stages are not rerun."""
    out = []
    outputs = [e for e in run.log if e["event"] == "agent_output"]
    counts: Dict[str, int] = {}
    for e in outputs:
        counts[e["stage"]] = counts.get(e["stage"], 0) + 1
    reruns = {s: n for s, n in counts.items() if s in UPSTREAM_ONCE and n > 1}
    out.append(check("workflow:no-completed-stage-rerun", not reruns, f"reruns: {reruns}"))
    idea_rounds = [e for e in outputs if e["stage"] == "investment_ideas"]
    regens = [e for e in run.log if e["event"] == "auto_regeneration"]
    ok, detail = True, ""
    for n, regen in enumerate(regens):
        if n + 1 >= len(idea_rounds):
            ok, detail = False, "regeneration without a following idea round"
            break
        before = {i["slot_number"]: i for i in idea_rounds[n]["output"]["ideas"]}
        after = {i["slot_number"]: i for i in idea_rounds[n + 1]["output"]["ideas"]}
        for slot in before:
            same = before[slot] == after[slot]
            if (slot in regen["slots"]) == same:  # a regenerated slot must change; a passing slot must not
                ok, detail = False, f"slot {slot} {'unchanged' if same else 'changed'} but regenerated slots were {regen['slots']}"
    out.append(check("workflow:selective-regeneration", ok, detail or f"{len(regens)} regeneration(s)"))
    return out

def checkpoints_and_escalation(run: RunResult) -> List[Dict[str, Any]]:
    log, out = run.log, []
    events = [e["event"] for e in log]
    decisions = [e for e in log if e["event"] == "advisor_decision"]
    out.append(check("workflow:every-agent-stage-reviewed", bool(decisions) or not run.completed))
    if "pdf_generated" in events:
        first_pdf = events.index("pdf_generated")
        last_cp = max((i for i, e in enumerate(log) if e["event"] == "advisor_decision" and e["stage"] in FINAL_CHECKPOINTS), default=-1)
        approving = log[last_cp]["decision"].get("action") == "approve" if last_cp >= 0 else False
        out.append(check("workflow:no-pdf-before-rm-approval", last_cp >= 0 and last_cp < first_pdf and approving))
    else:
        out.append(check("workflow:no-pdf-before-rm-approval", run.state.get("pdf_path") is None))
    from src.graph.routing import MAX_REGENERATIONS_PER_IDEA
    per_slot: Dict[int, int] = {}
    for e in log:
        if e["event"] == "auto_regeneration":
            for s in e["slots"]:
                per_slot[s] = per_slot.get(s, 0) + 1
    out.append(check("workflow:regeneration-limit-r10", all(n <= MAX_REGENERATIONS_PER_IDEA for n in per_slot.values()), str(per_slot)))
    if run.use_case == "post_meeting":
        out.append(check("workflow:agreement-never-changed-automatically",
                         all(e.get("agreement_changed") is False for e in log if e["event"] == "profile_updates_approved")
                         and not run.state["meeting_record"].agreement_changed))
    return out

def workflow_level(run: RunResult) -> List[Dict[str, Any]]:
    return routing(run) + selective_regeneration(run) + checkpoints_and_escalation(run)
