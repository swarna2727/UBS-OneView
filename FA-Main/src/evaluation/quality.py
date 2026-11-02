"""Level 2, output quality (deck slides 24-25): were the outputs good? Plus the per-use-case questions of slide 25.

UC1 briefing: is every point sourced?  UC2 opportunities: explained and allowed?  UC3 record: is the record faithful?
All deterministic over a finished run; the judge (judge.py) adds the qualities rules cannot score."""

import re
from typing import Any, Dict, List, Set

from src.evaluation import metrics as m
from src.evaluation.components import check
from src.evaluation.runner import RunResult
from src.tools.risk_engine import suitability_rule_engine
from src.tools.transcript_tools import index_lines, lines_support_amounts, transcript_parser
from src.tools.universe import approved_product_universe_tool

# --------------------------------------------------------------------------- shared output-quality checks

def fit_and_eligibility(run: RunResult) -> List[Dict[str, Any]]:
    """Delivered ideas pass an independent re-run of the rule engine, including client limits (E1)."""
    s = run.state
    ideas, profile, brief, om = s.get("candidate_ideas"), s.get("profile"), s.get("research_brief"), s.get("offer_map")
    if not (run.completed and ideas and profile and brief):
        return []
    approved = {p.ticker.upper() for p in approved_product_universe_tool()}
    engine = suitability_rule_engine(profile, ideas.ideas, brief, approved, offer_map=om)
    hard = [r for r in engine["portfolio_results"] if r.outcome in ("FAIL", "BLOCK")]
    hard += [r for rs in engine["idea_results"].values() for r in rs if r.outcome in ("FAIL", "BLOCK")]
    e1 = [r for rs in engine["idea_results"].values() for r in rs if r.rule_id == "E1"]
    return [check("quality:delivered-ideas-pass-rules", not hard, "; ".join(f"{r.rule_id}:{r.message}" for r in hard)),
            check("quality:delivered-ideas-eligible", bool(e1) and all(r.outcome == "PASS" for r in e1))]

def grounding_and_explainability(run: RunResult) -> List[Dict[str, Any]]:
    s = run.state
    ideas, brief = s.get("candidate_ideas"), s.get("research_brief")
    if not (run.completed and ideas and brief):
        return []
    return [check("quality:no-unsupported-claims", m.unsupported_claims(ideas.ideas, brief) == 0),
            check("quality:explainability-complete", m.explainability(ideas.ideas) == 1.0, f"{m.explainability(ideas.ideas)}")]

# --------------------------------------------------------------------------- UC1: is every point sourced?

def uc1_checks(run: RunResult) -> List[Dict[str, Any]]:
    s = run.state
    brief, research = s.get("meeting_brief"), s.get("research_brief")
    if not (brief and research):
        return [check("uc1:briefing-produced", False, "no briefing in the run")]
    known = {x.source_id for x in research.validated_sources}
    findings = research.findings
    points_sourced = all(f.cited_source_ids and f.publication_dates and set(f.cited_source_ids) <= known for f in findings)
    developments_cite = all(re.search(r"\((SRC|RM-NEWS)[^)]*\)", d) for d in brief.market_developments)
    forecast = [d for d in brief.market_developments if re.search(r"\b(will (rise|fall|reach)|forecast|price target)\b", d, re.I)]
    checks = {c.rule_id: c.outcome for c in brief.compliance_checks}
    return [
        check("uc1:source-and-date-on-each-point", points_sourced and developments_cite),
        check("uc1:impact-matches-the-data", brief.portfolio_impact == research.portfolio_impact and bool(brief.portfolio_impact)),
        check("uc1:no-unsupported-claims", checks.get("S1") == "PASS" and not forecast, f"S1={checks.get('S1')} forecasts={len(forecast)}"),
        check("uc1:limits-summary-present", bool(brief.limits_summary)),
    ]

# --------------------------------------------------------------------------- UC2: explained and allowed?

def blocked_tickers_seen(run: RunResult) -> Set[str]:
    """Tickers of every idea the critic ever failed on client limits (E1) during the run."""
    id_to_ticker: Dict[str, str] = {}
    for e in run.log:
        if e["event"] == "agent_output" and e.get("stage") == "investment_ideas":
            id_to_ticker.update({i["idea_id"]: i["ticker"] for i in e["output"]["ideas"]})
    blocked: Set[str] = set()
    for e in run.log:
        if e["event"] == "rule_results":
            for idea_id, results in (e.get("ideas") or {}).items():
                if any(r["rule_id"] == "E1" and r["outcome"] == "FAIL" for r in results) and idea_id in id_to_ticker:
                    blocked.add(id_to_ticker[idea_id])
    return blocked

def never_shown(run: RunResult, tickers: Set[str]) -> List[str]:
    """Blocked tickers that appear anywhere the RM could see (review payloads and workspace views)."""
    seen = " ".join(run.visible)
    return sorted(t for t in tickers if re.search(rf"\b{re.escape(t)}\b", seen))

def uc2_checks(run: RunResult) -> List[Dict[str, Any]]:
    s = run.state
    ideas, om = s.get("candidate_ideas"), s.get("offer_map")
    if not ideas:
        return [check("uc2:ideas-produced", False, "no ideas in the run")]
    each_has_why = all(i.thesis.strip() and i.client_fit_rationale.strip() and i.key_risks and i.cited_evidence_ids for i in ideas.ideas)
    lanes = [i.lane for i in sorted(ideas.ideas, key=lambda i: i.slot_number)]
    blocked = blocked_tickers_seen(run) | (set(om.blocked_tickers) & {i.ticker for i in ideas.ideas} if om else set())
    leaked = never_shown(run, blocked)
    return [
        check("uc2:each-idea-has-a-why", each_has_why),
        check("uc2:blocked-products-not-shown", not leaked, f"leaked: {leaked}; blocked ideas seen: {sorted(blocked)}"),
        check("uc2:two-lanes-labelled", lanes == ["ALIGNED", "ALIGNED", "EXPLORATORY"], str(lanes)),
        check("uc2:disclosures-attached", all(i.required_disclosures for i in ideas.ideas)),
    ]

# --------------------------------------------------------------------------- UC3: is the record faithful?

def uc3_checks(run: RunResult) -> List[Dict[str, Any]]:
    s = run.state
    record = s.get("meeting_record")
    if not record:
        return [check("uc3:record-produced", False, "no record in the run")]
    lines = index_lines(transcript_parser(run.raw_input.get("transcript", "")))
    summary = record.client_said + record.decisions
    faithful = all(l.source_lines and all(n in lines for n in l.source_lines)
                   and not lines_support_amounts(l.text, [lines[n] for n in l.source_lines]) for l in summary)
    delivered = run.completed
    actions_ok = all(a.owner and a.due_date for a in record.actions) if delivered else all(a.status == "FLAGGED" or (a.owner and a.due_date) for a in record.actions)
    approved_ids = {c.change_id for c in record.profile_changes if c.decision == "APPROVED"}
    updates = {u["change_id"] for u in (s.get("approved_profile_updates") or [])}
    reg_before = (run.raw_input.get("regulatory_profile") or {})
    reg_after = ((s.get("raw_input") or {}).get("regulatory_profile") or {})
    # An update may exist only after an approving RM decision; the log shows the order.
    order = [e["event"] for e in run.log]
    after_approval = ("profile_updates_approved" not in order) or (
        "advisor_decision" in order[:order.index("profile_updates_approved")])
    return [
        check("uc3:summary-matches-transcript", faithful and bool(summary)),
        check("uc3:actions-have-owner-and-date", actions_ok),
        check("uc3:no-change-without-rm-approval", updates <= approved_ids and after_approval and not record.agreement_changed
              and reg_before == reg_after, f"updates={sorted(updates)} approved={sorted(approved_ids)}"),
        check("uc3:every-change-checked-by-critic", all(c.check is not None for c in record.profile_changes)),
    ]

USE_CASE_CHECKS = {"pre_meeting": uc1_checks, "adhoc_request": uc2_checks, "post_meeting": uc3_checks}

def quality_level(run: RunResult) -> List[Dict[str, Any]]:
    return fit_and_eligibility(run) + grounding_and_explainability(run) + USE_CASE_CHECKS[run.use_case](run)
