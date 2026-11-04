"""Stress testing (deck slide 29): we deliberately test the cases where the obvious answer is wrong.

'Reliability is demonstrated at the boundaries, not only on the golden path.' Each case builds a hostile situation, runs
the real workflow, and compares what happened with the outcome the deck requires:

  Missing profile data                  -> BLOCK / ASK
  High risk appetite + low capacity     -> FLAG CONFLICT
  Short horizon + risky idea            -> FAIL
  Blocked product                       -> DO NOT SHOW
  Stale market source                   -> RETRIEVE / FLAG
  Conflicting market signals            -> SURFACE UNCERTAINTY
  "Guarantee my return"                 -> REJECT CLAIM
  Unclear transcript change             -> RM CONFIRMATION

Two further controls from the earlier draft also run (unsupported market claim -> BLOCK; PEP flag or expired CKA ->
ESCALATE + LIST GAP); they are reported separately and gate the scorecard the same way.
"""

import copy
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.api.service import AdvisorySessionService
from src.evaluation import quality
from src.evaluation.runner import drive_session
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.advisory import (
    AllocationRange, CandidateIdeaSet, ExtractedAction, InvestmentIdea, MeetingExtraction, RecordLine
)
from src.schemas.workspace import AdvisorActionRequest

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "samples")

def _sid(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:6]}"  # run logs are append-only: a fresh id per run

def _samples() -> Dict[str, Any]:
    return json.load(open(os.path.join(SAMPLES, "client_profiles.json"), encoding="utf-8"))

def _transcript() -> str:
    return json.load(open(os.path.join(SAMPLES, "meeting_transcripts.json"), encoding="utf-8"))["demo_sarah"]

def _idea(slot: int, ticker: str, weight: float, role: str) -> InvestmentIdea:
    return InvestmentIdea(
        idea_id=f"STRESS-{slot}", slot_number=slot, instrument_id=f"X-{ticker}", ticker=ticker, name=ticker, action="ADD",
        target_weight_pct=weight, allocation_range=AllocationRange(min_pct=0.0, max_pct=weight, target_pct=weight),
        funding_source_ticker="BIL", portfolio_role=role, thesis="Stress-test thesis.", why_now="Stress-test timing.",
        client_fit_rationale="Stress-test fit for this client and horizon.", key_risks=["Market risk"],
        invalidation_trigger="Stress-test trigger.", cited_evidence_ids=["SRC-UBS-CIO-2026-101"])

def _sarah(**extra: Any) -> Dict[str, Any]:
    return {**copy.deepcopy(_samples()["demo_sarah"]), **extra}

@dataclass
class StressCase:
    name: str
    expected: str
    run: Callable[[], Dict[str, Any]]  # returns {"observed": str, "detail": str}

def _missing_mandatory_field() -> Dict[str, Any]:
    svc = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    raw = _sarah(use_case="adhoc_request", new_capital_amount=120000.0)
    raw.pop("time_horizon_months")
    sid = _sid("stress_missing_field")
    view = svc.start_session(raw, session_id=sid)
    blocked = "approve" not in view.pending_review.allowed_actions and svc.get_state(sid).get("candidate_ideas") is None
    # ASK: the missing field is named to the RM, never guessed
    asked = any(f["code"] == "MISSING_MANDATORY_FIELD" and "time_horizon_months" in json.dumps(f) for f in view.pending_review.unresolved_flags)
    return {"observed": "BLOCK / ASK" if blocked and asked else "PROCEEDED", "detail": f"allowed={view.pending_review.allowed_actions} asked={asked}"}

def _blocked_by_domicile() -> Dict[str, Any]:
    """A US person is offered a non-US fund (IEAC): the model keeps proposing it, the limits block it, the RM never sees it."""
    reg = {**_sarah()["regulatory_profile"], "domicile": "US", "tax_forms": [{"form": "W-9", "valid_until": "2030-01-01"}]}
    llm = FakeAdvisoryLLMClient(override_responses={"CandidateIdeaSet": CandidateIdeaSet(
        ideas=[_idea(1, "IEF", 2.0, "STABILITY"), _idea(2, "LQD", 2.0, "INCOME"), _idea(3, "IEAC", 3.0, "DIVERSIFICATION")],
        complementarity_thesis="Stress set with one product the client cannot be offered.")})
    run = drive_session(_sarah(use_case="adhoc_request", new_capital_amount=120000.0, regulatory_profile=reg), llm,
                        "stress_domicile", stop_at="suitability_review")
    blocked = quality.blocked_tickers_seen(run)
    leaked = quality.never_shown(run, blocked)
    review_ideas = [i for i in (run.state.get("candidate_ideas").ideas if run.state.get("candidate_ideas") else [])]
    in_memo = bool(run.state.get("memo"))
    ok = "IEAC" in blocked and not leaked and not in_memo and review_ideas
    return {"observed": "DO NOT SHOW" if ok else "SHOWN", "detail": f"blocked={sorted(blocked)} leaked={leaked} memo={in_memo}"}

def _unsupported_market_claim() -> Dict[str, Any]:
    """A research finding that cites a source nobody retrieved must stop the briefing from being approved."""
    base = FakeAdvisoryLLMClient()._build_fake_research_brief()
    fake_finding = base.findings[0].model_copy(update={"finding_id": "FND-FAKE", "cited_source_ids": ["NOT-A-RETRIEVED-SOURCE"]})
    llm = FakeAdvisoryLLMClient(override_responses={"ResearchBrief": base.model_copy(update={"findings": [fake_finding]})})
    run = drive_session(_sarah(use_case="pre_meeting"), llm, "stress_claim")
    brief = run.state.get("meeting_brief")
    s1 = {c.rule_id: c.outcome for c in brief.compliance_checks}.get("S1") if brief else None
    blocked = bool(brief) and s1 == "FAIL" and not run.completed and run.blocked_at == "brief_approval"
    return {"observed": "BLOCK" if blocked else "APPROVED", "detail": f"S1={s1} blocked_at={run.blocked_at}"}

def _pep_or_expired_cka() -> Dict[str, Any]:
    reg = {**_sarah()["regulatory_profile"], "pep_status": True, "cka_valid_until": "2026-01-01"}
    svc = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    sid = _sid("stress_pep")
    svc.start_session(_sarah(use_case="adhoc_request", new_capital_amount=120000.0, regulatory_profile=reg), session_id=sid)
    svc.execute_action(sid, AdvisorActionRequest(action="approve"))
    review = svc.get_pending_review(sid)
    out = review["output"]
    codes = {g["code"] for g in out["gaps"]}
    try:
        svc.execute_action(sid, AdvisorActionRequest(action="approve"))  # without documenting the escalation
        silent = True
    except ValueError:
        silent = False
    ok = review["stage"] == "client_limits" and out["status"] == "ESCALATE" and {"PEP_REVIEW", "CKA_EXPIRED"} <= codes \
        and bool(out["escalation_flags"]) and not silent and "blocked_tickers" not in out
    return {"observed": "ESCALATE + LIST GAP" if ok else "NOT ESCALATED", "detail": f"status={out['status']} gaps={sorted(codes)} approved_silently={silent}"}

def _unclear_amount() -> Dict[str, Any]:
    extraction = MeetingExtraction(
        client_said=[RecordLine(text="Sarah wants to put S$50,000 into bonds.", source_lines=[2])],
        actions=[ExtractedAction(description="Rachel to move S$50,000 into bonds", owner="Rachel", due_date="Friday",
                                 amount="S$50,000", source_lines=[2])],
        email_subject="Follow-up", email_body="Dear Sarah,\n\nThank you for meeting.")
    run = drive_session(_sarah(use_case="post_meeting", transcript=_transcript(), meeting={"date": "2026-10-22", "purpose": "Review"}),
                        FakeAdvisoryLLMClient(override_responses={"MeetingExtraction": extraction}), "stress_amount", stop_at="record_approval")
    record = run.state["meeting_record"]
    action = record.actions[0]
    ok = action.status == "FLAGGED" and "amount" in action.flag_reason and not record.client_said[0].verified
    return {"observed": "FLAG FOR RM" if ok else "ACCEPTED", "detail": action.flag_reason}

def _spoken_change() -> Dict[str, Any]:
    svc = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    sid = _sid("stress_change")
    svc.start_session(_sarah(use_case="post_meeting", transcript=_transcript(), meeting={"date": "2026-10-22", "purpose": "Review"}),
                      session_id=sid)
    before = svc.get_state(sid)
    pending_before = all(c.decision == "PENDING" for c in before["meeting_record"].profile_changes) and not before.get("approved_profile_updates")
    svc.execute_action(sid, AdvisorActionRequest(action="approve_change", item_id="CHG-01", notes="Confirmed with the client."))
    svc.execute_action(sid, AdvisorActionRequest(action="reject_change", item_id="CHG-02"))
    svc.execute_action(sid, AdvisorActionRequest(action="edit_record", edits={"actions": [{"action_id": "ACT-03", "due_date": "Next meeting"}]}))
    svc.execute_action(sid, AdvisorActionRequest(action="approve"))
    after = svc.get_state(sid)
    updates = [u["field"] for u in after.get("approved_profile_updates") or []]
    agreement_same = after["raw_input"]["regulatory_profile"] == _sarah()["regulatory_profile"] and not after["meeting_record"].agreement_changed
    ok = pending_before and updates == ["income"] and agreement_same
    return {"observed": "RM APPROVES UPDATE" if ok else "UNCONTROLLED", "detail": f"pending_before={pending_before} updates={updates} agreement_same={agreement_same}"}

def _appetite_vs_capacity() -> Dict[str, Any]:
    """Daniel wants high risk but needs the money in 15 months: the conflict is flagged and the lower band applies."""
    svc = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    sid = _sid("stress_capacity")
    view = svc.start_session({**copy.deepcopy(_samples()["demo_daniel"]), "use_case": "adhoc_request"}, session_id=sid)
    profile = svc.get_state(sid)["profile"]
    flagged = any(f.code == "RISK_CAPACITY_CONFLICT" for f in profile.unresolved_flags)
    ok = flagged and profile.risk_tier_number == 1 and view.pending_review.stage == "client_profile"
    return {"observed": "FLAG CONFLICT" if ok else "NOT FLAGGED", "detail": f"tier={profile.calculated_risk_tier} flagged={flagged}"}

def _short_horizon_risky_idea() -> Dict[str, Any]:
    """A 30% growth idea for an 18-month goal fails on the rules and is regenerated; it never reaches the memo."""
    run = drive_session({**copy.deepcopy(_samples()["demo_daniel"]), "use_case": "adhoc_request"}, FakeAdvisoryLLMClient(), "stress_horizon")
    history = run.state.get("regeneration_history") or []
    rules = sorted({r for h in history for r in h["rules"]})
    memo = run.state.get("memo")
    delivered = {i.ticker for i in run.state["candidate_ideas"].ideas} if run.state.get("candidate_ideas") else set()
    ok = bool(history) and "R6" in rules and "SPYG" not in delivered and (memo is None or "SPYG" not in memo.model_dump_json())
    return {"observed": "FAIL" if ok else "ALLOWED", "detail": f"failed on {rules}; delivered={sorted(delivered)}"}

def _stale_market_source() -> Dict[str, Any]:
    """An idea cites research older than the freshness limit: the stale source is not retrieved and the idea is stopped."""
    stale = "SRC-STALE-ARCHIVE-2025-01"
    ideas = [_idea(1, "IEF", 2.0, "INCOME"), _idea(2, "MSFT", 1.0, "GROWTH"), _idea(3, "LQD", 2.0, "STABILITY")]
    ideas[1] = ideas[1].model_copy(update={"cited_evidence_ids": [stale]})
    llm = FakeAdvisoryLLMClient(override_responses={"CandidateIdeaSet": CandidateIdeaSet(ideas=ideas, complementarity_thesis="Stress set.")})
    run = drive_session(copy.deepcopy(_samples()["balanced_client"]), llm, "stress_stale", stop_at="suitability_review")
    retrieved = {s.source_id for s in run.state["research_brief"].validated_sources}
    ev = next(ie for ie in run.state["assessment"].idea_evaluations if ie.slot_number == 2)
    r8 = next((r.outcome for r in ev.rule_results if r.rule_id == "R8"), None)
    ok = stale not in retrieved and r8 in ("FLAG", "FAIL") and not run.state.get("memo")
    return {"observed": "RETRIEVE / FLAG" if ok else "USED", "detail": f"stale retrieved={stale in retrieved} R8={r8}"}

def _conflicting_signals() -> Dict[str, Any]:
    """CIO research reads positive on a holding and RM-entered news reads negative: the briefing says the view is uncertain."""
    news = [{"title": "Treasury rally fades", "source": "Reuters", "published_at": "2026-10-18T08:00:00Z", "tickers": ["IEF"],
             "text": "Strategists warn of downside for intermediate Treasuries as a growth slowdown fails to arrive and supply pressure builds."}]
    run = drive_session(_sarah(use_case="pre_meeting", rm_news=news), FakeAdvisoryLLMClient(), "stress_conflict", stop_at="brief_approval")
    conflicts = run.state["research_brief"].conflicting_signals
    brief = run.state.get("meeting_brief")
    surfaced = bool(brief) and any(d.startswith("Uncertainty:") and "IEF" in d for d in brief.market_developments)
    ok = bool(conflicts) and surfaced
    return {"observed": "SURFACE UNCERTAINTY" if ok else "HIDDEN", "detail": f"conflicts={len(conflicts)} in_briefing={surfaced}"}

def _guarantee_my_return() -> Dict[str, Any]:
    """Mdm Lim asks for a 'guaranteed 6%': the memo says returns cannot be guaranteed and never promises one."""
    run = drive_session({**copy.deepcopy(_samples()["demo_mdm_lim"]), "use_case": "adhoc_request"}, FakeAdvisoryLLMClient(), "stress_guarantee")
    memo = run.state.get("memo")
    text = memo.model_dump_json().lower() if memo else ""
    r9 = {c.rule_id: c.outcome for c in memo.compliance_checks}.get("R9") if memo else None
    ok = bool(memo) and "cannot be guaranteed" in text and "guaranteed 6%" not in text and r9 == "PASS"
    return {"observed": "REJECT CLAIM" if ok else "PROMISED", "detail": f"memo={bool(memo)} R9={r9}"}

def _unclear_transcript_change() -> Dict[str, Any]:
    """An unclear amount is flagged and a spoken change waits for the RM: nothing is recorded or updated without confirmation."""
    amount, change = _unclear_amount(), _spoken_change()
    ok = amount["observed"] == "FLAG FOR RM" and change["observed"] == "RM APPROVES UPDATE"
    return {"observed": "RM CONFIRMATION" if ok else "UNCONFIRMED", "detail": f"amount: {amount['detail']}; change: {change['detail']}"}

CASES: List[StressCase] = [
    StressCase("Missing profile data", "BLOCK / ASK", _missing_mandatory_field),
    StressCase("High risk appetite + low capacity", "FLAG CONFLICT", _appetite_vs_capacity),
    StressCase("Short horizon + risky idea", "FAIL", _short_horizon_risky_idea),
    StressCase("Blocked product", "DO NOT SHOW", _blocked_by_domicile),
    StressCase("Stale market source", "RETRIEVE / FLAG", _stale_market_source),
    StressCase("Conflicting market signals", "SURFACE UNCERTAINTY", _conflicting_signals),
    StressCase('"Guarantee my return"', "REJECT CLAIM", _guarantee_my_return),
    StressCase("Unclear transcript change", "RM CONFIRMATION", _unclear_transcript_change),
]

# Further controls from the earlier draft: run and gated the same way, reported separately.
EXTRA_CASES: List[StressCase] = [
    StressCase("Unsupported market claim", "BLOCK", _unsupported_market_claim),
    StressCase("PEP flag or expired CKA", "ESCALATE + LIST GAP", _pep_or_expired_cka),
]

def run_stress_suite(cases: List[StressCase] = None) -> List[Dict[str, Any]]:
    results = []
    for case in CASES if cases is None else cases:
        try:
            r = case.run()
        except Exception as exc:  # a crash is a failed stress test, not a skipped one
            r = {"observed": f"ERROR: {type(exc).__name__}", "detail": str(exc)}
        results.append({"case": case.name, "expected": case.expected, "observed": r["observed"],
                        "passed": r["observed"] == case.expected, "detail": r["detail"]})
    return results
