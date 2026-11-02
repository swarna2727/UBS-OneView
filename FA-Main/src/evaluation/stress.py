"""Stress testing (deck slide 26): we deliberately try to break it.

'Reliability means testing the cases where the obvious answer is wrong.' Each case builds a hostile situation, runs the
real workflow, and compares what happened with the outcome the deck requires:

  Missing mandatory field          -> BLOCK
  Product blocked by domicile      -> NOT SHOWN
  Unsupported market claim         -> BLOCK
  PEP flag or expired CKA          -> ESCALATE + LIST GAP
  Unclear amount in transcript     -> FLAG FOR RM
  Spoken change in circumstances   -> RM APPROVES UPDATE
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
    return {"observed": "BLOCK" if blocked else "PROCEEDED", "detail": f"allowed={view.pending_review.allowed_actions}"}

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
    return {"observed": "NOT SHOWN" if ok else "SHOWN", "detail": f"blocked={sorted(blocked)} leaked={leaked} memo={in_memo}"}

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

CASES: List[StressCase] = [
    StressCase("Missing mandatory field", "BLOCK", _missing_mandatory_field),
    StressCase("Product blocked by domicile", "NOT SHOWN", _blocked_by_domicile),
    StressCase("Unsupported market claim", "BLOCK", _unsupported_market_claim),
    StressCase("PEP flag or expired CKA", "ESCALATE + LIST GAP", _pep_or_expired_cka),
    StressCase("Unclear amount in transcript", "FLAG FOR RM", _unclear_amount),
    StressCase("Spoken change in circumstances", "RM APPROVES UPDATE", _spoken_change),
]

def run_stress_suite() -> List[Dict[str, Any]]:
    results = []
    for case in CASES:
        try:
            r = case.run()
        except Exception as exc:  # a crash is a failed stress test, not a skipped one
            r = {"observed": f"ERROR: {type(exc).__name__}", "detail": str(exc)}
        results.append({"case": case.name, "expected": case.expected, "observed": r["observed"],
                        "passed": r["observed"] == case.expected, "detail": r["detail"]})
    return results
