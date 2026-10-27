"""Stress-test cases 1-6 from the AAFS deck (slide 33), run end to end through the advisory graph."""

import copy
import pytest
from src.api.service import AdvisorySessionService
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.advisory import CandidateIdeaSet, InvestmentIdea, AllocationRange
from src.schemas.workspace import AdvisorActionRequest

def new_service(**fake_kwargs):
    return AdvisorySessionService(llm_client=FakeAdvisoryLLMClient(**fake_kwargs))

def act(service, sid, **kwargs):
    return service.execute_action(sid, AdvisorActionRequest(**kwargs))

def to_recommendation_review(service, sid):
    while True:
        review = service.get_pending_review(sid)
        if review["stage"] == "suitability_review":
            return review
        act(service, sid, action="approve")

def idea(slot, ticker, weight, role, source="SRC-UBS-CIO-2026-101"):
    return InvestmentIdea(
        idea_id=f"IDEA-{ticker}-{slot:02d}", slot_number=slot, instrument_id=f"X-{ticker}", ticker=ticker, name=ticker,
        action="ADD", target_weight_pct=weight,
        allocation_range=AllocationRange(min_pct=0.0, max_pct=weight, target_pct=weight),
        funding_source_ticker="BIL", portfolio_role=role, thesis="t", why_now="w", client_fit_rationale="f",
        key_risks=["r"], invalidation_trigger="i", cited_evidence_ids=[source]
    )

def idea_set(*ideas):
    return CandidateIdeaSet(ideas=list(ideas), complementarity_thesis="Stress-test set")

def test1_high_willingness_with_one_year_liquidity_need_flags_conflict_and_caps_growth(sample_profiles):
    """Test 1: the system prioritises liquidity and flags the conflict; growth is capped (R5, R6, L1)."""
    service = new_service()
    sid = "stress_1"
    view = service.start_session(sample_profiles["demo_daniel"], session_id=sid)
    profile = service.get_state(sid)["profile"]
    assert any(f.code == "RISK_CAPACITY_CONFLICT" for f in profile.unresolved_flags)
    assert profile.risk_tier_number == 1, "the lower of willingness and capacity applies"
    assert view.pending_review.stage == "client_profile"
    assert profile.near_term_liquidity_need > 0

    act(service, sid, action="approve")
    act(service, sid, action="approve")
    assessment = service.get_state(sid)["assessment"]
    growth = sum(i.target_weight_pct for i in service.get_state(sid)["candidate_ideas"].ideas if i.portfolio_role == "GROWTH")
    assert growth <= 30.0 or assessment.overall_status != "PASS"

def test2_heavy_tech_concentration_is_not_worsened_without_justification(balanced_client_data):
    """Test 2: adding to an already concentrated market is a hard FAIL, never silently accepted."""
    service = new_service(override_responses={"CandidateIdeaSet": idea_set(
        idea(1, "IEF", 2.0, "INCOME"), idea(2, "NVDA", 10.0, "GROWTH"), idea(3, "LQD", 2.0, "STABILITY"))})
    sid = "stress_2"
    service.start_session(balanced_client_data, session_id=sid)
    review = to_recommendation_review(service, sid)
    assessment = service.get_state(sid)["assessment"]
    assert assessment.overall_status == "FAIL"
    assert any(r.rule_id == "R7" and r.outcome == "FAIL" for ie in assessment.idea_evaluations for r in ie.rule_results)
    assert "lock_idea" in review["allowed_actions"]  # passing ideas can be locked, the failing one cannot
    failing = next(ie for ie in assessment.idea_evaluations if ie.status == "FAIL")
    with pytest.raises(ValueError):
        act(service, sid, action="lock_idea", idea_id=failing.idea_id)
    assert service.get_state(sid).get("memo") is None

def test3_guaranteed_return_request_never_yields_guaranteed_language(sample_profiles):
    """Test 3: Mdm Lim asks for a 'guaranteed 6%'; no prohibited wording reaches the memo (R9)."""
    service = new_service()
    sid = "stress_3"
    service.start_session(sample_profiles["demo_mdm_lim"], session_id=sid)
    while service.get_pending_review(sid)["stage"] != "advisor_approval":
        review = service.get_pending_review(sid)
        if review["stage"] == "suitability_review":
            for i in review["candidate_ideas"]:
                act(service, sid, action="lock_idea", idea_id=i["idea_id"], notes="Documented.")
        else:
            act(service, sid, action="approve")
    memo = service.get_state(sid)["memo"]
    text = memo.model_dump_json().lower()
    assert "cannot be guaranteed" in text
    assert "guaranteed 6%" not in text
    assert all(c.outcome == "PASS" for c in memo.compliance_checks)

def test4_missing_risk_information_stops_the_workflow(sample_profiles):
    """Test 4: with no risk willingness the workflow stops; nothing is recommended."""
    client = copy.deepcopy(sample_profiles["balanced_client"])
    client.pop("stated_risk_willingness")
    client["questionnaire_answers"] = None
    service = new_service()
    sid = "stress_4"
    view = service.start_session(client, session_id=sid)
    assert view.pending_review.stage == "client_profile"
    assert "approve" not in view.pending_review.allowed_actions
    assert any(f["code"] == "MISSING_MANDATORY_FIELD" for f in view.pending_review.unresolved_flags)
    assert service.get_state(sid).get("candidate_ideas") is None

def test5_idea_supported_by_stale_research_is_rejected(balanced_client_data):
    """Test 5: an idea citing a source older than the freshness limit fails evidence validation (R8)."""
    stale = "SRC-STALE-ARCHIVE-2025-01"
    service = new_service(override_responses={"CandidateIdeaSet": idea_set(
        idea(1, "IEF", 2.0, "INCOME"), idea(2, "MSFT", 1.0, "GROWTH", source=stale), idea(3, "LQD", 2.0, "STABILITY"))})
    sid = "stress_5"
    service.start_session(balanced_client_data, session_id=sid)
    to_recommendation_review(service, sid)
    state = service.get_state(sid)
    assert stale not in {s.source_id for s in state["research_brief"].validated_sources}
    stale_eval = next(ie for ie in state["assessment"].idea_evaluations if ie.slot_number == 2)
    assert stale_eval.status == "FAIL" and not stale_eval.evidence_verified
    assert any(r.rule_id == "R8" and r.outcome == "FAIL" for r in stale_eval.rule_results)
    assert state.get("memo") is None

def test6_only_the_rejected_idea_is_revised_and_other_work_is_preserved(balanced_client_data):
    """Test 6: when the critic rejects one of three ideas, the other two stay byte-for-byte identical."""
    service = new_service(override_responses={"CandidateIdeaSet": idea_set(
        idea(1, "IEF", 2.0, "INCOME"), idea(2, "NVDA", 10.0, "GROWTH"), idea(3, "LQD", 2.0, "STABILITY"))})
    sid = "stress_6"
    service.start_session(balanced_client_data, session_id=sid)
    to_recommendation_review(service, sid)
    log = service.get_run_log(sid)
    regens = [e for e in log if e["event"] == "auto_regeneration"]
    assert regens and all(e["slots"] == [2] for e in regens)
    idea_outputs = [e for e in log if e["event"] == "agent_output" and e["stage"] == "investment_ideas"]
    firsts = {i["slot_number"]: i for i in idea_outputs[0]["output"]["ideas"]}
    for later in idea_outputs[1:]:
        by_slot = {i["slot_number"]: i for i in later["output"]["ideas"]}
        assert by_slot[1] == firsts[1] and by_slot[3] == firsts[3]
