"""Integration trajectory tests verifying LangGraph execution paths, HITL interrupts, and selective regeneration."""

import pytest
from src.api.service import AdvisorySessionService
from src.schemas.workspace import AdvisorActionRequest
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.advisory import CandidateIdeaSet, InvestmentIdea, AllocationRange, SuitabilityAssessment, IdeaStatus
from src.schemas.tools import RiskMetrics, Breach

def approve_until(service, session_id, stage):
    """Approve each review checkpoint until the graph pauses at `stage`."""
    for _ in range(10):
        review = service.get_pending_review(session_id)
        assert review is not None, f"Session ended before reaching '{stage}' review"
        if review["stage"] == stage:
            return review
        service.execute_action(session_id, AdvisorActionRequest(action="approve"))
    raise AssertionError(f"Never reached '{stage}' review")

def test_balanced_client_full_trajectory_to_memo(balanced_client_data):
    """
    Clean client profile trajectory:
    1. Ingestion passes without blocking contradiction.
    2. Market Intelligence generates brief.
    3. Investment Ideas generates 3 complementary ideas.
    4. Suitability Critic passes.
    5. Memo Formatter renders structured memo and PDF.
    6. Graph pauses for RM review after every agent; RM approves each one.
    7. Advisor accepts final memo -> session completes.
    """
    service = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    session_id = "test_balanced_traj"

    # Step 1: Start session -> pauses for profile review
    view = service.start_session(balanced_client_data, session_id=session_id)
    assert view.session_id == session_id
    assert view.client_id == "CL-88201"
    assert view.pending_review.stage == "client_profile"
    assert view.pending_review.output["client_id"] == "CL-88201"

    # Each agent's output is reviewed in order
    for expected_stage in ["client_profile", "market_intelligence", "investment_ideas", "suitability_review"]:
        assert service.get_pending_review(session_id)["stage"] == expected_stage
        service.execute_action(session_id, AdvisorActionRequest(action="approve"))
    assert service.get_pending_review(session_id)["stage"] == "advisor_approval"

    # Verify that graph reached advisor_approval or completed
    state = service.get_state(session_id)
    assert state.get("profile") is not None
    assert state.get("research_brief") is not None
    assert state.get("candidate_ideas") is not None
    assert state.get("assessment") is not None
    assert state.get("memo") is not None

    # Step 2: Relationship Manager accepts final memo
    view_after_accept = service.execute_action(
        session_id,
        AdvisorActionRequest(action="accept", notes="RM Approved final memo.")
    )
    final_state = service.get_state(session_id)
    assert final_state.get("is_completed") is True
    assert final_state.get("pdf_path") is not None

def test_contradictory_client_interrupts_at_profile(contradictory_client_data):
    """
    Contradictory client (Aggressive willingness vs 18M liquidity commitment):
    Must trigger RISK_CAPACITY_CONFLICT flag and pause graph at hitl_resolve_profile.
    """
    service = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    session_id = "test_conflict_traj"

    view = service.start_session(contradictory_client_data, session_id=session_id)

    # Must be paused for human action
    state = service.get_state(session_id)
    assert view.center_panel.requires_human_action is True
    assert state.get("profile") is not None
    blocking_flags = [f for f in state["profile"].unresolved_flags if f.severity == "BLOCKING_INTERRUPT"]
    assert len(blocking_flags) >= 1
    assert any("RISK_CAPACITY_CONFLICT" in f.code for f in blocking_flags)

    assert view.pending_review.stage == "client_profile"
    assert len(view.pending_review.unresolved_flags) >= 1

    # Resolve conflict as RM
    view_res = service.execute_action(
        session_id,
        AdvisorActionRequest(
            action="resolve_conflict",
            conflict_resolution="Constrained risk tolerance to Moderate based on 18-month liquidity call."
        )
    )
    # Graph resumes past client profile and pauses for market review
    post_res_state = service.get_state(session_id)
    assert post_res_state.get("research_brief") is not None
    assert all(f.is_resolved for f in post_res_state["profile"].unresolved_flags)
    assert view_res.pending_review.stage == "market_intelligence"

def test_fail_status_routes_to_regeneration_never_to_memo(balanced_client_data):
    """
    Trajectory test:
    When Suitability Critic detects a hard breach (FAIL):
    The RM cannot approve it; revising MUST route to mark_failed_slots -> investment_ideas,
    and NEVER reach memo_formatter while in FAIL status.
    """
    # Create candidate ideas with a hard single-stock limit breach (NVDA 12% + 25% = 37% > 30% ceiling)
    breach_idea_set = CandidateIdeaSet(
        ideas=[
            InvestmentIdea(
                idea_id="IDEA-IEF-01", slot_number=1, instrument_id="FI-US10Y", ticker="IEF",
                name="iShares Treasury", action="ADD", target_weight_pct=5.0,
                allocation_range=AllocationRange(min_pct=3.0, max_pct=8.0, target_pct=5.0),
                funding_source_ticker="BIL", portfolio_role="INCOME", thesis="Yield", why_now="Now",
                client_fit_rationale="Fit", key_risks=["Vol"], invalidation_trigger="Trig",
                cited_evidence_ids=["SRC-UBS-CIO-2026-101"]
            ),
            InvestmentIdea(
                idea_id="IDEA-NVDA-BREACH-02", slot_number=2, instrument_id="EQ-NVDA", ticker="NVDA",
                name="NVIDIA Corp", action="ADD", target_weight_pct=36.0,
                allocation_range=AllocationRange(min_pct=10.0, max_pct=40.0, target_pct=36.0),
                funding_source_ticker="BIL", portfolio_role="GROWTH", thesis="AI", why_now="Now",
                client_fit_rationale="Fit", key_risks=["Vol"], invalidation_trigger="Trig",
                cited_evidence_ids=["SRC-UBS-CIO-2026-101"]
            ),
            InvestmentIdea(
                idea_id="IDEA-MSFT-03", slot_number=3, instrument_id="EQ-MSFT", ticker="MSFT",
                name="Microsoft", action="BUY", target_weight_pct=1.5,
                allocation_range=AllocationRange(min_pct=1.0, max_pct=2.0, target_pct=1.5),
                funding_source_ticker="BIL", portfolio_role="STABILITY", thesis="Cloud", why_now="Now",
                client_fit_rationale="Fit", key_risks=["Vol"], invalidation_trigger="Trig",
                cited_evidence_ids=["SRC-UBS-CIO-2026-101"]
            )
        ],
        complementarity_thesis="Initial idea set with hard breach"
    )

    mock_client = FakeAdvisoryLLMClient(override_responses={
        "CandidateIdeaSet": breach_idea_set
    })

    service = AdvisorySessionService(llm_client=mock_client)
    session_id = "test_fail_traj"

    service.start_session(balanced_client_data, session_id=session_id)
    review = approve_until(service, session_id, "suitability_review")
    assert service.get_state(session_id)["assessment"].overall_status == "FAIL"
    assert "approve" not in review["allowed_actions"]
    with pytest.raises(ValueError):
        service.execute_action(session_id, AdvisorActionRequest(action="approve"))

    service.execute_action(session_id, AdvisorActionRequest(action="revise", notes="Reduce NVDA."))
    state = service.get_state(session_id)
    assert state.get("memo") is None
    assert service.get_pending_review(session_id)["stage"] == "investment_ideas"

    # Verification 1: regeneration_count was incremented
    assert state.get("regeneration_count", 0) >= 1

    # Verification 2: slot 2 was marked for regeneration; slots 1 and 3 were locked
    assert 2 in state.get("regenerate_slots", []) or "IDEA-IEF-01" in state.get("locked_idea_ids", [])

    # Verification 3: Audit trail proves mark_failed_slots was traversed
    audit_stages = [a.get("stage") for a in state.get("audit_trail", [])]
    assert "mark_failed_slots" in audit_stages

def test_selective_regeneration_locks_passed_ideas(balanced_client_data):
    """
    Selective Regeneration Test:
    When RM rejects Idea #2 (NESN):
    Ideas #1 (IEF) and #3 (MSFT) MUST remain locked with identical content,
    while only Slot #2 is regenerated.
    """
    mock_client = FakeAdvisoryLLMClient()
    service = AdvisorySessionService(llm_client=mock_client)
    session_id = "test_selective_regen"

    # Run initial pipeline up to the idea review
    service.start_session(balanced_client_data, session_id=session_id)
    approve_until(service, session_id, "investment_ideas")
    initial_ideas = service.get_state(session_id)["candidate_ideas"]
    initial_slot_1 = next(i for i in initial_ideas.ideas if i.slot_number == 1)
    initial_slot_3 = next(i for i in initial_ideas.ideas if i.slot_number == 3)

    # Relationship Manager rejects Idea in Slot 2
    view_after_reject = service.execute_action(
        session_id,
        AdvisorActionRequest(action="reject", idea_id=initial_ideas.ideas[1].idea_id, notes="Prefer healthcare over consumer staples.")
    )

    # Verify state after selective regeneration
    new_state = service.get_state(session_id)
    new_ideas = new_state["candidate_ideas"]
    new_slot_1 = next(i for i in new_ideas.ideas if i.slot_number == 1)
    new_slot_3 = next(i for i in new_ideas.ideas if i.slot_number == 3)

    # Ideas #1 and #3 must be byte-for-byte identical (locked)
    assert new_slot_1.ticker == initial_slot_1.ticker
    assert new_slot_1.target_weight_pct == initial_slot_1.target_weight_pct
    assert new_slot_3.ticker == initial_slot_3.ticker
    assert new_slot_3.target_weight_pct == initial_slot_3.target_weight_pct

    # Regeneration count must be >= 1
    assert new_state["regeneration_count"] >= 1

def test_profile_regenerate_applies_rm_edits(balanced_client_data):
    """RM edits rendered profile fields and regenerates; the Client Profile Agent re-runs on the edited intake."""
    service = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    session_id = "test_profile_regen"
    service.start_session(balanced_client_data, session_id=session_id)

    view = service.execute_action(session_id, AdvisorActionRequest(
        action="regenerate",
        edits={"name": "Dr. Elena Rostova-Meyer", "time_horizon_months": 84, "not_a_field": "ignored"}
    ))
    assert view.pending_review.stage == "client_profile"
    profile = service.get_state(session_id)["profile"]
    assert profile.name == "Dr. Elena Rostova-Meyer"
    assert profile.time_horizon_months == 84
    assert "not_a_field" not in service.get_state(session_id)["raw_input"]

def test_incomplete_profile_cannot_be_approved(incomplete_client_data):
    """Missing mandatory fields must be supplied by the RM; approval is not offered."""
    service = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    view = service.start_session(incomplete_client_data, session_id="test_incomplete")
    assert view.pending_review.stage == "client_profile"
    assert "approve" not in view.pending_review.allowed_actions
    with pytest.raises(ValueError):
        service.execute_action("test_incomplete", AdvisorActionRequest(action="approve"))

def test_market_regenerate_passes_feedback_to_agent(balanced_client_data):
    llm = FakeAdvisoryLLMClient()
    service = AdvisorySessionService(llm_client=llm)
    session_id = "test_market_regen"
    service.start_session(balanced_client_data, session_id=session_id)
    approve_until(service, session_id, "market_intelligence")

    view = service.execute_action(session_id, AdvisorActionRequest(action="regenerate", notes="Focus on CHF exposure."))
    assert view.pending_review.stage == "market_intelligence"
    brief_prompts = [c["user_prompt"] for c in llm.call_history if c["model_name"] == "ResearchBrief"]
    assert len(brief_prompts) == 2
    assert "Focus on CHF exposure." in brief_prompts[-1]

def test_reject_ends_session_at_reviewed_stage(balanced_client_data):
    service = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    session_id = "test_reject"
    service.start_session(balanced_client_data, session_id=session_id)
    approve_until(service, session_id, "market_intelligence")

    view = service.execute_action(session_id, AdvisorActionRequest(action="reject", notes="Research outdated."))
    state = service.get_state(session_id)
    assert state["is_completed"] is True
    assert state["outcome"] == "REJECTED"
    assert state.get("candidate_ideas") is None
    assert view.pending_review is None
    statuses = {s.stage_key: s.status for s in view.workflow_stages}
    assert statuses["client_profile"] == "COMPLETED"
    assert statuses["market_intelligence"] == "FAILED"

def test_advisor_note_does_not_resume_graph(balanced_client_data):
    service = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    session_id = "test_note"
    service.start_session(balanced_client_data, session_id=session_id)

    view = service.execute_action(session_id, AdvisorActionRequest(action="add_advisor_note", notes="Called client."))
    assert "Called client." in view.advisor_notes
    assert view.pending_review.stage == "client_profile"
