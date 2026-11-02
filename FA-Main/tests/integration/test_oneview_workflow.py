"""OneView deck behaviours end to end: limits checkpoint, two lanes, blocked ideas never shown, UC3 record."""

import copy
import json
import os
import pytest
from src.api.service import AdvisorySessionService
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.advisory import MeetingExtraction, RecordLine, ExtractedAction, CircumstanceChange
from src.schemas.workspace import AdvisorActionRequest


def new_service(llm=None):
    return AdvisorySessionService(llm_client=llm or FakeAdvisoryLLMClient())


def act(service, sid, **kwargs):
    return service.execute_action(sid, AdvisorActionRequest(**kwargs))


def sarah(sample_profiles, **extra):
    return {**copy.deepcopy(sample_profiles["demo_sarah"]), **extra}


@pytest.fixture
def transcript():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "samples", "meeting_transcripts.json")
    return json.load(open(path))["demo_sarah"]


def record_session(service, sample_profiles, transcript, sid="uc3", llm_raw=None, **extra):
    raw = sarah(sample_profiles, use_case="post_meeting", transcript=transcript,
                meeting={"date": "2026-10-22", "purpose": "Quarterly review"}, **extra)
    return service.start_session(raw, session_id=sid)


# ---------------------------------------------------------------- orchestration and the limits checkpoint

def test_orchestration_plan_is_logged_and_the_rm_is_the_human_orchestrator(sample_profiles):
    service = new_service()
    service.start_session(sarah(sample_profiles, use_case="adhoc_request", new_capital_amount=120000.0), session_id="plan")
    plan = service.get_run_log("plan")[0]["orchestration"]
    assert plan == ["ClientProfileAgent", "CP:client_profile", "ClientLimitsLayer", "CP:client_limits", "MarketIntelligenceAgent",
                    "CP:market_intelligence", "OpportunityAgent", "FitEligibilityCritic", "CP:suitability_review",
                    "BriefRecordAgent", "CP:advisor_approval"]


def test_limits_run_before_research_and_the_rm_sees_areas_never_blocked_products(sample_profiles):
    service = new_service()
    service.start_session(sarah(sample_profiles, use_case="adhoc_request", new_capital_amount=120000.0,
                                regulatory_profile={**sample_profiles["demo_sarah"]["regulatory_profile"], "cka_valid_until": "2026-01-01"}),
                          session_id="lim")
    act(service, "lim", action="approve")
    review = service.get_pending_review("lim")
    assert review["stage"] == "client_limits" and service.get_state("lim").get("research_brief") is None
    assert "blocked_tickers" not in review["output"] and "HYG" not in json.dumps(review["output"])
    assert review["output"]["status"] == "ESCALATE"
    with pytest.raises(ValueError, match="escalation"):
        act(service, "lim", action="approve")  # open escalations need a documented note
    view = act(service, "lim", action="approve", notes="CKA refresh booked with the client for next week.")
    assert view.pending_review.stage == "market_intelligence"


def test_blocked_offer_map_cannot_be_approved_until_the_rm_supplies_the_data(sample_profiles):
    service = new_service()
    raw = sarah(sample_profiles, use_case="adhoc_request", new_capital_amount=120000.0)
    del raw["regulatory_profile"]
    service.start_session(raw, session_id="blocked")
    act(service, "blocked", action="approve")
    review = service.get_pending_review("blocked")
    assert review["stage"] == "client_limits" and review["allowed_actions"] == ["regenerate", "reject"]
    with pytest.raises(ValueError):
        act(service, "blocked", action="approve")
    view = act(service, "blocked", action="regenerate",
               edits={"regulatory_profile": sample_profiles["demo_sarah"]["regulatory_profile"]})
    assert view.pending_review.stage == "client_limits" and view.pending_review.output["status"] == "CLEAR"
    assert "approve" in view.pending_review.allowed_actions


# ---------------------------------------------------------------- UC2: two lanes, fit + eligibility, blocked ideas

def reach_ideas(service, sid, notes=None):
    while service.get_pending_review(sid)["stage"] != "suitability_review":
        stage = service.get_pending_review(sid)["stage"]
        act(service, sid, action="approve", notes=notes if stage == "client_limits" else None)
    return service.get_pending_review(sid)


def test_ideas_come_in_two_lanes_with_disclosures_ranges_and_fit_and_eligibility_checks(sample_profiles):
    service = new_service()
    service.start_session(sarah(sample_profiles, use_case="adhoc_request", new_capital_amount=120000.0), session_id="lanes")
    reach_ideas(service, "lanes")
    ideas = service.get_state("lanes")["candidate_ideas"].ideas
    assert [i.lane for i in ideas] == ["ALIGNED", "ALIGNED", "EXPLORATORY"]
    assert all(i.required_disclosures and i.indicative_amount_range and i.indicative_amount_range[0] <= i.indicative_amount_range[1] for i in ideas)
    for ev in service.get_state("lanes")["assessment"].idea_evaluations:
        assert set(ev.fit_check) == {"Risk range", "Horizon", "Concentration", "Liquidity"}
        assert ev.eligibility["Client limits"] == "ELIGIBLE" and ev.eligibility["Product risk disclosure"] == "REQUIRED"
        assert any(r.rule_id == "E1" and r.outcome == "PASS" for r in ev.rule_results)


def test_an_idea_blocked_by_client_limits_is_regenerated_and_never_shown(sample_profiles):
    """Expired CKA blocks the high-yield idea. Only that slot is regenerated; the product is never named to the RM."""
    service = new_service()
    reg = {**sample_profiles["demo_sarah"]["regulatory_profile"], "cka_valid_until": "2026-01-01"}
    service.start_session(sarah(sample_profiles, use_case="adhoc_request", new_capital_amount=120000.0, regulatory_profile=reg), session_id="blk")
    review = reach_ideas(service, "blk", notes="CKA refresh booked.")
    state = service.get_state("blk")
    first = state["idea_rounds"][0]
    slots = {i["idea_id"]: i["slot_number"] for i in first["ideas"]}
    assert {slots[k]: v for k, v in first["statuses"].items()} == {1: "PASS", 2: "FAIL", 3: "PASS"}
    assert state["idea_rounds"][1]["trigger"]["slots"] == [2] and state["idea_rounds"][1]["overall_status"] == "PASS"
    shown = json.dumps(review) + json.dumps(service.get_workspace_view("blk").model_dump(mode="json")) + json.dumps(state["idea_rounds"], default=str)
    assert "HYG" not in shown and "High Yield" not in shown and "iBoxx $ High Yield" not in shown
    assert "E1" in state["regeneration_history"][0]["rules"]


def test_blocked_idea_that_exhausts_regenerations_is_withheld_from_the_rm(sample_profiles):
    """If a blocked idea still reaches the RM after R10 escalation, its product is replaced, and it can never be locked."""
    always_hyg = FakeAdvisoryLLMClient()
    original = always_hyg._build_fake_candidate_ideas

    def stubborn(prompt):
        s = original(prompt)
        ideas = [i.model_copy(update={"ticker": "HYG", "instrument_id": "FI-USHY", "name": "iShares iBoxx $ High Yield Corporate Bond ETF"}) if i.slot_number == 2 else i
                 for i in s.ideas]
        return s.model_copy(update={"ideas": ideas})
    always_hyg._build_fake_candidate_ideas = stubborn

    service = new_service(always_hyg)
    reg = {**sample_profiles["demo_sarah"]["regulatory_profile"], "cka_valid_until": "2026-01-01"}
    service.start_session(sarah(sample_profiles, use_case="adhoc_request", new_capital_amount=120000.0, regulatory_profile=reg), session_id="stub")
    review = reach_ideas(service, "stub", notes="CKA refresh booked.")
    assert review["escalated"] is True
    blocked = next(i for i in review["candidate_ideas"] if i["slot_number"] == 2)
    assert blocked["ticker"] == "-" and "withheld" in blocked["name"]
    assert "HYG" not in json.dumps(review)
    with pytest.raises(ValueError, match="FAIL"):
        act(service, "stub", action="lock_idea", idea_id=blocked["idea_id"])


# ---------------------------------------------------------------- UC1: facts, sources, next briefing

def test_rm_entered_news_is_labelled_and_old_research_is_excluded(sample_profiles):
    service = new_service()
    news = [{"title": "Home loan rates ease", "source": "Straits Times", "text": "Mortgage rates eased this month.",
             "published_at": "2026-10-15T00:00:00Z", "tickers": ["AGGG"]}]
    service.start_session(sarah(sample_profiles, use_case="pre_meeting", rm_news=news), session_id="news")
    act(service, "news", action="approve"); act(service, "news", action="approve")
    brief = service.get_state("news")["research_brief"]
    ids = [s.source_id for s in brief.validated_sources]
    assert "RM-NEWS-01" in ids and "SRC-STALE-ARCHIVE-2025-01" not in ids
    assert next(s for s in brief.validated_sources if s.source_id == "RM-NEWS-01").publisher.startswith("RM-entered")
    assert ids.index("RM-NEWS-01") == len(ids) - 1  # CIO research first
    assert brief.rm_news_count == 1


def test_portfolio_impact_lists_what_moved_and_by_how_much_without_guessing(sample_profiles):
    service = new_service()
    service.start_session(sarah(sample_profiles, use_case="pre_meeting"), session_id="imp")
    act(service, "imp", action="approve"); act(service, "imp", action="approve"); act(service, "imp", action="approve")
    impact = {m.ticker: m for m in service.get_state("imp")["meeting_brief"].portfolio_impact}
    assert impact["ACWI"].move_pct == 4.8 and impact["ACWI"].contribution_pct == round(impact["ACWI"].weight_pct * 4.8 / 100, 2)


def test_previous_record_feeds_the_next_briefing(sample_profiles):
    service = new_service()
    prev = {"actions": [{"description": "Rachel to send the disclosure", "owner": "Rachel", "due_date": "Friday"}]}
    service.start_session(sarah(sample_profiles, use_case="pre_meeting", previous_meeting_record=prev), session_id="feed")
    act(service, "feed", action="approve"); act(service, "feed", action="approve"); act(service, "feed", action="approve")
    assert service.get_state("feed")["meeting_brief"].open_actions == ["Rachel to send the disclosure (owner Rachel, due Friday)"]


# ---------------------------------------------------------------- UC3: after the meeting

def test_uc3_summary_actions_checks_then_rm_approval_writes_the_record(sample_profiles, transcript):
    service = new_service()
    view = record_session(service, sample_profiles, transcript)
    assert view.pending_review.stage == "record_approval"
    record = service.get_state("uc3")["meeting_record"]
    assert [a.status for a in record.actions] == ["CONFIRMED", "CONFIRMED", "FLAGGED"]
    assert "no due date" in record.actions[2].flag_reason
    assert all(l.verified and l.source_lines for l in record.client_said + record.decisions)
    assert [c.field for c in record.profile_changes] == ["income", "employment"]
    assert all(c.check.status == "FLAG" and c.decision == "PENDING" for c in record.profile_changes)
    assert record.agreement_changed is False and record.agreement_type == "Advisory"
    allowed = view.pending_review.allowed_actions
    assert "approve" not in allowed and {"approve_change", "reject_change", "edit_record"} <= set(allowed)

    # Nothing is saved before the RM acts, and a flagged change needs the RM's reasoning.
    assert service.get_pdf_path("uc3") is None
    with pytest.raises(ValueError, match="notes"):
        act(service, "uc3", action="approve_change", item_id="CHG-01")
    act(service, "uc3", action="approve_change", item_id="CHG-01", notes="Sarah confirmed the new role; figures to follow.")
    act(service, "uc3", action="reject_change", item_id="CHG-02")
    view = act(service, "uc3", action="edit_record", edits={
        "actions": [{"action_id": "ACT-03", "due_date": "Next meeting"}],
        "email_body": "Dear Sarah,\n\nThank you for meeting. I will send the disclosure by Friday.\n\nKind regards,\nRachel"})
    assert "approve" in view.pending_review.allowed_actions
    view = act(service, "uc3", action="approve", notes="Confirmed.")
    state = service.get_state("uc3")
    assert view.outcome == "APPROVED" and os.path.exists(service.get_pdf_path("uc3"))
    assert state["meeting_record"].rm_confirmed and state["meeting_record"].follow_up_email.edited_by_rm
    assert [u["field"] for u in state["approved_profile_updates"]] == ["income"]
    assert state["raw_input"]["regulatory_profile"]["agreement_type"] == "Advisory"  # the agreement is never changed automatically
    assert any(e["event"] == "profile_updates_approved" and e["agreement_changed"] is False for e in service.get_run_log("uc3"))


def test_uc3_unclear_amount_is_flagged_for_the_rm(sample_profiles, transcript):
    extraction = MeetingExtraction(
        client_said=[RecordLine(text="Sarah wants to put S$50,000 into bonds.", source_lines=[2])],
        actions=[ExtractedAction(description="Rachel to move S$50,000 into bonds", owner="Rachel", due_date="Friday", amount="S$50,000", source_lines=[2])],
        email_subject="Follow-up", email_body="Dear Sarah,\n\nThanks for meeting.")
    service = new_service(FakeAdvisoryLLMClient(override_responses={"MeetingExtraction": extraction}))
    record_session(service, sample_profiles, transcript)
    record = service.get_state("uc3")["meeting_record"]
    assert record.actions[0].status == "FLAGGED" and "s$50000" in record.actions[0].flag_reason
    assert record.client_said[0].verified is False
    assert {c.rule_id: c.outcome for c in record.compliance_checks}["T1"] == "FLAG"
    assert "approve" not in service.get_pending_review("uc3")["allowed_actions"]


def test_uc3_line_that_does_not_trace_to_the_transcript_is_flagged(sample_profiles, transcript):
    extraction = MeetingExtraction(decisions=[RecordLine(text="Agreed to buy a bond fund.", source_lines=[99])],
                                   email_subject="Follow-up", email_body="Dear Sarah,\n\nThanks.")
    service = new_service(FakeAdvisoryLLMClient(override_responses={"MeetingExtraction": extraction}))
    record_session(service, sample_profiles, transcript)
    record = service.get_state("uc3")["meeting_record"]
    assert record.decisions[0].verified is False


def test_uc3_guarantee_wording_in_the_email_blocks_approval(sample_profiles, transcript):
    extraction = MeetingExtraction(email_subject="Follow-up", email_body="Dear Sarah,\n\nYour returns are guaranteed.")
    service = new_service(FakeAdvisoryLLMClient(override_responses={"MeetingExtraction": extraction}))
    view = record_session(service, sample_profiles, transcript)
    assert "approve" not in view.pending_review.allowed_actions
    view = act(service, "uc3", action="edit_record", edits={"email_body": "Dear Sarah,\n\nThanks for meeting."})
    assert "approve" in view.pending_review.allowed_actions


def test_uc3_spoken_domicile_change_is_checked_against_client_limits(sample_profiles, transcript):
    extraction = MeetingExtraction(
        changes=[CircumstanceChange(field="domicile", label="Domicile", after="US", text="I am relocating to the US.", source_lines=[4])],
        email_subject="Follow-up", email_body="Dear Sarah,\n\nThanks.")
    service = new_service(FakeAdvisoryLLMClient(override_responses={"MeetingExtraction": extraction}))
    record_session(service, sample_profiles, transcript)
    check = service.get_state("uc3")["meeting_record"].profile_changes[0].check
    assert check.status == "FLAG" and "Non-US-domiciled funds" in check.newly_blocked_areas


def test_uc3_needs_a_transcript(sample_profiles):
    with pytest.raises(ValueError, match="transcript"):
        new_service().start_session(sarah(sample_profiles, use_case="post_meeting"), session_id="none")


def test_uc3_reject_ends_without_a_record(sample_profiles, transcript):
    service = new_service()
    record_session(service, sample_profiles, transcript)
    view = act(service, "uc3", action="reject", notes="Wrong client.")
    assert view.outcome == "REJECTED" and service.get_pdf_path("uc3") is None
