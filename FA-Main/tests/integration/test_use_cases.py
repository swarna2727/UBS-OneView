"""Trajectory tests for the three use cases: separate triggers and paths over the same agents (layer T)."""

import copy
import os
import pytest
from src.api.service import AdvisorySessionService
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.workspace import AdvisorActionRequest
from src.tools.event_tools import MARKET_EVENTS


def new_service():
    return AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())


def act(service, sid, **kwargs):
    return service.execute_action(sid, AdvisorActionRequest(**kwargs))


def lock_all(service, sid):
    """CP3: the advisor locks in each idea individually; the last lock-in moves the set on."""
    view = None
    for idea in service.get_pending_review(sid)["candidate_ideas"]:
        view = act(service, sid, action="lock_idea", idea_id=idea["idea_id"])
    return view


def client(sample_profiles, key, **extra):
    return {**copy.deepcopy(sample_profiles[key]), **extra}


def test_pre_meeting_changed_profile_generates_ideas_then_brief(sample_profiles):
    """UC1: a change since the last review routes through ideas and the critic before the meeting brief."""
    service = new_service()
    sid = "uc1_changed"
    service.start_session(client(sample_profiles, "demo_sarah", use_case="pre_meeting"), session_id=sid)
    changes = service.get_state(sid)["profile_changes"]
    assert {c["field"] for c in changes} >= {"time_horizon_months", "goals"}
    act(service, sid, action="approve")
    view = act(service, sid, action="approve")  # include_ideas defaults to True because the profile changed
    assert view.pending_review.stage == "suitability_review"
    view = lock_all(service, sid)
    assert view.pending_review.stage == "brief_approval"
    brief = service.get_state(sid)["meeting_brief"]
    assert len(brief.recommendations) == 3 and brief.profile_changes
    assert service.get_pdf_path(sid) is None  # no PDF before approval
    view = act(service, sid, action="approve", notes="Approved for tomorrow.")
    assert view.outcome == "APPROVED" and os.path.exists(service.get_pdf_path(sid))
    assert service.get_state(sid).get("memo") is None  # the brief replaces the investment memo


def test_pre_meeting_unchanged_profile_skips_ideas(sample_profiles):
    """UC1: no change since the last review -> straight from research to the brief, no recommendations."""
    service = new_service()
    sid = "uc1_unchanged"
    service.start_session(client(sample_profiles, "demo_mdm_lim", use_case="pre_meeting"), session_id=sid)
    assert service.get_state(sid)["profile_changes"] == []
    act(service, sid, action="approve", notes="Confirmed.")
    view = act(service, sid, action="approve")
    assert view.pending_review.stage == "brief_approval"
    state = service.get_state(sid)
    assert state.get("candidate_ideas") is None and state["meeting_brief"].recommendations == []
    assert all(c.outcome == "PASS" for c in state["meeting_brief"].compliance_checks)  # R9: no 'guaranteed' wording


def test_pre_meeting_advisor_can_request_ideas_without_changes(sample_profiles):
    service = new_service()
    sid = "uc1_override"
    service.start_session(client(sample_profiles, "demo_mdm_lim", use_case="pre_meeting"), session_id=sid)
    act(service, sid, action="approve")
    view = act(service, sid, action="approve", edits={"include_ideas": True})
    assert view.pending_review.stage == "suitability_review"


def test_adhoc_request_adds_capital_and_regenerates_only_failed_idea(sample_profiles):
    """UC2: new capital is held as cash; Idea 2 fails R4, Ideas 1 and 3 are locked, only slot 2 is regenerated."""
    service = new_service()
    sid = "uc2_sarah"
    service.start_session(client(sample_profiles, "demo_sarah", use_case="adhoc_request", new_capital_amount=120000.0), session_id=sid)
    profile = service.get_state(sid)["profile"]
    assert profile.total_wealth == 920000.0 and profile.new_capital_amount == 120000.0
    assert abs(sum(h.weight_pct for h in profile.parsed_holdings) - 100) < 0.01
    act(service, sid, action="approve")
    act(service, sid, action="approve")
    rounds = service.get_state(sid)["idea_rounds"]
    assert rounds[0]["statuses"] == {"IDEA-ACWI-01": "PASS", "IDEA-HYG-02": "FAIL", "IDEA-GLD-03": "PASS"}
    assert rounds[1]["trigger"] == {"source": "auto_regeneration", "slots": [2]} and rounds[1]["overall_status"] == "PASS"
    assert [h["rules"] for h in service.get_state(sid)["regeneration_history"]] == [["R4"]]


def test_market_event_monitor_records_decision_without_trades(sample_profiles):
    """UC3: event research -> deterministic impact -> advisor decision; monitor needs a follow-up trigger."""
    event = MARKET_EVENTS["rates_up"]
    service = new_service()
    sid = "uc3_monitor"
    view = service.start_session(client(sample_profiles, "demo_sarah", use_case="market_event",
                                        market_event={"key": "rates_up", "title": event["title"], "description": event["description"]}),
                                 session_id=sid)
    assert view.pending_review.stage == "event_decision"  # no profile checkpoint for a confirmed client record
    impact = service.get_state(sid)["event_impact"]
    assert impact.portfolio_move_pct < 0 and impact.suggested_decision == "MONITOR"
    assert service.get_state(sid)["research_brief"].findings[0].theme == "Rates repricing"
    with pytest.raises(ValueError, match="rationale"):
        act(service, sid, action="maintain")
    with pytest.raises(ValueError, match="follow-up"):
        act(service, sid, action="monitor", notes="Watching rates.")
    view = act(service, sid, action="monitor", notes="Watching rates.", edits={"follow_up": "Review by 30 Nov"})
    state = service.get_state(sid)
    assert view.outcome == "APPROVED" and state["decision_record"].decision == "MONITOR"
    assert state.get("candidate_ideas") is None and os.path.exists(service.get_pdf_path(sid))


def test_market_event_revise_runs_ideas_critic_and_memo(sample_profiles):
    service = new_service()
    sid = "uc3_revise"
    service.start_session(client(sample_profiles, "demo_sarah", use_case="market_event", market_event={"key": "rates_up"}), session_id=sid)
    view = act(service, sid, action="revise", notes="Reduce duration.")
    assert view.pending_review.stage == "suitability_review"
    assert service.get_state(sid)["event_decision"]["decision"] == "REVISE"
    lock_all(service, sid)
    view = act(service, sid, action="approve")
    assert view.outcome == "APPROVED" and service.get_state(sid)["memo"] is not None


def test_market_event_with_missing_data_still_blocks_at_profile(sample_profiles):
    view = new_service().start_session(client(sample_profiles, "demo_aaron", use_case="market_event"), session_id="uc3_aaron")
    assert view.pending_review.stage == "client_profile" and "approve" not in view.pending_review.allowed_actions


def test_unknown_use_case_is_rejected(sample_profiles):
    with pytest.raises(ValueError, match="use_case"):
        new_service().start_session(client(sample_profiles, "demo_sarah", use_case="newsletter"))
