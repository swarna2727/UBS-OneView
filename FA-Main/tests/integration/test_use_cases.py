"""Trajectory tests for the three use cases: separate triggers and paths over the same agents (layer T)."""

import copy
import json
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


def by_slot(idea_round):
    """The critic's status per slot for one idea round (idea ids are opaque handles)."""
    slots = {i["idea_id"]: i["slot_number"] for i in idea_round["ideas"]}
    return {slots[k]: v for k, v in idea_round["statuses"].items()}


def lock_all(service, sid):
    """CP3: the advisor locks in each idea individually; the last lock-in moves the set on."""
    view = None
    for idea in service.get_pending_review(sid)["candidate_ideas"]:
        view = act(service, sid, action="lock_idea", idea_id=idea["idea_id"])
    return view


def client(sample_profiles, key, **extra):
    return {**copy.deepcopy(sample_profiles[key]), **extra}


def test_pre_meeting_four_checkpoints_then_brief_with_facts(sample_profiles):
    """UC1: verify client data -> check limits -> review sources -> approve briefing. Facts only; no ideas by default."""
    service = new_service()
    sid = "uc1_changed"
    service.start_session(client(sample_profiles, "demo_sarah", use_case="pre_meeting"), session_id=sid)
    changes = service.get_state(sid)["profile_changes"]
    assert {c["field"] for c in changes} >= {"time_horizon_months", "goals"}
    assert [service.get_pending_review(sid)["stage"]] == ["client_profile"]
    act(service, sid, action="approve")
    assert service.get_pending_review(sid)["stage"] == "client_limits"
    act(service, sid, action="approve")
    assert service.get_pending_review(sid)["stage"] == "market_intelligence"
    view = act(service, sid, action="approve")  # a changed profile no longer adds ideas by itself
    assert view.pending_review.stage == "brief_approval"
    state = service.get_state(sid)
    brief = state["meeting_brief"]
    assert brief.recommendations == [] and brief.profile_changes and state.get("candidate_ideas") is None
    assert brief.portfolio_impact and brief.limits_summary  # what moved, and what UBS can offer
    assert state["research_brief"].window_days == 90 and state["research_brief"].external_news_used is False
    assert {c.rule_id: c.outcome for c in brief.compliance_checks} == {"R9": "PASS", "S1": "PASS"}
    assert service.get_pdf_path(sid) is None  # no PDF before approval
    view = act(service, sid, action="approve", notes="Approved for tomorrow.")
    assert view.outcome == "APPROVED" and os.path.exists(service.get_pdf_path(sid))
    assert service.get_state(sid).get("memo") is None  # the brief replaces the investment memo


def test_pre_meeting_unchanged_profile_has_no_recommendations(sample_profiles):
    service = new_service()
    sid = "uc1_unchanged"
    service.start_session(client(sample_profiles, "demo_mdm_lim", use_case="pre_meeting"), session_id=sid)
    assert service.get_state(sid)["profile_changes"] == []
    act(service, sid, action="approve", notes="Confirmed.")
    act(service, sid, action="approve")
    view = act(service, sid, action="approve")
    assert view.pending_review.stage == "brief_approval"
    state = service.get_state(sid)
    assert state.get("candidate_ideas") is None and state["meeting_brief"].recommendations == []
    assert all(c.outcome == "PASS" for c in state["meeting_brief"].compliance_checks)  # R9: no 'guaranteed' wording


def test_pre_meeting_advisor_can_opt_in_to_ideas(sample_profiles):
    service = new_service()
    sid = "uc1_override"
    service.start_session(client(sample_profiles, "demo_sarah", use_case="pre_meeting"), session_id=sid)
    act(service, sid, action="approve")
    act(service, sid, action="approve")
    view = act(service, sid, action="approve", edits={"include_ideas": True})
    assert view.pending_review.stage == "suitability_review"
    view = lock_all(service, sid)
    assert view.pending_review.stage == "brief_approval"
    assert len(service.get_state(sid)["meeting_brief"].recommendations) == 3


def test_pre_meeting_rm_selects_the_points_to_raise(sample_profiles):
    """UC1 checkpoint 4 (deck slide 3): only the talking points the RM selects go into the approved brief."""
    service = new_service()
    sid = "uc1_select"
    service.start_session(client(sample_profiles, "demo_sarah", use_case="pre_meeting"), session_id=sid)
    for _ in range(3):
        act(service, sid, action="approve")
    drafted = service.get_state(sid)["meeting_brief"].talking_points
    assert len(drafted) >= 2
    with pytest.raises(ValueError, match="talking points"):
        act(service, sid, action="approve", edits={"talking_points": []})
    with pytest.raises(ValueError, match="talking points"):
        act(service, sid, action="approve", edits={"talking_points": ["A point the RM typed in"]})
    view = act(service, sid, action="approve", edits={"talking_points": drafted[:1]})
    brief = service.get_state(sid)["meeting_brief"]
    assert view.outcome == "APPROVED" and brief.talking_points == drafted[:1] and brief.points_not_raised == drafted[1:]


def test_adhoc_request_adds_capital_and_regenerates_only_failed_idea(sample_profiles):
    """UC2 (deck slides 8, 9, 31): two aligned ideas and one exploratory. Idea 2 is blocked by client limits (E1),
    Ideas 1 and 3 are locked and only Idea 2 is regenerated, as short-duration investment-grade bonds."""
    service = new_service()
    sid = "uc2_sarah"
    service.start_session(client(sample_profiles, "demo_sarah", use_case="adhoc_request", new_capital_amount=120000.0), session_id=sid)
    profile = service.get_state(sid)["profile"]
    assert profile.total_wealth == 920000.0 and profile.new_capital_amount == 120000.0
    assert abs(sum(h.weight_pct for h in profile.parsed_holdings) - 100) < 0.01
    act(service, sid, action="approve")  # confirm client data
    act(service, sid, action="approve")  # check limits
    act(service, sid, action="approve")  # review sources
    state = service.get_state(sid)
    rounds = state["idea_rounds"]
    assert by_slot(rounds[0]) == {1: "PASS", 2: "FAIL", 3: "PASS"}
    assert rounds[1]["trigger"] == {"source": "auto_regeneration", "slots": [2]} and rounds[1]["overall_status"] == "PASS"
    assert [h["rules"] for h in state["regeneration_history"]] == [["E1"]]
    ideas = {i.slot_number: i for i in state["candidate_ideas"].ideas}
    assert [ideas[s].lane for s in (1, 2, 3)] == ["ALIGNED", "ALIGNED", "EXPLORATORY"]
    assert ideas[2].ticker == "VCSH" and ideas[2].indicative_amount_range == [30000.0, 40000.0]
    assert "COCO" not in json.dumps(service.get_pending_review(sid), default=str)  # the blocked product is never shown


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
