"""Pure conditional routing functions for the advisory state graph."""

from typing import Literal
from src.graph.state import AdvisoryState

def route_after_profile(state: AdvisoryState) -> Literal["hitl_resolve_profile", "market_intelligence"]:
    """Route after ClientProfileAgent: if blocking flags exist, pause for RM resolution."""
    profile = state.get("profile")
    if not profile:
        return "hitl_resolve_profile"

    blocking_flags = [f for f in profile.unresolved_flags if f.severity == "BLOCKING_INTERRUPT"]
    if blocking_flags:
        return "hitl_resolve_profile"
    return "market_intelligence"

def route_after_critic(state: AdvisoryState) -> Literal["memo_formatter", "hitl_risk_exception", "mark_failed_slots"]:
    """
    Route after SuitabilityCriticAgent:
    - PASS: Route directly to Memo Formatter.
    - FLAG: Pause for RM exception review.
    - FAIL: Trigger selective regeneration if within retry limit, else escalate to RM.
    """
    assessment = state.get("assessment")
    if not assessment:
        return "hitl_risk_exception"

    overall_status = assessment.overall_status
    regen_count = state.get("regeneration_count", 0)
    max_regens = state.get("max_regenerations", 3)

    if overall_status == "PASS":
        return "memo_formatter"
    elif overall_status == "FLAG":
        return "hitl_risk_exception"
    elif overall_status == "FAIL":
        if regen_count < max_regens:
            return "mark_failed_slots"
        else:
            return "hitl_risk_exception"
    return "hitl_risk_exception"

def route_after_risk_exception(state: AdvisoryState) -> Literal["memo_formatter", "mark_failed_slots", "end_session"]:
    """Route based on RM decision during risk exception review."""
    human_action = state.get("human_action") or {}
    action_type = human_action.get("action", "")

    if action_type in ["override_flag", "accept"]:
        return "memo_formatter"
    elif action_type in ["reject", "revise"]:
        return "mark_failed_slots"
    elif action_type == "abort":
        return "end_session"
    return "memo_formatter"

def route_after_advisor_approval(state: AdvisoryState) -> Literal["end_session", "mark_failed_slots"]:
    """Route based on RM final review at advisor approval stage."""
    human_action = state.get("human_action") or {}
    action_type = human_action.get("action", "")

    if action_type in ["reject", "revise"]:
        return "mark_failed_slots"
    return "end_session"
