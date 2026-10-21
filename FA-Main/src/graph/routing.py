"""Pure conditional routing and review-action rules for the advisory state graph.

Every agent node is followed by a human review checkpoint. The review node
records its decision in ``state["review_outcome"]`` and the routers below map
that decision onto the next node.
"""

from typing import Literal, List, Optional, Dict, Any
from src.graph.state import AdvisoryState

# Review checkpoint stage keys (one per agent).
REVIEW_PROFILE = "client_profile"
REVIEW_MARKET = "market_intelligence"
REVIEW_IDEAS = "investment_ideas"
REVIEW_SUITABILITY = "suitability_review"
REVIEW_MEMO = "advisor_approval"

# Legacy action names accepted for backward compatibility, mapped to the
# canonical review action for a given stage.
_ACTION_ALIASES = {
    "accept": "approve",
    "resolve_conflict": "approve",
    "abort": "reject",
}

# Fields of the raw advisor intake the RM may edit when regenerating the profile.
EDITABLE_PROFILE_FIELDS = [
    "client_id", "name", "goals", "time_horizon_months", "stated_risk_willingness",
    "liquidity_schedule", "total_wealth", "experience_level", "constraints",
    "raw_holdings", "questionnaire_answers"
]

def normalize_action(stage: str, action: str) -> str:
    """Map legacy/alias action names onto the canonical action for a review stage."""
    if action == "override_flag" and stage == REVIEW_PROFILE:
        return "approve"
    return _ACTION_ALIASES.get(action, action)

def allowed_review_actions(stage: str, state: AdvisoryState) -> List[str]:
    """Return the canonical actions the RM may take at a review checkpoint."""
    if stage == REVIEW_PROFILE:
        profile = state.get("profile")
        missing = profile and any(
            f.code == "MISSING_MANDATORY_FIELD" and not f.is_resolved for f in profile.unresolved_flags
        )
        # Missing mandatory data cannot be approved away: the RM must supply it and regenerate.
        return ["regenerate", "reject"] if missing else ["approve", "regenerate", "reject"]

    if stage == REVIEW_MARKET:
        return ["approve", "regenerate", "reject"]

    if stage == REVIEW_IDEAS:
        return ["approve", "regenerate", "revise", "reject"]

    if stage == REVIEW_SUITABILITY:
        assessment = state.get("assessment")
        status = assessment.overall_status if assessment else "FAIL"
        actions: List[str] = []
        if status == "PASS":
            actions.append("approve")
        elif status == "FLAG":
            actions.append("override_flag")
        elif state.get("regeneration_count", 0) >= state.get("max_regenerations", 3):
            # Hard breaches can only be overridden once automatic regeneration is exhausted.
            actions.append("override_flag")
        actions.extend(["revise", "regenerate", "reject"])
        return actions

    if stage == REVIEW_MEMO:
        return ["approve", "regenerate", "revise", "reject"]

    return []

def resolve_review_outcome(stage: str, human_action: Dict[str, Any]) -> str:
    """
    Translate an RM action at a review checkpoint into a routing outcome:
    - 'approve': continue to the next agent
    - 'regenerate': re-run the agent that was just reviewed
    - 'regenerate_ideas': send specific (or failed) idea slots back to the Investment Idea Agent
    - 'reject': end the session
    """
    action = normalize_action(stage, human_action.get("action", ""))
    idea_id: Optional[str] = human_action.get("idea_id")

    if action in ("approve", "override_flag"):
        return "approve"
    if action == "regenerate":
        if stage == REVIEW_IDEAS:
            return "regenerate_ideas"
        return "regenerate"
    if action == "revise":
        return "regenerate_ideas"
    if action == "reject":
        # Rejecting a single idea regenerates that slot; rejecting the stage ends the session.
        if idea_id and stage in (REVIEW_IDEAS, REVIEW_SUITABILITY, REVIEW_MEMO):
            return "regenerate_ideas"
        return "reject"
    return "approve"

def route_after_profile_review(state: AdvisoryState) -> Literal["market_intelligence", "client_profile", "end_session"]:
    outcome = state.get("review_outcome")
    if outcome == "regenerate":
        return "client_profile"
    if outcome == "reject":
        return "end_session"
    return "market_intelligence"

def route_after_market_review(state: AdvisoryState) -> Literal["investment_ideas", "market_intelligence", "end_session"]:
    outcome = state.get("review_outcome")
    if outcome == "regenerate":
        return "market_intelligence"
    if outcome == "reject":
        return "end_session"
    return "investment_ideas"

def route_after_ideas_review(state: AdvisoryState) -> Literal["suitability_critic", "investment_ideas", "end_session"]:
    outcome = state.get("review_outcome")
    if outcome == "regenerate_ideas":
        return "investment_ideas"
    if outcome == "reject":
        return "end_session"
    return "suitability_critic"

def route_after_suitability_review(state: AdvisoryState) -> Literal["memo_formatter", "suitability_critic", "mark_failed_slots", "end_session"]:
    outcome = state.get("review_outcome")
    if outcome == "regenerate":
        return "suitability_critic"
    if outcome == "regenerate_ideas":
        return "mark_failed_slots"
    if outcome == "reject":
        return "end_session"
    return "memo_formatter"

def route_after_memo_review(state: AdvisoryState) -> Literal["end_session", "memo_formatter", "mark_failed_slots"]:
    outcome = state.get("review_outcome")
    if outcome == "regenerate":
        return "memo_formatter"
    if outcome == "regenerate_ideas":
        return "mark_failed_slots"
    return "end_session"
