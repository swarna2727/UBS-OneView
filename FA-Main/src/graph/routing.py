"""Pure conditional routing and review-action rules for the advisory state graph (slides 8 and 15).

Advisor review checkpoints follow the slide workflow:
  Client Profile -> [review] -> Market Intelligence -> [review] -> Investment Ideas -> Suitability Critic
  -> (FAIL: regenerate failed ideas automatically, max R10 times per idea) -> [recommendation review]
  -> Memo -> [final approval] -> PDF
Each review node records its decision in state["review_outcome"]; the routers map it to the next node.
"""

from typing import Literal, List, Optional, Dict, Any
from src.graph.state import AdvisoryState
from src.tools.risk_engine import load_policy

# Review checkpoint stage keys.
REVIEW_PROFILE = "client_profile"
REVIEW_MARKET = "market_intelligence"
REVIEW_RECOMMENDATIONS = "suitability_review"
REVIEW_MEMO = "advisor_approval"

MAX_REGENERATIONS_PER_IDEA = load_policy()["rules"]["R10_max_regenerations_per_idea"]

# Legacy action names accepted for backward compatibility.
_ACTION_ALIASES = {
    "accept": "approve",
    "resolve_conflict": "approve",
    "abort": "reject",
}

# Fields of the raw advisor intake the RM may edit when regenerating the profile.
EDITABLE_PROFILE_FIELDS = [
    "client_id", "name", "goals", "time_horizon_months", "stated_risk_willingness",
    "liquidity_schedule", "total_wealth", "experience_level", "constraints",
    "raw_holdings", "questionnaire_answers", "base_currency", "advisor_notes"
]

def normalize_action(stage: str, action: str) -> str:
    """Map legacy/alias action names onto the canonical action for a review stage."""
    if action == "override_flag" and stage == REVIEW_PROFILE:
        return "approve"
    return _ACTION_ALIASES.get(action, action)

def failed_slots(state: AdvisoryState) -> List[int]:
    assessment = state.get("assessment")
    if not assessment:
        return []
    return [ie.slot_number for ie in assessment.idea_evaluations if ie.status == "FAIL"]

def is_escalated(state: AdvisoryState) -> bool:
    """R10: a failed idea that has already been regenerated the maximum number of times goes to the advisor."""
    counts = state.get("slot_regen_counts") or {}
    return any(counts.get(slot, 0) >= MAX_REGENERATIONS_PER_IDEA for slot in failed_slots(state))

def allowed_review_actions(stage: str, state: AdvisoryState) -> List[str]:
    """Return the canonical actions the RM may take at a review checkpoint."""
    if stage == REVIEW_PROFILE:
        profile = state.get("profile")
        missing = profile and any(
            f.code == "MISSING_MANDATORY_FIELD" and not f.is_resolved for f in profile.unresolved_flags
        )
        # R1: missing mandatory data cannot be approved away; the RM must supply it and regenerate.
        return ["regenerate", "reject"] if missing else ["approve", "regenerate", "reject"]

    if stage == REVIEW_MARKET:
        return ["approve", "regenerate", "reject"]

    if stage == REVIEW_RECOMMENDATIONS:
        assessment = state.get("assessment")
        status = assessment.overall_status if assessment else "FAIL"
        # A FAIL can never reach the memo: the advisor can only revise, edit or reject.
        gate = {"PASS": ["approve"], "FLAG": ["override_flag"]}.get(status, [])
        return gate + ["revise", "edit_allocation", "regenerate", "reject"]

    if stage == REVIEW_MEMO:
        memo = state.get("memo")
        blocked = memo and any(r.outcome in ("FAIL", "BLOCK") for r in memo.compliance_checks)
        return ([] if blocked else ["approve"]) + ["regenerate", "revise", "edit_allocation", "reject"]

    return []

def resolve_review_outcome(stage: str, human_action: Dict[str, Any]) -> str:
    """
    Translate an RM action at a review checkpoint into a routing outcome:
    - 'approve': continue to the next step
    - 'regenerate': re-run the agent that was just reviewed (ideas: the whole set)
    - 'regenerate_ideas': send specific idea slots back to the Investment Idea Agent
    - 'revalidate': re-run the Suitability Critic after an allocation edit
    - 'reject': end the session
    """
    action = normalize_action(stage, human_action.get("action", ""))
    idea_id: Optional[str] = human_action.get("idea_id")

    if action in ("approve", "override_flag"):
        return "approve"
    if action == "edit_allocation":
        return "revalidate"
    if action == "regenerate":
        return "regenerate_ideas" if stage == REVIEW_RECOMMENDATIONS else "regenerate"
    if action == "revise":
        return "regenerate_ideas"
    if action == "reject":
        # Rejecting a single idea regenerates that slot; rejecting the stage ends the session.
        if idea_id and stage in (REVIEW_RECOMMENDATIONS, REVIEW_MEMO):
            return "regenerate_ideas"
        return "reject"
    raise ValueError(f"Unknown review action '{action}'")

def route_after_profile_review(state: AdvisoryState) -> Literal["market_intelligence", "client_profile", "end_session"]:
    return {"regenerate": "client_profile", "reject": "end_session"}.get(state.get("review_outcome"), "market_intelligence")

def route_after_market_review(state: AdvisoryState) -> Literal["investment_ideas", "market_intelligence", "end_session"]:
    return {"regenerate": "market_intelligence", "reject": "end_session"}.get(state.get("review_outcome"), "investment_ideas")

def route_after_critic(state: AdvisoryState) -> Literal["mark_failed_slots", "review_recommendations"]:
    """FAIL is regenerated automatically until R10's retry limit is hit; then the advisor decides."""
    assessment = state.get("assessment")
    if (assessment and assessment.overall_status == "FAIL" and failed_slots(state)
            and state.get("auto_regenerate", True) and not is_escalated(state)):
        return "mark_failed_slots"
    return "review_recommendations"

def route_after_recommendation_review(state: AdvisoryState) -> Literal["memo_formatter", "investment_ideas", "suitability_critic", "end_session"]:
    return {
        "regenerate_ideas": "investment_ideas",
        "revalidate": "suitability_critic",
        "reject": "end_session",
    }.get(state.get("review_outcome"), "memo_formatter")

def route_after_memo_review(state: AdvisoryState) -> Literal["generate_pdf", "memo_formatter", "investment_ideas", "suitability_critic", "end_session"]:
    return {
        "regenerate": "memo_formatter",
        "regenerate_ideas": "investment_ideas",
        "revalidate": "suitability_critic",
        "reject": "end_session",
    }.get(state.get("review_outcome"), "generate_pdf")
