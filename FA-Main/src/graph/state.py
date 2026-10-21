"""Typed AdvisoryState for LangGraph orchestration."""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from src.schemas.advisory import (
    ClientAdvisoryProfile, ResearchBrief, CandidateIdeaSet,
    SuitabilityAssessment, StructuredMemo
)

def append_list(current: List[Any], new_items: List[Any]) -> List[Any]:
    """Reducer to append items to a list in state."""
    if not current:
        return list(new_items or [])
    if not new_items:
        return list(current)
    return list(current) + list(new_items)

class AdvisoryState(TypedDict, total=False):
    """Shared typed state across the multi-agent financial advisory graph."""
    session_id: str
    raw_input: Dict[str, Any]
    profile: Optional[ClientAdvisoryProfile]
    research_brief: Optional[ResearchBrief]
    candidate_ideas: Optional[CandidateIdeaSet]
    assessment: Optional[SuitabilityAssessment]
    memo: Optional[StructuredMemo]
    pdf_path: Optional[str]

    # Selective regeneration & locking controls
    locked_idea_ids: List[str]
    regenerate_slots: List[int]
    idea_feedback: Optional[str]
    regeneration_count: int
    slot_regen_counts: Dict[int, int]  # automatic regenerations per idea slot (R10)
    allocation_edited: bool  # set when the advisor edits an allocation; the critic re-validates it
    auto_regenerate: bool  # False after an advisor edit: a resulting FAIL goes back to the advisor
    market_feedback: Optional[str]

    # Governance & Human-In-The-Loop tracking
    advisor_notes: Annotated[List[str], append_list]
    audit_trail: Annotated[List[Dict[str, Any]], append_list]
    current_stage: str
    requires_human_action: bool
    human_action: Optional[Dict[str, Any]]
    review_outcome: Optional[str]  # approve | regenerate | regenerate_ideas | reject
    outcome: Optional[str]  # APPROVED | REJECTED once the session ends
    rejected_at_stage: Optional[str]
    is_completed: bool
    error_message: Optional[str]
