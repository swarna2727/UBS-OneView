"""Advisor Workspace state contract models (Slide 20)."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.schemas.tools import Breach

class WorkflowStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage_key: Literal["client_profile", "market_intelligence", "investment_ideas", "suitability_review", "advisor_approval", "final_memo"]
    display_name: str
    status: Literal["PENDING", "IN_PROGRESS", "WAITING_HUMAN", "COMPLETED", "FAILED"]
    description: str

class AllocationDiffRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category_type: str # "Asset Class", "Region", "Single Stock"
    name: str # e.g. "Equities", "North America", "MSFT"
    before_pct: float
    after_pct: float
    delta_pct: float
    is_breached: bool = False
    limit_value: Optional[float] = None

class DecisionSpacePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_stage: str
    active_agent_name: str
    stage_summary: str
    allocation_diffs: List[AllocationDiffRow] = Field(default_factory=list)
    risk_warnings: List[Breach] = Field(default_factory=list)
    requires_human_action: bool = False
    interrupt_reason: Optional[str] = None
    allowed_actions: List[str] = Field(default_factory=list) # ["accept", "reject", "revise", "add_advisor_note", "resolve_conflict"]
    data_payload: Dict[str, Any] = Field(default_factory=dict)

class MemoPreviewSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str
    title: str
    page_number: int
    content_html_or_md: str
    is_highlighted: bool = False
    affected_by_stage: str

class InteractiveMemoPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sections: List[MemoPreviewSection] = Field(default_factory=list)
    active_highlight_section_ids: List[str] = Field(default_factory=list)
    pdf_download_url: Optional[str] = None

class AdvisorWorkspaceView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    client_id: str
    client_name: str
    workflow_stages: List[WorkflowStage]
    center_panel: DecisionSpacePayload
    right_panel: InteractiveMemoPreview
    advisor_notes: List[str] = Field(default_factory=list)
    pending_review: Optional["PendingReview"] = None
    outcome: Optional[str] = None  # APPROVED | REJECTED once the session has ended

class AdvisorActionRequest(BaseModel):
    """
    RM command at a review checkpoint. Canonical actions are 'approve', 'regenerate' and 'reject';
    'accept', 'resolve_conflict' and 'abort' are accepted as legacy aliases.
    """
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "approve", "regenerate", "reject", "revise", "override_flag", "edit_allocation", "add_advisor_note",
        "maintain", "monitor",  # UC3 market-event decision ('revise' sends the client to new ideas)
        "accept", "resolve_conflict", "abort"
    ]
    idea_id: Optional[str] = None
    notes: Optional[str] = None
    conflict_resolution: Optional[str] = None
    # 'regenerate' at the profile review: corrected raw intake fields.
    # 'edit_allocation': {"target_weight_pct": <new weight>} for idea_id; the critic re-validates it.
    edits: Optional[Dict[str, Any]] = None

class PendingReview(BaseModel):
    """The agent output currently paused for Relationship Manager review."""
    model_config = ConfigDict(extra="forbid")
    stage: str
    agent: str
    reason: str
    output: Optional[Dict[str, Any]] = None
    allowed_actions: List[str] = Field(default_factory=list)
    unresolved_flags: List[Dict[str, Any]] = Field(default_factory=list)
    editable_fields: List[str] = Field(default_factory=list)
    raw_input: Dict[str, Any] = Field(default_factory=dict)
    candidate_ideas: List[Dict[str, Any]] = Field(default_factory=list)
    escalated: bool = False
    regeneration_history: List[Dict[str, Any]] = Field(default_factory=list)

AdvisorWorkspaceView.model_rebuild()
