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

class AdvisorActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["accept", "reject", "revise", "add_advisor_note", "resolve_conflict", "override_flag"]
    idea_id: Optional[str] = None
    notes: Optional[str] = None
    conflict_resolution: Optional[str] = None
