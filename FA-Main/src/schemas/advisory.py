"""UBS-Aligned Multi-Agent Advisory Schemas (Pydantic v2)."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from src.schemas.tools import (
    Holding, StandardizedHoldings, QuantMetrics, SentimentBreakdown, 
    DocumentSnippet, Breach, RiskMetrics
)

# ==========================================
# 1. CLIENT ADVISORY PROFILE SCHEMAS (Slide 14)
# ==========================================

class LiquidityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    due_months: int = Field(..., ge=1)
    amount: float = Field(..., ge=0.0)
    purpose: str

class ProfileFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: Literal["MISSING_MANDATORY_FIELD", "RISK_CAPACITY_CONFLICT", "CONCENTRATION_WARNING", "HORIZON_MISMATCH"]
    severity: Literal["WARNING", "BLOCKING_INTERRUPT"]
    message: str
    affected_fields: List[str] = Field(default_factory=list)
    resolution_notes: Optional[str] = None
    is_resolved: bool = False

class ClientConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    excluded_sectors: List[str] = Field(default_factory=list)
    excluded_countries: List[str] = Field(default_factory=list)
    esg_mandate: bool = False
    max_single_stock_pct: float = Field(default=20.0, ge=1.0, le=100.0)

class ClientAdvisoryProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_id: str
    name: str
    goals: List[str] = Field(..., min_length=1)
    time_horizon_months: int = Field(..., ge=1)
    stated_risk_willingness: str
    calculated_risk_tier: str # e.g. "Conservative", "Moderate", "Balanced Growth", "Aggressive Growth", "Speculative"
    risk_tier_number: int = Field(..., ge=1, le=5)
    risk_score_range: List[float] = Field(..., min_length=2, max_length=2) # [min, max]
    liquidity_schedule: List[LiquidityEvent] = Field(default_factory=list)
    total_wealth: float = Field(..., ge=0.0)
    experience_level: str
    constraints: ClientConstraints
    parsed_holdings: List[Holding] = Field(default_factory=list)
    current_asset_class_weights: Dict[str, float] = Field(default_factory=dict)
    current_region_weights: Dict[str, float] = Field(default_factory=dict)
    current_currency_weights: Dict[str, float] = Field(default_factory=dict)
    unresolved_flags: List[ProfileFlag] = Field(default_factory=list)

    @field_validator("risk_score_range")
    @classmethod
    def validate_risk_range(cls, v: List[float]) -> List[float]:
        if v[0] > v[1]:
            raise ValueError("risk_score_range min must be <= max")
        return v

# ==========================================
# 2. MARKET INTELLIGENCE SCHEMAS (Slide 15)
# ==========================================

class ResearchFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str
    theme: str
    claim: str
    materiality_rank: int = Field(..., ge=1)
    relevance_to_client: str
    finding_type: Literal["QUANTITATIVE", "QUALITATIVE_SENTIMENT", "MACRO_POLICY"]
    cited_source_ids: List[str] = Field(..., min_length=1)
    publication_dates: List[str] = Field(..., min_length=1)
    freshness_verified: bool = True

class ResearchBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    macro_themes: List[str] = Field(..., min_length=1)
    quantitative_signals: Dict[str, QuantMetrics] = Field(default_factory=dict)
    sentiment_signals: Dict[str, SentimentBreakdown] = Field(default_factory=dict)
    portfolio_gaps_addressed: List[str] = Field(default_factory=list)
    findings: List[ResearchFinding] = Field(..., min_length=1)
    key_market_risks: List[str] = Field(..., min_length=1)
    validated_sources: List[DocumentSnippet] = Field(..., min_length=1)

# ==========================================
# 3. INVESTMENT IDEA SCHEMAS (Slide 16)
# ==========================================

class AllocationRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_pct: float = Field(..., ge=0.0, le=100.0)
    max_pct: float = Field(..., ge=0.0, le=100.0)
    target_pct: float = Field(..., ge=0.0, le=100.0)

    @model_validator(mode="after")
    def check_min_max(self) -> "AllocationRange":
        if self.min_pct > self.max_pct:
            raise ValueError(f"min_pct ({self.min_pct}) cannot exceed max_pct ({self.max_pct})")
        if not (self.min_pct <= self.target_pct <= self.max_pct):
            raise ValueError(f"target_pct ({self.target_pct}) must be within [min_pct, max_pct]")
        return self

class InvestmentIdea(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idea_id: str
    slot_number: int = Field(..., ge=1, le=3)
    instrument_id: str
    ticker: str
    name: str
    action: Literal["BUY", "ADD", "TRIM", "SELL"]
    target_weight_pct: float = Field(..., ge=0.0, le=100.0)
    allocation_range: AllocationRange
    funding_source_ticker: str = Field(default="BIL") # e.g. "BIL" (Cash) or holding to trim
    portfolio_role: Literal["GROWTH", "INCOME", "STABILITY", "DIVERSIFICATION"]
    thesis: str
    why_now: str
    client_fit_rationale: str
    key_risks: List[str] = Field(..., min_length=1)
    invalidation_trigger: str
    cited_evidence_ids: List[str] = Field(..., min_length=1)
    is_locked: bool = False
    rejection_feedback: Optional[str] = None

class CandidateIdeaSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ideas: List[InvestmentIdea] = Field(...)
    complementarity_thesis: str
    generation_round: int = Field(default=1, ge=1)

    @field_validator("ideas")
    @classmethod
    def validate_exactly_three_ideas(cls, v: List[InvestmentIdea]) -> List[InvestmentIdea]:
        if len(v) != 3:
            raise ValueError(f"CandidateIdeaSet must contain exactly 3 ideas, got {len(v)}")
        slots = [idea.slot_number for idea in v]
        if set(slots) != {1, 2, 3}:
            raise ValueError(f"CandidateIdeaSet must have ideas assigned to slots 1, 2, and 3. Got {slots}")
        tickers = [idea.ticker for idea in v]
        if len(set(tickers)) != 3:
            raise ValueError(f"CandidateIdeaSet ideas must be for 3 distinct instruments. Got {tickers}")
        return v

# ==========================================
# 4. SUITABILITY CRITIC SCHEMAS (Slide 17)
# ==========================================

class IdeaStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idea_id: str
    slot_number: int
    ticker: str
    status: Literal["PASS", "FLAG", "FAIL"]
    detected_breaches: List[Breach] = Field(default_factory=list)
    qualitative_notes: str
    evidence_verified: bool = True

class SuitabilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_status: Literal["PASS", "FLAG", "FAIL"]
    idea_evaluations: List[IdeaStatus] = Field(..., min_length=3, max_length=3)
    before_metrics: RiskMetrics
    after_metrics: RiskMetrics
    detected_breaches: List[Breach] = Field(default_factory=list)
    required_actions: List[str] = Field(default_factory=list)
    has_blocking_hard_breach: bool = False
    critic_notes: str

# ==========================================
# 5. MEMO FORMATTER SCHEMAS (Slide 18)
# ==========================================

class ExecutiveIdeaGlance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idea_id: str
    slot_number: int
    ticker: str
    name: str
    action: str
    target_weight_pct: float
    portfolio_role: str
    why_it_fits: str

class MemoPage1Executive(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_name: str
    client_id: str
    date: str
    investment_objective: str
    client_snapshot: str
    portfolio_snapshot: str
    ideas_at_a_glance: List[ExecutiveIdeaGlance] = Field(..., min_length=3, max_length=3)

class MemoPage2IdeaDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_number: int
    ticker: str
    name: str
    action: str
    allocation_range_str: str
    target_weight_pct: float
    portfolio_role: str
    thesis: str
    why_now: str
    client_fit_rationale: str
    key_risks: List[str]
    what_changes_view: str
    cited_sources_formatted: List[str]

class MemoPage3PortfolioView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    portfolio_level_impact_narrative: str
    scenario_analysis: Dict[str, str] # e.g. {"Equity Selloff (-10%)": "Portfolio down ~4.2%", "Rate Hike (+50bps)": "Yield cushioned"}
    risk_disclosures: List[str]
    market_developments: str
    audit_sources_summary: List[str]
    advisor_signature_block: str

class StructuredMemo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memo_id: str
    created_at: str
    page_1_executive: MemoPage1Executive
    page_2_details: List[MemoPage2IdeaDetail] = Field(..., min_length=3, max_length=3)
    page_3_portfolio_view: MemoPage3PortfolioView
    pdf_artifact_path: Optional[str] = None
    consistency_hash: str
