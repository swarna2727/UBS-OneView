"""UBS-Aligned Multi-Agent Advisory Schemas (Pydantic v2)."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from src.schemas.tools import (
    Holding, StandardizedHoldings, QuantMetrics, SentimentBreakdown, 
    DocumentSnippet, Breach, RiskMetrics, RuleResult, HoldingMove, ExcludedSource, MarketSignalConflict
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
    base_currency: str = "USD"
    near_term_liquidity_need: float = 0.0  # sum of liquidity events due within the liquidity-rule window
    advisor_notes: str = ""  # free-text advisor context; passed to agents as data, never as instructions
    new_capital_amount: float = 0.0  # UC2: new money to deploy, held as cash until the ideas are approved

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
    portfolio_impact: List[HoldingMove] = Field(default_factory=list)  # UC1: holdings moved over the window (facts only)
    window_days: int = 30  # look-back used for the freshness limit (UC1: 3 months)
    external_news_used: bool = False  # external news only by exception; the prototype never uses it
    rm_news_count: int = 0  # items the RM entered
    excluded_sources: List[ExcludedSource] = Field(default_factory=list)  # withdrawn, superseded or stale: retrieved but never used
    signal_conflicts: List[MarketSignalConflict] = Field(default_factory=list)  # deterministic: fed to the agent and the Critic
    conflicting_signals: List[str] = Field(default_factory=list)  # sources that disagree: surfaced as uncertainty, never resolved

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
    conviction: Literal["High", "Medium", "Low"] = "Medium"
    alternative_considered: str = ""  # AS-7: the alternative the advisor could choose instead
    is_locked: bool = False
    rejection_feedback: Optional[str] = None
    lane: Literal["ALIGNED", "EXPLORATORY"] = "ALIGNED"  # deck slide 9: slots 1-2 aligned, slot 3 exploratory
    indicative_amount_range: Optional[List[float]] = None  # [min, max] in the client's base currency
    required_disclosures: List[str] = Field(default_factory=list)  # attached by the client limits layer

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
    rule_results: List[RuleResult] = Field(default_factory=list)
    fit_check: Dict[str, str] = Field(default_factory=dict)  # risk range / horizon / concentration / liquidity -> PASS|FLAG|FAIL
    eligibility: Dict[str, str] = Field(default_factory=dict)  # client limits: eligible, disclosure, open gaps
    required_disclosures: List[str] = Field(default_factory=list)

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
    rule_results: List[RuleResult] = Field(default_factory=list)  # portfolio-level results

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
    conviction: str = "Medium"
    alternative_considered: str = ""

class MemoPage3PortfolioView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    portfolio_level_impact_narrative: str
    scenario_analysis: Dict[str, str] # e.g. {"Equity Selloff (-10%)": "Portfolio down ~4.2%", "Rate Hike (+50bps)": "Yield cushioned"}
    risk_disclosures: List[str]
    market_developments: str
    audit_sources_summary: List[str]
    advisor_signature_block: str

class ProfileChange(BaseModel):
    """One client fact that differs from the last review (UC1)."""
    model_config = ConfigDict(extra="forbid")
    field: str
    label: str
    before: str
    after: str

# ==========================================
# 6. USE CASE OUTPUTS: MEETING BRIEF (UC1) AND MARKET-EVENT DECISION (UC3)
# ==========================================

class BriefInsight(BaseModel):
    """One card of 'what the RM sees' (deck slide 9): what matters, and why it is in the briefing."""
    model_config = ConfigDict(extra="forbid")
    area: Literal["CLIENT", "PORTFOLIO", "MARKET", "DISCUSSION"]
    headline: str
    detail: str
    why_here: str  # where the point comes from: client profile, portfolio, CIO research (with dates)
    sources: List[str] = Field(default_factory=list)

class MeetingBrief(BaseModel):
    """UC1 client meeting brief: confirm the profile, share the market landscape, discuss next steps."""
    model_config = ConfigDict(extra="forbid")
    brief_id: str
    created_at: str
    client_name: str
    client_id: str
    meeting_purpose: str
    review_period: str
    profile_summary: List[str]
    profile_changes: List[ProfileChange] = Field(default_factory=list)
    key_insights: List[BriefInsight] = Field(default_factory=list)  # client, portfolio, market, discussion
    portfolio_summary: str
    portfolio_risks: List[str]
    market_developments: List[str]
    talking_points: List[str]
    points_not_raised: List[str] = Field(default_factory=list)  # drafted points the RM chose not to raise (kept for audit)
    questions_for_client: List[str]
    recommendations: List[ExecutiveIdeaGlance] = Field(default_factory=list)  # only when the profile changed
    portfolio_impact: List[HoldingMove] = Field(default_factory=list)  # facts only: what moved and by how much
    limits_summary: List[str] = Field(default_factory=list)  # what UBS can offer this client, gaps, disclosures
    open_actions: List[str] = Field(default_factory=list)  # carried from the last meeting's compliance record
    sources: List[str] = Field(default_factory=list)
    compliance_checks: List[RuleResult] = Field(default_factory=list)
    advisor_signature_block: str = "Pending Relationship Manager approval."
    advisor_approved: bool = False
    pdf_artifact_path: Optional[str] = None

class HoldingImpact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    name: str
    asset_class: str
    weight_pct: float
    shock_pct: float
    contribution_pct: float  # weight x shock, in portfolio percentage points

class EventImpact(BaseModel):
    """UC3 deterministic client impact of a market event: which holdings move, and does suitability still hold."""
    model_config = ConfigDict(extra="forbid")
    event_key: str
    event_title: str
    event_description: str
    holdings: List[HoldingImpact]
    portfolio_move_pct: float
    risk_score_before: float
    risk_score_after: float
    rule_results: List[RuleResult] = Field(default_factory=list)  # suitability re-check of the drifted portfolio
    suggested_decision: Literal["MAINTAIN", "MONITOR", "REVISE"]
    rationale: str

class EventDecisionRecord(BaseModel):
    """UC3 documented advisor decision when the advice is maintained or monitored (no new trades)."""
    model_config = ConfigDict(extra="forbid")
    record_id: str
    created_at: str
    client_name: str
    client_id: str
    event_title: str
    decision: Literal["MAINTAIN", "MONITOR", "REVISE"]
    rationale: str
    follow_up: str = ""
    portfolio_move_pct: float
    key_findings: List[str]
    sources: List[str] = Field(default_factory=list)
    advisor_signature_block: str = ""
    pdf_artifact_path: Optional[str] = None

class StructuredMemo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memo_id: str
    created_at: str
    page_1_executive: MemoPage1Executive
    page_2_details: List[MemoPage2IdeaDetail] = Field(..., min_length=3, max_length=3)
    page_3_portfolio_view: MemoPage3PortfolioView
    pdf_artifact_path: Optional[str] = None
    consistency_hash: str
    compliance_checks: List[RuleResult] = Field(default_factory=list)  # R9 wording + consistency validator
    advisor_approved: bool = False


# ==========================================
# 7. CLIENT LIMITS LAYER: WHAT UBS CAN OFFER (deck slide 12)
# ==========================================

class OfferedProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    name: str
    required_disclosures: List[str] = Field(default_factory=list)

class BlockedArea(BaseModel):
    """A category of product the client cannot be offered. Individual blocked products are never named to the RM."""
    model_config = ConfigDict(extra="forbid")
    area: str
    reason: str

class LimitGap(BaseModel):
    """Missing or lapsed regulatory data. Listed for the RM to resolve, never guessed."""
    model_config = ConfigDict(extra="forbid")
    code: Literal["MISSING_REGULATORY_FIELD", "CKA_MISSING", "CKA_EXPIRED", "CKA_EXPIRING", "TAX_FORM_MISSING",
                  "TAX_FORM_EXPIRED", "TAX_FORM_EXPIRING", "PEP_REVIEW", "NO_ADVICE_AGREEMENT"]
    message: str
    resolution: str

class OfferMap(BaseModel):
    """Structured output of the client limits layer: what UBS can offer this client."""
    model_config = ConfigDict(extra="forbid")
    client_id: str
    status: Literal["CLEAR", "GAPS", "ESCALATE", "BLOCKED"]
    regulatory_summary: Dict[str, str] = Field(default_factory=dict)
    eligible_products: List[OfferedProduct] = Field(default_factory=list)
    required_disclosures: List[str] = Field(default_factory=list)
    blocked_areas: List[BlockedArea] = Field(default_factory=list)
    gaps: List[LimitGap] = Field(default_factory=list)
    escalation_flags: List[str] = Field(default_factory=list)
    # Internal only: the Critic and the Opportunity Agent use it; the workspace never renders blocked tickers.
    blocked_tickers: List[str] = Field(default_factory=list)

    @property
    def eligible_tickers(self) -> List[str]:
        return [p.ticker for p in self.eligible_products]

    def rm_view(self) -> Dict[str, Any]:
        """The map as shown to the RM: blocked areas and gaps, never the blocked products themselves."""
        return self.model_dump(exclude={"blocked_tickers"})

# ==========================================
# 8. UC3: AFTER THE MEETING, NOTES / ACTIONS / COMPLIANCE RECORD (deck slides 13-15, 23)
# ==========================================

class RecordLine(BaseModel):
    """One factual line of the summary, traced to the transcript lines it comes from."""
    model_config = ConfigDict(extra="forbid")
    text: str
    source_lines: List[int] = Field(default_factory=list)
    verified: bool = True  # False: a cited line is missing, so the RM must check it

class CircumstanceChange(BaseModel):
    """A change in the client's circumstances spoken in the meeting. Extracted, not yet applied."""
    model_config = ConfigDict(extra="forbid")
    field: str  # income | employment | time_horizon_months | stated_risk_willingness | domicile | booking_centre | liquidity_need | other
    label: str
    before: str = ""
    after: str
    text: str
    source_lines: List[int] = Field(default_factory=list)

class ExtractedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    amount: Optional[str] = None
    source_lines: List[int] = Field(default_factory=list)

class MeetingExtraction(BaseModel):
    """What the Brief & Record Agent's model extracts from a transcript (before deterministic tracing and checks)."""
    model_config = ConfigDict(extra="forbid")
    client_said: List[RecordLine] = Field(default_factory=list)
    decisions: List[RecordLine] = Field(default_factory=list)
    changes: List[CircumstanceChange] = Field(default_factory=list)
    actions: List[ExtractedAction] = Field(default_factory=list)
    email_subject: str
    email_body: str

class RecordAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: str
    description: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    amount: Optional[str] = None
    status: Literal["CONFIRMED", "FLAGGED"]
    flag_reason: str = ""
    source_lines: List[int] = Field(default_factory=list)

class ProfileChangeCheck(BaseModel):
    """The Fit & Eligibility Critic's check of a spoken profile change against client limits and the risk band."""
    model_config = ConfigDict(extra="forbid")
    status: Literal["PASS", "FLAG", "FAIL"]
    implications: List[str] = Field(default_factory=list)
    new_gaps: List[str] = Field(default_factory=list)
    newly_blocked_areas: List[str] = Field(default_factory=list)
    escalation_flags: List[str] = Field(default_factory=list)

class ProfileChangeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    change_id: str
    field: str
    label: str
    before: str
    after: str
    text: str
    source_lines: List[int] = Field(default_factory=list)
    check: Optional[ProfileChangeCheck] = None
    decision: Literal["PENDING", "APPROVED", "REJECTED"] = "PENDING"
    decision_notes: str = ""

class FollowUpEmail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    body: str
    edited_by_rm: bool = False

class MeetingRecord(BaseModel):
    """UC3 output: confirmed summary, agreed actions, draft follow-up email and the compliance record."""
    model_config = ConfigDict(extra="forbid")
    record_id: str
    created_at: str
    client_name: str
    client_id: str
    meeting_date: str
    meeting_purpose: str
    agreement_type: str
    transcript_fingerprint: str
    transcript_lines: int
    client_said: List[RecordLine] = Field(default_factory=list)
    decisions: List[RecordLine] = Field(default_factory=list)
    actions: List[RecordAction] = Field(default_factory=list)
    profile_changes: List[ProfileChangeProposal] = Field(default_factory=list)
    follow_up_email: FollowUpEmail
    compliance_checks: List[RuleResult] = Field(default_factory=list)
    agreement_changed: bool = False  # invariant: the client's agreement is never changed automatically
    rm_confirmed: bool = False
    advisor_signature_block: str = "Pending Relationship Manager confirmation."
    pdf_artifact_path: Optional[str] = None
