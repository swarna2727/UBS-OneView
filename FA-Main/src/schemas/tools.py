"""Deterministic tool data transfer models."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_valid: bool
    missing_fields: List[str] = Field(default_factory=list)
    invalid_fields: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

class CalculatedRiskTier(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float = Field(..., ge=1.0, le=5.0)
    risk_tier: str # e.g. "Conservative", "Moderate", "Balanced Growth", "Aggressive Growth", "Speculative"
    risk_tier_number: int = Field(..., ge=1, le=5)
    capacity_score: float = Field(..., ge=1.0, le=5.0)
    willingness_score: float = Field(..., ge=1.0, le=5.0)
    recommended_min_horizon_months: int
    has_capacity_conflict: bool = False
    conflict_rationale: Optional[str] = None

class Holding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    name: str
    asset_class: str # "Equities", "Fixed Income", "Commodities", "Real Estate", "Cash Equivalents"
    region: str # "North America", "Europe", "Asia-Pacific", "Global", "Emerging Markets"
    currency: str # "USD", "EUR", "CHF", "GBP"
    weight_pct: float = Field(..., ge=0.0, le=100.0)
    risk_rating: int = Field(default=3, ge=1, le=7)
    complexity_rating: int = Field(default=1, ge=1, le=5)
    liquidity_tier: str = Field(default="Daily")
    sector: str = Field(default="Diversified")  # e.g. "Technology"; funds are "Diversified"
    instrument_type: Literal["Single Stock", "Fund", "Cash"] = "Fund"

class StandardizedHoldings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    holdings: List[Holding]
    total_weight_pct: float
    asset_class_weights: Dict[str, float]
    region_weights: Dict[str, float]
    currency_weights: Dict[str, float]
    weighted_risk_score: float
    is_balanced_sum: bool

class QuantMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    annualized_return_1y: Optional[float] = None
    annualized_return_3y: Optional[float] = None
    annualized_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    momentum_12_1: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    effective_duration: Optional[float] = None
    yield_to_maturity: Optional[float] = None
    credit_spread_bps: Optional[int] = None

class SentimentBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_sentiment: Literal["BULLISH", "NEUTRAL", "BEARISH"]
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    bullish_signals: List[str] = Field(default_factory=list)
    bearish_signals: List[str] = Field(default_factory=list)
    neutral_catalysts: List[str] = Field(default_factory=list)

class DocumentSnippet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    title: str
    publisher: str
    published_at: str
    retrieved_at: str
    text: str
    tickers: List[str] = Field(default_factory=list)
    asset_classes: List[str] = Field(default_factory=list)
    macro_themes: List[str] = Field(default_factory=list)
    is_fresh: bool = True
    age_days: Optional[int] = None

class ValidatedSnippets(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kept_snippets: List[DocumentSnippet]
    discarded_snippets: List[Dict[str, Any]] = Field(default_factory=list)
    rejection_reasons: Dict[str, str] = Field(default_factory=dict)

class ApprovedInstrument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instrument_id: str
    ticker: str
    name: str
    asset_class: str
    sub_asset_class: str
    region: str
    currency: str
    risk_rating: int
    complexity_rating: int
    liquidity_tier: str
    is_approved: bool
    status: str
    esg_rating: Optional[str] = None
    sector: str = "Diversified"
    instrument_type: str = "Fund"

class ExposureDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before_holdings: List[Holding]
    after_holdings: List[Holding]
    asset_class_delta: Dict[str, float] # e.g. {"Equities": +5.0, "Cash Equivalents": -5.0}
    region_delta: Dict[str, float]
    currency_delta: Dict[str, float]
    single_stock_weights: Dict[str, float] # after weights
    top_concentration_ticker: str
    top_concentration_weight: float
    total_after_weight: float
    unnormalized_after_weight: float = 100.0  # trade-set total before re-normalisation (R3 funding check)

class Breach(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str
    severity: Literal["FLAG", "FAIL"]
    metric_name: str
    before_value: float
    after_value: float
    threshold_value: float
    message: str

class RiskMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before_risk_score: float
    after_risk_score: float
    risk_score_delta: float
    max_single_stock_pct: float
    max_single_stock_ticker: str
    us_equity_pct: float
    liquidity_coverage_ratio: float
    unhedged_foreign_currency_pct: float
    growth_asset_pct: float = 0.0

class RuleResult(BaseModel):
    """Outcome of one suitability rule (R1-R10, L1) for the portfolio or a single idea."""
    model_config = ConfigDict(extra="forbid")
    rule_id: str
    name: str
    outcome: Literal["PASS", "FLAG", "FAIL", "BLOCK"]
    message: str
    idea_id: Optional[str] = None
    metric: Optional[str] = None
    before_value: Optional[float] = None
    after_value: Optional[float] = None
    threshold_value: Optional[float] = None

class RuleCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_status: Literal["PASS", "FLAG", "FAIL"]
    breaches: List[Breach] = Field(default_factory=list)
    passed_rules: List[str] = Field(default_factory=list)
