"""Deterministic risk engine and suitability rule engine (Slide 17)."""

import os
import yaml
from typing import List, Dict, Any, Tuple
from src.schemas.tools import Holding, RiskMetrics, RuleCheckResult, Breach
from src.tools.exposure import portfolio_exposure_calculator

CONFIG_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config", "suitability_policy.yaml"
)

def _load_policy() -> Dict[str, Any]:
    if os.path.exists(CONFIG_POLICY_PATH):
        with open(CONFIG_POLICY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "limits": {
            "single_stock": {"flag_threshold_pct": 20.0, "fail_threshold_pct": 30.0},
            "asset_class_drift": {"flag_threshold_pp": 10.0, "fail_threshold_pp": 20.0},
            "regional_concentration": {"us_equity_flag_pct": 50.0, "us_equity_fail_pct": 65.0},
            "risk_score_drift": {"flag_threshold": 0.3, "fail_threshold": 0.6},
            "liquidity_coverage": {"flag_min_coverage_ratio": 1.2, "fail_min_coverage_ratio": 1.0},
            "fx_exposure": {"unhedged_flag_pct": 25.0}
        }
    }

def calculate_portfolio_risk_metrics(
    holdings: List[Holding],
    liquidity_needs_amount: float = 100000.0,
    total_wealth: float = 1000000.0
) -> Tuple[float, float, str, float, float, float]:
    """Calculate core portfolio metrics: risk score, max single stock, US equity %, liquidity ratio, FX %."""
    total_wt = sum(h.weight_pct for h in holdings)
    if total_wt <= 0:
        return 1.0, 0.0, "NONE", 0.0, 999.0, 0.0

    weighted_risk = sum(h.risk_rating * h.weight_pct for h in holdings) / total_wt

    # Single stock concentration (excluding Cash / T-Bills / diversified ETFs like IEF, LQD, BIL)
    single_stocks = [
        h for h in holdings
        if h.asset_class == "Equities" and not h.ticker.startswith("ETF")
    ]
    if single_stocks:
        top_single = max(single_stocks, key=lambda x: x.weight_pct)
        max_single_stock_pct = top_single.weight_pct
        max_single_ticker = top_single.ticker
    else:
        max_single_stock_pct = 0.0
        max_single_ticker = "NONE"

    # US Equity concentration
    us_equity_pct = sum(
        h.weight_pct for h in holdings
        if h.asset_class == "Equities" and h.region in ["North America", "US"]
    )

    # Liquidity adequacy: liquid assets (Cash + T-Bills) vs near-term liquidity need
    liquid_holdings_value = sum(
        (h.weight_pct / 100.0) * total_wealth
        for h in holdings
        if h.liquidity_tier == "Daily" and h.asset_class in ["Cash Equivalents", "Fixed Income"]
    )
    liquidity_ratio = liquid_holdings_value / max(liquidity_needs_amount, 1.0)

    # Foreign currency unhedged (non-USD holdings)
    foreign_ccy_pct = sum(
        h.weight_pct for h in holdings if h.currency not in ["USD"]
    )

    return (
        round(weighted_risk, 2),
        round(max_single_stock_pct, 2),
        max_single_ticker,
        round(us_equity_pct, 2),
        round(liquidity_ratio, 2),
        round(foreign_ccy_pct, 2)
    )

def portfolio_risk_engine(
    current_portfolio: List[Holding],
    candidate_ideas: List[Any],
    liquidity_needs_amount: float = 100000.0,
    total_wealth: float = 1000000.0
) -> RiskMetrics:
    """
    Simulate after-portfolio and calculate Before vs After RiskMetrics.
    """
    trades = []
    for idea in candidate_ideas:
        if hasattr(idea, "ticker"):
            trades.append({
                "ticker": idea.ticker,
                "action": idea.action,
                "target_weight_pct": idea.target_weight_pct,
                "funding_source_ticker": getattr(idea, "funding_source_ticker", "BIL")
            })
        elif isinstance(idea, dict):
            trades.append({
                "ticker": idea.get("ticker"),
                "action": idea.get("action", "BUY"),
                "target_weight_pct": idea.get("target_weight_pct", 5.0),
                "funding_source_ticker": idea.get("funding_source_ticker", "BIL")
            })

    exposure_delta = portfolio_exposure_calculator(current_portfolio, trades)

    # Before metrics
    b_risk, b_stock_pct, b_stock_tkr, b_us, b_liq, b_fx = calculate_portfolio_risk_metrics(
        current_portfolio, liquidity_needs_amount, total_wealth
    )

    # After metrics
    a_risk, a_stock_pct, a_stock_tkr, a_us, a_liq, a_fx = calculate_portfolio_risk_metrics(
        exposure_delta.after_holdings, liquidity_needs_amount, total_wealth
    )

    return RiskMetrics(
        before_risk_score=b_risk,
        after_risk_score=a_risk,
        risk_score_delta=round(a_risk - b_risk, 2),
        max_single_stock_pct=a_stock_pct,
        max_single_stock_ticker=a_stock_tkr,
        us_equity_pct=a_us,
        liquidity_coverage_ratio=a_liq,
        unhedged_foreign_currency_pct=a_fx
    )

def suitability_rule_engine(
    client_profile: Any,
    proposed_holdings: List[Holding]
) -> RuleCheckResult:
    """
    Rule engine comparing post-trade portfolio to client constraints and institutional limits.
    Produces deterministic PASS, FLAG, or FAIL status with explicit Breach details.
    """
    policy = _load_policy()
    limits = policy.get("limits", {})

    breaches: List[Breach] = []
    passed: List[str] = []

    # Extract client constraints
    if hasattr(client_profile, "constraints"):
        constraints = client_profile.constraints
        max_allowed_stock = getattr(constraints, "max_single_stock_pct", 20.0)
        excluded_sectors = getattr(constraints, "excluded_sectors", [])
    elif isinstance(client_profile, dict):
        constraints = client_profile.get("constraints", {})
        max_allowed_stock = constraints.get("max_single_stock_pct", 20.0)
        excluded_sectors = constraints.get("excluded_sectors", [])
    else:
        max_allowed_stock = 20.0
        excluded_sectors = []

    total_wealth = getattr(client_profile, "total_wealth", 1000000.0) if hasattr(client_profile, "total_wealth") else client_profile.get("total_wealth", 1000000.0)

    # Calculate near term liquidity due
    liq_due = 50000.0
    if hasattr(client_profile, "liquidity_schedule"):
        for evt in client_profile.liquidity_schedule:
            due = getattr(evt, "due_months", 999) if hasattr(evt, "due_months") else evt.get("due_months", 999)
            amt = getattr(evt, "amount", 0.0) if hasattr(evt, "amount") else evt.get("amount", 0.0)
            if due <= 24:
                liq_due += amt

    # Calculate after-portfolio metrics
    risk, max_stock_pct, max_stock_tkr, us_pct, liq_ratio, fx_pct = calculate_portfolio_risk_metrics(
        proposed_holdings, liq_due, total_wealth
    )

    # 1. Single Stock Limit Check (>20% FLAG, >30% FAIL)
    single_stock_flag = limits.get("single_stock", {}).get("flag_threshold_pct", 20.0)
    single_stock_fail = limits.get("single_stock", {}).get("fail_threshold_pct", 30.0)

    if max_stock_pct > single_stock_fail:
        breaches.append(Breach(
            rule_id="RULE_SINGLE_STOCK_FAIL",
            severity="FAIL",
            metric_name="max_single_stock_concentration",
            before_value=0.0,
            after_value=max_stock_pct,
            threshold_value=single_stock_fail,
            message=f"CRITICAL HARD BREACH: Single stock concentration in {max_stock_tkr} is {max_stock_pct}%, exceeding hard limit of {single_stock_fail}%."
        ))
    elif max_stock_pct > single_stock_flag or max_stock_pct > max_allowed_stock:
        breaches.append(Breach(
            rule_id="RULE_SINGLE_STOCK_FLAG",
            severity="FLAG",
            metric_name="max_single_stock_concentration",
            before_value=0.0,
            after_value=max_stock_pct,
            threshold_value=min(single_stock_flag, max_allowed_stock),
            message=f"CONCENTRATION WARNING: Single stock {max_stock_tkr} is {max_stock_pct}%, exceeding soft guideline ({single_stock_flag}%)."
        ))
    else:
        passed.append("Single Stock Concentration Check")

    # 2. Regional / US Equity Limit (>50% FLAG, >65% FAIL)
    us_flag = limits.get("regional_concentration", {}).get("us_equity_flag_pct", 50.0)
    us_fail = limits.get("regional_concentration", {}).get("us_equity_fail_pct", 65.0)

    if us_pct > us_fail:
        breaches.append(Breach(
            rule_id="RULE_US_EQUITY_FAIL",
            severity="FAIL",
            metric_name="us_equity_concentration",
            before_value=0.0,
            after_value=us_pct,
            threshold_value=us_fail,
            message=f"CRITICAL HARD BREACH: US equity exposure is {us_pct}%, exceeding hard ceiling of {us_fail}%."
        ))
    elif us_pct > us_flag:
        breaches.append(Breach(
            rule_id="RULE_US_EQUITY_FLAG",
            severity="FLAG",
            metric_name="us_equity_concentration",
            before_value=0.0,
            after_value=us_pct,
            threshold_value=us_flag,
            message=f"REGIONAL DRIFT WARNING: US equity moves to {us_pct}%, exceeding recommended threshold ({us_flag}%)."
        ))
    else:
        passed.append("Regional Equity Concentration Check")

    # 3. Liquidity Coverage Adequacy (<1.0x FAIL, <1.2x FLAG)
    liq_flag = limits.get("liquidity_coverage", {}).get("flag_min_coverage_ratio", 1.2)
    liq_fail = limits.get("liquidity_coverage", {}).get("fail_min_coverage_ratio", 1.0)

    if liq_ratio < liq_fail:
        breaches.append(Breach(
            rule_id="RULE_LIQUIDITY_FAIL",
            severity="FAIL",
            metric_name="liquidity_coverage_ratio",
            before_value=2.0,
            after_value=liq_ratio,
            threshold_value=liq_fail,
            message=f"CRITICAL HARD BREACH: Liquid reserves ({liq_ratio:.2f}x) fail to cover near-term obligations (< {liq_fail}x)."
        ))
    elif liq_ratio < liq_flag:
        breaches.append(Breach(
            rule_id="RULE_LIQUIDITY_FLAG",
            severity="FLAG",
            metric_name="liquidity_coverage_ratio",
            before_value=2.0,
            after_value=liq_ratio,
            threshold_value=liq_flag,
            message=f"LIQUIDITY TIGHTENING WARNING: Liquidity coverage ratio is {liq_ratio:.2f}x (guideline >= {liq_flag}x)."
        ))
    else:
        passed.append("Liquidity Coverage Check")

    # Determine overall status
    if any(b.severity == "FAIL" for b in breaches):
        overall_status = "FAIL"
    elif any(b.severity == "FLAG" for b in breaches):
        overall_status = "FLAG"
    else:
        overall_status = "PASS"

    return RuleCheckResult(
        overall_status=overall_status,
        breaches=breaches,
        passed_rules=passed
    )
