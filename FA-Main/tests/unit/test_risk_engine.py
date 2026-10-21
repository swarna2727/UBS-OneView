"""Unit tests for deterministic portfolio risk engine and suitability rule engine."""

import pytest
from src.schemas.tools import Holding
from src.tools.risk_engine import suitability_rule_engine, calculate_portfolio_risk_metrics

def make_portfolio(single_stock_pct: float, us_equity_pct: float, cash_pct: float) -> list[Holding]:
    holdings = []
    if single_stock_pct > 0:
        holdings.append(Holding(
            ticker="NVDA",
            name="NVIDIA Corporation",
            asset_class="Equities",
            region="North America",
            currency="USD",
            weight_pct=single_stock_pct,
            risk_rating=5
        ))

    other_us = max(0.0, us_equity_pct - single_stock_pct)
    if other_us > 0:
        holdings.append(Holding(
            ticker="MSFT",
            name="Microsoft Corporation",
            asset_class="Equities",
            region="North America",
            currency="USD",
            weight_pct=other_us,
            risk_rating=4
        ))

    rem_equity = max(0.0, 100.0 - single_stock_pct - other_us - cash_pct)
    if rem_equity > 0:
        holdings.append(Holding(
            ticker="IEF",
            name="iShares 7-10 Year Treasury Bond",
            asset_class="Fixed Income",
            region="North America",
            currency="USD",
            weight_pct=rem_equity,
            risk_rating=2
        ))

    if cash_pct > 0:
        holdings.append(Holding(
            ticker="BIL",
            name="SPDR 1-3 Month T-Bill ETF",
            asset_class="Cash Equivalents",
            region="North America",
            currency="USD",
            weight_pct=cash_pct,
            risk_rating=1
        ))

    return holdings

def test_hard_concentration_limit_triggers_fail():
    """Single stock > 30% must deterministically trigger a FAIL status gate."""
    # 35% NVDA -> Hard Breach
    holdings = make_portfolio(single_stock_pct=35.0, us_equity_pct=45.0, cash_pct=15.0)

    client_profile = {
        "total_wealth": 1000000.0,
        "constraints": {"max_single_stock_pct": 20.0, "excluded_sectors": []},
        "liquidity_schedule": [{"due_months": 12, "amount": 50000.0}]
    }

    result = suitability_rule_engine(client_profile, holdings)
    assert result.overall_status == "FAIL"

    fail_breaches = [b for b in result.breaches if b.severity == "FAIL"]
    assert len(fail_breaches) >= 1
    assert "single_stock" in fail_breaches[0].metric_name.lower() or "SINGLE_STOCK" in fail_breaches[0].rule_id
    assert fail_breaches[0].after_value == 35.0

def test_soft_concentration_limit_triggers_flag():
    """Single stock between 20% and 30% must trigger a soft FLAG for RM exception review."""
    # 25% NVDA -> Mild breach (FLAG)
    holdings = make_portfolio(single_stock_pct=25.0, us_equity_pct=45.0, cash_pct=15.0)

    client_profile = {
        "total_wealth": 1000000.0,
        "constraints": {"max_single_stock_pct": 20.0, "excluded_sectors": []},
        "liquidity_schedule": [{"due_months": 12, "amount": 50000.0}]
    }

    result = suitability_rule_engine(client_profile, holdings)
    assert result.overall_status == "FLAG"

    flag_breaches = [b for b in result.breaches if b.severity == "FLAG"]
    assert len(flag_breaches) >= 1
    assert any("single_stock" in b.metric_name.lower() for b in flag_breaches)

def test_regional_us_equity_drift_triggers_flag():
    """US Equity > 50% must trigger a FLAG when single stock limits are respected."""
    # NVDA 18%, MSFT 18%, TSM 18% (all < 20%), Cash 20%, IEF 26%
    # Total US equity = 18 + 18 + 18 = 54% (exceeds 50% flag, below 65% fail)
    holdings = [
        Holding(ticker="NVDA", name="NVIDIA", asset_class="Equities", region="North America", currency="USD", weight_pct=18.0, risk_rating=5),
        Holding(ticker="MSFT", name="Microsoft", asset_class="Equities", region="North America", currency="USD", weight_pct=18.0, risk_rating=4),
        Holding(ticker="AAPL", name="Apple", asset_class="Equities", region="North America", currency="USD", weight_pct=18.0, risk_rating=4),
        Holding(ticker="IEF", name="Treasury", asset_class="Fixed Income", region="North America", currency="USD", weight_pct=26.0, risk_rating=2),
        Holding(ticker="BIL", name="T-Bills", asset_class="Cash Equivalents", region="North America", currency="USD", weight_pct=20.0, risk_rating=1)
    ]

    client_profile = {
        "total_wealth": 1000000.0,
        "constraints": {"max_single_stock_pct": 20.0, "excluded_sectors": []},
        "liquidity_schedule": [{"due_months": 12, "amount": 50000.0}]
    }

    result = suitability_rule_engine(client_profile, holdings)
    assert result.overall_status == "FLAG"
    assert any("us_equity" in b.metric_name.lower() for b in result.breaches)

def test_compliant_portfolio_triggers_pass():
    """Diversified portfolio within all limits must trigger PASS."""
    # NVDA 15%, MSFT 15% (US equity 30%), Cash 20%, IEF 50%
    holdings = make_portfolio(single_stock_pct=15.0, us_equity_pct=30.0, cash_pct=20.0)

    client_profile = {
        "total_wealth": 1000000.0,
        "constraints": {"max_single_stock_pct": 20.0, "excluded_sectors": []},
        "liquidity_schedule": [{"due_months": 12, "amount": 50000.0}]
    }

    result = suitability_rule_engine(client_profile, holdings)
    assert result.overall_status == "PASS"
    assert len(result.breaches) == 0
