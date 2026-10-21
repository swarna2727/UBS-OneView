"""Unit tests for Client Profile Agent tools and conflict detection."""

import pytest
from src.tools.profile_tools import (
    required_field_validator, risk_profile_calculator, portfolio_parser
)

def test_missing_mandatory_fields():
    incomplete = {
        "client_id": "CL-01",
        "name": "Test Client"
        # Missing goals, time_horizon, stated_risk_willingness, total_wealth, raw_holdings
    }
    res = required_field_validator(incomplete)
    assert not res.is_valid
    assert "goals" in res.missing_fields
    assert "time_horizon_months" in res.missing_fields
    assert "stated_risk_willingness" in res.missing_fields

def test_risk_capacity_vs_willingness_conflict():
    """Client with 'Aggressive' willingness but an 18-month 75% liquidity drain must trigger conflict."""
    contradictory_data = {
        "stated_risk_willingness": "Very High / Aggressive",
        "time_horizon_months": 18,
        "total_wealth": 1000000.0,
        "liquidity_schedule": [
            {"due_months": 12, "amount": 750000.0, "purpose": "Venture Capital Call"}
        ]
    }

    tier = risk_profile_calculator(contradictory_data)
    assert tier.has_capacity_conflict
    assert tier.willingness_score >= 4.0
    assert tier.capacity_score <= 2.0
    assert "conflicts" in tier.conflict_rationale.lower() or "constrained" in tier.conflict_rationale.lower()
    # Effective tier should be constrained to Conservative/Moderate
    assert tier.risk_tier_number <= 2

def test_portfolio_parser_validates_weight_sum():
    # Valid 100% sum
    raw_holdings = [
        {"ticker": "MSFT", "weight_pct": 50.0, "asset_class": "Equities"},
        {"ticker": "IEF", "weight_pct": 50.0, "asset_class": "Fixed Income"}
    ]
    parsed = portfolio_parser(raw_holdings)
    assert parsed.is_balanced_sum
    assert parsed.total_weight_pct == 100.0

    # Invalid 75% sum
    bad_holdings = [
        {"ticker": "MSFT", "weight_pct": 50.0, "asset_class": "Equities"},
        {"ticker": "IEF", "weight_pct": 25.0, "asset_class": "Fixed Income"}
    ]
    parsed_bad = portfolio_parser(bad_holdings)
    assert not parsed_bad.is_balanced_sum
    assert parsed_bad.total_weight_pct == 75.0
