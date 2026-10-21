"""Unit tests for the deterministic suitability rule engine R1-R10 (slide 27)."""

from types import SimpleNamespace
import pytest
from src.agents.client_profile import ClientProfileAgent
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.advisory import InvestmentIdea, AllocationRange
from src.schemas.tools import DocumentSnippet
from src.tools.risk_engine import (
    suitability_rule_engine, comparative_outcome, prohibited_wording, worst_outcome
)

APPROVED = {"MSFT", "NVDA", "NESN", "SPYG", "VYMI", "IEF", "LQD", "IEAC", "GLD", "BIL"}
BRIEF = SimpleNamespace(validated_sources=[
    DocumentSnippet(source_id="SRC-FRESH", title="t", publisher="UBS CIO", published_at="2026-10-10T00:00:00Z",
                    retrieved_at="2026-10-21T00:00:00Z", text="x"),
    DocumentSnippet(source_id="SRC-OLD", title="t", publisher="UBS CIO", published_at="2026-09-01T00:00:00Z",
                    retrieved_at="2026-10-21T00:00:00Z", text="x"),
])

def holding(ticker, weight, asset_class="Equities", region="North America", risk=4):
    return {"ticker": ticker, "name": ticker, "asset_class": asset_class, "region": region,
            "currency": "USD", "weight_pct": weight, "risk_rating": risk}

def make_profile(holdings, willingness="Balanced Growth", horizon=84, liquidity=None, max_single=30.0, **extra):
    raw = {
        "client_id": "CL-T", "name": "Test", "goals": ["Growth"], "time_horizon_months": horizon,
        "stated_risk_willingness": willingness, "total_wealth": 1_000_000.0,
        "liquidity_schedule": liquidity or [], "raw_holdings": holdings,
        "constraints": {"max_single_stock_pct": max_single}, **extra
    }
    return ClientProfileAgent(FakeAdvisoryLLMClient()).run(raw)

def idea(slot, ticker, weight, action="ADD", funding="BIL", evidence=("SRC-FRESH",), thesis="Quality exposure."):
    return InvestmentIdea(
        idea_id=f"I{slot}-{ticker}", slot_number=slot, instrument_id=f"X-{ticker}", ticker=ticker, name=ticker,
        action=action, target_weight_pct=weight,
        allocation_range=AllocationRange(min_pct=0.0, max_pct=max(weight, 1.0), target_pct=weight),
        funding_source_ticker=funding, portfolio_role="GROWTH", thesis=thesis, why_now="Now.",
        client_fit_rationale="Fits.", key_risks=["Drawdown"], invalidation_trigger="Thesis breaks.",
        cited_evidence_ids=list(evidence)
    )

def run(profile, ideas):
    return suitability_rule_engine(profile, ideas, BRIEF, APPROVED)

def result(results, rule_id, metric=None):
    return next(r for r in results if r.rule_id == rule_id and (metric is None or r.metric == metric))

def idea_outcome(engine, idea_id):
    return worst_outcome([r.outcome for r in engine["idea_results"][idea_id]])

# Balanced book: US equity 25%, risk score ~2.2, plenty of cash.
BALANCED = [holding("MSFT", 10), holding("SPYG", 15), holding("IEF", 35, "Fixed Income", risk=2),
            holding("BIL", 40, "Cash Equivalents", risk=1)]

def test_comparative_outcome_existing_vs_worsened():
    assert comparative_outcome(20, 25, 30) == "PASS"
    assert comparative_outcome(42, 42, 30) == "FLAG"   # existing, not worsened
    assert comparative_outcome(42, 40, 30) == "FLAG"   # existing, improved but still above
    assert comparative_outcome(42, 55, 30) == "FAIL"   # worsened
    assert comparative_outcome(25, 35, 30) == "FAIL"   # new breach
    assert comparative_outcome(1.2, 0.8, 1.0, higher_is_worse=False) == "FAIL"

def test_r1_missing_mandatory_fields_block():
    profile = make_profile(BALANCED)
    raw_missing = ClientProfileAgent(FakeAdvisoryLLMClient()).run({"client_id": "C", "name": "N", "raw_holdings": BALANCED})
    assert result(run(profile, [idea(1, "IEF", 5)])["portfolio_results"], "R1").outcome == "PASS"
    r1 = result(run(raw_missing, [idea(1, "IEF", 5)])["portfolio_results"], "R1")
    assert r1.outcome == "BLOCK"
    assert "time_horizon_months" in r1.message and "liquidity_schedule" in r1.message

def test_r2_off_universe_instrument_fails():
    engine = run(make_profile(BALANCED), [idea(1, "TSLA", 2)])
    assert result(engine["idea_results"]["I1-TSLA"], "R2").outcome == "FAIL"

def test_r3_unfunded_purchase_fails():
    no_cash = [holding("MSFT", 10), holding("NESN", 20, region="Europe", risk=3), holding("LQD", 70, "Fixed Income", risk=3)]
    engine = run(make_profile(no_cash), [idea(1, "GLD", 5, action="BUY", funding="BIL")])
    r3 = result(engine["portfolio_results"], "R3")
    assert r3.outcome == "FAIL" and r3.after_value == 105.0
    assert idea_outcome(engine, "I1-GLD") == "FAIL"

def test_r4_new_breach_of_risk_band_fails():
    # Moderate client (band ceiling 2.6) moving cash into equities
    engine = run(make_profile(BALANCED, willingness="Moderate"), [idea(1, "NVDA", 20)])
    r4 = result(engine["portfolio_results"], "R4")
    assert r4.before_value <= 2.6 < r4.after_value
    assert r4.outcome == "FAIL"

def test_r4_existing_breach_not_worsened_is_flag():
    risky = [holding("MSFT", 25), holding("NVDA", 25, risk=5), holding("SPYG", 25), holding("BIL", 25, "Cash Equivalents", risk=1)]
    engine = run(make_profile(risky, willingness="Moderate"), [idea(1, "SPYG", 10, action="TRIM", funding="BIL")])
    assert result(engine["portfolio_results"], "R4").outcome == "FLAG"

def test_r5_capacity_conflict_flags_until_advisor_resolves():
    profile = make_profile(BALANCED, willingness="Very High / Aggressive", horizon=18,
                           liquidity=[{"due_months": 12, "amount": 600_000.0, "purpose": "House"}])
    assert result(run(profile, [idea(1, "IEF", 5)])["portfolio_results"], "R5").outcome == "FLAG"
    resolved = profile.model_copy(update={"unresolved_flags": [
        f.model_copy(update={"is_resolved": True, "resolution_notes": "Agreed with client"}) for f in profile.unresolved_flags
    ]})
    r5 = result(run(resolved, [idea(1, "IEF", 5)])["portfolio_results"], "R5")
    assert r5.outcome == "PASS" and "Agreed with client" in r5.message

def test_r6_growth_cap_on_short_horizon_attributed_to_growth_idea():
    profile = make_profile(BALANCED, willingness="Moderate", horizon=24)  # growth 25% before
    engine = run(profile, [idea(1, "IEF", 5), idea(2, "VYMI", 10)])
    r6 = result(engine["portfolio_results"], "R6")
    assert (r6.before_value, r6.after_value, r6.outcome) == (25.0, 35.0, "FAIL")
    assert idea_outcome(engine, "I2-VYMI") == "FAIL"
    assert idea_outcome(engine, "I1-IEF") == "PASS"

def test_r6_not_applied_to_long_horizon():
    engine = run(make_profile(BALANCED, horizon=84), [idea(2, "VYMI", 10)])
    assert not any(r.rule_id == "R6" for r in engine["portfolio_results"])

def test_r7_existing_market_concentration_is_flag_and_worsening_it_fails():
    us_heavy = [holding("MSFT", 14), holding("SPYG", 28), holding("IEF", 25, "Fixed Income", risk=2),
                holding("BIL", 33, "Cash Equivalents", risk=1)]  # US equity 42%
    profile = make_profile(us_heavy)

    neutral = run(profile, [idea(1, "IEF", 5)])
    r7 = result(neutral["portfolio_results"], "R7", "market:North America")
    assert r7.outcome == "FLAG" and r7.before_value == 42.0

    worse = run(profile, [idea(1, "IEF", 5), idea(2, "SPYG", 13), idea(3, "IEAC", 5)])
    r7 = result(worse["portfolio_results"], "R7", "market:North America")
    assert r7.outcome == "FAIL" and r7.after_value == 55.0
    assert idea_outcome(worse, "I2-SPYG") == "FAIL"
    assert idea_outcome(worse, "I1-IEF") == "PASS" and idea_outcome(worse, "I3-IEAC") == "PASS"

def test_r7_instrument_limit_uses_client_single_stock_constraint():
    engine = run(make_profile(BALANCED, max_single=12.0), [idea(1, "MSFT", 15, funding="BIL")])
    r7 = result(engine["portfolio_results"], "R7", "instrument:MSFT")
    assert r7.threshold_value == 12.0 and r7.outcome == "FAIL"

def test_r8_missing_source_fails_and_stale_source_flags():
    engine = run(make_profile(BALANCED), [idea(1, "IEF", 5, evidence=("SRC-NOPE",)), idea(2, "LQD", 5, evidence=("SRC-OLD",))])
    assert result(engine["idea_results"]["I1-IEF"], "R8").outcome == "FAIL"
    assert result(engine["idea_results"]["I2-LQD"], "R8").outcome == "FLAG"

@pytest.mark.parametrize("text,hit", [
    ("This fund offers a guaranteed 6% income.", True),
    ("A risk-free way to earn yield.", True),
    ("Returns are not guaranteed and capital is at risk.", False),
    ("A 6% return cannot be guaranteed.", False),
    ("There is no guarantee of future performance.", False),
])
def test_r9_prohibited_wording_ignores_negations(text, hit):
    assert bool(prohibited_wording(text)) is hit

def test_r9_guarantee_wording_in_idea_fails():
    engine = run(make_profile(BALANCED), [idea(1, "IEF", 5, thesis="Delivers a guaranteed return.")])
    assert result(engine["idea_results"]["I1-IEF"], "R9").outcome == "FAIL"

def test_l1_liquidity_shortfall_existing_flag_and_worsened_fail():
    thin = [holding("MSFT", 20), holding("SPYG", 60), holding("BIL", 20, "Cash Equivalents", risk=1)]
    profile = make_profile(thin, liquidity=[{"due_months": 12, "amount": 400_000.0, "purpose": "House"}])
    assert result(run(profile, [idea(1, "GLD", 5, action="ADD", funding="SPYG")])["portfolio_results"], "L1").outcome == "FLAG"
    assert result(run(profile, [idea(1, "GLD", 5)])["portfolio_results"], "L1").outcome == "FAIL"

def test_portfolio_fail_blames_smallest_set_of_largest_contributors():
    """R4 breach from three risk-adding ideas: only the idea whose removal fixes the breach is regenerated."""
    profile = make_profile(BALANCED, willingness="Moderate")  # risk 2.10, ceiling 2.60
    engine = run(profile, [idea(1, "LQD", 5), idea(2, "NVDA", 15), idea(3, "IEAC", 5)])
    r4 = result(engine["portfolio_results"], "R4")
    assert r4.outcome == "FAIL"
    assert idea_outcome(engine, "I2-NVDA") == "FAIL"
    assert idea_outcome(engine, "I1-LQD") == "PASS" and idea_outcome(engine, "I3-IEAC") == "PASS"

def test_r3_overdrawn_cash_blames_largest_purchase():
    thin_cash = [holding("MSFT", 10), holding("IEF", 80, "Fixed Income", risk=2), holding("BIL", 10, "Cash Equivalents", risk=1)]
    engine = run(make_profile(thin_cash), [idea(1, "LQD", 3), idea(2, "IEAC", 8), idea(3, "GLD", 2)])
    assert result(engine["portfolio_results"], "R3").outcome == "FAIL"
    assert [i for i in ("I1-LQD", "I2-IEAC", "I3-GLD") if idea_outcome(engine, i) == "FAIL"] == ["I2-IEAC"]
