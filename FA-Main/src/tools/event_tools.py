"""Market-event impact and stress tools (UC3, memo stress view). Deterministic: no model calls."""

from typing import Any, Dict, List

from src.schemas.advisory import EventImpact, HoldingImpact
from src.schemas.tools import Holding
from src.tools.risk_engine import evaluate_portfolio_rules, portfolio_metrics

# Illustrative event presets: price move (%) by asset class, with per-instrument overrides where the
# exposure differs materially (duration, credit quality, sector). Calibrate before any pilot.
MARKET_EVENTS: Dict[str, Dict[str, Any]] = {
    "rates_up": {
        "title": "Interest-rate expectations rise",
        "description": "Markets price in fewer rate cuts after strong inflation data; 10-year yields rise about 75bps in two weeks.",
        "asset_class": {"Equities": -4.0, "Fixed Income": -4.5, "Real Estate": -7.0, "Commodities": -2.0, "Cash Equivalents": 0.3},
        "instrument": {"IEF": -5.5, "LQD": -6.0, "AGGG": -4.5, "IEAC": -4.0, "HYG": -3.0, "BIL": 0.3,
                       "SPYG": -6.0, "MSFT": -6.5, "NVDA": -8.0},
    },
    "equity_selloff": {
        "title": "Global equity sell-off",
        "description": "A growth scare drives global equities about 12% lower; high-quality bonds and gold rally.",
        "asset_class": {"Equities": -12.0, "Fixed Income": 1.5, "Real Estate": -9.0, "Commodities": 2.0, "Cash Equivalents": 0.1},
        "instrument": {"MSFT": -16.0, "NVDA": -22.0, "TSM": -18.0, "SPYG": -15.0, "HYG": -4.0, "GLD": 4.0, "IEF": 2.5},
    },
    "credit_widening": {
        "title": "Credit spreads widen",
        "description": "Corporate credit spreads widen about 80bps on default concerns; government bonds hold up.",
        "asset_class": {"Equities": -5.0, "Fixed Income": -2.0, "Real Estate": -5.0, "Commodities": 0.0, "Cash Equivalents": 0.1},
        "instrument": {"HYG": -6.0, "LQD": -3.5, "IEAC": -2.5, "AGGG": -1.5, "IEF": 1.0, "GLD": 1.5},
    },
}

# Stress scenarios shown in the suitability view and the memo PDF, as asset-class price moves (%).
STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "Equity sell-off": {"Equities": -15.0, "Real Estate": -10.0, "Commodities": -5.0, "Fixed Income": 1.0, "Cash Equivalents": 0.0},
    "Rates +100bps": {"Equities": -4.0, "Real Estate": -6.0, "Commodities": -2.0, "Fixed Income": -6.0, "Cash Equivalents": 0.2},
    "Credit spreads widen": {"Equities": -6.0, "Real Estate": -5.0, "Commodities": 0.0, "Fixed Income": -3.0, "Cash Equivalents": 0.0},
}

# Suggestion thresholds; the advisor always makes the decision.
REVISE_MOVE_PCT = -5.0
MONITOR_MOVE_PCT = -2.0


def stress_impact(holdings: List[Holding]) -> Dict[str, float]:
    """Estimated portfolio move (%) under each stress scenario."""
    return {name: round(sum(h.weight_pct * shocks.get(h.asset_class, 0.0) for h in holdings) / 100.0, 2)
            for name, shocks in STRESS_SCENARIOS.items()}


def market_event_impact(profile: Any, event_key: str, description: str = "") -> EventImpact:
    """Which holdings are affected, how much the portfolio moves, and whether suitability still holds after the drift."""
    event = MARKET_EVENTS.get(event_key, MARKET_EVENTS["rates_up"])
    holdings: List[Holding] = profile.parsed_holdings
    rows: List[HoldingImpact] = []
    drifted: List[Holding] = []
    for h in holdings:
        shock = event["instrument"].get(h.ticker.upper(), event["asset_class"].get(h.asset_class, 0.0))
        rows.append(HoldingImpact(ticker=h.ticker, name=h.name, asset_class=h.asset_class, weight_pct=h.weight_pct,
                                  shock_pct=shock, contribution_pct=round(h.weight_pct * shock / 100.0, 2)))
        drifted.append(h.model_copy(update={"weight_pct": h.weight_pct * (1 + shock / 100.0)}))
    move = round(sum(r.contribution_pct for r in rows), 2)
    total = sum(h.weight_pct for h in drifted) or 1.0
    drifted = [h.model_copy(update={"weight_pct": round(h.weight_pct / total * 100.0, 2)}) for h in drifted]

    before = portfolio_metrics(holdings, profile)
    after_profile = profile.model_copy(update={"total_wealth": profile.total_wealth * (1 + move / 100.0)})
    after = portfolio_metrics(drifted, after_profile)
    rules = [r for r in evaluate_portfolio_rules(after_profile, before, after) if r.rule_id != "R3"]

    outcomes = {r.outcome for r in rules}
    if "FAIL" in outcomes or move <= REVISE_MOVE_PCT:
        suggestion, why = "REVISE", "a suitability rule now fails or the estimated loss is material"
    elif "FLAG" in outcomes or move <= MONITOR_MOVE_PCT:
        suggestion, why = "MONITOR", "the portfolio is affected but every suitability rule still holds"
    else:
        suggestion, why = "MAINTAIN", "the estimated impact is small and suitability is unchanged"
    hit = sorted(rows, key=lambda r: r.contribution_pct)[0] if rows else None
    rationale = (f"Estimated portfolio move {move:+.2f}%"
                 + (f", led by {hit.ticker} ({hit.contribution_pct:+.2f} pts)" if hit else "")
                 + f". Suggested {suggestion.lower()}: {why}.")
    return EventImpact(
        event_key=event_key if event_key in MARKET_EVENTS else "rates_up",
        event_title=event["title"], event_description=description or event["description"],
        holdings=rows, portfolio_move_pct=move,
        risk_score_before=before["risk_score"], risk_score_after=after["risk_score"],
        rule_results=rules, suggested_decision=suggestion, rationale=rationale,
    )
