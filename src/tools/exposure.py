"""Portfolio exposure and delta calculator tool (Slide 16)."""

import os
import json
from typing import List, Dict, Any
from src.schemas.tools import Holding, ExposureDelta

DATA_UNIVERSE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "universe", "approved_products.json"
)

def _get_instrument_metadata(ticker: str) -> Dict[str, Any]:
    if os.path.exists(DATA_UNIVERSE_PATH):
        with open(DATA_UNIVERSE_PATH, "r", encoding="utf-8") as f:
            products = json.load(f)
            for p in products:
                if p["ticker"].upper() == ticker.upper():
                    return p
    return {
        "name": ticker,
        "asset_class": "Equities",
        "region": "North America",
        "currency": "USD",
        "risk_rating": 3,
        "complexity_rating": 1,
        "liquidity_tier": "Daily"
    }

def portfolio_exposure_calculator(
    current_holdings: List[Holding],
    proposed_trades: List[Dict[str, Any]]
) -> ExposureDelta:
    """
    Deterministically simulate proposed trades against current portfolio holdings.
    Computes before-and-after allocations, asset class drift, region drift,
    and single-stock concentrations.
    """
    # Build dictionary of current weights
    holdings_dict: Dict[str, Dict[str, Any]] = {}
    for h in current_holdings:
        holdings_dict[h.ticker.upper()] = {
            "name": h.name,
            "asset_class": h.asset_class,
            "region": h.region,
            "currency": h.currency,
            "weight_pct": h.weight_pct,
            "risk_rating": h.risk_rating,
            "complexity_rating": h.complexity_rating,
            "liquidity_tier": h.liquidity_tier
        }

    # Apply proposed trades
    for trade in proposed_trades:
        ticker = str(trade.get("ticker", "")).strip().upper()
        action = str(trade.get("action", "BUY")).strip().upper()
        target_pct = float(trade.get("target_weight_pct", 0.0))
        funding_source = str(trade.get("funding_source_ticker", "BIL")).strip().upper()

        meta = _get_instrument_metadata(ticker)

        if action in ["BUY", "ADD"]:
            current_weight = holdings_dict.get(ticker, {}).get("weight_pct", 0.0)
            delta = target_pct - current_weight if target_pct > current_weight else target_pct

            # Increase ticker weight
            if ticker in holdings_dict:
                holdings_dict[ticker]["weight_pct"] += delta
            else:
                holdings_dict[ticker] = {
                    "name": meta.get("name", ticker),
                    "asset_class": meta.get("asset_class", "Equities"),
                    "region": meta.get("region", "North America"),
                    "currency": meta.get("currency", "USD"),
                    "weight_pct": delta,
                    "risk_rating": meta.get("risk_rating", 3),
                    "complexity_rating": meta.get("complexity_rating", 1),
                    "liquidity_tier": meta.get("liquidity_tier", "Daily")
                }

            # Deduct delta from funding source (e.g. BIL or CASH)
            if funding_source in holdings_dict:
                holdings_dict[funding_source]["weight_pct"] = max(
                    0.0, holdings_dict[funding_source]["weight_pct"] - delta
                )
            else:
                # Deduct from cash or lowest risk holding
                for h_ticker in ["BIL", "CASH", "IEF"]:
                    if h_ticker in holdings_dict:
                        holdings_dict[h_ticker]["weight_pct"] = max(
                            0.0, holdings_dict[h_ticker]["weight_pct"] - delta
                        )
                        break

        elif action in ["TRIM", "SELL"]:
            if ticker in holdings_dict:
                current_weight = holdings_dict[ticker]["weight_pct"]
                delta = current_weight - target_pct if current_weight > target_pct else current_weight
                holdings_dict[ticker]["weight_pct"] = target_pct
                if funding_source in holdings_dict:
                    holdings_dict[funding_source]["weight_pct"] += delta
                else:
                    holdings_dict["BIL"] = {
                        "name": "SPDR 1-3 Month T-Bill ETF",
                        "asset_class": "Cash Equivalents",
                        "region": "North America",
                        "currency": "USD",
                        "weight_pct": delta,
                        "risk_rating": 1,
                        "complexity_rating": 1,
                        "liquidity_tier": "Daily"
                    }

    # Normalize weights if there's minor floating drift
    total_post_weight = sum(item["weight_pct"] for item in holdings_dict.values())
    after_holdings: List[Holding] = []
    for tkr, item in holdings_dict.items():
        if item["weight_pct"] > 0.001:
            norm_wt = (item["weight_pct"] / total_post_weight) * 100.0 if total_post_weight > 0 else 0.0
            after_holdings.append(Holding(
                ticker=tkr,
                name=item["name"],
                asset_class=item["asset_class"],
                region=item["region"],
                currency=item["currency"],
                weight_pct=round(norm_wt, 2),
                risk_rating=item["risk_rating"],
                complexity_rating=item["complexity_rating"],
                liquidity_tier=item["liquidity_tier"]
            ))

    # Calculate Deltas
    before_ac: Dict[str, float] = {}
    for h in current_holdings:
        before_ac[h.asset_class] = before_ac.get(h.asset_class, 0.0) + h.weight_pct

    after_ac: Dict[str, float] = {}
    single_stock_weights: Dict[str, float] = {}
    for h in after_holdings:
        after_ac[h.asset_class] = after_ac.get(h.asset_class, 0.0) + h.weight_pct
        single_stock_weights[h.ticker] = h.weight_pct

    asset_class_delta: Dict[str, float] = {}
    all_acs = set(list(before_ac.keys()) + list(after_ac.keys()))
    for ac in all_acs:
        asset_class_delta[ac] = round(after_ac.get(ac, 0.0) - before_ac.get(ac, 0.0), 2)

    # Top concentration
    top_ticker = max(single_stock_weights, key=single_stock_weights.get) if single_stock_weights else "NONE"
    top_weight = single_stock_weights.get(top_ticker, 0.0)

    return ExposureDelta(
        before_holdings=current_holdings,
        after_holdings=after_holdings,
        asset_class_delta=asset_class_delta,
        region_delta={},
        currency_delta={},
        single_stock_weights=single_stock_weights,
        top_concentration_ticker=top_ticker,
        top_concentration_weight=top_weight,
        total_after_weight=round(sum(h.weight_pct for h in after_holdings), 2)
    )
