"""Deterministic tools for Client Profile Agent (Slide 14)."""

from typing import Dict, Any, List
from src.schemas.tools import ValidationResult, CalculatedRiskTier, StandardizedHoldings, Holding
from src.tools.exposure import get_instrument_metadata

# R1: objective (goals), horizon, risk profile, liquidity need and investable amount must be present.
MANDATORY_FIELDS = [
    "client_id",
    "name",
    "goals",
    "time_horizon_months",
    "stated_risk_willingness",
    "liquidity_schedule",
    "total_wealth",
    "raw_holdings"
]

# An explicitly empty list is a valid answer for these ("no near-term liquidity needs").
EMPTY_LIST_ALLOWED = {"liquidity_schedule"}

def required_field_validator(profile_dict: Dict[str, Any]) -> ValidationResult:
    """Validate mandatory fields in raw advisor-entered client profile."""
    missing = []
    invalid = {}
    warnings = []

    for field in MANDATORY_FIELDS:
        val = profile_dict.get(field)
        empty_list = isinstance(val, list) and len(val) == 0 and field not in EMPTY_LIST_ALLOWED
        if val is None or val == "" or empty_list:
            missing.append(field)

    # Specific validity checks
    if "time_horizon_months" in profile_dict:
        try:
            th = int(profile_dict["time_horizon_months"])
            if th <= 0:
                invalid["time_horizon_months"] = "Time horizon must be greater than 0 months."
        except (ValueError, TypeError):
            invalid["time_horizon_months"] = "Time horizon must be an integer."

    if "total_wealth" in profile_dict:
        try:
            tw = float(profile_dict["total_wealth"])
            if tw < 0:
                invalid["total_wealth"] = "Total wealth cannot be negative."
        except (ValueError, TypeError):
            invalid["total_wealth"] = "Total wealth must be a number."

    is_valid = (len(missing) == 0 and len(invalid) == 0)
    return ValidationResult(
        is_valid=is_valid,
        missing_fields=missing,
        invalid_fields=invalid,
        warnings=warnings
    )

def risk_profile_calculator(profile_data: Dict[str, Any]) -> CalculatedRiskTier:
    """
    Deterministically calculate risk willingness vs risk capacity.
    Flag if client is 'high risk willingness' but has a short-term liquidity need (e.g. <= 18 months).
    """
    stated_willingness = str(profile_data.get("stated_risk_willingness", "")).lower()
    horizon_months = int(profile_data.get("time_horizon_months", 36) or 36)
    total_wealth = float(profile_data.get("total_wealth", 1000000.0) or 1000000.0)
    liquidity_schedule = profile_data.get("liquidity_schedule", [])

    # 1. Quantify willingness (1 to 5)
    if "speculative" in stated_willingness:
        willingness_score = 5.0
    elif "aggressive" in stated_willingness or "high" in stated_willingness:
        willingness_score = 4.0
    elif "balanced" in stated_willingness or "growth" in stated_willingness:
        willingness_score = 3.0
    elif "moderate" in stated_willingness:
        willingness_score = 2.0
    else:
        willingness_score = 1.0 # Conservative default

    # 2. Quantify capacity (1 to 5) based on time horizon and near-term liquidity commitments
    short_term_liquidity_needed = 0.0
    for event in liquidity_schedule:
        if isinstance(event, dict) and event.get("due_months", 999) <= 18:
            short_term_liquidity_needed += float(event.get("amount", 0.0))

    liquidity_drain_ratio = short_term_liquidity_needed / max(total_wealth, 1.0)

    # Base capacity from time horizon
    if horizon_months >= 84: # 7+ years
        base_capacity = 5.0
    elif horizon_months >= 60: # 5 years
        base_capacity = 4.0
    elif horizon_months >= 36: # 3 years
        base_capacity = 3.0
    elif horizon_months >= 24: # 2 years
        base_capacity = 2.0
    else: # < 2 years (e.g. 18 months)
        base_capacity = 1.5

    # Penalize capacity if heavy liquidity drain is due in <= 18 months
    if liquidity_drain_ratio >= 0.50:
        capacity_score = 1.0
    elif liquidity_drain_ratio >= 0.25:
        capacity_score = min(base_capacity, 2.0)
    else:
        capacity_score = base_capacity

    # R5: any gap where willingness exceeds capacity is flagged; the lower band is applied below.
    has_conflict = False
    conflict_msg = None
    if willingness_score > capacity_score:
        has_conflict = True
        conflict_msg = (
            f"Risk willingness ({stated_willingness}, score {willingness_score:.1f}) exceeds risk capacity "
            f"(score {capacity_score:.1f}) given a {horizon_months}-month horizon and "
            f"{short_term_liquidity_needed:,.0f} liquidity need ({liquidity_drain_ratio*100:.1f}% of wealth) due within 18 months. "
            f"The lower (capacity) band is applied."
        )

    # Final calculated tier is the constrained minimum of capacity and willingness
    final_score = min(willingness_score, capacity_score)
    tier_number = max(1, min(5, int(round(final_score))))

    tier_names = {
        1: "Conservative",
        2: "Moderate",
        3: "Balanced Growth",
        4: "Aggressive Growth",
        5: "Speculative"
    }
    recommended_min_horizon = {
        1: 12,
        2: 24,
        3: 36,
        4: 60,
        5: 84
    }

    return CalculatedRiskTier(
        score=round(final_score, 2),
        risk_tier=tier_names[tier_number],
        risk_tier_number=tier_number,
        capacity_score=round(capacity_score, 2),
        willingness_score=round(willingness_score, 2),
        recommended_min_horizon_months=recommended_min_horizon[tier_number],
        has_capacity_conflict=has_conflict,
        conflict_rationale=conflict_msg
    )

def portfolio_parser(raw_holdings: List[Dict[str, Any]]) -> StandardizedHoldings:
    """
    Parse, standardize, and sum portfolio holdings.
    Verify total weights and compute exposures deterministically.
    """
    holdings: List[Holding] = []
    total_weight = 0.0
    asset_class_weights: Dict[str, float] = {}
    region_weights: Dict[str, float] = {}
    currency_weights: Dict[str, float] = {}
    weighted_risk_sum = 0.0

    for item in raw_holdings:
        ticker = str(item.get("ticker", "")).strip().upper()
        name = str(item.get("name", ticker)).strip()
        asset_class = str(item.get("asset_class", "Equities")).strip()
        region = str(item.get("region", "Global")).strip()
        currency = str(item.get("currency", "USD")).strip().upper()
        weight_pct = float(item.get("weight_pct", 0.0))
        risk_rating = int(item.get("risk_rating", 3))
        complexity = int(item.get("complexity_rating", 1))
        liquidity = str(item.get("liquidity_tier", "Daily"))
        # Sector / instrument type from the input, else from the approved universe, else inferred.
        meta = get_instrument_metadata(ticker)
        if asset_class == "Cash Equivalents":
            default_type = "Cash"
        elif "ticker" in meta:  # known instrument
            default_type = meta.get("instrument_type", "Fund")
        else:
            default_type = "Single Stock" if asset_class == "Equities" else "Fund"
        sector = str(item.get("sector") or meta.get("sector", "Diversified"))
        instrument_type = str(item.get("instrument_type") or default_type)

        holding = Holding(
            ticker=ticker,
            name=name,
            asset_class=asset_class,
            region=region,
            currency=currency,
            weight_pct=weight_pct,
            risk_rating=risk_rating,
            complexity_rating=complexity,
            liquidity_tier=liquidity,
            sector=sector,
            instrument_type=instrument_type
        )
        holdings.append(holding)

        total_weight += weight_pct
        asset_class_weights[asset_class] = asset_class_weights.get(asset_class, 0.0) + weight_pct
        region_weights[region] = region_weights.get(region, 0.0) + weight_pct
        currency_weights[currency] = currency_weights.get(currency, 0.0) + weight_pct
        weighted_risk_sum += (risk_rating * weight_pct)

    # Check balanced sum (allow small float deviation)
    is_balanced = abs(total_weight - 100.0) <= 0.5
    weighted_risk_score = round(weighted_risk_sum / max(total_weight, 1.0), 2)

    return StandardizedHoldings(
        holdings=holdings,
        total_weight_pct=round(total_weight, 2),
        asset_class_weights={k: round(v, 2) for k, v in asset_class_weights.items()},
        region_weights={k: round(v, 2) for k, v in region_weights.items()},
        currency_weights={k: round(v, 2) for k, v in currency_weights.items()},
        weighted_risk_score=weighted_risk_score,
        is_balanced_sum=is_balanced
    )

def with_new_capital(raw_holdings: List[Dict[str, Any]], total_wealth: float, new_capital: float) -> List[Dict[str, Any]]:
    """UC2: add new money to the portfolio as cash (BIL) and re-weight the existing holdings over the larger total."""
    if new_capital <= 0 or total_wealth <= 0:
        return [dict(h) for h in raw_holdings]
    scale = total_wealth / (total_wealth + new_capital)
    cash_pct = new_capital / (total_wealth + new_capital) * 100.0
    holdings = [{**h, "weight_pct": round(float(h.get("weight_pct", 0.0)) * scale, 4)} for h in raw_holdings]
    cash = next((h for h in holdings if str(h.get("ticker", "")).upper() == "BIL"), None)
    if cash is None:
        holdings.append({"ticker": "BIL", "name": "Cash and money market funds", "asset_class": "Cash Equivalents",
                         "region": "Singapore", "currency": "SGD", "weight_pct": 0.0, "risk_rating": 1})
        cash = holdings[-1]
    cash["weight_pct"] = round(cash["weight_pct"] + cash_pct, 4)
    return holdings

# Client facts compared against the last review (UC1). Wealth only counts as a change above this drift.
CHANGE_FIELDS = {
    "goals": "Investment objective",
    "time_horizon_months": "Time horizon (months)",
    "stated_risk_willingness": "Risk willingness",
    "liquidity_schedule": "Liquidity needs",
    "experience_level": "Investment experience",
    "total_wealth": "Investable wealth",
}
WEALTH_CHANGE_THRESHOLD = 0.10

def _describe(field: str, value: Any) -> str:
    if value in (None, "", []):
        return "Not recorded"
    if field == "goals":
        return "; ".join(value) if isinstance(value, list) else str(value)
    if field == "liquidity_schedule":
        return "; ".join(f"{e.get('purpose', 'Need')}: {float(e.get('amount', 0)):,.0f} in {e.get('due_months')} months"
                         for e in value) or "None"
    if field == "total_wealth":
        return f"{float(value):,.0f}"
    return str(value)

def profile_change_detector(current: Dict[str, Any], last_review: Dict[str, Any]) -> List[Dict[str, str]]:
    """Deterministically list client facts that changed since the last review. No last review -> no changes."""
    if not last_review:
        return []
    changes = []
    for field, label in CHANGE_FIELDS.items():
        before, after = last_review.get(field), current.get(field)
        if field == "total_wealth":
            if not before or not after or abs(float(after) - float(before)) / float(before) < WEALTH_CHANGE_THRESHOLD:
                continue
        elif field == "liquidity_schedule":
            key = lambda events: sorted((int(e.get("due_months", 0)), float(e.get("amount", 0))) for e in events or [])
            if key(before) == key(after):
                continue
        elif before == after:
            continue
        changes.append({"field": field, "label": label, "before": _describe(field, before), "after": _describe(field, after)})
    return changes
