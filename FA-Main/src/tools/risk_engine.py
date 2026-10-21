"""Deterministic portfolio risk engine and suitability rule engine R1-R10 (slides 19 and 27).

Code, not prompts, decides what may pass: every rule outcome here is computed from the
client profile, the holdings and the approved data files. No LLM is involved.
"""

import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import yaml
from src.schemas.tools import Holding, RiskMetrics, RuleResult
from src.tools.exposure import portfolio_exposure_calculator

CONFIG_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config", "suitability_policy.yaml"
)

DEFAULT_REFERENCE_DATE = "2026-10-21T00:00:00Z"  # matches the research corpus snapshot
NO_LIQUIDITY_NEED_RATIO = 999.0
OUTCOME_ORDER = {"PASS": 0, "FLAG": 1, "FAIL": 2, "BLOCK": 3}

_policy_cache: Optional[Dict[str, Any]] = None

def load_policy() -> Dict[str, Any]:
    global _policy_cache
    if _policy_cache is None:
        with open(CONFIG_POLICY_PATH, "r", encoding="utf-8") as f:
            _policy_cache = yaml.safe_load(f)
    return _policy_cache

def _rule(rule_key: str) -> Dict[str, Any]:
    return load_policy()["rules"].get(rule_key) or {}

def worst_outcome(outcomes: List[str]) -> str:
    return max(outcomes, key=lambda o: OUTCOME_ORDER[o], default="PASS")

def comparative_outcome(before: float, after: float, limit: float, higher_is_worse: bool = True) -> str:
    """PASS within limit; FLAG for an existing breach not made worse; FAIL for a new or worsened breach."""
    breached = (lambda v: v > limit + 1e-9) if higher_is_worse else (lambda v: v < limit - 1e-9)
    if not breached(after):
        return "PASS"
    worsened = (after > before + 1e-9) if higher_is_worse else (after < before - 1e-9)
    if breached(before) and not worsened:
        return load_policy().get("existing_breach_outcome", "FLAG")
    return load_policy().get("worsened_breach_outcome", "FAIL")

# ==========================================
# Portfolio metrics
# ==========================================

def near_term_liquidity_need(profile: Any) -> float:
    window = _rule("L1_liquidity_coverage").get("window_months", 24)
    return sum(e.amount for e in getattr(profile, "liquidity_schedule", []) if e.due_months <= window)

def portfolio_metrics(holdings: List[Holding], profile: Any) -> Dict[str, Any]:
    """Deterministic metrics the rules are evaluated on."""
    r6, r7 = _rule("R6_short_horizon_growth"), _rule("R7_concentration")
    growth_classes = set(r6.get("growth_asset_classes", ["Equities"]))
    diversified_regions = set(r7.get("diversified_regions", []))
    diversified_sectors = set(r7.get("diversified_sectors", ["Diversified"]))

    total = sum(h.weight_pct for h in holdings) or 1.0
    risk_score = sum(h.risk_rating * h.weight_pct for h in holdings) / total

    markets: Dict[str, float] = {}
    sectors: Dict[str, float] = {}
    instruments: Dict[str, float] = {}
    for h in holdings:
        if h.asset_class == "Equities" and h.region not in diversified_regions:
            markets[h.region] = markets.get(h.region, 0.0) + h.weight_pct
        if h.instrument_type == "Single Stock":
            instruments[h.ticker] = instruments.get(h.ticker, 0.0) + h.weight_pct
            if h.sector not in diversified_sectors:
                sectors[h.sector] = sectors.get(h.sector, 0.0) + h.weight_pct

    need = near_term_liquidity_need(profile)
    wealth = getattr(profile, "total_wealth", 0.0)
    liquid_value = sum(
        h.weight_pct / 100.0 * wealth for h in holdings
        if h.liquidity_tier == "Daily" and h.asset_class in ("Cash Equivalents", "Fixed Income")
    )
    liquidity_ratio = liquid_value / need if need > 0 else NO_LIQUIDITY_NEED_RATIO

    top_ticker, top_pct = max(instruments.items(), key=lambda kv: kv[1], default=("NONE", 0.0))
    return {
        "risk_score": round(risk_score, 2),
        "growth_pct": round(sum(h.weight_pct for h in holdings if h.asset_class in growth_classes), 2),
        "market": {k: round(v, 2) for k, v in markets.items()},
        "sector": {k: round(v, 2) for k, v in sectors.items()},
        "instrument": {k: round(v, 2) for k, v in instruments.items()},
        "liquidity_ratio": round(liquidity_ratio, 2),
        "top_instrument": (top_ticker, round(top_pct, 2)),
        "us_equity_pct": round(markets.get("North America", 0.0), 2),
        "fx_pct": round(sum(h.weight_pct for h in holdings if h.currency != "USD"), 2),
    }

def metric_value(metrics: Dict[str, Any], metric: str) -> float:
    """Look up 'risk_score', 'growth_pct', 'liquidity_ratio' or a 'market:<k>' style key."""
    if ":" in metric:
        dim, key = metric.split(":", 1)
        return metrics[dim].get(key, 0.0)
    return metrics[metric]

def to_risk_metrics(before: Dict[str, Any], after: Dict[str, Any]) -> RiskMetrics:
    return RiskMetrics(
        before_risk_score=before["risk_score"],
        after_risk_score=after["risk_score"],
        risk_score_delta=round(after["risk_score"] - before["risk_score"], 2),
        max_single_stock_pct=after["top_instrument"][1],
        max_single_stock_ticker=after["top_instrument"][0],
        us_equity_pct=after["us_equity_pct"],
        liquidity_coverage_ratio=after["liquidity_ratio"],
        unhedged_foreign_currency_pct=after["fx_pct"],
        growth_asset_pct=after["growth_pct"]
    )

# ==========================================
# Profile-level rules (R1, R5)
# ==========================================

def evaluate_profile_rules(profile: Any) -> List[RuleResult]:
    results: List[RuleResult] = []
    missing = [f for f in profile.unresolved_flags if f.code == "MISSING_MANDATORY_FIELD" and not f.is_resolved]
    results.append(RuleResult(
        rule_id="R1", name="Mandatory fields present",
        outcome="BLOCK" if missing else "PASS",
        message=("Missing or invalid: " + ", ".join(sorted({fld for f in missing for fld in f.affected_fields})))
        if missing else "All mandatory client fields are present."
    ))

    conflicts = [f for f in profile.unresolved_flags if f.code == "RISK_CAPACITY_CONFLICT"]
    unresolved = [f for f in conflicts if not f.is_resolved]
    if unresolved:
        results.append(RuleResult(rule_id="R5", name="Risk willingness vs capacity", outcome="FLAG", message=unresolved[0].message))
    else:
        note = f" Advisor resolution: {conflicts[0].resolution_notes}" if conflicts else ""
        results.append(RuleResult(
            rule_id="R5", name="Risk willingness vs capacity", outcome="PASS",
            message=f"Lower band applied: {profile.calculated_risk_tier}.{note}" if conflicts
            else "Risk willingness does not exceed capacity."
        ))
    return results

# ==========================================
# Portfolio-level rules (R3, R4, R6, R7, L1)
# ==========================================

def evaluate_portfolio_rules(
    profile: Any,
    before: Dict[str, Any],
    after: Dict[str, Any],
    unnormalized_after_weight: float = 100.0
) -> List[RuleResult]:
    results: List[RuleResult] = []

    # R3 Allocations sum to 100%
    tol = _rule("R3_allocations_sum").get("tolerance_pp", 0.01)
    off_by = abs(unnormalized_after_weight - 100.0)
    results.append(RuleResult(
        rule_id="R3", name="Allocations sum to 100%", metric="allocation_total",
        outcome="PASS" if off_by <= tol else "FAIL",
        after_value=round(unnormalized_after_weight, 2), threshold_value=100.0,
        message="Proposed trades are fully funded." if off_by <= tol
        else f"Proposed trades leave the portfolio at {unnormalized_after_weight:.2f}% (must be 100% +/- {tol} pt): a purchase has no funding source."
    ))

    # R4 Proposed portfolio risk score within the client's risk band
    band_max = profile.risk_score_range[1]
    outcome = comparative_outcome(before["risk_score"], after["risk_score"], band_max)
    results.append(RuleResult(
        rule_id="R4", name="Portfolio risk within client band", metric="risk_score", outcome=outcome,
        before_value=before["risk_score"], after_value=after["risk_score"], threshold_value=band_max,
        message=f"Portfolio risk score {before['risk_score']:.2f} -> {after['risk_score']:.2f} "
                f"vs {profile.calculated_risk_tier} band ceiling {band_max:.2f}."
    ))

    # R6 Growth-asset cap for short horizons
    r6 = _rule("R6_short_horizon_growth")
    if profile.time_horizon_months <= r6.get("max_horizon_months", 36):
        cap = r6.get("max_growth_pct", 30.0)
        outcome = comparative_outcome(before["growth_pct"], after["growth_pct"], cap)
        results.append(RuleResult(
            rule_id="R6", name="Growth-asset cap for short horizon", metric="growth_pct", outcome=outcome,
            before_value=before["growth_pct"], after_value=after["growth_pct"], threshold_value=cap,
            message=f"{profile.time_horizon_months}-month horizon: growth assets {before['growth_pct']:.1f}% -> "
                    f"{after['growth_pct']:.1f}% (cap {cap:.0f}%)."
        ))

    # R7 Single market / sector / instrument concentration
    limit = _rule("R7_concentration").get("limit_pct", 30.0)
    instrument_limit = min(limit, profile.constraints.max_single_stock_pct)
    labels = {"market": "equity market", "sector": "sector", "instrument": "single instrument"}
    any_r7 = False
    for dim in ("market", "sector", "instrument"):
        dim_limit = instrument_limit if dim == "instrument" else limit
        for key in sorted(set(before[dim]) | set(after[dim])):
            b, a = before[dim].get(key, 0.0), after[dim].get(key, 0.0)
            outcome = comparative_outcome(b, a, dim_limit)
            if outcome == "PASS":
                continue
            any_r7 = True
            existing = "Existing concentration" if b > dim_limit else "Concentration"
            change = "worsened" if a > b + 1e-9 else "not worsened"
            results.append(RuleResult(
                rule_id="R7", name="Concentration limit", metric=f"{dim}:{key}", outcome=outcome,
                before_value=b, after_value=a, threshold_value=dim_limit,
                message=f"{existing} in {labels[dim]} '{key}': {b:.1f}% -> {a:.1f}% vs {dim_limit:.0f}% limit ({change})."
            ))
    if not any_r7:
        results.append(RuleResult(rule_id="R7", name="Concentration limit", outcome="PASS",
                                  message=f"No market, sector or instrument above {limit:.0f}%."))

    # L1 Liquidity coverage of near-term needs
    min_ratio = _rule("L1_liquidity_coverage").get("min_coverage_ratio", 1.0)
    outcome = comparative_outcome(before["liquidity_ratio"], after["liquidity_ratio"], min_ratio, higher_is_worse=False)
    results.append(RuleResult(
        rule_id="L1", name="Liquidity coverage", metric="liquidity_ratio", outcome=outcome,
        before_value=before["liquidity_ratio"], after_value=after["liquidity_ratio"], threshold_value=min_ratio,
        message="No near-term liquidity needs." if after["liquidity_ratio"] >= NO_LIQUIDITY_NEED_RATIO
        else f"Liquid assets cover near-term needs {before['liquidity_ratio']:.2f}x -> {after['liquidity_ratio']:.2f}x (minimum {min_ratio:.1f}x)."
    ))
    return results

# ==========================================
# Idea-level rules (R2, R8, R9)
# ==========================================

_NEGATION = re.compile(r"(not|cannot be|can't be|no|never|without)\s+(\w+\s+){0,2}$", re.IGNORECASE)

def prohibited_wording(text: str) -> List[str]:
    """R9: prohibited claims in client-facing text, ignoring negated uses such as 'returns are not guaranteed'."""
    hits = []
    for pattern in _rule("R9_prohibited_wording").get("patterns", []):
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if not _NEGATION.search(text[max(0, m.start() - 30):m.start()]):
                hits.append(m.group(0))
    return hits

def _source_age_days(published_at: str, reference_date_iso: str) -> int:
    ref = datetime.fromisoformat(reference_date_iso.replace("Z", "+00:00"))
    try:
        pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 10_000
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    return max(0, (ref - pub).days)

def evaluate_idea_rules(
    idea: Any,
    research_brief: Any,
    approved_tickers: set,
    reference_date_iso: str = DEFAULT_REFERENCE_DATE
) -> List[RuleResult]:
    results: List[RuleResult] = []

    # R2 Approved universe
    in_universe = idea.ticker.upper() in approved_tickers
    results.append(RuleResult(
        rule_id="R2", name="Approved universe", idea_id=idea.idea_id,
        outcome="PASS" if in_universe else "FAIL",
        message=f"{idea.ticker} is in the approved universe." if in_universe
        else f"{idea.ticker} is not in the approved universe."
    ))

    # R8 Evidence present and fresh
    max_age = _rule("R8_evidence_freshness").get("max_age_days", 30)
    sources = {s.source_id: s for s in getattr(research_brief, "validated_sources", [])}
    missing = [e for e in idea.cited_evidence_ids if e not in sources]
    stale = [e for e in idea.cited_evidence_ids if e in sources
             and _source_age_days(sources[e].published_at, reference_date_iso) > max_age]
    if missing or not idea.cited_evidence_ids:
        outcome, msg = "FAIL", f"Cites sources not in the research brief: {', '.join(missing) or 'none cited'}."
    elif stale:
        outcome, msg = "FLAG", f"Cites sources older than {max_age} days: {', '.join(stale)}."
    else:
        outcome, msg = "PASS", f"All {len(idea.cited_evidence_ids)} cited sources retrieved and within {max_age} days."
    results.append(RuleResult(rule_id="R8", name="Evidence sourced and fresh", idea_id=idea.idea_id, outcome=outcome, message=msg))

    # R9 Prohibited wording in the idea's client-facing text
    text = " ".join([idea.thesis, idea.why_now, idea.client_fit_rationale, idea.invalidation_trigger,
                     idea.alternative_considered, *idea.key_risks])
    hits = prohibited_wording(text)
    results.append(RuleResult(
        rule_id="R9", name="No prohibited wording", idea_id=idea.idea_id,
        outcome="FAIL" if hits else "PASS",
        message=f"Prohibited wording: {', '.join(sorted(set(hits)))}." if hits else "No guarantee or risk-free wording."
    ))
    return results

# ==========================================
# Full assessment
# ==========================================

def _trades(ideas: List[Any]) -> List[Dict[str, Any]]:
    return [{
        "ticker": i.ticker, "action": i.action, "target_weight_pct": i.target_weight_pct,
        "funding_source_ticker": i.funding_source_ticker
    } for i in ideas]

def _contributors(result: RuleResult, ideas: List[Any], before: Dict[str, Any],
                  solo: Dict[str, Tuple[Dict[str, Any], float]]) -> List[Any]:
    """
    Smallest set of ideas, largest effect first, whose removal would bring a failing metric back
    to an acceptable level (the limit, or the pre-trade value for an existing breach). Effects are
    treated as additive, which holds for the weighted-average metrics used here.
    """
    if result.metric == "allocation_total":
        tol = _rule("R3_allocations_sum").get("tolerance_pp", 0.01)
        unfunded_alone = [i for i in ideas if abs(solo[i.idea_id][1] - 100.0) > tol]
        if unfunded_alone:
            return unfunded_alone
        # Each purchase is fundable alone but together they overdraw the source: drop the largest first.
        shortfall, chosen = abs(result.after_value - 100.0), []
        for idea in sorted((i for i in ideas if i.action in ("BUY", "ADD")), key=lambda i: i.target_weight_pct, reverse=True):
            if shortfall <= tol:
                break
            chosen.append(idea)
            shortfall -= idea.target_weight_pct
        return chosen

    sign = -1.0 if result.metric == "liquidity_ratio" else 1.0  # lower liquidity is worse
    b = metric_value(before, result.metric)
    effects = sorted(
        ((sign * (metric_value(solo[i.idea_id][0], result.metric) - b), i) for i in ideas),
        key=lambda e: e[0], reverse=True
    )
    acceptable = max(result.threshold_value, b) if sign > 0 else min(result.threshold_value, b)
    excess = sign * (result.after_value - acceptable)
    chosen, removed = [], 0.0
    for effect, idea in effects:
        if effect <= 1e-9 or removed >= excess - 1e-9:
            break
        chosen.append(idea)
        removed += effect
    return chosen

def suitability_rule_engine(
    profile: Any,
    ideas: List[Any],
    research_brief: Any,
    approved_tickers: set,
    reference_date_iso: str = DEFAULT_REFERENCE_DATE
) -> Dict[str, Any]:
    """
    Evaluate R1-R10 for a candidate idea set.
    Returns portfolio-level results, per-idea results (idea rules plus portfolio FAILs the
    idea contributes to), before/after RiskMetrics and the after-holdings.
    """
    holdings = profile.parsed_holdings
    exposure = portfolio_exposure_calculator(holdings, _trades(ideas))
    before = portfolio_metrics(holdings, profile)
    after = portfolio_metrics(exposure.after_holdings, profile)

    portfolio_results = evaluate_profile_rules(profile) + evaluate_portfolio_rules(
        profile, before, after, exposure.unnormalized_after_weight
    )

    per_idea: Dict[str, List[RuleResult]] = {
        i.idea_id: evaluate_idea_rules(i, research_brief, approved_tickers, reference_date_iso) for i in ideas
    }

    # Attribute each portfolio-level FAIL to the ideas that individually push that metric the wrong way.
    solo: Dict[str, Tuple[Dict[str, Any], float]] = {}
    for idea in ideas:
        solo_exposure = portfolio_exposure_calculator(holdings, _trades([idea]))
        solo[idea.idea_id] = (portfolio_metrics(solo_exposure.after_holdings, profile), solo_exposure.unnormalized_after_weight)

    for result in portfolio_results:
        if result.outcome != "FAIL" or not result.metric:
            continue
        for idea in _contributors(result, ideas, before, solo) or ideas:  # no single cause: revisit all ideas
            per_idea[idea.idea_id].append(result.model_copy(update={"idea_id": idea.idea_id}))

    return {
        "portfolio_results": portfolio_results,
        "idea_results": per_idea,
        "before_metrics": to_risk_metrics(before, before),
        "after_metrics": to_risk_metrics(before, after),
        "after_holdings": exposure.after_holdings
    }
