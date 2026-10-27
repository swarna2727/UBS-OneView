"""Evaluation metrics (slide 30). All pure functions over idea sets: no LLM, so scores are reproducible."""

from typing import Any, Dict, List, Optional, Sequence
from src.schemas.advisory import ClientAdvisoryProfile, CandidateIdeaSet, ResearchBrief, InvestmentIdea
from src.tools.risk_engine import suitability_rule_engine
from src.tools.universe import approved_product_universe_tool

HARD_OUTCOMES = ("FAIL", "BLOCK")

def suitability_violations(profile: ClientAdvisoryProfile, ideas: CandidateIdeaSet, brief: ResearchBrief) -> int:
    """Count of distinct hard rule breaches (FAIL/BLOCK) the deterministic engine finds in a recommendation set."""
    approved = {p.ticker.upper() for p in approved_product_universe_tool()}
    engine = suitability_rule_engine(profile, ideas.ideas, brief, approved)
    keys = {(r.rule_id, r.metric, r.idea_id) for r in engine["portfolio_results"] if r.outcome in HARD_OUTCOMES}
    for results in engine["idea_results"].values():
        keys |= {(r.rule_id, r.metric, r.idea_id) for r in results if r.outcome in HARD_OUTCOMES and r.rule_id in ("R2", "R8", "R9")}
    return len(keys)

def unsupported_claims(ideas: Sequence[InvestmentIdea], brief: ResearchBrief) -> int:
    """Ideas whose evidence is missing, or cites a source that is not a validated, fresh source of the brief."""
    fresh = {s.source_id for s in brief.validated_sources}
    return sum(1 for i in ideas if not i.cited_evidence_ids or not set(i.cited_evidence_ids) <= fresh)

def explainability(ideas: Sequence[InvestmentIdea]) -> float:
    """AS-7: share of the five questions answered per idea (what, client factors, evidence, risks, alternative)."""
    if not ideas:
        return 0.0
    score = 0
    for i in ideas:
        score += sum([
            bool(i.thesis.strip()), bool(i.client_fit_rationale.strip()), bool(i.cited_evidence_ids),
            bool(i.key_risks), bool((i.alternative_considered or "").strip()),
        ])
    return round(score / (5 * len(ideas)), 3)

def _signature(ideas: Sequence[InvestmentIdea]) -> set:
    return {(i.ticker, i.action, round(i.target_weight_pct, 1)) for i in ideas}

def set_distance(a: Sequence[InvestmentIdea], b: Sequence[InvestmentIdea]) -> float:
    """1 - Jaccard similarity of (ticker, action, weight): 0 = identical sets, 1 = nothing in common."""
    sa, sb = _signature(a), _signature(b)
    return round(1 - len(sa & sb) / len(sa | sb), 3) if sa | sb else 0.0

def personalization_sensitivity(base: Sequence[InvestmentIdea], perturbed: Sequence[InvestmentIdea]) -> float:
    """Recommendation sensitivity to changed client inputs: the larger, the more the set responds."""
    return set_distance(base, perturbed)

def output_consistency(runs: List[Sequence[InvestmentIdea]]) -> float:
    """Share of repeat runs of identical inputs that reproduce the first run's recommendation set."""
    if len(runs) < 2:
        return 1.0
    return round(sum(1 for r in runs[1:] if set_distance(runs[0], r) == 0.0) / (len(runs) - 1), 3)
