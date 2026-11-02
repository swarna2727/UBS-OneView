"""Level 1, component evaluation (deck slide 24): did each part work?

Schema validation, calculation accuracy and source validity. Pure deterministic checks over a finished run."""

from datetime import datetime, timezone
from typing import Any, Dict, List
from pydantic import TypeAdapter, ValidationError

from src.evaluation.runner import RunResult
from src.schemas.advisory import (
    ClientAdvisoryProfile, OfferMap, ResearchBrief, CandidateIdeaSet, SuitabilityAssessment, MeetingBrief,
    StructuredMemo, MeetingRecord, ProfileChangeProposal
)
from src.tools.exposure import portfolio_exposure_calculator
from src.tools.market_tools import DATA_MOVES_PATH
from src.tools.risk_engine import DEFAULT_REFERENCE_DATE

STAGE_MODELS: Dict[str, Any] = {
    "client_profile": ClientAdvisoryProfile, "client_limits": OfferMap, "market_intelligence": ResearchBrief,
    "investment_ideas": CandidateIdeaSet, "suitability_critic": SuitabilityAssessment, "meeting_brief": MeetingBrief,
    "memo_formatter": StructuredMemo, "meeting_record": MeetingRecord, "profile_change_check": List[ProfileChangeProposal],
}

def check(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}

def schema_validation(run: RunResult) -> List[Dict[str, Any]]:
    """Every agent output in the run log re-validates against its schema."""
    results = []
    for e in run.log:
        model = STAGE_MODELS.get(e.get("stage")) if e["event"] == "agent_output" else None
        if model is None:
            continue
        try:
            TypeAdapter(model).validate_python(e["output"])
            results.append(check(f"schema:{e['stage']}", True))
        except ValidationError as exc:
            results.append(check(f"schema:{e['stage']}", False, str(exc.errors()[0]["msg"])))
    if not results:
        results.append(check("schema:any-output", False, "no agent output in the run log"))
    return results

def _risk_score(holdings: List[Any]) -> float:
    total = sum(h.weight_pct for h in holdings) or 1.0
    return round(sum(h.risk_rating * h.weight_pct for h in holdings) / total, 2)

def calculation_accuracy(run: RunResult) -> List[Dict[str, Any]]:
    """Numbers the system reports match an independent recomputation."""
    s, out = run.state, []
    profile = s.get("profile")
    if profile:
        out.append(check("calc:holdings-sum-100", abs(sum(h.weight_pct for h in profile.parsed_holdings) - 100) < 0.01))
    assessment, ideas = s.get("assessment"), s.get("candidate_ideas")
    if profile and assessment and ideas:
        trades = [{"ticker": i.ticker, "action": i.action, "target_weight_pct": i.target_weight_pct,
                   "funding_source_ticker": i.funding_source_ticker} for i in ideas.ideas]
        after = portfolio_exposure_calculator(profile.parsed_holdings, trades).after_holdings
        out.append(check("calc:risk-score-after", abs(_risk_score(after) - assessment.after_metrics.after_risk_score) <= 0.01,
                         f"recomputed {_risk_score(after)} vs reported {assessment.after_metrics.after_risk_score}"))
        out.append(check("calc:risk-score-before", abs(_risk_score(profile.parsed_holdings) - assessment.before_metrics.after_risk_score) <= 0.01))
    brief = s.get("research_brief")
    if brief and brief.portfolio_impact:
        import json
        moves = json.load(open(DATA_MOVES_PATH))["moves"]
        ok = all((m.move_pct is None and m.ticker.upper() not in moves) or
                 (m.move_pct == moves.get(m.ticker.upper()) and m.contribution_pct == round(m.weight_pct * m.move_pct / 100, 2))
                 for m in brief.portfolio_impact)
        out.append(check("calc:portfolio-impact", ok, "weight x move recomputed from the market-data file"))
    om = s.get("offer_map")
    if om:
        eligible, blocked = set(om.eligible_tickers), set(om.blocked_tickers)
        out.append(check("calc:offer-map-partition", not (eligible & blocked) and bool(eligible | blocked),
                         "no product is both eligible and blocked"))
    return out

def source_validity(run: RunResult) -> List[Dict[str, Any]]:
    """Every source is real, dated and inside the window; everything cited is a retrieved source."""
    s, out = run.state, []
    brief = s.get("research_brief")
    if not brief:
        return out
    ref = datetime.fromisoformat(DEFAULT_REFERENCE_DATE.replace("Z", "+00:00"))
    ids = {src.source_id for src in brief.validated_sources}
    bad_dates, stale = [], []
    for src in brief.validated_sources:
        try:
            pub = datetime.fromisoformat(src.published_at.replace("Z", "+00:00"))
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if (ref - pub).days > brief.window_days:
                stale.append(src.source_id)
        except ValueError:
            bad_dates.append(src.source_id)
    out.append(check("source:dates-parse", not bad_dates, ",".join(bad_dates)))
    out.append(check("source:within-window", not stale, f"older than {brief.window_days} days: {','.join(stale)}"))
    cited = {c for f in brief.findings for c in f.cited_source_ids}
    out.append(check("source:findings-cite-retrieved", cited <= ids, ",".join(sorted(cited - ids))))
    ideas = s.get("candidate_ideas")
    if ideas:
        idea_cited = {c for i in ideas.ideas for c in i.cited_evidence_ids}
        out.append(check("source:ideas-cite-retrieved", idea_cited <= ids, ",".join(sorted(idea_cited - ids))))
    rm_news = [x for x in brief.validated_sources if x.source_id.startswith("RM-NEWS")]
    out.append(check("source:rm-news-labelled", all(x.publisher.startswith("RM-entered") for x in rm_news)))
    out.append(check("source:no-external-news", brief.external_news_used is False))
    return out

def component_level(run: RunResult) -> List[Dict[str, Any]]:
    return schema_validation(run) + calculation_accuracy(run) + source_validity(run)
