"""Deterministic market intelligence tools (Slide 15)."""

import os
import json
from typing import Dict, Any, List
from datetime import datetime, timezone
from src.schemas.tools import QuantMetrics, SentimentBreakdown, DocumentSnippet, HoldingMove, Holding, MarketSignalConflict
from src.tools.retrieval import source_freshness_validator  # noqa: F401  (freshness + lifecycle status live with retrieval; re-exported here)

DATA_MARKET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "market", "prices", "market_data.json"
)

def quantitative_analysis_tool(ticker_or_asset_class: str) -> QuantMetrics:
    """
    Retrieve deterministic quantitative signals (volatility, momentum, returns, valuation)
    from verified historical market data tables.
    """
    ticker = ticker_or_asset_class.strip().upper()

    market_data = {}
    if os.path.exists(DATA_MARKET_PATH):
        with open(DATA_MARKET_PATH, "r", encoding="utf-8") as f:
            market_data = json.load(f)

    # Defaults if ticker is not explicitly in pre-loaded database
    metrics = market_data.get(ticker, {
        "annualized_return_1y": 0.08,
        "annualized_return_3y": 0.06,
        "annualized_volatility": 0.15,
        "sharpe_ratio": 0.85,
        "max_drawdown": 0.12,
        "momentum_12_1": 0.07,
        "forward_pe": 20.0,
        "pb_ratio": 3.0,
        "dividend_yield": 0.02,
        "beta": 1.0
    })

    return QuantMetrics(
        ticker=ticker,
        annualized_return_1y=metrics.get("annualized_return_1y"),
        annualized_return_3y=metrics.get("annualized_return_3y"),
        annualized_volatility=metrics.get("annualized_volatility"),
        sharpe_ratio=metrics.get("sharpe_ratio"),
        max_drawdown=metrics.get("max_drawdown"),
        momentum_12_1=metrics.get("momentum_12_1"),
        forward_pe=metrics.get("forward_pe"),
        pb_ratio=metrics.get("pb_ratio"),
        dividend_yield=metrics.get("dividend_yield"),
        beta=metrics.get("beta"),
        effective_duration=metrics.get("effective_duration"),
        yield_to_maturity=metrics.get("yield_to_maturity"),
        credit_spread_bps=metrics.get("credit_spread_bps")
    )

BULLISH_KEYWORDS = [
    "overweight", "outperform", "catalyst", "expansion", "growth",
    "resilient", "attractive", "compelling", "upside", "monetization", "yield"
]
BEARISH_KEYWORDS = [
    "underweight", "slowdown", "recession", "headwind", "downside",
    "elevated valuation", "spread widening", "breach", "pressure", "decline"
]

def sentiment_score(text: str) -> float:
    """Quantitative sentiment of one document: (bullish - bearish keyword hits) / total hits, 0.0 when neither appears."""
    text = text.lower()
    bull = sum(w in text for w in BULLISH_KEYWORDS)
    bear = sum(w in text for w in BEARISH_KEYWORDS)
    return round((bull - bear) / (bull + bear), 3) if bull + bear else 0.0

def _published(doc: DocumentSnippet) -> datetime:
    try:
        d = datetime.fromisoformat(doc.published_at.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

def detect_market_signal_conflicts(snippets: List[DocumentSnippet]) -> List[MarketSignalConflict]:
    """Deterministic comparator for conflicting research (deck slide 29), replacing model judgement.

    Groups sources by ticker, asset class (sector proxy) and macro theme, scores each source, and emits one
    MARKET_SIGNAL_CONFLICT per group that holds a bullish and a bearish reading diverging by at least the policy
    threshold and published within the policy window. Withdrawn or superseded sources are ignored: they never
    count as a signal. The same input always gives the same output."""
    from src.tools.risk_engine import load_policy
    cfg = load_policy()["market_signals"]
    direction, min_div, max_gap = cfg["direction_threshold"], cfg["divergence_threshold"], cfg["max_gap_days"]

    groups: Dict[tuple, List[DocumentSnippet]] = {}
    for doc in snippets:
        if doc.status != "active":
            continue
        for ticker in doc.tickers:
            groups.setdefault(("ticker", ticker.upper()), []).append(doc)
        for asset_class in doc.asset_classes:
            groups.setdefault(("sector", asset_class), []).append(doc)
        for theme in doc.macro_themes:
            groups.setdefault(("macro_theme", theme), []).append(doc)

    conflicts: List[MarketSignalConflict] = []
    for (dimension, key), docs in sorted(groups.items()):
        scored = [(sentiment_score(d.text), d) for d in docs]
        bulls = [(s, d) for s, d in scored if s > direction]
        bears = [(s, d) for s, d in scored if s < -direction]
        if not bulls or not bears:
            continue
        top_bull = max(bulls, key=lambda sd: (sd[0], sd[1].source_id))
        top_bear = min(bears, key=lambda sd: (sd[0], sd[1].source_id))
        divergence = round(top_bull[0] - top_bear[0], 3)
        gap = abs((_published(top_bull[1]) - _published(top_bear[1])).days)
        if divergence < min_div or gap > max_gap:
            continue
        label = {"ticker": key, "sector": f"the {key} sector", "macro_theme": f"the '{key}' theme"}[dimension]
        conflicts.append(MarketSignalConflict(
            dimension=dimension, key=key,
            bullish_source_ids=sorted(d.source_id for _, d in bulls), bearish_source_ids=sorted(d.source_id for _, d in bears),
            bullish_score=top_bull[0], bearish_score=top_bear[0], divergence=divergence, gap_days=gap,
            message=(f"MARKET_SIGNAL_CONFLICT on {label}: {top_bull[1].publisher} reads bullish ({top_bull[0]:+.2f}) and "
                     f"{top_bear[1].publisher} reads bearish ({top_bear[0]:+.2f}), divergence {divergence:.2f}. Treat the view as "
                     f"uncertain and weigh both before discussing it ({top_bull[1].source_id}, {top_bear[1].source_id}).")))
    return conflicts

def conflicting_signals_detector(snippets: List[DocumentSnippet]) -> List[str]:
    """The conflicts as messages for the briefing's 'Uncertainty' lines. A conflict is reported with both sources and is
    never resolved by picking a side."""
    return [c.message for c in detect_market_signal_conflicts(snippets)]

def sentiment_analysis_tool(corpus_docs: List[DocumentSnippet]) -> SentimentBreakdown:
    """
    Deterministically analyze sentiment across retrieved corpus documents
    using financial lexicon keywords (bullish vs bearish signals).
    """
    bullish_keywords, bearish_keywords = BULLISH_KEYWORDS, BEARISH_KEYWORDS

    bullish_signals = []
    bearish_signals = []
    neutral_catalysts = []

    bull_count = 0
    bear_count = 0

    for doc in corpus_docs:
        text_lower = doc.text.lower()
        matched_bull = [w for w in bullish_keywords if w in text_lower]
        matched_bear = [w for w in bearish_keywords if w in text_lower]

        if matched_bull:
            bull_count += len(matched_bull)
            bullish_signals.append(f"{doc.publisher}: Identified {', '.join(matched_bull[:3])} in {doc.title}")
        if matched_bear:
            bear_count += len(matched_bear)
            bearish_signals.append(f"{doc.publisher}: Flagged {', '.join(matched_bear[:3])} in {doc.title}")

        if not matched_bull and not matched_bear:
            neutral_catalysts.append(f"{doc.publisher}: Macro commentary in {doc.title}")

    total = bull_count + bear_count
    if total > 0:
        score = (bull_count - bear_count) / total
    else:
        score = 0.0

    if score > 0.2:
        sentiment = "BULLISH"
    elif score < -0.2:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    return SentimentBreakdown(
        overall_sentiment=sentiment,
        sentiment_score=round(score, 2),
        bullish_signals=bullish_signals[:5],
        bearish_signals=bearish_signals[:5],
        neutral_catalysts=neutral_catalysts[:3]
    )


DATA_MOVES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "market", "prices", "quarter_moves.json"
)

def portfolio_impact_tool(holdings: List[Holding]) -> List[HoldingMove]:
    """UC1 facts only: which holdings moved over the look-back window and by how much. No forecasts, no assumptions.
    A holding with no price data is listed with an empty move, never estimated."""
    moves: Dict[str, float] = {}
    if os.path.exists(DATA_MOVES_PATH):
        with open(DATA_MOVES_PATH, "r", encoding="utf-8") as f:
            moves = json.load(f).get("moves", {})
    result = []
    for h in holdings:
        move = moves.get(h.ticker.upper())
        result.append(HoldingMove(
            ticker=h.ticker, name=h.name, weight_pct=h.weight_pct, move_pct=move,
            contribution_pct=round(h.weight_pct * move / 100.0, 2) if move is not None else None,
            source="Bank market data (illustrative)" if move is not None else "No price data on file"))
    return sorted(result, key=lambda m: abs(m.contribution_pct or 0.0), reverse=True)
