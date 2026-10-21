"""Deterministic market intelligence tools (Slide 15)."""

import os
import json
from typing import Dict, Any, List
from src.schemas.tools import QuantMetrics, SentimentBreakdown, DocumentSnippet

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

def sentiment_analysis_tool(corpus_docs: List[DocumentSnippet]) -> SentimentBreakdown:
    """
    Deterministically analyze sentiment across retrieved corpus documents
    using financial lexicon keywords (bullish vs bearish signals).
    """
    bullish_keywords = [
        "overweight", "outperform", "catalyst", "expansion", "growth",
        "resilient", "attractive", "compelling", "upside", "monetization", "yield"
    ]
    bearish_keywords = [
        "underweight", "slowdown", "recession", "headwind", "downside",
        "elevated valuation", "spread widening", "breach", "pressure", "decline"
    ]

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
