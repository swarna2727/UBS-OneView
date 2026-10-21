"""Deterministic Fake LLM Client for unit testing and offline execution."""

import json
import re
from typing import Type, TypeVar, Optional, Dict, Any
from pydantic import BaseModel
from src.schemas.advisory import (
    ResearchBrief, CandidateIdeaSet, InvestmentIdea, AllocationRange,
    SuitabilityAssessment, IdeaStatus, StructuredMemo,
    MemoPage1Executive, MemoPage2IdeaDetail, MemoPage3PortfolioView,
    ExecutiveIdeaGlance, ResearchFinding
)
from src.schemas.tools import QuantMetrics, SentimentBreakdown, DocumentSnippet, RiskMetrics

T = TypeVar("T", bound=BaseModel)

class FakeAdvisoryLLMClient:
    """Mock LLM client returning realistic, schema-valid advisory objects."""

    def model_for(self, agent_name: Optional[str] = None) -> str:
        return "offline-deterministic"

    def __init__(self, override_responses: Optional[Dict[str, Any]] = None):
        self.override_responses = override_responses or {}
        self.call_history = []
        self._slot_attempts = {1: 0, 2: 0, 3: 0}

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.2,
        agent_name: Optional[str] = None
    ) -> T:
        model_name = response_model.__name__
        self.call_history.append({
            "model_name": model_name,
            "agent_name": agent_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        })

        if model_name in self.override_responses:
            override_val = self.override_responses[model_name]
            if isinstance(override_val, BaseModel):
                return override_val # type: ignore
            elif isinstance(override_val, dict):
                return response_model.model_validate(override_val)

        if response_model == ResearchBrief:
            return self._build_fake_research_brief() # type: ignore
        elif response_model == CandidateIdeaSet:
            return self._build_fake_candidate_ideas(user_prompt) # type: ignore
        elif response_model == StructuredMemo:
            return self._build_fake_memo(user_prompt) # type: ignore
        else:
            raise ValueError(f"No mock response generator for model {model_name}")

    def _build_fake_research_brief(self) -> ResearchBrief:
        return ResearchBrief(
            macro_themes=[
                "Quality Yield & Duration Anchoring",
                "Secular AI Infrastructure Monetization",
                "Defensive Barbell Allocation"
            ],
            quantitative_signals={
                "MSFT": QuantMetrics(
                    ticker="MSFT",
                    annualized_return_1y=0.224,
                    annualized_return_3y=0.168,
                    annualized_volatility=0.215,
                    sharpe_ratio=1.04,
                    max_drawdown=0.142,
                    momentum_12_1=0.185,
                    forward_pe=29.5
                ),
                "IEF": QuantMetrics(
                    ticker="IEF",
                    annualized_return_1y=0.052,
                    annualized_return_3y=-0.018,
                    annualized_volatility=0.078,
                    sharpe_ratio=0.66,
                    max_drawdown=0.092,
                    momentum_12_1=0.045
                ),
                "NESN": QuantMetrics(
                    ticker="NESN",
                    annualized_return_1y=0.042,
                    annualized_return_3y=0.021,
                    annualized_volatility=0.138,
                    sharpe_ratio=0.30,
                    max_drawdown=0.115,
                    momentum_12_1=0.035
                )
            },
            sentiment_signals={
                "Macro": SentimentBreakdown(
                    overall_sentiment="BULLISH",
                    sentiment_score=0.45,
                    bullish_signals=["UBS CIO overweight High Quality Fixed Income", "Enterprise cloud demand steady"],
                    bearish_signals=["Premium multiples in mega-cap tech warrant discipline"]
                )
            },
            portfolio_gaps_addressed=[
                "Client fixed income duration is underweight vs 5-year investment horizon",
                "Lack of defensive European consumer staple exposure"
            ],
            findings=[
                ResearchFinding(
                    finding_id="FND-01",
                    theme="Quality Yield",
                    claim="Intermediate Treasuries and IG Corporate Credit offer 2.3% real yields and strong recession hedging.",
                    materiality_rank=1,
                    relevance_to_client="Provides ballast for capital preservation goal while maintaining steady yield.",
                    finding_type="QUANTITATIVE",
                    cited_source_ids=["SRC-UBS-CIO-2026-101", "SRC-FED-MINUTES-2026-09"],
                    publication_dates=["2026-10-10", "2026-09-28"],
                    freshness_verified=True
                ),
                ResearchFinding(
                    finding_id="FND-02",
                    theme="AI Infrastructure",
                    claim="Cloud software leaders retain pricing power with robust balance sheets, despite elevated P/E multiples.",
                    materiality_rank=2,
                    relevance_to_client="Satisfies long-term growth mandate with best-of-breed platform.",
                    finding_type="QUALITATIVE_SENTIMENT",
                    cited_source_ids=["SRC-UBS-CIO-2026-102"],
                    publication_dates=["2026-10-14"],
                    freshness_verified=True
                )
            ],
            key_market_risks=[
                "Sticky services inflation postponing central bank easing",
                "Valuation compression in high-beta US technology"
            ],
            validated_sources=[
                DocumentSnippet(
                    source_id="SRC-UBS-CIO-2026-101",
                    title="UBS CIO Global Wealth Management: Q4 Macro Strategy & Asset Allocation",
                    publisher="UBS Chief Investment Office",
                    published_at="2026-10-10T08:00:00Z",
                    retrieved_at="2026-10-15T10:00:00Z",
                    text="UBS CIO maintains an overweight stance on high-quality Fixed Income...",
                    tickers=["MSFT", "LQD", "IEF", "NESN"],
                    is_fresh=True,
                    age_days=11
                ),
                DocumentSnippet(
                    source_id="SRC-UBS-CIO-2026-102",
                    title="UBS CIO Sector Insights: Secular Demand in Semis and Cloud Software",
                    publisher="UBS Chief Investment Office",
                    published_at="2026-10-14T09:30:00Z",
                    retrieved_at="2026-10-16T11:00:00Z",
                    text="Enterprise cloud monetization and generative AI inference workloads...",
                    tickers=["MSFT", "TSM"],
                    is_fresh=True,
                    age_days=7
                ),
                DocumentSnippet(
                    source_id="SRC-FED-MINUTES-2026-09",
                    title="Federal Reserve Policy Review & Rate Trajectory",
                    publisher="Federal Open Market Committee",
                    published_at="2026-09-28T18:00:00Z",
                    retrieved_at="2026-10-02T12:00:00Z",
                    text="Benchmark policy rates are poised for gradual recalibration...",
                    tickers=["IEF", "BIL"],
                    is_fresh=True,
                    age_days=23
                )
            ]
        )

    # Offline idea pools per slot. A slot's first entry is the initial idea; each regeneration of
    # that slot moves to the next entry, so selective regeneration produces a genuinely different idea.
    IDEA_POOLS = {
        1: [
            ("VYMI", "EQ-VYMI", "Vanguard International High Dividend Yield ETF", 3.0, "INCOME",
             "Diversified non-US dividend payers add an income stream outside the existing US equity exposure.",
             "Dividend yields outside the US remain above their 10-year average while payout ratios are stable.",
             "Adds income the client asked for without adding to US equity concentration.",
             ["Non-US equity drawdown", "Currency translation risk"], "Dividend cuts across European banks and energy.",
             "Medium", "A US dividend fund (rejected: adds to US concentration).", ["SRC-UBS-CIO-2026-101"]),
            ("AZN", "EQ-AZN", "AstraZeneca PLC", 2.0, "INCOME",
             "Defensive healthcare franchise with a growing dividend and late-stage oncology pipeline.",
             "Recent pipeline read-outs de-risk earnings while the shares trade below sector multiples.",
             "Adds defensive income outside the US.", ["Trial failures", "Drug pricing reform"],
             "Two consecutive late-stage trial failures.", "Medium", "A global healthcare fund.", ["SRC-UBS-CIO-2026-101"]),
            ("NESN", "EQ-NESN", "Nestle S.A.", 3.0, "STABILITY",
             "Defensive global franchise with steady 3.5% dividend yield and low beta to broader market shocks.",
             "Provides portfolio stabilization against equity volatility while inflation pressures normalize.",
             "Aligns with capital preservation objective and complements tech exposure.",
             ["FX headwind from Swiss Franc appreciation", "Commodity packaging cost swings"],
             "Organic sales growth decelerating below 2.0% for two consecutive quarters.", "Low",
             "A consumer staples fund.", ["SRC-UBS-CIO-2026-101"]),
        ],
        2: [
            ("SPYG", "EQ-SPYG", "SPDR Portfolio S&P 500 Growth ETF", 10.0, "GROWTH",
             "US large-cap growth leaders continue to compound earnings through AI-driven productivity gains.",
             "Earnings revisions for US growth names remain positive into the next reporting season.",
             "Supports long-term growth over the client's horizon and adds an equity return source.",
             ["Equity drawdown", "Rate sensitivity of long-duration growth stocks", "Single-market exposure"],
             "Forward earnings revisions turning negative for two consecutive months.", "Medium",
             "A global equity fund (lower US weight).", ["SRC-UBS-CIO-2026-101"]),
            ("IEAC", "FI-EUCORP", "iShares Core EUR Corporate Bond ETF", 8.0, "DIVERSIFICATION",
             "European investment-grade credit offers carry with lower duration risk than US Treasuries.",
             "ECB easing supports spread compression while all-in yields remain near multi-year highs.",
             "Diversifies away from US exposure and adds stable income consistent with the client's band.",
             ["Credit spread widening", "EUR currency risk"], "Euro area IG spreads widening above 180bps.",
             "High", "US investment-grade credit (rejected: adds to US exposure).", ["SRC-EURO-CREDIT-2026-10"]),
            ("GLD", "CM-GOLD", "SPDR Gold Shares", 4.0, "DIVERSIFICATION",
             "Gold diversifies equity and rate risk while central-bank buying supports demand.",
             "Real yields are expected to fall as policy rates are cut.", "Adds a diversifier with low equity correlation.",
             ["No income", "Price volatility"], "Real yields rising above 2.5%.", "Low",
             "Inflation-linked bonds.", ["SRC-GOLD-MONITOR-2026-10"]),
        ],
        3: [
            ("LQD", "FI-USIG", "iShares iBoxx $ Investment Grade Corporate Bond ETF", 3.0, "STABILITY",
             "High-quality corporate credit locks in attractive all-in yields with modest default risk.",
             "Investment-grade yields remain near multi-year highs as the easing cycle begins.",
             "Adds stable income and dampens portfolio volatility in line with the client's risk band.",
             ["Credit spread widening", "Duration volatility"], "IG spreads widening above 150bps.",
             "High", "Short-dated Treasuries (lower yield).", ["SRC-UBS-CIO-2026-101"]),
            ("IEF", "FI-US10Y", "iShares 7-10 Year Treasury Bond ETF", 3.0, "STABILITY",
             "Lock in attractive real yields in high-quality intermediate Treasuries to lengthen duration.",
             "Monetary policy transition creates asymmetric upside during potential economic slowdown.",
             "Directly addresses the client's income goal with the highest credit quality.",
             ["Unexpected resurgence in core services inflation", "Duration volatility"],
             "Fed hawkish surprise re-accelerating terminal rate forecasts above 5.5%.", "Medium",
             "Holding cash (lower expected return).", ["SRC-UBS-CIO-2026-101", "SRC-FED-MINUTES-2026-09"]),
            ("MSFT", "EQ-MSFT", "Microsoft Corporation", 1.5, "GROWTH",
             "Secular enterprise software leader monetizing generative AI across extensive customer base.",
             "Accelerating commercial cloud bookings and operating margin expansion.",
             "Fulfills long-term growth mandate with high quality balance sheet.",
             ["Multiple compression from high valuation", "Antitrust scrutiny"],
             "Azure growth decelerating below 20% on constant currency basis.", "Low",
             "A diversified technology fund.", ["SRC-UBS-CIO-2026-101", "SRC-UBS-CIO-2026-102"]),
        ],
    }

    def _build_fake_candidate_ideas(self, user_prompt: str) -> CandidateIdeaSet:
        # A new idea set starts every slot at its first idea; a regeneration request advances only the listed slots.
        if "Is Regeneration Request: True" in user_prompt:
            match = re.search(r"Slots to Generate/Regenerate: \[([\d, ]*)\]", user_prompt)
            for slot in [int(x) for x in match.group(1).split(",") if x.strip()] if match else []:
                self._slot_attempts[slot] += 1
        else:
            self._slot_attempts = {1: 0, 2: 0, 3: 0}

        ideas = []
        for slot, pool in self.IDEA_POOLS.items():
            (ticker, inst, name, weight, role, thesis, why_now, fit, risks, trigger,
             conviction, alternative, evidence) = pool[min(self._slot_attempts[slot], len(pool) - 1)]
            ideas.append(InvestmentIdea(
                idea_id=f"IDEA-{ticker}-{slot:02d}", slot_number=slot, instrument_id=inst, ticker=ticker, name=name,
                action="ADD", target_weight_pct=weight,
                allocation_range=AllocationRange(min_pct=round(weight * 0.5, 1), max_pct=round(weight * 1.5, 1), target_pct=weight),
                funding_source_ticker="BIL", portfolio_role=role, thesis=thesis, why_now=why_now,
                client_fit_rationale=fit, key_risks=risks, invalidation_trigger=trigger,
                cited_evidence_ids=evidence, conviction=conviction, alternative_considered=alternative
            ))
        return CandidateIdeaSet(
            ideas=ideas,
            complementarity_thesis=f"Income, growth and stability roles across {', '.join(i.ticker for i in ideas)}, funded from the cash buffer.",
            generation_round=1 + max(self._slot_attempts.values())
        )

    def _build_fake_memo(self, user_prompt: str) -> StructuredMemo:
        return StructuredMemo(
            memo_id="MEMO-UBS-2026-991",
            created_at="2026-10-21T03:00:00Z",
            page_1_executive=MemoPage1Executive(
                client_name="Dr. Elena Rostova",
                client_id="CL-88201",
                date="21 October 2026",
                investment_objective="Balanced capital preservation with selective growth and enhanced fixed income yield.",
                client_snapshot="Moderate risk tolerance, 5-year time horizon, well-defined liquidity requirements.",
                portfolio_snapshot="Current portfolio: 40% Equities, 40% Fixed Income, 5% Commodities, 15% Cash/T-Bills.",
                ideas_at_a_glance=[
                    ExecutiveIdeaGlance(
                        idea_id="IDEA-IEF-01",
                        slot_number=1,
                        ticker="IEF",
                        name="iShares 7-10 Year Treasury Bond ETF",
                        action="ADD",
                        target_weight_pct=5.0,
                        portfolio_role="INCOME",
                        why_it_fits="Extends duration and locks in 4.1% sovereign yield."
                    ),
                    ExecutiveIdeaGlance(
                        idea_id="IDEA-NESN-02",
                        slot_number=2,
                        ticker="NESN",
                        name="Nestle S.A.",
                        action="ADD",
                        target_weight_pct=5.0,
                        portfolio_role="STABILITY",
                        why_it_fits="Low beta quality defensive dividend stream."
                    ),
                    ExecutiveIdeaGlance(
                        idea_id="IDEA-MSFT-03",
                        slot_number=3,
                        ticker="MSFT",
                        name="Microsoft Corporation",
                        action="BUY",
                        target_weight_pct=4.0,
                        portfolio_role="GROWTH",
                        why_it_fits="High quality secular AI infrastructure compounding."
                    )
                ]
            ),
            page_2_details=[
                MemoPage2IdeaDetail(
                    slot_number=1,
                    ticker="IEF",
                    name="iShares 7-10 Year Treasury Bond ETF",
                    action="ADD",
                    allocation_range_str="3.0% - 8.0% (Target: 5.0%)",
                    target_weight_pct=5.0,
                    portfolio_role="INCOME",
                    thesis="Lock in attractive real yields in high-quality intermediate Treasuries to lengthen duration.",
                    why_now="Monetary policy transition creates asymmetric upside during potential economic slowdown.",
                    client_fit_rationale="Directly addresses client's underweight duration and income goals.",
                    key_risks=["Unexpected resurgence in inflation", "Duration volatility"],
                    what_changes_view="Fed hawkish surprise re-accelerating terminal rate forecasts above 5.5%.",
                    cited_sources_formatted=["[SRC-UBS-CIO-2026-101] UBS CIO (2026-10-10)", "[SRC-FED-MINUTES-2026-09] FOMC (2026-09-28)"]
                ),
                MemoPage2IdeaDetail(
                    slot_number=2,
                    ticker="NESN",
                    name="Nestle S.A.",
                    action="ADD",
                    allocation_range_str="3.0% - 7.0% (Target: 5.0%)",
                    target_weight_pct=5.0,
                    portfolio_role="STABILITY",
                    thesis="Defensive global franchise with steady 3.5% dividend yield and low beta to broader market shocks.",
                    why_now="Provides portfolio stabilization against equity volatility while inflation pressures normalize.",
                    client_fit_rationale="Aligns with capital preservation objective and complements tech exposure.",
                    key_risks=["FX headwind from Swiss Franc appreciation", "Packaging cost inflation"],
                    what_changes_view="Organic sales growth decelerating below 2.0% for two consecutive quarters.",
                    cited_sources_formatted=["[SRC-UBS-CIO-2026-101] UBS CIO (2026-10-10)"]
                ),
                MemoPage2IdeaDetail(
                    slot_number=3,
                    ticker="MSFT",
                    name="Microsoft Corporation",
                    action="BUY",
                    allocation_range_str="2.0% - 6.0% (Target: 4.0%)",
                    target_weight_pct=4.0,
                    portfolio_role="GROWTH",
                    thesis="Secular enterprise software leader monetizing generative AI across extensive customer base.",
                    why_now="Accelerating commercial cloud bookings and operating margin expansion.",
                    client_fit_rationale="Fulfills long-term growth mandate with high quality balance sheet.",
                    key_risks=["Multiple compression from high valuation (29.5x forward P/E)", "Antitrust scrutiny"],
                    what_changes_view="Azure growth decelerating below 20% on constant currency basis.",
                    cited_sources_formatted=["[SRC-UBS-CIO-2026-101] UBS CIO (2026-10-10)", "[SRC-UBS-CIO-2026-102] UBS CIO (2026-10-14)"]
                )
            ],
            page_3_portfolio_view=MemoPage3PortfolioView(
                portfolio_level_impact_narrative="The recommended rebalancing shifts 14% of portfolio from T-Bills into high-conviction core allocations. Equities increase moderately from 40% to 49% while Fixed Income moves from 40% to 45%, leaving a disciplined 6% liquid cash reserve.",
                scenario_analysis={
                    "Global Equity Shock (-15%)": "Estimated portfolio drawdown limited to -5.8% cushioned by duration and defensive staples.",
                    "Yield Curve Steepening (+50bps)": "Negative duration impact of -0.38% offset by 4.8% carry and defensive dividend yield.",
                    "Tech Multiple Compression (-20%)": "Portfolio impact isolated to -0.8% due to strict single-stock position sizing."
                },
                risk_disclosures=[
                    "Past performance is not indicative of future returns. Investments are subject to market risks.",
                    "UBS Financial Services Inc. does not provide tax or legal advice. Consult personal counsel.",
                    "Suitability confirmed under UBS Chief Investment Office Guidelines (Issue Q4 2026)."
                ],
                market_developments="UBS CIO maintains a constructive stance on balanced portfolios, emphasizing quality yield and secular tech leadership.",
                audit_sources_summary=[
                    "[SRC-UBS-CIO-2026-101] UBS CIO Global Wealth Management Q4 Macro Strategy (2026-10-10)",
                    "[SRC-UBS-CIO-2026-102] UBS CIO Sector Insights: Semis & Cloud (2026-10-14)",
                    "[SRC-FED-MINUTES-2026-09] Federal Reserve FOMC Minutes (2026-09-28)"
                ],
                advisor_signature_block="Prepared by: Principal Wealth Advisor, UBS Advisory Solutions\nValidated by: Automated Suitability Engine (PASS)\nClient Approval Date: 21-Oct-2026"
            ),
            pdf_artifact_path=None,
            consistency_hash="VALID-PASS-HASH-88201"
        )
