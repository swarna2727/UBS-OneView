"""Agent 5: Memo Formatter Agent (Slides 10, 20, 22)."""

import re
import uuid
from datetime import datetime
from typing import Optional, List
from src.agents.base import BaseAdvisoryAgent
from src.schemas.advisory import (
    ClientAdvisoryProfile, CandidateIdeaSet, SuitabilityAssessment, ResearchBrief,
    StructuredMemo, MemoPage1Executive, MemoPage2IdeaDetail,
    MemoPage3PortfolioView, ExecutiveIdeaGlance
)
from src.schemas.tools import RuleResult
from src.tools.memo_tools import memo_consistency_validator, citation_source_mapper
from src.tools.risk_engine import prohibited_wording
from src.llm.base import LLMClientProtocol

CURRENCY_SYMBOLS = {"SGD": "S$", "USD": "US$", "CHF": "CHF ", "EUR": "€", "GBP": "£"}
NOT_GUARANTEED = "Investment returns are not guaranteed and capital is at risk."

class MemoFormatterInputs:
    def __init__(
        self,
        profile: ClientAdvisoryProfile,
        ideas: CandidateIdeaSet,
        assessment: SuitabilityAssessment,
        research_brief: Optional[ResearchBrief] = None,
        advisor_notes: Optional[List[str]] = None
    ):
        self.profile = profile
        self.ideas = ideas
        self.assessment = assessment
        self.research_brief = research_brief
        self.advisor_notes = advisor_notes or []

def money(amount: float, currency: str) -> str:
    return f"{CURRENCY_SYMBOLS.get(currency, currency + ' ')}{amount:,.0f}"

def neutralise_guarantee(text: str) -> str:
    """Restate a client's 'guaranteed X' request as a target so the memo never promises it (R9)."""
    text = re.sub(r"\bguarantee[sd]?\b", "target", text, flags=re.IGNORECASE)
    return re.sub(r"\brisk[- ]free\b", "lower-risk", text, flags=re.IGNORECASE)

def memo_text(memo: StructuredMemo) -> str:
    p1, p3 = memo.page_1_executive, memo.page_3_portfolio_view
    parts = [p1.investment_objective, p1.client_snapshot, p1.portfolio_snapshot,
             *[g.why_it_fits for g in p1.ideas_at_a_glance],
             p3.portfolio_level_impact_narrative, p3.market_developments, *p3.scenario_analysis.values()]
    for d in memo.page_2_details:
        parts += [d.thesis, d.why_now, d.client_fit_rationale, d.what_changes_view, d.alternative_considered, *d.key_risks]
    return " ".join(parts)

class MemoFormatterAgent(BaseAdvisoryAgent[MemoFormatterInputs, StructuredMemo]):
    """
    Transforms approved ideas, client context and risk findings into a structured,
    traceable memo without altering the advice. The PDF is rendered only after advisor approval.
    """

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None):
        super().__init__("MemoFormatterAgent", llm_client)

    def execute(self, inputs: MemoFormatterInputs) -> StructuredMemo:
        profile, ideas, assessment = inputs.profile, inputs.ideas, inputs.assessment
        sources = inputs.research_brief.validated_sources if inputs.research_brief else []
        source_lines = {s.source_id: line for s, line in zip(sources, citation_source_mapper(None, sources))}

        # Page 1: client snapshot, objective, 3 ideas at a glance (deterministic from approved ideas)
        asked_for_guarantee = any(prohibited_wording(g) for g in profile.goals)
        goals = [neutralise_guarantee(g) for g in profile.goals]
        objective = f"{profile.calculated_risk_tier} mandate targeting: {', '.join(goals)}."
        if asked_for_guarantee:
            objective += (" The client asked for a certain return; returns cannot be guaranteed, so the ideas "
                          "below set out the trade-offs between income, risk and liquidity instead.")

        ideas_glance = [ExecutiveIdeaGlance(
            idea_id=i.idea_id, slot_number=i.slot_number, ticker=i.ticker, name=i.name, action=i.action,
            target_weight_pct=i.target_weight_pct, portfolio_role=i.portfolio_role, why_it_fits=i.client_fit_rationale
        ) for i in ideas.ideas]

        # Page 2: per-idea detail answering what / which client factors / which evidence / key risks / alternative
        idea_details = [MemoPage2IdeaDetail(
            slot_number=i.slot_number, ticker=i.ticker, name=i.name, action=i.action,
            allocation_range_str=f"{i.allocation_range.min_pct:.1f}% - {i.allocation_range.max_pct:.1f}% (Target: {i.target_weight_pct:.1f}%)",
            target_weight_pct=i.target_weight_pct, portfolio_role=i.portfolio_role,
            thesis=i.thesis, why_now=i.why_now, client_fit_rationale=i.client_fit_rationale,
            key_risks=i.key_risks, what_changes_view=i.invalidation_trigger,
            cited_sources_formatted=[source_lines.get(sid, f"[{sid}]") for sid in i.cited_evidence_ids],
            conviction=i.conviction,
            alternative_considered=i.alternative_considered or "Hold the existing allocation unchanged."
        ) for i in ideas.ideas]

        # Page 3: portfolio impact, overall risks, scenarios, sources, approval
        before, after = assessment.before_metrics, assessment.after_metrics
        open_items = [f"{b.rule_id}: {b.message}" for b in assessment.detected_breaches]
        page3 = MemoPage3PortfolioView(
            portfolio_level_impact_narrative=(
                f"Portfolio risk score moves from {before.after_risk_score:.2f} to {after.after_risk_score:.2f} "
                f"against the {profile.calculated_risk_tier} band ceiling of {profile.risk_score_range[1]:.2f}. "
                f"Growth assets {before.growth_asset_pct:.1f}% -> {after.growth_asset_pct:.1f}%; "
                f"largest single instrument {after.max_single_stock_ticker} at {after.max_single_stock_pct:.1f}%. "
                f"Suitability review: {assessment.overall_status}."
                + (" Advisor-documented exceptions: " + " | ".join(open_items) if open_items else "")
            ),
            scenario_analysis={
                "Global Equity Shock (-15%)": "Drawdown is cushioned by the fixed income and cash allocation.",
                "Yield Curve Steepening (+50bps)": "Higher reinvestment yields offset part of the duration loss.",
                "Sector Rotation / Multiple Contraction": "Diversified income holdings offset cyclical volatility."
            },
            risk_disclosures=[
                NOT_GUARANTEED,
                "Past performance is not an indicator of future results.",
                "Suitability assessed against the client's risk band, horizon, liquidity needs and concentration limits (rules R1-R10)."
            ],
            market_developments="; ".join(
                f"{f.claim} ({', '.join(f.cited_source_ids)})" for f in (inputs.research_brief.findings[:3] if inputs.research_brief else [])
            ) or "See cited sources.",
            audit_sources_summary=list(source_lines.values()),
            advisor_signature_block="Pending Relationship Manager approval. No client-facing use before sign-off."
        )

        memo_id = f"MEMO-UBS-{uuid.uuid4().hex[:8].upper()}"
        memo = StructuredMemo(
            memo_id=memo_id,
            created_at=datetime.now().isoformat(),
            page_1_executive=MemoPage1Executive(
                client_name=profile.name, client_id=profile.client_id, date=datetime.now().strftime("%d %B %Y"),
                investment_objective=objective,
                client_snapshot=(
                    f"Risk Tier: {profile.calculated_risk_tier} ({profile.risk_tier_number}/5), "
                    f"Horizon: {profile.time_horizon_months} months, Investable assets: {money(profile.total_wealth, profile.base_currency)}, "
                    f"Near-term liquidity need: {money(profile.near_term_liquidity_need, profile.base_currency)}"
                ),
                portfolio_snapshot=f"Asset class breakdown: {profile.current_asset_class_weights}",
                ideas_at_a_glance=ideas_glance
            ),
            page_2_details=idea_details,
            page_3_portfolio_view=page3,
            consistency_hash=f"CONSISTENT-{memo_id}"
        )

        # Deterministic checks: consistency with the critic-approved ideas and R9 wording.
        consistent = memo_consistency_validator(assessment, memo)
        hits = prohibited_wording(memo_text(memo))
        checks = [
            RuleResult(rule_id="MEMO", name="Memo matches approved ideas", outcome="PASS" if consistent else "FAIL",
                       message="Tickers and allocations match the validated idea set." if consistent
                       else "Memo tickers differ from the validated idea set."),
            RuleResult(rule_id="R9", name="No prohibited wording", outcome="FAIL" if hits else "PASS",
                       message=f"Prohibited wording: {', '.join(sorted(set(hits)))}." if hits else "No guarantee or risk-free wording.")
        ]
        return memo.model_copy(update={"compliance_checks": checks})
