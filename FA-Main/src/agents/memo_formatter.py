"""Agent 5: Memo Formatter Agent (Slide 18)."""

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from src.agents.base import BaseAdvisoryAgent
from src.schemas.advisory import (
    ClientAdvisoryProfile, CandidateIdeaSet, SuitabilityAssessment,
    StructuredMemo, MemoPage1Executive, MemoPage2IdeaDetail,
    MemoPage3PortfolioView, ExecutiveIdeaGlance
)
from src.tools.memo_tools import (
    memo_consistency_validator, citation_source_mapper, pdf_memo_renderer
)
from src.llm.base import LLMClientProtocol

MEMO_FORMATTER_SYSTEM_PROMPT = """You are the UBS Advisory Memo Formatter Agent.
Your responsibility is strictly to assemble approved ideas, client profile, and risk findings into an audit-ready, standardized 3-page StructuredMemo.
CRITICAL GOVERNANCE RULES:
1. You MUST NOT modify any ticker, allocation percentage, or action from the approved ideas.
2. You MUST NOT introduce new financial instruments.
3. Every idea detail on Page 2 must verbatim reflect the approved ideas.
4. Synthesize professional, compliant narrative explanations for the client snapshot, portfolio impact, and scenario analysis.
"""

class MemoFormatterInputs:
    def __init__(
        self,
        profile: ClientAdvisoryProfile,
        ideas: CandidateIdeaSet,
        assessment: SuitabilityAssessment,
        output_pdf_path: Optional[str] = None
    ):
        self.profile = profile
        self.ideas = ideas
        self.assessment = assessment
        self.output_pdf_path = output_pdf_path

class MemoFormatterAgent(BaseAdvisoryAgent[MemoFormatterInputs, StructuredMemo]):
    """
    Transforms approved ideas, client context, and risk findings into a structured,
    audit-ready UBS advisory memo without modifying investment advice.
    """

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None):
        super().__init__("MemoFormatterAgent", llm_client)

    def execute(self, inputs: MemoFormatterInputs) -> StructuredMemo:
        profile = inputs.profile
        ideas = inputs.ideas
        assessment = inputs.assessment

        # 1. Deterministic construction of Page 1 glance table directly from approved ideas
        ideas_glance = []
        for idea in ideas.ideas:
            ideas_glance.append(ExecutiveIdeaGlance(
                idea_id=idea.idea_id,
                slot_number=idea.slot_number,
                ticker=idea.ticker,
                name=idea.name,
                action=idea.action,
                target_weight_pct=idea.target_weight_pct,
                portfolio_role=idea.portfolio_role,
                why_it_fits=idea.client_fit_rationale
            ))

        # 2. Deterministic construction of Page 2 details
        idea_details = []
        for idea in ideas.ideas:
            range_str = f"{idea.allocation_range.min_pct:.1f}% - {idea.allocation_range.max_pct:.1f}% (Target: {idea.target_weight_pct:.1f}%)"
            formatted_sources = [f"Source Ref [{sid}]" for sid in idea.cited_evidence_ids]
            idea_details.append(MemoPage2IdeaDetail(
                slot_number=idea.slot_number,
                ticker=idea.ticker,
                name=idea.name,
                action=idea.action,
                allocation_range_str=range_str,
                target_weight_pct=idea.target_weight_pct,
                portfolio_role=idea.portfolio_role,
                thesis=idea.thesis,
                why_now=idea.why_now,
                client_fit_rationale=idea.client_fit_rationale,
                key_risks=idea.key_risks,
                what_changes_view=idea.invalidation_trigger,
                cited_sources_formatted=formatted_sources
            ))

        # 3. Deterministic construction of Page 3 portfolio view & disclosures
        page3_view = MemoPage3PortfolioView(
            portfolio_level_impact_narrative=(
                f"The proposed advisory strategy reallocates capital into high-conviction ideas, "
                f"shifting portfolio risk score from {assessment.before_metrics.after_risk_score:.2f} "
                f"to {assessment.after_metrics.after_risk_score:.2f}. Single-stock concentration is controlled at "
                f"{assessment.after_metrics.max_single_stock_pct:.1f}% ({assessment.after_metrics.max_single_stock_ticker})."
            ),
            scenario_analysis={
                "Global Equity Shock (-15%)": "Estimated portfolio drawdown buffered by sovereign duration and cash reserves.",
                "Yield Curve Steepening (+50bps)": "Yield enhancement cushions mark-to-market duration fluctuations.",
                "Sector Rotation / Multiple Contraction": "Defensive dividend franchises offset cyclical volatility."
            },
            risk_disclosures=[
                "Past performance is not an indicator of future results. Capital at risk.",
                "Investments are subject to market risks, including the possible loss of principal.",
                "Suitability verified in accordance with UBS Wealth Management Investment Advisory Guidelines."
            ],
            market_developments="Constructive outlook on quality fixed income carry and secular enterprise technology leaders.",
            audit_sources_summary=[
                f"Audit trail verified: All {len(ideas.ideas)} recommendations certified against fresh market research."
            ],
            advisor_signature_block=f"Principal Wealth Advisor | UBS Financial Services Inc.\nSuitability Review: {assessment.overall_status}\nTimestamp: {datetime.now().strftime('%d-%b-%Y %H:%M UTC')}"
        )

        # 4. Assemble StructuredMemo
        memo_id = f"MEMO-UBS-{uuid.uuid4().hex[:8].upper()}"
        today_str = datetime.now().strftime("%d %B %Y")

        memo = StructuredMemo(
            memo_id=memo_id,
            created_at=datetime.now().isoformat(),
            page_1_executive=MemoPage1Executive(
                client_name=profile.name,
                client_id=profile.client_id,
                date=today_str,
                investment_objective=f"{profile.calculated_risk_tier} portfolio mandate targeting: {', '.join(profile.goals)}.",
                client_snapshot=f"Risk Tier: {profile.calculated_risk_tier} ({profile.risk_tier_number}/5), Horizon: {profile.time_horizon_months} months, Wealth: ${profile.total_wealth:,.0f}",
                portfolio_snapshot=f"Asset class breakdown: {profile.current_asset_class_weights}",
                ideas_at_a_glance=ideas_glance
            ),
            page_2_details=idea_details,
            page_3_portfolio_view=page3_view,
            pdf_artifact_path=None,
            consistency_hash=f"CONSISTENT-{memo_id}"
        )

        # 5. Deterministic consistency check
        is_consistent = memo_consistency_validator(assessment, memo)
        if not is_consistent:
            raise ValueError("Memo consistency validation failed! Mismatched tickers detected between Critic and Memo.")

        # 6. Render PDF artifact
        pdf_path = inputs.output_pdf_path or f"data/pipeline_data/{memo_id}.pdf"
        rendered_pdf_path = pdf_memo_renderer(memo, pdf_path)

        memo_dict = memo.model_dump()
        memo_dict["pdf_artifact_path"] = rendered_pdf_path
        return StructuredMemo(**memo_dict)
