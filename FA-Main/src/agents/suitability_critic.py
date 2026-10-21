"""Agent 4: Suitability Critic Agent (Slide 17)."""

from typing import Optional, List, Dict, Any
from src.agents.base import BaseAdvisoryAgent
from src.schemas.advisory import (
    ClientAdvisoryProfile, CandidateIdeaSet, SuitabilityAssessment, IdeaStatus, ResearchBrief
)
from src.schemas.tools import RiskMetrics
from src.tools.risk_engine import portfolio_risk_engine, suitability_rule_engine
from src.tools.exposure import portfolio_exposure_calculator
from src.tools.evidence import evidence_integrity_check
from src.llm.base import LLMClientProtocol

class SuitabilityCriticInputs:
    def __init__(
        self,
        profile: ClientAdvisoryProfile,
        candidate_ideas: CandidateIdeaSet,
        research_brief: ResearchBrief
    ):
        self.profile = profile
        self.candidate_ideas = candidate_ideas
        self.research_brief = research_brief

class SuitabilityCriticAgent(BaseAdvisoryAgent[SuitabilityCriticInputs, SuitabilityAssessment]):
    """
    Independent risk engine and critic validating candidate ideas against client profile
    and overall portfolio impact. Emits PASS, FLAG, or FAIL status gates.
    """

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None):
        super().__init__("SuitabilityCriticAgent", llm_client)

    def execute(self, inputs: SuitabilityCriticInputs) -> SuitabilityAssessment:
        profile = inputs.profile
        idea_set = inputs.candidate_ideas
        brief = inputs.research_brief

        # 1. Deterministic evidence integrity check
        evidence_status = evidence_integrity_check(idea_set.ideas, brief)

        # 2. Deterministic portfolio risk engine
        risk_metrics = portfolio_risk_engine(
            current_portfolio=profile.parsed_holdings,
            candidate_ideas=idea_set.ideas,
            liquidity_needs_amount=50000.0,
            total_wealth=profile.total_wealth
        )

        # 3. Simulate after-holdings
        trades = [{
            "ticker": i.ticker,
            "action": i.action,
            "target_weight_pct": i.target_weight_pct,
            "funding_source_ticker": i.funding_source_ticker
        } for i in idea_set.ideas]
        exposure_delta = portfolio_exposure_calculator(profile.parsed_holdings, trades)

        # 4. Deterministic suitability rule engine (breach detection)
        rule_result = suitability_rule_engine(profile, exposure_delta.after_holdings)

        # Check evidence failure: if any idea lacks evidence, it's a hard FAIL
        detected_breaches = list(rule_result.breaches)
        if not evidence_status.all_evidence_valid:
            rule_result_status = "FAIL"
            for unsupported in evidence_status.unsupported_ideas:
                self.logger.warning(f"Evidence integrity failure: {unsupported}")
        else:
            rule_result_status = rule_result.overall_status

        # 5. Evaluate individual ideas
        idea_evaluations: List[IdeaStatus] = []
        for idea in idea_set.ideas:
            idea_breaches = [
                b for b in detected_breaches
                if idea.ticker.upper() in b.message.upper()
            ]

            # Specific per-idea checks
            if any(b.severity == "FAIL" for b in idea_breaches) or any(idea.ticker.upper() in u.upper() for u in evidence_status.unsupported_ideas):
                status = "FAIL"
                notes = f"Idea breaches hard risk limit or lacks fresh verified evidence."
            elif any(b.severity == "FLAG" for b in idea_breaches):
                status = "FLAG"
                notes = f"Idea triggers soft concentration/drift warning ({idea_breaches[0].metric_name})."
            else:
                status = "PASS"
                notes = f"Complies with risk tier, portfolio limits, and product universe."

            idea_evaluations.append(IdeaStatus(
                idea_id=idea.idea_id,
                slot_number=idea.slot_number,
                ticker=idea.ticker,
                status=status,
                detected_breaches=idea_breaches,
                qualitative_notes=notes,
                evidence_verified=idea.idea_id not in [u.split(':')[0] for u in evidence_status.unsupported_ideas]
            ))

        # Overall Status is deterministic: FAIL if any FAIL, FLAG if any FLAG, else PASS
        if any(ie.status == "FAIL" for ie in idea_evaluations) or rule_result_status == "FAIL":
            overall_status = "FAIL"
            critic_summary = "Hard suitability breach detected. Requires selective regeneration of failed ideas."
            has_hard_breach = True
        elif any(ie.status == "FLAG" for ie in idea_evaluations) or rule_result_status == "FLAG":
            overall_status = "FLAG"
            critic_summary = "Soft guideline breach detected. Pausing graph for Relationship Manager exception review."
            has_hard_breach = False
        else:
            overall_status = "PASS"
            critic_summary = "All candidate ideas and post-trade allocations pass UBS suitability standards."
            has_hard_breach = False

        # Required actions
        required_actions = []
        if overall_status == "FAIL":
            failed_slots = [ie.slot_number for ie in idea_evaluations if ie.status == "FAIL"]
            required_actions.append(f"Regenerate candidate ideas for slots: {failed_slots}")
        elif overall_status == "FLAG":
            required_actions.append("RM must review concentration and regional drift warnings and confirm exception.")
        else:
            required_actions.append("Proceed to structured advisory memo formatting.")

        return SuitabilityAssessment(
            overall_status=overall_status,
            idea_evaluations=idea_evaluations,
            before_metrics=RiskMetrics(
                before_risk_score=risk_metrics.before_risk_score,
                after_risk_score=risk_metrics.before_risk_score,
                risk_score_delta=0.0,
                max_single_stock_pct=exposure_delta.top_concentration_weight,
                max_single_stock_ticker=exposure_delta.top_concentration_ticker,
                us_equity_pct=risk_metrics.us_equity_pct,
                liquidity_coverage_ratio=risk_metrics.liquidity_coverage_ratio,
                unhedged_foreign_currency_pct=risk_metrics.unhedged_foreign_currency_pct
            ),
            after_metrics=risk_metrics,
            detected_breaches=detected_breaches,
            required_actions=required_actions,
            has_blocking_hard_breach=has_hard_breach,
            critic_notes=critic_summary
        )
