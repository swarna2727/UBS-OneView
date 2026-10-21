"""Agent 4: Suitability Critic Agent (Slides 19, 21, 27)."""

from typing import Optional, List
from src.agents.base import BaseAdvisoryAgent
from src.schemas.advisory import (
    ClientAdvisoryProfile, CandidateIdeaSet, SuitabilityAssessment, IdeaStatus, ResearchBrief
)
from src.schemas.tools import Breach, RuleResult
from src.tools.risk_engine import suitability_rule_engine, worst_outcome
from src.tools.universe import approved_product_universe_tool
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

def to_breach(r: RuleResult) -> Breach:
    """Project a FLAG/FAIL rule result onto the Breach shape the workspace renders."""
    return Breach(
        rule_id=r.rule_id,
        severity="FAIL" if r.outcome in ("FAIL", "BLOCK") else "FLAG",
        metric_name=r.metric or r.name,
        before_value=r.before_value or 0.0,
        after_value=r.after_value or 0.0,
        threshold_value=r.threshold_value or 0.0,
        message=f"[{r.rule_id}] {r.message}"
    )

class SuitabilityCriticAgent(BaseAdvisoryAgent[SuitabilityCriticInputs, SuitabilityAssessment]):
    """
    Independent critic validating candidate ideas against the client profile and the
    post-trade portfolio using the code-based rule engine (R1-R10). Emits PASS / FLAG / FAIL
    per idea and overall. It cannot approve its own exceptions: FLAGs go to the advisor and
    FAILs are regenerated.
    """

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None):
        super().__init__("SuitabilityCriticAgent", llm_client)

    def execute(self, inputs: SuitabilityCriticInputs) -> SuitabilityAssessment:
        profile = inputs.profile
        ideas = inputs.candidate_ideas.ideas
        approved = {p.ticker.upper() for p in approved_product_universe_tool()}

        engine = suitability_rule_engine(profile, ideas, inputs.research_brief, approved)
        portfolio_results: List[RuleResult] = engine["portfolio_results"]

        idea_evaluations: List[IdeaStatus] = []
        for idea in ideas:
            results = engine["idea_results"][idea.idea_id]
            status = worst_outcome([r.outcome for r in results])
            status = "FAIL" if status == "BLOCK" else status
            issues = [r for r in results if r.outcome != "PASS"]
            idea_evaluations.append(IdeaStatus(
                idea_id=idea.idea_id,
                slot_number=idea.slot_number,
                ticker=idea.ticker,
                status=status,
                detected_breaches=[to_breach(r) for r in issues],
                qualitative_notes="; ".join(f"{r.rule_id}: {r.message}" for r in issues)
                or "Passes R2 universe, R8 evidence and R9 wording checks; no portfolio breach attributed.",
                evidence_verified=all(r.outcome == "PASS" for r in results if r.rule_id == "R8"),
                rule_results=results
            ))

        portfolio_issues = [r for r in portfolio_results if r.outcome != "PASS"]
        overall = worst_outcome([r.outcome for r in portfolio_results] + [ie.status for ie in idea_evaluations])
        overall = "FAIL" if overall == "BLOCK" else overall

        failed_slots = [ie.slot_number for ie in idea_evaluations if ie.status == "FAIL"]
        if overall == "FAIL":
            notes = f"Hard suitability breach. Ideas in slots {failed_slots} must be regenerated; they cannot reach the memo."
            required = [f"Regenerate slots {failed_slots}."]
        elif overall == "FLAG":
            notes = "Judgment case: the advisor must document an exception or request a revision before the memo."
            required = [f"Advisor decision on: {', '.join(sorted({r.rule_id for r in portfolio_issues} | {b.rule_id for ie in idea_evaluations for b in ie.detected_breaches}))}."]
        else:
            notes = "All ideas and the proposed portfolio pass R1-R10."
            required = ["Proceed to memo drafting after advisor review."]

        # De-duplicate breaches: portfolio-level issues plus idea-only issues (R2/R8/R9).
        breaches = [to_breach(r) for r in portfolio_issues]
        breaches += [b for ie in idea_evaluations for b in ie.detected_breaches if b.rule_id in ("R2", "R8", "R9")]

        return SuitabilityAssessment(
            overall_status=overall,
            idea_evaluations=idea_evaluations,
            before_metrics=engine["before_metrics"],
            after_metrics=engine["after_metrics"],
            detected_breaches=breaches,
            required_actions=required,
            has_blocking_hard_breach=overall == "FAIL",
            critic_notes=notes,
            rule_results=portfolio_results
        )
