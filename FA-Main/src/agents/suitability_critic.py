"""Check | Fit & Eligibility Critic (deck slide 22): independently tests each idea, and the resulting portfolio, against the client's circumstances and limits."""

from typing import Any, Dict, Optional, List
from src.agents.base import BaseAdvisoryAgent
from src.schemas.advisory import (
    ClientAdvisoryProfile, CandidateIdeaSet, SuitabilityAssessment, IdeaStatus, ResearchBrief,
    OfferMap, ProfileChangeCheck, ProfileChangeProposal
)
from src.tools.client_limits import profile_change_limits_check
from src.schemas.tools import Breach, RuleResult
from src.mcp_tools.gateway import portfolio_risk_engine
from src.tools.risk_engine import worst_outcome
from src.tools.universe import approved_product_universe_tool
from src.llm.base import LLMClientProtocol

class SuitabilityCriticInputs:
    def __init__(
        self,
        profile: ClientAdvisoryProfile,
        candidate_ideas: CandidateIdeaSet,
        research_brief: ResearchBrief,
        offer_map: Optional[OfferMap] = None
    ):
        self.profile = profile
        self.candidate_ideas = candidate_ideas
        self.research_brief = research_brief
        self.offer_map = offer_map  # what UBS can offer this client; None skips the E1 eligibility rule

class ProfileChangeInputs:
    """UC3: spoken profile changes the Critic checks against client limits and the risk band."""
    def __init__(self, raw_input: Dict[str, Any], proposals: List[ProfileChangeProposal]):
        self.raw_input = raw_input
        self.proposals = proposals

# Fit check labels (deck slide 10) and the rules behind them.
FIT_RULES = {"Risk range": "R4", "Horizon": "R6", "Concentration": "R7", "Liquidity": "L1"}

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

def _worst(outcomes: List[str]) -> str:
    outcomes = ["FAIL" if o == "BLOCK" else o for o in outcomes]
    return worst_outcome(outcomes)

class FitEligibilityCriticAgent(BaseAdvisoryAgent[Any, Any]):
    """
    Independent critic validating ideas against the client profile, the post-trade portfolio and the client's
    limits using the code-based rule engine (R1-R10, E1). Emits PASS / FLAG / FAIL per idea and overall.
    Passing risk is not the same as being eligible or suitable. It cannot approve its own exceptions: FLAGs go
    to the advisor, FAILs are regenerated, and an idea blocked by client limits is never shown.
    In UC3 it also checks each spoken profile change against the limits and the risk band.
    """

    PROMPT_VERSION = "v2"

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None):
        super().__init__("FitEligibilityCritic", llm_client)

    def execute(self, inputs: Any) -> Any:
        if isinstance(inputs, ProfileChangeInputs):
            return self._check_profile_changes(inputs)
        return self._assess_ideas(inputs)

    def _check_profile_changes(self, inputs: ProfileChangeInputs) -> List[ProfileChangeProposal]:
        """Attach a PASS / FLAG / FAIL check to every proposed change. Changes stay PENDING: only the RM approves them."""
        return [p.model_copy(update={"check": profile_change_limits_check(inputs.raw_input, p.field, p.after)}) for p in inputs.proposals]

    def _assess_ideas(self, inputs: SuitabilityCriticInputs) -> SuitabilityAssessment:
        profile = inputs.profile
        ideas = inputs.candidate_ideas.ideas
        approved = {p.ticker.upper() for p in approved_product_universe_tool()}

        engine = portfolio_risk_engine(profile, ideas, inputs.research_brief, approved, offer_map=inputs.offer_map)
        portfolio_results: List[RuleResult] = engine["portfolio_results"]

        offer_map = inputs.offer_map
        idea_evaluations: List[IdeaStatus] = []
        for idea in ideas:
            results = engine["idea_results"][idea.idea_id]
            fit_check = {
                label: _worst([r.outcome for r in results if r.rule_id == rule]
                              + [r.outcome for r in portfolio_results if r.rule_id == rule and r.outcome != "FAIL"])
                for label, rule in FIT_RULES.items()
            }
            e1 = next((r for r in results if r.rule_id == "E1"), None)
            offered = next((p for p in offer_map.eligible_products if p.ticker.upper() == idea.ticker.upper()), None) if offer_map else None
            eligibility = {} if offer_map is None else {
                "Client limits": {"PASS": "ELIGIBLE", "FLAG": "ESCALATE", "FAIL": "BLOCKED"}[e1.outcome] if e1 else "UNKNOWN",
                "Product risk disclosure": "REQUIRED" if offered and offered.required_disclosures else "NONE",
                "Open gaps": "NONE" if not offer_map.gaps else f"{len(offer_map.gaps)} OPEN",
            }
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
                or "Passes the universe, evidence, wording and client-limits checks; no portfolio breach attributed.",
                evidence_verified=all(r.outcome == "PASS" for r in results if r.rule_id == "R8"),
                rule_results=results,
                fit_check=fit_check,
                eligibility=eligibility,
                required_disclosures=offered.required_disclosures if offered else []
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
        breaches += [b for ie in idea_evaluations for b in ie.detected_breaches if b.rule_id in ("R2", "R8", "R9", "E1", "M1")]

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


SuitabilityCriticAgent = FitEligibilityCriticAgent  # earlier name, kept for imports
