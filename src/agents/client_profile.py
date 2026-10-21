"""Agent 1: Client Profile Agent (Slide 14)."""

from typing import Dict, Any, List, Optional
from src.agents.base import BaseAdvisoryAgent
from src.schemas.advisory import (
    ClientAdvisoryProfile, ProfileFlag, ClientConstraints, LiquidityEvent
)
from src.tools.profile_tools import (
    required_field_validator, risk_profile_calculator, portfolio_parser
)
from src.llm.base import LLMClientProtocol

class ClientProfileAgent(BaseAdvisoryAgent[Dict[str, Any], ClientAdvisoryProfile]):
    """
    Ingests, standardizes, reconciles, and validates advisor-entered client inputs.
    Strictly forbids inferring high-stakes data or recommending investments.
    """

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None):
        super().__init__("ClientProfileAgent", llm_client)

    def execute(self, inputs: Dict[str, Any]) -> ClientAdvisoryProfile:
        # 1. Deterministic mandatory field validation
        val_result = required_field_validator(inputs)

        flags: List[ProfileFlag] = []
        if not val_result.is_valid:
            for missing_field in val_result.missing_fields:
                flags.append(ProfileFlag(
                    code="MISSING_MANDATORY_FIELD",
                    severity="BLOCKING_INTERRUPT",
                    message=f"Mandatory client profile field is missing: '{missing_field}'.",
                    affected_fields=[missing_field]
                ))
            for inv_field, inv_msg in val_result.invalid_fields.items():
                flags.append(ProfileFlag(
                    code="MISSING_MANDATORY_FIELD",
                    severity="BLOCKING_INTERRUPT",
                    message=f"Field '{inv_field}' has invalid value: {inv_msg}",
                    affected_fields=[inv_field]
                ))

        # 2. Deterministic risk calculation & capacity vs willingness check
        calc_risk = risk_profile_calculator(inputs)
        if calc_risk.has_capacity_conflict:
            flags.append(ProfileFlag(
                code="RISK_CAPACITY_CONFLICT",
                severity="BLOCKING_INTERRUPT",
                message=calc_risk.conflict_rationale or "Risk willingness conflicts with short-term liquidity capacity.",
                affected_fields=["stated_risk_willingness", "time_horizon_months", "liquidity_schedule"]
            ))

        # 3. Deterministic portfolio holdings parsing
        raw_holdings = inputs.get("raw_holdings", [])
        parsed_holdings_data = portfolio_parser(raw_holdings)
        if not parsed_holdings_data.is_balanced_sum:
            flags.append(ProfileFlag(
                code="CONCENTRATION_WARNING",
                severity="BLOCKING_INTERRUPT",
                message=f"Holdings total weight is {parsed_holdings_data.total_weight_pct}%, which does not equal 100%.",
                affected_fields=["raw_holdings"]
            ))

        # Build Liquidity Schedule
        liquidity_schedule = []
        for item in inputs.get("liquidity_schedule", []):
            if isinstance(item, dict):
                liquidity_schedule.append(LiquidityEvent(
                    due_months=int(item.get("due_months", 12)),
                    amount=float(item.get("amount", 0.0)),
                    purpose=str(item.get("purpose", "General Liquidity"))
                ))

        # Constraints
        raw_constraints = inputs.get("constraints", {})
        constraints = ClientConstraints(
            excluded_sectors=raw_constraints.get("excluded_sectors", []),
            excluded_countries=raw_constraints.get("excluded_countries", []),
            esg_mandate=raw_constraints.get("esg_mandate", False),
            max_single_stock_pct=float(raw_constraints.get("max_single_stock_pct", 20.0))
        )

        # Risk range definition
        tier_num = calc_risk.risk_tier_number
        tier_ranges = {
            1: [1.0, 1.8],
            2: [1.8, 2.6],
            3: [2.6, 3.4],
            4: [3.4, 4.2],
            5: [4.2, 5.0]
        }

        # Guardrail: Never infer missing high-stakes information
        goals = inputs.get("goals", [])
        if isinstance(goals, str):
            goals = [g.strip() for g in goals.split(",") if g.strip()]

        return ClientAdvisoryProfile(
            client_id=str(inputs.get("client_id", "CL-UNASSIGNED")),
            name=str(inputs.get("name", "Unnamed Client")),
            goals=goals or ["Capital Preservation"],
            time_horizon_months=int(inputs.get("time_horizon_months", 36) or 36),
            stated_risk_willingness=str(inputs.get("stated_risk_willingness", "Balanced")),
            calculated_risk_tier=calc_risk.risk_tier,
            risk_tier_number=tier_num,
            risk_score_range=tier_ranges.get(tier_num, [2.5, 3.5]),
            liquidity_schedule=liquidity_schedule,
            total_wealth=float(inputs.get("total_wealth", 1000000.0) or 1000000.0),
            experience_level=str(inputs.get("experience_level", "Knowledgeable")),
            constraints=constraints,
            parsed_holdings=parsed_holdings_data.holdings,
            current_asset_class_weights=parsed_holdings_data.asset_class_weights,
            current_region_weights=parsed_holdings_data.region_weights,
            current_currency_weights=parsed_holdings_data.currency_weights,
            unresolved_flags=flags
        )
