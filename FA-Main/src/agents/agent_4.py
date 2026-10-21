"""Agent 4: Risk & Compliance Validator - Identifies regulatory and risk requirements."""

from typing import Dict, Any

from src.agents.base_agent import BaseAgent
from src.config.prompt_templates import AGENT_4_PROMPT_TEMPLATE

class RiskComplianceValidator(BaseAgent):
    """Identifies regulatory requirements and risk management controls."""
    
    def __init__(self, api_key: str):
        super().__init__("RiskComplianceValidator", api_key)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strategy for regulatory compliance and risk management requirements."""
        
        # Get strategy context from previous agents
        agent_1_output = data.get('agent_1_output', {})
        agent_3_output = data.get('agent_3_output', {})
        
        selected_idea = None
        for idea in agent_1_output.get('investment_ideas', []):
            if idea.get('idea_id') == agent_1_output.get('selected_idea_id'):
                selected_idea = idea
                break
        
        strategy_rules = agent_3_output.get('strategy_rules', {})
        
        # Build prompt using template
        prompt = AGENT_4_PROMPT_TEMPLATE.format(
            strategy_name=selected_idea.get('strategy_name', 'Unknown') if selected_idea else 'Unknown',
            asset_class=selected_idea.get('asset_class', 'Unknown') if selected_idea else 'Unknown',
            time_horizon=selected_idea.get('time_horizon', 'Unknown') if selected_idea else 'Unknown',
            selection_criteria=strategy_rules.get('selection_criteria', {}).get('primary_criteria', 'Not specified'),
            weighting_method=strategy_rules.get('weighting_methodology', {}).get('method', 'Not specified'),
            position_sizing=strategy_rules.get('position_sizing', {}).get('max_position_size', 'Not specified'),
            risk_controls=strategy_rules.get('risk_controls', {}).get('failure_mode_mitigation', 'Not specified')
        )

        # Generate risk compliance analysis using Gemini
        try:
            response_data = self.gemini_client.generate_structured_json(
                prompt,
                {
                    "type": "object",
                    "properties": {
                        "risk_compliance": {
                            "type": "object",
                            "properties": {
                                "regulatory_requirements": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "regulation": {"type": "string"},
                                            "requirement": {"type": "string"},
                                            "compliance_action": {"type": "string"}
                                        },
                                        "additionalProperties": True
                                    }
                                },
                                "position_limits": {
                                    "type": "object",
                                    "properties": {
                                        "max_single_position": {"type": "string"},
                                        "max_sector_concentration": {"type": "string"},
                                        "max_geographic_concentration": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "liquidity_requirements": {
                                    "type": "object",
                                    "properties": {
                                        "minimum_liquidity": {"type": "string"},
                                        "liquidity_monitoring": {"type": "string"},
                                        "stress_testing": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "risk_metrics": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "compliance_procedures": {
                                    "type": "object",
                                    "properties": {
                                        "monitoring_frequency": {"type": "string"},
                                        "reporting_schedule": {"type": "string"},
                                        "escalation_procedures": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "stress_testing": {
                                    "type": "object",
                                    "properties": {
                                        "scenarios": {"type": "array", "items": {"type": "string"}},
                                        "frequency": {"type": "string"},
                                        "thresholds": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "escalation_procedures": {
                                    "type": "object",
                                    "properties": {
                                        "risk_breach_protocol": {"type": "string"},
                                        "regulatory_breach_protocol": {"type": "string"},
                                        "emergency_procedures": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                }
                            },
                            "additionalProperties": True
                        }
                    },
                    "additionalProperties": True
                }
            )
        except Exception as e:
            self.logger.warning(f"JSON generation failed, using fallback: {str(e)}")
            # Fallback to neutral, cross-asset risk compliance
            response_data = {
                "risk_compliance": {
                    "regulatory_requirements": [
                        {
                            "regulation": "Investment Management Oversight",
                            "requirement": "Fiduciary duty, best execution, conflict management",
                            "compliance_action": "Ongoing oversight, periodic attestations, and monitoring"
                        },
                        {
                            "regulation": "Risk Governance",
                            "requirement": "Independent risk controls and reporting",
                            "compliance_action": "Daily risk metrics, monthly reporting to governance body"
                        }
                    ],
                    "position_limits": {
                        "max_single_position": "By asset class/instrument (e.g., equities 5%, futures 10%, FX pair 15%)",
                        "max_sector_concentration": "Sector/asset bucket limits appropriate to strategy",
                        "max_geographic_concentration": "Geographic limits appropriate to mandate"
                    },
                    "liquidity_requirements": {
                        "minimum_liquidity": "Maintain buffer of liquid instruments to meet obligations",
                        "liquidity_monitoring": "Daily liquidity and capacity assessment",
                        "stress_testing": "Periodic liquidity stress tests across asset classes"
                    },
                    "risk_metrics": [
                        "Volatility (ex-ante/ex-post)",
                        "Drawdown",
                        "Value at Risk (VaR)",
                        "Tracking Error (if benchmarked)"
                    ],
                    "compliance_procedures": {
                        "monitoring_frequency": "Daily position/risk monitoring; weekly reviews",
                        "reporting_schedule": "Periodic performance/risk/compliance reporting",
                        "escalation_procedures": "Immediate escalation on limit breaches"
                    },
                    "stress_testing": {
                        "scenarios": [
                            "Broad risk-off shock",
                            "Interest rate shift",
                            "Volatility spike",
                            "Liquidity contraction"
                        ],
                        "frequency": "Monthly with quarterly comprehensive review",
                        "thresholds": "Thresholds defined per mandate and asset class"
                    },
                    "escalation_procedures": {
                        "risk_breach_protocol": "Reduce exposures, notify risk committee",
                        "regulatory_breach_protocol": "Notify compliance and authorities as required",
                        "emergency_procedures": "Activate contingency plan in extreme events"
                    }
                }
            }
        
        self.logger.info("Generated risk and compliance analysis")
        
        # Return data with agent 4 output added
        data['agent_4_output'] = {
            'risk_compliance': response_data.get('risk_compliance', {}),
            'additional_fields': {}
        }
        
        # Add any additional fields from the response
        for key, value in response_data.items():
            if key not in ['risk_compliance']:
                data['agent_4_output']['additional_fields'][key] = value
        
        return data
