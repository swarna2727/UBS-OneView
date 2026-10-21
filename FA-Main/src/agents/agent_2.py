"""Agent 2: Signal Modelling Specialist - Develops testable signals and identifies failure modes."""

from typing import Dict, Any

from src.agents.base_agent import BaseAgent
from src.config.prompt_templates import AGENT_2_PROMPT_TEMPLATE

class SignalModellingSpecialist(BaseAgent):
    """Develops testable signals and identifies failure scenarios for investment strategies."""
    
    def __init__(self, api_key: str):
        super().__init__("SignalModellingSpecialist", api_key)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop signals and failure modes for the selected strategy."""
        
        # Get selected strategy from agent 1 output
        agent_1_output = data.get('agent_1_output', {})
        selected_idea_id = agent_1_output.get('selected_idea_id')
        investment_ideas = agent_1_output.get('investment_ideas', [])
        
        # Find the selected idea
        selected_idea = None
        for idea in investment_ideas:
            if idea.get('idea_id') == selected_idea_id:
                selected_idea = idea
                break
        
        if not selected_idea:
            raise ValueError(f"Selected idea {selected_idea_id} not found in investment ideas")
        
        # Build prompt using template
        prompt = AGENT_2_PROMPT_TEMPLATE.format(
            strategy_name=selected_idea.get('strategy_name', 'Unknown'),
            description=selected_idea.get('description', 'No description'),
            signal_hypothesis=selected_idea.get('signal_hypothesis', 'No hypothesis'),
            market_inefficiency=selected_idea.get('market_inefficiency', 'Not specified'),
            asset_class=selected_idea.get('asset_class', 'Unknown'),
            time_horizon=selected_idea.get('time_horizon', 'Unknown')
        )

        # Generate signals using Gemini with flexible schema
        try:
            response_data = self.gemini_client.generate_structured_json(
                prompt,
                {
                    "type": "object",
                    "properties": {
                        "signals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {                                     
                                    "price_condition": {"type": "string"},
                                    "volume_condition": {"type": "string"},
                                    "long_entry": {"type": "string"},
                                    "short_entry": {"type": "string"},
                                    "stop_loss": {"type": "string"},
                                    "take_profit": {"type": "string"},
                                    "holding_period": {"type": "string"},
                                    "trend_filters": {"type": "string"},
                                    "exit": {"type": "string"}
                                },
                                "additionalProperties": True
                            }
                        },
                        "theoretical_validations": {
                            "type": "object",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "behavioral_finance": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "supporting_evidence": {"type": "array"},
                                            "assumptions": {"type": "array"},
                                            "potential_issues": {"type": "array"}
                                        },
                                        "additionalProperties": True
                                    },
                                    "empirical_evidence": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "historical_performance": {"type": "array"},
                                            "data_limitations": {"type": "array"},
                                            "potential_issues": {"type": "array"}
                                        },
                                        "additionalProperties": True
                                    },
                                    "analytical_modeling": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "testable_components": {"type": "array"},
                                            "simplifying_assumptions": {"type": "array"},
                                            "limitations": {"type": "array"}
                                        },
                                        "additionalProperties": True
                                    },
                                    "edge_cases": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "examples": {"type": "array"},
                                            "potential_impact": {"type": "array"},
                                            "mitigation": {"type": "array"}
                                        },
                                        "additionalProperties": True
                                    }
                                }
                            },
                            
                        },
                        "plausibility_risk_assessment": {
                            "type": "object",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "pros": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "points": {"type": "array"},
                                        },
                                        "additionalProperties": True
                                    },
                                    "cons": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "points": {"type": "array"},
                                        },
                                        "additionalProperties": True
                                    },
                                    "flagged_issues": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "points": {"type": "array"},
                                        },
                                        "additionalProperties": True
                                    }
                                },
                                "additionalProperties": True
                            }
                        },
                        "final_outcome": {
                            "type": "object",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "decision": {"type": "string"},
                                    "ranking": {"type": "string"}, 
                                    "evidence": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "points": {"type": "array"},
                                        },
                                        "additionalProperties": True
                                    }
                                },
                                "additionalProperties": True
                            }
                        },
                        "failure_modes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "failure_name": {"type": "string"},
                                    "trigger_conditions": {"type": "string"},
                                    "potential_impact": {"type": "string"},
                                    "severity_level": {"type": "string"},
                                    "mitigation_strategy": {"type": "string"},
                                    "early_warning_indicators": {"type": "string"}
                                },
                                "additionalProperties": True
                            },
                            "minItems": 3,
                            "maxItems": 3
                        }
                    },
                    "required": ["signals", "failure_modes"],
                    "additionalProperties": True
                }
            )
        except Exception as e:
            self.logger.warning(f"JSON generation failed, using fallback: {str(e)}")
            # Fallback to basic structure
            response_data = {
                "signals": [
                    {
                        "price_condition": "Stock with earnings surprise > +5% for long, < -5% for short",
                        "volume_condition": "Average daily trading volume > 500,000 shares",
                        "long_entry": "Buy at next market open after positive earnings surprise",
                        "short_entry": "Short at next market open after negative earnings surprise",
                        "stop_loss": "5% below entry price",
                        "take_profit": "15% above entry price or at end of holding period",
                        "holding_period": "2–4 weeks post-earnings announcement",
                        "trend_filters": "Exclude stocks in strong long-term downtrend for long trades",
                        "exit": "Exit when stop-loss or take-profit is hit, or holding period ends"
                    }
                ],
                "failure_modes": [
                    {
                        "failure_name": "Liquidity Crunch",
                        "trigger_conditions": "Daily trading volume drops below 200,000 shares or bid-ask spreads widen significantly",
                        "potential_impact": "Large slippage, inability to exit positions, or distorted backtest results",
                        "severity_level": "High",
                        "mitigation_strategy": "Apply strict liquidity filters and cap position size",
                        "early_warning_indicators": "Declining average daily volume, widening spreads, reduced order book depth"
                    },
                    {
                        "failure_name": "Earnings Restatement Risk",
                        "trigger_conditions": "Company issues revised or corrected earnings results after initial release",
                        "potential_impact": "Signal invalidation leading to unexpected price reversals and losses",
                        "severity_level": "Medium",
                        "mitigation_strategy": "Exclude companies with frequent restatements, monitor regulatory filings closely",
                        "early_warning_indicators": "High auditor turnover, unusual delays in filing reports, frequent earnings adjustments"
                    },
                    {
                        "failure_name": "Macro Shock Override",
                        "trigger_conditions": "Broad market shock (e.g., interest rate surprise, geopolitical crisis) occurs within holding period",
                        "potential_impact": "Earnings drift effect overwhelmed by market-wide selloff, strategy underperforms",
                        "severity_level": "High",
                        "mitigation_strategy": "Introduce market regime filters, reduce exposure during high volatility",
                        "early_warning_indicators": "Sharp rise in VIX, bond yield spikes, global equity correlations increasing"
                    }
                ]
            }
        
        self.logger.info(f"Generated {len(response_data.get('signals', []))} signals and {len(response_data.get('failure_modes', []))} failure modes")
        
        # Return data with agent 2 output added
        data['agent_2_output'] = {
            'signals': response_data.get('signals', []),
            'failure_modes': response_data.get('failure_modes', []),
            'additional_fields': {}
        }
        
        # Add any additional fields from the response
        for key, value in response_data.items():
            if key not in ['signals', 'failure_modes']:
                data['agent_2_output']['additional_fields'][key] = value
        
        return data
