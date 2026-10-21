"""Agent 3: Strategy Definition Engineer - Converts signals into executable trading rules."""

from typing import Dict, Any

from src.agents.base_agent import BaseAgent
from src.config.prompt_templates import AGENT_3_PROMPT_TEMPLATE

class StrategyDefinitionEngineer(BaseAgent):
    """Converts signals into executable trading rules with concise heuristics."""
    
    def __init__(self, api_key: str):
        super().__init__("StrategyDefinitionEngineer", api_key)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert signals into executable trading rules."""
        
        # Get strategy context from agent 1
        agent_1_output = data.get('agent_1_output', {})
        selected_idea = None
        for idea in agent_1_output.get('investment_ideas', []):
            if idea.get('idea_id') == agent_1_output.get('selected_idea_id'):
                selected_idea = idea
                break
        
        # Get signals context from agent 2
        agent_2_output = data.get('agent_2_output', {})
        signals = agent_2_output.get('signals', [])
        failure_modes = agent_2_output.get('failure_modes', [])
        
        # Build prompt using template
        prompt = AGENT_3_PROMPT_TEMPLATE.format(
            strategy_name=selected_idea.get('strategy_name', 'Unknown') if selected_idea else 'Unknown',
            description=selected_idea.get('description', 'No description') if selected_idea else 'No description',
            asset_class=selected_idea.get('asset_class', 'Unknown') if selected_idea else 'Unknown',
            time_horizon=selected_idea.get('time_horizon', 'Unknown') if selected_idea else 'Unknown',
            signals_context=self._format_signals_context(signals),
            failure_modes_context=self._format_failure_modes_context(failure_modes)
        )

        # Generate strategy rules using Gemini
        try:
            response_data = self.gemini_client.generate_structured_json(
                prompt,
                {
                    "type": "object",
                    "properties": {
                        "strategy_rules": {
                            "type": "object",
                            "properties": {
                                "selection_criteria": {
                                    "type": "object",
                                    "properties": {
                                        "primary_criteria": {"type": "string"},
                                        "secondary_criteria": {"type": "string"},
                                        "exclusion_criteria": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "weighting_methodology": {
                                    "type": "object",
                                    "properties": {
                                        "method": {"type": "string"},
                                        "parameters": {"type": "string"},
                                        "rebalancing_frequency": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "position_sizing": {
                                    "type": "object",
                                    "properties": {
                                        "max_position_size": {"type": "string"},
                                        "min_position_size": {"type": "string"},
                                        "sizing_method": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "execution_guidelines": {
                                    "type": "object",
                                    "properties": {
                                        "entry_conditions": {"type": "string"},
                                        "exit_conditions": {"type": "string"},
                                        "risk_management": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "rebalancing_framework": {
                                    "type": "object",
                                    "properties": {
                                        "frequency": {"type": "string"},
                                        "triggers": {"type": "string"},
                                        "methodology": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "risk_controls": {
                                    "type": "object",
                                    "properties": {
                                        "failure_mode_mitigation": {"type": "string"},
                                        "position_limits": {"type": "string"},
                                        "monitoring_requirements": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "monitoring_framework": {
                                    "type": "object",
                                    "properties": {
                                        "performance_metrics": {"type": "string"},
                                        "reporting_frequency": {"type": "string"},
                                        "validation_approach": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "heuristic_rules": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 3,
                                    "maxItems": 12
                                }
                            },
                            "additionalProperties": True
                        }
                    },
                    "required": ["strategy_rules"],
                    "additionalProperties": True
                }
            )
        except Exception as e:
            self.logger.warning(f"JSON generation failed, using fallback: {str(e)}")
            # Fallback to basic strategy rules
            response_data = {
                "strategy_rules": {
                    "selection_criteria": {
                        "primary_criteria": "Based on fundamental analysis, technical indicators, and market signals",
                        "secondary_criteria": "Additional screening criteria including liquidity and volatility filters",
                        "exclusion_criteria": "Exclude companies with poor governance, high debt levels, or regulatory issues"
                    },
                    "weighting_methodology": {
                        "method": "Equal or risk-targeted weighting depending on asset class",
                        "parameters": "Monthly rebalancing with 5% deviation threshold",
                        "rebalancing_frequency": "Monthly"
                    },
                    "position_sizing": {
                        "max_position_size": "Per-asset-class limits (e.g., single-name equity 5%, single future 10%, FX pair 15%)",
                        "min_position_size": "Operationally feasible minimum size",
                        "sizing_method": "Risk-adjusted sizing based on volatility and correlation"
                    },
                    "execution_guidelines": {
                        "entry_conditions": "Enter positions when signals align and risk metrics are favorable",
                        "exit_conditions": "Exit on stop-loss triggers, take-profit levels, or signal reversal",
                        "risk_management": "Continuous monitoring with automated risk controls"
                    },
                    "rebalancing_framework": {
                        "frequency": "Monthly rebalancing",
                        "triggers": "5% deviation from target weights or significant market regime change",
                        "methodology": "Systematic rebalancing to target weights with transaction cost consideration"
                    },
                    "risk_controls": {
                        "failure_mode_mitigation": "Diversification across instruments and sectors; regular stress testing",
                        "position_limits": "Cross-asset position/concentration limits varying by instrument type",
                        "monitoring_requirements": "Daily position monitoring, weekly risk reporting, monthly strategy review"
                    },
                    "monitoring_framework": {
                        "performance_metrics": "Sharpe ratio, maximum drawdown, alpha, beta, tracking error",
                        "reporting_frequency": "Daily performance updates, weekly risk reports, monthly strategy reviews",
                        "validation_approach": "Out-of-sample testing, walk-forward analysis, scenario testing"
                    },
                    "heuristic_rules": [
                        "Select instruments with strongest composite signal scores above threshold.",
                        "Exclude instruments breaching liquidity and governance/eligibility filters.",
                        "Target equal or risk-weight base; tilt weights by normalized signal strength.",
                        "Rebalance monthly or when weight drift exceeds threshold.",
                        "Apply cross-asset position/concentration caps by instrument class.",
                        "Exit on signal reversal or stop-loss at predefined drawdown.",
                        "Halt adds during risk breach; resume after remediation.",
                        "Review failure-mode indicators weekly and adjust thresholds if triggered."
                    ]
                }
            }
        
        self.logger.info("Generated strategy rules and heuristic rules")

        # Agent 3 selects the final idea (reselection always allowed)
        try:
            ideas = agent_1_output.get('investment_ideas', [])
            by_idea = data.get('agent_2_output', {}).get('by_idea', {})
            selected_id = None
            best_score = float('-inf')
            # Score each idea based on available signals and simple diversification heuristic
            for idea in ideas:
                idea_id = idea.get('idea_id')
                if not idea_id:
                    continue
                a2_bundle = by_idea.get(idea_id, {})
                signals = a2_bundle.get('signals', [])
                failure_modes = a2_bundle.get('failure_modes', [])
                # Basic score: signal count minus failure modes penalty (no thematic bonus/penalty)
                score = len(signals) - 0.5 * len(failure_modes)
                if score > best_score:
                    best_score = score
                    selected_id = idea_id

            if selected_id:
                data.setdefault('agent_3_output', {})['selected_idea_id'] = selected_id
                # Mirror into Agent 1 output for downstream consumers
                data.setdefault('agent_1_output', {})['selected_idea_id'] = selected_id
                self.logger.info(f"Agent 3 selected idea: {selected_id}")
        except Exception as sel_err:
            self.logger.warning(f"Agent 3 selection step skipped due to error: {str(sel_err)}")
        
        # Ensure we have the basic structure even if AI doesn't generate all fields
        strategy_rules = response_data.get('strategy_rules', {})
        
        # Provide default values for missing required fields
        if 'selection_criteria' not in strategy_rules:
            strategy_rules['selection_criteria'] = {
                'primary_criteria': 'To be defined based on market analysis',
                'secondary_criteria': 'Additional screening criteria',
                'exclusion_criteria': 'Standard exclusions apply'
            }
        
        if 'weighting_methodology' not in strategy_rules:
            strategy_rules['weighting_methodology'] = {
                'method': 'Equal weighting with rebalancing',
                'parameters': 'Monthly rebalancing',
                'rebalancing_frequency': 'Monthly'
            }
        
        if 'position_sizing' not in strategy_rules:
            strategy_rules['position_sizing'] = {
                'max_position_size': '5% per position',
                'min_position_size': '0.5% per position',
                'sizing_method': 'Risk-adjusted sizing'
            }
        
        if 'execution_guidelines' not in strategy_rules:
            strategy_rules['execution_guidelines'] = {
                'entry_conditions': 'Based on signal triggers',
                'exit_conditions': 'Stop loss and take profit levels',
                'risk_management': 'Position limits and monitoring'
            }
        
        if 'heuristic_rules' not in strategy_rules:
            strategy_rules['heuristic_rules'] = [
                'Select assets with strongest signals above threshold',
                'Equal-weight base with signal tilts',
                'Monthly rebalance or >5% drift',
                'Position and sector caps enforced'
            ]
        
        # Return data with agent 3 output added
        data['agent_3_output'] = {
            'strategy_rules': strategy_rules,
            'additional_fields': {}
        }
        
        # Add any additional fields from the response
        for key, value in response_data.items():
            if key not in ['strategy_rules']:
                data['agent_3_output']['additional_fields'][key] = value
        
        return data
    
    def _format_signals_context(self, signals: list) -> str:
        """Format signals for the prompt."""
        if not signals:
            return "No signals provided"
        
        context = "Signals:\n"
        for i, signal in enumerate(signals, 1):
            context += f"{i}. {signal.get('signal_name', 'Unknown')}: {signal.get('calculation_method', 'No method')}\n"
        
        return context
    
    def _format_failure_modes_context(self, failure_modes: list) -> str:
        """Format failure modes for the prompt."""
        if not failure_modes:
            return "No failure modes provided"
        
        context = "Failure Modes:\n"
        for i, failure in enumerate(failure_modes, 1):
            context += f"{i}. {failure.get('failure_name', 'Unknown')}: {failure.get('trigger_conditions', 'No conditions')}\n"
        
        return context
