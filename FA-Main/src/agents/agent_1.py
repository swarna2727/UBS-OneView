"""Agent 1: Investment Idea Generator - Creates 3 original investment strategies."""

import uuid
from typing import Dict, Any

from src.agents.base_agent import BaseAgent
from src.config.prompt_templates import AGENT_1_PROMPT_TEMPLATE, MARKET_CONTEXT_PROMPT_TEMPLATE

class InvestmentIdeaGenerator(BaseAgent):
    """Generates 3 original investment strategy ideas."""
    
    def __init__(self, api_key: str):
        super().__init__("InvestmentIdeaGenerator", api_key)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3 investment strategy ideas and update market context dynamically."""
        
        # Extract input data (contextless mode tolerated)
        input_data = data.get('input', {})
        market_conditions = input_data.get('market_conditions', '')
        investment_themes = input_data.get('investment_themes', [])
        constraints = input_data.get('constraints', '')
        
        # Build prompt using template
        prompt = AGENT_1_PROMPT_TEMPLATE

        # Generate ideas using Gemini with flexible schema
        try:
            response_data = self.gemini_client.generate_structured_json(
                prompt,
                {
                    "type": "object",
                    "properties": {
                        "investment_ideas": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "idea_id": {"type": "string"},
                                    "strategy_name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "signal_hypothesis": {"type": "string"},
                                    "originality_rationale": {"type": "string"},
                                    "time_horizon": {"type": "string"},
                                    "asset_class": {"type": "string"},
                                    "market_inefficiency": {"type": "string"}
                                },
                                "required": ["idea_id", "strategy_name", "description", "signal_hypothesis", "originality_rationale", "time_horizon", "asset_class"],
                                "additionalProperties": True
                            },
                            "minItems": 3,
                            "maxItems": 3
                        }
                    },
                    "required": ["investment_ideas"],
                    "additionalProperties": True
                }
            )
            
            # Apply diversification logic to reduce AI bias
            response_data = self._diversify_investment_ideas(response_data, data)
            
        except Exception as e:
            self.logger.warning(f"JSON generation failed, using fallback: {str(e)}")
            # Fallback to diversified, cross-asset investment ideas
            transaction_id = data.get('metadata', {}).get('transaction_id', str(uuid.uuid4()))
            response_data = {
                "investment_ideas": [
                    {
                        "idea_id": f"idea_{transaction_id[:8]}_1",
                        "strategy_name": "Rates Curve Carry",
                        "description": "Harvest term-structure carry by overweighting points on the sovereign yield curve with favorable roll-down, subject to duration and risk limits.",
                        "signal_hypothesis": "Positive expected roll-down and stable macro conditions lead to excess returns on targeted maturity buckets versus benchmark.",
                        "originality_rationale": "Systematic curve positioning with explicit carry/roll targeting rather than discretionary duration bets.",
                        "time_horizon": "6-18 months",
                        "asset_class": "Fixed Income",
                        "market_inefficiency": "Persistent term-structure premia and slow-moving investor rebalancing"
                    },
                    {
                        "idea_id": f"idea_{transaction_id[:8]}_2",
                        "strategy_name": "G10 FX Carry-Momentum",
                        "description": "Allocate to currency pairs exhibiting carry and recent trend persistence, hedging tail risk via volatility filters and position caps.",
                        "signal_hypothesis": "Currencies with positive interest differentials and supportive momentum outperform on a risk-adjusted basis.",
                        "originality_rationale": "Combines carry and momentum with explicit risk gating rather than naive carry.",
                        "time_horizon": "6-12 months",
                        "asset_class": "FX",
                        "market_inefficiency": "Slow adjustment to macro differentials and behavioral trend-following effects"
                    },
                    {
                        "idea_id": f"idea_{transaction_id[:8]}_3",
                        "strategy_name": "Commodity Trend Allocation",
                        "description": "Systematic long/short allocation across major commodity sectors based on medium-term trend signals with position and sector caps.",
                        "signal_hypothesis": "Medium-term price trends in commodities persist due to supply/demand adjustment lags.",
                        "originality_rationale": "Cross-commodity approach with sector diversification and explicit risk limits.",
                        "time_horizon": "6-18 months",
                        "asset_class": "Commodities",
                        "market_inefficiency": "Behavioral and structural delays in supply responses create persistent trends"
                    }
                ]
            }
        
        # Validate we have exactly 3 ideas
        ideas = response_data.get('investment_ideas', [])
        if len(ideas) != 3:
            raise ValueError(f"Agent 1 must generate exactly 3 ideas, but got {len(ideas)}")
        
        # Add unique IDs to each idea
        transaction_id = data.get('metadata', {}).get('transaction_id', str(uuid.uuid4()))
        for i, idea in enumerate(ideas):
            if 'idea_id' not in idea or not idea['idea_id']:
                idea['idea_id'] = f"idea_{transaction_id[:8]}_{i+1}"
        
        self.logger.info(f"Generated {len(ideas)} investment ideas")
        
        # Update input data with any additional parameters from the response
        if 'additional_parameters' in response_data:
            data['input']['additional_parameters'].update(response_data['additional_parameters'])
        
        # Return data with agent 1 output added
        data['agent_1_output'] = {
            'investment_ideas': ideas,
            'additional_fields': {}
        }
        
        # Allow additional properties in agent output
        for key, value in response_data.items():
            if key not in ['investment_ideas']:
                data['agent_1_output']['additional_fields'][key] = value
        
        return data
    
    # Context generation removed for contextless mode
    
    def _diversify_investment_ideas(self, response_data: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply diversification logic to reduce AI bias towards technology/AI solutions."""
        ideas = response_data.get('investment_ideas', [])
        
        if len(ideas) < 3:
            return response_data
        
        # Simple check: if all 3 ideas contain tech keywords, replace the last one
        tech_keywords = ['ai', 'artificial intelligence', 'technology', 'tech', 'software', 'digital']
        tech_count = sum(1 for idea in ideas if any(keyword in idea.get('strategy_name', '').lower() for keyword in tech_keywords))
        
        if tech_count >= 2:
            self.logger.info(f"Detected AI bias ({tech_count}/3 ideas), applying diversification")
            
            # Simple non-tech strategy
            ideas[-1].update({
                "strategy_name": "Value Recovery Strategy",
                "description": "Target undervalued companies in traditional sectors with strong fundamentals and recovery potential.",
                "signal_hypothesis": "Companies with strong balance sheets and temporary challenges will outperform as conditions normalize.",
                "originality_rationale": "Focus on fundamental value rather than growth narratives.",
                "time_horizon": "12-24 months",
                "asset_class": "Public Equities (Value)",
                "market_inefficiency": "Market overreaction to short-term challenges in traditional sectors"
            })
            ideas[-1]['idea_id'] = f"idea_{data.get('metadata', {}).get('transaction_id', str(uuid.uuid4()))[:8]}_3"
        
        response_data['investment_ideas'] = ideas
        return response_data
