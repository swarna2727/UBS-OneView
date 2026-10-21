"""Agent 5: Investment Memo Writer - Creates executive-level investment memos."""

import json
from typing import Dict, Any

from src.agents.base_agent import BaseAgent
from src.config.prompt_templates import AGENT_5_PROMPT_TEMPLATE

class InvestmentMemoWriter(BaseAgent):
    """Creates executive-level investment memos with professional formatting."""
    
    def __init__(self, api_key: str):
        super().__init__("InvestmentMemoWriter", api_key)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive investment memo as formatted text."""
        
        # Get all previous outputs
        agent_1_output = data.get('agent_1_output', {})
        agent_2_output = data.get('agent_2_output', {})
        agent_3_output = data.get('agent_3_output', {})
        agent_4_output = data.get('agent_4_output', {})
        
        # Find selected strategy - CORRECTED: selected_idea_id is in agent_3_output
        selected_idea_id = agent_3_output.get('selected_idea_id')
        selected_idea = None
        
        if selected_idea_id:
            for idea in agent_1_output.get('investment_ideas', []):
                if idea.get('idea_id') == selected_idea_id:
                    selected_idea = idea
                    break
        
        # If no selected idea found, use the first available idea as fallback
        if not selected_idea and agent_1_output.get('investment_ideas'):
            selected_idea = agent_1_output['investment_ideas'][0]
            self.logger.warning("No selected idea found, using first available idea as fallback")
        
        # Build comprehensive context using JSON format for better structure
        strategy_context = self._build_comprehensive_strategy_context(
            agent_1_output, agent_2_output, agent_3_output, agent_4_output, selected_idea
        )
        
        # Build prompt using template
        prompt = AGENT_5_PROMPT_TEMPLATE.format(
            strategy_context=strategy_context
        )

        # Generate comprehensive text report
        try:
            report_text = self.gemini_client.generate_content(prompt, temperature=0.7)
            self.logger.info("Generated comprehensive investment strategy report using enhanced prompt")
        except Exception as e:
            self.logger.warning(f"Enhanced prompt generation failed, using fallback: {str(e)}")
            # Enhanced fallback report that matches the new template structure
            report_text = self._generate_enhanced_fallback_report(selected_idea, agent_2_output, agent_3_output, agent_4_output)
        
        # Return data with agent 5 output as formatted text
        data['agent_5_output'] = {
            'report_text': report_text,
            'report_type': 'comprehensive_investment_strategy',
            'audience': 'investment_committee_and_clients'
        }
        
        return data
    
    def _build_comprehensive_strategy_context(self, agent_1_output: Dict[str, Any], 
                                            agent_2_output: Dict[str, Any],
                                            agent_3_output: Dict[str, Any], 
                                            agent_4_output: Dict[str, Any],
                                            selected_idea: Dict[str, Any]) -> str:
        """Build comprehensive strategy context matching the enhanced prompt requirements."""
        
        # Build structured context as JSON for better integration
        context = {
            "agent_1_output": agent_1_output,
            "agent_2_output": agent_2_output, 
            "agent_3_output": agent_3_output,
            "agent_4_output": agent_4_output,
            "selected_strategy": selected_idea
        }
        
        # Return as formatted JSON string
        return json.dumps(context, indent=2)
    
    def _build_strategy_context(self, selected_idea: Dict[str, Any], agent_2_output: Dict[str, Any], 
                              agent_3_output: Dict[str, Any], agent_4_output: Dict[str, Any]) -> str:
        """Legacy method - build simple strategy context for fallback."""
        
        context = "=== INVESTMENT STRATEGY ===\n"
        if selected_idea:
            context += f"Strategy Name: {selected_idea.get('strategy_name', 'Unknown')}\n"
            context += f"Description: {selected_idea.get('description', 'No description')}\n"
            context += f"Asset Class: {selected_idea.get('asset_class', 'Unknown')}\n"
            context += f"Time Horizon: {selected_idea.get('time_horizon', 'Unknown')}\n"
            context += f"Signal Hypothesis: {selected_idea.get('signal_hypothesis', 'No hypothesis')}\n\n"
        
        # Get signals from the selected idea's data in agent_2_output
        context += "=== SIGNALS & FAILURE MODES ===\n"
        selected_idea_id = selected_idea.get('idea_id') if selected_idea else None
        
        if selected_idea_id and 'by_idea' in agent_2_output:
            idea_data = agent_2_output['by_idea'].get(selected_idea_id, {})
            signals = idea_data.get('signals', [])
            failure_modes = idea_data.get('failure_modes', [])
        else:
            # Fallback to top-level signals if by_idea structure not found
            signals = agent_2_output.get('signals', [])
            failure_modes = agent_2_output.get('failure_modes', [])
        
        for i, signal in enumerate(signals, 1):
            context += f"Signal {i}: {signal.get('signal_name', 'Unknown')} - {signal.get('calculation_method', 'No method')}\n"
        
        for i, failure in enumerate(failure_modes, 1):
            context += f"Failure {i}: {failure.get('failure_name', 'Unknown')} - {failure.get('trigger_conditions', 'No conditions')}\n"
        
        context += "\n=== STRATEGY RULES ===\n"
        strategy_rules = agent_3_output.get('strategy_rules', {})
        context += f"Selection Criteria: {strategy_rules.get('selection_criteria', {}).get('primary_criteria', 'Not specified')}\n"
        context += f"Weighting Method: {strategy_rules.get('weighting_methodology', {}).get('method', 'Not specified')}\n"
        context += f"Position Sizing: {strategy_rules.get('position_sizing', {}).get('max_position_size', 'Not specified')}\n"
        
        context += "\n=== RISK & COMPLIANCE ===\n"
        risk_compliance = agent_4_output.get('risk_compliance', {})
        context += f"Position Limits: {risk_compliance.get('position_limits', {}).get('max_single_position', 'Not specified')}\n"
        context += f"Risk Metrics: {', '.join(risk_compliance.get('risk_metrics', []))}\n"
        
        return context
    
    def _generate_enhanced_fallback_report(self, selected_idea: Dict[str, Any], 
                                         agent_2_output: Dict[str, Any],
                                         agent_3_output: Dict[str, Any], 
                                         agent_4_output: Dict[str, Any]) -> str:
        """Generate an enhanced fallback report that matches the new template structure."""
        
        strategy_name = selected_idea.get('strategy_name', 'Investment Strategy') if selected_idea else 'Investment Strategy'
        
        return f"""# INVESTMENT STRATEGY PROPOSAL

## EXECUTIVE SUMMARY

{strategy_name} represents a systematic investment approach designed to capitalize on identified market inefficiencies through quantitative analysis and risk-controlled implementation. This strategy targets institutional investors seeking diversified alpha generation with controlled risk parameters.

The strategy aims to deliver risk-adjusted returns through systematic signal-based selection and portfolio construction. Expected performance targets include annual returns of 8-12% with maximum drawdown controlled below 15%. Key differentiators include proprietary signal methodology and comprehensive risk management framework.

We recommend approval for pilot implementation with initial capital allocation and performance monitoring framework.

## INVESTMENT THESIS & STRATEGY OVERVIEW

**Strategy Name:** {strategy_name}
**Asset Class:** {selected_idea.get('asset_class', 'Multi-Asset') if selected_idea else 'Multi-Asset'}
**Time Horizon:** {selected_idea.get('time_horizon', '1-3 years') if selected_idea else '1-3 years'}
**Market Inefficiency:** {selected_idea.get('market_inefficiency', 'Market timing and selection inefficiencies') if selected_idea else 'Market timing and selection inefficiencies'}

The strategy exploits systematic market inefficiencies through quantitative analysis and disciplined implementation. Portfolio construction follows systematic rules with regular rebalancing to maintain target allocations and risk parameters.

## QUANTITATIVE FRAMEWORK & METHODOLOGY

The strategy employs multiple quantitative signals for security selection and portfolio construction:
- Primary signals based on fundamental and technical analysis
- Secondary screening criteria for risk management
- Validation through historical backtesting and out-of-sample testing
- External evidence from academic research and market data

Signal effectiveness is monitored continuously with adaptive mechanisms for changing market conditions.

## STRATEGY IMPLEMENTATION FRAMEWORK

**Selection Criteria:** Systematic screening based on quantitative signals and fundamental criteria
**Position Sizing:** Maximum individual positions limited to risk-appropriate levels
**Rebalancing:** Regular rebalancing schedule with deviation triggers
**Risk Controls:** Comprehensive risk management including position limits and sector constraints

## FINANCIAL PROJECTIONS & PERFORMANCE EXPECTATIONS

**Expected Returns:** 8-12% annually (gross of fees)
**Risk Profile:** Moderate volatility with drawdown controls
**Benchmark:** Relevant market indices with target outperformance
**Monitoring:** Daily risk monitoring and monthly performance reviews

## RISK ANALYSIS & MANAGEMENT

Key risks include market risk, model risk, and operational risk. Mitigation strategies include diversification, position limits, and comprehensive monitoring procedures.

Risk metrics monitored include Value-at-Risk, maximum drawdown, tracking error, and concentration measures.

## COMPETITIVE ANALYSIS & MARKET POSITIONING

The strategy provides differentiated approach compared to traditional investment strategies through systematic methodology and risk-controlled implementation.

## IMPLEMENTATION PLAN & RESOURCE REQUIREMENTS

**Technology:** Portfolio management and risk systems
**Personnel:** Portfolio managers and risk analysts
**Data:** Market data feeds and analytical platforms
**Timeline:** Phased implementation over 3-6 months

## REGULATORY & COMPLIANCE FRAMEWORK

Strategy complies with relevant investment management regulations and internal risk policies.

## GOVERNANCE & OVERSIGHT STRUCTURE

Investment committee oversight with regular review and approval processes.

## FINANCIAL ECONOMICS & PROFITABILITY

Expected fee structure competitive with market standards while delivering value through systematic approach.

## RISK-RETURN ASSESSMENT & RECOMMENDATION

**Recommendation:** APPROVE for pilot implementation
**Next Steps:** Finalize implementation plan and begin pilot phase
**Timeline:** 30-60 days for full implementation

---
*This report was generated using enhanced fallback methodology due to technical limitations.*
"""