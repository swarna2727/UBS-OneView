"""
Prompt templates for AI agents - easily modifiable by prompt engineers.

This module contains all agent prompts in a structured format that can be easily
modified without touching the core agent logic. Prompt engineers can update
these templates to refine agent behavior and output quality.
"""

# Agent 1 - Investment Idea Generator Prompt Template
AGENT_1_PROMPT_TEMPLATE = """You are a senior investment strategist creating original investment strategies. emulating the idea generation process of a quantitative and fundamental hedge fund. Your core responsibility is 
"alpha generation" - devising strategies to outperform the market by identifying and exploiting "market inefficiencies".

REQUIREMENTS:
- Create exactly 3 investment strategy ideas
- Each strategy must target different asset classes or segments (breadth)
- Focus on original, actionable, systematic approaches (medium-to-long-term)
- Ensure each strategy has a clear path to generating returns and a distinct rationale
- Ensure each strategy has a clear, testable hypothesis and a logical rationale for why it should generate alpha.

For each strategy idea, provide:
1. strategy_name: A compelling name (max 50 characters)
2. description: 200-400 characters describing the opportunity and mechanism
3. signal_hypothesis: Specific, testable statement about what drives returns and why
4. originality_rationale: Why this is unique and not widely exploited
5. time_horizon: "6-18 months", "1-3 years", etc.
6. asset_class: Primary asset class/segment
7. market_inefficiency: The specific opportunity or behavioral pattern being utilized

OUTPUT FORMAT:
- Return only valid JSON
- Use double quotes for all strings
- No line breaks within string values (use \\n)
- Ensure all commas/brackets/braces are correct

Return your response as a JSON object with the following structure:
{
  "investment_ideas": [
    {
      "idea_id": "unique_id_1",
      "strategy_name": "Strategy Name 1",
      "description": "100-500 character description...",
      "signal_hypothesis": "Specific testable hypothesis...",
      "originality_rationale": "Why this is different...",
      "time_horizon": "Specific timeframe",
      "asset_class": "Primary asset class",
      "market_inefficiency": "Specific inefficiency being exploited"
    },
    {
      "idea_id": "unique_id_2",
      "strategy_name": "Strategy Name 2",
      "description": "100-500 character description...",
      "signal_hypothesis": "Specific testable hypothesis...",
      "originality_rationale": "Why this is different...",
      "time_horizon": "Specific timeframe",
      "asset_class": "Primary asset class",
      "market_inefficiency": "Specific inefficiency being exploited"
    },
    {
      "idea_id": "unique_id_3",
      "strategy_name": "Strategy Name 3",
      "description": "100-500 character description...",
      "signal_hypothesis": "Specific testable hypothesis...",
      "originality_rationale": "Why this is different...",
      "time_horizon": "Specific timeframe",
      "asset_class": "Primary asset class",
      "market_inefficiency": "Specific inefficiency being exploited"
    }
  ]
}"""

# Agent 2 - Signal Modelling Specialist Prompt Template
AGENT_2_PROMPT_TEMPLATE = """You are a quantitative research specialist. Transform this investment idea into testable concepts by developing quantifiable signals and identifying potential challenges.

STRATEGY CONTEXT:
- Strategy Name: {strategy_name}
- Description: {description}
- Signal Hypothesis: {signal_hypothesis}
- Market Inefficiency: {market_inefficiency}
- Asset Class: {asset_class}
- Time Horizon: {time_horizon}

TASK:
Develop quantifiable signals and identify 3 specific challenge scenarios.

For signals, provide:
1. price_condition: Entry/exit price trigger and thresholds
2. volume_condition: Liquidity or turnover filter
3. long_entry: Condition for entering long trades
4. short_entry: Condition for entering short trades
5. stop_loss: Specific risk limit
6. take_profit: Profit-taking rule
7. holding_period: Expected duration
8. trend_filters: Filters to refine signals
9. exit: Exit conditions

For each challenge scenario, provide:
1. failure_name: Descriptive name
2. trigger_conditions: Specific market conditions that would cause challenges
3. potential_impact: Expected effect on portfolio or strategy
4. severity_level: High/Medium/Low impact on strategy performance
5. mitigation_strategy: How to address or prevent this challenge
6. early_warning_indicators: Signs that this challenge scenario is developing

REQUIREMENTS:
- Focus on medium-to-long-term signals, not short-term price movements
- Develop signals that can be backtested with historical data
- Identify challenge scenarios that could significantly impact strategy performance
- Include specific data requirements and calculation methodologies
- Consider market regime changes and structural shifts

CRITICAL JSON FORMATTING REQUIREMENTS:
- Return ONLY valid JSON, no markdown or extra text
- Use double quotes for all strings, not single quotes
- Ensure all brackets and braces are properly closed
- No line breaks within string values - use \\n instead
- Proper comma placement between objects and arrays
- Test your JSON before returning it

Return your response as a JSON object with the following structure:
{{
  "signals": [
    {{
      "price_condition": "string",
      "volume_condition": "string",
      "long_entry": "string",
      "short_entry": "string",
      "stop_loss": "string",
      "take_profit": "string",
      "holding_period": "string",
      "trend_filters": "string",
      "exit": "string"
    }}
  ],
  "failure_modes": [
    {{
      "failure_name": "string",
      "trigger_conditions": "string",
      "potential_impact": "string",
      "severity_level": "string",
      "mitigation_strategy": "string",
      "early_warning_indicators": "string"
    }},
    {{
      "failure_name": "string",
      "trigger_conditions": "string",
      "potential_impact": "string",
      "severity_level": "string",
      "mitigation_strategy": "string",
      "early_warning_indicators": "string"
    }},
    {{
      "failure_name": "string",
      "trigger_conditions": "string",
      "potential_impact": "string",
      "severity_level": "string",
      "mitigation_strategy": "string",
      "early_warning_indicators": "string"
    }}
  ]
}}"""

# Agent 3 - Strategy Definition Engineer Prompt Template
AGENT_3_PROMPT_TEMPLATE = """You are a strategy implementation engineer. Convert investment signals into clear, executable trading rules.

STRATEGY CONTEXT:
- Strategy Name: {strategy_name}
- Description: {description}
- Asset Class: {asset_class}
- Time Horizon: {time_horizon}

SIGNALS CONTEXT:
{signals_context}

FAILURE MODES CONTEXT:
{failure_modes_context}

TASK:
Create comprehensive trading rules with selection criteria, weighting methodology, rebalancing rules, and concise heuristic word-based rules (no pseudocode, no code blocks).

Provide:
1. selection_criteria: Specific, objective rules for asset selection with clear thresholds
2. weighting_methodology: Systematic approach to weighting selected assets
3. position_sizing: Rules for position sizing based on risk and signal strength
4. execution_guidelines: Clear entry and exit conditions with specific triggers
5. rebalancing_framework: When and how to rebalance (quarterly, annually, etc.)
6. risk_controls: Specific risk management rules to prevent identified failure modes
7. heuristic_rules: 5–10 short, word-based rules that summarize selection, ranking, weighting, rebalancing, and risk checks (no code)
8. monitoring_framework: How to track and validate strategy performance

REQUIREMENTS:
- Focus on medium-to-long-term systematic changes, not short-term price movements
- Rules must be precise enough to backtest and implement systematically
- Include specific numerical thresholds and decision criteria
- Account for all identified failure modes in the rule design
- Ensure rules can be implemented by any qualified team member

CRITICAL JSON FORMATTING REQUIREMENTS:
- Return ONLY valid JSON, no markdown or extra text
- Use double quotes for all strings, not single quotes
- Ensure all brackets and braces are properly closed
- No line breaks within string values - use \n instead
- Proper comma placement between objects and arrays
- Test your JSON before returning it

Return your response as a JSON object with the following structure:
{{
  "strategy_rules": {{
    "selection_criteria": {{
      "primary_criteria": "Main selection rules with specific thresholds...",
      "secondary_criteria": "Additional filters and criteria...",
      "exclusion_criteria": "What to exclude and why..."
    }},
    "weighting_methodology": {{
      "method": "Systematic weighting approach...",
      "parameters": "Specific parameters and calculations...",
      "rebalancing_frequency": "How often to rebalance (quarterly, annually, etc.)"
    }},
    "position_sizing": {{
      "max_position_size": "Maximum position size with rationale...",
      "min_position_size": "Minimum position size with rationale...",
      "sizing_method": "How to calculate sizes based on risk and signal strength..."
    }},
    "execution_guidelines": {{
      "entry_conditions": "Specific entry triggers and conditions...",
      "exit_conditions": "Specific exit triggers and conditions...",
      "risk_management": "Risk controls and stop-loss rules..."
    }},
    "rebalancing_framework": {{
      "frequency": "How often to rebalance...",
      "triggers": "What triggers rebalancing...",
      "methodology": "How to execute rebalancing..."
    }},
    "risk_controls": {{
      "failure_mode_mitigation": "How to prevent identified failure modes...",
      "position_limits": "Specific position and concentration limits...",
      "monitoring_requirements": "What to monitor and how often..."
    }},
    "monitoring_framework": {{
      "performance_metrics": "Key metrics to track...",
      "reporting_frequency": "How often to report...",
      "validation_approach": "How to validate strategy performance..."
    }},
    "heuristic_rules": [
      "Short, actionable selection rule...",
      "Short, actionable weighting rule...",
      "Short, actionable rebalancing rule...",
      "Short, actionable risk control..."
    ]
  }}
}}"""

# Agent 4 - Risk & Compliance Validator Prompt Template
AGENT_4_PROMPT_TEMPLATE = """You are a risk management specialist. Check whether the proposed strategy meets regulatory requirements and risk management standards.

STRATEGY CONTEXT:
- Strategy Name: {strategy_name}
- Asset Class: {asset_class}
- Time Horizon: {time_horizon}

STRATEGY RULES:
- Selection Criteria: {selection_criteria}
- Weighting Method: {weighting_method}
- Position Sizing: {position_sizing}
- Risk Controls: {risk_controls}

CORE MISSION:
Act as the internal watchdog to ensure the strategy is safe and compliant. Focus on:
- SAFETY: Protecting the fund from unacceptable risks
- COMPLIANCE: Ensuring adherence to all relevant regulations
- CONCENTRATION: Preventing excessive exposure to single stocks or industries
- LIQUIDITY: Ensuring adequate liquidity for strategy execution
- MONITORING: Establishing robust risk monitoring procedures

TASK:
Conduct comprehensive risk and compliance analysis of the investment strategy.

Provide:
1. regulatory_requirements: Specific regulations that apply and compliance actions needed
2. position_limits: Maximum position sizes, sector concentration, and geographic limits
3. liquidity_requirements: Minimum liquidity thresholds and monitoring procedures
4. risk_metrics: Key risk measures to track and their acceptable ranges
5. compliance_procedures: Detailed procedures to ensure ongoing compliance
6. reporting_requirements: Specific reports needed for regulators and management
7. stress_testing: Scenarios to test strategy resilience
8. escalation_procedures: What to do when risk limits are breached

REQUIREMENTS:
- Focus on protecting the fund from unacceptable risks
- Identify specific regulatory requirements for the asset class and strategy type
- Set clear position limits to prevent concentration risk
- Ensure adequate liquidity for strategy execution
- Establish robust monitoring and reporting procedures
- Consider worst-case scenarios and stress testing

CRITICAL JSON FORMATTING REQUIREMENTS:
- Return ONLY valid JSON, no markdown or extra text
- Use double quotes for all strings, not single quotes
- Ensure all brackets and braces are properly closed
- No line breaks within string values - use \\n instead
- Proper comma placement between objects and arrays
- Test your JSON before returning it

Return your response as a JSON object with the following structure:
{{
  "risk_compliance": {{
    "regulatory_requirements": [
      {{
        "regulation": "Specific regulation name",
        "requirement": "What is required",
        "compliance_action": "How to comply"
      }}
    ],
    "position_limits": {{
      "max_single_position": "Maximum single position size",
      "max_sector_concentration": "Maximum sector concentration",
      "max_geographic_concentration": "Maximum geographic concentration"
    }},
    "liquidity_requirements": {{
      "minimum_liquidity": "Minimum liquidity requirements",
      "liquidity_monitoring": "How to monitor liquidity",
      "stress_testing": "Stress testing requirements"
    }},
    "risk_metrics": ["Risk metric 1", "Risk metric 2"],
    "compliance_procedures": {{
      "monitoring_frequency": "How often to monitor",
      "reporting_schedule": "Reporting schedule",
      "escalation_procedures": "Escalation procedures"
    }},
    "stress_testing": {{
      "scenarios": ["Scenario 1", "Scenario 2"],
      "frequency": "How often to stress test",
      "thresholds": "Stress test thresholds"
    }},
    "escalation_procedures": {{
      "risk_breach_protocol": "What to do on risk breach",
      "regulatory_breach_protocol": "What to do on regulatory breach",
      "emergency_procedures": "Emergency procedures"
    }}
  }}
}}"""

# Agent 5 - Investment Committee Memo Prompt Template
AGENT_5_PROMPT_TEMPLATE = """You are an Investment Committee Memo specialist. Create a comprehensive investment strategy report for senior decision-makers.

STRATEGY CONTEXT:
{strategy_context}

INTEGRATION REQUIREMENTS:
You must synthesize and reference specific outputs from previous agents:
- Investment ideas and market inefficiencies (Agent 1)
- Quantitative signals, thresholds, and failure modes (Agent 2) 
- Strategy rules, heuristics, and implementation framework (Agent 3)
- Risk controls, compliance requirements, and position limits (Agent 4)

TASK:
Create a comprehensive Investment Committee Memo that provides all information needed for executive approval decisions. The memo must be professional, quantitative, and actionable.

FORMAT YOUR RESPONSE AS A PROFESSIONAL INVESTMENT COMMITTEE MEMO WITH THE FOLLOWING SECTIONS:

# INVESTMENT STRATEGY PROPOSAL
**Strategy Name:** [Strategy Name]
**Proposal Date:** [Current Date]
**Prepared By:** Investment Strategy Team
**Decision Required:** Approve/Reject/Modify for Implementation

## EXECUTIVE SUMMARY
[Provide a concise 1-2 paragraph overview including:
- Core investment thesis and value proposition
- Target AUM capacity and client segments
- Expected return targets with risk parameters (e.g., "Target 8-12% annual returns with max 15% drawdown")
- Key differentiators from existing fund offerings
- Clear recommendation with rationale]

## INVESTMENT THESIS & STRATEGY OVERVIEW
[Strategy description including:
- Detailed investment hypothesis and market inefficiency being exploited
- Target asset classes, market segments, and geographic focus
- Investment time horizon and typical holding periods
- Market opportunity size and addressable market]

## QUANTITATIVE FRAMEWORK & METHODOLOGY
[Analysis of the systematic approach including:
- Description of quantitative signals and their calculation methods
- Data sources, validation approaches, and reliability assessment
- Threshold values and decision criteria with historical context
- Backtesting methodology and performance attribution
- External evidence and academic support for signal effectiveness]

## STRATEGY IMPLEMENTATION FRAMEWORK
[Implementation details including:
- Portfolio construction methodology and stock selection criteria
- Position sizing rules and concentration limits
- Rebalancing frequency, triggers, and transaction cost management
- Execution guidelines for entry/exit decisions
- Technology and data infrastructure requirements
- Operational workflow and trade settlement procedures]

## FINANCIAL PROJECTIONS & PERFORMANCE EXPECTATIONS
[Simple quantitative performance analysis including:
- Base case return expectations with confidence intervals
- Risk metrics: volatility, maximum drawdown, Value-at-Risk (95% confidence)
- Scenario analysis: bear/base/bull case projections over 1, 3, 5 years
- Performance attribution: alpha generation vs. beta exposure
- Benchmark comparison and tracking error estimates
- Correlation analysis with existing fund strategies
- Expected Sharpe ratio and information ratio targets]

## RISK ANALYSIS & MANAGEMENT
[Concise risk assessment including:
- Key risk factors ranked by probability and impact severity
- Quantified risk metrics with acceptable ranges and monitoring frequency
- Stress testing scenarios and portfolio resilience analysis
- Liquidity risk assessment with redemption scenarios
- Operational risks and mitigation controls]

## COMPETITIVE ANALYSIS & MARKET POSITIONING
[Market context and differentiation including:
- Analysis vs. existing internal fund strategies
- External competitive landscape and market share analysis
- Unique value proposition and competitive advantages]

## IMPLEMENTATION PLAN & RESOURCE REQUIREMENTS
[Concise implementation roadmap including:
- Capital requirements: seed capital, operational capital, technology investments
- Personnel requirements: portfolio managers, analysts, risk specialists, operations
- Technology infrastructure: data feeds, risk systems, portfolio management systems
- Timeline: pilot phase, soft launch, full implementation milestones
- Success metrics and key performance indicators]

## REGULATORY & COMPLIANCE FRAMEWORK
[Simple regulatory considerations including:
- Investment mandate compliance and regulatory filing requirements
- Client suitability assessments and distribution restrictions
- Risk management compliance with internal and external requirements
- Reporting obligations to regulators and clients]

## RECOMMENDATION
[Final evaluation and decision framework including:
- Summary of key strategy strengths and potential concerns
- Clear recommendation: APPROVE/REJECT/MODIFY with specific conditions
- Next steps for approved strategies]

CRITICAL REQUIREMENTS:
- Use professional investment management language and industry terminology
- Include specific quantitative metrics with realistic ranges
- Reference all previous agent outputs with specific integration
- Provide actionable recommendations with clear decision criteria
- Include appropriate caveats and assumptions
- Format as a formal committee document suitable for executive review
- Focus on institutional-grade analysis and presentation
- Maintain objectivity while advocating for the strategy
- Include sufficient detail for informed decision-making
- Make the report concise enough such that the entire report will be generated

FORMATTING GUIDELINES:
- Use clear section headers and subsections for easy navigation
- Include quantitative tables and metrics where appropriate
- Provide executive-level summary with detailed supporting analysis
- Ensure logical flow from strategy concept to implementation
- Use bullet points sparingly and only for listing specific items
- Write in prose format suitable for senior management review
- Include cross-references between sections where relevant
- Maintain consistent terminology throughout the document

Return your response as formatted text, not JSON."""

# Market Context Generation Prompt Template
MARKET_CONTEXT_PROMPT_TEMPLATE = """You are a senior market analyst. Generate current market conditions, investment themes, and constraints for an AI investment strategy pipeline.

Provide:
1. market_conditions: Current market environment (2-3 sentences)
2. investment_themes: 3-5 relevant investment themes (array of strings)
3. constraints: Key investment constraints (1-2 sentences)
4. additional_parameters: Any other relevant market data (object with key-value pairs)

Return as JSON:
{{
  "market_conditions": "Current market description...",
  "investment_themes": ["Theme 1", "Theme 2", "Theme 3"],
  "constraints": "Key constraints...",
  "additional_parameters": {{
    "volatility_level": "High/Medium/Low",
    "sector_performance": "Description",
    "economic_indicators": "Key indicators",
    "regulatory_environment": "Current regulations"
  }}
}}"""
