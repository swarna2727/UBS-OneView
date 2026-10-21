# Prompt Engineering Guide

This guide explains how to modify agent prompts for the Investment Strategy Pipeline system.

## Overview

The system uses 5 AI agents, each with specific roles and responsibilities. All prompts are stored in `src/config/prompt_templates.py` for easy modification by prompt engineers.

## Agent Roles and Responsibilities

### Agent 1: Investment Idea Generator
**Role**: Creates original investment strategies that exploit unique market opportunities
**Focus**: Originality, market inefficiencies, systematic approaches
**Key Outputs**: 3 investment ideas with market inefficiency analysis

### Agent 2: Signal Modelling Specialist  
**Role**: Validates ideas and develops testable signals (per-idea evaluation)
**Focus**: Feasibility, testability, robustness, external evidence
**Key Outputs**: Quantifiable signals, failure modes, backtesting requirements — aggregated under `agent_2_output.by_idea[idea_id]`

### Agent 3: Strategy Definition Engineer
**Role**: Converts signals into executable trading rules and selects the final idea
**Focus**: Clarity, systematic rules, medium-to-long-term focus
**Key Outputs**: Selection criteria, weighting methodology, position sizing, execution guidelines, rebalancing framework, risk controls, monitoring framework, and `heuristic_rules` (word-based, no pseudocode). Also sets `selected_idea_id`.

### Agent 4: Risk & Compliance Validator
**Role**: Ensures safety and regulatory compliance
**Focus**: Risk protection, regulatory adherence, concentration limits
**Key Outputs**: Risk metrics, compliance procedures, stress testing

### Agent 5: Investment Committee Memo Writer
**Role**: Creates comprehensive investment strategy reports
**Focus**: Logical soundness, risk control, implementation feasibility
**Key Outputs**: Professional investment strategy report (not email)

## Modifying Prompts

### 1. Access Prompt Templates
All prompts are stored in `src/config/prompt_templates.py`:

```python
# Agent 1 prompt template
AGENT_1_PROMPT_TEMPLATE = """Your prompt here..."""

# Agent 2 prompt template  
AGENT_2_PROMPT_TEMPLATE = """Your prompt here..."""

# And so on...
```

### 2. Key Template Variables
Each template uses specific variables that are populated at runtime:

**Agent 1**:
- `{market_conditions}`: Current market environment
- `{investment_themes}`: Comma-separated investment themes
- `{constraints}`: Investment constraints

**Agent 2** (per-idea calls via orchestrator):
- `{strategy_name}`: Candidate strategy name
- `{description}`: Candidate description
- `{signal_hypothesis}`: Hypothesis
- `{market_inefficiency}`: Inefficiency
- `{asset_class}`: Asset class
- `{time_horizon}`: Time horizon

**Agent 3**:
- `{strategy_name}`: Strategy name
- `{description}`: Strategy description
- `{asset_class}`: Asset class
- `{time_horizon}`: Time horizon
- `{signals_context}`: Formatted per-idea signals context
- `{failure_modes_context}`: Formatted per-idea failure modes context

**Agent 4**:
- `{strategy_name}`: Strategy name
- `{asset_class}`: Asset class
- `{time_horizon}`: Time horizon
- `{selection_criteria}`: Selection criteria from Agent 3
- `{weighting_method}`: Weighting method from Agent 3
- `{position_sizing}`: Position sizing from Agent 3
- `{risk_controls}`: Risk controls from Agent 3

**Agent 5**:
- `{strategy_context}`: Comprehensive strategy context from all previous agents

### 3. Best Practices for Prompt Engineering

#### Focus on Role Clarity
- Each agent has a specific role - ensure prompts reinforce this
- Use clear, action-oriented language
- Emphasize the agent's core mission and concerns

#### Maintain Consistency
- Use consistent terminology across agents
- Ensure output formats match the expected JSON schemas
- Keep the same level of detail and professionalism

#### Test Incrementally
- Modify one agent at a time
- Test the full pipeline after each change
- Verify that outputs flow correctly between agents

#### Consider the Full Pipeline
- Each agent builds on previous outputs
- Ensure prompts account for data from previous agents
- Maintain the logical flow from idea generation to final report

### 4. Common Modifications

#### Adjusting Output Detail
```python
# To increase detail in Agent 1 descriptions
"description": "Detailed explanation of the strategy and the market inefficiency it exploits (300-500 characters)"

# To add more specific requirements
"REQUIREMENTS:
- Each strategy must target different asset classes/segments
- Focus on strategies that exploit specific market inefficiencies
- Include specific quantitative metrics where possible"
```

#### Changing Focus Areas
```python
# To emphasize different aspects in Agent 2
"Focus on:
- FEASIBILITY: Can this strategy be systematically implemented?
- TESTABILITY: What specific metrics can validate this approach?
- ROBUSTNESS: Under what conditions would this strategy fail?
- EVIDENCE: What external factors support this strategy's viability?"
```

#### Modifying Output Structure
```python
# To add new fields to Agent 1 output
"7. market_inefficiency: The specific market inefficiency or behavioral pattern being exploited
8. expected_alpha: Expected alpha generation potential
9. implementation_complexity: Low/Medium/High complexity rating"
```

### 5. Testing Your Changes

1. **Import Test**: Ensure the modified templates import correctly
```python
python -c "from src.config.prompt_templates import AGENT_1_PROMPT_TEMPLATE; print('✅ Template imports successfully')"
```

2. **Agent Test**: Test individual agents with mock data
```python
python -c "from src.agents.agent_1 import InvestmentIdeaGenerator; print('✅ Agent imports successfully')"
```

3. **Pipeline Test**: Run the full pipeline to ensure end-to-end functionality
```python
python -c "from src.pipeline import InvestmentPipeline; print('✅ Pipeline imports successfully')"
```

### 6. Schema Updates

If you add new fields to agent outputs, you may need to update:

1. **Agent JSON Schema**: In the agent's `generate_structured_json` call
2. **Master Schema**: In `master_schema.json`
3. **Agent Processing**: In the agent's `process` method

### 7. Example Updates

#### A) Encouraging Breadth in Agent 1
"""
REQUIREMENTS:
- Each strategy must target different asset classes/segments
- Avoid repetitive theme dominance; encourage distinct rationales
- Be explicit about the market inefficiency exploited
"""

#### B) Ensuring Per-Idea Parity in Agent 2
"""
Return consistent detail across ideas to enable fair comparison by Agent 3.
For each idea: provide 2–3 signals, 3 failure modes, and backtesting needs.
"""

#### C) Heuristic Rules in Agent 3
"""
Provide concise, word-based `heuristic_rules` (no pseudocode). Focus on selection, weighting, rebalancing, risk checks, and monitoring.
"""

```python
# In AGENT_1_PROMPT_TEMPLATE, add ESG requirements:
"""
REQUIREMENTS:
- Each strategy must target different asset classes/segments
- Focus on strategies that exploit specific market inefficiencies
- Ensure ideas are truly original and not obvious factor/momentum/value plays
- Consider ESG factors and sustainable investing opportunities
- Include ESG impact assessment for each strategy
- Be specific about why the opportunity exists and why others haven't exploited it
- Consider behavioral, structural, or informational advantages
"""

# Add ESG field to JSON structure:
"""
7. market_inefficiency: The specific market inefficiency or behavioral pattern being exploited
8. esg_impact: ESG impact assessment (Positive/Neutral/Negative)
9. sustainability_metrics: Key sustainability metrics to track
"""
```

### 8. Troubleshooting

#### Common Issues:
- **Import Errors**: Check that all template variables are properly formatted
- **JSON Schema Errors**: Ensure new fields are added to both template and schema
- **Pipeline Failures**: Verify that agent outputs flow correctly to next agents
- **PDF Formatting**: Check that new content formats properly in PDF output

#### Debug Steps:
1. Check logs in `logs/pipeline_*.log`
2. Test individual agents with mock data
3. Verify JSON schema validation
4. Check PDF generation with sample data

## Conclusion

The prompt template system allows for easy modification of agent behavior without touching core logic. Focus on role clarity, maintain consistency, and test incrementally to ensure successful modifications.
