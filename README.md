# AI Investment Strategy Pipeline

A production-ready sequential AI agent pipeline for professional investment strategy development using Gemini 2.5 Flash.

## System Overview

This system consists of 5 specialized AI agents that collaborate in multi-round cycles on a shared JSON schema, culminating in a comprehensive investment memo:

1. **Investment Idea Generator** - Creates 3 original investment strategies spanning different asset classes
2. **Signal Modelling Specialist** - Develops testable signals and identifies failure scenarios for each candidate idea (per-idea evaluation)
3. **Strategy Definition Engineer** - Converts signals into executable trading rules with concise heuristic rules (no pseudocode) and selects the final idea
4. **Risk & Compliance Validator** - Identifies regulatory requirements and risk management controls
5. **Investment Memo Writer** - Creates executive-level investment memo with PDF output

## Features

- **Flat, Multi-Round Collaboration**: Agents 1–4 collaborate across 1–3 rounds on a shared schema
- **Per-Idea Evaluation**: Agent 2 evaluates signals/failure modes for each candidate idea, enabling fair comparison
- **Clear Selection Ownership**: Agent 3 selects (and may reselect) the final idea each round based on comparative evidence
- **Professional Output**: Generates publication-ready investment memos in PDF format
- **Comprehensive Validation**: JSON schema validation at each step with error handling
- **Executive Interface**: Streamlit frontend with preset scenarios and custom inputs
- **Data Lineage**: Complete transaction tracking with unique IDs and processing times
- **Risk Management**: Built-in compliance checks and risk metric specifications

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configuration

1. Get a Google API key for Gemini 2.5 Flash from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a `.env` file:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Running the Application

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

## Usage

### Web Interface

1. **Enter API Key**: Provide your Google Gemini API key
2. **Choose Input Method**: 
   - Select from preset scenarios (Technology Growth, ESG Sustainable, Value Recovery)
   - Or create custom input with market conditions, themes, and constraints
3. **Rounds**: Choose number of collaboration rounds (default 2)
4. **Run Pipeline**: Execute the collaborative pipeline
5. **Download Results**: Get PDF investment memo and JSON data files

### Programmatic Usage

```python
from src.pipeline import InvestmentPipeline

# Initialize pipeline
pipeline = InvestmentPipeline(api_key="your_api_key")

# Define input data
input_data = {
    "market_conditions": "Current bullish market with tech focus",
    "investment_themes": ["AI", "Cloud Computing", "Cybersecurity"],
    "constraints": "Large-cap focus, max 5% individual positions"
}

# Run pipeline (collaboration rounds)
result = pipeline.run(
    input_data=input_data,
    rounds=2
)

# Access results
print(f"Transaction ID: {result['transaction_id']}")
print(f"PDF Path: {result['pdf_path']}")
print(f"Processing Time: {result['total_processing_time_ms']}ms")
```

## Architecture

### Data Flow (High Level)
```
User Input → Round 1: [A1 → A2(per idea) → A3 → A4] → Round 2: [A1 → A2(per idea) → A3 → A4] → … → A5 → PDF
```

### File Structure
```
src/
├── agents/              # AI agent implementations
│   ├── agent_1_idea_generator.py
│   ├── agent_2_signal_modelling.py
│   ├── agent_3_strategy_definition.py
│   ├── agent_4_risk_compliance.py
│   └── agent_5_memo_writer.py
├── schemas/             # JSON validation schemas
├── utils/               # Utilities (Gemini client, validation, PDF)
└── pipeline.py          # Main orchestrator
```

### Agent Specifications

**Agent 1 - Investment Idea Generator**
- Input: Market conditions, themes, constraints
- Output: 3 investment ideas with strategy names, descriptions, hypotheses
- Validation: Ensures 3 ideas across different asset classes

**Agent 2 - Signal Modelling Specialist**  
- Input: Each candidate idea from Agent 1 (evaluated per idea)
- Output: 2-3 quantifiable signals, 3 failure modes, backtesting requirements (per idea)
- Validation: Specific data sources and numerical thresholds required

**Agent 3 - Strategy Definition Engineer**
- Input: Per-idea signals and failure modes from Agent 2  
- Output: Selection criteria, weighting methodology, position sizing, execution guidelines, rebalancing framework, risk controls, monitoring framework, and `heuristic_rules` (concise, word-based). Selects `selected_idea_id` each round.
- Validation: All rules must be quantifiable and executable

**Agent 4 - Risk & Compliance Validator**
- Input: Strategy rules from Agent 3
- Output: Position limits, liquidity constraints, regulatory flags, risk metrics
- Validation: Institutional-grade risk management standards

**Agent 5 - Investment Memo Writer**
- Input: All previous agents' outputs
- Output: Executive memo with summary, risks, approvals, timeline, resources
- Validation: Professional formatting suitable for executive review

## Output Examples

### Investment Memo Structure
- **Executive Summary** (2-3 sentences)
- **Strategy Overview** (1 paragraph) 
- **Key Risks** (3 risks with severity and mitigation)
- **Approval Reasons** (3 reasons with supporting data)
- **Implementation Timeline** (specific milestones)
- **Resource Requirements** (capital, personnel, technology, data)

### Generated Files
- `agent{i}_{ClassName}_round{N}_{transaction_id}.json` - Per-agent snapshots
- `round{N}_{transaction_id}.json` - Per-round snapshot
- `{transaction_id}_complete.json` - Final JSON
- `{transaction_id}_final_memo.pdf` - Professional PDF report
- `schema_final.json` and `schema_final_{transaction_id}.json` - PRD alias and tx-scoped final schema
- `final_report.txt` and `final_report_{transaction_id}.txt` - PRD alias and tx-scoped committee text

## Performance Targets

- **Complete Pipeline**: Under 3 minutes end-to-end
- **Individual Agents**: 20-40 seconds each
- **JSON Validation**: 95%+ schema compliance with fallbacks
- **Error Recovery**: Automatic retry with intelligent fallbacks

## Production Features

- **Robust Error Handling**: Automatic retry with exponential backoff
- **Safety Filter Bypass**: Optimized for financial content
- **Rate Limiting**: Intelligent API call management
- **Comprehensive Logging**: Full transaction tracking and debugging
- **Fallback Mechanisms**: Graceful degradation on failures
- **Selection Control**: Agent 3 owns selection; Agent 2 compares all ideas; optional convergence early-stop

## Customization

### Adding New Agents
1. Extend `BaseAgent` class
2. Implement `process()` method
3. Define output schema
4. Add to pipeline orchestrator

### Custom Schemas
Update schemas in `src/schemas/json_schemas.py` for specialized requirements.

### PDF Formatting
Modify `src/utils/pdf_generator.py` for custom report layouts.

## How It Works

1) Rounds and Shared Schema
- The pipeline runs 1–3 rounds. In each round, Agents 1–4 operate on a shared JSON schema. The orchestrator merges outputs with metadata (proposed_by, confidence, comments, history).

2) Per-Idea Signals (Agent 2)
- For each idea produced by Agent 1, Agent 2 is run separately. Outputs are aggregated under `agent_2_output.by_idea[idea_id]` (signals, failure modes, backtesting).

3) Idea Selection (Agent 3)
- Agent 3 consumes `by_idea` to produce rules and select `selected_idea_id` based on comparative evidence. Heuristic scoring favors breadth by counting signals and penalizing failure modes. Reselection is allowed every round.

4) Synthesis (Agent 5)
- The final committee agent produces a human-readable report (`report_text`) which is rendered to PDF. PRD-compatible aliases are written for convenience.

5) Observability
- Per-agent and per-round snapshots are written for auditability, with per-round timing and optional early convergence stop.

## Troubleshooting

**Common Issues:**
- **API Key Invalid**: Verify Gemini 2.5 Pro access and key format
- **JSON Validation Fails**: Check agent output formats against schemas
- **PDF Generation Error**: Ensure reportlab dependencies installed
- **Processing Timeout**: Increase timeout limits for complex strategies

**Debug Mode:**
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-agent`)
3. Add tests for new functionality
4. Submit pull request with detailed description

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Create GitHub issue with detailed error information
- Include transaction ID and log files
- Specify Gemini API key status (without sharing the key)
