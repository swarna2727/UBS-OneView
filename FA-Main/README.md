# UBS Human-Led Multi-Agent Financial Advisory System

A human-led multi-agent investment advisory application for UBS financial advisors (Relationship Managers), built with **LangGraph**, **Pydantic v2**, **OpenRouter**, **FastAPI** and **Streamlit**. It follows the *Personalized Financial Advisory (AAFS v2)* design: the app is the advisor's decision workspace, and the PDF memo is the final advisory artifact.

---

## 1. Design Principles

1. **Clear role boundaries**: five agents, each owning one kind of reasoning, hand off typed Pydantic objects.
2. **Human accountability by design**: the advisor confirms inputs, reviews research, decides on recommendations and approves the memo. No PDF exists before advisor sign-off.
3. **Evidence before recommendation**: every market claim carries a source and publication date; sources older than 30 days are excluded (R8).
4. **Deterministic control where possible**: risk scores, exposures, allocation totals and every suitability rule run in pure Python (`src/tools/`, linted to never import LLM code). Models handle language, relevance and synthesis.
5. **Lowest effective level of automation**: research and drafting are automated; client understanding, exceptions and final approval stay with the advisor.

---

## 2. Use Cases: Three Triggers, One Set of Agents

The use cases are connected in the client lifecycle (prepare → recommend → reassess) but each has its own trigger and its own path through the LangGraph graph. The use case is set when a session starts (`use_case` in the intake payload; the API defaults to `adhoc_request`).

| Use case | Trigger | Path | Checkpoints | Output |
|---|---|---|---|---|
| **UC1 Prepare** · `pre_meeting` | “I'm meeting the client tomorrow” (a quarterly review; a simulated calendar prompt on the landing page shows the future-state trigger) | Profile (with the changes since `last_review`) → market landscape since the last review → **only if the profile changed**: ideas → critic → approval → meeting brief | CP1 confirm profile · CP2 review research (choose whether to add ideas; on by default when the profile changed) · CP3 approve ideas (if any) · CP4 approve brief | Meeting brief PDF: profile to confirm, changes, portfolio risks, market landscape, talking points, questions, ideas (if any), with charts |
| **UC2 Recommend** · `adhoc_request` | “The client has S$120k to deploy” | New capital added as cash → research → ideas → critic (FAIL ideas regenerated, passing ideas locked) → approval → memo | CP1 confirm inputs · CP2 research · CP3 approve ideas · CP4 sign off memo | 3-page investment memo PDF with allocation, risk-band and stress charts |
| **UC3 Reassess** · `market_event` | “Markets just moved” (event preset + description) | Client record → event research → deterministic client impact + rule re-check → advisor decision | Decide: **maintain** / **monitor** (follow-up trigger required) / **revise** (→ ideas → critic → CP3 → memo → CP4) | Decision record PDF with the impact chart, or a revised memo |

Within one browser session the client record carries forward: an approved UC1 brief becomes the new `last_review`, and approved UC2/UC3 trades are assumed executed, so the next workflow starts from the updated portfolio.

### UC2 workflow and checkpoints (UC1 and UC3 reuse these nodes)

```mermaid
flowchart TD
    START([Advisor enters client]) --> CP[1. Client Profile Agent]
    CP --> R1{{"Advisor: confirm inputs / resolve conflicts"}}
    R1 -->|"Regenerate with edited fields"| CP
    R1 -->|"Approve"| MI[2. Market Intelligence Agent]
    MI --> R2{{"Advisor: inspect research and sources"}}
    R2 -->|"Regenerate with feedback"| MI
    R2 -->|"Approve"| IG[3. Investment Idea Agent]
    IG --> SC[4. Suitability Critic + rule engine]
    SC -->|"FAIL, retries left (R10)"| SEL[Lock passing ideas, regenerate failed ones]
    SEL --> IG
    SC -->|"PASS / FLAG / FAIL escalated"| R3{{"Advisor: recommendation review"}}
    R3 -->|"Revise idea / regenerate all"| IG
    R3 -->|"Edit allocation"| SC
    R3 -->|"Accept (PASS) / documented exception (FLAG)"| MF[5. Memo Formatter Agent]
    MF --> R4{{"Advisor: final approval"}}
    R4 -->|"Regenerate memo"| MF
    R4 -->|"Revise idea"| IG
    R4 -->|"Edit allocation"| SC
    R4 -->|"Approve"| PDF[Generate PDF] --> DONE([APPROVED])
    R1 & R2 & R3 & R4 -->|"Reject"| REJ([REJECTED])
```

| Checkpoint | Advisor actions | Notes |
|---|---|---|
| Client profile | `approve`, `regenerate` (with `edits` to the intake fields), `reject` | Approve is unavailable while mandatory fields are missing (R1). Approving resolves flags such as R5 with the advisor's note. |
| Research brief | `approve`, `regenerate` (notes become feedback), `reject` | Advisor can inspect every source and its date. |
| Recommendations | `approve` (PASS only), `override_flag` (FLAG, note required), `revise` (one idea via `idea_id`, others locked), `edit_allocation`, `regenerate` (all ideas), `reject` | A FAIL can never be approved. FAILed ideas are regenerated automatically up to 2 times per idea before reaching the advisor (R10). An edited allocation is re-validated and returned to the advisor, never auto-regenerated. |
| Final approval | `approve` (generates the PDF), `regenerate`, `revise`, `edit_allocation`, `reject` | Approve is unavailable if the memo fails R9 wording or the consistency check. |
| Market-event decision (UC3) | `maintain`, `monitor` (`edits.follow_up` required), `revise` | Notes are required. The impact tool suggests a decision; the advisor makes it. A market event never places a trade. |
| Meeting brief (UC1) | `approve` (generates the PDF), `regenerate`, `reject` | Approve is unavailable if the brief fails the R9 wording check. |

`add_advisor_note` can be sent at any time; it is logged but never acts as a review decision. Advisor notes and feedback are passed to models as quoted data, not instructions.

---

## 3. Suitability Rule Engine (`src/tools/risk_engine.py`, `config/suitability_policy.yaml`)

| # | Check | Default | Outcome |
|---|---|---|---|
| R1 | Mandatory fields: objective, horizon, risk profile, liquidity need, investable amount | any missing | BLOCK |
| R2 | Instrument in the approved universe file | not found | FAIL |
| R3 | Post-trade allocations sum to 100% (purchases are funded) | ±0.01 pt | FAIL |
| R4 | Proposed portfolio risk score within the client's band | band ceiling | FAIL (FLAG if an existing breach is not worsened) |
| R5 | Risk willingness above risk capacity | any gap | FLAG, lower band applied |
| R6 | Growth-asset share when the horizon is 3 years or less | cap 30% | FAIL (FLAG if existing, not worsened) |
| R7 | Single market, sector or instrument share after the change | 30% (instrument: client limit if lower) | FLAG existing, FAIL if new or worsened |
| R8 | Every cited source retrieved and within the freshness limit | 30 days | FAIL missing, FLAG stale |
| R9 | Prohibited wording ("guaranteed", "risk-free"); negated uses allowed | any match | FAIL |
| R10 | Automatic regenerations of the same idea | max 2 | Escalate to advisor |
| L1 | Liquid assets cover liquidity needs due within 24 months | 1.0x | FAIL (FLAG if existing, not worsened) |

Thresholds are illustrative defaults held in configuration and must be calibrated against UBS suitability policy before any pilot. A portfolio-level FAIL is attributed to the smallest set of ideas whose removal fixes it, so only those slots are regenerated.

---

## 4. The Five Agents

| Agent | Role | Deterministic tools |
|---|---|---|
| Client Profile (`client_profile.py`) | Standardise advisor inputs, compute risk band (lower of willingness and capacity), flag gaps and conflicts | `required_field_validator`, `risk_profile_calculator`, `portfolio_parser` |
| Market Intelligence (`market_intelligence.py`) | Retrieve and synthesise client-relevant research with sources and dates | `retrieval_rag_tool`, `source_freshness_validator`, `quantitative_analysis_tool`, `sentiment_analysis_tool` |
| Investment Idea (`investment_idea.py`) | Three complementary ideas from the approved universe, each with role, conviction, risks, invalidation trigger and alternative considered | `approved_product_universe_tool`, `portfolio_exposure_calculator` |
| Suitability Critic (`suitability_critic.py`) | PASS / FLAG / FAIL per idea and overall from the rule engine | `suitability_rule_engine` |
| Memo Formatter (`memo_formatter.py`) | Formats approved artifacts only: the 3-page memo (UC2, UC3 revise), the meeting brief (UC1) and the decision record (UC3); restates guarantee requests as targets; R9 and consistency checks | `memo_consistency_validator`, `citation_source_mapper`, PDF renderers with charts (`memo_tools.py`, `pdf_charts.py`) |

Deterministic UC tools: `profile_change_detector` (UC1, changes since the last review), `with_new_capital` (UC2, new money held as cash), `market_event_impact` (UC3, contribution by holding, drifted-portfolio rule re-check and a suggested decision) and `stress_impact` (`src/tools/event_tools.py`). Event shocks and stress scenarios are illustrative and must be calibrated before a pilot.

---

## 5. Models (OpenRouter)

The Streamlit workspace has two run modes, chosen on the landing page. **Offline · test data stubs** uses `FakeAdvisoryLLMClient`, so every demo runs locally without a key or external model calls. **Live · OpenRouter** runs the same LangGraph workflow in-process with `OpenRouterLLMClient`; it needs `OPENROUTER_API_KEY` in the environment or in a `.env` file. The REST API uses OpenRouter when the key is set and otherwise falls back to the offline client.

```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_MODEL=anthropic/claude-sonnet-4.5             # default model for every agent
export OPENROUTER_MODEL_INVESTMENTIDEAAGENT=openai/gpt-4.1      # optional per-agent override
```

Agent outputs are validated against their Pydantic schema; invalid output is retried once with the validation errors, then the agent raises `LLMUnavailableError` ("data unavailable") instead of guessing. The model and prompt version of every agent call are written to the run log.

---

## 6. Audit Run Log

Each session is one run ID. `data/runs/<session_id>.jsonl` (override with `RUN_LOG_DIR`) records the client inputs, policy version, every agent output with model and prompt version, rule results, automatic regenerations, advisor decisions and notes, PDF generation and the final outcome. `GET /sessions/{id}/audit` returns it.

---

## 7. Running

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py                                   # offline advisor prototype
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload      # REST API
python -m pytest tests/ -v                                       # offline test suite
```

### Advisor workspace (Streamlit)

1. **Landing page**: pick the run mode (offline stubs or live OpenRouter), then open a client or enter a new one. A simulated calendar card ("Quarterly review — Sarah Tan, tomorrow") previews the future-state trigger for UC1. In live mode, client details are sent to the model provider and use credits.
2. **Client home**: "What do you need today?" with one card per use case (meeting date for UC1, amount for UC2, event preset and description for UC3) and the client's history in this session.
3. **Workspace**: the left panel lists only the steps of the active workflow; optional branches (UC1 ideas, UC3 revise) appear once taken. Each step is one page: labelled tabs show the intermediate output, and the advisor checkpoint sits at the bottom of the page with the button to the next page. Steps not reached yet are locked; amber marks the checkpoint waiting for you.

| Page | Intermediate output (tabs) | Checkpoint at the bottom |
|---|---|---|
| Client profile | Field form with required fields (*) as a control, changes since the last review (UC1), exposure donuts and risk band, flags and rules | CP1: save changes & re-check, confirm, or reject. Confirm is disabled while a required field is missing (R1) |
| Market landscape / Research | Findings with sources, risk/return and Sharpe charts, sentiment, source age vs the 30-day limit | CP2: approve (UC1: add 3 ideas or not), re-run with feedback, reject |
| Event & client impact (UC3) | What changed, contribution by holding, suitability re-check, advice under review | Decide: maintain / monitor / revise |
| Candidate ideas | **Candidate idea set** per round, rationale, proposed weights, lock/regenerate grid with critic feedback | Next page |
| Suitability check | **System-checked suitable set**: verdicts, before → after vs rule limits, exposure shift and stress, rule matrix | Next page |
| Advisor approval | **Human-approved set**: checked set, approved set, idea lineage | CP3: approve / exception, revise one idea, edit allocation, regenerate, reject |
| Memo (UC2, UC3 revise) / Meeting brief (UC1) | **Memo formatting**: memo pages or brief sections | CP4: approve → PDF download |
| Decision record (UC3) | Decision, rationale, follow-up, impact chart | PDF download |
| Audit & performance | Decision timeline, latency per agent call, tokens and estimated cost, notes, raw run log | – |

The idea-set lineage is kept in graph state: `idea_rounds` stores every candidate set with the critic's verdicts and what triggered it (initial, automatic regeneration, advisor revision, allocation edit), and `approved_idea_set` stores the set the advisor approved at CP3. The memo is formatted from that set.

### Demo clients (`data/samples/client_profiles.json`)

| Client | Profile | Control shown |
|---|---|---|
| Sarah Tan, 35 | Illustrative S$800k portfolio, moderate, S$300k home down-payment goal in 8 years; 45% global equities (ACWI), 30% investment-grade bonds (AGGG), 15% cash/MMF, 10% gold. Last review (Jul 2026) had a 10-year horizon | Demo client for all three use cases. UC1: the horizon change is detected, so 3 ideas are added to the brief. UC2 (S$120k): Idea 2 (high yield) fails R4, Ideas 1 and 3 are locked and only Idea 2 is regenerated. UC3: rates shock → suggested *monitor* |
| Daniel, 29 | S$220k, "comfortable with high risk", S$180k deposit in 15 months | R5 willingness vs capacity: band lowered to Conservative. The first candidate set over-allocates growth (SPYG 30%), so the critic FAILs it on R4/R6/L1, locks the other two ideas and regenerates only slot 2 |
| Mdm Lim, 62 | S$1.2M, low-moderate risk, retiring in 3 years, wants a "guaranteed 6%" | Memo restates the request as a target and states returns cannot be guaranteed (R9) |
| Aaron, 45 | S$400k, moderate, no horizon given | R1 blocks the workflow until the horizon is supplied |

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sessions` | Start a session with the client intake; runs to the first advisor checkpoint |
| `GET` | `/sessions/{id}` | Workspace view, including `pending_review` (agent output, `allowed_actions`) |
| `POST` | `/sessions/{id}/actions` | Advisor decision: `approve`, `regenerate`, `reject`, `revise`, `override_flag`, `edit_allocation`, `add_advisor_note` (optional `idea_id`, `notes`, `edits`). 400 if not allowed at the current checkpoint |
| `GET` | `/sessions/{id}/pdf` | Approved memo PDF (exists only after final approval) |
| `GET` | `/sessions/{id}/audit` | Audit trail and full run log |
| `GET` | `/health` | Health check |

### Tests

- `tests/unit/test_risk_engine.py`: every rule R1-R10 and L1, existing-vs-worsened outcomes, idea attribution.
- `tests/unit/test_openrouter_client.py`: schema validation, retry on invalid output, fail-safe error, per-agent models.
- `tests/integration/test_trajectories.py`: full trajectory with PDF only after approval; automatic regeneration then R10 escalation; a FAIL never reaches the memo; selective revision locks other ideas; allocation edits re-validated; FLAG needs a documented exception; demo-client controls; audit log contents; idea-set lineage (candidate → checked → approved, cleared again after a revision).
- `tests/integration/test_use_cases.py`: UC1 with and without profile changes (and the advisor override), UC2 new capital and selective regeneration, UC3 maintain/monitor validation, the decision record and the revise path, R1 still blocking UC3, unknown use case rejected.
