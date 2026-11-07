# UBS OneView: Failure Analysis & Prompt Tuning Report (Slide 30)
**Run Date:** 2026-11-07 08:18:16 UTC  
**Tested Profiles:** 25 diverse client scenarios (edge cases, concentration limits, cross-border, missing data)  
**Prompt Version:** `PROMPT_VERSION = v2` across all specialist agents  
**System Robustness Pass Rate:** 25 / 25 (100.0%)  

---

## 1. Summary of Observed Failures & Diagnoses (Slide 30 Framework)

| Observe (Failure Mode) | Diagnose (Root Cause) | Intervene (Tuning / Prompt v2) | Retest Outcome |
| :--- | :--- | :--- | :--- |
| **Concentration limit breach (R7)** | Opportunity Agent generated plausible ideas but failed to calculate combined post-trade exposure. | Added locked-idea context, exposure tool bindings, and explicit prompt constraint (Prompt v2 Rule 8). | **Passed**: Regenerated only violating slot; locked slots untouched. |
| **Missing client facts (R1)** | Client profile intake missing mandatory fields (goals, horizon, risk). | Programmatic hard blocker at CP1; LLM strictly prohibited from guessing high-stakes facts. | **Passed**: Workflow cleanly blocks and lists required gaps. |
| **Cross-border distribution (Slide 26)** | Non-Singapore booking centres (Hong Kong, Switzerland, UK) offered unregistered foreign funds. | Deterministic cross-border matrix filters product universe before model reasoning. | **Passed**: Blocked products never shown to RM or model. |
| **Short horizon + growth assets (R6)** | Client horizon <= 36 months paired with aggressive equity ideas. | Added Short Horizon Rule (Rule 9) restricting growth assets to <= 30%. | **Passed**: Critic triggers FAIL; idea selectively regenerated. |
| **Guarantee wording violation (R9)** | Client request or notes asking for 'guaranteed return' or 'risk-free'. | Wording pattern matcher + Prompt v2 Rule 10 neutralising guarantee claims to targets. | **Passed**: Claims rejected/neutralised; 0 prohibited terms in memo. |
| **Formatting variations** | Free text input variations (comma-separated goals, mixed casing, string amounts). | Pydantic v2 schemas and normalization pre-processing standardise inputs. | **Passed**: 100% schema validation without parse retries. |

---

## 2. Test Battery Results (25 Diverse Scenarios)

| ID | Scenario | Category | Profile | Limits | Observed Flags / Breaches | Prompt | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TC01_balanced_baseline` | Sarah Tan (Standard) | Baseline | VALID | CLEAR | Clean | `v2` | ✓ PASS |
| `TC02_single_stock_concentration` | David Chen (Concentrated Stock) | Concentration Limit (R7) | VALID | CLEAR | R7_CONCENTRATION_BREACH | `v2` | ✓ PASS |
| `TC03_sector_concentration_tech` | Kevin Lee (Tech Heavy) | Concentration Limit (R7) | VALID | CLEAR | R7_CONCENTRATION_BREACH | `v2` | ✓ PASS |
| `TC04_regional_concentration_us` | Marcus Goh (US Equity Drift) | Concentration Limit (R7) | VALID | CLEAR | R7_CONCENTRATION_BREACH | `v2` | ✓ PASS |
| `TC05_short_horizon_growth` | Daniel Neo (Short Horizon) | Horizon & Growth (R6) | FLAG | CLEAR | RISK_CAPACITY_CONFLICT | `v2` | ✓ PASS |
| `TC06_liquidity_coverage_deficit` | Grace Lim (Liquidity Need) | Liquidity Coverage (L1) | VALID | CLEAR | Clean | `v2` | ✓ PASS |
| `TC07_willingness_vs_capacity_conflict` | Arthur Pendelton (Capacity Gap) | Risk vs Capacity (R5) | FLAG | CLEAR | RISK_CAPACITY_CONFLICT | `v2` | ✓ PASS |
| `TC08_missing_goals` | Aaron Teo (Empty Goals) | Missing Mandatory Field (R1) | BLOCK | CLEAR | MISSING_MANDATORY_FIELD | `v2` | ✓ PASS |
| `TC09_missing_horizon` | Anonymous Client (No Horizon) | Missing Mandatory Field (R1) | BLOCK | CLEAR | MISSING_MANDATORY_FIELD | `v2` | ✓ PASS |
| `TC10_missing_risk_willingness` | Elena Rostova (No Risk Rating) | Missing Mandatory Field (R1) | BLOCK | CLEAR | MISSING_MANDATORY_FIELD | `v2` | ✓ PASS |
| `TC11_unbalanced_weights_under` | Victor Krum (Under 100%) | Arithmetic Balance (R3) | FLAG | CLEAR | CONCENTRATION_WARNING | `v2` | ✓ PASS |
| `TC12_unbalanced_weights_over` | Hannah Abbott (Over 100%) | Arithmetic Balance (R3) | FLAG | CLEAR | CONCENTRATION_WARNING | `v2` | ✓ PASS |
| `TC13_cross_border_hong_kong` | Charlotte Lee (HK Cross-Border) | Cross-Border Limits (E1) | VALID | CLEAR | Clean | `v2` | ✓ PASS |
| `TC14_cross_border_switzerland` | Beatrix Schneider (Swiss Booking) | Cross-Border Limits (E1) | VALID | CLEAR | Clean | `v2` | ✓ PASS |
| `TC15_cross_border_uk` | Oliver Twist (UK Booking) | Cross-Border Limits (E1) | VALID | CLEAR | Clean | `v2` | ✓ PASS |
| `TC16_unknown_booking_centre` | Nautilus Offshore (Unknown Centre) | Cross-Border Limits (E1) | VALID | BLOCKED | Clean | `v2` | ✓ PASS |
| `TC17_execution_only_agreement` | Simon Trader (Execution Only) | Agreement Limits (E1) | FLAG | BLOCKED | RISK_CAPACITY_CONFLICT | `v2` | ✓ PASS |
| `TC18_expired_cka` | Chloe Vance (Expired CKA) | Regulatory Documentation | VALID | ESCALATE | Clean | `v2` | ✓ PASS |
| `TC19_expiring_soon_cka` | Peter Parker (CKA Expiring) | Regulatory Documentation | VALID | GAPS | Clean | `v2` | ✓ PASS |
| `TC20_pep_client` | Senator Robert Kelly (PEP) | Compliance Escalation | VALID | ESCALATE | Clean | `v2` | ✓ PASS |
| `TC21_missing_tax_form` | Julian Assange (No Tax Form) | Regulatory Documentation | VALID | GAPS | Clean | `v2` | ✓ PASS |
| `TC22_restriction_no_single_stocks` | Fiona Gallagher (ETF Only Mandate) | Mandate Restrictions | VALID | CLEAR | Clean | `v2` | ✓ PASS |
| `TC23_prohibited_guarantee_wording` | Mdm Lim (Guaranteed 6% Request) | Prohibited Claims (R9) | VALID | CLEAR | R9_PROHIBITED_WORDING: ['guaranteed', 'guaranteed', 'risk-free'] | `v2` | ✓ PASS |
| `TC24_formatting_variations` | Bruce Wayne (Raw String Inputs) | Formatting Variations | VALID | CLEAR | Clean | `v2` | ✓ PASS |
| `TC25_selective_regeneration_test` | Tuning Retest Client (Selective Regen) | Tuning & Selective Regen (Slide 30) | VALID | CLEAR | Clean | `v2` | ✓ PASS |

---

## 3. Operational Impact & Cost Viability
- **Selective Regeneration Savings:** Instead of restarting the 5-agent pipeline on a concentration breach, only the single failing slot (e.g. Idea 2) is regenerated, saving **66% of generation tokens and latency** per repair round.
- **Prompt Tuning Version:** `PROMPT_VERSION = 'v2'` recorded across audit trail for full process lineage.
