"""Evaluation framework (slides 30-32): deterministic metrics and the baseline-vs-multi-agent harness."""

from src.evaluation import metrics as m
from src.evaluation.harness import run_comparison, render_markdown, perturb
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.advisory import CandidateIdeaSet, InvestmentIdea, AllocationRange

def make_idea(slot, ticker, weight=2.0, source="SRC-UBS-CIO-2026-101", alt="Hold cash instead."):
    return InvestmentIdea(
        idea_id=f"I-{ticker}", slot_number=slot, instrument_id=f"X-{ticker}", ticker=ticker, name=ticker, action="ADD",
        target_weight_pct=weight, allocation_range=AllocationRange(min_pct=0, max_pct=weight, target_pct=weight),
        portfolio_role="INCOME", thesis="t", why_now="w", client_fit_rationale="f", key_risks=["r"],
        invalidation_trigger="i", cited_evidence_ids=[source], alternative_considered=alt
    )

def test_explainability_counts_the_five_questions():
    full = [make_idea(1, "IEF"), make_idea(2, "LQD"), make_idea(3, "GLD")]
    assert m.explainability(full) == 1.0
    partial = [make_idea(1, "IEF", alt=""), make_idea(2, "LQD"), make_idea(3, "GLD")]
    assert m.explainability(partial) == round(14 / 15, 3)

def test_set_distance_and_consistency():
    a = [make_idea(1, "IEF"), make_idea(2, "LQD"), make_idea(3, "GLD")]
    b = [make_idea(1, "IEF"), make_idea(2, "LQD"), make_idea(3, "MSFT")]
    assert m.set_distance(a, a) == 0.0 and m.set_distance(a, b) == 0.5
    assert m.output_consistency([a, a, a]) == 1.0 and m.output_consistency([a, b]) == 0.0

def test_perturb_makes_client_more_conservative(balanced_client_data):
    changed = perturb(balanced_client_data)
    assert changed["stated_risk_willingness"] == "Conservative" and changed["time_horizon_months"] <= 24
    assert perturb(balanced_client_data) is not balanced_client_data

def test_unsupported_claims_flags_stale_or_missing_sources(balanced_client_data):
    from src.agents.client_profile import ClientProfileAgent
    from src.agents.market_intelligence import MarketIntelligenceAgent
    llm = FakeAdvisoryLLMClient()
    brief = MarketIntelligenceAgent(llm).run(ClientProfileAgent(llm).run(balanced_client_data))
    ideas = [make_idea(1, "IEF"), make_idea(2, "LQD", source="SRC-STALE-ARCHIVE-2025-01"), make_idea(3, "GLD", source="NOPE")]
    assert m.unsupported_claims(ideas, brief) == 2

def test_harness_runs_both_systems_and_renders_report(balanced_client_data):
    results = run_comparison(balanced_client_data, FakeAdvisoryLLMClient)
    assert set(results) == {"single_llm", "multi_agent"}
    for system in results.values():
        assert {"suitability_violations", "unsupported_claims", "personalization", "explainability",
                "consistency", "latency_s", "llm_calls"} <= set(system)
    # The single LLM has no critic, so its unchecked set carries the deterministic engine's findings.
    assert results["multi_agent"]["suitability_violations"] == 0
    report = render_markdown({"balanced_client": results}, offline=True)
    assert "Single LLM" in report and "Multi-agent" in report
