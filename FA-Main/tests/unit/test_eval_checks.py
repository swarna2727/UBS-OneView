"""The evaluators themselves are tested: each check passes on a healthy run and FAILS when the defect it guards against is seeded.

A test suite that cannot fail proves nothing (deck slide 26: test the cases where the obvious answer is wrong)."""

import copy
import dataclasses
import pytest

from src.evaluation import components, quality, trajectory
from src.evaluation.judge import CRITERIA, RUBRIC_MODELS, HeuristicJudge, JudgeVerdict, LLMJudge, mean_score
from src.evaluation.performance import BUCKETS, performance_report
from src.evaluation.runner import drive_session
from src.evaluation.scorecard import scenarios, evaluate_run, judge_run
from src.llm.fake import FakeAdvisoryLLMClient


@pytest.fixture(scope="module")
def runs():
    return {uc: drive_session(raw, FakeAdvisoryLLMClient(), f"evalchk_{uc}") for uc, raw in scenarios().items()}


def failed(checks):
    return {c["check"] for c in checks if not c["passed"]}


def tamper(run, **state_changes):
    return dataclasses.replace(run, state={**run.state, **state_changes})


def test_healthy_runs_pass_every_level(runs):
    for uc, run in runs.items():
        assert run.completed, f"{uc} blocked at {run.blocked_at}"
        for fn in (components.component_level, quality.quality_level, trajectory.workflow_level):
            assert failed(fn(run)) == set(), (uc, fn.__name__, failed(fn(run)))


# ------------------------------------------------------------------ component level

def test_schema_validation_catches_a_corrupt_agent_output(runs):
    run = copy.deepcopy(runs["adhoc_request"])
    next(e for e in run.log if e["event"] == "agent_output" and e["stage"] == "suitability_critic")["output"]["overall_status"] = "MAYBE"
    assert "schema:suitability_critic" in failed(components.schema_validation(run))


def test_schema_validation_fails_on_an_empty_log(runs):
    assert failed(components.schema_validation(dataclasses.replace(runs["pre_meeting"], log=[]))) == {"schema:any-output"}


def test_calculation_accuracy_catches_a_wrong_portfolio_impact(runs):
    run = copy.deepcopy(runs["pre_meeting"])
    brief = run.state["research_brief"]
    brief.portfolio_impact[0] = brief.portfolio_impact[0].model_copy(update={"contribution_pct": 99.0})
    assert "calc:portfolio-impact" in failed(components.calculation_accuracy(run))


def test_calculation_accuracy_catches_a_wrong_risk_score(runs):
    run = copy.deepcopy(runs["adhoc_request"])
    a = run.state["assessment"]
    run.state["assessment"] = a.model_copy(update={"after_metrics": a.after_metrics.model_copy(update={"after_risk_score": 4.9})})
    assert "calc:risk-score-after" in failed(components.calculation_accuracy(run))


def test_calculation_accuracy_catches_a_product_both_eligible_and_blocked(runs):
    run = copy.deepcopy(runs["pre_meeting"])
    om = run.state["offer_map"]
    run.state["offer_map"] = om.model_copy(update={"blocked_tickers": om.blocked_tickers + [om.eligible_tickers[0]]})
    assert "calc:offer-map-partition" in failed(components.calculation_accuracy(run))


def test_source_validity_catches_stale_uncited_and_unlabelled_sources(runs):
    run = copy.deepcopy(runs["adhoc_request"])
    brief = run.state["research_brief"]
    run.state["research_brief"] = brief.model_copy(update={"window_days": 1, "external_news_used": True})
    bad = failed(components.source_validity(run))
    assert {"source:within-window", "source:no-external-news"} <= bad
    run = copy.deepcopy(runs["adhoc_request"])
    ideas = run.state["candidate_ideas"]
    run.state["candidate_ideas"] = ideas.model_copy(update={"ideas": [ideas.ideas[0].model_copy(update={"cited_evidence_ids": ["GHOST"]})] + ideas.ideas[1:]})
    assert "source:ideas-cite-retrieved" in failed(components.source_validity(run))


def test_source_validity_catches_a_withdrawn_source_in_use_and_an_unexplained_exclusion(runs):
    run = copy.deepcopy(runs["pre_meeting"])
    brief = run.state["research_brief"]
    withdrawn = brief.validated_sources[0].model_copy(update={"status": "withdrawn"})
    run.state["research_brief"] = brief.model_copy(update={"validated_sources": [withdrawn] + brief.validated_sources[1:]})
    assert "source:all-active" in failed(components.source_validity(run))
    run = copy.deepcopy(runs["pre_meeting"])
    brief = run.state["research_brief"]
    assert brief.excluded_sources, "the demo corpus holds superseded, withdrawn and stale notes"
    run.state["research_brief"] = brief.model_copy(update={"excluded_sources": [brief.excluded_sources[0].model_copy(update={"reason": ""})]})
    assert "source:exclusions-explained" in failed(components.source_validity(run))


# ------------------------------------------------------------------ output quality, per use case

def test_uc1_catches_an_unsourced_development_and_a_wrong_impact(runs):
    run = copy.deepcopy(runs["pre_meeting"])
    brief = run.state["meeting_brief"]
    run.state["meeting_brief"] = brief.model_copy(update={"market_developments": ["Markets will rise next quarter."], "portfolio_impact": []})
    bad = failed(quality.uc1_checks(run))
    assert {"uc1:source-and-date-on-each-point", "uc1:impact-matches-the-data", "uc1:no-unsupported-claims"} <= bad


def test_uc2_catches_missing_why_wrong_lanes_and_a_leaked_blocked_product(runs):
    run = copy.deepcopy(runs["adhoc_request"])
    ideas = run.state["candidate_ideas"]
    broken = [ideas.ideas[0].model_copy(update={"thesis": " ", "lane": "EXPLORATORY"})] + ideas.ideas[1:]
    run.state["candidate_ideas"] = ideas.model_copy(update={"ideas": broken})
    assert {"uc2:each-idea-has-a-why", "uc2:two-lanes-labelled"} <= failed(quality.uc2_checks(run))
    leaked = tamper(runs["adhoc_request"], offer_map=runs["adhoc_request"].state["offer_map"].model_copy(update={"blocked_tickers": ["ACWI"]}))
    assert "uc2:blocked-products-not-shown" in failed(quality.uc2_checks(dataclasses.replace(leaked, visible=['{"ticker": "ACWI"}'])))


def test_fit_and_eligibility_catches_a_delivered_idea_outside_client_limits(runs):
    run = copy.deepcopy(runs["adhoc_request"])
    om = run.state["offer_map"]
    ticker = run.state["candidate_ideas"].ideas[0].ticker
    run.state["offer_map"] = om.model_copy(update={"blocked_tickers": om.blocked_tickers + [ticker],
                                                   "eligible_products": [p for p in om.eligible_products if p.ticker != ticker]})
    assert "quality:delivered-ideas-pass-rules" in failed(quality.fit_and_eligibility(run))


def test_uc3_catches_an_untraceable_summary_missing_owner_and_an_unapproved_update(runs):
    run = copy.deepcopy(runs["post_meeting"])
    rec = run.state["meeting_record"]
    said = [rec.client_said[0].model_copy(update={"source_lines": [99]})] + rec.client_said[1:]
    actions = [rec.actions[0].model_copy(update={"owner": None})] + rec.actions[1:]
    run.state["meeting_record"] = rec.model_copy(update={"client_said": said, "actions": actions})
    assert {"uc3:summary-matches-transcript", "uc3:actions-have-owner-and-date"} <= failed(quality.uc3_checks(run))
    run = copy.deepcopy(runs["post_meeting"])
    run.state["approved_profile_updates"] = [{"change_id": "CHG-99"}]  # never proposed or approved
    assert "uc3:no-change-without-rm-approval" in failed(quality.uc3_checks(run))
    run = copy.deepcopy(runs["post_meeting"])
    run.state["meeting_record"] = run.state["meeting_record"].model_copy(update={"agreement_changed": True})
    assert "uc3:no-change-without-rm-approval" in failed(quality.uc3_checks(run))


# ------------------------------------------------------------------ workflow level

def test_trajectory_tokens_follow_the_deck_order(runs):
    tokens = trajectory.observed(runs["pre_meeting"].log)
    assert tokens[:4] == ["agent:client_profile", "cp:client_profile", "agent:client_limits", "cp:client_limits"]
    assert tokens[-2:] == ["pdf", "end:APPROVED"]


def test_routing_catches_a_wrong_order_and_a_wrong_agent(runs):
    run = copy.deepcopy(runs["pre_meeting"])
    i, j = [k for k, e in enumerate(run.log) if e["event"] == "agent_output"][1:3]  # swap limits and market intelligence
    run.log[i], run.log[j] = run.log[j], run.log[i]
    assert "workflow:expected-order" in failed(trajectory.routing(run))
    uc3 = copy.deepcopy(runs["post_meeting"])
    uc3.log.append({"event": "agent_output", "stage": "market_intelligence"})
    assert "workflow:no-wrong-agents" in failed(trajectory.routing(uc3))


def test_selective_regeneration_is_verified_from_the_log(runs):
    run = runs["adhoc_request"]
    assert [t for t in trajectory.observed(run.log) if t.startswith("regen")] == ["regen:2"]
    bad = copy.deepcopy(run)
    rounds = [e for e in bad.log if e["event"] == "agent_output" and e["stage"] == "investment_ideas"]
    rounds[1]["output"]["ideas"][0]["thesis"] = "Changed although it passed."  # a passing slot was touched
    assert "workflow:selective-regeneration" in failed(trajectory.selective_regeneration(bad))
    rerun = copy.deepcopy(run)
    rerun.log.append({"event": "agent_output", "stage": "market_intelligence"})
    rerun.log.append({"event": "agent_output", "stage": "market_intelligence"})
    assert "workflow:no-completed-stage-rerun" in failed(trajectory.selective_regeneration(rerun))


def test_checkpoint_checks_catch_a_pdf_before_approval_and_endless_regeneration(runs):
    run = copy.deepcopy(runs["pre_meeting"])
    pdf = next(k for k, e in enumerate(run.log) if e["event"] == "pdf_generated")
    run.log.insert(0, run.log.pop(pdf))
    assert "workflow:no-pdf-before-rm-approval" in failed(trajectory.checkpoints_and_escalation(run))
    many = copy.deepcopy(runs["adhoc_request"])
    many.log += [{"event": "auto_regeneration", "slots": [2]}] * 3
    assert "workflow:regeneration-limit-r10" in failed(trajectory.checkpoints_and_escalation(many))


# ------------------------------------------------------------------ judge

def good_verdict(kind):
    return JudgeVerdict(scores={c: 4 for c in CRITERIA[kind]}, rationale="ok")


def test_llm_judge_scores_through_the_client_and_demands_every_criterion():
    reply = RUBRIC_MODELS["idea"].model_validate(good_verdict("idea").model_dump())
    llm = FakeAdvisoryLLMClient(override_responses={"IdeaVerdict": reply})
    verdict = LLMJudge(llm).judge("idea", {"thesis": "t"}, {"facts": []})
    assert verdict.scores == {"personalisation": 4, "grounding": 4, "explanation_clarity": 4}
    assert llm.call_history[-1]["agent_name"] == "OutputJudge" and "quoted data" in llm.call_history[-1]["system_prompt"]
    # every criterion is a named, required field, so a partial or empty score set never validates
    schema = RUBRIC_MODELS["idea"].model_json_schema()
    assert set(schema["$defs"]["IdeaScores"]["required"]) == set(CRITERIA["idea"])
    with pytest.raises(ValueError):
        RUBRIC_MODELS["idea"].model_validate({"scores": {"grounding": 5}, "rationale": "x"})


def test_judge_verdict_rejects_out_of_range_scores():
    with pytest.raises(ValueError):
        JudgeVerdict(scores={"grounding": 9}, rationale="x")


def test_heuristic_judge_punishes_ungrounded_ideas_and_added_advice():
    judge = HeuristicJudge()
    assert judge.judge("idea", {"cited_evidence_ids": [], "client_fit_rationale": "short"}, {"facts": []}).scores["grounding"] == 1
    clean = {"lines": [{"source_lines": [1], "verified": True}], "email": "Thanks for meeting."}
    advised = {"lines": [{"source_lines": [1], "verified": True}], "email": "I recommend you buy more bonds."}
    assert judge.judge("record", clean, {}).scores["no_added_advice"] == 5
    assert judge.judge("record", advised, {}).scores["no_added_advice"] == 1
    assert mean_score([good_verdict("idea")]) == 4.0


def test_judge_run_scores_delivered_ideas_and_records(runs):
    assert judge_run(runs["adhoc_request"], HeuristicJudge())["mean"] >= 3
    assert judge_run(runs["post_meeting"], HeuristicJudge())["judge"] == "heuristic-offline"
    assert judge_run(runs["pre_meeting"], HeuristicJudge()) == {}


# ------------------------------------------------------------------ performance and cost

def test_performance_report_breaks_latency_into_the_four_deck_buckets(runs):
    p = performance_report(runs["adhoc_request"])
    assert set(p["latency_by_bucket_ms"]) == set(BUCKETS) == {"Research", "Ideas", "Checks", "Record"}
    assert abs(sum(p["latency_by_bucket_ms"].values()) - p["end_to_end_ms"]) < 0.5
    assert abs(sum(p["latency_share"].values()) - 1.0) < 0.01 and p["llm_calls"] > 0
    assert p["cost_usd"] is None  # the offline model reports no token usage: shown as n/a, never invented


def test_selective_regeneration_saves_idea_generations(runs):
    p = performance_report(runs["adhoc_request"])
    assert p["regeneration_rounds"] == 1 and p["idea_generations_saved"] == 2  # 1 of 3 regenerated, 2 kept
    assert performance_report(runs["pre_meeting"])["idea_generations_saved"] == 0


def test_cost_is_estimated_from_reported_token_usage(runs):
    run = copy.deepcopy(runs["pre_meeting"])
    for e in run.log:
        if e["event"] == "agent_output":
            e["usage"] = {"prompt_tokens": 1_000_000, "completion_tokens": 100_000}
    p = performance_report(run)
    assert p["cost_usd"] == round(3.0 * sum(1 for e in run.log if e["event"] == "agent_output") * 1 + 15.0 * 0.1 * sum(1 for e in run.log if e["event"] == "agent_output"), 4)


def test_evaluate_run_bundles_levels_judge_performance_and_trajectory(runs):
    out = evaluate_run(runs["pre_meeting"], HeuristicJudge())
    assert set(out["levels"]) == {"schema", "deterministic", "competency", "trajectory"} and out["label"] == "UC1 Prepare" and out["trajectory"]
    assert out["performance"]["unapproved_actions"] == 0 and out["performance"]["manual_recovery_actions"] == []


def test_unapproved_action_rate_catches_a_pdf_without_rm_approval(runs):
    from src.evaluation.performance import consequential_actions, operational_summary
    run = runs["pre_meeting"]
    seeded = dataclasses.replace(run, log=[e for e in run.log if not (e["event"] == "advisor_decision" and e["stage"] == "brief_approval")])
    assert any(not a["approved"] for a in consequential_actions(seeded.log))
    assert operational_summary([seeded])["unapproved_action_rate_pct"] > 0
