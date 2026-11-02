"""The deck's six stress tests and the full scorecard, run end to end on the offline model (deck slides 24-27)."""

import json
import pytest

from src.evaluation.scorecard import failures, render_markdown, run_scorecard
from src.evaluation.stress import CASES, run_stress_suite

DECK_STRESS = {
    "Missing mandatory field": "BLOCK",
    "Product blocked by domicile": "NOT SHOWN",
    "Unsupported market claim": "BLOCK",
    "PEP flag or expired CKA": "ESCALATE + LIST GAP",
    "Unclear amount in transcript": "FLAG FOR RM",
    "Spoken change in circumstances": "RM APPROVES UPDATE",
}


def test_stress_cases_match_the_deck_table():
    assert {c.name: c.expected for c in CASES} == DECK_STRESS


@pytest.mark.parametrize("result", run_stress_suite(), ids=lambda r: r["case"])
def test_each_stress_case_produces_the_required_outcome(result):
    assert result["passed"], f"{result['case']}: expected {result['expected']}, observed {result['observed']} ({result['detail']})"


def test_a_crashing_stress_case_is_a_failure_not_a_skip(monkeypatch):
    from src.evaluation import stress
    monkeypatch.setattr(stress, "CASES", [stress.StressCase("Boom", "BLOCK", lambda: 1 / 0)])
    [r] = stress.run_stress_suite()
    assert not r["passed"] and r["observed"].startswith("ERROR")


@pytest.fixture(scope="module")
def scorecard():
    return run_scorecard(include_baseline=False)


def test_scorecard_passes_all_deterministic_checks_on_all_three_use_cases(scorecard):
    assert scorecard["failures"] == [], scorecard["failures"]
    assert [r["label"] for r in scorecard["runs"]] == ["UC1 Prepare", "UC2 Discover", "UC3 Record"]
    assert all(len(r["levels"]["component"]) and len(r["levels"]["output quality"]) and len(r["levels"]["workflow"]) for r in scorecard["runs"])


def test_scorecard_reports_personalisation_and_the_regeneration_saving(scorecard):
    pers = scorecard["personalization"]
    # The changed client must either get a different idea set, or be stopped by the controls: never the same advice.
    assert (pers["distance"] or 0) > 0 or pers.get("responded", "").startswith("blocked")
    uc2 = next(r for r in scorecard["runs"] if r["use_case"] == "adhoc_request")
    assert uc2["performance"]["idea_generations_saved"] == 2


def test_markdown_report_has_every_deck_section(scorecard):
    md = render_markdown(scorecard, offline=True)
    for heading in ("Component level", "Output Quality level", "Workflow level", "Stress tests", "Performance and cost",
                    "Personalisation", "ALL DETERMINISTIC CHECKS PASS", "Offline run"):
        assert heading in md
    for name, outcome in DECK_STRESS.items():
        assert f"| {name} | {outcome} | {outcome} | PASS |" in md
    for bucket in ("Research", "Ideas", "Checks", "Record"):
        assert bucket in md


def test_a_failing_check_is_reported_and_would_fail_the_run(scorecard):
    bad = json.loads(json.dumps(scorecard, default=str))
    bad["runs"][0]["levels"]["workflow"][0]["passed"] = False
    bad["stress"][0]["passed"] = False
    found = failures(bad)
    assert len(found) == 2 and any("Stress /" in f for f in found)
    assert "2 FAILURE(S)" in render_markdown({**bad, "failures": found, "baseline": None} if False else {**bad, "failures": found}, offline=True)


def test_scorecard_includes_the_specialist_vs_single_llm_hypothesis():
    result = run_scorecard(include_baseline=True)
    assert set(result["baseline"]) == {"single_llm", "multi_agent"}
    assert "specialists vs one financial-advisor LLM" in render_markdown(result, offline=True)


def test_cli_exit_code_reflects_the_deterministic_checks(tmp_path, monkeypatch):
    import sys
    from src.evaluation import scorecard as sc
    monkeypatch.setattr(sys, "argv", ["scorecard", "--out", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json"), "--no-baseline"])
    with pytest.raises(SystemExit) as exit_info:
        sc.main()
    assert exit_info.value.code == 0 and (tmp_path / "r.md").exists()
    assert json.loads((tmp_path / "r.json").read_text())["failures"] == []
