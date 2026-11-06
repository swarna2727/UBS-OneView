"""Deterministic market signal conflict detector (deck slide 29) and its use by the agent and the Critic."""

import copy
import random
import pytest
from src.schemas.tools import DocumentSnippet
from src.tools.market_tools import detect_market_signal_conflicts, conflicting_signals_detector, sentiment_score

BULL = "UBS overweight stance with attractive yield and compelling upside."
BEAR = "Analysts see downside and pressure as a slowdown and recession headwind builds."


def doc(source_id, text, tickers=("IEF",), published="2026-10-10T00:00:00Z", status="active", asset_classes=(), themes=(), publisher=None):
    return DocumentSnippet(source_id=source_id, title=source_id, publisher=publisher or source_id, published_at=published,
                           retrieved_at="2026-10-18T00:00:00Z", text=text, tickers=list(tickers), asset_classes=list(asset_classes),
                           macro_themes=list(themes), status=status)


def test_sentiment_score_is_a_quantitative_value_in_range():
    assert sentiment_score(BULL) == 1.0 and sentiment_score(BEAR) == -1.0
    assert sentiment_score("Neutral macro commentary.") == 0.0
    assert -1.0 <= sentiment_score("overweight but downside") <= 1.0


def test_bullish_vs_bearish_on_the_same_ticker_is_a_structured_conflict():
    [c] = detect_market_signal_conflicts([doc("A", BULL, published="2026-10-10T00:00:00Z"), doc("B", BEAR, published="2026-10-18T00:00:00Z")])
    assert (c.code, c.dimension, c.key, c.severity) == ("MARKET_SIGNAL_CONFLICT", "ticker", "IEF", "FLAG")
    assert c.bullish_source_ids == ["A"] and c.bearish_source_ids == ["B"]
    assert c.bullish_score == 1.0 and c.bearish_score == -1.0 and c.divergence == 2.0 and c.gap_days == 8
    assert "MARKET_SIGNAL_CONFLICT" in c.message and "A" in c.message and "B" in c.message


def test_agreeing_or_neutral_sources_do_not_conflict():
    assert detect_market_signal_conflicts([doc("A", BULL), doc("B", BULL)]) == []
    assert detect_market_signal_conflicts([doc("A", BEAR), doc("B", BEAR)]) == []
    assert detect_market_signal_conflicts([doc("A", BULL), doc("B", "Neutral macro commentary.")]) == []


def test_readings_more_than_the_window_apart_do_not_conflict():
    far = [doc("A", BULL, published="2026-08-01T00:00:00Z"), doc("B", BEAR, published="2026-10-10T00:00:00Z")]
    assert detect_market_signal_conflicts(far) == []
    near = [doc("A", BULL, published="2026-09-12T00:00:00Z"), doc("B", BEAR, published="2026-10-10T00:00:00Z")]  # 28 days
    assert len(detect_market_signal_conflicts(near)) == 1


def test_a_small_divergence_below_the_threshold_is_not_a_conflict():
    mild_bull = "overweight attractive but downside"       # 2 bull, 1 bear -> +0.33
    mild_bear = "underweight recession headwind but upside"  # 1 bull, 3 bear -> -0.5; divergence 0.83
    assert len(detect_market_signal_conflicts([doc("A", mild_bull), doc("B", mild_bear)])) == 1
    weaker_bear = "underweight but attractive growth"       # 3 bull, 1 bear -> +0.5: same side as A
    assert detect_market_signal_conflicts([doc("A", mild_bull), doc("B", weaker_bear)]) == []


def test_withdrawn_and_superseded_sources_never_create_a_conflict():
    assert detect_market_signal_conflicts([doc("A", BULL), doc("B", BEAR, status="withdrawn")]) == []
    assert detect_market_signal_conflicts([doc("A", BULL), doc("B", BEAR, status="superseded")]) == []


def test_conflicts_are_found_by_sector_and_macro_theme_too():
    docs = [doc("A", BULL, tickers=("X",), asset_classes=("Fixed Income",), themes=("Quality Yield",)),
            doc("B", BEAR, tickers=("Y",), asset_classes=("Fixed Income",), themes=("Quality Yield",))]
    found = {(c.dimension, c.key) for c in detect_market_signal_conflicts(docs)}
    assert found == {("sector", "Fixed Income"), ("macro_theme", "Quality Yield")}  # different tickers: no ticker conflict


def test_detection_is_deterministic_and_order_independent():
    docs = [doc("A", BULL), doc("B", BEAR), doc("C", BULL, tickers=("LQD",)), doc("D", BEAR, tickers=("LQD",))]
    first = detect_market_signal_conflicts(docs)
    for seed in range(5):
        shuffled = copy.deepcopy(docs)
        random.Random(seed).shuffle(shuffled)
        assert detect_market_signal_conflicts(shuffled) == first
    assert [c.key for c in first] == ["IEF", "LQD"]


def test_legacy_message_wrapper_matches_the_structured_conflicts():
    docs = [doc("A", BULL), doc("B", BEAR)]
    assert conflicting_signals_detector(docs) == [c.message for c in detect_market_signal_conflicts(docs)]


# ------------------------------------------------------------------ agent and Critic integration

NEWS = [{"title": "Treasury rally fades", "source": "Reuters", "published_at": "2026-10-18T08:00:00Z", "tickers": ["IEF"],
         "text": "Strategists warn of downside for intermediate Treasuries as a growth slowdown fails to arrive and supply pressure builds."}]


NEWS_VCSH = [{"title": "Short-duration credit loses momentum", "source": "Reuters", "published_at": "2026-10-18T08:00:00Z", "tickers": ["VCSH"],
              "text": "Strategists see downside for short-duration investment grade credit as spread pressure builds and a recession looms."}]


def test_research_brief_carries_structured_conflicts_and_the_model_is_told_not_to_resolve_them(sample_profiles):
    from src.agents.client_profile import ClientProfileAgent
    from src.agents.market_intelligence import MarketIntelligenceAgent
    from src.llm.fake import FakeAdvisoryLLMClient
    llm = FakeAdvisoryLLMClient()
    profile = ClientProfileAgent(llm).run(copy.deepcopy(sample_profiles["demo_sarah"]))
    brief = MarketIntelligenceAgent(llm).run(profile, rm_news=NEWS)
    [conflict] = [c for c in brief.signal_conflicts if c.dimension == "ticker"]
    assert conflict.key == "IEF" and "RM-NEWS-01" in conflict.bearish_source_ids
    assert brief.conflicting_signals and brief.conflicting_signals[0].startswith("MARKET_SIGNAL_CONFLICT")
    call = next(c for c in llm.call_history if c["agent_name"] == "MarketIntelligenceAgent")
    assert "DETERMINISTIC MARKET SIGNAL CONFLICTS" in call["user_prompt"] and "RM-NEWS-01" in call["user_prompt"]
    assert "never pick a side" in call["user_prompt"] and "do not resolve" in call["system_prompt"]


def test_default_research_has_no_conflict_so_demo_ideas_are_not_flagged(sample_profiles):
    from src.agents.client_profile import ClientProfileAgent
    from src.agents.market_intelligence import MarketIntelligenceAgent
    from src.llm.fake import FakeAdvisoryLLMClient
    llm = FakeAdvisoryLLMClient()
    brief = MarketIntelligenceAgent(llm).run(ClientProfileAgent(llm).run(copy.deepcopy(sample_profiles["demo_sarah"])))
    assert brief.signal_conflicts == []


def test_critic_flags_an_idea_that_leans_on_conflicted_research_and_passes_the_rest(sample_profiles):
    from src.api.service import AdvisorySessionService
    from src.llm.fake import FakeAdvisoryLLMClient
    from src.schemas.workspace import AdvisorActionRequest
    svc = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    raw = {**copy.deepcopy(sample_profiles["demo_sarah"]), "use_case": "adhoc_request", "new_capital_amount": 120000.0, "rm_news": NEWS_VCSH}
    svc.start_session(raw, session_id="m1")
    while svc.get_pending_review("m1")["stage"] != "suitability_review":
        svc.execute_action("m1", AdvisorActionRequest(action="approve"))
    assessment = svc.get_state("m1")["assessment"]
    by_ticker = {ie.ticker: next(r for r in ie.rule_results if r.rule_id == "M1") for ie in assessment.idea_evaluations}
    assert by_ticker["VCSH"].outcome == "FLAG" and "MARKET_SIGNAL_CONFLICT" in by_ticker["VCSH"].message
    assert by_ticker["ACWI"].outcome == "PASS" and by_ticker["BOTZ"].outcome == "PASS"  # citing a disagreeing source elsewhere is not a conflict
    idea = next(ie for ie in assessment.idea_evaluations if ie.ticker == "VCSH")
    assert idea.status == "FLAG" and any(b.rule_id == "M1" for b in assessment.detected_breaches)
    with pytest.raises(ValueError, match="notes"):  # a flagged idea can only be locked with the RM's documented reasoning
        svc.execute_action("m1", AdvisorActionRequest(action="lock_idea", idea_id=idea.idea_id))
    svc.execute_action("m1", AdvisorActionRequest(action="lock_idea", idea_id=idea.idea_id, notes="Weighed the Reuters view against the CIO."))
