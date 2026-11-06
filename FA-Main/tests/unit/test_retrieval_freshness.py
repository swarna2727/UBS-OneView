"""Unit tests for evidence retrieval and freshness validation."""

import pytest
from src.tools.retrieval import source_freshness_validator, retrieval_rag_tool
from src.schemas.tools import DocumentSnippet

def test_source_freshness_filtering():
    snippets = [
        DocumentSnippet(
            source_id="FRESH-01",
            title="Fresh Macro Strategy Note",
            publisher="UBS CIO",
            published_at="2026-10-15T00:00:00Z",
            retrieved_at="2026-10-18T00:00:00Z",
            text="Recent monetary policy commentary."
        ),
        DocumentSnippet(
            source_id="STALE-01",
            title="Old Market Note",
            publisher="Legacy Press",
            published_at="2025-01-01T00:00:00Z", # ~650 days old
            retrieved_at="2025-01-02T00:00:00Z",
            text="Old speculation."
        )
    ]

    res = source_freshness_validator(
        snippets,
        max_age_days=45,
        reference_date_iso="2026-10-21T00:00:00Z"
    )

    assert len(res.kept_snippets) == 1
    assert res.kept_snippets[0].source_id == "FRESH-01"
    assert len(res.discarded_snippets) == 1
    assert "STALE-01" in res.rejection_reasons

def test_retrieval_rag_tool():
    results = retrieval_rag_tool("Fixed Income duration", asset_universe=["IEF", "LQD"])
    assert len(results) > 0
    # Top result should mention fixed income or duration
    assert any("fixed income" in r.text.lower() or "duration" in r.text.lower() for r in results)


# ------------------------------------------------------------------ document lifecycle status (deck slide 26)

def snippet(source_id, status="active", published="2026-10-15T00:00:00Z", superseded_by=None):
    return DocumentSnippet(source_id=source_id, title=source_id, publisher="Pub", published_at=published,
                           retrieved_at="2026-10-18T00:00:00Z", text="Commentary.", status=status, superseded_by=superseded_by)

def validate(snippets, max_age_days=30):
    return source_freshness_validator(snippets, max_age_days=max_age_days, reference_date_iso="2026-10-21T00:00:00Z")

def test_status_defaults_to_active_and_rejects_unknown_values():
    assert snippet("A").status == "active"
    with pytest.raises(ValueError):
        snippet("A", status="deleted")

def test_withdrawn_and_superseded_are_dropped_even_when_recent():
    res = validate([snippet("OK"), snippet("WD", "withdrawn"), snippet("SS", "superseded", superseded_by="OK")])
    assert [s.source_id for s in res.kept_snippets] == ["OK"]
    assert res.rejection_reasons["WD"] == "Withdrawn research: the publisher has retracted this note."
    assert res.rejection_reasons["SS"] == "Superseded research: replaced by OK."

def test_dropped_snippets_record_category_and_explicit_reason():
    res = validate([snippet("WD", "withdrawn"), snippet("SS", "superseded"), snippet("OLD", published="2025-01-01T00:00:00Z")])
    assert res.dropped_snippets is res.discarded_snippets
    by_id = {d["source_id"]: d for d in res.dropped_snippets}
    assert by_id["WD"]["drop_category"] == "withdrawn" and by_id["WD"]["drop_reason"].startswith("Withdrawn research")
    assert by_id["SS"]["drop_category"] == "superseded" and by_id["SS"]["drop_reason"] == "Superseded research: a newer note replaced it."
    assert by_id["OLD"]["drop_category"] == "stale" and by_id["OLD"]["drop_reason"].startswith("Age > 30d")

def test_status_wins_over_age_for_an_old_withdrawn_note():
    res = validate([snippet("WD-OLD", "withdrawn", published="2025-01-01T00:00:00Z")])
    assert res.dropped_snippets[0]["drop_category"] == "withdrawn"

def test_freshness_validator_is_also_exposed_from_market_tools():
    from src.tools import market_tools
    assert market_tools.source_freshness_validator is source_freshness_validator

def test_corpus_carries_superseded_and_withdrawn_examples():
    corpus = {r.source_id: r for r in retrieval_rag_tool("Fixed Income European credit IEF IEAC", asset_universe=["IEF", "IEAC"])}
    assert corpus["SRC-UBS-CIO-2026-099"].status == "superseded" and corpus["SRC-UBS-CIO-2026-099"].superseded_by == "SRC-UBS-CIO-2026-101"
    assert corpus["SRC-BROKER-NOTE-2026-10"].status == "withdrawn"
    assert corpus["SRC-UBS-CIO-2026-101"].status == "active"

def test_the_research_brief_excludes_and_displays_lifecycle_status(balanced_client_data):
    from src.agents.client_profile import ClientProfileAgent
    from src.agents.market_intelligence import MarketIntelligenceAgent
    from src.llm.fake import FakeAdvisoryLLMClient
    llm = FakeAdvisoryLLMClient()
    brief = MarketIntelligenceAgent(llm).run(ClientProfileAgent(llm).run(balanced_client_data))
    used = {s.source_id for s in brief.validated_sources}
    excluded = {e.source_id: e for e in brief.excluded_sources}
    assert not used & {"SRC-UBS-CIO-2026-099", "SRC-BROKER-NOTE-2026-10", "SRC-STALE-ARCHIVE-2025-01"}
    assert excluded["SRC-UBS-CIO-2026-099"].category == "superseded" and "SRC-UBS-CIO-2026-101" in excluded["SRC-UBS-CIO-2026-099"].reason
    assert excluded["SRC-BROKER-NOTE-2026-10"].category == "withdrawn" and excluded["SRC-STALE-ARCHIVE-2025-01"].category == "stale"
    assert all(s.status == "active" for s in brief.validated_sources)
    assert all("SRC-UBS-CIO-2026-099" not in call["user_prompt"].split("VERIFIED FRESH EVIDENCE SNIPPETS")[1] for call in llm.call_history
               if call["agent_name"] == "MarketIntelligenceAgent")  # a withdrawn or superseded note never reaches the model as evidence
