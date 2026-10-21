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
