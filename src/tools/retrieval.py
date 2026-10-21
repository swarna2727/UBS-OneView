"""Retrieval and source freshness validation tools (Slide 15)."""

import os
import json
from datetime import datetime, timezone
from typing import List, Optional
from src.schemas.tools import DocumentSnippet, ValidatedSnippets

DATA_CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "research", "corpus.json"
)

def retrieval_rag_tool(query: str, asset_universe: Optional[List[str]] = None) -> List[DocumentSnippet]:
    """
    Retrieve document snippets from research corpus filtered by query terms
    and relevant asset universe tickers/asset classes.
    """
    if not os.path.exists(DATA_CORPUS_PATH):
        return []

    with open(DATA_CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    query_tokens = set(query.lower().replace(",", " ").replace(";", " ").split())
    results: List[DocumentSnippet] = []

    for item in corpus:
        text_lower = item.get("text", "").lower()
        title_lower = item.get("title", "").lower()
        tickers = [t.upper() for t in item.get("tickers", [])]
        themes = [th.lower() for th in item.get("macro_themes", [])]

        score = 0
        for token in query_tokens:
            if token in text_lower:
                score += 1
            if token in title_lower:
                score += 2
            if any(token == th for th in themes):
                score += 3

        # Universe relevance bonus
        if asset_universe:
            universe_set = set(t.upper() for t in asset_universe)
            if set(tickers).intersection(universe_set):
                score += 2

        if score > 0 or not query_tokens:
            snippet = DocumentSnippet(
                source_id=item["source_id"],
                title=item["title"],
                publisher=item["publisher"],
                published_at=item["published_at"],
                retrieved_at=item["retrieved_at"],
                text=item["text"],
                tickers=tickers,
                asset_classes=item.get("asset_classes", []),
                macro_themes=item.get("macro_themes", []),
                is_fresh=True
            )
            results.append((score, snippet))

    # Sort descending by relevance score
    results.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in results]

def source_freshness_validator(
    snippets: List[DocumentSnippet], 
    max_age_days: int = 45,
    reference_date_iso: Optional[str] = "2026-10-21T00:00:00Z"
) -> ValidatedSnippets:
    """
    Validate that research snippets are not stale (e.g. older than max_age_days).
    Every market-dependent claim requires explicit fresh evidence.
    """
    if reference_date_iso:
        try:
            ref_dt = datetime.fromisoformat(reference_date_iso.replace("Z", "+00:00"))
        except Exception:
            ref_dt = datetime.now(timezone.utc)
    else:
        ref_dt = datetime.now(timezone.utc)

    kept: List[DocumentSnippet] = []
    discarded = []
    reasons = {}

    for s in snippets:
        try:
            pub_dt = datetime.fromisoformat(s.published_at.replace("Z", "+00:00"))
            age_days = (ref_dt - pub_dt).days
        except Exception:
            age_days = 999

        if age_days < 0:
            age_days = 0

        if age_days > max_age_days:
            reason = f"Snippet published {age_days} days ago exceeds max allowed age ({max_age_days} days)."
            discarded.append(s.model_dump())
            reasons[s.source_id] = reason
        else:
            s_dict = s.model_dump()
            s_dict["age_days"] = age_days
            s_dict["is_fresh"] = True
            kept.append(DocumentSnippet(**s_dict))

    return ValidatedSnippets(
        kept_snippets=kept,
        discarded_snippets=discarded,
        rejection_reasons=reasons
    )
