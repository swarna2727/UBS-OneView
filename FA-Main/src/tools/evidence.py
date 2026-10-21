"""Evidence integrity and audit verification tool (Slide 17)."""

from typing import List, Any
from src.schemas.tools import EvidenceStatus

def evidence_integrity_check(candidate_ideas: List[Any], research_brief: Any) -> EvidenceStatus:
    """
    Deterministically verify that every candidate investment idea has valid,
    fresh, and verified evidence IDs matching the ResearchBrief.
    """
    valid_source_ids = set()
    stale_source_ids = set()

    # Collect source IDs from research brief
    if hasattr(research_brief, "validated_sources"):
        for src in research_brief.validated_sources:
            if getattr(src, "is_fresh", True):
                valid_source_ids.add(getattr(src, "source_id", ""))
            else:
                stale_source_ids.add(getattr(src, "source_id", ""))
    elif isinstance(research_brief, dict):
        for src in research_brief.get("validated_sources", []):
            if src.get("is_fresh", True):
                valid_source_ids.add(src.get("source_id", ""))
            else:
                stale_source_ids.add(src.get("source_id", ""))

    missing_sources = set()
    unsupported_ideas = []

    for idea in candidate_ideas:
        idea_id = getattr(idea, "idea_id", "") if hasattr(idea, "idea_id") else idea.get("idea_id", "")
        ticker = getattr(idea, "ticker", "") if hasattr(idea, "ticker") else idea.get("ticker", "")
        evidence_ids = getattr(idea, "cited_evidence_ids", []) if hasattr(idea, "cited_evidence_ids") else idea.get("cited_evidence_ids", [])

        if not evidence_ids:
            unsupported_ideas.append(f"{ticker} ({idea_id}): No evidence IDs provided.")
            continue

        idea_has_valid = False
        for eid in evidence_ids:
            if eid in valid_source_ids:
                idea_has_valid = True
            elif eid in stale_source_ids:
                unsupported_ideas.append(f"{ticker} ({idea_id}): Cites stale source {eid}.")
            else:
                missing_sources.add(eid)
                unsupported_ideas.append(f"{ticker} ({idea_id}): Cites unretrieved source {eid}.")

        if not idea_has_valid:
            unsupported_ideas.append(f"{ticker} ({idea_id}): Lacks any valid fresh evidence.")

    all_valid = (len(unsupported_ideas) == 0 and len(missing_sources) == 0)

    return EvidenceStatus(
        all_evidence_valid=all_valid,
        valid_source_ids=list(valid_source_ids),
        missing_source_ids=list(missing_sources),
        stale_source_ids=list(stale_source_ids),
        unsupported_ideas=list(set(unsupported_ideas))
    )
