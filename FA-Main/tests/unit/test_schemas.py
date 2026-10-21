"""Unit tests for Pydantic v2 schemas and validation guardrails."""

import pytest
from pydantic import ValidationError
from src.schemas.advisory import (
    CandidateIdeaSet, InvestmentIdea, AllocationRange
)

def make_idea(slot: int, ticker: str, role: str = "GROWTH") -> InvestmentIdea:
    return InvestmentIdea(
        idea_id=f"IDEA-{ticker}-{slot}",
        slot_number=slot,
        instrument_id=f"INST-{ticker}",
        ticker=ticker,
        name=f"Asset {ticker}",
        action="BUY",
        target_weight_pct=5.0,
        allocation_range=AllocationRange(min_pct=2.0, max_pct=8.0, target_pct=5.0),
        funding_source_ticker="BIL",
        portfolio_role=role, # type: ignore
        thesis="Investment thesis",
        why_now="Catalyst now",
        client_fit_rationale="Suitability fit",
        key_risks=["Market risk"],
        invalidation_trigger="Growth stops",
        cited_evidence_ids=["SRC-01"]
    )

def test_candidate_idea_set_must_have_exactly_three_ideas():
    # Exactly 3 distinct ideas -> valid
    valid_set = CandidateIdeaSet(
        ideas=[
            make_idea(1, "IEF", "INCOME"),
            make_idea(2, "NESN", "STABILITY"),
            make_idea(3, "MSFT", "GROWTH")
        ],
        complementarity_thesis="Cohesive diversified set"
    )
    assert len(valid_set.ideas) == 3

    # Only 2 ideas -> must raise ValidationError
    with pytest.raises(ValidationError) as exc:
        CandidateIdeaSet(
            ideas=[
                make_idea(1, "IEF", "INCOME"),
                make_idea(2, "NESN", "STABILITY")
            ],
            complementarity_thesis="Incomplete set"
        )
    assert "CandidateIdeaSet must contain exactly 3 ideas" in str(exc.value)

    # 4 ideas -> must raise ValidationError
    with pytest.raises(ValidationError) as exc:
        CandidateIdeaSet(
            ideas=[
                make_idea(1, "IEF", "INCOME"),
                make_idea(2, "NESN", "STABILITY"),
                make_idea(3, "MSFT", "GROWTH"),
                make_idea(3, "NVDA", "GROWTH")
            ],
            complementarity_thesis="Too many ideas"
        )
    assert "CandidateIdeaSet must contain exactly 3 ideas" in str(exc.value)

def test_candidate_idea_set_requires_slots_1_2_3():
    with pytest.raises(ValidationError) as exc:
        CandidateIdeaSet(
            ideas=[
                make_idea(1, "IEF", "INCOME"),
                make_idea(1, "NESN", "STABILITY"), # duplicate slot 1
                make_idea(3, "MSFT", "GROWTH")
            ],
            complementarity_thesis="Invalid slots"
        )
    assert "slots 1, 2, and 3" in str(exc.value)

def test_candidate_idea_set_requires_distinct_instruments():
    with pytest.raises(ValidationError) as exc:
        CandidateIdeaSet(
            ideas=[
                make_idea(1, "MSFT", "INCOME"),
                make_idea(2, "MSFT", "STABILITY"), # duplicate ticker
                make_idea(3, "IEF", "GROWTH")
            ],
            complementarity_thesis="Duplicate tickers"
        )
    assert "distinct instruments" in str(exc.value)

def test_allocation_range_bounds():
    # Valid range
    ar = AllocationRange(min_pct=2.0, max_pct=6.0, target_pct=4.0)
    assert ar.target_pct == 4.0

    # min > max
    with pytest.raises(ValidationError):
        AllocationRange(min_pct=6.0, max_pct=2.0, target_pct=4.0)

    # target outside [min, max]
    with pytest.raises(ValidationError):
        AllocationRange(min_pct=2.0, max_pct=6.0, target_pct=8.0)
