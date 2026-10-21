"""Unit tests for memo consistency validation and PDF rendering."""

import os
import pytest
from src.tools.memo_tools import memo_consistency_validator, citation_source_mapper, pdf_memo_renderer
from src.schemas.advisory import (
    IdeaStatus, StructuredMemo, MemoPage1Executive, MemoPage2IdeaDetail,
    MemoPage3PortfolioView, ExecutiveIdeaGlance
)
from src.schemas.tools import RiskMetrics

def test_memo_consistency_validator():
    # Critic approved tickers: IEF, NESN, MSFT
    critic_output = {
        "idea_evaluations": [
            {"ticker": "IEF", "status": "PASS"},
            {"ticker": "NESN", "status": "PASS"},
            {"ticker": "MSFT", "status": "PASS"}
        ]
    }

    # Consistent memo
    valid_memo = {
        "page_1_executive": {
            "ideas_at_a_glance": [
                {"ticker": "IEF"},
                {"ticker": "NESN"},
                {"ticker": "MSFT"}
            ]
        }
    }
    assert memo_consistency_validator(critic_output, valid_memo)

    # Inconsistent memo (altered ticker TSLA introduced)
    tampered_memo = {
        "page_1_executive": {
            "ideas_at_a_glance": [
                {"ticker": "IEF"},
                {"ticker": "NESN"},
                {"ticker": "TSLA"} # Alteration!
            ]
        }
    }
    assert not memo_consistency_validator(critic_output, tampered_memo)

def test_citation_source_mapper():
    evidence_list = [
        {"source_id": "SRC-01", "publisher": "UBS CIO", "title": "Macro Q4", "published_at": "2026-10-10T00:00:00Z"}
    ]
    citations = citation_source_mapper("dummy", evidence_list)
    assert len(citations) == 1
    assert "[SRC-01] UBS CIO" in citations[0]

def test_pdf_memo_renderer(tmp_path):
    memo = StructuredMemo(
        memo_id="TEST-MEMO-01",
        created_at="2026-10-21T00:00:00Z",
        page_1_executive=MemoPage1Executive(
            client_name="Test Client",
            client_id="CL-TEST",
            date="21-Oct-2026",
            investment_objective="Balanced Growth",
            client_snapshot="Moderate risk",
            portfolio_snapshot="Balanced",
            ideas_at_a_glance=[
                ExecutiveIdeaGlance(idea_id="1", slot_number=1, ticker="IEF", name="Treasury", action="ADD", target_weight_pct=5.0, portfolio_role="INCOME", why_it_fits="Yield"),
                ExecutiveIdeaGlance(idea_id="2", slot_number=2, ticker="NESN", name="Nestle", action="ADD", target_weight_pct=5.0, portfolio_role="STABILITY", why_it_fits="Defensive"),
                ExecutiveIdeaGlance(idea_id="3", slot_number=3, ticker="MSFT", name="Microsoft", action="BUY", target_weight_pct=5.0, portfolio_role="GROWTH", why_it_fits="Tech")
            ]
        ),
        page_2_details=[
            MemoPage2IdeaDetail(slot_number=i, ticker="TK", name="Name", action="ADD", allocation_range_str="3-7%", target_weight_pct=5.0, portfolio_role="ROLE", thesis="Thesis", why_now="Now", client_fit_rationale="Fit", key_risks=["Risk"], what_changes_view="Trigger", cited_sources_formatted=["SRC-01"])
            for i in [1, 2, 3]
        ],
        page_3_portfolio_view=MemoPage3PortfolioView(
            portfolio_level_impact_narrative="Balanced shift",
            scenario_analysis={"Shock": "Cushioned"},
            risk_disclosures=["Risk disclosure"],
            market_developments="Macro developments",
            audit_sources_summary=["Audit source"],
            advisor_signature_block="Signed by Advisor"
        ),
        consistency_hash="HASH"
    )

    pdf_file = os.path.join(str(tmp_path), "test_memo.pdf")
    out_path = pdf_memo_renderer(memo, pdf_file)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000
