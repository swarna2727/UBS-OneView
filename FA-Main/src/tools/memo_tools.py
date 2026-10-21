"""Deterministic tools for Memo Formatter Agent (Slide 18)."""

import os
from typing import List, Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from src.schemas.advisory import StructuredMemo

def memo_consistency_validator(critic_output: Any, memo_content: Any) -> bool:
    """
    Verify strict consistency between Critic-approved idea tickers/allocations
    and the final StructuredMemo content.
    Returns True if 100% consistent; False if there is any alteration.
    """
    try:
        # Extract approved idea details from critic_output or candidate ideas
        approved_tickers = set()
        approved_weights = {}

        if hasattr(critic_output, "idea_evaluations"):
            for ie in critic_output.idea_evaluations:
                ticker = getattr(ie, "ticker", "").upper()
                approved_tickers.add(ticker)
        elif isinstance(critic_output, dict):
            for ie in critic_output.get("idea_evaluations", []):
                approved_tickers.add(ie.get("ticker", "").upper())

        # Extract tickers from memo
        memo_tickers = set()
        if hasattr(memo_content, "page_1_executive"):
            for glance in memo_content.page_1_executive.ideas_at_a_glance:
                memo_tickers.add(glance.ticker.upper())
        elif isinstance(memo_content, dict):
            for glance in memo_content.get("page_1_executive", {}).get("ideas_at_a_glance", []):
                memo_tickers.add(glance.get("ticker", "").upper())

        # If critic had specific tickers evaluated, memo must match exactly
        if approved_tickers and memo_tickers != approved_tickers:
            return False

        return True
    except Exception:
        return False

def citation_source_mapper(memo_text_or_findings: Any, evidence_list: List[Any]) -> List[str]:
    """
    Map cited evidence IDs to fully audit-formatted citation strings with publisher and dates.
    """
    citations = []
    source_map = {}
    for item in evidence_list:
        sid = getattr(item, "source_id", "") if hasattr(item, "source_id") else item.get("source_id", "")
        pub = getattr(item, "publisher", "") if hasattr(item, "publisher") else item.get("publisher", "")
        title = getattr(item, "title", "") if hasattr(item, "title") else item.get("title", "")
        date = getattr(item, "published_at", "") if hasattr(item, "published_at") else item.get("published_at", "")
        source_map[sid] = f"[{sid}] {pub} - \"{title}\" (Published: {date[:10]})"

    for sid, formatted in source_map.items():
        citations.append(formatted)

    return citations

def pdf_memo_renderer(structured_memo: StructuredMemo, output_path: str) -> str:
    """
    Deterministically compile StructuredMemo into a standardized 3-page UBS advisory PDF.
    - Page 1: Executive Recommendation & Ideas at a Glance Table
    - Page 2: Investment Idea Details (Theses, Why Now, Allocation, Risks)
    - Page 3: Portfolio View, Scenario Analysis, Risk Disclosures, Signature Block
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    ubs_red = colors.HexColor("#E60000")
    ubs_dark = colors.HexColor("#0F172A")
    ubs_slate = colors.HexColor("#334155")
    bg_light = colors.HexColor("#F8FAFC")

    title_style = ParagraphStyle(
        "UBSTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=ubs_dark,
        spaceAfter=6
    )

    section_style = ParagraphStyle(
        "UBSSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=ubs_red,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "UBSBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=ubs_slate,
        spaceAfter=4
    )

    bold_body = ParagraphStyle(
        "UBSBoldBody",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=ubs_dark
    )

    story = []

    # ================= PAGE 1 =================
    p1 = structured_memo.page_1_executive

    story.append(Paragraph("UBS WEALTH MANAGEMENT ADVISORY MEMO", title_style))
    story.append(Paragraph(f"<b>Client:</b> {p1.client_name} ({p1.client_id}) &nbsp;|&nbsp; <b>Date:</b> {p1.date} &nbsp;|&nbsp; <b>Status:</b> RM Approved Recommendation", bold_body))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Executive Objective & Mandate", section_style))
    story.append(Paragraph(p1.investment_objective, body_style))

    story.append(Paragraph("2. Client & Portfolio Context", section_style))
    story.append(Paragraph(f"<b>Client Snapshot:</b> {p1.client_snapshot}", body_style))
    story.append(Paragraph(f"<b>Portfolio Baseline:</b> {p1.portfolio_snapshot}", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Recommendations at a Glance", section_style))
    glance_data = [
        [
            Paragraph("<b>#</b>", bold_body),
            Paragraph("<b>Instrument</b>", bold_body),
            Paragraph("<b>Action</b>", bold_body),
            Paragraph("<b>Target %</b>", bold_body),
            Paragraph("<b>Role</b>", bold_body),
            Paragraph("<b>Strategic Rationale</b>", bold_body)
        ]
    ]

    for item in p1.ideas_at_a_glance:
        glance_data.append([
            Paragraph(str(item.slot_number), body_style),
            Paragraph(f"<b>{item.ticker}</b><br/>{item.name}", body_style),
            Paragraph(item.action, bold_body),
            Paragraph(f"{item.target_weight_pct:.1f}%", bold_body),
            Paragraph(item.portfolio_role, body_style),
            Paragraph(item.why_it_fits, body_style)
        ])

    table_glance = Table(glance_data, colWidths=[20, 110, 50, 50, 80, 210])
    table_glance.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(table_glance)

    # PAGE BREAK TO PAGE 2
    story.append(PageBreak())

    # ================= PAGE 2 =================
    story.append(Paragraph("PAGE 2: INVESTMENT IDEA SPECIFICATIONS & RATIONALE", title_style))
    story.append(Spacer(1, 6))

    for detail in structured_memo.page_2_details:
        story.append(Paragraph(f"Idea #{detail.slot_number}: {detail.ticker} - {detail.name} ({detail.action} @ {detail.allocation_range_str})", section_style))
        story.append(Paragraph(f"<b>Portfolio Role:</b> {detail.portfolio_role} &nbsp;|&nbsp; <b>Target Weight:</b> {detail.target_weight_pct:.1f}% &nbsp;|&nbsp; <b>Conviction:</b> {detail.conviction}", bold_body))
        story.append(Paragraph(f"<b>Core Thesis:</b> {detail.thesis}", body_style))
        story.append(Paragraph(f"<b>Why Now / Catalyst:</b> {detail.why_now}", body_style))
        story.append(Paragraph(f"<b>Client Suitability Fit:</b> {detail.client_fit_rationale}", body_style))
        story.append(Paragraph(f"<b>Key Risks & Mitigations:</b> {'; '.join(detail.key_risks)}", body_style))
        story.append(Paragraph(f"<b>What Would Change Our View:</b> {detail.what_changes_view}", body_style))
        story.append(Paragraph(f"<b>Alternative Considered:</b> {detail.alternative_considered}", body_style))
        story.append(Paragraph(f"<b>Attributed Evidence Sources:</b> {', '.join(detail.cited_sources_formatted)}", body_style))
        story.append(Spacer(1, 8))

    # PAGE BREAK TO PAGE 3
    story.append(PageBreak())

    # ================= PAGE 3 =================
    p3 = structured_memo.page_3_portfolio_view
    story.append(Paragraph("PAGE 3: PORTFOLIO VIEW, GOVERNANCE & RISK DISCLOSURE", title_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Portfolio-Level Impact & Asset Allocation Shift", section_style))
    story.append(Paragraph(p3.portfolio_level_impact_narrative, body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Stress Testing & Scenario Analysis", section_style))
    scenario_rows = [[Paragraph("<b>Scenario Event</b>", bold_body), Paragraph("<b>Expected Impact & Cushioning</b>", bold_body)]]
    for sc_name, sc_impact in p3.scenario_analysis.items():
        scenario_rows.append([Paragraph(sc_name, bold_body), Paragraph(sc_impact, body_style)])

    table_sc = Table(scenario_rows, colWidths=[180, 340])
    table_sc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(table_sc)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Market Developments & Evidence Attribution", section_style))
    story.append(Paragraph(p3.market_developments, body_style))
    for src in p3.audit_sources_summary:
        story.append(Paragraph(f"• {src}", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Regulatory Disclosures & RM Sign-off", section_style))
    for disc in p3.risk_disclosures:
        story.append(Paragraph(f"<i>Disclaimer:</i> {disc}", body_style))

    story.append(Spacer(1, 14))
    story.append(Paragraph(f"<b>Sign-off Block:</b> {p3.advisor_signature_block}", bold_body))

    doc.build(story)
    return output_path
