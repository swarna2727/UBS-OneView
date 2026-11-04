"""UC2 investment-opportunities memo, laid out like a research note (Brief & Record Agent's renderer).

Page 1: title block, Figure 1 (the three ideas at a glance), a short summary and one bullet per idea.
Page 2: portfolio view, figures in pairs with a source line, key takeaways in the right margin.
Then: one block per idea (why, CIO view, risks, fit and eligibility), sources as linked titles, RM sign-off.

Deterministic: it formats approved content only and never changes ideas, allocations or findings. Internal source ids
never appear: sources are shown as title, publisher and date. Prototype output, not investment research.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.tools.memo_tools import readable_citations, source_index
from src.tools.pdf_charts import memo_figures
from src.tools.risk_engine import risk_band_points, risk_points

RED, INK, MUTED, RULE = colors.HexColor("#C8102E"), colors.HexColor("#1A1A1A"), colors.HexColor("#6B6B6B"), colors.HexColor("#BDBDBD")
BLUE, TAG_BG = colors.HexColor("#1C5CAB"), colors.HexColor("#E8C9C3")
PAGE_W, PAGE_H = A4
MARGIN = 40
BODY_W = PAGE_W - 2 * MARGIN - 12  # the page frame pads 6pt on each side
MAIN_W, SIDE_W = BODY_W * 0.66, BODY_W * 0.30
GAP = BODY_W - MAIN_W - SIDE_W

STYLES = {
    "title": ParagraphStyle("NTitle", fontName="Helvetica", fontSize=22, leading=26, textColor=INK),
    "subtitle": ParagraphStyle("NSub", fontName="Helvetica", fontSize=14, leading=18, textColor=RED, spaceBefore=4),
    "h1": ParagraphStyle("NH1", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=6, spaceAfter=4),
    "h2": ParagraphStyle("NH2", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=INK, spaceBefore=4, spaceAfter=2),
    "body": ParagraphStyle("NBody", fontName="Helvetica", fontSize=9, leading=12.5, textColor=INK, spaceAfter=4),
    "bullet": ParagraphStyle("NBullet", fontName="Helvetica", fontSize=9, leading=12.5, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=3),
    "callout": ParagraphStyle("NCallout", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=RED),
    "figure": ParagraphStyle("NFig", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=INK, spaceAfter=2),
    "source": ParagraphStyle("NSrc", fontName="Helvetica", fontSize=7, leading=9, textColor=MUTED, spaceBefore=2),
    "cell": ParagraphStyle("NCell", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=INK),
    "cellb": ParagraphStyle("NCellB", fontName="Helvetica-Bold", fontSize=7.5, leading=9.5, textColor=INK),
    "side": ParagraphStyle("NSide", fontName="Helvetica", fontSize=8, leading=10.5, textColor=INK, alignment=2),
    "sideb": ParagraphStyle("NSideB", fontName="Helvetica-Bold", fontSize=9, leading=11.5, textColor=RED, alignment=2),
    "small": ParagraphStyle("NSmall", fontName="Helvetica", fontSize=7, leading=9, textColor=MUTED),
}


def _p(text: Any, style: str = "body", index: Optional[Dict[str, Any]] = None) -> Paragraph:
    return Paragraph(readable_citations(text, index), STYLES[style])


def _money(amount: float, currency: str = "SGD") -> str:
    return f"{'S$' if currency == 'SGD' else currency + ' '}{amount:,.0f}"


def _first_sentence(text: str) -> str:
    head = text.split(". ")[0].strip()
    return head if head.endswith(".") else head + "."


def _strip_label(text: str) -> str:
    for label in ("CIO view:", "CIO research:", "Why now:"):
        if text.strip().lower().startswith(label.lower()):
            return text.strip()[len(label):].strip()
    return text


def _side_row(main: List[Any], side: List[Any]) -> Table:
    """Body text on the left, key takeaways in red in the right margin (as in a research note)."""
    t = Table([[main, side]], colWidths=[MAIN_W, SIDE_W + GAP])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (0, 0), GAP), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    return t


def _figure(number: int, title: str, drawing: Any, source: str) -> List[Any]:
    return [Paragraph(f"Figure {number}: {title}", STYLES["figure"]), drawing, Paragraph(f"Source: {source}", STYLES["source"])]


def _pair(left: List[Any], right: List[Any]) -> Table:
    t = Table([[left, right]], colWidths=[BODY_W / 2, BODY_W / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return t


def _chrome(title: str, date: str):
    """Running header and footer: brand and title at the top, date, page number and the prototype notice at the bottom."""
    def draw(canvas, doc):
        canvas.saveState()
        top = PAGE_H - 30
        if doc.page == 1:
            canvas.setFont("Helvetica-Bold", 17); canvas.setFillColor(RED); canvas.drawString(MARGIN, top - 6, "UBS")
            canvas.setFillColor(INK); canvas.setFont("Helvetica", 17); canvas.drawString(MARGIN + 38, top - 6, "OneView")
            canvas.setFont("Helvetica-Bold", 8); canvas.drawRightString(PAGE_W - MARGIN, top, title)
            canvas.setFont("Helvetica", 8); canvas.drawRightString(PAGE_W - MARGIN, top - 11, date)
        else:
            canvas.setFont("Helvetica", 8); canvas.setFillColor(INK); canvas.drawString(MARGIN, top, title)
            canvas.setFillColor(RED); canvas.drawRightString(PAGE_W - MARGIN, top, "UBS OneView")
            canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5); canvas.line(MARGIN, top - 5, PAGE_W - MARGIN, top - 5)
        canvas.setStrokeColor(INK); canvas.setLineWidth(0.8); canvas.line(MARGIN, 42, PAGE_W - MARGIN, 42)
        canvas.setFillColor(INK); canvas.setFont("Helvetica-Bold", 8); canvas.drawString(MARGIN, 30, title)
        canvas.setFont("Helvetica", 8); canvas.drawString(MARGIN + canvas.stringWidth(title, "Helvetica-Bold", 8) + 6, 30, date)
        canvas.setFillColor(RED); canvas.drawRightString(PAGE_W - MARGIN, 30, f"UBS OneView  {doc.page}")
        canvas.setFillColor(MUTED); canvas.setFont("Helvetica", 6.5)
        canvas.drawString(MARGIN, 20, "Prototype output for the Relationship Manager: illustrative ideas for discussion, not investment research or a recommendation.")
        canvas.restoreState()
    return draw


def research_note_memo_renderer(memo: Any, output_path: str, profile: Any, ideas: List[Any], assessment: Any = None,
                                sources: Optional[List[Any]] = None, regeneration_history: Optional[List[Dict[str, Any]]] = None) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    index = source_index(sources) if sources else None
    p1, p3 = memo.page_1_executive, memo.page_3_portfolio_view
    details = {d.slot_number: d for d in memo.page_2_details}
    ideas = sorted(ideas, key=lambda i: i.slot_number)
    evaluations = {ie.idea_id: ie for ie in (assessment.idea_evaluations if assessment else [])}
    current = {h.ticker.upper(): h.weight_pct for h in profile.parsed_holdings}
    cur = profile.base_currency
    first = profile.name.split()[0]
    capital = profile.new_capital_amount
    date = datetime.fromisoformat(memo.created_at[:19]).strftime("%d %B %Y").lstrip("0") if memo.created_at else p1.date
    title = "Investment Opportunities"
    before, after = (assessment.before_metrics.after_risk_score, assessment.after_metrics.after_risk_score) if assessment else (None, None)
    band = risk_band_points(profile.risk_score_range)

    def fit(idea: Any) -> str:
        return evaluations[idea.idea_id].status if idea.idea_id in evaluations else "—"

    def eligible(idea: Any) -> str:
        ie = evaluations.get(idea.idea_id)
        e1 = next((r for r in (ie.rule_results if ie else []) if r.rule_id == "E1"), None)
        return "Eligible" if e1 is None or e1.outcome == "PASS" else "Not eligible"

    def amount(idea: Any) -> str:
        r = idea.indicative_amount_range or []
        return f"{_money(r[0], cur)}–{_money(r[1], cur)}" if len(r) == 2 else "—"

    story: List[Any] = [Spacer(1, 18)]

    # ---------------- Page 1: title block, Figure 1, summary ----------------
    subtitle = f"{profile.name}: three ideas" + (f" for {_money(capital, cur)} of new capital" if capital else " for discussion")
    tags = Table([[Paragraph("<font color='white'><b>DISCOVER</b></font>", STYLES["cell"])],
                  [Paragraph(f"{profile.calculated_risk_tier} · {profile.time_horizon_months // 12}-year horizon", STYLES["cell"])]],
                 colWidths=[SIDE_W])
    tags.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), RED), ("BACKGROUND", (0, 1), (0, 1), TAG_BG),
                              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    side = [tags, Spacer(1, 8), Paragraph("<b>Prepared for</b>", STYLES["side"]), Paragraph("Relationship Manager", STYLES["side"]),
            Paragraph(f"{p1.client_name} · {p1.client_id}", STYLES["side"]), Spacer(1, 4),
            Paragraph("<b>Status</b>", STYLES["side"]), Paragraph("Signed off by the RM" if memo.advisor_approved else "Draft", STYLES["side"])]
    story.append(_side_row([Paragraph(title, STYLES["title"]), Paragraph(subtitle, STYLES["subtitle"])], side))
    story.append(Spacer(1, 12))

    head = ["Idea", "Lane", "Action", "Current", "Target", "Indicative amount", "Fit", "Eligibility"]
    rows = [[Paragraph(f"<b>{h}</b>", STYLES["cellb"]) for h in head]]
    for i in ideas:
        rows.append([Paragraph(f"<b>{i.ticker}</b> {i.name}", STYLES["cell"]), Paragraph(i.lane.title(), STYLES["cell"]),
                     Paragraph(i.action, STYLES["cell"]), Paragraph(f"{current.get(i.ticker.upper(), 0.0):.1f}%", STYLES["cell"]),
                     Paragraph(f"<font color='#1C5CAB'><b>{i.target_weight_pct:.1f}%</b></font>", STYLES["cell"]),
                     Paragraph(amount(i), STYLES["cell"]), Paragraph(fit(i), STYLES["cell"]), Paragraph(eligible(i), STYLES["cell"])])
    glance = Table(rows, colWidths=[w * BODY_W / 515 for w in (150, 52, 40, 42, 40, 92, 34, 65)], repeatRows=1)
    glance.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
                                ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story += _figure(1, "Three ideas for discussion, checked for fit and eligibility", glance,
                     "OneView Opportunity Agent, Fit &amp; Eligibility Critic, client limits")
    story.append(Spacer(1, 10))

    summary = (f"Ideas in two lanes for {first}: two build on the current portfolio and one explores a new theme within "
               f"{first}'s client limits.")
    if before is not None:
        summary += f" Portfolio risk moves from {risk_points(before)} to {risk_points(after)} on a 0–100 scale, against a {band} {profile.calculated_risk_tier} band."
    story.append(_p(summary))
    for i in ideas:
        d = details.get(i.slot_number)
        lead = _first_sentence(d.thesis if d else i.thesis)
        story.append(Paragraph(f"<b>{i.ticker} ({i.lane.lower()}):</b> {readable_citations(lead, index)} Indicative {amount(i)}.",
                               STYLES["bullet"], bulletText="•"))
    if any("E1" in h.get("rules", []) for h in regeneration_history or []):
        story.append(Paragraph("<b>Not shown:</b> one idea was blocked by client limits and replaced; the blocked product is withheld.",
                               STYLES["bullet"], bulletText="•"))
    story.append(Paragraph("<b>Next step:</b> discuss with the client. Nothing is traded without the client's confirmation.",
                           STYLES["bullet"], bulletText="•"))
    figs = memo_figures(profile, ideas)
    story.append(Spacer(1, 10))
    story.append(_pair(_figure(2, "Allocation: current vs proposed", figs["allocation"], "OneView Portfolio Risk Engine"),
                       _figure(3, "Proposed allocation by asset class", figs["mix"], "OneView Portfolio Risk Engine")))

    # ---------------- Page 2: portfolio view ----------------
    story.append(PageBreak())
    story.append(Paragraph("Portfolio view", STYLES["h1"]))
    takeaway = (f"Risk stays inside the {profile.calculated_risk_tier} band ({risk_points(after)} on 0–100)" if after is not None
                and after <= profile.risk_score_range[1] else "Check the portfolio risk against the client's band")
    story.append(_side_row([_p(p3.portfolio_level_impact_narrative, index=index)], [Paragraph(takeaway, STYLES["sideb"])]))
    story.append(Spacer(1, 6))
    story += _figure(4, f"Portfolio risk vs the {profile.calculated_risk_tier} band (0–100)", figs["risk"], "OneView Portfolio Risk Engine")
    story.append(Spacer(1, 8))
    scen = [Paragraph(f"<b>{k}:</b> {readable_citations(v, index)}", STYLES["bullet"], bulletText="•") for k, v in p3.scenario_analysis.items()]
    story.append(_pair(_figure(5, "Illustrative stress: estimated portfolio move", figs["stress"], "OneView stress scenarios (illustrative)"),
                       [Paragraph("Scenario analysis", STYLES["figure"])] + scen))

    # ---------------- Idea details, sources, sign-off (flow on from the portfolio view) ----------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("Idea details", STYLES["h1"]))
    for i in ideas:
        d = details.get(i.slot_number)
        if not d:
            continue
        main = [Paragraph(f"Idea {i.slot_number} · {i.ticker} · {i.name}", STYLES["h2"]),
                _p(f"<b>Why it fits {first}:</b> {d.client_fit_rationale}", index=index),
                _p(f"<b>CIO view:</b> {_strip_label(d.why_now)}", index=index),
                _p(f"<b>Risks to raise:</b> {'; '.join(d.key_risks)}.", index=index),
                Paragraph(f"Fit {fit(i)} · {eligible(i)} · Disclosures: {', '.join(x.replace(' disclosure', '') for x in i.required_disclosures) or 'none'}",
                          STYLES["small"])]
        cited = [source_index(sources).get(s) if sources else source_index().get(s) for s in i.cited_evidence_ids]
        side = [Paragraph(f"{i.lane.title()} · {amount(i)}", STYLES["sideb"])]
        side += [Paragraph(f"{c['short']} · {c['date']}", STYLES["side"]) for c in cited if c]
        story.append(KeepTogether([_side_row(main, side), Spacer(1, 6)]))

    story.append(Paragraph("Sources", STYLES["h1"]))
    for info in (index or source_index()).values():
        link = f"<link href='{info['url']}' color='#1C5CAB'><u>{info['title']}</u></link>" if info.get("url") else f"<b>{info['title']}</b>"
        story.append(Paragraph(f"{link} — {info['short']}, {info['date']}", STYLES["body"]))

    story.append(Paragraph("RM sign-off", STYLES["h1"]))
    story.append(_p(p3.advisor_signature_block, index=index))
    story.append(Spacer(1, 4))
    story += [Paragraph(readable_citations(r, index), STYLES["small"]) for r in p3.risk_disclosures]

    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=52, bottomMargin=56,
                            title=f"{title} · {p1.client_name}", author="UBS OneView (prototype)")
    chrome = _chrome(title, date)
    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    return output_path
