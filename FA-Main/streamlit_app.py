"""UBS OneView advisor workspace.

After choosing a client, the advisor picks today's purpose. Each purpose is its own trigger and workflow
over the same agents under the Relationship Manager (the human orchestrator):
  UC1 Prepare   - before the meeting: confirm the profile, check client limits, review sourced research, approve the briefing
  UC2 Discover  - ideas for new capital in two lanes (aligned, exploratory): fit + eligibility check -> keep / edit -> memo
  UC3 Record    - after the meeting: summary, actions, profile changes the RM approves, follow-up email, compliance record
  (Market-event reassessment from the earlier draft still runs in the API; it is not offered here.)
The sidebar lists the steps of the active workflow. Each step is one page: labelled tabs show the
intermediate output and the advisor checkpoint sits at the bottom of the page, like a "next page" button.
"""

from __future__ import annotations

import copy
import html
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.api.service import AdvisorySessionService
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.workspace import AdvisorActionRequest
from src.tools.event_tools import MARKET_EVENTS, stress_impact
from src.tools.exposure import portfolio_exposure_calculator
from src.tools.memo_tools import readable_citations, source_index
from src.tools.profile_tools import profile_change_detector, required_field_validator, risk_profile_calculator
from src.tools.risk_engine import evaluate_profile_rules, load_policy, reference_mix_gaps, risk_band_points, risk_points

load_dotenv()

ROOT = Path(__file__).parent
SAMPLE_PROFILES = json.loads((ROOT / "data" / "samples" / "client_profiles.json").read_text(encoding="utf-8"))
VISIBLE_CLIENT_KEYS = ("demo_sarah", "demo_daniel", "demo_mdm_lim", "demo_aaron")
CLIENT_CARD_SUMMARIES = {
    "demo_sarah": ("Home down payment · 8 years", "Demo client for all three use cases"),
    "demo_daniel": ("S$180k property deposit · 15 months", "R5: willingness above capacity; one idea fails and is regenerated"),
    "demo_mdm_lim": ("Retirement reserve · 3 years", "Asks for a 'guaranteed 6%': R9 wording control"),
    "demo_aaron": ("Long-term growth · horizon not recorded", "R1: required field missing blocks the workflow"),
}
PRE_MEETING, ADHOC, EVENT, POST_MEETING = "pre_meeting", "adhoc_request", "market_event", "post_meeting"
USE_CASES = {
    PRE_MEETING: ("Prepare", "Before the meeting", "“I'm meeting the client tomorrow. What should we discuss?”", ":material/event:"),
    ADHOC: ("Discover", "Investment opportunities", "“What could suit the client, and why?”", ":material/bolt:"),
    POST_MEETING: ("Record", "After the meeting", "“What did we agree, and is it on record?”", ":material/edit_note:"),
    EVENT: ("Reassess", "Market-event response", "“Markets just moved. Does this change anything for the client?”", ":material/trending_down:"),
}
HOME_CASES = (PRE_MEETING, ADHOC, POST_MEETING)  # the three moments in the deck
DEMO_TRANSCRIPTS = json.loads((Path(__file__).parent / "data" / "samples" / "meeting_transcripts.json").read_text(encoding="utf-8"))
OFFLINE, LIVE = "Offline · test data stubs", "Live · OpenRouter"

# Validated reference palette (dataviz skill): status colours never double as series colours.
STATUS_COLORS = {"PASS": "#0ca30c", "FLAG": "#fab219", "FAIL": "#d03b3b", "BLOCK": "#d03b3b"}
STATUS_ICONS = {"PASS": "✓", "FLAG": "!", "FAIL": "✕", "BLOCK": "■"}
BEFORE_COLOR, AFTER_COLOR = "#cfcfcc", "#c8102e"  # before (light grey) -> proposed (brand red)
SERIES = ["#c8102e", "#1f1f1f", "#8c8c8a", "#c9a227", "#d6d6d3", "#7a1f2b", "#4d4d4b", "#e79aa6"]  # red, black, greys, gold
MUTED = "#7a7a78"
RM_NAME = "Rachel Wong"  # the deck's illustrative Relationship Manager
# Assumed list prices used only for the cost estimate on the Audit screen (US$ per million tokens).
ASSUMED_PRICE_PER_MTOK = {"prompt": 3.0, "completion": 15.0}
RISK_OPTIONS = ["Conservative", "Low to moderate", "Moderate", "Balanced growth", "Aggressive growth"]
EXPERIENCE_OPTIONS = ["New investor", "Limited", "ETF investor", "Knowledgeable", "Experienced investor"]

# Workflow steps per use case: key, label, checkpoint (None = output page, no advisor decision).
STEPS = {
    PRE_MEETING: [("profile", "Client profile", "CP1"), ("limits", "Client limits", "Limits"), ("research", "Market landscape", "CP2"),
                  ("ideas", "Opportunities", None), ("suitability", "Fit & eligibility", None),
                  ("approval", "Lock in ideas", "CP3"), ("brief", "Meeting brief", "CP4")],
    ADHOC: [("profile", "Client inputs", "CP1"), ("limits", "Client limits", "Limits"), ("research", "Research", "CP2"),
            ("ideas", "Opportunities", None), ("suitability", "Fit & eligibility", None),
            ("approval", "Keep / edit ideas", "CP3"), ("memo", "Memo & sign-off", "CP4")],
    POST_MEETING: [("profile", "Client profile", None), ("meeting", "Meeting record", "Record")],
    EVENT: [("profile", "Client profile", None), ("event", "Event impact", "Decide"),
            ("ideas", "Candidate ideas", None), ("suitability", "Suitability check", None),
            ("approval", "Advisor approval", "CP3"), ("memo", "Memo & sign-off", "CP4"), ("record", "Decision record", None)],
}
IDEA_STEPS = ("ideas", "suitability", "approval")
STATUS_NAV_ICON = {"done": ":material/check_circle:", "action": ":material/pending_actions:",
                   "failed": ":material/cancel:", "locked": ":material/lock:"}

st.set_page_config(page_title="UBS OneView", layout="wide", initial_sidebar_state="expanded")

st.html("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  :root { --brand-red: #c8102e; --ink: #1a1a1a; --muted: #7a7a78; --line: #ebebe8; --canvas: #f4f4f2; --pink: #fdecee; }
  .stApp { background: var(--canvas); font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; color: var(--ink); }
  .block-container { max-width: 1440px; padding-top: 3.4rem; padding-bottom: 1.1rem; }
  /* The transparent header overlays the top of the page; let clicks reach buttons under it (menu stays clickable). */
  header[data-testid="stHeader"] { background: transparent; }
  header[data-testid="stHeader"], header[data-testid="stHeader"] * { pointer-events: none !important; }
  header[data-testid="stHeader"] button, header[data-testid="stHeader"] button * { pointer-events: auto !important; }
  [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap: .9rem; }
  h1, h2, h3, h4 { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; letter-spacing: -0.02em; color: var(--ink); }
  h2 { font-size: 1.3rem !important; font-weight: 600; padding: 0 !important; margin: 0 !important; }
  h4 { font-size: .95rem !important; font-weight: 600; padding: .2rem 0 0 !important; margin: 0 0 .2rem !important; }

  /* cards: every bordered container is a white rounded card with a soft shadow */
  [class*="st-key-card_"], .st-key-footer, .st-key-footer_done {
    background: #fff; border: 1px solid #efefec !important; border-radius: 14px !important;
    box-shadow: 0 1px 2px rgba(20, 20, 20, .04), 0 6px 18px rgba(20, 20, 20, .04); }
  .st-key-footer { border-left: 3px solid var(--brand-red) !important; }
  .st-key-footer_done { border-left: 3px solid #0ca30c !important; }
  [data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 10px; background: #fff; }
  [data-testid="stExpander"] summary { font-size: .82rem; }

  /* sidebar: slim, white, small-caps section labels, pink active item */
  section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); width: 16rem !important; min-width: 16rem !important; }
  [data-testid="stSidebarHeader"] { height: 2.4rem; padding-top: .6rem; }
  [data-testid="stSidebarUserContent"] { padding-top: 0; padding-bottom: 5rem; }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .15rem; }
  [data-testid="stSidebar"] .stButton button { justify-content: flex-start; min-height: 2.1rem; padding: .3rem .7rem; border: none !important;
    background: transparent; color: #3a3a38; box-shadow: none !important; border-radius: 8px; }
  [data-testid="stSidebar"] .stButton button p { text-align: left; font-size: .86rem; font-weight: 500; }
  [data-testid="stSidebar"] .stButton button > div, [data-testid="stSidebar"] .stButton button span { justify-content: flex-start; }
  [data-testid="stSidebar"] .stButton button > div { width: 100%; }
  [data-testid="stSidebar"] .stHtml, [data-testid="stSidebar"] .stMarkdown { margin-bottom: .2rem; }
  [data-testid="stSidebar"] .stButton button:hover { background: #f5f5f3; color: var(--ink); }
  [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] { background: var(--pink); color: var(--brand-red); }
  [data-testid="stSidebar"] .stButton button:disabled { color: #b5b5b2; background: transparent; }
  .side-label { display: block; color: var(--muted); font-size: .66rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; margin: 1.1rem 0 .35rem .7rem; }
  .user-chip { position: fixed; bottom: 0; left: 0; width: 16rem; padding: .8rem 1rem; border-top: 1px solid var(--line); background: #fff;
    display: flex; align-items: center; gap: .6rem; font-size: .8rem; line-height: 1.25; z-index: 5; }
  .avatar { width: 1.9rem; height: 1.9rem; border-radius: 50%; background: var(--brand-red); color: #fff; display: inline-flex;
    align-items: center; justify-content: center; font-weight: 600; font-size: .78rem; flex: none; }
  .avatar.lg { width: 2.4rem; height: 2.4rem; font-size: .9rem; }

  /* type */
  .eyebrow { color: var(--muted); font-size: .68rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; }
  .card-label { color: var(--ink); font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
  .subtle { color: var(--muted); font-size: .86rem; }
  .brand { font-weight: 700; font-size: 1.05rem; letter-spacing: -.02em; padding: .2rem .7rem 0; }
  .brand span { font-weight: 500; color: var(--ink); }
  .brand b { color: var(--brand-red); }
  .client-chip { border-radius: 10px; padding: .55rem .7rem; margin: .6rem 0 0; font-size: .8rem; line-height: 1.4; background: #f7f7f5;
    display: flex; gap: .55rem; align-items: center; }
  .client-chip b { font-size: .88rem; }

  /* top bar on every workflow page */
  .topbar { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; padding: .2rem 0 .4rem; border-bottom: 1px solid var(--line); }
  .topbar .title { font-size: 1.3rem; font-weight: 600; letter-spacing: -.02em; }

  /* status pills */
  .pill { display: inline-block; border-radius: 99px; padding: .08rem .55rem; font-size: .72rem; font-weight: 600; border: 1px solid; margin-right: .25rem; white-space: nowrap; }
  .pill.PASS { color: #0a7d0a; border-color: #bfe3bf; background: #eef8ee; }
  .pill.FLAG { color: #8a5d00; border-color: #f4dc9c; background: #fff8e6; }
  .pill.FAIL, .pill.BLOCK { color: #b42f2f; border-color: #f0c2c2; background: #fdeeee; }
  .pill.INFO { color: #4d4d4b; border-color: var(--line); background: #f7f7f5; }

  /* stepper (idea lineage) */
  .lineage { display: flex; align-items: center; gap: .4rem; flex-wrap: wrap; margin: .2rem 0 0; }
  .lineage .step { display: flex; align-items: center; gap: .55rem; flex: 1 1 150px; font-size: .68rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: var(--muted); }
  .lineage .step .dot { width: 1.7rem; height: 1.7rem; border-radius: 50%; border: 1.5px solid #d6d6d3; display: inline-flex; align-items: center;
    justify-content: center; font-size: .75rem; color: var(--muted); background: #fff; flex: none; }
  .lineage .step b { display: block; font-size: .86rem; letter-spacing: 0; text-transform: none; color: var(--ink); font-weight: 600; }
  .lineage .step.done .dot { border-color: var(--brand-red); color: var(--brand-red); }
  .lineage .step.active .dot { background: var(--brand-red); border-color: var(--brand-red); color: #fff; }
  .lineage .step.active { color: var(--brand-red); }
  .lineage .arrow { flex: 0 0 1.6rem; height: 1.5px; background: #dcdcd9; }

  /* content blocks */
  .card-ticker { font-size: 1.15rem; font-weight: 700; letter-spacing: -.02em; }
  .kv { font-size: .85rem; line-height: 1.55; margin-bottom: .5rem; }
  .kv b { color: var(--muted); font-weight: 600; }
  .stats { display: flex; gap: 1.4rem; margin: .5rem 0 .6rem; font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
  .stats b { display: block; font-size: 1.05rem; color: var(--ink); text-transform: none; letter-spacing: 0; }
  .callout { background: #fff; border: 1px solid #efefec; border-left: 3px solid var(--brand-red); border-radius: 12px; padding: .75rem 1rem; font-size: .88rem;
    box-shadow: 0 1px 2px rgba(20, 20, 20, .04); }
  .src { font-size: .84rem; line-height: 1.45; margin: 0 0 .45rem; }
  .src a { font-weight: 600; text-decoration: none; color: var(--ink); }
  .src a:hover { color: var(--brand-red); }
  .num { font-size: 1.6rem; font-weight: 300; color: #c9c9c6; line-height: 1; }
  table.grid { border-collapse: collapse; width: 100%; font-size: .83rem; background: #fff; border-radius: 12px; overflow: hidden; }
  table.grid th, table.grid td { border-bottom: 1px solid var(--line); padding: .5rem .65rem; text-align: left; vertical-align: top; }
  table.grid th { color: var(--muted); font-weight: 600; font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; background: #fafaf8; }
  .usecase-trigger { color: var(--muted); font-size: .86rem; line-height: 1.45; min-height: 2.6rem; margin: .25rem 0 .35rem; }
  .cal { border: 1px solid #efefec; border-radius: 12px; padding: .7rem 1rem; font-size: .88rem; background: #fff; }

  /* widgets */
  [data-testid="stMetric"] { background: #fff; border: 1px solid #efefec; border-radius: 12px; padding: .7rem .95rem; box-shadow: 0 1px 2px rgba(20, 20, 20, .04); }
  [data-testid="stMetricLabel"] p { font-size: .66rem !important; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  [data-testid="stMetricValue"] { font-size: 1.15rem; font-weight: 600; }
  .stButton button, .stDownloadButton button, .stFormSubmitButton button { border-radius: 8px; font-weight: 600; }
  [data-testid="stBaseButton-secondary"] { background: #fff; border-color: #e2e2df; }
  [data-baseweb="tab-list"] { gap: 1.2rem; }
  [data-baseweb="tab"] p { font-size: .84rem; font-weight: 500; }
  [data-testid="stTextInputRootElement"], [data-baseweb="select"] > div, [data-testid="stNumberInputContainer"], textarea { border-radius: 8px !important; }
</style>
""")


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------

def esc(text: Any) -> str:
    """Escape for HTML blocks; '$' is encoded so Streamlit never reads it as LaTeX. Source ids become publisher · date."""
    return html.escape(readable_citations(text, st.session_state.get("source_index"))).replace("$", "&#36;")


def md(text: Any) -> str:
    return readable_citations(text, st.session_state.get("source_index")).replace("$", r"\$")


def source_links(source_ids: list[str] | None = None) -> str:
    """Sources as linked titles with publisher and date (all sources of this session when no ids are given)."""
    index = st.session_state.get("source_index") or source_index()
    rows = []
    for sid in dict.fromkeys(source_ids if source_ids is not None else list(index)):
        info = index.get(sid) or source_index().get(sid)
        if not info:
            continue
        title = html.escape(info["title"])
        link = f"<a href='{html.escape(info['url'])}' target='_blank'>{title}</a>" if info.get("url") else f"<b>{title}</b>"
        rows.append(f"<div class='src'>{link}<br><span class='subtle'>{html.escape(info['short'])} · {html.escape(info['date'])}</span></div>")
    return "".join(rows)


def fmt_money(amount: float | None, currency: str = "SGD") -> str:
    if amount is None:
        return "Not recorded"
    symbol = "S$" if currency == "SGD" else f"{currency} "
    return f"{symbol}{amount:,.0f}"


def fmt_horizon(months: int | None) -> str:
    if not months:
        return "Not recorded"
    return f"{months // 12} years" if months % 12 == 0 else f"{months} months"


def pill(status: str, text: str | None = None) -> str:
    return f"<span class='pill {esc(status)}'>{STATUS_ICONS.get(status, '')} {esc(text or status)}</span>"


def info_pill(text: str) -> str:
    return f"<span class='pill INFO'>{esc(text)}</span>"


_CARD_IDS = iter(range(1, 1_000_000))


def card(height: int | str | None = None) -> Any:
    """A white rounded card (bordered container with a key the stylesheet targets)."""
    kwargs = {"height": height} if height is not None else {}
    return st.container(border=True, key=f"card_{next(_CARD_IDS)}", **kwargs)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def kv(label: str, value: Any) -> str:
    return f"<div class='kv'><b>{esc(label)}</b><br>{esc(value)}</div>"


# --------------------------------------------------------------------------------------
# Client records (lifecycle across use cases) and session state
# --------------------------------------------------------------------------------------

def client_record(key: str) -> dict[str, Any]:
    """The client's record for this browser session: updated after each completed workflow."""
    records = st.session_state.setdefault("records", {})
    if key not in records:
        records[key] = {"raw": copy.deepcopy(SAMPLE_PROFILES.get(key, {})), "history": []}
    return records[key]


def new_service() -> AdvisorySessionService:
    if st.session_state.get("mode") == LIVE:
        from src.llm.openrouter import OpenRouterLLMClient
        return AdvisorySessionService(llm_client=OpenRouterLLMClient())
    return AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())


def load_context() -> dict[str, Any]:
    """Everything a page needs, read once per rerun from the paused LangGraph session."""
    service = st.session_state.get("service")
    view = st.session_state.get("workspace")
    if not service or not view:
        return {"state": {}, "view": None, "pending": None, "stage": None, "outcome": None, "use_case": None}
    view = service.get_workspace_view(view.session_id)
    st.session_state.workspace = view
    state = service.get_state(view.session_id)
    if state.get("research_brief"):
        st.session_state.source_index = source_index(state["research_brief"].validated_sources)
    pending = view.pending_review
    return {"state": state, "view": view, "pending": pending, "stage": pending.stage if pending else None,
            "outcome": view.outcome, "use_case": state.get("use_case", ADHOC), "sid": view.session_id}


def visible_steps(ctx: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """The active workflow's steps; optional branches appear once the workflow takes them."""
    uc, state = ctx["use_case"], ctx["state"]
    steps = []
    for key, label, cp in STEPS[uc]:
        if uc == PRE_MEETING and key in IDEA_STEPS and state.get("include_ideas") is False:
            continue
        if uc == EVENT:
            decision = (state.get("event_decision") or {}).get("decision")
            if key in IDEA_STEPS + ("memo",) and decision != "REVISE":
                continue
            if key == "record" and decision not in ("MAINTAIN", "MONITOR"):
                continue
        steps.append((key, label, cp))
    return steps + [("audit", "Audit & performance", None)]


def nav_status(key: str, ctx: dict[str, Any]) -> str:
    """done | action | failed | locked | open for one sidebar entry."""
    state, stage, outcome = ctx["state"], ctx["stage"], ctx["outcome"]
    rejected_at = state.get("rejected_at_stage") if outcome == "REJECTED" else None
    if key == "audit":
        return "open" if state else "locked"
    if key == "profile":
        if rejected_at == "client_profile":
            return "failed"
        if stage == "client_profile":
            return "action"
        return "done" if state.get("profile") and (state.get("research_brief") or state.get("meeting_record") or outcome) else ("open" if state.get("profile") else "locked")
    if key == "limits":
        if not state.get("offer_map"):
            return "locked"
        if rejected_at == "client_limits":
            return "failed"
        return "action" if stage == "client_limits" else "done"
    if key == "meeting":
        if not state.get("meeting_record"):
            return "locked"
        if rejected_at == "record_approval":
            return "failed"
        return "action" if stage == "record_approval" else ("done" if outcome == "APPROVED" else "open")
    if key == "research":
        if not state.get("research_brief"):
            return "locked"
        if rejected_at == "market_intelligence":
            return "failed"
        return "action" if stage == "market_intelligence" else "done"
    if key == "event":
        if not state.get("event_impact"):
            return "locked"
        return "action" if stage == "event_decision" else "done"
    if key == "ideas":
        return "done" if state.get("idea_rounds") else "locked"
    if key == "suitability":
        assessment = state.get("assessment")
        if not assessment:
            return "locked"
        return "done" if assessment.overall_status != "FAIL" else "failed"
    if key == "approval":
        if not state.get("assessment"):
            return "locked"
        if rejected_at == "suitability_review":
            return "failed"
        if stage == "suitability_review":
            return "action"
        return "done" if state.get("approved_idea_set") else "open"
    if key == "memo":
        if rejected_at == "advisor_approval":
            return "failed"
        if stage == "advisor_approval":
            return "action"
        return "done" if outcome == "APPROVED" and state.get("memo") else "locked"
    if key == "brief":
        if rejected_at == "brief_approval":
            return "failed"
        if stage == "brief_approval":
            return "action"
        return "done" if outcome == "APPROVED" and state.get("meeting_brief") else "locked"
    if key == "record":
        return "done" if state.get("decision_record") else "locked"
    return "open"


def screen_for_pending(previous_stage: str | None, ctx: dict[str, Any]) -> str:
    """After an advisor action, open the page that shows what the workflow produced next."""
    uc, state = ctx["use_case"], ctx["state"]
    if ctx["outcome"] == "APPROVED":
        if uc == PRE_MEETING:
            return "brief"
        if uc == POST_MEETING:
            return "meeting"
        return "record" if state.get("decision_record") else "memo"
    if ctx["outcome"] == "REJECTED":
        return "audit"
    stage = ctx["stage"]
    if stage == "suitability_review":
        return "ideas" if previous_stage in ("market_intelligence", "event_decision") else "suitability"
    return {"client_profile": "profile", "client_limits": "limits", "market_intelligence": "research", "event_decision": "event",
            "advisor_approval": "memo", "brief_approval": "brief", "record_approval": "meeting"}.get(stage, "audit")


def submit_action(action: str, notes: str = "", **kwargs: Any) -> None:
    service = st.session_state.service
    view = st.session_state.workspace
    previous_stage = view.pending_review.stage if view.pending_review else None
    try:
        request = AdvisorActionRequest(action=action, notes=notes.strip() or None, **kwargs)
        with st.spinner("Agents are working…"):
            st.session_state.workspace = service.execute_action(view.session_id, request)
        st.session_state.error = None
        if action != "add_advisor_note":
            ctx = load_context()
            st.session_state.screen = screen_for_pending(previous_stage, ctx)
            if action in ("lock_idea", "unlock_idea") and ctx["stage"] == "suitability_review":
                st.session_state.screen = "approval"  # keep locking in the remaining ideas on the same page
            if ctx["outcome"]:
                record_lifecycle(ctx)
    except Exception as exc:  # surfaced to the advisor instead of a blank screen
        st.session_state.error = str(exc)
    st.rerun()


def _raw_holdings(holdings: list[Any]) -> list[dict[str, Any]]:
    return [{"ticker": h.ticker, "name": h.name, "asset_class": h.asset_class, "region": h.region, "currency": h.currency,
             "weight_pct": round(h.weight_pct, 2), "risk_rating": h.risk_rating} for h in holdings if h.weight_pct > 0.01]


def record_lifecycle(ctx: dict[str, Any]) -> None:
    """Carry a completed workflow into the client record, so the next use case starts from it."""
    done = st.session_state.setdefault("recorded_sessions", set())
    if ctx["sid"] in done:
        return
    done.add(ctx["sid"])
    record = client_record(st.session_state.client_key)
    state, uc = ctx["state"], ctx["use_case"]
    raw = record["raw"]
    entry = {"use_case": uc, "date": date.today().isoformat(), "outcome": ctx["outcome"], "session": ctx["sid"], "summary": ""}
    if ctx["outcome"] == "APPROVED":
        confirmed = {k: v for k, v in state.get("raw_input", {}).items() if k not in ("use_case", "meeting", "market_event", "new_capital_amount", "transcript", "previous_meeting_record", "rm_news")}
        raw.update(confirmed)
        approved = state.get("approved_idea_set")
        if uc == PRE_MEETING:
            raw["last_review"] = {**{k: copy.deepcopy(raw.get(k)) for k in ("goals", "time_horizon_months", "stated_risk_willingness",
                                                                          "liquidity_schedule", "total_wealth", "experience_level")},
                                  "date": date.today().isoformat()}
            n = len(state.get("profile_changes") or [])
            entry["summary"] = f"Meeting brief approved · {n} profile change(s)" + (" · 3 ideas discussed" if approved else "")
        elif approved:
            # Assume the approved trades were executed, so later reviews start from the new portfolio.
            ideas = state["candidate_ideas"].ideas
            delta = portfolio_exposure_calculator(state["profile"].parsed_holdings, trades_from(ideas))
            raw["raw_holdings"] = _raw_holdings(delta.after_holdings)
            raw["total_wealth"] = state["profile"].total_wealth
            entry["summary"] = "Approved: " + ", ".join(f"{i.action} {i.ticker} {i.target_weight_pct:.1f}%" for i in ideas)
            record["current_advice"] = {"date": entry["date"], "ideas": [i.model_dump() for i in ideas], "use_case": uc}
        if uc == POST_MEETING and state.get("meeting_record"):
            rec = state["meeting_record"]
            record["previous_meeting_record"] = rec.model_dump(mode="json")  # feeds the next pre-meeting briefing
            entry["summary"] = (f"Record confirmed · {len(rec.actions)} action(s) · "
                                f"{len(state.get('approved_profile_updates') or [])} profile update(s) approved")
        if uc == EVENT and state.get("event_decision"):
            d = state["event_decision"]
            entry["summary"] = f"{d['decision'].title()}: {d['rationale']}" + (f" · follow-up: {d['follow_up']}" if d.get("follow_up") else "") \
                + (f" · {entry['summary']}" if approved else "")
    else:
        entry["summary"] = f"Rejected at {state.get('rejected_at_stage', 'a checkpoint').replace('_', ' ')}"
    record["history"].append(entry)


def go(screen: str) -> None:
    st.session_state.screen = screen


def reset_to_client_list() -> None:
    for key in [k for k in st.session_state.keys() if k not in ("mode", "records", "recorded_sessions")]:
        del st.session_state[key]


def back_to_client() -> None:
    for key in [k for k in st.session_state.keys() if k not in ("mode", "records", "recorded_sessions", "client_key")]:
        del st.session_state[key]


def select_client(client_key: str, raw: dict[str, Any] | None = None) -> None:
    reset_to_client_list()
    st.session_state.client_key = client_key
    if raw is not None:
        st.session_state.setdefault("records", {})[client_key] = {"raw": raw, "history": []}
    client_record(client_key)


def start_case(case: str, client_key: str | None = None) -> None:
    if client_key:
        select_client(client_key)
    record = client_record(st.session_state.client_key)
    raw = copy.deepcopy(record["raw"])
    raw["use_case"] = case
    if case == PRE_MEETING and record.get("previous_meeting_record"):
        raw["previous_meeting_record"] = record["previous_meeting_record"]
    if case == PRE_MEETING:
        meeting_day = st.session_state.get("uc1_date") or (date.today() + timedelta(days=1))
        raw["meeting"] = {"purpose": st.session_state.get("uc1_purpose") or "Quarterly portfolio review", "date": str(meeting_day)}
    elif case == ADHOC:
        raw["new_capital_amount"] = float(st.session_state.get("uc2_amount") or 0)
    elif case == POST_MEETING:
        raw["transcript"] = st.session_state.get("uc3_transcript") or ""
        raw["meeting"] = {"purpose": st.session_state.get("uc3_purpose") or "Client meeting", "date": str(date.today())}
    else:
        key = st.session_state.get("uc3_event") or "rates_up"
        raw["market_event"] = {"key": key, "title": MARKET_EVENTS[key]["title"],
                               "description": (st.session_state.get("uc3_desc") or MARKET_EVENTS[key]["description"]).strip()}
    st.session_state.case = case
    st.session_state.error = None
    try:
        service = new_service()
        with st.spinner("Agents are preparing the first checkpoint…"):
            st.session_state.workspace = service.start_session(raw)
        st.session_state.service = service
        st.session_state.screen = screen_for_pending(None, load_context())
    except Exception as exc:
        st.session_state.case = None
        st.session_state.error = str(exc)


# --------------------------------------------------------------------------------------
# Derived data for the charts
# --------------------------------------------------------------------------------------

def trades_from(ideas: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for idea in ideas:
        get = idea.get if isinstance(idea, dict) else lambda k, i=idea: getattr(i, k)
        rows.append({"ticker": get("ticker"), "action": get("action"), "target_weight_pct": get("target_weight_pct"),
                     "funding_source_ticker": get("funding_source_ticker")})
    return rows


def exposure_frame(profile: Any, ideas: list[Any], dimension: str) -> pd.DataFrame:
    """Before vs after weights by asset class or region, from the deterministic exposure calculator."""
    delta = portfolio_exposure_calculator(profile.parsed_holdings, trades_from(ideas))
    rows: dict[str, dict[str, float]] = {}
    for label, holdings in (("Before", delta.before_holdings), ("After", delta.after_holdings)):
        for holding in holdings:
            rows.setdefault(getattr(holding, dimension), {"Before": 0.0, "After": 0.0})[label] += holding.weight_pct
    return pd.DataFrame([{"Category": k, **v} for k, v in rows.items()]).sort_values("After", ascending=False)


def holding_weights(holdings: list[Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for holding in holdings:
        weights[holding.ticker] = weights.get(holding.ticker, 0.0) + holding.weight_pct
    return weights


def stress_frame(profile: Any, ideas: list[Any]) -> pd.DataFrame:
    delta = portfolio_exposure_calculator(profile.parsed_holdings, trades_from(ideas))
    rows = []
    for label, holdings in (("Before", delta.before_holdings), ("After", delta.after_holdings)):
        rows += [{"Scenario": k, "Portfolio": label, "Impact %": v} for k, v in stress_impact(holdings).items()]
    return pd.DataFrame(rows)


def idea_lineage(state: dict[str, Any]) -> dict[str, int]:
    rounds = state.get("idea_rounds") or []
    generated = {idea["idea_id"] for r in rounds for idea in r["ideas"]}
    latest = rounds[-1] if rounds else None
    checked = sum(1 for s in (latest["statuses"].values() if latest else []) if s in ("PASS", "FLAG"))
    approved = len((state.get("approved_idea_set") or {}).get("ideas", []))
    final = state.get("memo") or state.get("meeting_brief")
    in_final = 3 if final and approved else 0
    return {"generated": len(generated), "rounds": len(rounds), "checked": checked, "approved": approved, "final": in_final}


def checkpoint_decisions(state: dict[str, Any], stage: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for entry in state.get("audit_trail") or []:
        key = entry.get("stage", "")
        if key.startswith("review_") and (stage is None or key == f"review_{stage}"):
            action = entry.get("human_action") or {}
            rows.append({"Checkpoint": key.removeprefix("review_").replace("_", " ").title(), "Decision": action.get("action", ""),
                         "Advisor note": action.get("notes") or "", "Time": entry.get("timestamp", "")[11:16]})
    return rows


# --------------------------------------------------------------------------------------
# Charts (Altair; marks and colours follow the dataviz reference palette)
# --------------------------------------------------------------------------------------

def _pop_title(chart: Any) -> str | None:
    """Remove a chart's Vega title (and its layers'); a Vega title collapses the plot under Streamlit's sizing."""
    found = None
    title = getattr(chart, "title", alt.Undefined)
    if title is not alt.Undefined:
        found = title.text if isinstance(title, alt.TitleParams) else str(title)
        chart.title = alt.Undefined
    for layer in getattr(chart, "layer", None) or []:
        layer_title = _pop_title(layer)  # strip every layer, not just the first one with a title
        found = found or layer_title
    return found


def show_chart(chart: alt.Chart) -> None:
    chart = chart.copy(deep=True)
    title = _pop_title(chart)
    if title:
        html_block(f"<div style='font-weight:650;font-size:.92rem;padding-top:.2rem'>{esc(title)}</div>")
    st.altair_chart(chart.configure_view(strokeWidth=0).configure(background="transparent", font="Inter"), width="stretch")


def donut(weights: dict[str, float], title: str, height: int = 200) -> alt.Chart:
    frame = pd.DataFrame([{"Category": k, "Weight": round(v, 1)} for k, v in sorted(weights.items(), key=lambda kv: -kv[1])])
    order = list(frame["Category"])
    return alt.Chart(frame, title=title).mark_arc(innerRadius=58, outerRadius=80, stroke="#ffffff", strokeWidth=2).encode(
        theta=alt.Theta("Weight:Q", stack=True),
        color=alt.Color("Category:N", scale=alt.Scale(domain=order, range=SERIES[:len(order)]), legend=alt.Legend(title=None, labelLimit=150)),
        order=alt.Order("Weight:Q", sort="descending"),
        tooltip=["Category", alt.Tooltip("Weight:Q", format=".1f", title="Weight %")],
    ).properties(height=height)


def before_after_bars(frame: pd.DataFrame, title: str, height: int = 220) -> alt.Chart:
    long = frame.melt(id_vars="Category", value_vars=["Before", "After"], var_name="Portfolio", value_name="Weight")
    return alt.Chart(long, title=title).mark_bar(cornerRadiusEnd=4, height=11).encode(
        y=alt.Y("Category:N", sort=list(frame["Category"]), title=None, axis=alt.Axis(labelLimit=160)),
        yOffset=alt.YOffset("Portfolio:N", sort=["Before", "After"]),
        x=alt.X("Weight:Q", title="Portfolio weight %"),
        color=alt.Color("Portfolio:N", scale=alt.Scale(domain=["Before", "After"], range=[BEFORE_COLOR, AFTER_COLOR]), legend=alt.Legend(title=None)),
        tooltip=["Category", "Portfolio", alt.Tooltip("Weight:Q", format=".1f", title="Weight %")],
    ).properties(height=height)


def metric_vs_limit(rule: Any, unit: str = "") -> alt.Chart:
    """One rule as a small multiple: before and after bars against the rule's limit line."""
    show = risk_points if rule.metric == "risk_score" else (lambda v: v)
    num = ".0f" if rule.metric == "risk_score" else ".2f"
    frame = pd.DataFrame([{"Portfolio": "Before", "Value": show(rule.before_value)}, {"Portfolio": "After", "Value": show(rule.after_value)}])
    limit = show(rule.threshold_value)
    domain_max = max(frame["Value"].max(), limit) * 1.25 or 1
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Portfolio:N", sort=["Before", "After"], title=None, scale=alt.Scale(paddingInner=0.3)),
        x=alt.X("Value:Q", title=unit or None, scale=alt.Scale(domain=[0, domain_max])),
        color=alt.Color("Portfolio:N", scale=alt.Scale(domain=["Before", "After"], range=[BEFORE_COLOR, AFTER_COLOR]), legend=None),
        tooltip=["Portfolio", alt.Tooltip("Value:Q", format=num)])
    labels = bars.mark_text(align="left", dx=4, color=MUTED).encode(text=alt.Text("Value:Q", format=num), color=alt.value(MUTED))
    rule_line = alt.Chart(pd.DataFrame([{"Limit": limit}])).mark_rule(strokeDash=[4, 3], color=STATUS_COLORS["FAIL"], strokeWidth=2).encode(
        x="Limit:Q", tooltip=[alt.Tooltip("Limit:Q", format=num, title="Limit")])
    return alt.layer(bars, labels, rule_line, title=f"{STATUS_ICONS.get(rule.outcome, '')} {rule.rule_id} · {rule.name}").properties(height=120)


def risk_scale_chart(points: list[dict[str, Any]], band: list[float]) -> alt.Chart:
    """Scores come in on the rules' 1-5 scale and are plotted on the 0-100 scale the RM sees."""
    frame = pd.DataFrame([{**p, "Score": risk_points(p["Score"])} for p in points])
    order = list(frame["Measure"]) + ["Applied band"]
    y_scale = alt.Scale(domain=order)
    band_rect = alt.Chart(pd.DataFrame([{"lo": risk_points(band[0]), "hi": risk_points(band[1]), "y": "Applied band"}])).mark_bar(color="#cde2fb", cornerRadius=4).encode(
        x=alt.X("lo:Q", scale=alt.Scale(domain=[0, 100]), title="Risk (0 = lowest, 100 = highest)"), x2="hi:Q",
        y=alt.Y("y:N", scale=y_scale, title=None, axis=alt.Axis(labelLimit=160)),
        tooltip=[alt.Tooltip("lo:Q", title="Band from"), alt.Tooltip("hi:Q", title="Band to")])
    dots = alt.Chart(frame).mark_point(filled=True, size=170, stroke="#ffffff", strokeWidth=2).encode(
        x="Score:Q", y=alt.Y("Measure:N", scale=y_scale),
        color=alt.Color("Measure:N", scale=alt.Scale(domain=list(frame["Measure"]), range=SERIES[:len(frame)]), legend=None),
        tooltip=["Measure", alt.Tooltip("Score:Q", format=".0f")])
    labels = dots.mark_text(dx=14, align="left", color=MUTED).encode(text=alt.Text("Score:Q", format=".0f"), color=alt.value(MUTED))
    return alt.layer(band_rect, dots, labels).properties(height=46 * len(order) + 40)


def reference_mix_chart(profile: Any) -> alt.Chart:
    """Current allocation vs the CIO reference mix for the client's risk band (slide 9 portfolio card)."""
    mix = (load_policy().get("cio_reference_mix") or {}).get("mixes", {}).get(profile.calculated_risk_tier, {})
    classes = sorted(set(mix) | set(profile.current_asset_class_weights), key=lambda c: -mix.get(c, 0))
    rows = [{"Category": c, "Portfolio": label, "Weight": round(value, 1)} for c in classes
            for label, value in (("Current", profile.current_asset_class_weights.get(c, 0.0)), ("CIO mix", float(mix.get(c, 0))))]
    return alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=3, height=8).encode(
        y=alt.Y("Category:N", sort=classes, title=None, axis=alt.Axis(labelLimit=110)),
        yOffset=alt.YOffset("Portfolio:N", sort=["Current", "CIO mix"]),
        x=alt.X("Weight:Q", title="%"),
        color=alt.Color("Portfolio:N", scale=alt.Scale(domain=["Current", "CIO mix"], range=[AFTER_COLOR, BEFORE_COLOR]), legend=None),
        tooltip=["Category", "Portfolio", alt.Tooltip("Weight:Q", format=".1f")]).properties(height=150, background="transparent")


def status_heatmap(frame: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(frame).encode(
        x=alt.X("Scope:N", title=None, sort=list(dict.fromkeys(frame["Scope"])), axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("Rule:N", title=None, sort=list(dict.fromkeys(frame["Rule"])), axis=alt.Axis(labelLimit=240)))
    cells = base.mark_rect(cornerRadius=4, stroke="#ffffff", strokeWidth=2).encode(
        color=alt.Color("Outcome:N", scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())), legend=None),
        tooltip=["Scope", "Rule", "Outcome", "Message"])
    text = base.mark_text(fontWeight="bold").encode(
        text="Label:N", color=alt.condition(alt.datum.Outcome == "FLAG", alt.value("#0b0b0b"), alt.value("#ffffff")))
    return alt.layer(cells, text).properties(height=min(32 * frame["Rule"].nunique() + 40, 380))


def signed_bars(frame: pd.DataFrame, label: str, value: str, title: str, height: int = 220) -> alt.Chart:
    order = list(frame.sort_values(value)[label])
    bars = alt.Chart(frame, title=title).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y(f"{label}:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X(f"{value}:Q", title="Portfolio % points"),
        color=alt.condition(alt.datum[value] >= 0, alt.value(SERIES[0]), alt.value(SERIES[7])),
        tooltip=list(frame.columns))
    zero = alt.Chart(pd.DataFrame([{value: 0}])).mark_rule(color=MUTED).encode(x=f"{value}:Q")
    return (bars + zero).properties(height=height)


# --------------------------------------------------------------------------------------
# Page chrome: header, lineage strip, checkpoint footer
# --------------------------------------------------------------------------------------

PAGE_ROLES = {
    "profile": "Client Profile Agent · Understand", "limits": "Client Limits layer (rules) · Understand",
    "research": "Market Intelligence Agent · Research",
    "event": "Market Intelligence + Fit & Eligibility Critic · Reassess", "ideas": "Opportunity Agent · Discover",
    "suitability": "Fit & Eligibility Critic + rule engine · Check", "approval": "Advisor review · Decide",
    "memo": "Brief & Record Agent · Communicate", "brief": "Brief & Record Agent · Communicate",
    "meeting": "Brief & Record Agent + Fit & Eligibility Critic · Communicate",
    "record": "Brief & Record Agent · Communicate", "audit": "Governance · Audit",
}


def page_header(ctx: dict[str, Any], key: str, title: str, subtitle: str, status_html: str = "") -> None:
    """Same header on every page: step n of N · role · checkpoint, title with status on the right, one-line purpose."""
    steps = [k for k, *_ in visible_steps(ctx) if k != "audit"]
    cp = next((c for k, _, c in STEPS[ctx["use_case"]] if k == key), None)
    position = f"Step {steps.index(key) + 1} of {len(steps)} · " if key in steps else ""
    checkpoint = f" · Checkpoint {cp[2:]}" if cp and cp.startswith("CP") else (" · Advisor decision" if cp else "")
    html_block(f"<div class='topbar'><div><div class='eyebrow'>{esc(position + PAGE_ROLES[key] + checkpoint)}</div>"
               f"<div class='title'>{esc(title)}</div></div><div>{status_html}</div></div>")


def lineage_strip(state: dict[str, Any], active: str, final_label: str) -> None:
    counts = idea_lineage(state)
    steps = [
        ("ideas", "Candidate set", f"{counts['generated']} ideas"),
        ("suitability", "System-checked set", f"{counts['checked']}/3 suitable"),
        ("approval", "Human-approved set", f"{counts['approved']}/3" if counts["approved"] else "pending"),
        ("final", final_label, f"{counts['final']} ideas" if counts["final"] else "pending"),
    ]
    parts = []
    order = [k for k, *_ in steps]
    for i, (key, label, value) in enumerate(steps):
        if i:
            parts.append("<div class='arrow'></div>")
        state = "active" if key == active else ("done" if order.index(key) < order.index(active) else "")
        parts.append(f"<div class='step {state}'><span class='dot'>{i + 1}</span><div>{esc(label)}<b>{esc(value)}</b></div></div>")
    html_block(f"<div class='lineage'>{''.join(parts)}</div>")


def next_step(ctx: dict[str, Any], current: str) -> tuple[str, str] | None:
    steps = visible_steps(ctx)
    keys = [k for k, *_ in steps]
    if current not in keys:
        return None
    for key, label, _ in steps[keys.index(current) + 1:]:
        if nav_status(key, ctx) != "locked":
            return key, label
    return None


def done_footer(ctx: dict[str, Any], current: str, message: str) -> None:
    """Footer for a page whose checkpoint is complete (or that has none): status plus the next page."""
    with st.container(border=True, key="footer_done"):
        left, right = st.columns([4, 1.3], vertical_alignment="center")
        left.markdown(md(message))
        target = next_step(ctx, current)
        if target:
            right.button(f"Next: {target[1]}", icon=":material/arrow_forward:", type="primary", width="stretch",
                         on_click=go, args=(target[0],), key=f"next_{current}")


def decision_summary(ctx: dict[str, Any], stage: str, label: str) -> str:
    rows = checkpoint_decisions(ctx["state"], stage)
    if not rows:
        return f"{label} not reached yet."
    last = rows[-1]
    return f"✓ **{label}** · decision **{last['Decision']}** at {last['Time']}" + (f" · “{last['Advisor note']}”" if last["Advisor note"] else "")


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------

def render_sidebar(ctx: dict[str, Any]) -> None:
    raw = client_record(st.session_state.client_key)["raw"]
    uc = ctx["use_case"]
    with st.sidebar:
        html_block(brand_html())
        mode = "Live model" if st.session_state.get("mode") == LIVE else "Offline"
        html_block(f"<div class='client-chip'>{avatar(raw.get('name', 'Client'))}<div><b>{esc(raw.get('name', 'Client'))}</b><br>"
                   f"<span class='subtle'>{esc(USE_CASES[uc][0])} · {esc(mode)}</span></div></div>"
                   "<span class='side-label'>Workflow</span>")
        current = st.session_state.get("screen")
        action_keys = []
        for i, (key, label, checkpoint) in enumerate(visible_steps(ctx)):
            status = nav_status(key, ctx)
            if status == "action":
                action_keys.append(key)
            number = f"{i + 1} · " if key != "audit" else ""
            state_text = {"action": "Waiting for your decision", "done": "Completed", "failed": "Needs attention",
                          "locked": "Not reached yet"}.get(status, "Available")
            st.button(f"{number}{label}", key=f"nav_{key}",
                      icon=STATUS_NAV_ICON.get(status, ":material/radio_button_unchecked:"), width="stretch",
                      type="primary" if key == current else "secondary", disabled=status == "locked",
                      on_click=go, args=(key,), help=f"Advisor checkpoint {checkpoint} · {state_text}" if checkpoint else state_text)
        if action_keys:
            st.html("<style>" + "".join(
                f".st-key-nav_{k} button {{ border-color: #fab219 !important; box-shadow: inset 3px 0 0 #fab219; }}" for k in action_keys
            ) + "</style>")
        if ctx["outcome"] == "APPROVED":
            st.success("Workflow complete · PDF ready", icon=":material/verified:")
        elif ctx["outcome"] == "REJECTED":
            st.error("Workflow rejected", icon=":material/block:")
        html_block("<span class='side-label'>Navigate</span>")
        st.button("Back to client", icon=":material/arrow_back:", on_click=back_to_client, width="stretch", key="nav_back")
        st.button("All clients", icon=":material/groups:", on_click=reset_to_client_list, width="stretch", key="nav_all_clients")
        html_block(user_chip())


def brand_html() -> str:
    return "<div class='brand'><b>UBS</b> <span>OneView</span></div>"


def avatar(name: str, large: bool = False) -> str:
    initials = "".join(w[0] for w in str(name).replace("Mdm ", "").split()[:2]).upper() or "?"
    return f"<span class='avatar{' lg' if large else ''}'>{esc(initials)}</span>"


def user_chip() -> str:
    return f"<div class='user-chip'>{avatar(RM_NAME)}<div><b>{esc(RM_NAME)}</b><br><span class='subtle' style='font-size:.74rem'>Relationship Manager</span></div></div>"


def render_home_sidebar() -> None:
    """Landing and client home: brand, the client list and the RM."""
    with st.sidebar:
        html_block(brand_html() + "<span class='side-label'>Clients</span>")
        current = st.session_state.get("client_key")
        for key in VISIBLE_CLIENT_KEYS:
            raw = client_record(key)["raw"]
            st.button(raw["name"], key=f"side_{key}", icon=":material/person:", width="stretch",
                      type="primary" if key == current else "secondary", on_click=select_client, args=(key,))
        html_block("<span class='side-label'>New request</span>")
        st.button("Client intake", key="side_intake", icon=":material/person_add:", width="stretch", on_click=reset_to_client_list)
        html_block(user_chip())


# --------------------------------------------------------------------------------------
# Page: client profile (field form with required fields; CP1)
# --------------------------------------------------------------------------------------

def _norm_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not str(r.get("ticker") or "").strip():
            continue
        out.append({"ticker": str(r["ticker"]).strip().upper(), "name": str(r.get("name") or "").strip(),
                    "asset_class": str(r.get("asset_class") or "Equities"), "region": str(r.get("region") or "Global"),
                    "currency": str(r.get("currency") or "SGD"), "weight_pct": round(float(r.get("weight_pct") or 0), 2),
                    "risk_rating": int(r.get("risk_rating") or 3)})
    return out


def _norm_liquidity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("due_months") in (None, "") or r.get("amount") in (None, "") or pd.isna(r.get("due_months")) or pd.isna(r.get("amount")):
            continue
        out.append({"due_months": int(r["due_months"]), "amount": float(r["amount"]), "purpose": str(r.get("purpose") or "Planned need")})
    return out


def profile_form(ctx: dict[str, Any], raw: dict[str, Any], editable: bool) -> tuple[dict[str, Any], list[str]]:
    """The client fact form. Returns (edits vs the current inputs, missing required fields)."""
    sid, uc = ctx["sid"], ctx["use_case"]
    k = lambda name: f"pf_{sid}_{name}"
    left, right = st.columns(2)
    with left:
        goals = st.text_area("Investment objective(s) * — one per line", value="\n".join(raw.get("goals") or []), height=80,
                             key=k("goals"), disabled=not editable)
        c1, c2 = st.columns(2)
        horizon = c1.number_input("Time horizon (months) *", min_value=0, step=6, value=int(raw.get("time_horizon_months") or 0),
                                  key=k("horizon"), disabled=not editable, help="0 = not recorded (R1 blocks the workflow)")
        wealth = c2.number_input("Investable wealth *", min_value=0.0, step=10000.0, value=float(raw.get("total_wealth") or 0),
                                 key=k("wealth"), disabled=not editable)
        risk_opts = RISK_OPTIONS if raw.get("stated_risk_willingness") in RISK_OPTIONS else [raw.get("stated_risk_willingness") or "Moderate"] + RISK_OPTIONS
        exp_opts = EXPERIENCE_OPTIONS if raw.get("experience_level") in EXPERIENCE_OPTIONS else [raw.get("experience_level") or "ETF investor"] + EXPERIENCE_OPTIONS
        c1, c2 = st.columns(2)
        risk = c1.selectbox("Risk willingness *", risk_opts, index=risk_opts.index(raw.get("stated_risk_willingness") or risk_opts[0]),
                            key=k("risk"), disabled=not editable)
        experience = c2.selectbox("Investment experience", exp_opts, index=exp_opts.index(raw.get("experience_level") or exp_opts[0]),
                                  key=k("exp"), disabled=not editable)
        capital = None
        if uc == ADHOC:
            capital = st.number_input("New capital to deploy *", min_value=0.0, step=10000.0, value=float(raw.get("new_capital_amount") or 0),
                                      key=k("capital"), disabled=not editable, help="Added to the portfolio as cash; the ideas deploy it")
    with right:
        st.caption("Liquidity needs * — planned cash needs (an empty table means none)")
        liquidity = st.data_editor(pd.DataFrame(raw.get("liquidity_schedule") or [], columns=["due_months", "amount", "purpose"]),
                                   num_rows="dynamic", hide_index=True, width="stretch", height=110, key=k("liq"), disabled=not editable,
                                   column_config={"due_months": st.column_config.NumberColumn("Due (months)", min_value=1),
                                                  "amount": st.column_config.NumberColumn("Amount", min_value=0.0, format="%.0f"),
                                                  "purpose": st.column_config.TextColumn("Purpose")})
        st.caption("Holdings * — weights must total 100%")
        holdings = st.data_editor(pd.DataFrame(_norm_holdings(raw.get("raw_holdings") or [])), num_rows="dynamic", hide_index=True,
                                  width="stretch", height=180, key=k("hold"), disabled=not editable,
                                  column_config={"weight_pct": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0, format="%.1f"),
                                                 "risk_rating": st.column_config.NumberColumn("Risk 1–5", min_value=1, max_value=5)})
    values = {
        "goals": [g.strip() for g in goals.splitlines() if g.strip()],
        "time_horizon_months": int(horizon) if horizon else None,
        "total_wealth": float(wealth),
        "stated_risk_willingness": risk,
        "experience_level": experience,
        "liquidity_schedule": _norm_liquidity(liquidity.replace({pd.NA: None}).to_dict(orient="records")),
        "raw_holdings": _norm_holdings(holdings.replace({pd.NA: None}).to_dict(orient="records")),
    }
    if capital is not None:
        values["new_capital_amount"] = float(capital)
    missing = []
    if not values["goals"]:
        missing.append("investment objective")
    if not values["time_horizon_months"]:
        missing.append("time horizon")
    if values["total_wealth"] <= 0:
        missing.append("investable wealth")
    total = sum(h["weight_pct"] for h in values["raw_holdings"])
    if not values["raw_holdings"] or abs(total - 100) > 0.5:
        missing.append(f"holdings totalling 100% (now {total:.1f}%)")
    if uc == ADHOC and not values.get("new_capital_amount"):
        missing.append("new capital to deploy")
    current = {**raw, "raw_holdings": _norm_holdings(raw.get("raw_holdings") or []),
               "liquidity_schedule": _norm_liquidity(raw.get("liquidity_schedule") or [])}
    edits = {f: v for f, v in values.items() if v != current.get(f) and not (v is None and current.get(f) in (None, 0))}
    return edits, missing


def render_profile_generation(profile: Any, raw: dict[str, Any]) -> None:
    """Show how the profile was generated, step by step, so the advisor can review each one."""
    steps = st.columns(4)
    with steps[0], card(height="stretch"):
        st.markdown("**1 · Ingest**  \nAdvisor-verified inputs")
        st.json({k: raw.get(k) for k in ("goals", "time_horizon_months", "stated_risk_willingness", "total_wealth",
                                         "new_capital_amount", "liquidity_schedule") if k in raw}, expanded=1)
    with steps[1], card(height="stretch"):
        st.markdown("**2 · Required-field validator**  \nDeterministic")
        check = required_field_validator(raw)
        if check.is_valid:
            html_block(pill("PASS", "All mandatory fields present"))
        else:
            html_block(pill("FAIL", "Missing: " + ", ".join(check.missing_fields)))
    with steps[2], card(height="stretch"):
        st.markdown("**3 · Risk-profile calculator**  \nDeterministic, not the LLM")
        try:
            risk = risk_profile_calculator(raw)
            st.metric("Risk willingness", f"{risk_points(risk.willingness_score)} / 100")
            st.metric("Risk capacity", f"{risk_points(risk.capacity_score)} / 100")
            st.metric("Band (lower of the two)", risk.risk_tier)
        except Exception as exc:
            st.info(f"The risk band cannot be calculated yet: {exc}")
    with steps[3], card(height="stretch"):
        st.markdown("**4 · Standardised profile**  \nWhat the downstream agents receive")
        st.json({"risk_tier": profile.calculated_risk_tier, "risk_band_0_100": risk_band_points(profile.risk_score_range),
                 "time_horizon_months": profile.time_horizon_months, "near_term_liquidity_need": profile.near_term_liquidity_need,
                 "asset_class_weights": profile.current_asset_class_weights,
                 "unresolved_flags": [f.code for f in profile.unresolved_flags]}, expanded=1)


def render_profile(ctx: dict[str, Any]) -> None:
    state, uc = ctx["state"], ctx["use_case"]
    profile = state.get("profile")
    raw = state.get("raw_input") or {}
    pending = ctx["stage"] == "client_profile"
    open_flags = [f for f in (profile.unresolved_flags if profile else []) if not f.is_resolved]
    status_html = pill("FLAG", f"{len(open_flags)} flag(s) to resolve") if open_flags else pill("PASS", "No open flags")
    subtitle = {PRE_MEETING: "Confirm or update what the client tells you. Required fields (*) must be complete to continue.",
                ADHOC: "Confirm the request: amount to deploy, horizon, liquidity and risk. Required fields (*) must be complete.",
                EVENT: "From the client record (confirmed at the last review). Shown for reference.",
                POST_MEETING: "From the client record. Shown for reference; spoken changes are proposed on the meeting record."}[uc]
    page_header(ctx, "profile", "Client profile", subtitle, status_html if profile else "")
    if not profile:
        st.info("The client profile appears when the workflow starts.")
        return
    changes = profile_change_detector(raw, raw.get("last_review") or {}) if uc == PRE_MEETING else []
    tab_names = ["Client details *"]
    if uc == PRE_MEETING:
        tab_names.append(f"Changes since last review ({len(changes)})")
    tab_names += ["Exposure & risk", "Flags & rules", "How it was generated"]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        edits, missing = profile_form(ctx, raw, editable=pending)
    i = 1
    if uc == PRE_MEETING:
        with tabs[i]:
            last = raw.get("last_review") or {}
            if not last:
                st.info("No previous review on record: this is treated as the first review.")
            elif not changes:
                st.success(f"Nothing has changed since the last review on {last.get('date', '—')}. "
                           "The brief is facts and points to discuss; ideas are added only if you ask for them at Checkpoint 2.")
            else:
                html_block(f"<div class='callout'><b>{len(changes)} change(s) since the last review on {esc(last.get('date', '—'))}.</b> "
                           "They appear in the briefing for you to confirm with the client. Ideas are optional (Checkpoint 2).</div>")
                html_block("<table class='grid'><tr><th>Client fact</th><th>Last review</th><th>Now</th></tr>" + "".join(
                    f"<tr><td>{esc(c['label'])}</td><td>{esc(c['before'])}</td><td><b>{esc(c['after'])}</b></td></tr>" for c in changes) + "</table>")
        i += 1
    with tabs[i]:
        left, mid, right = st.columns([1, 1, 1.3])
        with left:
            show_chart(donut(profile.current_asset_class_weights, "Asset class"))
        with mid:
            show_chart(donut(profile.current_region_weights, "Region"))
        with right:
            calc = risk_profile_calculator(raw) if raw.get("time_horizon_months") else None
            total = sum(h.weight_pct for h in profile.parsed_holdings) or 1
            points = [{"Measure": "Current portfolio", "Score": round(sum(h.risk_rating * h.weight_pct for h in profile.parsed_holdings) / total, 2)}]
            if calc:
                points = [{"Measure": "Risk willingness", "Score": calc.willingness_score}, {"Measure": "Risk capacity", "Score": calc.capacity_score}] + points
            show_chart(risk_scale_chart(points, profile.risk_score_range).properties(title="Risk band = lower of willingness and capacity"))
        if profile.new_capital_amount:
            st.caption(f"Includes {fmt_money(profile.new_capital_amount, profile.base_currency)} of new capital held as cash (BIL) until the ideas deploy it.")
    with tabs[i + 1]:
        left, right = st.columns(2)
        with left:
            st.markdown("**Gaps and conflicts detected**")
            if not profile.unresolved_flags:
                html_block(pill("PASS", "No gaps or conflicts"))
            for flag in profile.unresolved_flags:
                status = "FAIL" if flag.severity == "BLOCKING_INTERRUPT" else "FLAG"
                resolved = f" · resolved: {flag.resolution_notes}" if flag.is_resolved else ""
                html_block(f"<div class='callout' style='margin-bottom:.4rem'>{pill(status, flag.code.replace('_', ' ').title())}"
                           f"<br>{esc(flag.message)}<span class='subtle'>{esc(resolved)}</span></div>")
        with right:
            st.markdown("**Profile rules (deterministic)**")
            html_block("<table class='grid'><tr><th>Rule</th><th>Outcome</th><th>Detail</th></tr>" + "".join(
                f"<tr><td>{esc(r.rule_id)} · {esc(r.name)}</td><td>{pill(r.outcome)}</td><td>{esc(r.message)}</td></tr>"
                for r in evaluate_profile_rules(profile)) + "</table>")
    with tabs[i + 2]:
        render_profile_generation(profile, raw)

    if not pending:
        msg = decision_summary(ctx, "client_profile", "Checkpoint 1 · profile confirmed") if uc not in (EVENT, POST_MEETING) else \
            "Client record loaded from the last confirmed review. No profile checkpoint is needed for this workflow."
        done_footer(ctx, "profile", msg)
        return
    allowed = ctx["pending"].allowed_actions
    with st.container(border=True, key="footer"):
        left, mid, right = st.columns([3, 1.4, 1], vertical_alignment="bottom")
        with left:
            note = st.text_input("Checkpoint 1 · Confirm the client profile — note (optional)", key=f"cp1_note_{ctx['sid']}",
                                 placeholder="e.g. Confirmed with the client by phone: purchase brought forward to 2034.")
            if missing:
                st.error("Required before you can continue: " + ", ".join(missing), icon=":material/rule:")
            elif edits:
                st.warning(f"You changed {', '.join(e.replace('_', ' ') for e in edits)}. Save to re-run the profile check.", icon=":material/edit:")
        with mid:
            if edits:
                if st.button("Save changes & re-check", type="primary", icon=":material/refresh:", width="stretch", disabled=bool(missing)):
                    submit_action("regenerate", note or "Advisor updated client inputs.", edits=edits)
            else:
                if st.button("Confirm profile · Next", type="primary", icon=":material/arrow_forward:", width="stretch",
                             disabled=bool(missing) or "approve" not in allowed):
                    submit_action("approve", note)
        with right, st.popover("More actions", icon=":material/more_horiz:", width="stretch"):
            if st.button("Reject and end the session", icon=":material/block:", width="stretch", key="cp1_reject"):
                submit_action("reject", note)


# --------------------------------------------------------------------------------------
# Page: client limits (what UBS can offer this client; UC1 / UC2 checkpoint)
# --------------------------------------------------------------------------------------

def render_limits(ctx: dict[str, Any]) -> None:
    state = ctx["state"]
    offer = state.get("offer_map")
    pending = ctx["stage"] == "client_limits"
    status_pill = {"CLEAR": pill("PASS", "Clear"), "GAPS": pill("FLAG", f"{len(offer.gaps)} gap(s)") if offer else "",
                   "ESCALATE": pill("FLAG", "Escalation open"), "BLOCKED": pill("FAIL", "Blocked: data missing")}.get(offer.status, "") if offer else ""
    page_header(ctx, "limits", "What UBS can offer this client",
                "A rule layer, not an AI judgement. A blocked product is never shown; a gap is listed, never guessed.", status_pill)
    if not offer:
        st.info("The client limits appear once the client profile is confirmed.")
        return
    tabs = st.tabs([f"Eligible products ({len(offer.eligible_products)})", f"Blocked areas ({len(offer.blocked_areas)})",
                    f"Gaps & escalations ({len(offer.gaps)})", "Regulatory profile"])
    with tabs[0]:
        if offer.eligible_products:
            from src.tools.exposure import get_instrument_metadata
            classes: dict[str, float] = {}
            for prod in offer.eligible_products:
                cls = (get_instrument_metadata(prod.ticker) or {}).get("asset_class", "Other")
                classes[cls] = classes.get(cls, 0) + 1
            disclosures: dict[str, int] = {}
            for prod in offer.eligible_products:
                for d in prod.required_disclosures:
                    disclosures[d.replace(" disclosure", "")] = disclosures.get(d.replace(" disclosure", ""), 0) + 1
            left, right = st.columns(2)
            with left:
                show_chart(donut(classes, "Eligible products by asset class"))
            with right:
                frame = pd.DataFrame([{"Disclosure": k, "Products": v} for k, v in disclosures.items()])
                show_chart(alt.Chart(frame, title="Disclosures required (number of products)").mark_bar(cornerRadiusEnd=4, height=16, color=SERIES[0]).encode(
                    y=alt.Y("Disclosure:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)), x=alt.X("Products:Q", title=None),
                    tooltip=["Disclosure", "Products"]).properties(height=36 * len(frame) + 30))
            with st.expander(f"Product list ({len(offer.eligible_products)})"):
                st.markdown(" · ".join(f"**{md(prod.ticker)}** {md(prod.name)}" for prod in offer.eligible_products))
        else:
            st.error("Nothing can be offered until the gaps are resolved.", icon=":material/block:")
    with tabs[1]:
        if offer.blocked_areas:
            html_block("<table class='grid'><tr><th>Blocked area</th><th>Why</th></tr>" + "".join(
                f"<tr><td>{esc(a.area)}</td><td>{esc(a.reason)}</td></tr>" for a in offer.blocked_areas) + "</table>")
        else:
            st.success("No area is blocked for this client.")
    with tabs[2]:
        for flag in offer.escalation_flags:
            st.warning(flag, icon=":material/priority_high:")
        for gap in offer.gaps:
            st.markdown(f"**{md(gap.message)}**  \n{md(gap.resolution)}")
        if not offer.gaps and not offer.escalation_flags:
            st.success("No open gaps or escalations.")
    reg = (state.get("raw_input") or {}).get("regulatory_profile") or {}
    with tabs[3]:
        c1, c2, c3 = st.columns(3)
        domicile = c1.text_input("Domicile", value=reg.get("domicile", ""), disabled=not pending, key=f"reg_dom_{ctx['sid']}")
        booking = c2.text_input("Booking centre", value=reg.get("booking_centre", ""), disabled=not pending, key=f"reg_book_{ctx['sid']}")
        agreement = c3.selectbox("Agreement type", ["", "Advisory", "Discretionary", "Execution-only"],
                                 index=["", "Advisory", "Discretionary", "Execution-only"].index(reg.get("agreement_type", "")) if reg.get("agreement_type", "") in ("", "Advisory", "Discretionary", "Execution-only") else 0,
                                 disabled=not pending, key=f"reg_agr_{ctx['sid']}")
        d1, d2, d3 = st.columns(3)
        klass = d1.selectbox("Investor classification", ["", "Retail", "Accredited", "Institutional"],
                             index=["", "Retail", "Accredited", "Institutional"].index(reg.get("investor_classification", "")) if reg.get("investor_classification", "") in ("", "Retail", "Accredited", "Institutional") else 0,
                             disabled=not pending, key=f"reg_cls_{ctx['sid']}")
        cka = d2.text_input("CKA valid until (YYYY-MM-DD)", value=reg.get("cka_valid_until", "") or "", disabled=not pending, key=f"reg_cka_{ctx['sid']}")
        pep = d3.checkbox("Politically exposed person (PEP)", value=bool(reg.get("pep_status")), disabled=not pending, key=f"reg_pep_{ctx['sid']}")
        forms = reg.get("tax_forms") or [{}]
        form_name = forms[0].get("form", "W-8BEN")
        tax_until = st.text_input(f"Tax form {form_name} valid until (YYYY-MM-DD)", value=forms[0].get("valid_until", "") or "",
                                  disabled=not pending, key=f"reg_tax_{ctx['sid']}")
        edits = {"domicile": domicile.strip(), "booking_centre": booking.strip(), "agreement_type": agreement,
                 "investor_classification": klass, "cka_valid_until": cka.strip() or None, "pep_status": pep,
                 "tax_forms": [{"form": form_name, "valid_until": tax_until.strip()}] if tax_until.strip() else []}
        changed = {k: v for k, v in edits.items() if v != (reg.get(k) if k != "tax_forms" else (reg.get(k) or []))
                   and not (v in ("", None) and reg.get(k) in ("", None))}
    if not pending:
        done_footer(ctx, "limits", decision_summary(ctx, "client_limits", "Client limits confirmed"))
        return
    allowed = ctx["pending"].allowed_actions
    with st.container(border=True, key="footer"):
        left, mid, right = st.columns([3, 1.4, 1], vertical_alignment="bottom")
        with left:
            needs_note = bool(offer.escalation_flags)
            note = st.text_input("Check limits — note" + (" (required: escalation open, say how it is being handled)" if needs_note else " (optional)"),
                                 key=f"lim_note_{ctx['sid']}", placeholder="e.g. CKA refresh booked for next week; specified products stay off the table until then.")
            if offer.status == "BLOCKED":
                st.error("Supply the missing regulatory data in the Regulatory profile tab, then save to re-check.", icon=":material/rule:")
        with mid:
            if changed:
                if st.button("Save changes & re-check", type="primary", icon=":material/refresh:", width="stretch"):
                    submit_action("regenerate", note or "Advisor updated the regulatory profile.", edits={"regulatory_profile": edits})
            elif st.button("Confirm limits · Next", type="primary", icon=":material/arrow_forward:", width="stretch",
                           disabled="approve" not in allowed or (needs_note and not note.strip())):
                submit_action("approve", note)
        with right, st.popover("More actions", icon=":material/more_horiz:", width="stretch"):
            if st.button("Reject and end the session", icon=":material/block:", width="stretch", key="lim_reject"):
                submit_action("reject", note)


# --------------------------------------------------------------------------------------
# Page: meeting record (UC3: summary, actions, profile changes, follow-up email, compliance record)
# --------------------------------------------------------------------------------------

def render_meeting(ctx: dict[str, Any]) -> None:
    state = ctx["state"]
    record = state.get("meeting_record")
    approved = ctx["outcome"] == "APPROVED"
    page_header(ctx, "meeting", "After the meeting: what was agreed",
                "OneView drafts; you confirm. Every line links to the transcript, and the client's agreement is never changed automatically.",
                (pill("PASS", "Confirmed · record saved") if approved else pill("FLAG", "Draft: your confirmation needed")) if record else "")
    if not record:
        st.info("The record is drafted from the transcript when the workflow starts.")
        return
    pending = ctx["stage"] == "record_approval"
    open_actions = [a for a in record.actions if a.status == "FLAGGED"]
    undecided = [c for c in record.profile_changes if c.decision == "PENDING"]
    tabs = st.tabs(["Summary", f"Actions ({len(record.actions)})", f"Profile changes ({len(record.profile_changes)})",
                    "Follow-up email", "Compliance checks"])
    with tabs[0]:
        left, right = st.columns(2)
        for col, title, lines in ((left, "What the client said", record.client_said), (right, "Decisions made", record.decisions)):
            with col, card(height="stretch"):
                st.markdown(f"**{title}**")
                if not lines:
                    st.caption("Nothing recorded.")
                for line in lines:
                    refs = ", ".join(str(n) for n in line.source_lines) or "none"
                    st.markdown(f"- {md(line.text)} {info_pill('transcript line ' + refs)}" + ("" if line.verified else " " + pill("FLAG", "check this line")),
                                unsafe_allow_html=True)
    edits: dict[str, Any] = {}
    with tabs[1]:
        for a in record.actions:
            with card():
                head = f"**{md(a.description)}** " + pill("PASS" if a.status == "CONFIRMED" else "FLAG", a.status.title())
                st.markdown(head, unsafe_allow_html=True)
                if a.flag_reason:
                    st.caption(f"Flagged: {a.flag_reason}")
                c1, c2 = st.columns(2)
                owner = c1.text_input("Owner", value=a.owner or "", key=f"own_{a.action_id}_{ctx['sid']}", disabled=not pending)
                due = c2.text_input("Due", value=a.due_date or "", key=f"due_{a.action_id}_{ctx['sid']}", disabled=not pending)
                if (owner.strip() or None) != a.owner or (due.strip() or None) != a.due_date:
                    edits.setdefault("actions", []).append({"action_id": a.action_id, "owner": owner.strip() or None, "due_date": due.strip() or None})
    with tabs[2]:
        if not record.profile_changes:
            st.success("No change in the client's circumstances was spoken.")
        for c in record.profile_changes:
            with card():
                status = c.check.status if c.check else "FLAG"
                st.markdown(f"**{md(c.label)}**: {md(c.before or '—')} → **{md(c.after)}** " + pill(status, f"Critic: {status}")
                            + info_pill(c.decision.title()) + info_pill("transcript line " + ", ".join(map(str, c.source_lines))), unsafe_allow_html=True)
                st.caption(f"“{c.text}”")
                for line in (c.check.implications if c.check else []):
                    st.markdown(f"- {md(line)}")
                for line in (c.check.new_gaps if c.check else []) + [f"Newly blocked: {a}" for a in (c.check.newly_blocked_areas if c.check else [])]:
                    st.markdown(f"- :orange[{md(line)}]")
                if pending:
                    b1, b2, b3 = st.columns([3, 1, 1], vertical_alignment="bottom")
                    why = b1.text_input("Your reasoning (required unless the Critic passed it)", key=f"chg_note_{c.change_id}_{ctx['sid']}")
                    if b2.button("Approve update", key=f"chg_ok_{c.change_id}", icon=":material/check:", width="stretch"):
                        submit_action("approve_change", why, item_id=c.change_id)
                    if b3.button("Reject", key=f"chg_no_{c.change_id}", icon=":material/close:", width="stretch"):
                        submit_action("reject_change", why, item_id=c.change_id)
    with tabs[3]:
        subject = st.text_input("Subject", value=record.follow_up_email.subject, key=f"em_sub_{ctx['sid']}", disabled=not pending)
        body = st.text_area("Draft email (you edit and send it)", value=record.follow_up_email.body, height=200, key=f"em_body_{ctx['sid']}", disabled=not pending)
        if subject != record.follow_up_email.subject:
            edits["email_subject"] = subject
        if body != record.follow_up_email.body:
            edits["email_body"] = body
    with tabs[4]:
        for r in record.compliance_checks:
            st.markdown(f"{pill(r.outcome)} **{md(r.rule_id)} {md(r.name)}** · {md(r.message)}", unsafe_allow_html=True)
        st.caption(f"Transcript fingerprint {record.transcript_fingerprint} · {record.transcript_lines} lines · agreement type {record.agreement_type}")

    if approved:
        with st.container(border=True, key="footer_done"):
            left, right = st.columns([3, 1.3], vertical_alignment="center")
            left.markdown(decision_summary(ctx, "record_approval", "Record confirmed") + " · it feeds the next pre-meeting briefing.")
            with right:
                render_pdf_download(ctx, "Download compliance record PDF")
        return
    if not pending:
        return
    allowed = ctx["pending"].allowed_actions
    with st.container(border=True, key="footer"):
        left, mid, right = st.columns([3, 1.4, 1], vertical_alignment="bottom")
        with left:
            note = st.text_input("Confirm the record — note (printed on the record)", key=f"rec_note_{ctx['sid']}")
            if open_actions:
                st.warning(f"{len(open_actions)} action(s) still need an owner and a due date.", icon=":material/rule:")
            if undecided:
                st.warning(f"{len(undecided)} profile change(s) still need your decision.", icon=":material/rule:")
        with mid:
            if edits:
                if st.button("Save edits", type="primary", icon=":material/save:", width="stretch"):
                    submit_action("edit_record", note, edits=edits)
            elif st.button("Confirm · save record", type="primary", icon=":material/verified:", width="stretch", disabled="approve" not in allowed):
                submit_action("approve", note)
        with right, st.popover("More actions", icon=":material/more_horiz:", width="stretch"):
            if st.button("Re-draft from the transcript", icon=":material/refresh:", width="stretch"):
                submit_action("regenerate", note or "Re-draft.")
            if st.button("Reject and end the session", icon=":material/block:", width="stretch", key="rec_reject"):
                submit_action("reject", note)


# --------------------------------------------------------------------------------------
# Page: research / market landscape (CP2)
# --------------------------------------------------------------------------------------

def research_tabs(brief: Any, profile: Any, names: list[str]) -> None:
    tabs = st.tabs(names)
    with tabs[0]:
        findings = sorted(brief.findings, key=lambda f: f.materiality_rank)
        for col, finding in zip(st.columns(max(len(findings[:3]), 1)), findings[:3]):
            with col, card(height="stretch"):
                html_block(f"<div class='num'>{finding.materiality_rank:02d}</div><b>{esc(finding.theme)}</b>" + source_links(finding.cited_source_ids))
                with st.expander("Details"):
                    st.markdown(f"{md(finding.claim)}\n\n**For this client:** {md(finding.relevance_to_client)}")
        left, right = st.columns([1, 1.2])
        with right:
            html_block("<div class='card-label' style='margin-top:.3rem'>Key market risks</div>" + "".join(info_pill(r) for r in brief.key_market_risks))
        with left:
            if brief.conflicting_signals:
                # Slide 29: conflicting market signals are surfaced as uncertainty, never resolved by picking a side.
                html_block(f"<div class='callout'>{pill('FLAG', 'Sources disagree')}<br>" + "<br>".join(esc(c) for c in brief.conflicting_signals) + "</div>")
            cites: dict[str, float] = {}
            for finding in brief.findings:
                for sid in finding.cited_source_ids:
                    label = (st.session_state.get("source_index") or {}).get(sid, {}).get("short", "Source")
                    cites[label] = cites.get(label, 0) + 1
            show_chart(donut(cites, "Evidence by publisher", height=170))
    with tabs[1]:
        frame = pd.DataFrame([m.model_dump() for m in brief.quantitative_signals.values()])
        if frame.empty:
            st.info("No quantitative signals were retrieved.")
        else:
            held = {h.ticker for h in profile.parsed_holdings}
            frame["Held"] = frame["ticker"].map(lambda t: "In portfolio" if t in held else "Universe")
            left, right = st.columns(2)
            held_scale = alt.Scale(domain=["In portfolio", "Universe"], range=[SERIES[0], SERIES[1]])
            with left:
                base = alt.Chart(frame.dropna(subset=["annualized_volatility", "annualized_return_1y"]), title="Risk vs return (1y)").encode(
                    x=alt.X("annualized_volatility:Q", title="Volatility (annualised)", axis=alt.Axis(format="%")),
                    y=alt.Y("annualized_return_1y:Q", title="1y return", axis=alt.Axis(format="%")))
                dots = base.mark_point(filled=True, size=120, stroke="#ffffff", strokeWidth=2).encode(
                    color=alt.Color("Held:N", scale=held_scale, legend=alt.Legend(title=None)),
                    tooltip=["ticker", alt.Tooltip("annualized_return_1y:Q", format=".1%"), alt.Tooltip("annualized_volatility:Q", format=".1%")])
                show_chart((dots + base.mark_text(dx=9, align="left", color=MUTED).encode(text="ticker:N")).properties(height=240))
            with right:
                bar = alt.Chart(frame.dropna(subset=["sharpe_ratio"]), title="Sharpe ratio").mark_bar(cornerRadiusEnd=4, height=16).encode(
                    y=alt.Y("ticker:N", sort="-x", title=None), x=alt.X("sharpe_ratio:Q", title="Sharpe"),
                    color=alt.Color("Held:N", scale=held_scale, legend=None), tooltip=["ticker", alt.Tooltip("sharpe_ratio:Q", format=".2f")])
                show_chart(bar.properties(height=240))
            labels = {"ticker": "Ticker", "annualized_return_1y": "1y return", "annualized_volatility": "Volatility",
                      "max_drawdown": "Max drawdown", "momentum_12_1": "Momentum 12-1", "forward_pe": "Fwd P/E", "yield_to_maturity": "Yield"}
            percent = {"annualized_return_1y", "annualized_volatility", "max_drawdown", "momentum_12_1", "yield_to_maturity"}
            st.dataframe(frame[list(labels)], hide_index=True, width="stretch", height=min(35 * len(frame) + 40, 190),
                         column_config={c: st.column_config.NumberColumn(label, format="percent" if c in percent else "%.1f")
                                        if c != "ticker" else st.column_config.TextColumn(label) for c, label in labels.items()})
    with tabs[2]:
        left, right = st.columns([1, 1.3])
        sentiments = brief.sentiment_signals
        with left:
            frame = pd.DataFrame([{"Scope": k, "Score": v.sentiment_score, "Label": v.overall_sentiment} for k, v in sentiments.items()])
            if not frame.empty:
                chart = alt.Chart(frame, title="Sentiment (−1 bearish … +1 bullish)").mark_bar(cornerRadiusEnd=4, height=26).encode(
                    y=alt.Y("Scope:N", title=None), x=alt.X("Score:Q", scale=alt.Scale(domain=[-1, 1]), title=None),
                    color=alt.condition(alt.datum.Score >= 0, alt.value(SERIES[0]), alt.value(SERIES[7])),
                    tooltip=["Scope", "Label", alt.Tooltip("Score:Q", format=".2f")])
                zero = alt.Chart(pd.DataFrame([{"x": 0}])).mark_rule(color=MUTED).encode(x="x:Q")
                show_chart((chart + zero).properties(height=70 * len(frame) + 40))
        with right:
            for k, v in sentiments.items():
                tones = {"Positive": len(v.bullish_signals), "Negative": len(v.bearish_signals), "Neutral": len(v.neutral_catalysts)}
                show_chart(donut({tone: n for tone, n in tones.items() if n}, f"Sources by tone · {k}"))
    with tabs[3]:
        max_age = load_policy()["rules"]["R8_evidence_freshness"]["max_age_days"]
        index = st.session_state.get("source_index") or {}
        frame = pd.DataFrame([{"Source": f"{index.get(s.source_id, {}).get('short', 'Source')} · {index.get(s.source_id, {}).get('date', '')}",
                               "Title": s.title, "Age (days)": s.age_days or 0, "Status": "PASS" if s.is_fresh else "FLAG"}
                              for s in brief.validated_sources])
        left, right = st.columns([1, 1.3])
        with left:
            order = list(frame.sort_values("Age (days)", ascending=False)["Source"])
            bars = alt.Chart(frame, title=f"Source age vs {max_age}-day freshness limit (R8)").mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("Source:N", title=None, scale=alt.Scale(domain=order, paddingInner=0.35), axis=alt.Axis(labelLimit=200)),
                x=alt.X("Age (days):Q", title="Age (days)", scale=alt.Scale(domain=[0, max(max_age * 1.2, frame['Age (days)'].max() + 3)])),
                color=alt.Color("Status:N", scale=alt.Scale(domain=["PASS", "FLAG"], range=[STATUS_COLORS["PASS"], STATUS_COLORS["FLAG"]]), legend=alt.Legend(title=None)),
                tooltip=["Source", "Title", "Age (days)"])
            limit = alt.Chart(pd.DataFrame([{"Age (days)": max_age}])).mark_rule(strokeDash=[4, 3], color=STATUS_COLORS["FAIL"], strokeWidth=2).encode(x="Age (days):Q")
            show_chart((bars + limit).properties(height=45 * len(frame) + 60))
        with right, card():
            html_block(source_links([s.source_id for s in brief.validated_sources]))


def render_research(ctx: dict[str, Any]) -> None:
    state, uc = ctx["state"], ctx["use_case"]
    brief = state.get("research_brief")
    raw = state.get("raw_input") or {}
    since = (raw.get("last_review") or {}).get("date")
    title = "Market landscape: the last 3 months" if uc == PRE_MEETING else "Evidence-backed research brief"
    subtitle = (f"CIO research first, then news you entered; facts only, no forecasts. Every claim has a dated source (last review: {since or 'none'})."
                if uc == PRE_MEETING else "Client-relevant findings with quantitative and qualitative signals; every claim has a dated source.")
    stale = [s for s in brief.validated_sources if not s.is_fresh] if brief else []
    status_html = (pill("FLAG", f"{len(stale)} stale source(s)") if stale else pill("PASS", f"All sources ≤ {brief.window_days} days")) if brief else ""
    page_header(ctx, "research", title, subtitle, status_html)
    if not brief:
        st.info("Research appears after Checkpoint 1 is confirmed.")
        return
    research_tabs(brief, state["profile"], ["Findings", "Quantitative signals", "Sentiment", "Sources & freshness"])

    if ctx["stage"] != "market_intelligence":
        msg = decision_summary(ctx, "market_intelligence", "Checkpoint 2 · research approved")
        if uc == PRE_MEETING:
            msg += " · ideas for the brief: **" + ("yes (your choice)" if state.get("include_ideas") else "no") + "**"
        done_footer(ctx, "research", msg)
        return
    changes = state.get("profile_changes") or []
    with st.container(border=True, key="footer"):
        left, mid, right = st.columns([3, 1.4, 1], vertical_alignment="bottom")
        with left:
            note = st.text_input("Checkpoint 2 · Review the research — note (required to re-run with feedback)", key=f"cp2_note_{ctx['sid']}",
                                 placeholder="e.g. Relevant to the home goal. Or: focus more on short-duration bonds.")
            include = None
            if uc == PRE_MEETING:
                include = st.checkbox(f"Also prepare 3 investment ideas for this meeting ({len(changes)} profile change(s) since last review)",
                                      value=False, key=f"cp2_ideas_{ctx['sid']}",
                                      help="Off by default: the briefing is facts and points to discuss. Ideas are the Discover workflow")
        with mid:
            label = "Approve · Next" if uc != PRE_MEETING else ("Approve · generate ideas" if include else "Approve · draft brief")
            if st.button(label, type="primary", icon=":material/arrow_forward:", width="stretch"):
                submit_action("approve", note, **({"edits": {"include_ideas": bool(include)}} if uc == PRE_MEETING else {}))
        with right, st.popover("More actions", icon=":material/more_horiz:", width="stretch"):
            if st.button("Re-run research with my feedback", icon=":material/refresh:", width="stretch", disabled=not note.strip()):
                submit_action("regenerate", note)
            if st.button("Reject and end the session", icon=":material/block:", width="stretch", key="cp2_reject"):
                submit_action("reject", note)


# --------------------------------------------------------------------------------------
# Page: market event & client impact (UC3 decision)
# --------------------------------------------------------------------------------------

def render_event(ctx: dict[str, Any]) -> None:
    state = ctx["state"]
    impact = state.get("event_impact")
    brief = state.get("research_brief")
    if not impact:
        page_header(ctx, "event", "Event & client impact", "")
        st.info("The client impact appears once the event research has run.")
        return
    suggestion = {"MAINTAIN": "PASS", "MONITOR": "FLAG", "REVISE": "FAIL"}[impact.suggested_decision]
    page_header(ctx, "event", impact.event_title,
                "What changed, where the client is exposed, and whether suitability still holds. A market event triggers a review, not a trade.",
                pill(suggestion, f"Suggested: {impact.suggested_decision.title()}"))
    tabs = st.tabs(["What changed", "Client impact", "Suitability re-check", "Advice under review"])
    with tabs[0]:
        left, right = st.columns([1.4, 1])
        with left:
            html_block(f"<div class='callout'><b>Event</b><br>{esc(impact.event_description)}</div>")
            with st.container(border=False):
                for f in sorted(brief.findings, key=lambda f: f.materiality_rank)[:3]:
                    with card():
                        html_block(f"<b>{esc(f.theme)}</b><div class='kv'>{esc(f.claim)}</div>"
                                   f"<div class='kv'><b>Client relevance</b><br>{esc(f.relevance_to_client)}</div>"
                                   f"<div class='subtle' style='font-size:.78rem'>{esc(', '.join(f.cited_source_ids))}</div>")
        with right:
            a, b = st.columns(2)
            a.metric("Estimated portfolio move", f"{impact.portfolio_move_pct:+.2f}%")
            b.metric("Portfolio risk (0-100)", risk_points(impact.risk_score_after), risk_points(impact.risk_score_after) - risk_points(impact.risk_score_before), delta_color="inverse")
            st.markdown("**Key market risks**")
            st.markdown("\n".join(f"- {md(r)}" for r in brief.key_market_risks))
    with tabs[1]:
        frame = pd.DataFrame([{"Holding": f"{h.ticker} · {h.asset_class}", "Weight %": h.weight_pct, "Assumed move %": h.shock_pct,
                               "Contribution": h.contribution_pct} for h in impact.holdings])
        left, right = st.columns([1.3, 1])
        with left:
            show_chart(signed_bars(frame, "Holding", "Contribution", "Estimated contribution to the portfolio move (% points)"))
        with right:
            st.dataframe(frame, hide_index=True, width="stretch", height=min(35 * len(frame) + 40, 260),
                         column_config={"Weight %": st.column_config.NumberColumn(format="%.1f"), "Assumed move %": st.column_config.NumberColumn(format="%+.1f"),
                                        "Contribution": st.column_config.NumberColumn(format="%+.2f")})
            st.caption("Assumed moves come from an illustrative event scenario (deterministic tool), not a forecast.")
    with tabs[2]:
        html_block("<table class='grid'><tr><th>Rule</th><th>Outcome</th><th>Detail</th></tr>" + "".join(
            f"<tr><td>{esc(r.rule_id)} · {esc(r.name)}</td><td>{pill(r.outcome)}</td><td>{esc(r.message)}</td></tr>" for r in impact.rule_results) + "</table>")
        html_block(f"<div class='callout' style='margin-top:.6rem'><b>Why {impact.suggested_decision.lower()}</b><br>{esc(impact.rationale)}</div>")
    with tabs[3]:
        advice = client_record(st.session_state.client_key).get("current_advice")
        if advice:
            st.markdown(f"**Advice approved on {advice['date']}** (assumed implemented)")
            html_block("<table class='grid'><tr><th>Idea</th><th>Action</th><th>Target</th><th>Role</th></tr>" + "".join(
                f"<tr><td><b>{esc(i['ticker'])}</b> · {esc(i['name'])}</td><td>{esc(i['action'])}</td><td>{i['target_weight_pct']:.1f}%</td>"
                f"<td>{esc(i['portfolio_role'].title())}</td></tr>" for i in advice["ideas"]) + "</table>")
        else:
            st.info("No recommendation has been approved in this session yet; the review covers the current portfolio.")
        show_chart(donut(state["profile"].current_asset_class_weights, "Current portfolio by asset class", height=180))

    if ctx["stage"] != "event_decision":
        d = state.get("event_decision") or {}
        done_footer(ctx, "event", f"✓ **Decision recorded: {d.get('decision', '').title()}** · “{d.get('rationale', '')}”"
                    + (f" · follow-up: {d['follow_up']}" if d.get("follow_up") else ""))
        return
    with st.container(border=True, key="footer"):
        c1, c2, c3 = st.columns([1.6, 2.4, 1.1], vertical_alignment="bottom")
        options = ["Maintain", "Monitor", "Revise"]
        with c1:
            choice = st.radio("Advisor decision", options, index=options.index(impact.suggested_decision.title()),
                              key=f"cp_event_choice_{ctx['sid']}", horizontal=True)
        with c2:
            note = st.text_input("Decision rationale (required)", key=f"cp_event_note_{ctx['sid']}",
                                 placeholder="e.g. Bond losses are within tolerance; the home goal is still on track.")
            follow = st.text_input("Monitoring trigger or follow-up date (required for Monitor)", key=f"cp_event_follow_{ctx['sid']}",
                                   placeholder="e.g. Review if 10-year yields exceed 4.75% or by 30 Nov", disabled=choice != "Monitor")
        with c3:
            ready = note.strip() and (choice != "Monitor" or follow.strip())
            label = {"Maintain": "Record · maintain", "Monitor": "Record · monitor", "Revise": "Revise · generate ideas"}[choice]
            if st.button(label, type="primary", icon=":material/arrow_forward:", width="stretch", disabled=not ready):
                submit_action(choice.lower(), note, **({"edits": {"follow_up": follow}} if choice == "Monitor" else {}))


# --------------------------------------------------------------------------------------
# Pages: candidate ideas, suitability check, approval (CP3)
# --------------------------------------------------------------------------------------

def idea_card(idea: dict[str, Any], status: str | None, tag: str | None = None, height: int | str = "stretch") -> None:
    with card(height=height):
        badges = (pill(status) if status else "") + (info_pill(tag) if tag else "") + info_pill(idea.get("lane", "ALIGNED").title())
        stats = (f"<div class='stats'><div>Action<b>{esc(idea['action'])}</b></div><div>Target weight<b>{idea['target_weight_pct']:.1f}%</b></div>"
                 f"<div>Conviction<b>{esc(idea['conviction'])}</b></div></div>")
        amount = idea.get("indicative_amount_range") or []
        html_block(f"<div class='eyebrow'>Idea {idea['slot_number']} · {esc(idea['portfolio_role'].title())}</div>"
                   f"<div class='card-ticker'>{esc(idea['ticker'])}</div><div class='subtle'>{esc(idea['name'])}</div>{badges}{stats}"
                   + (f"<div class='subtle'>Indicative {fmt_money(amount[0])}–{fmt_money(amount[1])}</div>" if len(amount) == 2 and not idea.get("withheld") else ""))


def round_trigger_text(trigger: dict[str, Any]) -> str:
    slots = ", ".join(str(s) for s in trigger.get("slots", []))
    return {"initial": "Initial candidate set (all 3 slots)",
            "auto_regeneration": f"Automatic regeneration of slot {slots} after a critic FAIL",
            "advisor_revision": f"Advisor revision of slot {slots}",
            "allocation_edit": "Advisor edited an allocation; re-validated"}.get(trigger.get("source"), trigger.get("source", ""))


def final_label(ctx: dict[str, Any]) -> str:
    return "Meeting brief" if ctx["use_case"] == PRE_MEETING else "Memo"


def render_ideas(ctx: dict[str, Any]) -> None:
    state = ctx["state"]
    rounds = state.get("idea_rounds") or []
    page_header(ctx, "ideas", "Candidate idea set",
                "Three complementary ideas from the approved universe, generated as one portfolio-aware set. The agent cannot approve its own ideas.",
                info_pill(f"{len(rounds)} round(s)"))
    if not rounds:
        st.info("Candidate ideas appear after the research checkpoint.")
        return
    lineage_strip(state, "ideas", final_label(ctx))
    tabs = st.tabs(["Candidate set", "Idea rationale", "Proposed allocation", "Lock / regenerate rounds"])
    labels = {r["round"]: f"Round {r['round']}" + (" · original" if r["round"] == 1 else "") for r in rounds}
    with tabs[0]:
        chosen = st.pills("Generation round", list(labels), format_func=labels.get, default=1, key=f"round_pick_{ctx['sid']}",
                          label_visibility="collapsed") or 1
        rnd = next(r for r in rounds if r["round"] == chosen)
        st.caption(round_trigger_text(rnd["trigger"]))
        regenerated = set(rnd["trigger"].get("slots", [])) if rnd["trigger"].get("source") != "initial" else set()
        for col, idea in zip(st.columns(3), sorted(rnd["ideas"], key=lambda i: i["slot_number"])):
            with col:
                tag = None if not regenerated else ("Regenerated" if idea["slot_number"] in regenerated else "Locked")
                idea_card(idea, rnd["statuses"].get(idea["idea_id"]), tag)
    ideas = state["candidate_ideas"].model_dump()["ideas"]
    with tabs[1]:
        pick = st.pills("Idea", [i["slot_number"] for i in ideas], default=1, key=f"rationale_pick_{ctx['sid']}", label_visibility="collapsed",
                        format_func=lambda s: next(f"Idea {i['slot_number']} · {i['ticker']}" for i in ideas if i["slot_number"] == s)) or 1
        idea = next(i for i in ideas if i["slot_number"] == pick)
        left, right = st.columns([1.2, 1])
        with left, card(height="stretch"):
            html_block(kv("Why it suits the client", idea["client_fit_rationale"])
                       + "<div class='kv'><b>Risks to raise</b><br>" + "".join(info_pill(r) for r in idea["key_risks"]) + "</div>"
                       + "<div class='kv'><b>Sources</b></div>" + source_links(idea["cited_evidence_ids"]))
            with st.expander("Full rationale"):
                st.markdown(f"**Thesis:** {md(idea['thesis'])}\n\n**Why now:** {md(idea['why_now'])}\n\n"
                            f"**What would change our view:** {md(idea['invalidation_trigger'])}\n\n"
                            f"**Alternative considered:** {md(idea['alternative_considered'] or '—')}")
        with right:
            current = holding_weights(state["profile"].parsed_holdings).get(idea["ticker"], 0.0)
            rng = idea["allocation_range"]
            frame = pd.DataFrame([{"Weight": "Current", "Value": current}, {"Weight": "Range low", "Value": rng["min_pct"]},
                                  {"Weight": "Target", "Value": idea["target_weight_pct"]}, {"Weight": "Range high", "Value": rng["max_pct"]}])
            show_chart(alt.Chart(frame, title=f"{idea['ticker']}: portfolio weight %").mark_bar(cornerRadiusEnd=4, height=18).encode(
                y=alt.Y("Weight:N", sort=list(frame["Weight"]), title=None), x=alt.X("Value:Q", title=None),
                color=alt.condition(alt.datum.Weight == "Current", alt.value(BEFORE_COLOR), alt.value(AFTER_COLOR)),
                tooltip=["Weight", alt.Tooltip("Value:Q", format=".1f")]).properties(height=170))
    with tabs[2]:
        current = holding_weights(state["profile"].parsed_holdings)
        frame = pd.DataFrame([{"Category": f"{i['ticker']} ({i['action']})", "Before": current.get(i["ticker"], 0.0), "After": i["target_weight_pct"]} for i in ideas])
        left, right = st.columns([1.4, 1])
        with left:
            show_chart(before_after_bars(frame, "Current vs proposed weight per idea", height=200))
        with right:
            show_chart(donut({f"{i['portfolio_role'].title()} · {i['ticker']}": i["target_weight_pct"] for i in ideas}, "Target weight by role"))
    with tabs[3]:
        header = "<tr><th>Round</th><th>Trigger</th>" + "".join(f"<th>Slot {s}</th>" for s in (1, 2, 3)) + "<th>Set verdict</th></tr>"
        body = ""
        for rnd in rounds:
            regen = set(rnd["trigger"].get("slots", [])) if rnd["trigger"].get("source") != "initial" else set()
            cells = ""
            for s in (1, 2, 3):
                idea = next(i for i in rnd["ideas"] if i["slot_number"] == s)
                lock = "" if not regen else (" · regenerated" if s in regen else " · locked")
                cells += f"<td><b>{esc(idea['ticker'])}</b> {pill(rnd['statuses'].get(idea['idea_id'], 'PASS'))}<span class='subtle'>{lock}</span></td>"
            body += f"<tr><td>{rnd['round']}</td><td>{esc(round_trigger_text(rnd['trigger']))}</td>{cells}<td>{pill(rnd['overall_status'])}</td></tr>"
        html_block(f"<table class='grid'>{header}{body}</table>")
        history = state.get("regeneration_history") or []
        if history:
            st.markdown("**Sent back by the critic**")
            st.dataframe(pd.DataFrame([{"Slot": h["slot"], "Rejected idea": h["ticker"], "Attempt": h["attempt"], "Rules failed": ", ".join(h["rules"]),
                                        "Critic feedback": h["reason"]} for h in history]), hide_index=True, width="stretch", height=140)
        else:
            st.caption("No idea was sent back.")
    done_footer(ctx, "ideas", f"Candidate set generated in {len(rounds)} round(s). The Suitability Critic checks every set before you see it.")


def render_suitability(ctx: dict[str, Any]) -> None:
    state = ctx["state"]
    assessment = state.get("assessment")
    page_header(ctx, "suitability", "System-checked suitable set",
                "Deterministic rules calculate measurable risk; the critic assesses each idea. No FAIL can reach the client.",
                pill(assessment.overall_status, f"Overall {assessment.overall_status}") if assessment else "")
    if not assessment:
        st.info("The suitability check runs automatically after the candidate set is generated.")
        return
    lineage_strip(state, "suitability", final_label(ctx))
    ideas, profile = state["candidate_ideas"].ideas, state["profile"]
    tabs = st.tabs(["Idea verdicts", "Portfolio before → after", "Exposure shift & stress", "Rule matrix"])
    with tabs[0]:
        evaluations = {ie.idea_id: ie for ie in assessment.idea_evaluations}
        for col, idea in zip(st.columns(3), sorted(ideas, key=lambda i: i.slot_number)):
            ie = evaluations.get(idea.idea_id)
            with col, card(height="stretch"):
                html_block(f"<div class='eyebrow'>Idea {idea.slot_number} · {esc(idea.portfolio_role.title())}</div>"
                           f"<div class='card-ticker'>{esc(idea.ticker)} {pill(ie.status) if ie else ''}</div>"
                           f"<div class='subtle'>{esc(idea.name)} · {idea.action} {idea.target_weight_pct:.1f}%</div>")
                issues = [r for r in (ie.rule_results if ie else []) if r.outcome != "PASS"]
                for r in issues:
                    html_block(f"<div class='kv'>{pill(r.outcome, r.rule_id)} {esc(r.message)}</div>")
                if not issues:
                    html_block(f"<div class='kv'>{pill('PASS', 'All idea rules')} {len(ie.rule_results) if ie else 0} checks passed.</div>")

    with tabs[1]:
        m = assessment.after_metrics
        a, b, c, d = st.columns(4)
        a.metric("Portfolio risk (0-100)", risk_points(m.after_risk_score),
                 f"{risk_points(m.after_risk_score) - risk_points(m.before_risk_score):+d} vs before", delta_color="inverse")
        b.metric("Client band", risk_band_points(profile.risk_score_range), profile.calculated_risk_tier, delta_color="off", delta_arrow="off")
        c.metric("Growth assets", f"{m.growth_asset_pct:.1f}%")
        d.metric("Liquidity coverage (L1)", "No near-term need" if m.liquidity_coverage_ratio >= 999 else f"{m.liquidity_coverage_ratio:.2f}x")
        comparable = [r for r in assessment.rule_results if r.before_value is not None and r.after_value is not None
                      and r.threshold_value is not None and not (r.metric == "liquidity_ratio" and r.after_value >= 999)]
        units = {"risk_score": "risk (0-100)", "growth_pct": "% of portfolio", "liquidity_ratio": "coverage (x)"}
        if comparable:
            cols = st.columns(min(len(comparable), 3))
            for i, rule in enumerate(comparable):
                with cols[i % len(cols)]:
                    show_chart(metric_vs_limit(rule, units.get(rule.metric or "", "%")))
            st.caption("Light = current · dark = proposed · red dashed = limit")
    with tabs[2]:
        left, mid, right = st.columns(3)
        with left:
            show_chart(before_after_bars(exposure_frame(profile, ideas, "asset_class"), "Asset class", height=210))
        with mid:
            show_chart(before_after_bars(exposure_frame(profile, ideas, "region"), "Region", height=210))
        with right:
            chart = alt.Chart(stress_frame(profile, ideas), title="Illustrative stress impact").mark_bar(cornerRadiusEnd=4, height=11).encode(
                y=alt.Y("Scenario:N", title=None, axis=alt.Axis(labelLimit=160)), yOffset=alt.YOffset("Portfolio:N", sort=["Before", "After"]),
                x=alt.X("Impact %:Q", title="Estimated portfolio move %"),
                color=alt.Color("Portfolio:N", scale=alt.Scale(domain=["Before", "After"], range=[BEFORE_COLOR, AFTER_COLOR]), legend=alt.Legend(title=None)),
                tooltip=["Scenario", "Portfolio", alt.Tooltip("Impact %:Q", format=".2f")])
            show_chart(chart.properties(height=210))
        st.caption("Illustrative stress scenarios, not a forecast")
    with tabs[3]:
        rows = [{"Scope": "Portfolio", "Rule": f"{r.rule_id} {r.name}", "Outcome": r.outcome, "Label": STATUS_ICONS[r.outcome], "Message": r.message}
                for r in assessment.rule_results]
        for ie in sorted(assessment.idea_evaluations, key=lambda e: e.slot_number):
            rows += [{"Scope": f"{ie.slot_number} · {ie.ticker}", "Rule": f"{r.rule_id} {r.name}", "Outcome": r.outcome,
                      "Label": STATUS_ICONS[r.outcome], "Message": r.message} for r in ie.rule_results]
        frame = pd.DataFrame(rows)
        left, right = st.columns([1.2, 1])
        with left:
            show_chart(status_heatmap(frame))
        with right:
            counts = frame["Outcome"].value_counts()
            a, b, c = st.columns(3)
            a.metric("✓ PASS", int(counts.get("PASS", 0)))
            b.metric("! FLAG", int(counts.get("FLAG", 0)))
            c.metric("✕ FAIL", int(counts.get("FAIL", 0) + counts.get("BLOCK", 0)))
            st.dataframe(frame[frame["Outcome"] != "PASS"][["Scope", "Rule", "Outcome", "Message"]], hide_index=True, width="stretch", height=200)
    done_footer(ctx, "suitability", f"Suitability result **{assessment.overall_status}**. You decide on the checked set at the next checkpoint.")


def render_approval(ctx: dict[str, Any]) -> None:
    state = ctx["state"]
    assessment = state.get("assessment")
    approved = state.get("approved_idea_set")
    status_html = pill("PASS", "Approved by advisor") if approved else (pill("FLAG", "Waiting for your decision") if ctx["stage"] == "suitability_review" else "")
    page_header(ctx, "approval", "Human-approved idea set",
                f"Lock in each idea; only a fully locked-in set is formatted into the {final_label(ctx).lower()}. A FAIL can never be locked.", status_html)
    if not assessment:
        st.info("This checkpoint opens once the suitability check has run.")
        return
    lineage_strip(state, "approval", final_label(ctx))
    ideas = state["candidate_ideas"].ideas
    statuses = {ie.idea_id: ie.status for ie in assessment.idea_evaluations}
    tabs = st.tabs(["Checked set", "Human-approved set", "Idea lineage"])
    locked_ids = set(ctx["pending"].locked_idea_ids) if ctx["stage"] == "suitability_review" else {i["idea_id"] for i in (approved or {}).get("ideas", [])}
    with tabs[0]:
        left, right = st.columns([1.3, 1])
        with left:
            html_block("<table class='grid'><tr><th>Idea</th><th>Verdict</th><th>Locked in</th><th>Action</th><th>Why it fits</th></tr>" + "".join(
                f"<tr><td><b>{esc(i.ticker)}</b> · {esc(i.portfolio_role.title())}</td><td>{pill(statuses.get(i.idea_id, 'PASS'))}</td>"
                f"<td>{pill('PASS', 'Locked') if i.idea_id in locked_ids else info_pill('Not yet')}</td>"
                f"<td>{i.action} {i.target_weight_pct:.1f}%</td><td>{esc(i.client_fit_rationale)}</td></tr>"
                for i in sorted(ideas, key=lambda x: x.slot_number)) + "</table>")
        with right:
            html_block(f"<div class='callout'><b>Critic</b><br>{esc(assessment.critic_notes)}</div>")
            m = assessment.after_metrics
            st.metric("Proposed portfolio risk (0-100)", risk_points(m.after_risk_score), f"client band {risk_band_points(state['profile'].risk_score_range)}", delta_color="off", delta_arrow="off")
    with tabs[1]:
        if not approved:
            st.info("No set has been approved yet. Lock in all three ideas to approve the set that is formatted next.")
        else:
            for col, idea in zip(st.columns(3), sorted(approved["ideas"], key=lambda i: i["slot_number"])):
                with col:
                    idea_card(idea, None, "Advisor approved")
            if approved.get("notes"):
                html_block(f"<div class='callout'><b>Advisor rationale</b><br>{esc(approved['notes'])}</div>")
    with tabs[2]:
        rounds = state.get("idea_rounds") or []
        final_ids = {i["idea_id"] for i in (approved or {}).get("ideas", [])}
        latest = rounds[-1]["statuses"] if rounds else {}
        seen: dict[str, dict[str, Any]] = {}
        for rnd in rounds:
            for idea in rnd["ideas"]:
                entry = seen.setdefault(idea["idea_id"], {"Slot": idea["slot_number"], "Idea": f"{idea['ticker']} · {idea['name']}",
                                                          "First round": rnd["round"], "Verdicts": []})
                entry["Verdicts"].append(rnd["statuses"].get(idea["idea_id"], ""))
        rows = []
        for idea_id, entry in seen.items():
            if idea_id in final_ids:
                fate = "Locked in"
            elif idea_id in latest:
                fate = "Awaiting advisor" if latest[idea_id] != "FAIL" else "Failing, needs action"
            else:
                fate = "Replaced (critic FAIL)" if "FAIL" in entry["Verdicts"] else "Replaced (advisor revision)"
            rows.append({**entry, "Verdicts": " → ".join(entry["Verdicts"]), "Outcome": fate})
        st.dataframe(pd.DataFrame(rows).sort_values(["Slot", "First round"]), hide_index=True, width="stretch")

    if ctx["stage"] != "suitability_review":
        done_footer(ctx, "approval", decision_summary(ctx, "suitability_review", "Checkpoint 3 · idea set approved"))
        return
    pending = ctx["pending"]
    allowed = pending.allowed_actions
    locked = set(pending.locked_idea_ids)
    flagged = assessment.overall_status == "FLAG"
    idea_label = lambda i: f"Idea {i.slot_number} · {i.ticker} ({statuses.get(i.idea_id, '')})"
    with st.container(border=True, key="footer"):
        cols = st.columns([2.6, 1, 1, 1, 1.1], vertical_alignment="bottom")
        with cols[0]:
            note = st.text_input("Checkpoint 3 · Lock in each idea — rationale" + (" (required: FLAG)" if flagged else ""),
                                 key=f"cp3_note_{ctx['sid']}", placeholder="Why this idea suits the client, or what to change.")
            st.caption(f"Locked in: {len(locked)} of {len(ideas)}. The {final_label(ctx).lower()} is drafted once all three are locked; a FAIL cannot be locked.")
        # No common accept button: the advisor locks in each idea individually.
        for col, idea in zip(cols[1:4], sorted(ideas, key=lambda x: x.slot_number)):
            with col:
                verdict = statuses.get(idea.idea_id, "PASS")
                if idea.idea_id in locked:
                    if st.button(f"Unlock {idea.ticker}", icon=":material/lock:", width="stretch", key=f"unlock_{idea.idea_id}",
                                 disabled="unlock_idea" not in allowed):
                        submit_action("unlock_idea", note, idea_id=idea.idea_id)
                else:
                    needs_note = flagged or verdict == "FLAG"
                    if st.button(f"Lock in {idea.ticker}", icon=":material/lock_open:", type="primary", width="stretch", key=f"lock_{idea.idea_id}",
                                 disabled=verdict == "FAIL" or "lock_idea" not in allowed or (needs_note and not note.strip()),
                                 help="A FAIL idea cannot be locked" if verdict == "FAIL" else ("Needs a rationale" if needs_note else None)):
                        submit_action("lock_idea", note, idea_id=idea.idea_id)
        with cols[4], st.popover("More actions", icon=":material/more_horiz:", width="stretch"):
            target = st.selectbox("Idea", ideas, format_func=idea_label, key=f"cp3_idea_{ctx['sid']}")
            if st.button("Revise this idea", icon=":material/edit:", width="stretch", disabled=not note.strip() or "revise" not in allowed):
                submit_action("revise", note, idea_id=target.idea_id)
            weight = st.number_input("New target weight %", 0.0, 100.0, float(target.target_weight_pct), 0.5, key=f"cp3_w_{ctx['sid']}_{target.idea_id}")
            if st.button("Apply weight and re-check", icon=":material/tune:", width="stretch",
                         disabled=not note.strip() or "edit_allocation" not in allowed or weight == target.target_weight_pct):
                submit_action("edit_allocation", note, idea_id=target.idea_id, edits={"target_weight_pct": weight})
            st.divider()
            if st.button("Regenerate all three ideas", icon=":material/refresh:", width="stretch", disabled=not note.strip() or "regenerate" not in allowed):
                submit_action("regenerate", note)
            if st.button("Reject and end the session", icon=":material/block:", width="stretch", key="cp3_reject"):
                submit_action("reject", note)


# --------------------------------------------------------------------------------------
# Page: memo (UC2, UC3 revise; CP4)
# --------------------------------------------------------------------------------------

def render_pdf_download(ctx: dict[str, Any], label: str) -> None:
    pdf_path = st.session_state.service.get_pdf_path(ctx["sid"])
    pdf_file = Path(pdf_path) if pdf_path else None
    if pdf_file and not pdf_file.is_absolute():
        pdf_file = ROOT / pdf_file
    if pdf_file and pdf_file.is_file():
        st.download_button(label, pdf_file.read_bytes(), file_name=pdf_file.name, mime="application/pdf",
                           type="primary", icon=":material/download:", width="stretch", key=f"pdf_{ctx['sid']}")
    else:
        st.warning("The PDF file is not at the expected path.")


def render_memo(ctx: dict[str, Any]) -> None:
    state = ctx["state"]
    memo = state.get("memo")
    approved = ctx["outcome"] == "APPROVED"
    page_header(ctx, "memo", "Investment memo",
                "Formats the approved set into a 3-page memo with charts; it cannot change ideas, allocations or risk findings.",
                (pill("PASS", "Advisor approved · PDF generated") if approved else pill("FLAG", "Draft · PDF only after sign-off")) if memo else "")
    if not memo or nav_status("memo", ctx) == "locked":
        st.info("The memo is drafted after the idea set is approved at Checkpoint 3.")
        return
    lineage_strip(state, "final", final_label(ctx))
    page1, page3 = memo.page_1_executive, memo.page_3_portfolio_view
    tabs = st.tabs(["Page 1 · Executive recommendation", "Page 2 · Idea details", "Page 3 · Portfolio view", "Compliance checks"])
    with tabs[0]:
        # Same as Figure 1 of the PDF: the three ideas at a glance, then the allocation shift.
        ideas, profile = state["candidate_ideas"].ideas, state["profile"]
        verdicts = {ie.idea_id: ie.status for ie in state["assessment"].idea_evaluations} if state.get("assessment") else {}
        current = holding_weights(profile.parsed_holdings)
        rows = "".join(
            f"<tr><td><b>{esc(i.ticker)}</b> · {esc(i.name)}</td><td>{esc(i.lane.title())}</td><td>{esc(i.action)}</td>"
            f"<td>{current.get(i.ticker, 0.0):.1f}%</td><td><b>{i.target_weight_pct:.1f}%</b></td>"
            f"<td>{(fmt_money(i.indicative_amount_range[0]) + '–' + fmt_money(i.indicative_amount_range[1])) if i.indicative_amount_range else '—'}</td>"
            f"<td>{pill(verdicts.get(i.idea_id, 'PASS'))}</td></tr>" for i in sorted(ideas, key=lambda x: x.slot_number))
        html_block("<table class='grid'><tr><th>Idea</th><th>Lane</th><th>Action</th><th>Current</th><th>Target</th><th>Indicative amount</th><th>Fit</th></tr>"
                   + rows + "</table>")
        left, right = st.columns([1.3, 1])
        with left:
            show_chart(before_after_bars(exposure_frame(profile, ideas, "asset_class"), "Allocation: current vs proposed", height=180))
        with right:
            after = exposure_frame(profile, ideas, "asset_class")
            show_chart(donut(dict(zip(after["Category"], after["After"])), "Proposed allocation", height=180))
    with tabs[1]:
        for col, d in zip(st.columns(3), memo.page_2_details):
            with col, card(height="stretch"):
                html_block(f"<div class='eyebrow'>Idea {d.slot_number} · {esc(d.portfolio_role.title())} · {esc(d.conviction)} conviction</div>"
                           f"<div class='card-ticker'>{esc(d.ticker)}</div><div class='subtle'>{esc(d.action)} · {esc(d.allocation_range_str)}</div>"
                           + kv("Why it suits the client", d.client_fit_rationale)
                           + "<div class='kv'><b>Risks to raise</b><br>" + "".join(info_pill(r) for r in d.key_risks) + "</div>")
    with tabs[2]:
        ideas, profile = state["candidate_ideas"].ideas, state["profile"]
        left, right = st.columns([1.2, 1])
        with left:
            show_chart(before_after_bars(exposure_frame(profile, ideas, "asset_class"), "Asset allocation: current vs proposed", height=190))
            html_block(f"<div class='callout'><b>Portfolio-level impact</b><br>{esc(page3.portfolio_level_impact_narrative)}</div>")
        with right, card(height="stretch"):
            st.markdown("**Scenario analysis**")
            st.markdown("\n".join(f"- **{md(k)}**: {md(v)}" for k, v in page3.scenario_analysis.items()))
            st.markdown("**Risk disclosures**")
            st.markdown("\n".join(f"- {md(r)}" for r in page3.risk_disclosures))
            st.caption(md(page3.advisor_signature_block))
    with tabs[3]:
        html_block("<table class='grid'><tr><th>Check</th><th>Outcome</th><th>Detail</th></tr>" + "".join(
            f"<tr><td>{esc(r.rule_id)} · {esc(r.name)}</td><td>{pill(r.outcome)}</td><td>{esc(r.message)}</td></tr>" for r in memo.compliance_checks) + "</table>")
        st.markdown("**Sources**")
        html_block(source_links())

    if approved:
        with st.container(border=True, key="footer_done"):
            left, right = st.columns([3, 1.3], vertical_alignment="center")
            left.markdown(decision_summary(ctx, "advisor_approval", "Checkpoint 4 · memo signed off") + " · the PDF includes the portfolio charts.")
            with right:
                render_pdf_download(ctx, "Download memo PDF")
        return
    if ctx["stage"] != "advisor_approval":
        done_footer(ctx, "memo", "The memo is waiting for the idea set to be approved again.")
        return
    allowed = ctx["pending"].allowed_actions
    ideas = state["candidate_ideas"].ideas
    with st.container(border=True, key="footer"):
        left, mid, right = st.columns([3, 1.4, 1.1], vertical_alignment="bottom")
        with left:
            note = st.text_input("Checkpoint 4 · Sign off the memo — note (printed in the approval block)", key=f"cp4_note_{ctx['sid']}")
            if "approve" not in allowed:
                st.error("The memo failed a compliance check (R9 wording or consistency). Regenerate it.", icon=":material/block:")
        with mid:
            if st.button("Approve · generate PDF", type="primary", icon=":material/verified:", width="stretch", disabled="approve" not in allowed):
                submit_action("approve", note)
        with right, st.popover("More actions", icon=":material/more_horiz:", width="stretch"):
            st.caption("Changes need a note.")
            if st.button("Regenerate the memo", icon=":material/refresh:", width="stretch", disabled=not note.strip() or "regenerate" not in allowed):
                submit_action("regenerate", note)
            target = st.selectbox("Idea", ideas, format_func=lambda i: f"Idea {i.slot_number} · {i.ticker}", key=f"cp4_rev_{ctx['sid']}")
            if st.button("Revise this idea", icon=":material/edit:", width="stretch", disabled=not note.strip() or "revise" not in allowed):
                submit_action("revise", note, idea_id=target.idea_id)
            st.divider()
            if st.button("Reject and end the session", icon=":material/block:", width="stretch", key="cp4_reject"):
                submit_action("reject", note)


# --------------------------------------------------------------------------------------
# Page: meeting brief (UC1; CP4)
# --------------------------------------------------------------------------------------

def render_brief(ctx: dict[str, Any]) -> None:
    state = ctx["state"]
    brief = state.get("meeting_brief")
    approved = ctx["outcome"] == "APPROVED"
    page_header(ctx, "brief", "Client meeting brief",
                "Confirm the profile, share the market landscape and agree next steps. The PDF (with charts) is generated after your approval.",
                (pill("PASS", "Approved · PDF generated") if approved else pill("FLAG", "Draft")) if brief else "")
    if not brief:
        st.info("The brief is drafted after the research checkpoint (and the idea approval, if ideas were requested).")
        return
    if brief.recommendations:
        lineage_strip(state, "final", final_label(ctx))
    profile = state["profile"]
    tabs = st.tabs(["What matters", "Profile & changes", "Portfolio & risks", "Market landscape", "Talking points & questions"]
                   + (["Ideas to discuss"] if brief.recommendations else []))
    with tabs[0]:
        # Deck slide 9: relevance filtering, not summarisation. Each card says why it is in the briefing.
        if brief.key_insights:
            cols = st.columns(len(brief.key_insights))
            for n, (col, ins) in enumerate(zip(cols, brief.key_insights), start=1):
                with col, card(height="stretch"):
                    html_block(f"<div class='eyebrow'>{n:02d} · {esc(ins.area)}</div><div class='card-ticker'>{esc(ins.headline)}</div>")
                    if ins.area == "PORTFOLIO":
                        show_chart(reference_mix_chart(profile))
                        st.caption("Dark = current · light = CIO mix")
                    elif ins.area == "MARKET":
                        html_block(source_links([s for s in ins.sources if s in (st.session_state.get("source_index") or {})]))
                    else:
                        html_block(f"<div class='kv'>{esc(ins.detail)}</div>")
                    with st.expander("Why is this here?"):
                        st.markdown(md(ins.why_here))
        else:
            st.info("Nothing stands out for this client this period: no client notes since the last review, no gap to the reference mix.")
    with tabs[1]:
        left, right = st.columns([1, 1.2])
        with left, card(height="stretch"):
            html_block(f"<div class='eyebrow'>{esc(brief.meeting_purpose)} · {esc(brief.review_period)}</div>")
            st.markdown("\n".join(f"- {md(s)}" for s in brief.profile_summary))
        with right:
            if brief.profile_changes:
                html_block("<table class='grid'><tr><th>Changed since last review</th><th>Before</th><th>Now</th></tr>" + "".join(
                    f"<tr><td>{esc(c.label)}</td><td>{esc(c.before)}</td><td><b>{esc(c.after)}</b></td></tr>" for c in brief.profile_changes) + "</table>")
            else:
                st.success("No change in the client's profile since the last review.")
    with tabs[2]:
        left, mid, right = st.columns([1, 1.2, 1.1])
        with left:
            show_chart(donut(profile.current_asset_class_weights, "Current allocation"))
        with mid:
            frame = pd.DataFrame([{"Scenario": k, "Impact %": v} for k, v in stress_impact(profile.parsed_holdings).items()])
            show_chart(signed_bars(frame, "Scenario", "Impact %", "Illustrative stress: estimated portfolio move %", height=170))
        with right, card(height="stretch"):
            st.markdown("**Portfolio risks to raise**")
            st.markdown("\n".join(f"- {md(r)}" for r in brief.portfolio_risks))
    with tabs[3]:
        research = state.get("research_brief")
        if research and research.conflicting_signals:
            html_block(f"<div class='callout'>{pill('FLAG', 'Sources disagree')}<br>" + "<br>".join(esc(c) for c in research.conflicting_signals) + "</div>")
        for col, finding in zip(st.columns(3), sorted(research.findings, key=lambda f: f.materiality_rank)[:3] if research else []):
            with col, card(height="stretch"):
                html_block(f"<b>{finding.materiality_rank} · {esc(finding.theme)}</b>" + source_links(finding.cited_source_ids))
                with st.expander("Details"):
                    st.markdown(f"{md(finding.claim)}\n\n**For this client:** {md(finding.relevance_to_client)}")
    selecting = not approved and ctx["stage"] == "brief_approval"
    with tabs[4]:
        left, right = st.columns(2)
        with left, card(height="stretch"):
            if selecting:
                # Checkpoint 4 (deck slide 3): the RM selects the points to raise; only those go into the PDF.
                st.markdown("**Talking points** · tick the points to raise with the client")
                for i, t in enumerate(brief.talking_points):
                    st.checkbox(md(t), value=True, key=f"cp4b_pt_{ctx['sid']}_{i}")
            else:
                st.markdown("**Talking points**\n" + "\n".join(f"{i + 1}. {md(t)}" for i, t in enumerate(brief.talking_points)))
                if brief.points_not_raised:
                    st.caption(f"{len(brief.points_not_raised)} drafted point(s) not selected by the RM (kept on the audit record).")
        with right, card(height="stretch"):
            st.markdown("**Questions for the client**\n" + "\n".join(f"- {md(q)}" for q in brief.questions_for_client))
    if brief.recommendations:
        with tabs[5]:
            html_block("<table class='grid'><tr><th>#</th><th>Idea</th><th>Action</th><th>Target</th><th>Role</th><th>Why it fits</th></tr>" + "".join(
                f"<tr><td>{g.slot_number}</td><td><b>{esc(g.ticker)}</b> · {esc(g.name)}</td><td>{esc(g.action)}</td><td>{g.target_weight_pct:.1f}%</td>"
                f"<td>{esc(g.portfolio_role.title())}</td><td>{esc(g.why_it_fits)}</td></tr>" for g in brief.recommendations) + "</table>")
            show_chart(before_after_bars(exposure_frame(profile, state["candidate_ideas"].ideas, "asset_class"), "Asset allocation if implemented", height=170))

    if approved:
        with st.container(border=True, key="footer_done"):
            left, right = st.columns([3, 1.3], vertical_alignment="center")
            left.markdown(decision_summary(ctx, "brief_approval", "Checkpoint 4 · brief approved") + " · the PDF includes the allocation and risk charts.")
            with right:
                render_pdf_download(ctx, "Download meeting brief PDF")
        return
    if ctx["stage"] != "brief_approval":
        return
    allowed = ctx["pending"].allowed_actions
    with st.container(border=True, key="footer"):
        left, mid, right = st.columns([3, 1.4, 1], vertical_alignment="bottom")
        selected = [t for i, t in enumerate(brief.talking_points) if st.session_state.get(f"cp4b_pt_{ctx['sid']}_{i}", True)]
        note = left.text_input(f"Checkpoint 4 · Approve the brief with {len(selected)} of {len(brief.talking_points)} talking points "
                               "— note (printed on the brief)", key=f"cp4b_note_{ctx['sid']}")
        with mid:
            if st.button("Approve · generate PDF", type="primary", icon=":material/verified:", width="stretch",
                         disabled="approve" not in allowed or not selected):
                submit_action("approve", note, edits={"talking_points": selected})
        with right, st.popover("More actions", icon=":material/more_horiz:", width="stretch"):
            if st.button("Regenerate the brief (needs a note)", icon=":material/refresh:", width="stretch", disabled=not note.strip()):
                submit_action("regenerate", note)
            if st.button("Reject and end the session", icon=":material/block:", width="stretch", key="cp4b_reject"):
                submit_action("reject", note)


# --------------------------------------------------------------------------------------
# Page: decision record (UC3 maintain / monitor)
# --------------------------------------------------------------------------------------

def render_record(ctx: dict[str, Any]) -> None:
    record = ctx["state"].get("decision_record")
    page_header(ctx, "record", "Market-event decision record",
                "The documented decision. No trades are placed; the record PDF includes the impact chart.",
                pill("PASS", "Recorded") if record else "")
    if not record:
        st.info("The record is created when you decide to maintain or monitor.")
        return
    impact = ctx["state"]["event_impact"]
    left, right = st.columns([1.2, 1])
    with left, card(height="stretch"):
        html_block(f"<div class='eyebrow'>{esc(record.event_title)}</div><div class='card-ticker'>{esc(record.decision.title())}</div>"
                   + kv("Rationale", record.rationale) + (kv("Follow-up / monitoring trigger", record.follow_up) if record.follow_up else "")
                   + kv("Estimated portfolio move", f"{record.portfolio_move_pct:+.2f}%") + kv("Recorded", record.advisor_signature_block))
    with right:
        frame = pd.DataFrame([{"Holding": h.ticker, "Contribution": h.contribution_pct} for h in impact.holdings])
        show_chart(signed_bars(frame, "Holding", "Contribution", "Contribution to the move (% points)", height=200))
    with st.container(border=True, key="footer_done"):
        a, b = st.columns([3, 1.3], vertical_alignment="center")
        a.markdown("✓ **Decision recorded.** It is saved to the client's history; a later review starts from it.")
        with b:
            render_pdf_download(ctx, "Download decision record PDF")


# --------------------------------------------------------------------------------------
# Page: audit & performance
# --------------------------------------------------------------------------------------

def render_audit(ctx: dict[str, Any]) -> None:
    state, view = ctx["state"], ctx["view"]
    log = st.session_state.service.get_run_log(view.session_id)
    page_header(ctx, "audit", "Audit trail & performance",
                f"Run {view.session_id}: inputs, every agent output with model and prompt version, rule results and advisor decisions.")
    tabs = st.tabs(["Decision timeline", "Latency & cost", "Advisor notes", "Raw run log"])
    with tabs[0]:
        rows = []
        for e in log:
            detail = ""
            if e["event"] == "agent_output":
                detail = f"{e.get('agent', '')} · {e.get('model', '')} · prompt {e.get('prompt_version', '')}"
            elif e["event"] == "advisor_decision":
                d = e.get("decision") or {}
                detail = f"{d.get('action', '')} → {e.get('review_outcome', '')}" + (f" · “{d['notes']}”" if d.get("notes") else "")
            elif e["event"] == "auto_regeneration":
                detail = f"slots {e.get('slots')} · " + " ".join(e.get("reasons", []))
            elif e["event"] == "rule_results":
                outcomes = [r["outcome"] for r in e.get("portfolio", [])]
                detail = f"{outcomes.count('PASS')} pass · {outcomes.count('FLAG')} flag · {outcomes.count('FAIL')} fail (portfolio)"
            elif e["event"] == "event_impact":
                imp = e.get("impact") or {}
                detail = f"move {imp.get('portfolio_move_pct')}% · suggested {imp.get('suggested_decision')}"
            elif e["event"] in ("run_started", "run_ended", "pdf_generated"):
                detail = e.get("outcome") or e.get("pdf_path") or f"policy {e.get('policy_version', '')} · {e.get('llm_client', '')}"
            rows.append({"Time (UTC)": e["timestamp"][11:19], "Event": e["event"].replace("_", " "), "Stage": e.get("stage", ""), "Detail": detail})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=430)
    with tabs[1]:
        calls = [e for e in state.get("audit_trail") or [] if e.get("duration_ms") is not None]
        frame = pd.DataFrame([{"Call": f"{i + 1}. {e['agent'].replace('Agent', '')}", "Agent": e["agent"], "Latency (ms)": e["duration_ms"],
                               "Model": e.get("model", ""), "Prompt": e.get("prompt_version", ""),
                               "Prompt tokens": (e.get("usage") or {}).get("prompt_tokens", 0),
                               "Completion tokens": (e.get("usage") or {}).get("completion_tokens", 0)} for i, e in enumerate(calls)])
        if frame.empty:
            st.info("No agent calls recorded yet.")
        else:
            prompt_t, completion_t = frame["Prompt tokens"].sum(), frame["Completion tokens"].sum()
            cost = prompt_t / 1e6 * ASSUMED_PRICE_PER_MTOK["prompt"] + completion_t / 1e6 * ASSUMED_PRICE_PER_MTOK["completion"]
            a, b, c, d = st.columns(4)
            a.metric("Agent calls", len(frame))
            b.metric("Total agent latency", f"{frame['Latency (ms)'].sum() / 1000:.2f} s", help="Pilot target: p95 ≤ 2 min end-to-end")
            c.metric("Tokens", f"{int(prompt_t + completion_t):,}", "offline stubs use none" if prompt_t + completion_t == 0 else None, delta_color="off", delta_arrow="off")
            d.metric("Estimated model cost", f"US${cost:.3f}", "target ≤ US$0.50 per memo", delta_color="off", delta_arrow="off",
                     help=f"Assumed rates: US${ASSUMED_PRICE_PER_MTOK['prompt']}/M prompt, US${ASSUMED_PRICE_PER_MTOK['completion']}/M completion tokens")
            agents = list(dict.fromkeys(frame["Agent"]))
            chart = alt.Chart(frame, title="Latency per agent call").mark_bar(cornerRadiusEnd=4, height=16).encode(
                y=alt.Y("Call:N", sort=list(frame["Call"]), title=None), x=alt.X("Latency (ms):Q"),
                color=alt.Color("Agent:N", scale=alt.Scale(domain=agents, range=SERIES[:len(agents)]), legend=None),
                tooltip=["Agent", "Model", alt.Tooltip("Latency (ms):Q", format=",.1f")])
            show_chart(chart.properties(height=max(26 * len(frame) + 40, 160)))
    with tabs[2]:
        left, right = st.columns([1.2, 1])
        with left, card(height="stretch"):
            st.markdown("**Notes and exceptions on this run**")
            st.markdown("\n".join(f"- {md(n)}" for n in state.get("advisor_notes") or []) or "No notes yet.")
        with right:
            note = st.text_area("Add an advisor note", key="free_note", height=120, help="Logged to the run log. A note never acts as a review decision.")
            if st.button("Log note", icon=":material/note_add:", disabled=not note.strip(), width="stretch"):
                submit_action("add_advisor_note", note)
    with tabs[3]:
        with st.container(height=440):
            st.json(log, expanded=1)


# --------------------------------------------------------------------------------------
# Landing and client home
# --------------------------------------------------------------------------------------

def render_landing() -> None:
    top_l, top_r = st.columns([3, 2], vertical_alignment="bottom")
    with top_l:
        html_block("<div class='eyebrow'>UBS OneView · Agentic Fintech Consultancy pilot</div>")
        st.markdown("## Advisory workspace")
    with top_r:
        mode = st.segmented_control("Run mode", [OFFLINE, LIVE], default=st.session_state.get("mode", OFFLINE), key="mode_pick",
                                    selection_mode="single", width="stretch")
        st.session_state.mode = mode or OFFLINE
    if st.session_state.mode == LIVE and not os.environ.get("OPENROUTER_API_KEY"):
        st.warning("Live mode needs OPENROUTER_API_KEY in the environment or a .env file. Until then, use offline mode.")
    if st.session_state.get("error"):
        st.error(st.session_state.error)

    tomorrow = (date.today() + timedelta(days=1)).strftime("%a %d %b")
    with st.container(border=False):
        left, right = st.columns([4, 1.2], vertical_alignment="center")
        left.markdown(f"<div class='cal'><b>Calendar · {tomorrow} · 10:00 · Quarterly review — Sarah Tan</b> "
                      "<span class='subtle'>· future-state preview</span></div>", unsafe_allow_html=True)
        right.button("Prepare meeting brief", icon=":material/event:", type="primary", width="stretch", key="cal_prepare",
                     on_click=start_case, args=(PRE_MEETING, "demo_sarah"))

    tabs = st.tabs(["Clients", "New client intake"])
    with tabs[0]:
        for row_start in range(0, len(VISIBLE_CLIENT_KEYS), 2):
            for col, key in zip(st.columns(2), VISIBLE_CLIENT_KEYS[row_start:row_start + 2]):
                raw = client_record(key)["raw"]
                summary, control = CLIENT_CARD_SUMMARIES[key]
                last = (raw.get("last_review") or {}).get("date", "none")
                with col, card():
                    left, right = st.columns([3, 1.1], vertical_alignment="center")
                    with left:
                        html_block(f"<div style='display:flex;gap:.75rem;align-items:center'>{avatar(raw['name'], large=True)}<div>"
                                   f"<b style='font-size:1rem'>{esc(raw['name'])}</b>"
                                   f"<div class='subtle' style='font-size:.8rem'>{esc(raw['client_id'])} · {esc(summary)} · "
                                   f"{esc(raw.get('stated_risk_willingness', ''))} · last review {esc(last)}</div></div></div>")
                    right.button("Open", key=f"open_{key}", type="primary", width="stretch", on_click=select_client, args=(key,))
    with tabs[1]:
        render_intake()


def render_intake() -> None:
    with st.form("client_intake", border=False):
        c1, c2, c3, c4 = st.columns([1, 1.6, 1, 1])
        client_id = c1.text_input("Client ID *", value=f"CL-{uuid4().hex[:6].upper()}")
        name = c2.text_input("Client name *")
        currency = c3.selectbox("Base currency", ["SGD", "USD", "EUR", "GBP", "HKD"])
        wealth = c4.number_input("Investable wealth *", min_value=0.0, value=500000.0, step=10000.0)
        c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
        goals = c1.text_input("Investment objective(s) * — separate with ';'", placeholder="Retirement income in 10 years; preserve capital")
        horizon = c2.number_input("Time horizon (months) *", min_value=1, value=60)
        risk = c3.selectbox("Risk willingness *", RISK_OPTIONS, index=2)
        experience = c4.selectbox("Experience", EXPERIENCE_OPTIONS, index=2)
        left, right = st.columns([1.6, 1])
        with left:
            holdings = st.data_editor(pd.DataFrame([
                {"ticker": "ACWI", "name": "Global equities", "asset_class": "Equities", "region": "Global", "currency": "USD", "weight_pct": 50.0, "risk_rating": 3},
                {"ticker": "AGGG", "name": "Investment-grade bonds", "asset_class": "Fixed Income", "region": "Global", "currency": currency, "weight_pct": 35.0, "risk_rating": 2},
                {"ticker": "BIL", "name": "Cash / money market", "asset_class": "Cash Equivalents", "region": "Singapore", "currency": currency, "weight_pct": 15.0, "risk_rating": 1},
            ]), num_rows="dynamic", hide_index=True, width="stretch", height=170,
                column_config={"weight_pct": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0),
                               "risk_rating": st.column_config.NumberColumn("Risk 1–5", min_value=1, max_value=5)})
        with right:
            liquidity = st.data_editor(pd.DataFrame([{"due_months": 24, "amount": 50000.0, "purpose": "Planned expense"}]),
                                       num_rows="dynamic", hide_index=True, width="stretch", height=110,
                                       column_config={"due_months": st.column_config.NumberColumn("Due (months)", min_value=1),
                                                      "amount": st.column_config.NumberColumn("Amount", min_value=0.0),
                                                      "purpose": st.column_config.TextColumn("Purpose")})
            notes = st.text_input("Advisor context (optional)")
        submitted = st.form_submit_button("Create client and continue", type="primary")
    if not submitted:
        return
    rows = _norm_holdings(holdings.replace({pd.NA: None}).to_dict(orient="records"))
    total = sum(r["weight_pct"] for r in rows)
    goal_list = [g.strip() for g in goals.split(";") if g.strip()]
    if not name.strip() or not goal_list:
        st.error("Enter the client name and at least one investment objective.")
        return
    if abs(total - 100) > 0.5:
        st.error(f"Holding weights must total 100% (currently {total:.1f}%).")
        return
    raw = {"client_id": client_id.strip(), "name": name.strip(), "base_currency": currency, "goals": goal_list,
           "time_horizon_months": int(horizon), "stated_risk_willingness": risk,
           "liquidity_schedule": _norm_liquidity(liquidity.replace({pd.NA: None}).to_dict(orient="records")),
           "total_wealth": float(wealth), "experience_level": experience, "raw_holdings": rows,
           "constraints": {"excluded_sectors": [], "esg_mandate": False, "max_single_stock_pct": 20.0}, "advisor_notes": notes.strip()}
    select_client(f"custom_{client_id.strip()}", raw)
    st.rerun()


def render_client_home() -> None:
    record = client_record(st.session_state.client_key)
    raw = record["raw"]
    top_l, top_r = st.columns([4, 1], vertical_alignment="bottom")
    with top_l:
        html_block(f"<div class='eyebrow'>{esc(raw.get('client_id', ''))} · {esc(st.session_state.get('mode', OFFLINE))}</div>")
        st.markdown(f"## {md(raw.get('name', 'Client'))}")
        html_block(f"<div class='subtle'>{esc(' · '.join(raw.get('goals', [])))}</div>")
    with top_r:
        st.button("All clients", icon=":material/arrow_back:", on_click=reset_to_client_list, width="stretch")
    if st.session_state.get("error"):
        st.error(st.session_state.error)
    a, b, c, d = st.columns(4)
    a.metric("Investable wealth", fmt_money(raw.get("total_wealth"), raw.get("base_currency", "SGD")))
    b.metric("Time horizon", fmt_horizon(raw.get("time_horizon_months")))
    c.metric("Stated risk willingness", raw.get("stated_risk_willingness", "—"))
    d.metric("Last review", (raw.get("last_review") or {}).get("date", "None on record"))

    st.markdown("#### What do you need today?")
    live_blocked = st.session_state.get("mode") == LIVE and not os.environ.get("OPENROUTER_API_KEY")
    last_run = {h["use_case"]: h for h in record["history"]}
    for col, key in zip(st.columns(3), HOME_CASES):
        verb, label, trigger, icon = USE_CASES[key]
        with col, card(height="stretch"):
            html_block(f"<div class='eyebrow'>{esc(verb)}</div><b style='font-size:1.05rem'>{esc(label)}</b>"
                       f"<div class='usecase-trigger'>{esc(trigger)}</div>")
            if key == PRE_MEETING:
                c1, c2 = st.columns([1, 1.3])
                c1.date_input("Meeting date", value=date.today() + timedelta(days=1), key="uc1_date")
                c2.text_input("Meeting", value="Quarterly portfolio review", key="uc1_purpose")
            elif key == ADHOC:
                st.number_input("New capital to deploy", min_value=0.0, step=10000.0, value=120000.0, key="uc2_amount")
            elif key == POST_MEETING:
                st.text_input("Meeting", value="Quarterly portfolio review", key="uc3_purpose")
                st.text_area("Meeting notes or transcript", key="uc3_transcript", height=68,
                             value=DEMO_TRANSCRIPTS.get(st.session_state.client_key, ""),
                             placeholder="Paste the transcript: one line per turn, e.g. 'Rachel: …' / 'Sarah: …'")
            else:
                st.selectbox("Market event", list(MARKET_EVENTS), format_func=lambda k: MARKET_EVENTS[k]["title"], key="uc3_event",
                             on_change=lambda: st.session_state.update(uc3_desc=MARKET_EVENTS[st.session_state.uc3_event]["description"]))
                st.session_state.setdefault("uc3_desc", MARKET_EVENTS[st.session_state.get("uc3_event") or "rates_up"]["description"])
                st.text_area("What happened", key="uc3_desc", height=68)
            if key in last_run:
                st.caption(f"Last run {last_run[key]['date']}: {last_run[key]['summary'][:90]}")
            st.button("Start", key=f"start_{key}", icon=icon, type="primary", width="stretch", on_click=start_case, args=(key,),
                      disabled=live_blocked or (key == ADHOC and not st.session_state.get("uc2_amount"))
                      or (key == POST_MEETING and not (st.session_state.get("uc3_transcript") or DEMO_TRANSCRIPTS.get(st.session_state.client_key, "")).strip()))
    if record["history"]:
        with st.expander(f"Client history this session ({len(record['history'])})"):
            st.dataframe(pd.DataFrame([{"Date": h["date"], "Use case": USE_CASES[h["use_case"]][1], "Outcome": h["outcome"], "Summary": h["summary"]}
                                       for h in record["history"]]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

PAGES = {
    "profile": render_profile, "limits": render_limits, "meeting": render_meeting, "research": render_research, "event": render_event, "ideas": render_ideas,
    "suitability": render_suitability, "approval": render_approval, "memo": render_memo, "brief": render_brief,
    "record": render_record, "audit": render_audit,
}


def main() -> None:
    st.session_state.setdefault("mode", OFFLINE)
    if not st.session_state.get("client_key"):
        render_home_sidebar()
        render_landing()
        return
    if not st.session_state.get("case") or not st.session_state.get("workspace"):
        render_home_sidebar()
        render_client_home()
        return
    ctx = load_context()
    screen = st.session_state.get("screen")
    keys = [k for k, *_ in visible_steps(ctx)]
    if screen not in keys or nav_status(screen, ctx) == "locked":
        screen = st.session_state.screen = screen_for_pending(None, ctx) if ctx["stage"] or ctx["outcome"] else keys[0]
    render_sidebar(ctx)
    if st.session_state.get("error"):
        st.error(st.session_state.error, icon=":material/error:")
    PAGES[screen](ctx)


if __name__ == "__main__":
    main()
