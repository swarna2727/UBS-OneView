"""UBS Prism offline advisory prototype."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd
import streamlit as st

from src.agents.client_profile import ClientProfileAgent
from src.agents.market_intelligence import MarketIntelligenceAgent
from src.api.service import AdvisorySessionService
from src.llm.fake import FakeAdvisoryLLMClient
from src.schemas.workspace import AdvisorActionRequest


ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "data" / "samples" / "client_profiles.json"
SAMPLE_PROFILES = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
CASE_LABELS = {
    "meeting": "Client meeting preparation",
    "investment": "Investment review",
    "market": "Market event response",
}
VISIBLE_CLIENT_KEYS = ("demo_sarah", "demo_daniel", "demo_mdm_lim", "demo_aaron")
CLIENT_CARD_SUMMARIES = {
    "demo_sarah": "Home down payment · 8 years",
    "demo_daniel": "S$180k property deposit · 18 months",
    "demo_mdm_lim": "Retirement reserve · 3 years",
    "demo_aaron": "Long-term growth · horizon not recorded",
}
SCREENS = [
    "Overview",
    "Client profile",
    "Portfolio",
    "Research",
    "Recommendations",
    "Advisor review",
    "Brief",
]


st.set_page_config(page_title="UBS Prism | Advisory workspace", page_icon=None, layout="wide")

st.html("""
<style>
    :root { --prism-red: #c8102e; }
    .block-container { max-width: 1440px; padding-top: 2.1rem; padding-bottom: 4rem; }
    h1, h2, h3 { letter-spacing: -0.035em; }
    h1 { font-size: clamp(2rem, 3vw, 2.65rem); font-weight: 620; }
    h2 { font-size: 1.42rem; font-weight: 610; margin-top: .25rem; }
    h3 { font-size: 1.05rem; font-weight: 620; }
    [data-testid="stMetric"] { background: var(--secondary-background-color); border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent); border-radius: 12px; padding: 1rem 1.1rem; min-height: 112px; }
    [data-testid="stMetricLabel"] { color: color-mix(in srgb, var(--text-color) 70%, transparent); }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
    div[data-testid="stSegmentedControl"] { margin: .6rem 0 1.3rem; }
    div[data-testid="stSegmentedControl"] button { min-height: 2.7rem; }
    .eyebrow { color: var(--prism-red); font-size: .75rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
    .subtle { color: color-mix(in srgb, var(--text-color) 68%, transparent); }
    .card-title { font-size: 1.02rem; font-weight: 650; margin: 0 0 .35rem; }
    .checkpoint { border-left: 2px solid color-mix(in srgb, var(--text-color) 17%, transparent); padding: .1rem 0 .8rem 1rem; margin-left: .38rem; }
    .checkpoint.active { border-left: 2px solid var(--prism-red); }
    .checkpoint-card { border: 1px solid color-mix(in srgb, var(--text-color) 15%, transparent); border-radius: 12px; padding: .9rem 1rem; min-height: 92px; }
    .checkpoint-card.complete { border-color: color-mix(in srgb, #16834a 52%, transparent); background: color-mix(in srgb, #16834a 9%, var(--secondary-background-color)); }
    .checkpoint-card.complete .checkpoint-status { color: color-mix(in srgb, #16834a 62%, var(--text-color)); }
    .checkpoint-card.active { border-color: color-mix(in srgb, var(--prism-red) 48%, transparent); }
    .checkpoint-status { font-weight: 650; margin-top: .35rem; }
    .status { display: inline-block; border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent); border-radius: 99px; padding: .2rem .55rem; font-size: .76rem; }
    .callout { padding: 1rem 1.15rem; border: 1px solid color-mix(in srgb, var(--text-color) 15%, transparent); border-radius: 12px; background: var(--secondary-background-color); }
    .stButton button { border-radius: 8px; min-height: 2.6rem; }
    @media (prefers-color-scheme: dark) { .eyebrow { color: #ff8091; } }
</style>
""")


def offline_service() -> AdvisorySessionService:
    return AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())


def profile_agent() -> ClientProfileAgent:
    return ClientProfileAgent(llm_client=FakeAdvisoryLLMClient())


def research_agent() -> MarketIntelligenceAgent:
    return MarketIntelligenceAgent(llm_client=FakeAdvisoryLLMClient())


def fmt_money(amount: float, currency: str = "SGD") -> str:
    symbol = "S$" if currency == "SGD" else f"{currency} "
    return f"{symbol}{amount:,.0f}"


def fmt_horizon(months: int | None) -> str:
    if not months:
        return "Not recorded"
    if months % 12 == 0:
        return f"{months // 12} years"
    return f"{months} months"


def start_case(case: str) -> None:
    client_key = st.session_state.get("client_key", VISIBLE_CLIENT_KEYS[0])
    raw = dict(SAMPLE_PROFILES[client_key])
    raw["raw_holdings"] = [dict(item) for item in raw["raw_holdings"]]
    raw["liquidity_schedule"] = [dict(item) for item in raw["liquidity_schedule"]]
    raw["constraints"] = dict(raw["constraints"])
    if case == "investment":
        raw.setdefault("new_capital_amount", 120000.0)
    st.session_state.case = case
    st.session_state.screen = "Overview"
    st.session_state.raw_profile = raw
    st.session_state.profile = None
    st.session_state.research = None
    st.session_state.meeting_brief_approved = False
    st.session_state.custom_stage = "profile"
    st.session_state.advisor_note = ""
    st.session_state.completed_custom = set()
    st.session_state.service = None
    st.session_state.workspace = None
    st.session_state.error = None
    if case == "investment":
        try:
            service = offline_service()
            view = service.start_session(raw)
            st.session_state.service = service
            st.session_state.workspace = view
        except Exception as exc:  # surfaced to advisor, rather than leaving a blank screen
            st.session_state.error = str(exc)
    else:
        try:
            st.session_state.profile = profile_agent().execute(raw)
        except Exception as exc:
            st.session_state.error = str(exc)


def select_client(client_key: str) -> None:
    """Open a client workspace without starting the advisory process yet."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.client_key = client_key
    st.session_state.raw_profile = dict(SAMPLE_PROFILES[client_key])
    st.session_state.raw_profile["raw_holdings"] = [dict(item) for item in st.session_state.raw_profile.get("raw_holdings", [])]
    st.session_state.raw_profile["liquidity_schedule"] = [dict(item) for item in st.session_state.raw_profile.get("liquidity_schedule", [])]
    st.session_state.raw_profile["constraints"] = dict(st.session_state.raw_profile.get("constraints", {}))
    st.session_state.profile = profile_agent().execute(st.session_state.raw_profile)
    st.session_state.screen = "Overview"
    st.session_state.event_decision = None
    st.session_state.market_revision_pending = False


def advisor_review_required() -> bool:
    """Whether the Advisor review screen currently has an action to complete."""
    case = st.session_state.get("case")
    if case == "investment":
        view = st.session_state.get("workspace")
        if st.session_state.get("market_revision_pending") and view and view.outcome:
            return True
        return bool(view and view.pending_review and not view.outcome)
    if case in {"meeting", "market"}:
        return st.session_state.get("custom_stage") != "complete"
    return False


def finish_market_response() -> None:
    """Complete a revised market response from a Streamlit button callback."""
    st.session_state.market_revision_pending = False
    st.session_state.case = "market"
    st.session_state.custom_stage = "complete"
    st.session_state.screen = "Advisor review"


def record_market_decision() -> None:
    """Persist the advisor's market decision before Streamlit rebuilds widgets."""
    choice = st.session_state.get("event_choice", "Maintain current advice")
    reason = st.session_state.get("event_rationale", "").strip()
    follow_up = st.session_state.get("event_followup", "").strip()
    st.session_state.event_decision = {"choice": choice, "reason": reason, "follow_up": follow_up}
    st.session_state.completed_custom.add("decision")
    if choice.startswith("Revise"):
        st.session_state.market_revision_pending = True
        raw = dict(st.session_state.raw_profile)
        raw["advisor_notes"] += " Market event context: interest-rate expectations shifted; reassess existing bond exposure."
        try:
            service = offline_service()
            st.session_state.workspace = service.start_session(raw)
            st.session_state.service = service
            st.session_state.case = "investment"
            st.session_state.screen = "Advisor review"
        except Exception as exc:
            st.session_state.error = str(exc)
    else:
        st.session_state.custom_stage = "complete"


def current_stage() -> str:
    if st.session_state.get("case") == "investment":
        view = st.session_state.get("workspace")
        if view and view.pending_review:
            return view.pending_review.stage
        return view.center_panel.current_stage if view else "client_profile"
    return st.session_state.get("custom_stage", "profile")


def workflow_items() -> list[tuple[str, str]]:
    if st.session_state.get("case") == "investment":
        view = st.session_state.get("workspace")
        if view:
            labels = {
                "COMPLETED": "Completed",
                "WAITING_HUMAN": "In review",
                "IN_PROGRESS": "In progress",
                "PENDING": "Not started",
                "FAILED": "Needs attention",
            }
            return [(stage.display_name, labels.get(stage.status, stage.status.title())) for stage in view.workflow_stages]
    case = st.session_state.get("case")
    if case == "meeting":
        keys = [("Client profile", "profile"), ("Market research", "research"), ("Meeting brief", "brief")]
    else:
        keys = [("Client profile", "profile"), ("Market research", "research"), ("Advisor decision", "decision")]
    items = []
    active = st.session_state.get("custom_stage", "profile")
    done = st.session_state.get("completed_custom", set())
    for label, key in keys:
        status = "Completed" if key in done else ("In review" if key == active else "Not started")
        items.append((label, status))
    return items


def show_workflow() -> None:
    st.markdown("<div class='eyebrow'>Workflow checkpoints</div>", unsafe_allow_html=True)
    items = workflow_items()
    rows = [items] if len(items) <= 3 else [items[:3], items[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, (label, status) in zip(cols, row):
            with col, st.container(border=True, height=104):
                st.caption(label)
                if status == "Completed":
                    st.badge(status, icon=":material/check_circle:", color="green")
                elif status in ("In review", "In progress"):
                    st.badge(status, icon=":material/schedule:", color="orange")
                elif status == "Needs attention":
                    st.badge(status, icon=":material/warning:", color="red")
                else:
                    st.badge(status, color="gray")


def render_profile(profile: Any, raw: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Client", profile.name)
    c2.metric("Investment horizon", f"{profile.time_horizon_months // 12} years")
    c3.metric("Risk profile", profile.calculated_risk_tier)
    c4.metric("Base currency", profile.base_currency)
    st.subheader("Client goals and context")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("<div class='callout'>", unsafe_allow_html=True)
        st.markdown("**Primary goal**")
        st.write("\n".join(f"- {goal}" for goal in profile.goals))
        st.markdown("**Investment experience**")
        st.write(profile.experience_level)
        st.markdown("**Advisor context**")
        st.text(profile.advisor_notes)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='callout'>", unsafe_allow_html=True)
        st.markdown("**Liquidity plan**")
        if profile.liquidity_schedule:
            for event in profile.liquidity_schedule:
                st.write(f"{event.purpose}: {fmt_money(event.amount, profile.base_currency)} in {event.due_months // 12} years")
        else:
            st.write("No scheduled liquidity event recorded.")
        st.markdown("**Risk range**")
        st.write(f"{profile.risk_score_range[0]:.1f}–{profile.risk_score_range[1]:.1f} / 5")
        st.markdown("</div>", unsafe_allow_html=True)
    wealth = raw.get("total_wealth")
    st.caption(f"Prototype data: total portfolio value {fmt_money(wealth, profile.base_currency)} is an illustrative assumption; the assignment specifies the allocation mix and S$120,000 new capital, but not total wealth.".replace("$", r"\$"))
    weights_total = sum(h.weight_pct for h in profile.parsed_holdings) or 1
    portfolio_risk = sum(h.risk_rating * h.weight_pct for h in profile.parsed_holdings) / weights_total
    if portfolio_risk > profile.risk_score_range[1]:
        st.warning(
            f"The rule model estimates current portfolio risk at {portfolio_risk:.2f}, above this "
            f"{profile.calculated_risk_tier} profile’s {profile.risk_score_range[1]:.2f} ceiling. "
            "Review the client’s confirmed risk tolerance and current mix before approving new ideas."
        )
    if profile.unresolved_flags:
        st.subheader("Profile flags")
        for flag in profile.unresolved_flags:
            st.warning(flag.message)
    render_profile_generation(profile, raw)


def render_profile_generation(profile: Any, raw: dict[str, Any]) -> None:
    """Slide 25: show how the profile was generated so the advisor can review each step."""
    from src.tools.profile_tools import required_field_validator, risk_profile_calculator
    with st.expander("How this profile was generated (review each step)"):
        st.markdown("**1. Ingest** · advisor-verified inputs")
        st.json({k: raw.get(k) for k in ("goals", "time_horizon_months", "stated_risk_willingness", "total_wealth", "liquidity_schedule") if k in raw}, expanded=False)
        st.markdown("**2. Required-field validator** · deterministic")
        check = required_field_validator(raw)
        if check.is_valid:
            st.success("All mandatory fields are present.")
        else:
            st.error("Missing: " + ", ".join(check.missing_fields))
        st.markdown("**3. Risk-profile calculator** · deterministic, not the LLM")
        try:
            risk = risk_profile_calculator(raw)
            r1, r2, r3 = st.columns(3)
            r1.metric("Risk willingness", f"{risk.willingness_score:.1f} / 5")
            r2.metric("Risk capacity", f"{risk.capacity_score:.1f} / 5")
            r3.metric("Final band (lower of the two)", risk.risk_tier)
            if risk.has_capacity_conflict:
                st.warning(risk.conflict_rationale or "Willingness exceeds capacity.")
        except Exception as exc:
            st.info(f"Risk band cannot be calculated yet: {exc}")
        st.markdown("**4. Standardised profile** · what downstream agents receive")
        st.json({
            "risk_tier": profile.calculated_risk_tier, "risk_score_range": profile.risk_score_range,
            "time_horizon_months": profile.time_horizon_months,
            "near_term_liquidity_need": profile.near_term_liquidity_need,
            "asset_class_weights": profile.current_asset_class_weights,
            "unresolved_flags": [f.code for f in profile.unresolved_flags],
        }, expanded=False)


def render_portfolio(profile: Any, raw: dict[str, Any]) -> None:
    holdings = profile.parsed_holdings
    rows = [{"Holding": h.name, "Ticker": h.ticker, "Asset class": h.asset_class, "Region": h.region, "Currency": h.currency, "Weight": h.weight_pct} for h in holdings]
    total = sum(row["Weight"] for row in rows)
    st.subheader("Current portfolio")
    st.caption("Holdings and portfolio mix from the client profile")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=100, format="%.1f%%")})
    a, b, c = st.columns(3)
    a.metric("Portfolio value", fmt_money(raw["total_wealth"], profile.base_currency))
    b.metric("Allocation total", f"{total:.0f}%", "Balanced" if abs(total - 100) < 0.01 else "Needs review")
    c.metric("Available new capital", fmt_money(raw.get("new_capital_amount", 0), profile.base_currency))
    st.markdown("**Allocation by asset class**")
    groups: dict[str, float] = {}
    for holding in holdings:
        groups[holding.asset_class] = groups.get(holding.asset_class, 0) + holding.weight_pct
    st.dataframe(pd.DataFrame([{"Asset class": key, "Portfolio weight": value} for key, value in groups.items()]), hide_index=True, width="stretch")


def render_research(brief: Any) -> None:
    if brief is None:
        st.info("Research appears here after the client profile checkpoint is approved.")
        return
    st.subheader("Evidence-backed market brief")
    st.caption("Each finding shows its verified source and publication date.")
    for finding in brief.findings:
        with st.container(border=True):
            st.markdown(f"**{finding.theme}** · {finding.finding_type.replace('_', ' ').title()}")
            st.write(finding.claim)
            st.caption(f"Client relevance: {finding.relevance_to_client}")
            st.caption("Evidence: " + ", ".join(f"{sid} ({date})" for sid, date in zip(finding.cited_source_ids, finding.publication_dates)))
    if brief.key_market_risks:
        st.markdown("**Key risks to keep in view**")
        for risk in brief.key_market_risks:
            st.write(f"- {risk}")
    with st.expander("Sources reviewed"):
        for source in brief.validated_sources:
            st.write(f"**{source.source_id}** · {source.title} · {source.published_at}")
            st.caption(source.text)


def render_ideas(view: Any) -> None:
    if not view:
        st.info("Recommendations appear after the profile and research checkpoints.")
        return
    state = st.session_state.service.get_state(view.session_id)
    ideas = state.get("candidate_ideas")
    assessment = state.get("assessment")
    if not ideas:
        st.info("The investment idea checkpoint has not been reached yet.")
        return
    st.caption(ideas.complementarity_thesis)
    grid = st.columns(3)
    statuses = {item.idea_id: item for item in assessment.idea_evaluations} if assessment else {}
    for col, idea in zip(grid, ideas.ideas):
        with col:
            with st.container(border=True):
                st.markdown(f"<div class='eyebrow'>Idea {idea.slot_number} · {idea.portfolio_role.title()}</div>", unsafe_allow_html=True)
                st.markdown(f"### {idea.ticker}")
                st.markdown(f"**{idea.name}**")
                st.metric("Target allocation", f"{idea.target_weight_pct:.1f}%", f"{idea.action.title()} · {idea.conviction} conviction")
                st.write(idea.thesis)
                st.caption(f"Why now: {idea.why_now}")
                st.caption(f"Client fit: {idea.client_fit_rationale}")
                st.caption(f"Main risk: {'; '.join(idea.key_risks)}")
                status = statuses.get(idea.idea_id)
                if status:
                    if status.status == "PASS": st.success("Suitability check passed")
                    elif status.status == "FLAG": st.warning("Advisor review required")
                    else: st.error("Suitability check failed")
                render_lock_control(view, idea, status)
    if assessment:
        st.markdown("### Portfolio-level suitability")
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall status", assessment.overall_status)
        c2.metric("Hard breaches", "Yes" if assessment.has_blocking_hard_breach else "None")
        c3.metric("Required actions", str(len(assessment.required_actions)))
        if assessment.detected_breaches:
            for breach in assessment.detected_breaches:
                st.warning(f"{breach.rule_id} · {breach.message} (observed {breach.after_value:.1f}; limit {breach.threshold_value:.1f})")
        else:
            st.success("No portfolio-level rule breaches were detected.")
        with st.expander("Review rule checks"):
            checks = [{"Rule": r.rule_id, "Check": r.name, "Result": r.outcome, "Observed": r.after_value, "Limit": r.threshold_value} for r in assessment.rule_results]
            st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")


def render_lock_control(view: Any, idea: Any, status: Any) -> None:
    """Slide 25: each idea is locked in individually; there is no common accept button."""
    pending = view.pending_review
    if not pending or pending.stage != "suitability_review":
        return
    locked = idea.idea_id in pending.locked_idea_ids
    note = st.session_state.get("advisor_note", "")
    if locked:
        st.success("Locked in")
        if "unlock_idea" in pending.allowed_actions and st.button("Unlock", key=f"unlock_{idea.idea_id}", width="stretch"):
            submit_action("unlock_idea", note, idea_id=idea.idea_id)
        return
    needs_note = view.center_panel.data_payload.get("suitability_status") == "FLAG" or (status and status.status == "FLAG")
    failed = bool(status and status.status == "FAIL")
    if st.button(
        "Lock in this idea", key=f"lock_{idea.idea_id}", type="primary", width="stretch",
        disabled=failed or "lock_idea" not in pending.allowed_actions or (needs_note and not note.strip()),
    ):
        submit_action("lock_idea", note, idea_id=idea.idea_id)
    if needs_note and not note.strip() and not failed:
        st.caption("Record an exception rationale in the advisor note to lock this idea.")


def submit_action(action: str, notes: str = "", **kwargs: Any) -> None:
    service = st.session_state.service
    view = st.session_state.workspace
    try:
        request = AdvisorActionRequest(action=action, notes=notes or None, **kwargs)
        st.session_state.workspace = service.execute_action(view.session_id, request)
        st.session_state.error = None
    except Exception as exc:
        st.session_state.error = str(exc)
    st.rerun()


def render_allocation_editor(view: Any) -> None:
    state = st.session_state.service.get_state(view.session_id)
    ideas = state.get("candidate_ideas")
    if not ideas:
        return
    suffix = view.session_id
    with st.expander("Edit an idea allocation"):
        selected = st.selectbox(
            "Idea to edit",
            ideas.ideas,
            format_func=lambda idea: f"Idea {idea.slot_number} · {idea.ticker} · {idea.name}",
            key=f"allocation_idea_{suffix}",
        )
        weight = st.number_input(
            "New target allocation (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(selected.target_weight_pct),
            step=0.5,
            key=f"allocation_weight_{suffix}_{selected.idea_id}",
        )
        note = st.text_input(
            "Reason for the allocation change",
            key=f"allocation_note_{suffix}",
            placeholder="Record why this change fits the client and portfolio.",
        )
        if st.button("Recheck suitability", key=f"recheck_{suffix}", disabled=not note.strip(), type="primary", width="stretch"):
            submit_action("edit_allocation", note, idea_id=selected.idea_id, edits={"target_weight_pct": weight})


def approve_custom_profile() -> None:
    st.session_state.completed_custom.add("profile")
    st.session_state.custom_stage = "research"
    try:
        st.session_state.research = research_agent().execute(st.session_state.profile)
    except Exception as exc:
        st.session_state.error = str(exc)


def approve_custom_research() -> None:
    st.session_state.completed_custom.add("research")
    if st.session_state.case == "meeting":
        st.session_state.custom_stage = "brief"
    else:
        st.session_state.custom_stage = "decision"


def render_advisor_review() -> None:
    case = st.session_state.case
    st.subheader("Advisor checkpoint")
    st.write("Review the evidence and client impact, record your rationale, then choose the next step.")
    if st.session_state.error:
        st.error(st.session_state.error)
    if case == "investment":
        view = st.session_state.workspace
        if not view:
            st.error("The advisory workflow could not start. Restart the review to try again.")
            return
        pending = view.pending_review
        stage = pending.stage if pending else view.center_panel.current_stage
        st.markdown(f"**Current checkpoint:** {pending.agent if pending else stage.replace('_', ' ').title()}")
        if pending:
            st.markdown(f"<div class='callout'>{pending.reason}</div>", unsafe_allow_html=True)
        st.text_area("Advisor rationale or review note", key="advisor_note", placeholder="Record the evidence, suitability reasoning, or follow-up needed.")
        if stage == "client_profile":
            st.caption("Confirm the profile and portfolio details before research begins.")
            current_risk = st.session_state.raw_profile.get("stated_risk_willingness", "Moderate")
            risk_options = ["Conservative", "Moderate", "Balanced Growth", "Aggressive Growth"]
            selected_risk = st.selectbox(
                "Client-confirmed risk tolerance",
                risk_options,
                index=risk_options.index(current_risk) if current_risk in risk_options else 1,
                key=f"confirmed_risk_{view.session_id}",
            )
            st.caption("Change this only when the advisor has confirmed a correction with the client.")
            if selected_risk != current_risk:
                if st.button("Save confirmed profile correction", type="primary", width="stretch", disabled=not st.session_state.advisor_note.strip()):
                    st.session_state.raw_profile["stated_risk_willingness"] = selected_risk
                    submit_action("regenerate", st.session_state.advisor_note, edits={"stated_risk_willingness": selected_risk})
            elif st.button("Confirm client profile", type="primary", width="stretch"):
                submit_action("approve", st.session_state.advisor_note)
        elif stage == "market_intelligence":
            st.caption("Check source dates and client relevance on the Research screen.")
            if st.button("Approve research brief", type="primary", width="stretch"):
                submit_action("approve", st.session_state.advisor_note)
        elif stage == "suitability_review":
            assessment = st.session_state.service.get_state(view.session_id).get("assessment")
            if assessment:
                st.markdown(f"**Suitability result:** {assessment.overall_status}")
                rule_issues = [rule for rule in assessment.rule_results if rule.outcome in {"FLAG", "FAIL", "BLOCK"}]
                for issue in rule_issues:
                    message = f"{issue.rule_id} · {issue.message}"
                    if issue.outcome in {"FAIL", "BLOCK"}:
                        st.error(message)
                    else:
                        st.warning(message)
            st.caption("Review recommendations and rule results on the Recommendations screen.")
            if pending:
                total = len(pending.candidate_ideas)
                st.markdown(f"**Ideas locked in:** {len(pending.locked_idea_ids)} of {total}")
                st.caption("Lock in each idea on the Recommendations screen. The memo is drafted once all three are locked. "
                           "A FLAG needs a documented rationale; a FAIL cannot be locked.")
            if pending and "revise" in pending.allowed_actions:
                if st.button("Revise ideas using this rationale", type="primary", width="stretch", disabled=not st.session_state.advisor_note.strip()):
                    submit_action("revise", st.session_state.advisor_note)
            if pending and "regenerate" in pending.allowed_actions:
                if st.button("Regenerate ideas", width="stretch", disabled=not st.session_state.advisor_note.strip()):
                    submit_action("regenerate", st.session_state.advisor_note)
            if pending and "edit_allocation" in pending.allowed_actions:
                render_allocation_editor(view)
            if pending and "reject" in pending.allowed_actions:
                if st.button("Reject this recommendation set", width="stretch"):
                    submit_action("reject", st.session_state.advisor_note)
        elif stage in {"advisor_approval", "final_memo"}:
            st.caption("Read the completed memo on the Brief screen before recording final sign-off.")
            if pending and "approve" in pending.allowed_actions:
                if st.button("Approve final memo", type="primary", width="stretch"):
                    submit_action("approve", st.session_state.advisor_note)
            elif pending and "reject" in pending.allowed_actions:
                if st.button("Return memo for revision", width="stretch"):
                    submit_action("reject", st.session_state.advisor_note)
            if pending and "revise" in pending.allowed_actions:
                if st.button("Revise recommendations", type="primary", width="stretch", disabled=not st.session_state.advisor_note.strip()):
                    submit_action("revise", st.session_state.advisor_note)
            if pending and "regenerate" in pending.allowed_actions:
                if st.button("Regenerate memo", width="stretch", disabled=not st.session_state.advisor_note.strip()):
                    submit_action("regenerate", st.session_state.advisor_note)
            if pending and "edit_allocation" in pending.allowed_actions:
                render_allocation_editor(view)
        else:
            st.info("The workflow is complete or is not paused at a review checkpoint.")
        if view.outcome:
            st.success(f"Workflow outcome: {view.outcome}")
            if view.outcome == "APPROVED":
                render_approved_memo_download(view)
            if view.outcome == "APPROVED" and st.session_state.get("market_revision_pending"):
                st.button("Finish market response", type="primary", width="stretch", on_click=finish_market_response)
    elif case == "meeting":
        stage = st.session_state.custom_stage
        client_name = st.session_state.raw_profile.get("name", "the client")
        if stage == "profile":
            st.caption(f"Verify {client_name}’s stated goal, investment horizon, risk profile, and portfolio.")
            if st.button("Confirm client profile", type="primary", width="stretch"):
                approve_custom_profile(); st.rerun()
        elif stage == "research":
            st.caption("Review evidence and citations on the Research screen.")
            if st.button("Approve research and prepare brief", type="primary", width="stretch"):
                approve_custom_research(); st.rerun()
        elif stage == "brief":
            st.caption("Review the meeting brief on the Brief screen, then approve it.")
            if st.button("Approve meeting brief", type="primary", width="stretch"):
                st.session_state.completed_custom.add("brief")
                st.session_state.custom_stage = "complete"
                st.rerun()
        else:
            st.success("Meeting preparation is approved and ready to use.")
            st.subheader("Choose the next review")
            st.caption("Your client and meeting preparation is complete. Choose either review; the investment review is not required to respond to a market event.")
            investment_col, market_col = st.columns(2)
            with investment_col:
                st.markdown("**Investment review**")
                st.caption("Assess investment ideas, portfolio suitability, and the advisory memo.")
                st.button(
                    "Review investments",
                    type="primary",
                    width="stretch",
                    key="choose_investment_review",
                    on_click=start_case,
                    args=("investment",),
                )
            with market_col:
                st.markdown("**Market-event response**")
                st.caption("Assess an event’s relevance, record your decision, and document follow-up.")
                st.button(
                    "Respond to a market event",
                    width="stretch",
                    key="choose_market_response",
                    on_click=start_case,
                    args=("market",),
                )
    else:
        stage = st.session_state.custom_stage
        if stage == "profile":
            st.caption("Verify client data and portfolio exposures before assessing the event.")
            if st.button("Confirm client profile", type="primary", width="stretch"):
                approve_custom_profile(); st.rerun()
        elif stage == "research":
            st.caption("Review the event context and evidence on the Research screen.")
            if st.button("Approve event research", type="primary", width="stretch"):
                approve_custom_research(); st.rerun()
        elif stage == "decision":
            st.markdown("**Event:** Interest-rate expectations have shifted. Existing bond exposures may respond differently as expectations change.")
            choice = st.radio("Advisor decision", ["Maintain current advice", "Monitor and follow up", "Revise the advice"], key="event_choice")
            goal = (st.session_state.profile.goals[0] if st.session_state.get("profile") and st.session_state.profile.goals else "the client’s stated objectives")
            reason = st.text_area("Decision rationale", key="event_rationale", placeholder=f"Explain how the evidence and {goal} inform this decision.")
            follow_up = st.text_input("Monitoring trigger or follow-up date", key="event_followup", placeholder="Required when monitoring")
            st.button(
                "Record advisor decision",
                type="primary",
                width="stretch",
                disabled=not reason.strip() or (choice.startswith("Monitor") and not follow_up.strip()),
                on_click=record_market_decision,
            )
        elif stage == "complete":
            st.success("The market response is recorded. No trade was placed.")
            if st.session_state.event_decision:
                st.write(f"**Decision:** {st.session_state.event_decision['choice']}")
                st.write(f"**Rationale:** {st.session_state.event_decision['reason']}")
                if st.session_state.event_decision["follow_up"]:
                    st.write(f"**Follow-up:** {st.session_state.event_decision['follow_up']}")


def render_approved_memo_download(view: Any) -> None:
    """Show the finalized memo download at sign-off and on the Brief screen."""
    pdf_path = st.session_state.service.get_pdf_path(view.session_id)
    pdf_file = Path(pdf_path) if pdf_path else None
    if pdf_file and not pdf_file.is_absolute():
        pdf_file = ROOT / pdf_file
    if pdf_file and pdf_file.is_file():
        st.success("Final advisor-approved memo is ready to download.")
        st.download_button(
            "Download approved memo PDF",
            pdf_file.read_bytes(),
            file_name=pdf_file.name,
            mime="application/pdf",
            type="primary",
            width="stretch",
            key="approved_memo_pdf_download",
        )
    else:
        st.warning("The review is approved, but its PDF file is not available at the expected path.")


def render_brief() -> None:
    case = st.session_state.case
    if case == "investment":
        view = st.session_state.workspace
        if not view:
            st.info("The advisory brief is available after the workflow starts.")
            return
        state = st.session_state.service.get_state(view.session_id)
        memo = state.get("memo")
        if not memo:
            st.info("The memo is prepared after the suitability checkpoint is complete.")
            return
        page = memo.page_1_executive
        st.subheader("Advisory memo preview")
        st.markdown(f"### {page.client_name} · {page.date}")
        st.text(page.investment_objective)
        st.markdown("**Client snapshot**")
        st.text(page.client_snapshot)
        st.markdown("**Portfolio snapshot**")
        st.text(page.portfolio_snapshot)
        st.markdown("**Recommendations at a glance**")
        st.dataframe(pd.DataFrame([{"Idea": row.slot_number, "Instrument": row.ticker, "Action": row.action, "Target": f"{row.target_weight_pct:.1f}%", "Role": row.portfolio_role, "Fit": row.why_it_fits} for row in page.ideas_at_a_glance]), hide_index=True, width="stretch")
        for idea in memo.page_2_details:
            with st.expander(f"Idea {idea.slot_number}: {idea.ticker} — {idea.name}"):
                st.write(idea.thesis)
                st.write(f"**Why now:** {idea.why_now}")
                st.write(f"**Key risks:** {'; '.join(idea.key_risks)}")
                st.write(f"**Evidence:** {'; '.join(idea.cited_sources_formatted)}")
        st.markdown("**Portfolio impact and governance**")
        st.write(memo.page_3_portfolio_view.portfolio_level_impact_narrative)
        for disclosure in memo.page_3_portfolio_view.risk_disclosures:
            st.caption(disclosure)
        approved = view.outcome == "APPROVED"
        if approved:
            render_approved_memo_download(view)
        else:
            st.info("The PDF download becomes available after the advisor approves the final memo.")
    elif case == "meeting":
        profile = st.session_state.get("profile")
        research = st.session_state.get("research")
        if not profile or "profile" not in st.session_state.completed_custom:
            st.info("The meeting brief is prepared once the client profile has been confirmed.")
            return
        st.subheader(f"Meeting brief · {profile.name}")
        st.markdown("**Purpose**")
        goal_text = "; ".join(profile.goals) if profile.goals else "Confirm the client’s investment objectives"
        st.write(f"Review {profile.name}’s objectives ({goal_text}) and decide whether the current portfolio and liquidity plan remain aligned.")
        st.markdown("**Client snapshot**")
        st.write(f"{profile.calculated_risk_tier} risk profile · {profile.experience_level} · {profile.time_horizon_months // 12}-year horizon")
        st.markdown("**Discussion points**")
        mix = ", ".join(f"{weight:.0f}% {asset.lower()}" for asset, weight in profile.current_asset_class_weights.items())
        discussion_points = [
            f"Confirm the goal, timing, and expected cash requirement: {goal_text}.",
            f"Review the current asset mix: {mix or 'not available'}.",
            "Confirm near-term liquidity needs and any amount available to invest.",
            "Review currency exposure and comfort with market volatility.",
        ]
        st.write("\n".join(f"{index}. {point}" for index, point in enumerate(discussion_points, start=1)))
        st.markdown("**Research to discuss**")
        if research:
            for finding in research.findings[:3]:
                st.write(f"- {finding.theme}: {finding.claim} ({', '.join(finding.cited_source_ids)})")
        st.caption("Advisor-prepared discussion guide. Recommendations and client facts require RM confirmation.")
    else:
        st.subheader("Market-event decision record")
        decision = st.session_state.get("event_decision")
        if decision:
            st.write(f"**Decision:** {decision['choice']}")
            st.write(f"**Rationale:** {decision['reason']}")
            if decision["follow_up"]:
                st.write(f"**Follow-up:** {decision['follow_up']}")
            st.caption("The advisor records the decision here; no trade is placed from this workspace.")
        else:
            st.info("The decision record is available after research review.")


def render_overview() -> None:
    case = st.session_state.get("case")
    label = CASE_LABELS.get(case, "Client overview")
    st.markdown(f"<div class='eyebrow'>{label}</div>", unsafe_allow_html=True)
    st.title("Advisory workspace")
    st.markdown("Prepare a client conversation with evidence, suitability checks, and clear advisor sign-off.")
    st.write("")
    profile = st.session_state.get("profile")
    view = st.session_state.get("workspace")
    if st.session_state.error:
        st.error(st.session_state.error)
    if profile or view:
        if view:
            name = view.client_name
            stage = view.pending_review.stage if view.pending_review else view.center_panel.current_stage
        else:
            name = profile.name
            stage = st.session_state.custom_stage
        a, b, c = st.columns(3)
        a.metric("Client", name)
        b.metric("Active checkpoint", stage.replace("_", " ").title())
        c.metric("Review status", "In progress")
        st.markdown("#### Workspace screens")
        st.write("Open a labeled screen to inspect the client, portfolio, research, recommendations, advisor decisions, or brief.")
    else:
        st.info("The client workspace is loading. Restart the review if details do not appear.")


def reset_to_client_list() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def start_online_session(api_url: str, raw_input: dict[str, Any]) -> None:
    try:
        response = httpx.post(f"{api_url}/sessions", json=raw_input, timeout=120.0)
        response.raise_for_status()
        st.session_state.online_workspace = response.json()
        st.session_state.online_error = None
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1200]
        st.session_state.online_error = f"The API rejected the intake ({exc.response.status_code}): {detail}"
        st.session_state.online_workspace = None
    except httpx.RequestError as exc:
        st.session_state.online_error = f"Could not reach {api_url}. Start the REST API and try again. ({exc})"
        st.session_state.online_workspace = None
    st.rerun()


def render_online_result(api_url: str) -> None:
    if st.session_state.get("online_error"):
        st.error(st.session_state.online_error)
    online_workspace = st.session_state.get("online_workspace")
    if online_workspace:
        st.success(f"Online review started for {online_workspace.get('client_name', 'the client')}.")
        st.write(f"**Session:** `{online_workspace.get('session_id', 'unknown')}`")
        st.write(f"**Current stage:** {online_workspace.get('center_panel', {}).get('current_stage', 'Profile review')}")
        pending = online_workspace.get("pending_review") or {}
        if pending:
            st.write(f"**Advisor checkpoint:** {pending.get('stage', 'Profile review')} · {pending.get('agent', 'Advisory team')}")
            st.write(pending.get("reason", "Review the client profile and confirm the information."))
        st.link_button("Open API workspace", f"{api_url}/docs")


def render_online_existing_clients(api_url: str) -> None:
    st.title("Existing clients")
    st.markdown("Choose a sample client to start an online review with the configured model provider.")
    for row_start in range(0, len(VISIBLE_CLIENT_KEYS), 2):
        cards = st.columns(2)
        for col, client_key in zip(cards, VISIBLE_CLIENT_KEYS[row_start:row_start + 2]):
            raw = SAMPLE_PROFILES[client_key]
            with col, st.container(border=True):
                st.markdown(f"<div class='eyebrow'>{raw['client_id']}</div>", unsafe_allow_html=True)
                st.markdown(f"#### {raw['name']}")
                st.write(CLIENT_CARD_SUMMARIES.get(client_key, raw.get("goals", ["Objectives not provided"])[0]))
                risk = raw.get("stated_risk_willingness", "Not recorded")
                risk_label = risk if "risk" in risk.lower() else f"{risk} risk"
                st.caption(f"{risk_label} · {fmt_horizon(raw.get('time_horizon_months'))}")
                if st.button("Start online review", key=f"online_{client_key}", type="primary", width="stretch"):
                    start_online_session(api_url, json.loads(json.dumps(raw)))


def render_online_intake(api_url: str) -> None:
    st.title("New client")
    st.markdown("Enter client-confirmed details to start an online advisory session.")
    with st.form("online_client_intake"):
        st.markdown("#### Client details")
        id_col, name_col, currency_col = st.columns([1, 2, 1])
        with id_col:
            client_id = st.text_input(
                "Client ID",
                value=st.session_state.get("online_default_client_id", f"CL-{uuid4().hex[:8].upper()}"),
            )
        with name_col:
            name = st.text_input("Client name")
        with currency_col:
            currency = st.selectbox("Base currency", ["SGD", "USD", "EUR", "GBP", "HKD"])

        goal_text = st.text_area("Investment objectives", placeholder="Enter one objective per line")
        horizon_col, risk_col, wealth_col, experience_col = st.columns(4)
        with horizon_col:
            horizon_months = st.number_input("Time horizon (months)", min_value=1, value=60, step=1)
        with risk_col:
            risk_willingness = st.selectbox(
                "Risk willingness", ["Conservative", "Low to moderate", "Moderate", "Balanced growth", "Aggressive growth", "High risk"]
            )
        with wealth_col:
            total_wealth = st.number_input("Total wealth", min_value=0.0, value=0.0, step=10000.0)
        with experience_col:
            experience = st.selectbox("Investment experience", ["New investor", "ETF investor", "Experienced investor", "Professional investor"])

        st.markdown("#### Current portfolio")
        st.caption("Enter the client-confirmed holdings. Weights must total 100%.")
        holdings_seed = pd.DataFrame([{
            "ticker": "", "name": "", "asset_class": "Equities", "region": "Global",
            "currency": currency, "weight_pct": 0.0, "risk_rating": 3,
        }])
        holdings_frame = st.data_editor(
            holdings_seed,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "name": st.column_config.TextColumn("Holding name", required=True),
                "asset_class": st.column_config.TextColumn("Asset class", required=True),
                "region": st.column_config.TextColumn("Region", required=True),
                "currency": st.column_config.TextColumn("Currency", required=True),
                "weight_pct": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0, step=0.1),
                "risk_rating": st.column_config.NumberColumn("Risk 1–5", min_value=1, max_value=5, step=1),
            },
        )

        st.markdown("#### Liquidity needs")
        st.caption("Add planned cash needs, or leave the table empty if the client has none to record.")
        liquidity_frame = st.data_editor(
            pd.DataFrame(columns=["due_months", "amount", "purpose"]),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "due_months": st.column_config.NumberColumn("Due in months", min_value=1, step=1),
                "amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=1000.0),
                "purpose": st.column_config.TextColumn("Purpose"),
            },
        )
        advisor_notes = st.text_area("Advisor context", placeholder="Optional notes for the review")
        submitted = st.form_submit_button("Start online review", type="primary", width="stretch")

    if not submitted:
        return

    goals = [line.strip() for line in goal_text.splitlines() if line.strip()]
    holdings = holdings_frame.replace({pd.NA: None}).to_dict(orient="records")
    holdings = [row for row in holdings if str(row.get("ticker") or "").strip()]
    try:
        total_weight = sum(float(row.get("weight_pct") or 0) for row in holdings)
    except (TypeError, ValueError):
        total_weight = -1
    if not name.strip() or not client_id.strip() or not goals:
        st.error("Enter a client ID, client name, and at least one investment objective.")
        return
    if total_wealth <= 0:
        st.error("Enter the client-confirmed total wealth.")
        return
    if not holdings or abs(total_weight - 100.0) > 0.1:
        st.error(f"Enter at least one client holding and make the weights total 100% (currently {total_weight:.1f}%).")
        return

    liquidity_schedule = []
    for row in liquidity_frame.to_dict(orient="records"):
        if not row.get("purpose") and row.get("amount") is None and row.get("due_months") is None:
            continue
        if not row.get("purpose") or row.get("amount") is None or row.get("due_months") is None:
            st.error("Complete all fields for each liquidity need, or remove the unfinished row.")
            return
        liquidity_schedule.append({
            "due_months": int(row["due_months"]),
            "amount": float(row["amount"]),
            "purpose": str(row["purpose"]).strip(),
        })

    raw_input = {
        "client_id": client_id.strip(),
        "name": name.strip(),
        "base_currency": currency,
        "goals": goals,
        "time_horizon_months": int(horizon_months),
        "stated_risk_willingness": risk_willingness,
        "liquidity_schedule": liquidity_schedule,
        "total_wealth": float(total_wealth),
        "experience_level": experience,
        "raw_holdings": holdings,
        "constraints": {"excluded_sectors": [], "excluded_countries": [], "esg_mandate": False, "max_single_stock_pct": 20.0},
        "advisor_notes": advisor_notes.strip(),
    }
    start_online_session(api_url, raw_input)


def render_online_mode() -> None:
    st.title("Online advisory")
    st.markdown("Use a sample client or enter a new client profile.")
    st.info("Starting a review sends client details to the configured OpenRouter model provider. Model usage may consume credits.")
    api_url = st.text_input(
        "Advisory API address",
        value=st.session_state.get("online_api_url", "http://127.0.0.1:8000"),
        help="The REST API must be running with an OpenRouter key configured.",
    ).strip().rstrip("/")
    st.session_state.online_api_url = api_url
    source = st.segmented_control(
        "Client source",
        ["Existing clients", "New client"],
        default="Existing clients",
        key="online_client_source",
        selection_mode="single",
        label_visibility="collapsed",
        width="content",
    )
    if source == "New client":
        render_online_intake(api_url)
    else:
        render_online_existing_clients(api_url)
    render_online_result(api_url)


def render_client_picker() -> None:
    st.title("Clients")
    st.markdown("Select a client to open their advisory workspace.")
    for row_start in range(0, len(VISIBLE_CLIENT_KEYS), 2):
        cards = st.columns(2)
        for col, client_key in zip(cards, VISIBLE_CLIENT_KEYS[row_start:row_start + 2]):
            raw = SAMPLE_PROFILES[client_key]
            with col, st.container(border=True):
                st.markdown(f"<div class='eyebrow'>{raw['client_id']}</div>", unsafe_allow_html=True)
                st.markdown(f"#### {raw['name']}")
                st.write(CLIENT_CARD_SUMMARIES.get(client_key, raw.get("goals", ["Objectives not provided"])[0]))
                risk = raw.get("stated_risk_willingness", "Not recorded")
                risk_label = risk if "risk" in risk.lower() else f"{risk} risk"
                st.caption(f"{risk_label} · {fmt_horizon(raw.get('time_horizon_months'))}")
                if st.button("Open client", key=f"open_{client_key}", type="primary", width="stretch"):
                    select_client(client_key)
                    st.rerun()


def render_client_home() -> None:
    profile = st.session_state.profile
    raw = st.session_state.raw_profile
    top_a, top_b = st.columns([4, 1])
    with top_a:
        st.title(profile.name)
        st.markdown("Client relationship overview")
    with top_b:
        if st.button("All clients", width="stretch"):
            reset_to_client_list()
            st.rerun()
    a, b, c = st.columns(3)
    a.metric("Client ID", profile.client_id)
    b.metric("Risk profile", profile.calculated_risk_tier)
    c.metric("Portfolio value", fmt_money(raw.get("total_wealth", 0), profile.base_currency))
    st.markdown("### Objectives")
    st.write(" · ".join(profile.goals) if profile.goals else "Objectives need advisor confirmation.")
    missing = [field for flag in profile.unresolved_flags if flag.code == "MISSING_MANDATORY_FIELD" for field in flag.affected_fields]
    if missing:
        st.warning("Client information needs confirmation: " + ", ".join(sorted(set(missing))) + ". Suitability checks will stop until the required information is recorded.")
    st.markdown("### Advisory process")
    st.write("Client and meeting preparation → Choose an investment review or a market-event response")
    if st.button("Start client review", key=f"start_review_{st.session_state.client_key}", type="primary", width="stretch"):
        start_case("meeting")
        st.rerun()


def main() -> None:
    st.markdown("<div class='eyebrow'>UBS Prism</div>", unsafe_allow_html=True)
    if not st.session_state.get("client_key"):
        mode = st.segmented_control(
            "Advisory mode",
            ["Offline clients", "Online mode"],
            default="Offline clients",
            key="landing_mode",
            selection_mode="single",
            label_visibility="collapsed",
            width="content",
        )
        if mode == "Online mode":
            render_online_mode()
        else:
            render_client_picker()
        return
    if not st.session_state.get("case"):
        render_client_home()
        return

    case = st.session_state.case
    client_name = st.session_state.raw_profile.get("name", "Client")
    top_a, top_b = st.columns([4, 1])
    with top_a:
        st.title("Advisory workspace")
        st.markdown(f"<span class='subtle'>{client_name} · {CASE_LABELS[case]}</span>", unsafe_allow_html=True)
    with top_b:
        if st.button("All clients", width="stretch"):
            reset_to_client_list()
            st.rerun()
    show_workflow()
    action_needed = advisor_review_required()
    review_label = "Advisor review · Needed" if action_needed else "Advisor review"
    screen_options = [review_label if item == "Advisor review" else item for item in SCREENS]
    previous_screen = st.session_state.get(
        "screen", st.session_state.get("workspace_screen", "Overview")
    )
    selected_screen = previous_screen if previous_screen in screen_options else (
        review_label if str(previous_screen).startswith("Advisor review") else "Overview"
    )
    st.session_state.screen = selected_screen
    if action_needed:
        st.badge("Advisor review needed", color="orange")
        if selected_screen != review_label:
            st.html("""
            <style>
                .st-key-screen [role="radiogroup"] > button:nth-child(6)[aria-checked="false"],
                .st-key-screen [role="radiogroup"] > label:nth-child(6):has(input:not(:checked)),
                .st-key-screen [role="radiogroup"] [role="radio"]:nth-child(6):not([aria-checked="true"]) {
                    background-color: #3a2a1d !important;
                    border-color: #8b5e2a !important;
                    color: #ee9e32 !important;
                    box-shadow: none !important;
                }
                .st-key-screen [role="radiogroup"] > button:nth-child(6)[aria-checked="false"]:hover,
                .st-key-screen [role="radiogroup"] > label:nth-child(6):has(input:not(:checked)):hover,
                .st-key-screen [role="radiogroup"] [role="radio"]:nth-child(6):not([aria-checked="true"]):hover {
                    background-color: #493522 !important;
                    border-color: #ad7730 !important;
                    color: #f5b24f !important;
                }
            </style>
            """)
    selected_screen = st.segmented_control(
        "Workspace screen",
        screen_options,
        key="screen",
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
    )
    screen = (
        "Advisor review"
        if selected_screen and selected_screen.startswith("Advisor review")
        else selected_screen
    )
    if action_needed and screen != "Advisor review":
        st.warning("Advisor review is waiting for your action. Open the highlighted tab to continue.")
    if screen == "Overview":
        render_overview()
    elif screen == "Client profile":
        profile = st.session_state.get("profile")
        if profile is None and st.session_state.get("workspace"):
            state = st.session_state.service.get_state(st.session_state.workspace.session_id)
            profile = state.get("profile")
        if profile:
            render_profile(profile, st.session_state.raw_profile)
        else:
            st.info("Client profile details will appear when the workflow starts.")
    elif screen == "Portfolio":
        profile = st.session_state.get("profile")
        if profile is None and st.session_state.get("workspace"):
            profile = st.session_state.service.get_state(st.session_state.workspace.session_id).get("profile")
        if profile:
            render_portfolio(profile, st.session_state.raw_profile)
        else:
            st.info("Portfolio details will appear when the workflow starts.")
    elif screen == "Research":
        research = st.session_state.get("research")
        if research is None and st.session_state.get("workspace"):
            research = st.session_state.service.get_state(st.session_state.workspace.session_id).get("research_brief")
        render_research(research)
        if case == "market":
            st.markdown("### Event context")
            profile = st.session_state.get("profile")
            goal = profile.goals[0] if profile and profile.goals else "the client’s stated objectives"
            st.write(f"Interest-rate expectations shifted. Review existing fixed-income exposure and its effect on {goal.lower()}.")
            st.caption("Illustrative rate scenario; live market data is not connected.")
    elif screen == "Recommendations":
        render_ideas(st.session_state.get("workspace"))
        if case in {"meeting", "market"}:
            st.info("This workflow does not generate a new trade recommendation unless the advisor selects Revise in the market-event decision.")
    elif screen == "Advisor review":
        render_advisor_review()
    elif screen == "Brief":
        render_brief()

    st.divider()
    st.caption("Advisor approval required · no orders are placed")

if __name__ == "__main__":
    main()
