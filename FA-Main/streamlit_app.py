"""UBS Advisor Workspace - 3-Panel Human-Led Multi-Agent Advisory System (Slide 20)."""

import os
import json
import streamlit as st
from typing import Dict, Any, Optional

from src.api.service import AdvisorySessionService
from src.schemas.workspace import AdvisorActionRequest, AdvisorWorkspaceView

# Page Configuration
st.set_page_config(
    page_title="UBS Advisory Workspace | Human-Led Multi-Agent System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Institutional Wealth Management aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .top-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-bottom: 3px solid #E60000;
        padding: 1.2rem 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stage-item {
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 6px;
        font-size: 0.9rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .stage-completed { background: #ECFDF5; border-left: 4px solid #10B981; color: #065F46; }
    .stage-active { background: #EFF6FF; border-left: 4px solid #3B82F6; color: #1E40AF; font-weight: 600; }
    .stage-waiting { background: #FEF3C7; border-left: 4px solid #F59E0B; color: #92400E; font-weight: 600; }
    .stage-pending { background: #F8FAFC; border-left: 4px solid #CBD5E1; color: #64748B; }

    .warning-box {
        background: #FEF2F2;
        border: 1px solid #F87171;
        border-left: 5px solid #EF4444;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.8rem 0;
        color: #991B1B;
    }
    .flag-box {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 5px solid #F59E0B;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.8rem 0;
        color: #92400E;
    }
    .pass-box {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 5px solid #22C55E;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.8rem 0;
        color: #166534;
    }
    .memo-highlight {
        background-color: #FEF08A;
        border-left: 4px solid #EAB308;
        padding: 0.6rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
    }
    .memo-normal {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.6rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load sample profiles
def load_sample_profiles() -> Dict[str, Any]:
    sample_path = os.path.join(os.path.dirname(__file__), "data", "samples", "client_profiles.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Initialize Session State
if "service" not in st.session_state:
    st.session_state.service = AdvisorySessionService()
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "workspace_view" not in st.session_state:
    st.session_state.workspace_view = None

service: AdvisorySessionService = st.session_state.service

def submit_action(session_id: str, **kwargs: Any) -> None:
    """Send an RM review decision to the graph and refresh the workspace."""
    try:
        with st.spinner("Applying advisor decision..."):
            st.session_state.workspace_view = service.execute_action(session_id, AdvisorActionRequest(**kwargs))
        st.rerun()
    except ValueError as e:
        st.error(str(e))

def split_csv(text: str) -> list:
    return [t.strip() for t in text.split(",") if t.strip()]

def review_buttons(session_id: str, review, key: str, approve_label: str = "✅ Approve & Continue",
                   regenerate_label: str = "🔄 Regenerate", edits: Optional[Dict[str, Any]] = None) -> None:
    """Shared approve / regenerate / reject gate shown under every agent's output."""
    notes = st.text_area("RM notes / feedback for the agent:", key=f"notes_{key}")
    cols = st.columns(3)
    with cols[0]:
        if "approve" in review.allowed_actions:
            if st.button(approve_label, type="primary", use_container_width=True, key=f"approve_{key}"):
                submit_action(session_id, action="approve", notes=notes or None)
        elif "override_flag" in review.allowed_actions:
            if st.button("⚠️ Approve with Exception Override", type="primary", use_container_width=True, key=f"override_{key}"):
                if not notes:
                    st.error("An exception override needs an RM justification note.")
                else:
                    submit_action(session_id, action="override_flag", notes=notes)
        else:
            st.caption("⛔ Failing ideas cannot be approved: revise, edit or reject them.")
    with cols[1]:
        if "regenerate" in review.allowed_actions:
            if st.button(regenerate_label, use_container_width=True, key=f"regen_{key}"):
                submit_action(session_id, action="regenerate", notes=notes or None, edits=edits)
    with cols[2]:
        if "reject" in review.allowed_actions:
            if st.button("⛔ Reject & End Session", use_container_width=True, key=f"reject_{key}"):
                submit_action(session_id, action="reject", notes=notes or None)

# Header Banner
st.markdown("""
<div class="top-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h2 style="margin:0; font-size:1.6rem; letter-spacing:-0.5px;">UBS Advisory Workspace</h2>
            <p style="margin:4px 0 0 0; opacity:0.85; font-size:0.9rem;">Human-Led Multi-Agent Investment Advisory Architecture & Portfolio Risk Governance</p>
        </div>
        <div style="text-align:right;">
            <span style="background:#E60000; padding:4px 10px; border-radius:4px; font-weight:bold; font-size:0.8rem;">RELATIONSHIP MANAGER WORKSPACE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Client Selection & Session Controls
with st.sidebar:
    st.header("👤 Client Session Intake")

    sample_profiles = load_sample_profiles()
    profile_options = list(sample_profiles.keys()) if sample_profiles else ["Custom Intake"]

    selected_profile_key = st.selectbox(
        "Select Client Profile:",
        options=profile_options,
        format_func=lambda x: {
            "demo_sarah": "Sarah, 38 (Demo: concentration FLAG + selective regeneration)",
            "demo_daniel": "Daniel, 29 (Demo: willingness vs capacity, R5/R6)",
            "demo_mdm_lim": "Mdm Lim, 62 (Demo: 'guaranteed 6%' request, R9)",
            "demo_aaron": "Aaron, 45 (Demo: missing horizon blocks, R1)",
            "balanced_client": "Dr. Elena Rostova (Balanced Growth 5Y)",
            "contradictory_client": "Marcus Vance (Contradiction: Aggressive vs 18M Need)",
            "incomplete_client": "Anonymous (Missing Fields Test)"
        }.get(x, x)
    )

    client_payload = sample_profiles.get(selected_profile_key, {})

    if st.button("🚀 Initialize Advisory Pipeline", type="primary", use_container_width=True):
        with st.spinner("Executing multi-agent advisory graph..."):
            session_id = f"sess_{selected_profile_key[:8]}_{os.urandom(2).hex()}"
            view = service.start_session(raw_input=client_payload, session_id=session_id)
            st.session_state.current_session_id = session_id
            st.session_state.workspace_view = view
            st.rerun()

    st.markdown("---")
    st.subheader("📋 Session Context")
    if st.session_state.current_session_id:
        st.write(f"**Session ID:** `{st.session_state.current_session_id}`")
        if st.session_state.workspace_view:
            wv = st.session_state.workspace_view
            st.write(f"**Client:** {wv.client_name}")
            st.write(f"**ID:** {wv.client_id}")
    else:
        st.info("No active advisory session. Click 'Initialize Advisory Pipeline' to begin.")

# Main 3-Panel Layout
view: Optional[AdvisorWorkspaceView] = st.session_state.workspace_view

if view is not None:
    col_left, col_center, col_right = st.columns([1, 2, 1.4], gap="medium")

    # ==========================================
    # 1. LEFT PANEL: WORKFLOW PROGRESS
    # ==========================================
    with col_left:
        st.markdown("### 📌 1. Workflow Progress")
        st.caption("Agent Orchestration & HITL Checkpoints")

        for stage in view.workflow_stages:
            status_badge = {
                "COMPLETED": "✅ Done",
                "IN_PROGRESS": "🔄 Active",
                "WAITING_HUMAN": "⚠️ RM Action",
                "PENDING": "⏳ Pending",
                "FAILED": "❌ Failed"
            }.get(stage.status, stage.status)

            css_class = {
                "COMPLETED": "stage-completed",
                "IN_PROGRESS": "stage-active",
                "WAITING_HUMAN": "stage-waiting",
                "PENDING": "stage-pending",
                "FAILED": "warning-box"
            }.get(stage.status, "stage-pending")

            st.markdown(f"""
            <div class="stage-item {css_class}">
                <div>
                    <b>{stage.display_name}</b><br/>
                    <small style="opacity:0.85;">{stage.description}</small>
                </div>
                <span>{status_badge}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📝 Advisor Notes Log")
        if view.advisor_notes:
            for note in view.advisor_notes:
                st.info(note)
        else:
            st.caption("No notes recorded yet.")

        # Quick Add Advisor Note
        with st.expander("➕ Add Advisor Note"):
            note_text = st.text_input("Note:", key="input_new_advisor_note")
            if st.button("Save Note"):
                if note_text:
                    service.execute_action(
                        view.session_id,
                        AdvisorActionRequest(action="add_advisor_note", notes=note_text)
                    )
                    st.session_state.workspace_view = service.get_workspace_view(view.session_id)
                    st.rerun()

    # ==========================================
    # 2. CENTER PANEL: DECISION SPACE
    # ==========================================
    with col_center:
        st.markdown("### ⚖️ 2. Decision Space & Risk Controls")
        cp = view.center_panel
        st.markdown(f"**Stage:** `{cp.current_stage.upper()}` &nbsp;|&nbsp; **Active Agent:** `{cp.active_agent_name}`")
        st.write(cp.stage_summary)

        review = view.pending_review
        stage = review.stage if review else None
        output = (review.output or {}) if review else {}

        if view.outcome:
            if view.outcome == "APPROVED":
                st.success("✅ Advisory session approved and signed off by the Relationship Manager.")
            else:
                st.error("⛔ Advisory session was rejected by the Relationship Manager.")
        elif review:
            st.warning(f"🛡️ RM REVIEW REQUIRED — {review.agent}: {review.reason}")

        # ---------- Agent 1: Client Profile review ----------
        if stage == "client_profile":
            st.markdown("#### 👤 Client Profile Agent Output")
            for flag in review.unresolved_flags:
                box_class = "warning-box" if flag.get("severity") == "BLOCKING_INTERRUPT" else "flag-box"
                st.markdown(f"""
                <div class="{box_class}">
                    <b>[{flag.get('severity')}] {flag.get('code')}</b><br/>
                    {flag.get('message')}<br/>
                    <small>Fields: {', '.join(flag.get('affected_fields', []))}</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                f"**Calculated Risk Tier:** `{output.get('calculated_risk_tier')} ({output.get('risk_tier_number')}/5)` &nbsp;|&nbsp; "
                f"**Horizon:** `{output.get('time_horizon_months')} months` &nbsp;|&nbsp; "
                f"**Total Wealth:** `${output.get('total_wealth', 0):,.0f}`"
            )
            st.caption(f"Asset class weights: {output.get('current_asset_class_weights')}")

            st.markdown("##### ✏️ Profile Fields (edit, then Regenerate to re-run the agent)")
            raw = review.raw_input
            raw_constraints = raw.get("constraints") or {}
            f1, f2 = st.columns(2)
            with f1:
                name = st.text_input("Client name", value=str(raw.get("name", "")))
                goals = st.text_input("Goals (comma-separated)", value=", ".join(raw.get("goals") or []))
                horizon = st.number_input("Time horizon (months)", min_value=0, value=int(raw.get("time_horizon_months") or 0))
                willingness = st.text_input("Stated risk willingness", value=str(raw.get("stated_risk_willingness", "")))
            with f2:
                client_id = st.text_input("Client ID", value=str(raw.get("client_id", "")))
                wealth = st.number_input("Total wealth", min_value=0.0, value=float(raw.get("total_wealth") or 0.0), step=10000.0)
                experience = st.text_input("Experience level", value=str(raw.get("experience_level", "")))
                max_single = st.number_input("Max single stock %", min_value=1.0, max_value=100.0,
                                             value=float(raw_constraints.get("max_single_stock_pct", 20.0)))
            excluded = st.text_input("Excluded sectors (comma-separated)", value=", ".join(raw_constraints.get("excluded_sectors", [])))
            advisor_context = st.text_area("Advisor notes (context for the agents; treated as data)", value=str(raw.get("advisor_notes", "")))
            esg = st.checkbox("ESG mandate", value=bool(raw_constraints.get("esg_mandate", False)))

            st.caption("Liquidity schedule")
            liquidity = st.data_editor(
                raw.get("liquidity_schedule") or [{"due_months": 12, "amount": 0.0, "purpose": ""}],
                num_rows="dynamic", use_container_width=True, key="edit_liquidity"
            )
            st.caption("Current holdings")
            holdings = st.data_editor(
                raw.get("raw_holdings") or [{"ticker": "", "name": "", "asset_class": "", "region": "", "currency": "USD", "weight_pct": 0.0, "risk_rating": 3}],
                num_rows="dynamic", use_container_width=True, key="edit_holdings"
            )

            def records(data: Any) -> list:
                rows = data.to_dict("records") if hasattr(data, "to_dict") else list(data)
                return [r for r in rows if any(v not in (None, "", 0, 0.0) for v in r.values())]

            edits = {
                "client_id": client_id, "name": name, "goals": split_csv(goals),
                "time_horizon_months": int(horizon), "stated_risk_willingness": willingness,
                "total_wealth": float(wealth), "experience_level": experience,
                "constraints": {**raw_constraints, "excluded_sectors": split_csv(excluded),
                                "esg_mandate": esg, "max_single_stock_pct": float(max_single)},
                "liquidity_schedule": records(liquidity), "raw_holdings": records(holdings),
                "advisor_notes": advisor_context
            }
            if "approve" not in review.allowed_actions:
                st.info("Mandatory fields are missing — complete them above and Regenerate. Approval is disabled.")
            review_buttons(view.session_id, review, "profile", regenerate_label="🔄 Regenerate with Edits", edits=edits)

        # ---------- Agent 2: Market Intelligence review ----------
        elif stage == "market_intelligence":
            st.markdown("#### 🌐 Market Intelligence Agent Output")
            st.write("**Macro themes:** " + ", ".join(output.get("macro_themes", [])))
            st.markdown("##### Findings")
            st.dataframe([{
                "Rank": f.get("materiality_rank"), "Theme": f.get("theme"), "Claim": f.get("claim"),
                "Type": f.get("finding_type"), "Sources": ", ".join(f.get("cited_source_ids", [])),
                "Published": ", ".join(f.get("publication_dates", []))
            } for f in sorted(output.get("findings", []), key=lambda f: f.get("materiality_rank", 99))],
                use_container_width=True, hide_index=True)
            st.markdown("##### Key market risks")
            for r in output.get("key_market_risks", []):
                st.write(f"- {r}")
            with st.expander("Validated evidence sources"):
                st.dataframe([{
                    "Source": src.get("source_id"), "Title": src.get("title"), "Publisher": src.get("publisher"),
                    "Published": src.get("published_at"), "Age (days)": src.get("age_days")
                } for src in output.get("validated_sources", [])], use_container_width=True, hide_index=True)
            review_buttons(view.session_id, review, "market")

        # ---------- Agents 3 + 4: validated recommendations review ----------
        elif stage == "suitability_review":
            status = output.get("overall_status")
            verdicts = {ie["idea_id"]: ie for ie in output.get("idea_evaluations", [])}
            st.markdown(f"#### 💡 Recommendations after Suitability Review — **{status}**")
            if review.escalated:
                st.error("R10 escalation: an idea still fails after 2 automatic regenerations. Revise it with guidance, edit it, or reject it.")
            st.caption(output.get("critic_notes", ""))
            if review.regeneration_history:
                with st.expander(f"🔁 Ideas replaced automatically in this session ({len(review.regeneration_history)})"):
                    for h in review.regeneration_history:
                        st.markdown(f"- Slot {h['slot']}: **{h['ticker']}** failed {', '.join(h['rules'])} "
                                    f"(attempt {h['attempt']}) — {h['reason']}")

            portfolio_issues = [r for r in output.get("rule_results", []) if r["outcome"] != "PASS"]
            for r in portfolio_issues:
                box_class = "warning-box" if r["outcome"] in ("FAIL", "BLOCK") else "flag-box"
                st.markdown(f'<div class="{box_class}"><b>[{r["outcome"]}] {r["rule_id"]} {r["name"]}</b><br/>{r["message"]}</div>',
                            unsafe_allow_html=True)
            if not portfolio_issues and status == "PASS":
                st.markdown('<div class="pass-box"><b>✅ ALL SUITABILITY RULES PASSED</b><br/>R1-R10 and liquidity coverage pass for the proposed portfolio.</div>',
                            unsafe_allow_html=True)

            if cp.allocation_diffs:
                st.markdown("##### 📊 Before vs. After (rule engine)")
                def fmt(metric_name: str, v: Optional[float]) -> str:
                    if v is None:
                        return "—"
                    return f"{v:.2f}" if ("risk score" in metric_name or "liquidity" in metric_name) else f"{v:.1f}%"
                st.dataframe([{
                    "Rule": d.category_type, "Metric": d.name,
                    "Before": fmt(d.name, d.before_pct), "After": fmt(d.name, d.after_pct),
                    "Limit": fmt(d.name, d.limit_value), "Status": "⚠️" if d.is_breached else "✓"
                } for d in cp.allocation_diffs], use_container_width=True, hide_index=True)

            for idea in review.candidate_ideas:
                verdict = verdicts.get(idea["idea_id"], {})
                badge = {"PASS": "✅ PASS", "FLAG": "⚠️ FLAG", "FAIL": "⛔ FAIL"}.get(verdict.get("status"), "")
                with st.expander(f"{badge} · Slot {idea['slot_number']}: {idea['ticker']} - {idea['name']} ({idea['action']} {idea['target_weight_pct']}%)", expanded=True):
                    st.write(f"**Role:** `{idea['portfolio_role']}` · **Conviction:** `{idea.get('conviction', 'Medium')}` · "
                             f"**Range:** {idea['allocation_range']['min_pct']}% - {idea['allocation_range']['max_pct']}%")
                    st.write(f"**Thesis:** {idea['thesis']}")
                    st.write(f"**Why now:** {idea['why_now']}")
                    st.write(f"**Why it suits the client:** {idea['client_fit_rationale']}")
                    st.write(f"**Key risks:** {', '.join(idea['key_risks'])} · **What would change our view:** {idea['invalidation_trigger']}")
                    st.write(f"**Alternative considered:** {idea.get('alternative_considered') or '—'}")
                    st.caption(f"Evidence: {', '.join(idea['cited_evidence_ids'])}")
                    issues = [r for r in verdict.get("rule_results", []) if r["outcome"] != "PASS"]
                    for r in issues:
                        st.markdown(f"- **{r['rule_id']} {r['outcome']}**: {r['message']}")

                    c1, c2 = st.columns(2)
                    with c1:
                        idea_notes = st.text_input("Revision guidance:", key=f"idea_notes_{idea['idea_id']}")
                        if st.button("↩️ Revise this idea", key=f"revise_{idea['idea_id']}", use_container_width=True):
                            submit_action(view.session_id, action="revise", idea_id=idea["idea_id"],
                                          notes=idea_notes or f"Replace {idea['ticker']} with a better-suited idea.")
                    with c2:
                        new_weight = st.number_input("Allocation %", min_value=0.0, max_value=100.0, step=0.5,
                                                     value=float(idea["target_weight_pct"]), key=f"weight_{idea['idea_id']}")
                        if st.button("✏️ Edit & re-validate", key=f"edit_{idea['idea_id']}", use_container_width=True):
                            submit_action(view.session_id, action="edit_allocation", idea_id=idea["idea_id"],
                                          edits={"target_weight_pct": new_weight})

            review_buttons(view.session_id, review, "recommendations", approve_label="✅ Accept & Draft Memo",
                           regenerate_label="🔄 Regenerate All Ideas")

        # ---------- Agent 5: Memo review & sign-off ----------
        elif stage == "advisor_approval":
            st.markdown("#### 📄 Memo Formatter Agent Output")
            exec_page = output.get("page_1_executive", {})
            st.write(f"**Memo ID:** `{output.get('memo_id')}`")
            st.write(f"**Objective:** {exec_page.get('investment_objective')}")
            st.write(f"**Client snapshot:** {exec_page.get('client_snapshot')}")
            st.info("Review the full memo in the preview panel, then sign off, regenerate the memo, or send an idea back.")
            for g in exec_page.get("ideas_at_a_glance", []):
                c1, c2 = st.columns([4, 1])
                c1.write(f"**{g.get('slot_number')}. {g.get('ticker')}** ({g.get('action')} {g.get('target_weight_pct')}%) — {g.get('portfolio_role')}")
                if c2.button("↩️ Revise idea", key=f"memo_revise_{g.get('idea_id')}"):
                    submit_action(view.session_id, action="revise", idea_id=g.get("idea_id"),
                                  notes=f"RM requested revision of {g.get('ticker')} at memo sign-off.")
            for check in output.get("compliance_checks", []):
                icon = "✓" if check["outcome"] == "PASS" else "⛔"
                st.write(f"{icon} **{check['rule_id']} {check['name']}:** {check['message']}")
            st.caption("The PDF is generated only after you approve.")
            review_buttons(view.session_id, review, "memo", approve_label="✅ Approve, Sign Off & Generate PDF",
                           regenerate_label="🔄 Regenerate Memo")

    # ==========================================
    # 3. RIGHT PANEL: INTERACTIVE MEMO PREVIEW
    # ==========================================
    with col_right:
        st.markdown("### 📄 3. Interactive Memo Preview")
        st.caption("Live Preview Highlighting Stage-Affected Content")

        if view.right_panel.pdf_download_url:
            pdf_path = service.get_pdf_path(view.session_id)
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "📥 Download Final UBS Advisory Memo PDF",
                    data=pdf_bytes,
                    file_name=f"UBS_Advisory_Memo_{view.session_id}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )

        if view.right_panel.sections:
            for sec in view.right_panel.sections:
                with st.container(border=True):
                    st.caption(f"{sec.title} (Page {sec.page_number})")
                    if sec.is_highlighted:
                        st.markdown(":orange[**Affected by the current stage**]")
                    st.markdown(sec.content_html_or_md)
        else:
            st.info("Advisory memo sections will appear dynamically as ideas and risk reviews are finalized.")

else:
    # Welcome view when no session is active
    st.info("👋 Welcome to the UBS Multi-Agent Advisory System. Please select a client profile from the sidebar and click 'Initialize Advisory Pipeline'.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1. Client Profile Agent**
        - Ingests goals, liquidity, and holdings.
        - Compares willingness vs. capacity.
        - Flags missing or contradictory inputs.
        """)
    with col2:
        st.markdown("""
        **2. Market Intelligence & Idea Agents**
        - Retrieves fresh macro research & quant metrics.
        - Generates exactly 3 complementary ideas.
        - Supports selective regeneration.
        """)
    with col3:
        st.markdown("""
        **3. Critic & Memo Formatter Agents**
        - Deterministic portfolio risk calculations.
        - PASS, FLAG, and FAIL status gates.
        - Standardized 3-page UBS Advisory Memo PDF.
        """)