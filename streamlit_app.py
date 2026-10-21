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

        # Risk & Breach Warnings (if any)
        if cp.risk_warnings:
            for w in cp.risk_warnings:
                box_class = "warning-box" if w.severity == "FAIL" else "flag-box"
                st.markdown(f"""
                <div class="{box_class}">
                    <b>[{w.severity}] {w.metric_name}</b><br/>
                    {w.message}<br/>
                    <small>Threshold: {w.threshold_value}% | After Allocation: {w.after_value}%</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pass-box">
                <b>✅ ALL SUITABILITY RULES PASSED</b><br/>
                No hard limit breaches or concentration warnings detected.
            </div>
            """, unsafe_allow_html=True)

        # Allocation Diff Table
        if cp.allocation_diffs:
            st.markdown("#### 📊 Asset Allocation Shift (Before vs. After)")
            table_rows = []
            for d in cp.allocation_diffs:
                delta_str = f"+{d.delta_pct:.1f}%" if d.delta_pct > 0 else f"{d.delta_pct:.1f}%"
                status_icon = "⚠️" if d.is_breached else "✓"
                table_rows.append({
                    "Type": d.category_type,
                    "Name": d.name,
                    "Current %": f"{d.before_pct:.1f}%",
                    "Proposed %": f"{d.after_pct:.1f}%",
                    "Delta": delta_str,
                    "Limit": status_icon
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # Candidate Ideas Display
        candidate_ideas = cp.data_payload.get("candidate_ideas", [])
        if candidate_ideas:
            st.markdown("#### 💡 Candidate Ideas & Actions")
            for idea in candidate_ideas:
                with st.expander(f"Slot {idea.get('slot_number')}: {idea.get('ticker')} - {idea.get('name')} ({idea.get('action')})", expanded=True):
                    col_i1, col_i2 = st.columns([3, 1])
                    with col_i1:
                        st.write(f"**Role:** `{idea.get('portfolio_role')}` &nbsp;|&nbsp; **Target:** `{idea.get('target_weight_pct')}%` (Range: {idea.get('allocation_range', {}).get('min_pct')}% - {idea.get('allocation_range', {}).get('max_pct')}%)")
                        st.write(f"**Thesis:** {idea.get('thesis')}")
                        st.write(f"**Why Now:** {idea.get('why_now')}")
                        st.write(f"**Client Fit:** {idea.get('client_fit_rationale')}")
                        st.caption(f"Evidence Ref: {', '.join(idea.get('cited_evidence_ids', []))}")

                    with col_i2:
                        st.markdown("**RM Commands:**")
                        # Reject Button
                        if st.button("❌ Reject", key=f"btn_reject_{idea.get('idea_id')}"):
                            with st.spinner(f"Rejecting {idea.get('ticker')} and triggering selective regeneration..."):
                                service.execute_action(
                                    view.session_id,
                                    AdvisorActionRequest(
                                        action="reject",
                                        idea_id=idea.get("idea_id"),
                                        notes=f"RM rejected {idea.get('ticker')} for slot {idea.get('slot_number')}."
                                    )
                                )
                                st.session_state.workspace_view = service.get_workspace_view(view.session_id)
                                st.rerun()

                        # Revise Button
                        with st.popover("✏️ Revise"):
                            rev_notes = st.text_input("Revision Notes:", key=f"notes_{idea.get('idea_id')}")
                            if st.button("Submit Revision", key=f"btn_sub_rev_{idea.get('idea_id')}"):
                                service.execute_action(
                                    view.session_id,
                                    AdvisorActionRequest(
                                        action="revise",
                                        idea_id=idea.get("idea_id"),
                                        notes=rev_notes or "Rebalance to lower risk rating."
                                    )
                                )
                                st.session_state.workspace_view = service.get_workspace_view(view.session_id)
                                st.rerun()

        # Human-in-the-Loop Action Gate
        st.markdown("---")
        st.markdown("#### 🛡️ Relationship Manager Action Gate")

        if cp.requires_human_action:
            st.warning(f"ACTION REQUIRED: {cp.interrupt_reason or 'Please review and confirm checkpoint.'}")

            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                if "accept" in cp.allowed_actions:
                    if st.button("✅ Accept & Proceed", type="primary", use_container_width=True):
                        service.execute_action(view.session_id, AdvisorActionRequest(action="accept"))
                        st.session_state.workspace_view = service.get_workspace_view(view.session_id)
                        st.rerun()

            with col_a2:
                if "override_flag" in cp.allowed_actions:
                    if st.button("⚠️ Confirm Exception Override", use_container_width=True):
                        service.execute_action(view.session_id, AdvisorActionRequest(
                            action="override_flag",
                            notes="RM certified exception override based on client consultation."
                        ))
                        st.session_state.workspace_view = service.get_workspace_view(view.session_id)
                        st.rerun()

            with col_a3:
                if "resolve_conflict" in cp.allowed_actions:
                    if st.button("🔧 Resolve Profile Conflict", use_container_width=True):
                        service.execute_action(view.session_id, AdvisorActionRequest(
                            action="resolve_conflict",
                            conflict_resolution="Align with lower risk capacity due to near-term liquidity commitments."
                        ))
                        st.session_state.workspace_view = service.get_workspace_view(view.session_id)
                        st.rerun()
        else:
            if st.button("✅ Approve Recommendation & Render Memo PDF", type="primary", use_container_width=True):
                service.execute_action(view.session_id, AdvisorActionRequest(action="accept"))
                st.session_state.workspace_view = service.get_workspace_view(view.session_id)
                st.rerun()

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
                box_style = "memo-highlight" if sec.is_highlighted else "memo-normal"
                highlight_tag = "<b>[CURRENTLY AFFECTED BY ACTIVE STAGE]</b><br/>" if sec.is_highlighted else ""
                st.markdown(f"""
                <div class="{box_style}">
                    <small><b>{sec.title}</b> (Page {sec.page_number})</small><br/>
                    {highlight_tag}
                    <div style="font-size:0.85rem; margin-top:4px;">
                        {sec.content_html_or_md}
                    </div>
                </div>
                """, unsafe_allow_html=True)
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