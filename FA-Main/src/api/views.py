"""View projection transforming AdvisoryState into the 3-Panel Advisor Workspace contract (Slide 20)."""

from typing import List, Dict, Any, Optional
from src.graph.state import AdvisoryState
from src.schemas.workspace import (
    AdvisorWorkspaceView, WorkflowStage, DecisionSpacePayload,
    AllocationDiffRow, InteractiveMemoPreview, MemoPreviewSection, PendingReview
)

STAGE_CONFIG = [
    ("client_profile", "Client Profile", "Ingest & reconcile advisor inputs"),
    ("market_intelligence", "Market Intelligence", "Retrieve evidence-backed market data"),
    ("investment_ideas", "Investment Ideas", "Generate complementary candidate ideas"),
    ("suitability_review", "Risk Review", "Independent critic & portfolio risk checks"),
    ("final_memo", "Advisory Memo", "Audit-ready 3-page UBS advisory memo"),
    ("advisor_approval", "Advisor Sign-off", "RM final approval of the memo")
]

AGENT_NAMES = {
    "client_profile": "Client Profile Agent",
    "market_intelligence": "Market Intelligence Agent",
    "investment_ideas": "Investment Idea Agent",
    "suitability_review": "Suitability Critic Agent",
    "final_memo": "Memo Formatter Agent",
    "advisor_approval": "Memo Formatter Agent"
}

def build_workspace_view(
    state: AdvisoryState,
    session_id: str,
    pending_review: Optional[Dict[str, Any]] = None
) -> AdvisorWorkspaceView:
    """Project internal graph state (and the review checkpoint it is paused at) into the 3-panel view."""
    # While paused, the review checkpoint defines the active stage.
    current_stage_key = pending_review["stage"] if pending_review else state.get("current_stage", "client_profile")
    outcome = state.get("outcome")
    rejected_at = state.get("rejected_at_stage")
    profile = state.get("profile")
    assessment = state.get("assessment")
    ideas = state.get("candidate_ideas")
    memo = state.get("memo")
    pdf_path = state.get("pdf_path")
    human_action = state.get("human_action") or {}

    client_id = profile.client_id if profile else state.get("raw_input", {}).get("client_id", "CL-PENDING")
    client_name = profile.name if profile else state.get("raw_input", {}).get("name", "New Client Advisory")

    # 1. Left Panel: Workflow Progress Stages
    stages: List[WorkflowStage] = []
    found_current = False

    stage_order = [s[0] for s in STAGE_CONFIG]
    if outcome == "APPROVED":
        current_idx = len(stage_order)
    elif outcome == "REJECTED" and rejected_at in stage_order:
        current_idx = stage_order.index(rejected_at)
    else:
        current_idx = stage_order.index(current_stage_key) if current_stage_key in stage_order else 0

    for idx, (s_key, s_name, s_desc) in enumerate(STAGE_CONFIG):
        if outcome == "REJECTED" and idx == current_idx:
            status = "FAILED"
        elif idx < current_idx:
            status = "COMPLETED"
        elif idx == current_idx:
            if state.get("requires_human_action"):
                status = "WAITING_HUMAN"
            else:
                status = "IN_PROGRESS"
        else:
            status = "PENDING"

        stages.append(WorkflowStage(
            stage_key=s_key, # type: ignore
            display_name=s_name,
            status=status, # type: ignore
            description=s_desc
        ))

    # 2. Center Panel: Decision Space with Allocation Diffs & Warnings
    allocation_diffs: List[AllocationDiffRow] = []
    risk_warnings = assessment.detected_breaches if assessment else []

    if profile and assessment:
        # Compare asset class weights before vs after
        before_holdings = assessment.before_metrics
        after_holdings = assessment.after_metrics

        # Generate asset class diff rows
        current_ac = profile.current_asset_class_weights
        for ac, b_wt in current_ac.items():
            # Estimate after weight or diff
            a_wt = b_wt
            if ideas:
                for idea in ideas.ideas:
                    if (ac == "Equities" and idea.portfolio_role in ["GROWTH", "STABILITY"]) or (ac == "Fixed Income" and idea.portfolio_role == "INCOME"):
                        a_wt += (idea.target_weight_pct * 0.5)
                    elif ac == "Cash Equivalents":
                        a_wt -= (idea.target_weight_pct * 0.5)

            delta = round(a_wt - b_wt, 2)
            allocation_diffs.append(AllocationDiffRow(
                category_type="Asset Class",
                name=ac,
                before_pct=round(b_wt, 1),
                after_pct=round(max(0.0, a_wt), 1),
                delta_pct=delta,
                is_breached=any(ac.lower() in b.message.lower() for b in risk_warnings)
            ))

        # Top single stock row
        allocation_diffs.append(AllocationDiffRow(
            category_type="Single Stock",
            name=after_holdings.max_single_stock_ticker,
            before_pct=round(before_holdings.max_single_stock_pct, 1),
            after_pct=round(after_holdings.max_single_stock_pct, 1),
            delta_pct=round(after_holdings.max_single_stock_pct - before_holdings.max_single_stock_pct, 1),
            is_breached=any("SINGLE_STOCK" in b.rule_id for b in risk_warnings)
        ))

    allowed_actions = list(pending_review.get("allowed_actions", [])) if pending_review else []
    allowed_actions.append("add_advisor_note")

    if pending_review:
        stage_summary = f"{pending_review['agent']} output for {client_name} is awaiting your review."
    elif outcome:
        stage_summary = f"Advisory session for {client_name} ended: {outcome}."
    else:
        stage_summary = (
            f"Reviewing recommendations for {client_name}. "
            f"Suitability Status: {assessment.overall_status if assessment else 'PENDING'}."
        )

    center_panel = DecisionSpacePayload(
        current_stage=current_stage_key,
        active_agent_name=AGENT_NAMES.get(current_stage_key, current_stage_key.replace('_', ' ').title()),
        stage_summary=stage_summary,
        allocation_diffs=allocation_diffs,
        risk_warnings=risk_warnings,
        requires_human_action=state.get("requires_human_action", False),
        interrupt_reason=state.get("error_message") or (pending_review["reason"] if pending_review else None),
        allowed_actions=allowed_actions,
        data_payload={
            "risk_tier": profile.calculated_risk_tier if profile else "N/A",
            "suitability_status": assessment.overall_status if assessment else "PENDING",
            "candidate_ideas": [i.model_dump() for i in ideas.ideas] if ideas else [],
            "critic_notes": assessment.critic_notes if assessment else "Pending review"
        }
    )

    # 3. Right Panel: Interactive Memo Preview with highlighting
    sections: List[MemoPreviewSection] = []
    active_highlights = []

    if memo:
        sections.append(MemoPreviewSection(
            section_id="sec_exec_summary",
            title="Page 1: Executive Recommendation",
            page_number=1,
            content_html_or_md=f"**Objective:** {memo.page_1_executive.investment_objective}\n\n**Client Snapshot:** {memo.page_1_executive.client_snapshot}",
            is_highlighted=(current_stage_key in ["client_profile", "market_intelligence"]),
            affected_by_stage="client_profile"
        ))

        glance_rows = "\n".join([
            f"- **{g.slot_number}. {g.ticker}** ({g.action} {g.target_weight_pct:.1f}%): {g.why_it_fits} [{g.portfolio_role}]"
            for g in memo.page_1_executive.ideas_at_a_glance
        ])
        sections.append(MemoPreviewSection(
            section_id="sec_glance_table",
            title="Page 1: Recommendations at a Glance",
            page_number=1,
            content_html_or_md=glance_rows,
            is_highlighted=(current_stage_key == "investment_ideas"),
            affected_by_stage="investment_ideas"
        ))

        for d in memo.page_2_details:
            sections.append(MemoPreviewSection(
                section_id=f"sec_idea_{d.slot_number}",
                title=f"Page 2: Idea #{d.slot_number} - {d.ticker} ({d.portfolio_role})",
                page_number=2,
                content_html_or_md=f"**Thesis:** {d.thesis}\n\n**Why Now:** {d.why_now}\n\n**Allocation:** {d.allocation_range_str}",
                is_highlighted=(current_stage_key in ["investment_ideas", "suitability_review"]),
                affected_by_stage="suitability_review"
            ))

        sections.append(MemoPreviewSection(
            section_id="sec_portfolio_view",
            title="Page 3: Portfolio View & Governance",
            page_number=3,
            content_html_or_md=f"**Impact:** {memo.page_3_portfolio_view.portfolio_level_impact_narrative}\n\n**Sign-off:** {memo.page_3_portfolio_view.advisor_signature_block}",
            is_highlighted=(current_stage_key in ["advisor_approval", "final_memo"]),
            affected_by_stage="final_memo"
        ))

        active_highlights = [s.section_id for s in sections if s.is_highlighted]

    right_panel = InteractiveMemoPreview(
        sections=sections,
        active_highlight_section_ids=active_highlights,
        pdf_download_url=f"/sessions/{session_id}/pdf" if pdf_path else None
    )

    return AdvisorWorkspaceView(
        session_id=session_id,
        client_id=client_id,
        client_name=client_name,
        workflow_stages=stages,
        center_panel=center_panel,
        right_panel=right_panel,
        advisor_notes=state.get("advisor_notes") or [],
        pending_review=PendingReview(**pending_review) if pending_review else None,
        outcome=outcome
    )
