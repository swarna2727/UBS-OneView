"""LangGraph orchestration layer for human-led multi-agent advisory system."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AdvisoryState
from src.graph.routing import (
    route_after_profile, route_after_critic,
    route_after_risk_exception, route_after_advisor_approval
)
from src.agents.client_profile import ClientProfileAgent
from src.agents.market_intelligence import MarketIntelligenceAgent
from src.agents.investment_idea import InvestmentIdeaAgent, InvestmentIdeaAgentInputs
from src.agents.suitability_critic import SuitabilityCriticAgent, SuitabilityCriticInputs
from src.agents.memo_formatter import MemoFormatterAgent, MemoFormatterInputs
from src.schemas.advisory import InvestmentIdea
from src.llm.base import LLMClientProtocol

def create_advisory_nodes(llm_client: Optional[LLMClientProtocol] = None):
    """Factory creating bound node functions for the advisory workflow."""
    profile_agent = ClientProfileAgent(llm_client)
    market_agent = MarketIntelligenceAgent(llm_client)
    idea_agent = InvestmentIdeaAgent(llm_client)
    critic_agent = SuitabilityCriticAgent(llm_client)
    memo_agent = MemoFormatterAgent(llm_client)

    def node_client_profile(state: AdvisoryState) -> Dict[str, Any]:
        raw_input = state.get("raw_input", {})
        profile = profile_agent.run(raw_input)
        audit = [{
            "stage": "client_profile",
            "timestamp": datetime.now().isoformat(),
            "flags_detected": len(profile.unresolved_flags)
        }]
        return {
            "profile": profile,
            "current_stage": "client_profile",
            "audit_trail": audit
        }

    def node_hitl_resolve_profile(state: AdvisoryState) -> Dict[str, Any]:
        """HITL interrupt checkpoint for input confirmation and contradiction resolution."""
        profile = state.get("profile")
        flags = profile.unresolved_flags if profile else []

        # Trigger interrupt for human resolution
        resume_val = interrupt({
            "stage": "client_profile",
            "reason": "Profile has unresolved flags or conflicts requiring advisor resolution.",
            "unresolved_flags": [f.model_dump() for f in flags],
            "allowed_actions": ["resolve_conflict", "accept", "override_flag"]
        })

        human_action = resume_val if isinstance(resume_val, dict) else {"action": str(resume_val)}

        # Update profile flags to resolved
        if profile:
            resolved_flags = []
            for f in profile.unresolved_flags:
                f_dict = f.model_dump()
                f_dict["is_resolved"] = True
                f_dict["resolution_notes"] = human_action.get("notes") or human_action.get("conflict_resolution") or "Resolved by RM"
                from src.schemas.advisory import ProfileFlag
                resolved_flags.append(ProfileFlag(**f_dict))
            profile_dict = profile.model_dump()
            profile_dict["unresolved_flags"] = resolved_flags
            from src.schemas.advisory import ClientAdvisoryProfile
            updated_profile = ClientAdvisoryProfile(**profile_dict)
        else:
            updated_profile = profile

        audit = [{
            "stage": "hitl_resolve_profile",
            "timestamp": datetime.now().isoformat(),
            "human_action": human_action
        }]

        return {
            "profile": updated_profile,
            "human_action": human_action,
            "audit_trail": audit
        }

    def node_market_intelligence(state: AdvisoryState) -> Dict[str, Any]:
        profile = state["profile"]
        brief = market_agent.run(profile)
        audit = [{
            "stage": "market_intelligence",
            "timestamp": datetime.now().isoformat(),
            "macro_themes": brief.macro_themes,
            "validated_sources_count": len(brief.validated_sources)
        }]
        return {
            "research_brief": brief,
            "current_stage": "market_intelligence",
            "audit_trail": audit
        }

    def node_investment_ideas(state: AdvisoryState) -> Dict[str, Any]:
        profile = state["profile"]
        brief = state["research_brief"]
        candidate_ideas = state.get("candidate_ideas")
        locked_ids = set(state.get("locked_idea_ids") or [])
        regenerate_slots = state.get("regenerate_slots") or []

        # Identify locked ideas
        locked_list: List[InvestmentIdea] = []
        if candidate_ideas and locked_ids:
            for idea in candidate_ideas.ideas:
                if idea.idea_id in locked_ids or (idea.slot_number not in regenerate_slots and regenerate_slots):
                    locked_list.append(idea)

        inputs = InvestmentIdeaAgentInputs(
            profile=profile,
            research_brief=brief,
            locked_ideas=locked_list,
            rejected_slots=regenerate_slots,
            feedback=state.get("idea_feedback")
        )

        new_idea_set = idea_agent.run(inputs)
        audit = [{
            "stage": "investment_ideas",
            "timestamp": datetime.now().isoformat(),
            "generation_round": new_idea_set.generation_round,
            "locked_slots_retained": [i.slot_number for i in locked_list],
            "regenerated_slots": regenerate_slots
        }]
        return {
            "candidate_ideas": new_idea_set,
            "current_stage": "investment_ideas",
            "regenerate_slots": [], # reset after generation
            "audit_trail": audit
        }

    def node_suitability_critic(state: AdvisoryState) -> Dict[str, Any]:
        profile = state["profile"]
        idea_set = state["candidate_ideas"]
        brief = state["research_brief"]

        inputs = SuitabilityCriticInputs(
            profile=profile,
            candidate_ideas=idea_set,
            research_brief=brief
        )

        assessment = critic_agent.run(inputs)
        audit = [{
            "stage": "suitability_critic",
            "timestamp": datetime.now().isoformat(),
            "overall_status": assessment.overall_status,
            "breaches_count": len(assessment.detected_breaches)
        }]
        return {
            "assessment": assessment,
            "current_stage": "suitability_review",
            "audit_trail": audit
        }

    def node_mark_failed_slots(state: AdvisoryState) -> Dict[str, Any]:
        """
        Selective regeneration preparation:
        Lock all ideas that PASSED or are not rejected.
        Mark failed or explicitly rejected slots for regeneration.
        Increment regeneration_count.
        """
        assessment = state.get("assessment")
        idea_set = state.get("candidate_ideas")
        human_action = state.get("human_action") or {}

        regen_slots = []
        locked_ids = []

        # Check if human explicitly rejected an idea by ID
        target_rejected_id = human_action.get("idea_id")

        if idea_set:
            for idea in idea_set.ideas:
                # Check critic evaluation
                critic_eval = next((ie for ie in assessment.idea_evaluations if ie.idea_id == idea.idea_id), None) if assessment else None

                is_failed = (critic_eval and critic_eval.status == "FAIL") or (target_rejected_id and idea.idea_id == target_rejected_id)

                if is_failed:
                    regen_slots.append(idea.slot_number)
                else:
                    locked_ids.append(idea.idea_id)

        # Fallback if no specific slot was marked as FAIL
        if not regen_slots:
            regen_slots = [2] # default to slot 2 for selective testing

        regen_count = state.get("regeneration_count", 0) + 1
        feedback = human_action.get("notes") or "Critic detected risk limit breach; re-align allocation and role."

        audit = [{
            "stage": "mark_failed_slots",
            "timestamp": datetime.now().isoformat(),
            "regeneration_round": regen_count,
            "marked_slots": regen_slots,
            "locked_ideas": locked_ids
        }]

        return {
            "locked_idea_ids": locked_ids,
            "regenerate_slots": regen_slots,
            "regeneration_count": regen_count,
            "idea_feedback": feedback,
            "audit_trail": audit
        }

    def node_hitl_risk_exception(state: AdvisoryState) -> Dict[str, Any]:
        """HITL interrupt checkpoint for soft risk breach / FLAG overrides or FAIL escalation."""
        assessment = state.get("assessment")
        breaches = assessment.detected_breaches if assessment else []

        resume_val = interrupt({
            "stage": "risk_review",
            "reason": "Suitability assessment raised FLAG (or reached max retries) requiring RM exception decision.",
            "breaches": [b.model_dump() for b in breaches],
            "allowed_actions": ["override_flag", "reject", "revise", "abort"]
        })

        human_action = resume_val if isinstance(resume_val, dict) else {"action": str(resume_val)}

        advisor_notes = []
        if human_action.get("notes"):
            advisor_notes.append(f"RM Exception Note: {human_action['notes']}")

        audit = [{
            "stage": "hitl_risk_exception",
            "timestamp": datetime.now().isoformat(),
            "human_action": human_action
        }]

        return {
            "human_action": human_action,
            "advisor_notes": advisor_notes,
            "audit_trail": audit
        }

    def node_memo_formatter(state: AdvisoryState) -> Dict[str, Any]:
        profile = state["profile"]
        ideas = state["candidate_ideas"]
        assessment = state["assessment"]
        session_id = state.get("session_id", "default_session")

        pdf_target = f"data/pipeline_data/{session_id}_final_memo.pdf"
        inputs = MemoFormatterInputs(
            profile=profile,
            ideas=ideas,
            assessment=assessment,
            output_pdf_path=pdf_target
        )

        memo = memo_agent.run(inputs)
        audit = [{
            "stage": "memo_formatter",
            "timestamp": datetime.now().isoformat(),
            "memo_id": memo.memo_id,
            "pdf_path": memo.pdf_artifact_path
        }]
        return {
            "memo": memo,
            "pdf_path": memo.pdf_artifact_path,
            "current_stage": "final_memo",
            "audit_trail": audit
        }

    def node_hitl_advisor_approval(state: AdvisoryState) -> Dict[str, Any]:
        """Final HITL approval of investment recommendation memo before PDF sign-off."""
        memo = state.get("memo")

        resume_val = interrupt({
            "stage": "advisor_approval",
            "reason": "Final recommendation memo ready for Relationship Manager approval.",
            "memo_id": memo.memo_id if memo else "N/A",
            "allowed_actions": ["accept", "reject", "revise", "add_advisor_note"]
        })

        human_action = resume_val if isinstance(resume_val, dict) else {"action": str(resume_val)}

        advisor_notes = []
        if human_action.get("notes"):
            advisor_notes.append(f"Advisor Sign-off Note: {human_action['notes']}")

        audit = [{
            "stage": "hitl_advisor_approval",
            "timestamp": datetime.now().isoformat(),
            "human_action": human_action
        }]

        return {
            "human_action": human_action,
            "advisor_notes": advisor_notes,
            "audit_trail": audit
        }

    def node_end_session(state: AdvisoryState) -> Dict[str, Any]:
        audit = [{
            "stage": "end_session",
            "timestamp": datetime.now().isoformat(),
            "status": "COMPLETED"
        }]
        return {
            "is_completed": True,
            "current_stage": "completed",
            "audit_trail": audit
        }

    return {
        "client_profile": node_client_profile,
        "hitl_resolve_profile": node_hitl_resolve_profile,
        "market_intelligence": node_market_intelligence,
        "investment_ideas": node_investment_ideas,
        "suitability_critic": node_suitability_critic,
        "mark_failed_slots": node_mark_failed_slots,
        "hitl_risk_exception": node_hitl_risk_exception,
        "memo_formatter": node_memo_formatter,
        "hitl_advisor_approval": node_hitl_advisor_approval,
        "end_session": node_end_session
    }

def build_advisory_graph(
    llm_client: Optional[LLMClientProtocol] = None,
    checkpointer: Optional[Any] = None
):
    """
    Construct the full LangGraph state machine with conditional routing edges,
    selective regeneration loops, and human-in-the-loop checkpoints.
    """
    nodes = create_advisory_nodes(llm_client)
    builder = StateGraph(AdvisoryState)

    # Add all nodes
    for name, node_func in nodes.items():
        builder.add_node(name, node_func)

    # 1. Entry point -> Client Profile Agent
    builder.add_edge(START, "client_profile")

    # 2. Conditional edge after Client Profile
    builder.add_conditional_edges(
        "client_profile",
        route_after_profile,
        {
            "hitl_resolve_profile": "hitl_resolve_profile",
            "market_intelligence": "market_intelligence"
        }
    )

    # 3. Resume from resolve profile -> Market Intelligence
    builder.add_edge("hitl_resolve_profile", "market_intelligence")

    # 4. Market Intelligence -> Investment Ideas
    builder.add_edge("market_intelligence", "investment_ideas")

    # 5. Investment Ideas -> Suitability Critic
    builder.add_edge("investment_ideas", "suitability_critic")

    # 6. Conditional edge after Suitability Critic (PASS, FLAG, FAIL)
    builder.add_conditional_edges(
        "suitability_critic",
        route_after_critic,
        {
            "memo_formatter": "memo_formatter",
            "hitl_risk_exception": "hitl_risk_exception",
            "mark_failed_slots": "mark_failed_slots"
        }
    )

    # 7. Selective regeneration loop: mark_failed_slots -> investment_ideas
    builder.add_edge("mark_failed_slots", "investment_ideas")

    # 8. Conditional edge from risk exception
    builder.add_conditional_edges(
        "hitl_risk_exception",
        route_after_risk_exception,
        {
            "memo_formatter": "memo_formatter",
            "mark_failed_slots": "mark_failed_slots",
            "end_session": "end_session"
        }
    )

    # 9. Memo Formatter -> Advisor Approval Checkpoint
    builder.add_edge("memo_formatter", "hitl_advisor_approval")

    # 10. Conditional edge after Advisor Approval
    builder.add_conditional_edges(
        "hitl_advisor_approval",
        route_after_advisor_approval,
        {
            "end_session": "end_session",
            "mark_failed_slots": "mark_failed_slots"
        }
    )

    # 11. End Session -> END
    builder.add_edge("end_session", END)

    saver = checkpointer or MemorySaver()
    return builder.compile(checkpointer=saver)
