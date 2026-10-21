"""LangGraph orchestration layer for human-led multi-agent advisory system."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AdvisoryState
from src.graph.routing import (
    REVIEW_PROFILE, REVIEW_MARKET, REVIEW_IDEAS, REVIEW_SUITABILITY, REVIEW_MEMO,
    EDITABLE_PROFILE_FIELDS, allowed_review_actions, resolve_review_outcome,
    route_after_profile_review, route_after_market_review, route_after_ideas_review,
    route_after_suitability_review, route_after_memo_review
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

    def _request_review(stage: str, agent_name: str, reason: str, output: Any, state: AdvisoryState, **extra: Any) -> Dict[str, Any]:
        """Pause the graph so the RM can inspect an agent's output and approve, regenerate or reject it."""
        resume_val = interrupt({
            "stage": stage,
            "agent": agent_name,
            "reason": reason,
            "output": output.model_dump() if output is not None else None,
            "allowed_actions": allowed_review_actions(stage, state),
            **extra
        })
        return resume_val if isinstance(resume_val, dict) else {"action": str(resume_val)}

    def _review_audit(stage: str, human_action: Dict[str, Any], outcome: str) -> List[Dict[str, Any]]:
        return [{
            "stage": f"review_{stage}",
            "timestamp": datetime.now().isoformat(),
            "human_action": human_action,
            "review_outcome": outcome
        }]

    def node_review_profile(state: AdvisoryState) -> Dict[str, Any]:
        """HITL checkpoint: RM reviews the rendered client profile, resolving any flags."""
        profile = state.get("profile")
        blocking = [f for f in profile.unresolved_flags if f.severity == "BLOCKING_INTERRUPT"] if profile else []
        reason = (
            "Profile has unresolved flags or conflicts requiring advisor resolution."
            if blocking else "Review the standardized client profile before market research begins."
        )
        human_action = _request_review(
            REVIEW_PROFILE, "Client Profile Agent", reason, profile, state,
            unresolved_flags=[f.model_dump() for f in profile.unresolved_flags] if profile else [],
            editable_fields=EDITABLE_PROFILE_FIELDS,
            raw_input=state.get("raw_input", {})
        )
        outcome = resolve_review_outcome(REVIEW_PROFILE, human_action)
        update: Dict[str, Any] = {"human_action": human_action, "review_outcome": outcome}

        if outcome == "approve" and profile:
            # Mark profile flags as resolved by the RM
            notes = human_action.get("notes") or human_action.get("conflict_resolution") or "Resolved by RM"
            resolved_flags = [f.model_copy(update={"is_resolved": True, "resolution_notes": notes}) for f in profile.unresolved_flags]
            update["profile"] = profile.model_copy(update={"unresolved_flags": resolved_flags})
        elif outcome == "regenerate":
            edits = {k: v for k, v in (human_action.get("edits") or {}).items() if k in EDITABLE_PROFILE_FIELDS}
            update["raw_input"] = {**state.get("raw_input", {}), **edits}
        elif outcome == "reject":
            update["rejected_at_stage"] = REVIEW_PROFILE

        update["audit_trail"] = _review_audit(REVIEW_PROFILE, human_action, outcome)
        return update

    def node_market_intelligence(state: AdvisoryState) -> Dict[str, Any]:
        profile = state["profile"]
        brief = market_agent.run(profile, feedback=state.get("market_feedback"))
        audit = [{
            "stage": "market_intelligence",
            "timestamp": datetime.now().isoformat(),
            "macro_themes": brief.macro_themes,
            "validated_sources_count": len(brief.validated_sources)
        }]
        return {
            "research_brief": brief,
            "current_stage": "market_intelligence",
            "market_feedback": None,
            "audit_trail": audit
        }

    def node_review_market(state: AdvisoryState) -> Dict[str, Any]:
        """HITL checkpoint: RM reviews the evidence-backed research brief."""
        human_action = _request_review(
            REVIEW_MARKET, "Market Intelligence Agent",
            "Review the research brief and its cited evidence before ideas are generated.",
            state.get("research_brief"), state
        )
        outcome = resolve_review_outcome(REVIEW_MARKET, human_action)
        update: Dict[str, Any] = {"human_action": human_action, "review_outcome": outcome}
        if outcome == "regenerate":
            update["market_feedback"] = human_action.get("notes")
        elif outcome == "reject":
            update["rejected_at_stage"] = REVIEW_MARKET
        update["audit_trail"] = _review_audit(REVIEW_MARKET, human_action, outcome)
        return update

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

    def node_review_ideas(state: AdvisoryState) -> Dict[str, Any]:
        """HITL checkpoint: RM reviews the three candidate ideas before the suitability critic runs."""
        idea_set = state.get("candidate_ideas")
        human_action = _request_review(
            REVIEW_IDEAS, "Investment Idea Agent",
            "Review the candidate ideas. Approve, regenerate a single idea, or regenerate the full set.",
            idea_set, state
        )
        outcome = resolve_review_outcome(REVIEW_IDEAS, human_action)
        update: Dict[str, Any] = {"human_action": human_action, "review_outcome": outcome}

        if outcome == "regenerate_ideas" and idea_set:
            target_id = human_action.get("idea_id")
            if target_id:
                regen_slots = [i.slot_number for i in idea_set.ideas if i.idea_id == target_id]
                locked_ids = [i.idea_id for i in idea_set.ideas if i.idea_id != target_id]
            else:
                regen_slots = [i.slot_number for i in idea_set.ideas]
                locked_ids = []
            update.update({
                "regenerate_slots": regen_slots,
                "locked_idea_ids": locked_ids,
                "regeneration_count": state.get("regeneration_count", 0) + 1,
                "idea_feedback": human_action.get("notes") or "RM requested regeneration."
            })
        elif outcome == "reject":
            update["rejected_at_stage"] = REVIEW_IDEAS

        update["audit_trail"] = _review_audit(REVIEW_IDEAS, human_action, outcome)
        return update

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

        # Fallback when nothing FAILED (e.g. RM asked to revise a FLAG set): regenerate every non-PASS slot,
        # or the whole set if all ideas passed.
        if not regen_slots and idea_set:
            regen_slots = [
                ie.slot_number for ie in (assessment.idea_evaluations if assessment else []) if ie.status != "PASS"
            ] or [i.slot_number for i in idea_set.ideas]
            locked_ids = [i.idea_id for i in idea_set.ideas if i.slot_number not in regen_slots]

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

    def node_review_suitability(state: AdvisoryState) -> Dict[str, Any]:
        """HITL checkpoint: RM reviews the critic's PASS / FLAG / FAIL assessment and any breaches."""
        assessment = state.get("assessment")
        status = assessment.overall_status if assessment else "FAIL"
        reason = {
            "PASS": "All suitability rules passed. Confirm the assessment to draft the advisory memo.",
            "FLAG": "Suitability assessment raised FLAG; an RM exception override is required to proceed.",
            "FAIL": "Suitability assessment found hard breaches; send failing ideas back for regeneration."
        }.get(status, "Review the suitability assessment.")
        human_action = _request_review(REVIEW_SUITABILITY, "Suitability Critic Agent", reason, assessment, state)
        outcome = resolve_review_outcome(REVIEW_SUITABILITY, human_action)

        advisor_notes = []
        if human_action.get("action") == "override_flag" and human_action.get("notes"):
            advisor_notes.append(f"RM Exception Note: {human_action['notes']}")

        update: Dict[str, Any] = {
            "human_action": human_action,
            "review_outcome": outcome,
            "advisor_notes": advisor_notes,
            "audit_trail": _review_audit(REVIEW_SUITABILITY, human_action, outcome)
        }
        if outcome == "reject":
            update["rejected_at_stage"] = REVIEW_SUITABILITY
        return update

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

    def node_review_memo(state: AdvisoryState) -> Dict[str, Any]:
        """Final HITL checkpoint: RM signs off on the advisory memo."""
        human_action = _request_review(
            REVIEW_MEMO, "Memo Formatter Agent",
            "Final recommendation memo ready for Relationship Manager approval.",
            state.get("memo"), state
        )
        outcome = resolve_review_outcome(REVIEW_MEMO, human_action)

        advisor_notes = []
        if human_action.get("notes"):
            advisor_notes.append(f"Advisor Sign-off Note: {human_action['notes']}")

        update: Dict[str, Any] = {
            "human_action": human_action,
            "review_outcome": outcome,
            "advisor_notes": advisor_notes,
            "audit_trail": _review_audit(REVIEW_MEMO, human_action, outcome)
        }
        if outcome == "reject":
            update["rejected_at_stage"] = REVIEW_MEMO
        return update

    def node_end_session(state: AdvisoryState) -> Dict[str, Any]:
        outcome = "REJECTED" if state.get("review_outcome") == "reject" else "APPROVED"
        audit = [{
            "stage": "end_session",
            "timestamp": datetime.now().isoformat(),
            "status": "COMPLETED",
            "outcome": outcome
        }]
        return {
            "is_completed": True,
            "outcome": outcome,
            "current_stage": "completed" if outcome == "APPROVED" else "rejected",
            "audit_trail": audit
        }

    return {
        "client_profile": node_client_profile,
        "review_profile": node_review_profile,
        "market_intelligence": node_market_intelligence,
        "review_market": node_review_market,
        "investment_ideas": node_investment_ideas,
        "review_ideas": node_review_ideas,
        "suitability_critic": node_suitability_critic,
        "review_suitability": node_review_suitability,
        "mark_failed_slots": node_mark_failed_slots,
        "memo_formatter": node_memo_formatter,
        "review_memo": node_review_memo,
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

    # Every agent is followed by an RM review checkpoint that can approve
    # (continue), regenerate (re-run the agent) or reject (end the session).
    builder.add_edge(START, "client_profile")
    builder.add_edge("client_profile", "review_profile")
    builder.add_conditional_edges(
        "review_profile",
        route_after_profile_review,
        {
            "market_intelligence": "market_intelligence",
            "client_profile": "client_profile",
            "end_session": "end_session"
        }
    )

    builder.add_edge("market_intelligence", "review_market")
    builder.add_conditional_edges(
        "review_market",
        route_after_market_review,
        {
            "investment_ideas": "investment_ideas",
            "market_intelligence": "market_intelligence",
            "end_session": "end_session"
        }
    )

    builder.add_edge("investment_ideas", "review_ideas")
    builder.add_conditional_edges(
        "review_ideas",
        route_after_ideas_review,
        {
            "suitability_critic": "suitability_critic",
            "investment_ideas": "investment_ideas",
            "end_session": "end_session"
        }
    )

    builder.add_edge("suitability_critic", "review_suitability")
    builder.add_conditional_edges(
        "review_suitability",
        route_after_suitability_review,
        {
            "memo_formatter": "memo_formatter",
            "suitability_critic": "suitability_critic",
            "mark_failed_slots": "mark_failed_slots",
            "end_session": "end_session"
        }
    )

    # Selective regeneration loop: mark_failed_slots -> investment_ideas
    builder.add_edge("mark_failed_slots", "investment_ideas")

    builder.add_edge("memo_formatter", "review_memo")
    builder.add_conditional_edges(
        "review_memo",
        route_after_memo_review,
        {
            "end_session": "end_session",
            "memo_formatter": "memo_formatter",
            "mark_failed_slots": "mark_failed_slots"
        }
    )

    builder.add_edge("end_session", END)

    saver = checkpointer or MemorySaver()
    return builder.compile(checkpointer=saver)
