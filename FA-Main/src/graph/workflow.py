"""LangGraph orchestration layer for the human-led multi-agent advisory system.

One graph of five shared agents; the use case (pre-meeting review, ad-hoc request, market event) is set
when the session starts and the routers in routing.py send each one down its own path."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from src.graph.state import AdvisoryState
from src.graph.routing import (
    REVIEW_PROFILE, REVIEW_MARKET, REVIEW_RECOMMENDATIONS, REVIEW_MEMO, REVIEW_EVENT, REVIEW_BRIEF,
    PRE_MEETING, MARKET_EVENT,
    EDITABLE_PROFILE_FIELDS, MAX_REGENERATIONS_PER_IDEA, use_case,
    allowed_review_actions, resolve_review_outcome, failed_slots, is_escalated,
    route_after_client_profile, route_after_profile_review, route_after_market_intelligence,
    route_after_market_review, route_after_critic, route_after_event_review, route_after_brief_review,
    route_after_recommendation_review, route_after_memo_review
)
from src.agents.client_profile import ClientProfileAgent
from src.agents.market_intelligence import MarketIntelligenceAgent
from src.agents.investment_idea import InvestmentIdeaAgent, InvestmentIdeaAgentInputs
from src.agents.suitability_critic import SuitabilityCriticAgent, SuitabilityCriticInputs
from src.agents.memo_formatter import MemoFormatterAgent, MemoFormatterInputs, MeetingBriefInputs, DecisionRecordInputs
from src.audit.run_log import RunLog
from src.schemas.advisory import InvestmentIdea, CandidateIdeaSet
from src.tools.event_tools import market_event_impact
from src.tools.memo_tools import pdf_memo_renderer, meeting_brief_renderer, decision_record_renderer
from src.tools.pdf_charts import portfolio_change_charts, profile_charts, event_charts
from src.tools.profile_tools import profile_change_detector, risk_profile_calculator
from src.llm.base import LLMClientProtocol

PDF_DIR = "data/pipeline_data"

def checkpoint_serializer() -> JsonPlusSerializer:
    """Allow-list the advisory Pydantic models that are stored in graph checkpoints."""
    from src.schemas import advisory, tools
    allowed = [
        (module.__name__, name)
        for module in (advisory, tools)
        for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)

def create_advisory_nodes(llm_client: Optional[LLMClientProtocol] = None, run_log: Optional[RunLog] = None):
    """Factory creating bound node functions for the advisory workflow."""
    profile_agent = ClientProfileAgent(llm_client)
    market_agent = MarketIntelligenceAgent(llm_client)
    idea_agent = InvestmentIdeaAgent(llm_client)
    critic_agent = SuitabilityCriticAgent(llm_client)
    memo_agent = MemoFormatterAgent(llm_client)
    log = run_log or RunLog()

    def _agent_audit(state: AdvisoryState, stage: str, agent, output, **extra: Any) -> List[Dict[str, Any]]:
        """Record an agent's output with its model and prompt version (AS-8)."""
        log.record(state.get("session_id", "default_session"), "agent_output", stage=stage, **agent.last_run, output=output, **extra)
        return [{"stage": stage, "timestamp": datetime.now().isoformat(), **agent.last_run, **extra}]

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

    def _review_audit(state: AdvisoryState, stage: str, human_action: Dict[str, Any], outcome: str) -> List[Dict[str, Any]]:
        log.record(state.get("session_id", "default_session"), "advisor_decision", stage=stage,
                   decision=human_action, review_outcome=outcome)
        return [{
            "stage": f"review_{stage}",
            "timestamp": datetime.now().isoformat(),
            "human_action": human_action,
            "review_outcome": outcome
        }]

    def _regenerate_slots(state: AdvisoryState, human_action: Dict[str, Any]) -> Dict[str, Any]:
        """Advisor-requested regeneration: one idea (others locked) or the whole set. Resets R10 counters for those slots."""
        idea_set = state.get("candidate_ideas")
        target_id = human_action.get("idea_id")
        ideas = idea_set.ideas if idea_set else []
        slots = [i.slot_number for i in ideas if i.idea_id == target_id] if target_id else [i.slot_number for i in ideas]
        counts = dict(state.get("slot_regen_counts") or {})
        for slot in slots:
            counts[slot] = 0
        return {
            "regenerate_slots": slots,
            "locked_idea_ids": [i.idea_id for i in ideas if i.slot_number not in slots],
            "slot_regen_counts": counts,
            "regeneration_count": state.get("regeneration_count", 0) + 1,
            "idea_feedback": human_action.get("notes") or "Advisor requested a different idea for this slot."
        }

    def _research_focus(state: AdvisoryState) -> Optional[str]:
        raw = state.get("raw_input", {})
        if use_case(state) == PRE_MEETING:
            since = (raw.get("last_review") or {}).get("date", "the last review")
            return (f"Quarterly client review: market developments since {since} (roughly the last quarter) that matter "
                    "for this client's holdings and goals.")
        if use_case(state) == MARKET_EVENT:
            event = raw.get("market_event") or {}
            return (f"MARKET EVENT: {event.get('title', 'Market development')}: {event.get('description', '')} "
                    "Explain what changed, which holdings are exposed and whether the investment thesis is affected.")
        capital = raw.get("new_capital_amount")
        return f"Ad-hoc request: the client wants to deploy {float(capital):,.0f} of new capital." if capital else None

    def _idea_context(state: AdvisoryState) -> str:
        if use_case(state) == PRE_MEETING:
            changes = state.get("profile_changes") or []
            return "Profile changes since the last review: " + ("; ".join(
                f"{c['label']}: {c['before']} -> {c['after']}" for c in changes) or "none (advisor requested ideas)")
        if use_case(state) == MARKET_EVENT:
            impact, decision = state.get("event_impact"), state.get("event_decision") or {}
            return (f"Market event: {impact.event_title if impact else ''}. {impact.rationale if impact else ''} "
                    f"Advisor decided to revise the advice: {decision.get('rationale', '')}")
        capital = state.get("raw_input", {}).get("new_capital_amount")
        return f"Client request: deploy {float(capital):,.0f} of new capital." if capital else "Client request for investment ideas."

    def _apply_allocation_edit(state: AdvisoryState, human_action: Dict[str, Any]) -> Dict[str, Any]:
        """AS-6: an advisor-edited allocation is applied, then the critic re-validates it before any memo."""
        idea_set: CandidateIdeaSet = state["candidate_ideas"]
        new_weight = float((human_action.get("edits") or {})["target_weight_pct"])
        updated: List[InvestmentIdea] = []
        for idea in idea_set.ideas:
            if idea.idea_id == human_action.get("idea_id"):
                rng = idea.allocation_range
                idea = idea.model_copy(update={
                    "target_weight_pct": new_weight,
                    "allocation_range": rng.model_copy(update={
                        "min_pct": min(rng.min_pct, new_weight), "max_pct": max(rng.max_pct, new_weight), "target_pct": new_weight
                    })
                })
            updated.append(idea)
        return {"candidate_ideas": idea_set.model_copy(update={"ideas": updated}), "allocation_edited": True}

    # ------------------------------------------------------------------
    # Agent 1: Client Profile + review
    # ------------------------------------------------------------------

    def node_client_profile(state: AdvisoryState) -> Dict[str, Any]:
        raw = state.get("raw_input", {})
        profile = profile_agent.run(raw)
        changes = profile_change_detector(raw, raw.get("last_review") or {}) if use_case(state) == PRE_MEETING else []
        return {
            "profile": profile,
            "profile_changes": changes,
            "current_stage": "client_profile",
            "audit_trail": _agent_audit(state, "client_profile", profile_agent, profile,
                                        flags_detected=len(profile.unresolved_flags))
        }

    def node_review_profile(state: AdvisoryState) -> Dict[str, Any]:
        """Advisor confirms inputs and resolves missing or contradictory information (input confirmation / conflict resolution)."""
        profile = state.get("profile")
        flagged = bool(profile and profile.unresolved_flags)
        reason = (
            "Profile has unresolved flags or conflicts requiring advisor resolution."
            if flagged else "Confirm the standardized client profile before market research begins."
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
            notes = human_action.get("notes") or human_action.get("conflict_resolution") or "Resolved by RM"
            resolved_flags = [f.model_copy(update={"is_resolved": True, "resolution_notes": notes}) for f in profile.unresolved_flags]
            update["profile"] = profile.model_copy(update={"unresolved_flags": resolved_flags})
        elif outcome == "regenerate":
            edits = {k: v for k, v in (human_action.get("edits") or {}).items() if k in EDITABLE_PROFILE_FIELDS}
            update["raw_input"] = {**state.get("raw_input", {}), **edits}
        elif outcome == "reject":
            update["rejected_at_stage"] = REVIEW_PROFILE

        update["audit_trail"] = _review_audit(state, REVIEW_PROFILE, human_action, outcome)
        return update

    # ------------------------------------------------------------------
    # Agent 2: Market Intelligence + review
    # ------------------------------------------------------------------

    def node_market_intelligence(state: AdvisoryState) -> Dict[str, Any]:
        brief = market_agent.run(state["profile"], feedback=state.get("market_feedback"), focus=_research_focus(state))
        return {
            "research_brief": brief,
            "current_stage": "market_intelligence",
            "market_feedback": None,
            "audit_trail": _agent_audit(state, "market_intelligence", market_agent, brief,
                                        validated_sources_count=len(brief.validated_sources))
        }

    def node_review_market(state: AdvisoryState) -> Dict[str, Any]:
        """Advisor inspects the underlying sources and can challenge the relevance of retrieved research."""
        human_action = _request_review(
            REVIEW_MARKET, "Market Intelligence Agent",
            "Review the research brief and its cited evidence before ideas are generated.",
            state.get("research_brief"), state
        )
        outcome = resolve_review_outcome(REVIEW_MARKET, human_action)
        update: Dict[str, Any] = {"human_action": human_action, "review_outcome": outcome}
        if outcome == "approve" and use_case(state) == PRE_MEETING:
            # Default: new recommendations only when the client's profile changed; the advisor can override.
            default = bool(state.get("profile_changes"))
            update["include_ideas"] = bool((human_action.get("edits") or {}).get("include_ideas", default))
        if outcome == "regenerate":
            update["market_feedback"] = human_action.get("notes")
        elif outcome == "reject":
            update["rejected_at_stage"] = REVIEW_MARKET
        update["audit_trail"] = _review_audit(state, REVIEW_MARKET, human_action, outcome)
        return update

    # ------------------------------------------------------------------
    # Agents 3 + 4: Investment Ideas -> Suitability Critic (auto-regenerate FAILs) -> review
    # ------------------------------------------------------------------

    def node_investment_ideas(state: AdvisoryState) -> Dict[str, Any]:
        candidate_ideas = state.get("candidate_ideas")
        locked_ids = set(state.get("locked_idea_ids") or [])
        regenerate_slots = state.get("regenerate_slots") or []

        locked_list: List[InvestmentIdea] = []
        if candidate_ideas and regenerate_slots:
            locked_list = [i for i in candidate_ideas.ideas if i.idea_id in locked_ids and i.slot_number not in regenerate_slots]

        new_idea_set = idea_agent.run(InvestmentIdeaAgentInputs(
            profile=state["profile"],
            research_brief=state["research_brief"],
            locked_ideas=locked_list,
            rejected_slots=regenerate_slots,
            feedback=state.get("idea_feedback"),
            use_case=use_case(state),
            use_case_context=_idea_context(state)
        ))
        feedback = state.get("idea_feedback") or ""
        if not regenerate_slots:
            trigger = {"source": "initial", "slots": [1, 2, 3]}
        elif feedback.startswith("Suitability critic rejected"):
            trigger = {"source": "auto_regeneration", "slots": sorted(regenerate_slots)}
        else:
            trigger = {"source": "advisor_revision", "slots": sorted(regenerate_slots), "notes": feedback}
        return {
            "candidate_ideas": new_idea_set,
            "current_stage": "investment_ideas",
            "regenerate_slots": [],
            "idea_round_trigger": trigger,
            "approved_idea_set": None,
            "audit_trail": _agent_audit(state, "investment_ideas", idea_agent, new_idea_set,
                                        locked_slots_retained=[i.slot_number for i in locked_list],
                                        regenerated_slots=regenerate_slots)
        }

    def node_suitability_critic(state: AdvisoryState) -> Dict[str, Any]:
        assessment = critic_agent.run(SuitabilityCriticInputs(
            profile=state["profile"],
            candidate_ideas=state["candidate_ideas"],
            research_brief=state["research_brief"]
        ))
        audit = _agent_audit(state, "suitability_critic", critic_agent, assessment, overall_status=assessment.overall_status)
        log.record(state.get("session_id", "default_session"), "rule_results",
                   portfolio=assessment.rule_results,
                   ideas={ie.idea_id: ie.rule_results for ie in assessment.idea_evaluations})
        trigger = (
            {"source": "allocation_edit", "slots": []} if state.get("allocation_edited")
            else state.get("idea_round_trigger") or {"source": "initial", "slots": [1, 2, 3]}
        )
        idea_round = {
            "round": len(state.get("idea_rounds") or []) + 1,
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger,
            "ideas": state["candidate_ideas"].model_dump(mode="json")["ideas"],
            "statuses": {ie.idea_id: ie.status for ie in assessment.idea_evaluations},
            "overall_status": assessment.overall_status,
            "risk_score_after": assessment.after_metrics.after_risk_score
        }
        return {
            "assessment": assessment,
            "idea_rounds": [idea_round],
            "approved_idea_set": None,
            "current_stage": "suitability_review",
            # An advisor's own allocation edit is re-validated and returned to the advisor, never auto-regenerated.
            "auto_regenerate": not state.get("allocation_edited", False),
            "allocation_edited": False,
            "audit_trail": audit
        }

    def node_mark_failed_slots(state: AdvisoryState) -> Dict[str, Any]:
        """Automatic selective regeneration: lock PASS/FLAG ideas, regenerate FAIL ideas, feed back the rule messages."""
        assessment = state["assessment"]
        slots = failed_slots(state)
        counts = dict(state.get("slot_regen_counts") or {})
        for slot in slots:
            counts[slot] = counts.get(slot, 0) + 1
        reasons = [
            f"Slot {ie.slot_number} ({ie.ticker}) failed: {ie.qualitative_notes}"
            for ie in assessment.idea_evaluations if ie.slot_number in slots
        ]
        locked = [ie.idea_id for ie in assessment.idea_evaluations if ie.slot_number not in slots]
        regen_count = state.get("regeneration_count", 0) + 1
        log.record(state.get("session_id", "default_session"), "auto_regeneration",
                   slots=slots, attempt_by_slot=counts, reasons=reasons)
        history = [{
            "slot": ie.slot_number, "ticker": ie.ticker, "attempt": counts[ie.slot_number],
            "rules": sorted({r.rule_id for r in ie.rule_results if r.outcome in ("FAIL", "BLOCK")}),
            "reason": ie.qualitative_notes
        } for ie in assessment.idea_evaluations if ie.slot_number in slots]
        return {
            "locked_idea_ids": locked,
            "regenerate_slots": slots,
            "slot_regen_counts": counts,
            "regeneration_count": regen_count,
            "idea_feedback": "Suitability critic rejected these ideas: " + " ".join(reasons),
            "regeneration_history": history,
            "audit_trail": [{
                "stage": "mark_failed_slots", "timestamp": datetime.now().isoformat(),
                "regeneration_round": regen_count, "marked_slots": slots, "locked_ideas": locked
            }]
        }

    def node_review_recommendations(state: AdvisoryState) -> Dict[str, Any]:
        """Recommendation review / risk exception: advisor accepts, revises, edits or rejects ideas."""
        assessment = state.get("assessment")
        status = assessment.overall_status if assessment else "FAIL"
        escalated = is_escalated(state)
        if escalated:
            reason = (f"R10: an idea still fails after {MAX_REGENERATIONS_PER_IDEA} automatic regenerations. "
                      f"Revise it with guidance, edit its allocation, or reject it.")
        else:
            reason = {
                "PASS": "All ideas passed suitability. Accept to draft the memo, or revise individual ideas.",
                "FLAG": "Judgment case: document an exception to proceed, or request a revision.",
                "FAIL": "Hard breach: failing ideas cannot reach the memo. Revise, edit or reject them."
            }[status]
        human_action = _request_review(
            REVIEW_RECOMMENDATIONS, "Suitability Critic Agent", reason, assessment, state,
            candidate_ideas=state["candidate_ideas"].model_dump()["ideas"] if state.get("candidate_ideas") else [],
            escalated=escalated,
            regeneration_history=state.get("regeneration_history") or []
        )
        outcome = resolve_review_outcome(REVIEW_RECOMMENDATIONS, human_action)
        update: Dict[str, Any] = {"human_action": human_action, "review_outcome": outcome}
        if human_action.get("action") == "override_flag" and human_action.get("notes"):
            update["advisor_notes"] = [f"RM Exception Note: {human_action['notes']}"]
        if outcome == "approve":
            update["approved_idea_set"] = {
                "timestamp": datetime.now().isoformat(),
                "decision": human_action.get("action"),
                "notes": human_action.get("notes"),
                "overall_status": status,
                "ideas": state["candidate_ideas"].model_dump(mode="json")["ideas"]
            }
        if outcome == "regenerate_ideas":
            update.update(_regenerate_slots(state, human_action))
        elif outcome == "revalidate":
            update.update(_apply_allocation_edit(state, human_action))
        elif outcome == "reject":
            update["rejected_at_stage"] = REVIEW_RECOMMENDATIONS
        update["audit_trail"] = _review_audit(state, REVIEW_RECOMMENDATIONS, human_action, outcome)
        return update

    # ------------------------------------------------------------------
    # UC3: client impact of a market event + advisor decision
    # ------------------------------------------------------------------

    def node_event_impact(state: AdvisoryState) -> Dict[str, Any]:
        """Deterministic re-check: which holdings move and whether the drifted portfolio still passes the rules."""
        event = state.get("raw_input", {}).get("market_event") or {}
        impact = market_event_impact(state["profile"], event.get("key", "rates_up"), event.get("description", ""))
        log.record(state.get("session_id", "default_session"), "event_impact", impact=impact)
        return {
            "event_impact": impact,
            "current_stage": REVIEW_EVENT,
            "audit_trail": [{"stage": "event_impact", "timestamp": datetime.now().isoformat(),
                             "suggested_decision": impact.suggested_decision, "portfolio_move_pct": impact.portfolio_move_pct}]
        }

    def node_review_event(state: AdvisoryState) -> Dict[str, Any]:
        """The advisor decides: maintain, monitor (with a follow-up trigger) or revise the advice."""
        impact = state.get("event_impact")
        human_action = _request_review(
            REVIEW_EVENT, "Market Intelligence Agent + Suitability Critic",
            f"Review the event evidence and client impact, then decide. {impact.rationale if impact else ''}",
            impact, state
        )
        outcome = resolve_review_outcome(REVIEW_EVENT, human_action)
        decision = {"maintain": "MAINTAIN", "monitor": "MONITOR", "revise": "REVISE"}.get(human_action.get("action"), "MONITOR")
        update: Dict[str, Any] = {
            "human_action": human_action, "review_outcome": outcome,
            "event_decision": {"decision": decision, "rationale": human_action.get("notes") or "",
                               "follow_up": (human_action.get("edits") or {}).get("follow_up", ""),
                               "timestamp": datetime.now().isoformat()},
            "advisor_notes": [f"Market-event decision ({decision}): {human_action.get('notes') or ''}"],
        }
        if outcome == "new_ideas":
            update.update({"candidate_ideas": None, "locked_idea_ids": [], "regenerate_slots": [],
                           "idea_feedback": f"Market event revision: {human_action.get('notes') or ''}"})
        update["audit_trail"] = _review_audit(state, REVIEW_EVENT, human_action, outcome)
        return update

    def node_decision_record(state: AdvisoryState) -> Dict[str, Any]:
        """Maintain / monitor: document the decision (no trades) and render the record PDF."""
        session_id = state.get("session_id", "default_session")
        decision = state.get("event_decision") or {}
        record = memo_agent.run(DecisionRecordInputs(
            profile=state["profile"], impact=state["event_impact"], decision=decision.get("decision", "MONITOR"),
            rationale=decision.get("rationale", ""), follow_up=decision.get("follow_up", ""),
            research_brief=state.get("research_brief")))
        record = record.model_copy(update={"advisor_signature_block":
                                           f"Relationship Manager on {datetime.now().strftime('%d-%b-%Y %H:%M')}."})
        pdf_path = decision_record_renderer(record, state["event_impact"], f"{PDF_DIR}/{session_id}_decision_record.pdf",
                                            charts=event_charts(state["event_impact"]))
        record = record.model_copy(update={"pdf_artifact_path": pdf_path})
        log.record(session_id, "pdf_generated", record_id=record.record_id, pdf_path=pdf_path)
        return {
            "decision_record": record, "pdf_path": pdf_path,
            "audit_trail": _agent_audit(state, "decision_record", memo_agent, record, pdf_path=pdf_path)
        }

    # ------------------------------------------------------------------
    # UC1: meeting brief + approval -> PDF
    # ------------------------------------------------------------------

    def node_meeting_brief(state: AdvisoryState) -> Dict[str, Any]:
        raw = state.get("raw_input", {})
        since = (raw.get("last_review") or {}).get("date")
        approved = bool(state.get("approved_idea_set"))
        brief = memo_agent.run(MeetingBriefInputs(
            profile=state["profile"], research_brief=state["research_brief"],
            meeting_purpose=(raw.get("meeting") or {}).get("purpose", "Quarterly portfolio review"),
            review_period=f"since last review on {since}" if since else "first review on record",
            changes=state.get("profile_changes") or [],
            ideas=state.get("candidate_ideas") if approved else None,
            assessment=state.get("assessment") if approved else None))
        return {
            "meeting_brief": brief, "pdf_path": None, "current_stage": REVIEW_BRIEF,
            "audit_trail": _agent_audit(state, "meeting_brief", memo_agent, brief, brief_id=brief.brief_id)
        }

    def node_review_brief(state: AdvisoryState) -> Dict[str, Any]:
        """CP4 (UC1): the advisor approves what is used in the meeting. No PDF before approval."""
        human_action = _request_review(
            REVIEW_BRIEF, "Memo Formatter Agent",
            "Meeting brief ready. Approve the insights to discuss with the client; the PDF is generated after approval.",
            state.get("meeting_brief"), state
        )
        outcome = resolve_review_outcome(REVIEW_BRIEF, human_action)
        update: Dict[str, Any] = {"human_action": human_action, "review_outcome": outcome}
        if human_action.get("notes"):
            update["advisor_notes"] = [f"Meeting brief note: {human_action['notes']}"]
        if outcome == "reject":
            update["rejected_at_stage"] = REVIEW_BRIEF
        update["audit_trail"] = _review_audit(state, REVIEW_BRIEF, human_action, outcome)
        return update

    def node_generate_brief_pdf(state: AdvisoryState) -> Dict[str, Any]:
        session_id = state.get("session_id", "default_session")
        notes = (state.get("human_action") or {}).get("notes")
        brief = state["meeting_brief"].model_copy(update={
            "advisor_approved": True,
            "advisor_signature_block": f"Approved by Relationship Manager on {datetime.now().strftime('%d-%b-%Y %H:%M')}."
                                       + (f" Note: {notes}" if notes else "")})
        calc = risk_profile_calculator(state.get("raw_input", {}))
        charts = profile_charts(state["profile"], calc.willingness_score, calc.capacity_score)
        if brief.recommendations and state.get("candidate_ideas"):
            charts += portfolio_change_charts(state["profile"], state["candidate_ideas"].ideas)[:2]
        pdf_path = meeting_brief_renderer(brief, f"{PDF_DIR}/{session_id}_meeting_brief.pdf", charts=charts)
        brief = brief.model_copy(update={"pdf_artifact_path": pdf_path})
        log.record(session_id, "pdf_generated", brief_id=brief.brief_id, pdf_path=pdf_path)
        return {"meeting_brief": brief, "pdf_path": pdf_path,
                "audit_trail": [{"stage": "generate_brief_pdf", "timestamp": datetime.now().isoformat(), "pdf_path": pdf_path}]}

    # ------------------------------------------------------------------
    # Agent 5: Memo + final approval -> PDF
    # ------------------------------------------------------------------

    def node_memo_formatter(state: AdvisoryState) -> Dict[str, Any]:
        memo = memo_agent.run(MemoFormatterInputs(
            profile=state["profile"],
            ideas=state["candidate_ideas"],
            assessment=state["assessment"],
            research_brief=state.get("research_brief"),
            advisor_notes=state.get("advisor_notes") or []
        ))
        return {
            "memo": memo,
            "pdf_path": None,
            "current_stage": "final_memo",
            "audit_trail": _agent_audit(state, "memo_formatter", memo_agent, memo, memo_id=memo.memo_id)
        }

    def node_review_memo(state: AdvisoryState) -> Dict[str, Any]:
        """Final sign-off: no client-facing memo or PDF without advisor approval."""
        human_action = _request_review(
            REVIEW_MEMO, "Memo Formatter Agent",
            "Final recommendation memo ready for Relationship Manager approval. The PDF is generated only after approval.",
            state.get("memo"), state
        )
        outcome = resolve_review_outcome(REVIEW_MEMO, human_action)
        update: Dict[str, Any] = {"human_action": human_action, "review_outcome": outcome}
        if human_action.get("notes"):
            update["advisor_notes"] = [f"Advisor Sign-off Note: {human_action['notes']}"]
        if outcome == "regenerate_ideas":
            update.update(_regenerate_slots(state, human_action))
        elif outcome == "revalidate":
            update.update(_apply_allocation_edit(state, human_action))
        elif outcome == "reject":
            update["rejected_at_stage"] = REVIEW_MEMO
        update["audit_trail"] = _review_audit(state, REVIEW_MEMO, human_action, outcome)
        return update

    def node_generate_pdf(state: AdvisoryState) -> Dict[str, Any]:
        session_id = state.get("session_id", "default_session")
        memo = state["memo"]
        notes = (state.get("human_action") or {}).get("notes")
        exceptions = [n for n in state.get("advisor_notes") or [] if n.startswith("RM Exception Note:")]
        signature = (
            f"Approved by Relationship Manager on {datetime.now().strftime('%d-%b-%Y %H:%M')}."
            + (f" Note: {notes}" if notes else "")
            + f" Suitability review: {state['assessment'].overall_status}."
            + (" " + " ".join(exceptions) if exceptions else "")
        )
        approved = memo.model_copy(update={
            "advisor_approved": True,
            "page_3_portfolio_view": memo.page_3_portfolio_view.model_copy(update={"advisor_signature_block": signature})
        })
        pdf_path = pdf_memo_renderer(approved, f"{PDF_DIR}/{session_id}_final_memo.pdf",
                                     charts=portfolio_change_charts(state["profile"], state["candidate_ideas"].ideas))
        approved = approved.model_copy(update={"pdf_artifact_path": pdf_path})
        log.record(session_id, "pdf_generated", memo_id=approved.memo_id, pdf_path=pdf_path)
        return {
            "memo": approved,
            "pdf_path": pdf_path,
            "audit_trail": [{"stage": "generate_pdf", "timestamp": datetime.now().isoformat(), "pdf_path": pdf_path}]
        }

    def node_end_session(state: AdvisoryState) -> Dict[str, Any]:
        outcome = "REJECTED" if state.get("review_outcome") == "reject" else "APPROVED"
        log.record(state.get("session_id", "default_session"), "run_ended", outcome=outcome,
                   rejected_at_stage=state.get("rejected_at_stage"))
        return {
            "is_completed": True,
            "outcome": outcome,
            "current_stage": "completed" if outcome == "APPROVED" else "rejected",
            "audit_trail": [{"stage": "end_session", "timestamp": datetime.now().isoformat(), "status": "COMPLETED", "outcome": outcome}]
        }

    return {
        "client_profile": node_client_profile,
        "review_profile": node_review_profile,
        "market_intelligence": node_market_intelligence,
        "review_market": node_review_market,
        "investment_ideas": node_investment_ideas,
        "suitability_critic": node_suitability_critic,
        "mark_failed_slots": node_mark_failed_slots,
        "review_recommendations": node_review_recommendations,
        "event_impact": node_event_impact,
        "review_event": node_review_event,
        "decision_record": node_decision_record,
        "meeting_brief": node_meeting_brief,
        "review_brief": node_review_brief,
        "generate_brief_pdf": node_generate_brief_pdf,
        "memo_formatter": node_memo_formatter,
        "review_memo": node_review_memo,
        "generate_pdf": node_generate_pdf,
        "end_session": node_end_session
    }

def build_advisory_graph(
    llm_client: Optional[LLMClientProtocol] = None,
    checkpointer: Optional[Any] = None,
    run_log: Optional[RunLog] = None
):
    """
    Construct the LangGraph state machine: advisor review after the profile, the research
    brief, the validated recommendations and the memo; automatic selective regeneration of
    FAILed ideas (max R10 per idea) between the idea agent and the critic.
    """
    nodes = create_advisory_nodes(llm_client, run_log)
    builder = StateGraph(AdvisoryState)
    for name, node_func in nodes.items():
        builder.add_node(name, node_func)

    builder.add_edge(START, "client_profile")
    builder.add_conditional_edges("client_profile", route_after_client_profile, {
        "review_profile": "review_profile",
        "market_intelligence": "market_intelligence"
    })
    builder.add_conditional_edges("review_profile", route_after_profile_review, {
        "market_intelligence": "market_intelligence",
        "client_profile": "client_profile",
        "end_session": "end_session"
    })

    builder.add_conditional_edges("market_intelligence", route_after_market_intelligence, {
        "review_market": "review_market",
        "event_impact": "event_impact"
    })
    builder.add_conditional_edges("review_market", route_after_market_review, {
        "investment_ideas": "investment_ideas",
        "meeting_brief": "meeting_brief",
        "market_intelligence": "market_intelligence",
        "end_session": "end_session"
    })

    # UC3: client impact -> advisor decision -> decision record, or new ideas when the advice is revised.
    builder.add_edge("event_impact", "review_event")
    builder.add_conditional_edges("review_event", route_after_event_review, {
        "decision_record": "decision_record",
        "investment_ideas": "investment_ideas"
    })
    builder.add_edge("decision_record", "end_session")

    # UC1: meeting brief -> approval -> PDF.
    builder.add_edge("meeting_brief", "review_brief")
    builder.add_conditional_edges("review_brief", route_after_brief_review, {
        "generate_brief_pdf": "generate_brief_pdf",
        "meeting_brief": "meeting_brief",
        "end_session": "end_session"
    })
    builder.add_edge("generate_brief_pdf", "end_session")

    # Ideas are always validated by the critic before the advisor sees them.
    builder.add_edge("investment_ideas", "suitability_critic")
    builder.add_conditional_edges("suitability_critic", route_after_critic, {
        "mark_failed_slots": "mark_failed_slots",
        "review_recommendations": "review_recommendations"
    })
    builder.add_edge("mark_failed_slots", "investment_ideas")
    builder.add_conditional_edges("review_recommendations", route_after_recommendation_review, {
        "memo_formatter": "memo_formatter",
        "meeting_brief": "meeting_brief",
        "investment_ideas": "investment_ideas",
        "suitability_critic": "suitability_critic",
        "end_session": "end_session"
    })

    builder.add_edge("memo_formatter", "review_memo")
    builder.add_conditional_edges("review_memo", route_after_memo_review, {
        "generate_pdf": "generate_pdf",
        "memo_formatter": "memo_formatter",
        "investment_ideas": "investment_ideas",
        "suitability_critic": "suitability_critic",
        "end_session": "end_session"
    })
    builder.add_edge("generate_pdf", "end_session")
    builder.add_edge("end_session", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver(serde=checkpoint_serializer()))
