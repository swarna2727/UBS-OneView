"""Session management service for the UBS Multi-Agent Advisory System."""

import uuid
import logging
from typing import Dict, Any, Optional, List, Iterator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph.workflow import build_advisory_graph, checkpoint_serializer
from src.graph.state import AdvisoryState
from src.schemas.workspace import AdvisorWorkspaceView, AdvisorActionRequest
from src.api.views import build_workspace_view
from src.graph.routing import normalize_action, ADHOC_REQUEST, USE_CASES, REVIEW_EVENT
from src.llm.base import LLMClientProtocol
from src.llm.openrouter import default_llm_client
from src.audit.run_log import RunLog, file_fingerprint
from src.tools.risk_engine import CONFIG_POLICY_PATH

logger = logging.getLogger("AdvisorySessionService")

class AdvisorySessionService:
    """Manages active advisory graph sessions, checkpoints, and human-in-the-loop actions."""

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None, run_log: Optional[RunLog] = None):
        self.checkpointer = MemorySaver(serde=checkpoint_serializer())
        self.llm_client = llm_client or default_llm_client()  # one client shared by all agents
        self.run_log = run_log or RunLog()
        self.graph = build_advisory_graph(llm_client=llm_client, checkpointer=self.checkpointer, run_log=self.run_log)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(
        self,
        raw_input: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> AdvisorWorkspaceView:
        """Initialize and run graph until completion or first HITL interrupt."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        config = {"configurable": {"thread_id": sid}}

        initial_state = self._initial_state(sid, raw_input)
        logger.info(f"Starting advisory session: {sid}")
        self._log_run_started(sid, raw_input)

        try:
            self.graph.invoke(initial_state, config=config)
            return self._current_view(sid)
        except Exception as e:
            logger.error(f"Error in session {sid}: {e}")
            raise

    def _initial_state(self, sid: str, raw_input: Dict[str, Any]) -> AdvisoryState:
        use_case = raw_input.get("use_case") or ADHOC_REQUEST
        if use_case not in USE_CASES:
            raise ValueError(f"Unknown use_case '{use_case}'. Expected one of {list(USE_CASES)}.")
        return {
            "session_id": sid,
            "use_case": use_case,
            "raw_input": raw_input,
            "locked_idea_ids": [],
            "regenerate_slots": [],
            "regeneration_count": 0,
            "slot_regen_counts": {},
            "advisor_notes": [],
            "audit_trail": [],
            "current_stage": "client_profile",
            "requires_human_action": False,
            "is_completed": False
        }

    def _log_run_started(self, sid: str, raw_input: Dict[str, Any]) -> None:
        model_for = getattr(self.llm_client, "model_for", None)
        self.run_log.record(
            sid, "run_started", inputs=raw_input,
            policy_version=file_fingerprint(CONFIG_POLICY_PATH),
            llm_client=type(self.llm_client).__name__,
            default_model=model_for(None) if callable(model_for) else None
        )

    def _stream_progress(self, session_id: str, graph_input: Any) -> Iterator[Dict[str, Any]]:
        """Run the graph and yield one progress event per completed node, then the resulting workspace view."""
        config = {"configurable": {"thread_id": session_id}}
        for update in self.graph.stream(graph_input, config=config, stream_mode="updates"):
            for node, payload in update.items():
                if node == "__interrupt__":
                    continue
                yield {"event": "node_completed", "session_id": session_id, "node": node,
                       "stage": (payload or {}).get("current_stage")}
        view = self._current_view(session_id)
        yield {"event": "checkpoint", "session_id": session_id, "view": view.model_dump(mode="json")}

    def stream_session(self, raw_input: Dict[str, Any], session_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """start_session with real-time progress events (slide 26)."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        initial_state = self._initial_state(sid, raw_input)
        self._log_run_started(sid, raw_input)
        yield {"event": "session_started", "session_id": sid}
        yield from self._stream_progress(sid, initial_state)

    def stream_action(self, session_id: str, action_request: AdvisorActionRequest) -> Iterator[Dict[str, Any]]:
        """execute_action with real-time progress events. Validation errors raise before the stream starts."""
        pending = self._validate_action(session_id, action_request)
        if pending is None:  # advisor notes never resume the graph
            yield {"event": "checkpoint", "session_id": session_id,
                   "view": self.execute_action(session_id, action_request).model_dump(mode="json")}
            return
        yield from self._stream_progress(session_id, Command(resume=action_request.model_dump()))

    def _snapshot(self, session_id: str):
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = self.graph.get_state(config)
        if not state_snapshot.values:
            raise KeyError(f"Advisory session '{session_id}' not found.")
        return state_snapshot

    def get_pending_review(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the interrupt payload of the review checkpoint the graph is paused at, if any."""
        for task in self._snapshot(session_id).tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None

    def _current_view(self, session_id: str) -> AdvisorWorkspaceView:
        current_state = dict(self._snapshot(session_id).values)
        pending_review = self.get_pending_review(session_id)
        current_state["requires_human_action"] = pending_review is not None
        self.active_sessions[session_id] = current_state
        return build_workspace_view(current_state, session_id, pending_review)

    def get_workspace_view(self, session_id: str) -> AdvisorWorkspaceView:
        """Fetch current 3-panel workspace projection for a session."""
        return self._current_view(session_id)

    def execute_action(
        self,
        session_id: str,
        action_request: AdvisorActionRequest
    ) -> AdvisorWorkspaceView:
        """Resume execution from an HITL review checkpoint with an advisor command."""
        config = {"configurable": {"thread_id": session_id}}
        logger.info(f"Executing advisor action '{action_request.action}' on session: {session_id}")

        # Notes are recorded without resuming the graph, so they never act as a review decision.
        if action_request.action == "add_advisor_note":
            self._snapshot(session_id)
            if action_request.notes:
                self.run_log.record(session_id, "advisor_note", notes=action_request.notes)
                was_paused = self.get_pending_review(session_id) is not None
                self.graph.update_state(config, {"advisor_notes": [action_request.notes]})
                if was_paused:
                    # update_state clears the pending interrupt; re-enter the review node to restore it.
                    self.graph.invoke(None, config=config)
            return self._current_view(session_id)

        self._validate_action(session_id, action_request)

        try:
            self.graph.invoke(Command(resume=action_request.model_dump()), config=config)
            return self._current_view(session_id)
        except Exception as e:
            logger.error(f"Error executing action on session {session_id}: {e}")
            raise

    def _validate_action(self, session_id: str, action_request: AdvisorActionRequest) -> Optional[Dict[str, Any]]:
        """Check an advisor action against the current checkpoint; returns it, or None for a note-only action."""
        if action_request.action == "add_advisor_note":
            return None
        pending_review = self.get_pending_review(session_id)
        if pending_review is None:
            raise ValueError(f"Advisory session '{session_id}' is not waiting for an advisor review.")

        stage = pending_review["stage"]
        action = normalize_action(stage, action_request.action)
        if action not in pending_review.get("allowed_actions", []):
            raise ValueError(
                f"Action '{action_request.action}' is not allowed at the '{stage}' review. "
                f"Allowed: {pending_review.get('allowed_actions', [])}"
            )
        if action == "override_flag" and not (action_request.notes or "").strip():
            raise ValueError("An exception override must document the advisor's reasoning in 'notes'.")
        if stage == REVIEW_EVENT and not (action_request.notes or "").strip():
            raise ValueError("A market-event decision must document the advisor's rationale in 'notes'.")
        if action == "monitor" and not str((action_request.edits or {}).get("follow_up", "")).strip():
            raise ValueError("A 'monitor' decision needs a follow-up trigger or date in edits.follow_up.")
        if action in ("lock_idea", "unlock_idea"):
            ideas = pending_review.get("candidate_ideas") or []
            if action_request.idea_id not in {i["idea_id"] for i in ideas}:
                raise ValueError(f"{action} needs the idea_id of one of the three ideas.")
            if action == "lock_idea":
                assessment = self.get_state(session_id).get("assessment")
                status = assessment.overall_status if assessment else "FAIL"
                idea_status = next((ie.status for ie in assessment.idea_evaluations if ie.idea_id == action_request.idea_id), "FAIL")
                if idea_status == "FAIL" or status == "FAIL" and idea_status != "PASS":
                    raise ValueError("A FAIL idea cannot be locked in; revise, edit or reject it.")
                if (status == "FLAG" or idea_status == "FLAG") and not (action_request.notes or "").strip():
                    raise ValueError("Locking an idea in a FLAG set must document the advisor's exception in 'notes'.")
        if action == "edit_allocation":
            weight = (action_request.edits or {}).get("target_weight_pct")
            idea_ids = {i["idea_id"] for i in pending_review.get("candidate_ideas") or (self.get_state(session_id)["candidate_ideas"].model_dump()["ideas"])}
            if action_request.idea_id not in idea_ids or not isinstance(weight, (int, float)) or not 0 <= weight <= 100:
                raise ValueError("edit_allocation needs a valid idea_id and edits.target_weight_pct between 0 and 100.")
        return pending_review

    def get_state(self, session_id: str) -> AdvisoryState:
        """Return the raw internal graph state for debugging or assertion."""
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = self.graph.get_state(config)
        return state_snapshot.values

    def get_run_log(self, session_id: str) -> List[Dict[str, Any]]:
        """Full audit record for one run (AS-8)."""
        return self.run_log.read(session_id)

    def get_pdf_path(self, session_id: str) -> Optional[str]:
        """Get path to the rendered PDF memo artifact."""
        state = self.get_state(session_id)
        return state.get("pdf_path")
