"""Session management service for the UBS Multi-Agent Advisory System."""

import uuid
import logging
from typing import Dict, Any, Optional, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph.workflow import build_advisory_graph
from src.graph.state import AdvisoryState
from src.schemas.workspace import AdvisorWorkspaceView, AdvisorActionRequest
from src.api.views import build_workspace_view
from src.llm.base import LLMClientProtocol

logger = logging.getLogger("AdvisorySessionService")

class AdvisorySessionService:
    """Manages active advisory graph sessions, checkpoints, and human-in-the-loop actions."""

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None):
        self.checkpointer = MemorySaver()
        self.llm_client = llm_client
        self.graph = build_advisory_graph(llm_client=llm_client, checkpointer=self.checkpointer)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(
        self,
        raw_input: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> AdvisorWorkspaceView:
        """Initialize and run graph until completion or first HITL interrupt."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        config = {"configurable": {"thread_id": sid}}

        initial_state: AdvisoryState = {
            "session_id": sid,
            "raw_input": raw_input,
            "locked_idea_ids": [],
            "regenerate_slots": [],
            "regeneration_count": 0,
            "max_regenerations": 3,
            "advisor_notes": [],
            "audit_trail": [],
            "current_stage": "client_profile",
            "requires_human_action": False,
            "is_completed": False
        }

        logger.info(f"Starting advisory session: {sid}")

        # Run graph
        try:
            result = self.graph.invoke(initial_state, config=config)
            # Check if interrupted
            state_snapshot = self.graph.get_state(config)
            is_interrupted = len(state_snapshot.tasks) > 0 and any(len(t.interrupts) > 0 for t in state_snapshot.tasks)
            current_state = state_snapshot.values
            current_state["requires_human_action"] = is_interrupted

            self.active_sessions[sid] = current_state
            return build_workspace_view(current_state, sid)
        except Exception as e:
            logger.error(f"Error in session {sid}: {e}")
            raise

    def get_workspace_view(self, session_id: str) -> AdvisorWorkspaceView:
        """Fetch current 3-panel workspace projection for a session."""
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = self.graph.get_state(config)
        if not state_snapshot.values:
            raise KeyError(f"Advisory session '{session_id}' not found.")

        current_state = state_snapshot.values
        is_interrupted = len(state_snapshot.tasks) > 0 and any(len(t.interrupts) > 0 for t in state_snapshot.tasks)
        current_state["requires_human_action"] = is_interrupted
        return build_workspace_view(current_state, session_id)

    def execute_action(
        self,
        session_id: str,
        action_request: AdvisorActionRequest
    ) -> AdvisorWorkspaceView:
        """Resume execution from an HITL interrupt with advisor command."""
        config = {"configurable": {"thread_id": session_id}}
        logger.info(f"Executing advisor action '{action_request.action}' on session: {session_id}")

        action_dict = action_request.model_dump()

        try:
            # Resume graph using Command(resume=action_dict)
            self.graph.invoke(Command(resume=action_dict), config=config)

            state_snapshot = self.graph.get_state(config)
            is_interrupted = len(state_snapshot.tasks) > 0 and any(len(t.interrupts) > 0 for t in state_snapshot.tasks)
            current_state = state_snapshot.values
            current_state["requires_human_action"] = is_interrupted

            # If advisor added a note
            if action_request.action == "add_advisor_note" and action_request.notes:
                notes = current_state.get("advisor_notes") or []
                notes.append(action_request.notes)
                current_state["advisor_notes"] = notes

            self.active_sessions[session_id] = current_state
            return build_workspace_view(current_state, session_id)
        except Exception as e:
            logger.error(f"Error executing action on session {session_id}: {e}")
            raise

    def get_state(self, session_id: str) -> AdvisoryState:
        """Return the raw internal graph state for debugging or assertion."""
        config = {"configurable": {"thread_id": session_id}}
        state_snapshot = self.graph.get_state(config)
        return state_snapshot.values

    def get_pdf_path(self, session_id: str) -> Optional[str]:
        """Get path to the rendered PDF memo artifact."""
        state = self.get_state(session_id)
        return state.get("pdf_path")
