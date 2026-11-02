"""Run a complete OneView session as a scripted Relationship Manager and keep everything the evaluators need.

The evaluators read the final graph state and the append-only run log, so every check works the same way on the
offline scripted model and on a live model."""

import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.api.service import AdvisorySessionService
from src.llm.base import LLMClientProtocol
from src.schemas.workspace import AdvisorActionRequest

@dataclass
class RunResult:
    use_case: str
    session_id: str
    raw_input: Dict[str, Any]
    state: Dict[str, Any]
    log: List[Dict[str, Any]]
    llm: Any
    visible: List[str] = field(default_factory=list)  # everything the RM was shown, serialised (for "never shown" checks)
    completed: bool = False  # reached an approved end
    blocked_at: Optional[str] = None  # checkpoint where the scripted RM could not continue

def _act(service: AdvisorySessionService, sid: str, **kw: Any):
    return service.execute_action(sid, AdvisorActionRequest(**kw))

def _snapshot(service: AdvisorySessionService, sid: str, sink: List[str]) -> None:
    """Record every recommendation-bearing thing the RM can see at this point: ideas, replaced ideas, critic output,
    warnings and the workspace's idea panel. (Research text may legitimately mention a product; an idea may not.)"""
    review = service.get_pending_review(sid) or {}
    view = service.get_workspace_view(sid)
    payload = view.center_panel.data_payload
    parts = [review.get("candidate_ideas"), review.get("regeneration_history"),
             review.get("output") if review.get("stage") == "suitability_review" else None,
             payload.get("candidate_ideas"), payload.get("critic_notes"),
             [w.model_dump() for w in view.center_panel.risk_warnings]]
    sink.append(json.dumps(parts, default=str))

def drive_session(raw: Dict[str, Any], llm: LLMClientProtocol, session_id: str, rm_note: str = "Evaluation: reviewed by the scripted RM.",
                  stop_at: Optional[str] = None) -> RunResult:
    """Approve every checkpoint a careful RM could approve; stop where the workflow does not allow it (a block is a valid outcome)."""
    session_id = f"{session_id}-{uuid.uuid4().hex[:6]}"  # a run log is append-only: never reuse an id across runs
    service = AdvisorySessionService(llm_client=llm)
    raw = copy.deepcopy(raw)
    result_visible: List[str] = []
    service.start_session(raw, session_id=session_id)
    blocked_at = None
    for _ in range(60):
        review = service.get_pending_review(session_id)
        if review is None:
            break
        _snapshot(service, session_id, result_visible)
        stage, allowed = review["stage"], review["allowed_actions"]
        if stage == stop_at:
            break
        try:
            if stage == "suitability_review":
                locked = set(review.get("locked_idea_ids", []))
                nxt = next((i for i in review["candidate_ideas"] if i["idea_id"] not in locked), None)
                if "lock_idea" not in allowed or nxt is None:
                    blocked_at = stage
                    break
                _act(service, session_id, action="lock_idea", idea_id=nxt["idea_id"], notes=rm_note)
            elif stage == "record_approval":
                record = review["output"]
                changes = [c for c in record["profile_changes"] if c["decision"] == "PENDING"]
                if changes:
                    _act(service, session_id, action="approve_change", item_id=changes[0]["change_id"], notes=rm_note)
                    continue
                flagged = [a for a in record["actions"] if a["status"] == "FLAGGED"]
                if flagged:
                    _act(service, session_id, action="edit_record", edits={"actions": [
                        {"action_id": a["action_id"], "owner": a["owner"] or "RM", "due_date": a["due_date"] or "Next meeting"} for a in flagged]})
                    continue
                if "approve" not in allowed:
                    blocked_at = stage
                    break
                _act(service, session_id, action="approve", notes=rm_note)
            elif "approve" in allowed:
                _act(service, session_id, action="approve", notes=rm_note)
            else:
                blocked_at = stage
                break
        except ValueError:
            blocked_at = stage
            break
    state = dict(service.get_state(session_id))
    return RunResult(
        use_case=raw.get("use_case", "adhoc_request"), session_id=session_id, raw_input=raw, state=state,
        log=service.get_run_log(session_id), llm=llm, visible=result_visible,
        completed=state.get("outcome") == "APPROVED", blocked_at=blocked_at)
