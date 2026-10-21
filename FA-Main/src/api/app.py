"""FastAPI REST Application for the UBS Human-Led Multi-Agent Advisory System."""

import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.schemas.workspace import AdvisorWorkspaceView, AdvisorActionRequest
from src.api.service import AdvisorySessionService

app = FastAPI(
    title="UBS Human-Led Multi-Agent Advisory API",
    version="2.0.0",
    description="Compliant UBS Advisory Architecture API with HITL checkpoints and 3-Panel Workspace support."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
service = AdvisorySessionService()

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "HEALTHY", "architecture": "UBS_MULTI_AGENT_ADVISORY_V2"}

@app.post("/sessions", response_model=AdvisorWorkspaceView, status_code=status.HTTP_201_CREATED)
def create_advisory_session(payload: Dict[str, Any], session_id: Optional[str] = None):
    """
    Start a new client advisory session.
    Runs graph through Client Profile, Market Intelligence, Idea Generation,
    and Suitability Review, pausing at relevant HITL checkpoints.
    """
    try:
        workspace_view = service.start_session(raw_input=payload, session_id=session_id)
        return workspace_view
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start advisory session: {str(e)}"
        )

@app.get("/sessions/{session_id}", response_model=AdvisorWorkspaceView)
def get_advisory_session_workspace(session_id: str):
    """Retrieve the current 3-panel workspace projection for an advisory session."""
    try:
        return service.get_workspace_view(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advisory session '{session_id}' not found."
        )

@app.post("/sessions/{session_id}/actions", response_model=AdvisorWorkspaceView)
def execute_advisor_action(session_id: str, action_req: AdvisorActionRequest):
    """
    Execute Relationship Manager (RM) command:
    - 'accept': Approve recommendations or final advisory memo
    - 'reject': Reject specific candidate idea (triggers selective regeneration)
    - 'revise': Request revisions with specific notes (triggers selective regeneration)
    - 'override_flag': Confirm exception override for mild concentration or drift flag
    - 'add_advisor_note': Append commentary to advisory audit record
    - 'resolve_conflict': Resolve questionnaire/liquidity profile contradictions
    """
    try:
        return service.execute_action(session_id, action_req)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advisory session '{session_id}' not found."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to execute advisor action: {str(e)}"
        )

@app.get("/sessions/{session_id}/pdf")
def download_advisory_memo_pdf(session_id: str):
    """Download the finalized, audit-approved 3-page UBS Advisory Memo PDF artifact."""
    pdf_path = service.get_pdf_path(session_id)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Finalized PDF memo artifact is not yet approved or generated."
        )
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"UBS_Advisory_Memo_{session_id}.pdf"
    )

@app.get("/sessions/{session_id}/audit")
def get_session_audit_trail(session_id: str):
    """Get complete audit trail of agent executions, deterministic calculations, and RM actions."""
    try:
        state = service.get_state(session_id)
        return {
            "session_id": session_id,
            "current_stage": state.get("current_stage"),
            "regeneration_count": state.get("regeneration_count", 0),
            "advisor_notes": state.get("advisor_notes", []),
            "audit_trail": state.get("audit_trail", [])
        }
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advisory session '{session_id}' not found."
        )
