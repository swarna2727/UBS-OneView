"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"

def test_create_session_endpoint(balanced_client_data):
    response = client.post("/sessions", json=balanced_client_data)
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["client_id"] == "CL-88201"
    assert len(data["workflow_stages"]) == 6
    assert "center_panel" in data
    assert "right_panel" in data

def test_get_session_and_action_roundtrip(balanced_client_data):
    # 1. Create session
    create_resp = client.post("/sessions", json=balanced_client_data)
    assert create_resp.status_code == 201
    sid = create_resp.json()["session_id"]

    # 2. Get workspace
    get_resp = client.get(f"/sessions/{sid}")
    assert get_resp.status_code == 200
    assert get_resp.json()["session_id"] == sid

    # 3. Add advisor note action
    action_resp = client.post(f"/sessions/{sid}/actions", json={
        "action": "add_advisor_note",
        "notes": "Client requested emphasis on defensive income yield."
    })
    assert action_resp.status_code == 200
    assert "Client requested emphasis on defensive income yield." in action_resp.json()["advisor_notes"]

    # 4. Get audit trail
    audit_resp = client.get(f"/sessions/{sid}/audit")
    assert audit_resp.status_code == 200
    assert len(audit_resp.json()["audit_trail"]) > 0

def test_review_checkpoint_exposed_and_disallowed_action_rejected(incomplete_client_data):
    create_resp = client.post("/sessions", json=incomplete_client_data)
    assert create_resp.status_code == 201
    body = create_resp.json()
    assert body["pending_review"]["stage"] == "client_profile"
    assert "approve" not in body["pending_review"]["allowed_actions"]

    action_resp = client.post(f"/sessions/{body['session_id']}/actions", json={"action": "approve"})
    assert action_resp.status_code == 400
