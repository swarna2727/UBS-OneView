"""MCP transport for the deterministic tools: the flag defaults to direct calls, and the MCP stdio server returns
exactly what the direct calls return, tool by tool and through the full workflow."""

import copy

import pytest

from src.api.service import AdvisorySessionService
from src.llm.fake import FakeAdvisoryLLMClient
from src.mcp_tools import gateway
from src.schemas.workspace import AdvisorActionRequest


def test_transport_defaults_to_direct_and_rejects_unknown_values(monkeypatch):
    monkeypatch.delenv(gateway.TRANSPORT_ENV, raising=False)
    assert gateway.transport() == "direct"
    monkeypatch.setenv(gateway.TRANSPORT_ENV, "MCP")
    assert gateway.transport() == "mcp"
    monkeypatch.setenv(gateway.TRANSPORT_ENV, "grpc")
    with pytest.raises(ValueError, match="ONEVIEW_TOOL_TRANSPORT"):
        gateway.transport()


def test_server_lists_the_three_tools():
    from src.mcp_tools.server import server
    import asyncio
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"cio_research_search", "client_limits_check", "portfolio_risk_engine"}


def _both(monkeypatch, fn, *args, **kwargs):
    monkeypatch.setenv(gateway.TRANSPORT_ENV, "direct")
    direct = fn(*args, **kwargs)
    monkeypatch.setenv(gateway.TRANSPORT_ENV, "mcp")
    return direct, fn(*args, **kwargs)


def _dumped(value):
    if isinstance(value, list):
        return [_dumped(v) for v in value]
    if isinstance(value, dict):
        return {k: _dumped(v) for k, v in value.items()}
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def test_each_tool_returns_the_same_result_over_mcp(monkeypatch, sample_profiles):
    raw = copy.deepcopy(sample_profiles["demo_sarah"])
    direct, mcp = _both(monkeypatch, gateway.cio_research_search, "Macro Moderate yield inflation equities ACWI", ["ACWI", "MSFT"])
    assert direct and _dumped(direct) == _dumped(mcp)
    direct, mcp = _both(monkeypatch, gateway.client_limits_check, raw)
    assert _dumped(direct) == _dumped(mcp)

    # the risk engine on a real idea set: run the workflow to the Critic, then call the engine both ways
    monkeypatch.setenv(gateway.TRANSPORT_ENV, "direct")
    service = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
    service.start_session({**raw, "use_case": "adhoc_request", "new_capital_amount": 120000.0}, session_id="mcp-parity")
    for _ in range(3):
        service.execute_action("mcp-parity", AdvisorActionRequest(action="approve"))
    s = service.get_state("mcp-parity")
    approved = {"ACWI", "AGGG", "BIL", "GLD", "VCSH", "BOTZ", "COCO", "IEAC"}
    direct, mcp = _both(monkeypatch, gateway.portfolio_risk_engine, s["profile"], s["candidate_ideas"].ideas,
                        s["research_brief"], approved, offer_map=s["offer_map"])
    assert _dumped(direct) == _dumped(mcp)


def test_full_workflow_is_identical_over_mcp(monkeypatch, sample_profiles):
    """UC2 for Sarah end to end with each transport: same offer map, evidence, assessment and regenerations."""
    def run(transport):
        monkeypatch.setenv(gateway.TRANSPORT_ENV, transport)
        service = AdvisorySessionService(llm_client=FakeAdvisoryLLMClient())
        sid = f"mcp-{transport}"
        service.start_session({**copy.deepcopy(sample_profiles["demo_sarah"]), "use_case": "adhoc_request", "new_capital_amount": 120000.0}, session_id=sid)
        for _ in range(3):
            service.execute_action(sid, AdvisorActionRequest(action="approve"))
        s = service.get_state(sid)
        a = s["assessment"]
        return {"offer_map": s["offer_map"].model_dump(mode="json"),
                "sources": [x.source_id for x in s["research_brief"].validated_sources],
                "ideas": [(i.slot_number, i.ticker, i.target_weight_pct) for i in s["candidate_ideas"].ideas],
                "statuses": [(e.status, sorted((r.rule_id, r.outcome) for r in e.rule_results)) for e in a.idea_evaluations],
                "metrics": (a.before_metrics.model_dump(), a.after_metrics.model_dump()),
                "regenerations": [(r["slot"], r["ticker"], r["rules"]) for r in s.get("regeneration_history") or []]}
    assert run("direct") == run("mcp")
