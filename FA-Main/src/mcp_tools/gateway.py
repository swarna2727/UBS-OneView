"""The agents' single entry point to the deterministic tools, over the configured transport.

ONEVIEW_TOOL_TRANSPORT=direct (default): call the functions in src/tools/ in-process, exactly as before.
ONEVIEW_TOOL_TRANSPORT=mcp: call the same functions through the MCP stdio server (src/mcp_tools/server.py).
Both return the same Pydantic types, so agent code does not depend on the transport."""

import json
import os
from typing import Any, Dict, List, Optional

from src.schemas.advisory import OfferMap
from src.schemas.tools import DocumentSnippet, Holding, RiskMetrics, RuleResult
from src.tools.client_limits import client_limits_engine
from src.tools.retrieval import retrieval_rag_tool
from src.tools.risk_engine import suitability_rule_engine

TRANSPORT_ENV = "ONEVIEW_TOOL_TRANSPORT"
TRANSPORTS = ("direct", "mcp")


def transport() -> str:
    value = (os.environ.get(TRANSPORT_ENV) or "direct").strip().lower()
    if value not in TRANSPORTS:
        raise ValueError(f"{TRANSPORT_ENV} must be one of {TRANSPORTS}, not '{value}'.")
    return value


def _call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    from src.mcp_tools.client import get_client  # imported only when MCP is in use
    return get_client().call(name, arguments)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def cio_research_search(query: str, asset_universe: Optional[List[str]] = None) -> List[DocumentSnippet]:
    if transport() == "direct":
        return retrieval_rag_tool(query, asset_universe=asset_universe)
    out = _call("cio_research_search", {"query": query, "asset_universe": asset_universe})
    return [DocumentSnippet.model_validate(s) for s in out["snippets"]]


def client_limits_check(raw_input: Dict[str, Any]) -> OfferMap:
    if transport() == "direct":
        return client_limits_engine(raw_input)
    return OfferMap.model_validate(_call("client_limits_check", {"raw_input": _json(raw_input)}))


def portfolio_risk_engine(profile: Any, ideas: List[Any], research_brief: Any, approved_tickers: set,
                          offer_map: Any = None) -> Dict[str, Any]:
    if transport() == "direct":
        return suitability_rule_engine(profile, ideas, research_brief, approved_tickers, offer_map=offer_map)
    out = _call("portfolio_risk_engine", {
        "profile": profile.model_dump(mode="json"),
        "ideas": [i.model_dump(mode="json") for i in ideas],
        "approved_tickers": sorted(approved_tickers),
        "research_brief": research_brief.model_dump(mode="json") if research_brief is not None else None,
        "offer_map": offer_map.model_dump(mode="json") if offer_map is not None else None,
    })
    return {
        "portfolio_results": [RuleResult.model_validate(r) for r in out["portfolio_results"]],
        "idea_results": {k: [RuleResult.model_validate(r) for r in v] for k, v in out["idea_results"].items()},
        "before_metrics": RiskMetrics.model_validate(out["before_metrics"]),
        "after_metrics": RiskMetrics.model_validate(out["after_metrics"]),
        "after_holdings": [Holding.model_validate(h) for h in out["after_holdings"]],
    }
