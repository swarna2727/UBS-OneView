"""Minimal MCP server (stdio) exposing three deterministic tools, unchanged:

  cio_research_search    -> src.tools.retrieval.retrieval_rag_tool
  client_limits_check    -> src.tools.client_limits.client_limits_engine
  portfolio_risk_engine  -> src.tools.risk_engine.suitability_rule_engine

Run: python -m src.mcp_tools.server   (from FA-Main)

Arguments and results are JSON: Pydantic models are sent as model_dump(mode="json") and rebuilt with model_validate.
No model calls: like src/tools/, this module is covered by the architecture lint."""

from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from src.schemas.advisory import ClientAdvisoryProfile, InvestmentIdea, OfferMap, ResearchBrief
from src.tools.client_limits import client_limits_engine
from src.tools.retrieval import retrieval_rag_tool
from src.tools.risk_engine import suitability_rule_engine

server = MCPServer("oneview-tools", instructions="Deterministic UBS OneView tools: CIO research search, client limits, portfolio risk engine.")
# Tool return types are the builtin dict[str, Any]: the SDK sends that as the structured result itself, while
# typing.Dict is wrapped as {"result": ...}, which the gateway does not expect.


def _dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _ref(reference_date_iso: Optional[str]) -> Dict[str, str]:
    return {"reference_date_iso": reference_date_iso} if reference_date_iso else {}


@server.tool(description="Search the CIO research corpus. Returns document snippets with source IDs, dates and status.")
def cio_research_search(query: str, asset_universe: Optional[List[str]] = None) -> dict[str, Any]:
    return {"snippets": [_dump(s) for s in retrieval_rag_tool(query, asset_universe=asset_universe)]}


@server.tool(description="Map what UBS can offer this client (offer map): eligible products, blocked areas, gaps, escalations.")
def client_limits_check(raw_input: Dict[str, Any], reference_date_iso: Optional[str] = None) -> dict[str, Any]:
    return _dump(client_limits_engine(raw_input, **_ref(reference_date_iso)))


@server.tool(description="Run the suitability rules (R1-R10, E1) on a candidate idea set: per-idea and portfolio results, before/after risk metrics.")
def portfolio_risk_engine(profile: Dict[str, Any], ideas: List[Dict[str, Any]], approved_tickers: List[str],
                          research_brief: Optional[Dict[str, Any]] = None, offer_map: Optional[Dict[str, Any]] = None,
                          reference_date_iso: Optional[str] = None) -> dict[str, Any]:
    engine = suitability_rule_engine(
        ClientAdvisoryProfile.model_validate(profile),
        [InvestmentIdea.model_validate(i) for i in ideas],
        ResearchBrief.model_validate(research_brief) if research_brief is not None else None,
        set(approved_tickers),
        offer_map=OfferMap.model_validate(offer_map) if offer_map is not None else None,
        **_ref(reference_date_iso))
    return {
        "portfolio_results": [_dump(r) for r in engine["portfolio_results"]],
        "idea_results": {idea_id: [_dump(r) for r in results] for idea_id, results in engine["idea_results"].items()},
        "before_metrics": _dump(engine["before_metrics"]),
        "after_metrics": _dump(engine["after_metrics"]),
        "after_holdings": [_dump(h) for h in engine["after_holdings"]],
    }


if __name__ == "__main__":
    server.run("stdio")
