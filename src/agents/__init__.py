"""Agents package exports."""

from src.agents.base import BaseAdvisoryAgent
from src.agents.client_profile import ClientProfileAgent
from src.agents.market_intelligence import MarketIntelligenceAgent
from src.agents.investment_idea import InvestmentIdeaAgent, InvestmentIdeaAgentInputs
from src.agents.suitability_critic import SuitabilityCriticAgent, SuitabilityCriticInputs
from src.agents.memo_formatter import MemoFormatterAgent, MemoFormatterInputs

__all__ = [
    "BaseAdvisoryAgent",
    "ClientProfileAgent",
    "MarketIntelligenceAgent",
    "InvestmentIdeaAgent",
    "InvestmentIdeaAgentInputs",
    "SuitabilityCriticAgent",
    "SuitabilityCriticInputs",
    "MemoFormatterAgent",
    "MemoFormatterInputs"
]