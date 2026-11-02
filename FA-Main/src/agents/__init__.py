"""Agents package exports.

OneView agents (deck slide 16): Client Profile, Market Intelligence, Opportunity, Fit & Eligibility Critic and
Brief & Record, plus the Client Limits rule layer. The earlier class names remain as aliases."""

from src.agents.base import BaseAdvisoryAgent
from src.agents.client_profile import ClientProfileAgent
from src.agents.client_limits import ClientLimitsAgent
from src.agents.market_intelligence import MarketIntelligenceAgent
from src.agents.investment_idea import OpportunityAgent, InvestmentIdeaAgent, InvestmentIdeaAgentInputs
from src.agents.suitability_critic import (
    FitEligibilityCriticAgent, SuitabilityCriticAgent, SuitabilityCriticInputs, ProfileChangeInputs
)
from src.agents.memo_formatter import (
    BriefRecordAgent, MemoFormatterAgent, MemoFormatterInputs, MeetingBriefInputs, MeetingRecordInputs, DecisionRecordInputs
)

__all__ = [
    "BaseAdvisoryAgent",
    "ClientProfileAgent",
    "ClientLimitsAgent",
    "MarketIntelligenceAgent",
    "OpportunityAgent",
    "InvestmentIdeaAgent",
    "InvestmentIdeaAgentInputs",
    "FitEligibilityCriticAgent",
    "SuitabilityCriticAgent",
    "SuitabilityCriticInputs",
    "ProfileChangeInputs",
    "BriefRecordAgent",
    "MemoFormatterAgent",
    "MemoFormatterInputs",
    "MeetingBriefInputs",
    "MeetingRecordInputs",
    "DecisionRecordInputs"
]
