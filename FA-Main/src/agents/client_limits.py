"""Understand | Client Limits layer: what UBS can offer this client (deck slide 12).

A rule layer, not an AI judgement: it makes no model call. It runs before any briefing or idea is produced,
and is shared by UC1 and UC2. The Opportunity Agent may only choose from its eligible products; the Fit &
Eligibility Critic re-checks every idea against it."""

from typing import Any, Dict, Optional
from src.agents.base import BaseAdvisoryAgent
from src.schemas.advisory import OfferMap
from src.mcp_tools.gateway import client_limits_check
from src.llm.base import LLMClientProtocol

class ClientLimitsAgent(BaseAdvisoryAgent[Dict[str, Any], OfferMap]):
    """Read the regulatory profile, match the product universe, attach disclosures, mark blocked areas and gaps, publish the offer map."""

    PROMPT_VERSION = "rules-v2"  # no prompt: the version tracks the rule set in config/suitability_policy.yaml

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None):
        super().__init__("ClientLimitsLayer", llm_client)

    def execute(self, inputs: Dict[str, Any]) -> OfferMap:
        return client_limits_check(inputs)
