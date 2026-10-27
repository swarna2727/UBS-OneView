"""Slide 32 baseline: one general-purpose LLM turns client inputs + research directly into the idea set.

No profile validator, rule engine, critic, regeneration loop or advisor checkpoints; the output is scored
with the same deterministic metrics as the multi-agent workflow.
"""

import json
from typing import Any, Dict, Optional
from src.llm.base import LLMClientProtocol
from src.schemas.advisory import CandidateIdeaSet
from src.tools.retrieval import retrieval_rag_tool
from src.tools.universe import approved_product_universe_tool

BASELINE_SYSTEM_PROMPT = (
    "You are a UBS investment adviser. From the client data and research snippets, propose exactly three "
    "complementary investment ideas for the client as a CandidateIdeaSet. Cite source IDs for market claims."
)

def single_llm_baseline(raw_input: Dict[str, Any], llm_client: LLMClientProtocol) -> CandidateIdeaSet:
    snippets = retrieval_rag_tool("macro yield inflation equities fixed income")
    universe = [p.ticker for p in approved_product_universe_tool()]
    prompt = (
        f"CLIENT DATA:\n{json.dumps(raw_input, indent=2, default=str)}\n\n"
        f"APPROVED UNIVERSE: {universe}\n\n"
        f"RESEARCH:\n{json.dumps([s.model_dump() for s in snippets], indent=2)}"
    )
    return llm_client.generate_structured(
        system_prompt=BASELINE_SYSTEM_PROMPT, user_prompt=prompt,
        response_model=CandidateIdeaSet, temperature=0.2, agent_name="SingleLLMBaseline",
    )
