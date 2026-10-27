"""Base Agent class enforcing strict typed I/O, tool bindings, and prompt isolation."""

import time
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any, Optional, Dict, Type
from pydantic import BaseModel
from src.llm.base import LLMClientProtocol
from src.llm.openrouter import default_llm_client

InT = TypeVar("InT")
OutT = TypeVar("OutT")
T = TypeVar("T", bound=BaseModel)

class BaseAdvisoryAgent(ABC, Generic[InT, OutT]):
    """Base advisory agent with isolated role boundaries and deterministic tool bindings."""

    # Bump when the agent's prompt changes; recorded on every run for audit (AS-8).
    PROMPT_VERSION = "v1"

    def __init__(self, name: str, llm_client: Optional[LLMClientProtocol] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.llm = llm_client or default_llm_client()
        self.last_run: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, inputs: InT, **kwargs: Any) -> OutT:
        """Execute agent's isolated reasoning with deterministic tools and LLM synthesis."""
        pass

    def generate(self, system_prompt: str, user_prompt: str, response_model: Type[T], temperature: float = 0.2) -> T:
        """Call the LLM for this agent, routed to the agent's configured model."""
        return self.llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            temperature=temperature,
            agent_name=self.name
        )

    def run(self, inputs: InT, **kwargs: Any) -> OutT:
        """Execution wrapper with logging, timing and model/prompt version capture."""
        start_time = time.time()
        self.logger.info(f"Agent [{self.name}] starting execution.")
        if hasattr(self.llm, "last_call"):
            self.llm.last_call = {}  # so a deterministic agent never reports the previous agent's usage
        try:
            result = self.execute(inputs, **kwargs)
            duration = round((time.time() - start_time) * 1000, 2)
            model_for = getattr(self.llm, "model_for", None)
            last_call = getattr(self.llm, "last_call", None) or {}
            self.last_run = {
                "agent": self.name,
                "model": model_for(self.name) if callable(model_for) else type(self.llm).__name__,
                "prompt_version": self.PROMPT_VERSION,
                "duration_ms": duration,
                "usage": last_call.get("usage")  # token counts reported by the provider, if any
            }
            self.logger.info(f"Agent [{self.name}] completed in {duration}ms.")
            return result
        except Exception as e:
            self.logger.error(f"Agent [{self.name}] failed: {str(e)}")
            raise
