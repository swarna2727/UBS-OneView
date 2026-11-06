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
        self._usage: Dict[str, int] = {}
        self._reset_usage()

    def _reset_usage(self) -> None:
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "attempts": 0, "calls": 0}

    def _absorb_usage(self) -> None:
        """Add the call that just finished (success or failure) to this run's totals, then clear it so it is never counted twice."""
        last = getattr(self.llm, "last_call", None) or {}
        usage = last.get("usage") or {}
        self._usage["calls"] += 1
        self._usage["attempts"] += int(last.get("attempts") or 1)
        self._usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self._usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        if hasattr(self.llm, "last_call"):
            self.llm.last_call = {}

    def accumulated_usage(self) -> Optional[Dict[str, int]]:
        """Cumulative usage of this agent run, or None when the agent made no model call (a rule layer or tool-only step)."""
        if not self._usage["calls"]:
            return None
        u = self._usage
        return {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                "total_tokens": u["prompt_tokens"] + u["completion_tokens"], "attempts": u["attempts"], "calls": u["calls"]}

    @abstractmethod
    def execute(self, inputs: InT, **kwargs: Any) -> OutT:
        """Execute agent's isolated reasoning with deterministic tools and LLM synthesis."""
        pass

    def generate(self, system_prompt: str, user_prompt: str, response_model: Type[T], temperature: float = 0.2) -> T:
        """Call the LLM for this agent, routed to the agent's configured model."""
        try:
            return self.llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                temperature=temperature,
                agent_name=self.name
            )
        finally:
            self._absorb_usage()  # tokens are spent even when the model's output is rejected or the call fails

    def run(self, inputs: InT, **kwargs: Any) -> OutT:
        """Execution wrapper with logging, timing and model/prompt version capture."""
        start_time = time.time()
        self.logger.info(f"Agent [{self.name}] starting execution.")
        self._reset_usage()
        if hasattr(self.llm, "last_call"):
            self.llm.last_call = {}  # so a deterministic agent never reports the previous agent's usage
        try:
            result = self.execute(inputs, **kwargs)
            duration = round((time.time() - start_time) * 1000, 2)
            model_for = getattr(self.llm, "model_for", None)
            self.last_run = {
                "agent": self.name,
                "model": model_for(self.name) if callable(model_for) else type(self.llm).__name__,
                "prompt_version": self.PROMPT_VERSION,
                "duration_ms": duration,
                # Cumulative over every model call and retry in this run (deck slide 31); None for a tool-only agent.
                "usage": self.accumulated_usage()
            }
            self.logger.info(f"Agent [{self.name}] completed in {duration}ms.")
            return result
        except Exception as e:
            self.logger.error(f"Agent [{self.name}] failed: {str(e)}")
            # A failed run still spent tokens on every attempt: keep them rather than losing them with the exception.
            self.last_run = {"agent": self.name, "prompt_version": self.PROMPT_VERSION, "usage": self.accumulated_usage(), "error": str(e)}
            raise
