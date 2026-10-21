"""Base Agent class enforcing strict typed I/O, tool bindings, and prompt isolation."""

import time
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any, Optional
from src.llm.base import LLMClientProtocol
from src.llm.gemini import GeminiAdvisoryLLMClient

InT = TypeVar("InT")
OutT = TypeVar("OutT")

class BaseAdvisoryAgent(ABC, Generic[InT, OutT]):
    """Base advisory agent with isolated role boundaries and deterministic tool bindings."""

    def __init__(self, name: str, llm_client: Optional[LLMClientProtocol] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.llm = llm_client or GeminiAdvisoryLLMClient()

    @abstractmethod
    def execute(self, inputs: InT, **kwargs: Any) -> OutT:
        """Execute agent's isolated reasoning with deterministic tools and LLM synthesis."""
        pass

    def run(self, inputs: InT, **kwargs: Any) -> OutT:
        """Execution wrapper with logging and timing."""
        start_time = time.time()
        self.logger.info(f"Agent [{self.name}] starting execution.")
        try:
            result = self.execute(inputs, **kwargs)
            duration = round((time.time() - start_time) * 1000, 2)
            self.logger.info(f"Agent [{self.name}] completed in {duration}ms.")
            return result
        except Exception as e:
            self.logger.error(f"Agent [{self.name}] failed: {str(e)}")
            raise
