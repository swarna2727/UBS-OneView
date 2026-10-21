"""LLM Client Protocol and Base abstractions."""

from typing import Protocol, Type, TypeVar, Optional, Any
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMClientProtocol(Protocol):
    """Protocol for LLM interactions with enforced structured output validation."""

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.2,
        agent_name: Optional[str] = None
    ) -> T:
        """Generate content from the model and parse strictly into response_model."""
        ...
