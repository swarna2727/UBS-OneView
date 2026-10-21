"""Production Gemini LLM Client using google-genai SDK."""

import os
import json
import logging
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from src.llm.base import LLMClientProtocol
from src.llm.fake import FakeAdvisoryLLMClient

logger = logging.getLogger("GeminiAdvisoryLLMClient")
T = TypeVar("T", bound=BaseModel)

class GeminiAdvisoryLLMClient(LLMClientProtocol):
    """Client for Google Gemini using modern google-genai SDK with structured output schema."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self._fallback_fake = FakeAdvisoryLLMClient()
        self._client = None

        if self.api_key and not self.api_key.startswith("your_") and len(self.api_key) > 10:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini client with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai Client: {e}. Falling back to offline client.")
                self._client = None
        else:
            logger.info("No valid Gemini API key provided; running in deterministic offline mode.")

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.2
    ) -> T:
        """Call Gemini model with structured output constraint, or fall back if offline."""
        if self._client is None:
            return self._fallback_fake.generate_structured(
                system_prompt, user_prompt, response_model, temperature
            )

        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=f"{system_prompt}\n\nUSER REQUEST:\n{user_prompt}",
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_model
                )
            )

            if response and response.text:
                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                parsed_dict = json.loads(cleaned_text.strip())
                return response_model.model_validate(parsed_dict)
            else:
                raise ValueError("Empty response text from Gemini API")

        except Exception as e:
            logger.warning(f"Gemini API structured call failed: {e}. Using deterministic fallback.")
            return self._fallback_fake.generate_structured(
                system_prompt, user_prompt, response_model, temperature
            )
