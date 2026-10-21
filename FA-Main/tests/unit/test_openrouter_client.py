"""Unit tests for the OpenRouter client: typed output, retry on invalid output, fail-safe errors."""

import json
import httpx
import pytest
from pydantic import BaseModel
from src.llm.openrouter import OpenRouterLLMClient, LLMUnavailableError, default_llm_client
from src.llm.fake import FakeAdvisoryLLMClient

class Answer(BaseModel):
    ticker: str
    weight_pct: float

def make_client(replies, seen):
    """Client whose HTTP calls return each reply in turn (str content, or an int HTTP status)."""
    replies = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "boom"})
        return httpx.Response(200, json={"model": seen[-1]["model"], "choices": [{"message": {"content": reply}}]})

    return OpenRouterLLMClient(api_key="test-key", default_model="test/default", transport=httpx.MockTransport(handler))

def test_returns_validated_model_and_sends_schema():
    seen = []
    client = make_client(['{"ticker": "IEF", "weight_pct": 5.0}'], seen)
    result = client.generate_structured("sys", "user", Answer, agent_name="InvestmentIdeaAgent")
    assert result == Answer(ticker="IEF", weight_pct=5.0)
    assert seen[0]["model"] == "test/default"
    assert seen[0]["response_format"]["json_schema"]["name"] == "Answer"

def test_invalid_output_is_retried_with_validation_feedback():
    seen = []
    client = make_client(['{"ticker": "IEF"}', '```json\n{"ticker": "IEF", "weight_pct": 5}\n```'], seen)
    result = client.generate_structured("sys", "user", Answer)
    assert result.weight_pct == 5.0
    assert len(seen) == 2
    assert "failed schema validation" in seen[1]["messages"][-1]["content"]
    assert client.last_call["attempts"] == 2

def test_persistent_failure_raises_instead_of_guessing():
    client = make_client([500, 500], [])
    with pytest.raises(LLMUnavailableError, match="Data unavailable"):
        client.generate_structured("sys", "user", Answer)

def test_per_agent_model_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_MARKETINTELLIGENCEAGENT", "test/research-model")
    seen = []
    client = make_client(['{"ticker": "GLD", "weight_pct": 2}'], seen)
    client.generate_structured("sys", "user", Answer, agent_name="MarketIntelligenceAgent")
    assert seen[0]["model"] == "test/research-model"
    assert client.model_for("InvestmentIdeaAgent") == "test/default"

def test_default_client_is_offline_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert isinstance(default_llm_client(), FakeAdvisoryLLMClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert isinstance(default_llm_client(), OpenRouterLLMClient)
