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
        usage = None
        if isinstance(reply, tuple):  # (content, usage): the provider reports tokens for this attempt
            reply, usage = reply
        body = {"model": seen[-1]["model"], "choices": [{"message": {"content": reply}}]}
        if usage:
            body["usage"] = usage
        return httpx.Response(200, json=body)

    return OpenRouterLLMClient(api_key="test-key", default_model="test/default", transport=httpx.MockTransport(handler))

def test_returns_validated_model_and_sends_schema():
    seen = []
    client = make_client(['{"ticker": "IEF", "weight_pct": 5.0}'], seen)
    result = client.generate_structured("sys", "user", Answer, agent_name="InvestmentIdeaAgent")
    assert result == Answer(ticker="IEF", weight_pct=5.0)
    assert seen[0]["model"] == "test/default"
    assert seen[0]["response_format"]["json_schema"]["name"] == "Answer"

def test_schema_too_large_for_the_provider_is_resent_without_response_format():
    """Providers that cannot compile a large schema reject it with a 400; the schema stays in the prompt and the reply is still validated."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "The compiled grammar is too large, which would cause performance issues."}})
        return httpx.Response(200, json={"model": body["model"], "choices": [{"message": {"content": '{"ticker": "IEF", "weight_pct": 5}'}}]})

    client = OpenRouterLLMClient(api_key="test-key", default_model="test/default", transport=httpx.MockTransport(handler))
    assert client.generate_structured("sys", "user", Answer) == Answer(ticker="IEF", weight_pct=5.0)
    assert len(seen) == 2 and "response_format" not in seen[1] and "JSON schema" in seen[1]["messages"][0]["content"]

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


# ------------------------------------------------------------------ cumulative token accounting (deck slide 31)

def u(prompt, completion):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

def test_tokens_are_summed_over_every_attempt_not_overwritten():
    client = make_client([('{"ticker": "IEF"}', u(100, 20)), ('{"ticker": "IEF", "weight_pct": 5}', u(150, 30))], [])
    client.generate_structured("sys", "user", Answer)
    assert client.last_call["usage"] == {"prompt_tokens": 250, "completion_tokens": 50, "total_tokens": 300, "attempts": 2}
    assert client.last_call["attempts"] == 2

def test_a_single_attempt_reports_its_own_tokens():
    client = make_client([('{"ticker": "IEF", "weight_pct": 1}', u(40, 8))], [])
    client.generate_structured("sys", "user", Answer)
    assert client.last_call["usage"] == {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48, "attempts": 1}

def test_tokens_spent_on_a_failed_call_are_not_lost():
    client = make_client([('{"ticker": "IEF"}', u(100, 20)), ('{"nope": 1}', u(110, 25))], [])
    with pytest.raises(LLMUnavailableError):
        client.generate_structured("sys", "user", Answer)
    assert client.last_call["usage"] == {"prompt_tokens": 210, "completion_tokens": 45, "total_tokens": 255, "attempts": 2}
    assert client.usage_total["total_tokens"] == 255 and client.usage_total["attempts"] == 2

def test_the_run_counter_accumulates_across_calls_until_reset():
    client = make_client([('{"ticker": "A", "weight_pct": 1}', u(10, 1)), ('{"ticker": "B"}', u(20, 2)),
                          ('{"ticker": "B", "weight_pct": 2}', u(30, 3))], [])
    client.generate_structured("sys", "user", Answer)
    client.generate_structured("sys", "user", Answer)
    assert client.usage_total == {"prompt_tokens": 60, "completion_tokens": 6, "total_tokens": 66, "calls": 2, "attempts": 3}
    client.reset_usage()
    assert client.usage_total == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "attempts": 0}

def test_missing_usage_from_the_provider_counts_as_zero_not_an_error():
    client = make_client(['{"ticker": "IEF", "weight_pct": 1}'], [])
    client.generate_structured("sys", "user", Answer)
    assert client.last_call["usage"]["total_tokens"] == 0 and client.last_call["usage"]["attempts"] == 1

def test_an_attempt_past_the_deadline_is_dropped_and_retried_and_logged_on_its_own():
    """A reply held open with keep-alive bytes never trips the read timeout, so each attempt has a wall-clock deadline."""
    import time
    calls = []

    class Trickle(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(4):
                time.sleep(0.1)
                yield b" "

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, stream=Trickle())
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": '{"ticker": "IEF", "weight_pct": 5}'}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 4}})

    client = OpenRouterLLMClient(api_key="k", default_model="m", timeout_s=0.25, transport=httpx.MockTransport(handler))
    assert client.generate_structured("sys", "user", Answer).ticker == "IEF"
    first, second = client.last_call["attempt_log"]
    assert first["outcome"].startswith("failed: TimeoutException") and first["completion_tokens"] == 0
    assert second["outcome"] == "reply" and second["completion_tokens"] == 4 and client.last_call["attempts"] == 2

def test_a_gzip_reply_is_decoded_once():
    import gzip
    body = json.dumps({"model": "m", "choices": [{"message": {"content": '{"ticker": "IEF", "weight_pct": 5}'}}]}).encode()
    transport = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-encoding": "gzip"}, content=gzip.compress(body)))
    client = OpenRouterLLMClient(api_key="k", default_model="m", transport=transport)
    assert client.generate_structured("sys", "user", Answer).ticker == "IEF"
