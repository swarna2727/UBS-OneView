"""Cumulative latency and token accounting (deck slide 31): retries and multiple calls are never lost."""

import copy
import httpx
import json
import pytest
from pydantic import BaseModel

from src.agents.base import BaseAdvisoryAgent
from src.evaluation.performance import performance_report
from src.llm.openrouter import LLMUnavailableError, OpenRouterLLMClient


class Answer(BaseModel):
    ticker: str
    weight_pct: float


def client_with(replies):
    """OpenRouter client whose HTTP replies are (content, prompt_tokens, completion_tokens) in turn."""
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        content, p, c = queue.pop(0)
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}})
    return OpenRouterLLMClient(api_key="k", default_model="m", transport=httpx.MockTransport(handler))


class TwoCallAgent(BaseAdvisoryAgent):
    """An agent that calls the model twice in one run (multi-turn)."""
    def __init__(self, llm):
        super().__init__("TwoCallAgent", llm)

    def execute(self, inputs):
        a = self.generate("sys", "first", Answer)
        b = self.generate("sys", "second", Answer)
        return [a, b]


class ToolOnlyAgent(BaseAdvisoryAgent):
    def __init__(self, llm):
        super().__init__("ToolOnly", llm)

    def execute(self, inputs):
        return inputs


GOOD = '{"ticker": "IEF", "weight_pct": 5}'
BAD = '{"ticker": "IEF"}'


def test_agent_usage_is_cumulative_over_retries_and_multiple_calls():
    llm = client_with([(BAD, 100, 10), (GOOD, 120, 12), (GOOD, 50, 5)])  # call 1 retries once, call 2 succeeds first time
    agent = TwoCallAgent(llm)
    agent.run(None)
    assert agent.last_run["usage"] == {"prompt_tokens": 270, "completion_tokens": 27, "total_tokens": 297, "attempts": 3, "calls": 2}


def test_the_agent_spec_keys_are_present():
    llm = client_with([(GOOD, 7, 3)])
    agent = TwoCallAgent.__new__(TwoCallAgent)
    BaseAdvisoryAgent.__init__(agent, "OneCall", llm)
    agent.execute = lambda inputs: agent.generate("s", "u", Answer)
    agent.run(None)
    assert {"prompt_tokens", "completion_tokens", "total_tokens", "attempts"} <= set(agent.last_run["usage"])
    assert agent.last_run["usage"]["total_tokens"] == 10


def test_one_runs_usage_never_leaks_into_the_next():
    llm = client_with([(GOOD, 10, 1), (GOOD, 20, 2), (GOOD, 30, 3), (GOOD, 40, 4)])
    agent = TwoCallAgent(llm)
    agent.run(None)
    first = agent.last_run["usage"]["total_tokens"]
    agent.run(None)
    assert first == 33 + 0 and agent.last_run["usage"]["total_tokens"] == 77 and llm.usage_total["total_tokens"] == 110


def test_a_tool_only_agent_reports_no_usage_even_after_a_model_agent():
    llm = client_with([(GOOD, 10, 1), (GOOD, 10, 1)])
    TwoCallAgent(llm).run(None)
    tool = ToolOnlyAgent(llm)
    tool.run("x")
    assert tool.last_run["usage"] is None


def test_a_failed_run_keeps_the_tokens_every_attempt_spent():
    llm = client_with([(BAD, 100, 10), (BAD, 110, 11)])
    agent = TwoCallAgent(llm)
    with pytest.raises(LLMUnavailableError):
        agent.run(None)
    assert agent.last_run["usage"]["total_tokens"] == 231 and agent.last_run["usage"]["attempts"] == 2
    assert "error" in agent.last_run


def test_offline_agent_reports_attempts_and_zero_tokens(fake_llm, balanced_client_data):
    from src.agents.market_intelligence import MarketIntelligenceAgent
    from src.agents.client_profile import ClientProfileAgent
    profile = ClientProfileAgent(fake_llm).run(balanced_client_data)
    agent = MarketIntelligenceAgent(fake_llm)
    agent.run(profile)
    assert agent.last_run["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "attempts": 1, "calls": 1}


def test_performance_report_reads_the_cumulative_figures(fake_llm):
    from src.evaluation.runner import drive_session
    from src.evaluation.scorecard import scenarios
    run = copy.deepcopy(drive_session(scenarios()["pre_meeting"], fake_llm, "tok"))
    run.llm = object()  # a client with no call_history: counts come from the logged cumulative usage
    outputs = [e for e in run.log if e["event"] == "agent_output"]
    for e in outputs:
        e["usage"] = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, "attempts": 2, "calls": 1}
    p = performance_report(run)
    assert p["tokens"] == {"prompt": 1000 * len(outputs), "completion": 100 * len(outputs), "total": 1100 * len(outputs)}
    assert p["llm_calls"] == len(outputs) and p["llm_attempts"] == 2 * len(outputs)
    assert p["cost_usd"] == round(len(outputs) * (1000 / 1e6 * 3.0 + 100 / 1e6 * 15.0), 4)
