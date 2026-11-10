"""OpenRouter LLM client with typed structured output (Slide 26: model-agnostic, typed handoffs)."""

import os
import json
import logging
import time
from typing import Type, TypeVar, Optional, Dict, Any, List
import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("OpenRouterLLMClient")
T = TypeVar("T", bound=BaseModel)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

class LLMUnavailableError(RuntimeError):
    """Raised when the model cannot produce valid output. Agents surface this instead of guessing."""

def _agent_env_key(agent_name: str) -> str:
    # "MarketIntelligenceAgent" -> "OPENROUTER_MODEL_MARKETINTELLIGENCEAGENT"
    return "OPENROUTER_MODEL_" + "".join(c for c in agent_name.upper() if c.isalnum())

class OpenRouterLLMClient:
    """
    Calls OpenRouter's OpenAI-compatible chat completions API and validates the reply
    against the requested Pydantic model. Invalid output triggers one retry with the
    validation errors fed back to the model; a second failure raises LLMUnavailableError.

    Model selection: OPENROUTER_MODEL sets the default; OPENROUTER_MODEL_<AGENTNAME>
    (e.g. OPENROUTER_MODEL_INVESTMENTIDEAAGENT) overrides it for one agent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        max_retries: int = 1,
        timeout_s: float = 120.0,
        transport: Optional[httpx.BaseTransport] = None
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.default_model = default_model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        self.timeout_s = timeout_s  # wall-clock limit per HTTP attempt; a slower attempt is dropped and retried
        self._http = httpx.Client(timeout=timeout_s, transport=transport)
        self.last_call: Dict[str, Any] = {}
        self._attempts: List[Dict[str, Any]] = []
        self.reset_usage()

    # ---- token accounting (deck slide 31): every attempt is paid for, so every attempt is counted ----

    def reset_usage(self) -> None:
        """Start a new run-level counter. `usage_total` accumulates across every call and every retry since the reset."""
        self.usage_total: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "attempts": 0}

    @staticmethod
    def _usage_dict(prompt: int, completion: int, attempts: int) -> Dict[str, int]:
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion, "attempts": attempts}

    def _finish_call(self, model: str, prompt: int, completion: int, attempts: int, **extra: Any) -> None:
        """Record one logical call: `last_call.usage` holds the totals over ALL its attempts, and the run counter grows by them."""
        self.last_call = {"model": model, "attempts": attempts, "usage": self._usage_dict(prompt, completion, attempts),
                          "attempt_log": self._attempts, **extra}
        self.usage_total["prompt_tokens"] += prompt
        self.usage_total["completion_tokens"] += completion
        self.usage_total["total_tokens"] += prompt + completion
        self.usage_total["calls"] += 1
        self.usage_total["attempts"] += attempts

    def model_for(self, agent_name: Optional[str]) -> str:
        if agent_name:
            return os.environ.get(_agent_env_key(agent_name), self.default_model)
        return self.default_model

    def _chat(self, model: str, messages: List[Dict[str, str]], response_model: Type[BaseModel], temperature: float) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": "UBS OneView Advisory"}
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": False,
                    "schema": response_model.model_json_schema()
                }
            }
        }
        resp = self._post(headers, body)
        if resp.status_code == 400 and "grammar is too large" in resp.text:
            # Some providers cannot compile large schemas into constrained decoding. The schema is already in the
            # system prompt and every reply is validated against the Pydantic model, so resend without the format.
            logger.info(f"{response_model.__name__}: schema too large for constrained decoding; retrying without response_format.")
            body.pop("response_format")
            resp = self._post(headers, body)
        resp.raise_for_status()
        return resp.json()

    def _post(self, headers: Dict[str, str], body: Dict[str, Any]) -> httpx.Response:
        """One HTTP request with a total deadline. OpenRouter holds a slow request open with keep-alive bytes, so httpx's
        read timeout alone never fires: the reply is read in chunks and dropped once `timeout_s` has passed."""
        deadline = time.monotonic() + self.timeout_s
        with self._http.stream("POST", OPENROUTER_URL, headers=headers, json=body) as resp:
            chunks = []
            for chunk in resp.iter_bytes():
                if time.monotonic() > deadline:
                    raise httpx.TimeoutException(f"no complete reply within {self.timeout_s:.0f} s", request=resp.request)
                chunks.append(chunk)
            # iter_bytes has already decoded the body, so the encoding headers must not be passed on
            headers = {k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")}
            return httpx.Response(resp.status_code, headers=headers, content=b"".join(chunks), request=resp.request)

    @staticmethod
    def _extract_json(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        return text.strip()

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.2,
        agent_name: Optional[str] = None
    ) -> T:
        model = self.model_for(agent_name)
        schema_hint = json.dumps(response_model.model_json_schema())
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\nRespond with a single JSON object that validates against this JSON schema:\n{schema_hint}"},
            {"role": "user", "content": user_prompt}
        ]

        last_error: Optional[str] = None
        prompt_tokens = completion_tokens = 0  # summed over every attempt, including those that fail validation
        self._attempts = []  # per attempt: seconds, outcome and tokens, so a dropped or retried attempt is visible on its own
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                data = self._chat(model, messages, response_model, temperature)
                reported = data.get("usage") or {}
                self._attempts.append({"seconds": round(time.monotonic() - started, 3), "outcome": "reply",
                                       "prompt_tokens": int(reported.get("prompt_tokens") or 0),
                                       "completion_tokens": int(reported.get("completion_tokens") or 0)})
                prompt_tokens += int(reported.get("prompt_tokens") or 0)  # counted before parsing: a rejected reply still cost tokens
                completion_tokens += int(reported.get("completion_tokens") or 0)
                content = data["choices"][0]["message"]["content"] or ""
                result = response_model.model_validate_json(self._extract_json(content))
                self._finish_call(data.get("model", model), prompt_tokens, completion_tokens, attempt + 1)
                return result
            except ValidationError as e:
                self._attempts[-1]["outcome"] = "invalid output"
                last_error = str(e)
                logger.warning(f"[{agent_name}] {response_model.__name__} failed validation (attempt {attempt + 1}): {e}")
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Your previous output failed schema validation:\n{e}\nReturn corrected JSON only."}
                ]
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
                failed = f"failed: {type(e).__name__}: {e}"[:200]
                if len(self._attempts) > attempt:  # a reply came back but was malformed
                    self._attempts[-1]["outcome"] = failed
                else:  # no reply: dropped, timed out or refused
                    self._attempts.append({"seconds": round(time.monotonic() - started, 3), "outcome": failed,
                                           "prompt_tokens": 0, "completion_tokens": 0})
                last_error = str(e)
                logger.warning(f"[{agent_name}] OpenRouter call failed (attempt {attempt + 1}): {e}")

        self._finish_call(model, prompt_tokens, completion_tokens, self.max_retries + 1, error=last_error)
        raise LLMUnavailableError(
            f"{agent_name or 'Agent'}: model '{model}' did not return a valid {response_model.__name__} "
            f"after {self.max_retries + 1} attempts ({last_error}). Data unavailable."
        )

def default_llm_client():
    """OpenRouter when OPENROUTER_API_KEY is set; otherwise the deterministic offline client."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterLLMClient()
    from src.llm.fake import FakeAdvisoryLLMClient
    logger.info("OPENROUTER_API_KEY not set; running in deterministic offline mode.")
    return FakeAdvisoryLLMClient()
