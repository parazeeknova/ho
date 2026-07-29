import asyncio
import json
import re
import time
from typing import Any

from generalcompute import GeneralCompute

from src.configuration import LLMConfig, get_config
from src.logging import get_logger
from src.retry import _is_transient

logger = get_logger("llm")


# Token bucket gating ALL LLM calls across every agent.
def _llm_state():
    cfg = get_config().llm
    return {
        "rate": cfg.token_rate,
        "max": cfg.token_max,
        "count": cfg.token_max,
        "last": time.monotonic(),
    }


_token_lock = asyncio.Lock()
_token_state = _llm_state()
_rate_limit_hit_at = 0.0


def _rate_penalty_secs() -> float:
    return get_config().llm.rate_penalty_secs


async def _acquire_llm_token() -> None:
    """Wait until a rate-limit token is available (token bucket)."""
    global _rate_limit_hit_at
    while True:
        remaining = _rate_penalty_secs() - (time.monotonic() - _rate_limit_hit_at)
        if remaining > 0:
            await asyncio.sleep(remaining)
            continue

        async with _token_lock:
            cfg = get_config().llm
            now = time.monotonic()
            elapsed = now - _token_state["last"]
            _token_state["last"] = now
            _token_state["count"] = min(
                cfg.token_max, _token_state["count"] + elapsed * cfg.token_rate
            )
            if _token_state["count"] >= 1.0:
                _token_state["count"] -= 1.0
                return
            wait = (1.0 - _token_state["count"]) / cfg.token_rate
        await asyncio.sleep(wait)


async def _mark_rate_limited() -> None:
    """Called when the LLM returns a 429 or rate-limit error."""
    global _rate_limit_hit_at
    async with _token_lock:
        _rate_limit_hit_at = time.monotonic()
        _token_state["count"] = 0.0


DIM = "\033[2m"
ITALIC = "\033[3m"
RESET = "\033[0m"

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same_job": {"type": "boolean"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["same_job", "confidence"],
}


class ContextManager:
    def __init__(self, verbose: bool = True, config: LLMConfig | None = None) -> None:
        self.cumulative_output_tokens = 0
        self.verbose = verbose
        cfg = config or get_config().llm
        self.model = cfg.model
        self._max_retries = cfg.max_retries
        self._retry_delay = cfg.retry_delay
        self._max_tokens = cfg.max_tokens
        try:
            self._client = GeneralCompute(api_key=cfg.api_key)
        except ValueError:
            self._client = None

    async def aclose(self) -> None:
        pass

    async def chat(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        current_prompt = prompt
        if len(current_prompt) > 120000:
            current_prompt = current_prompt[:120000]

        if schema is not None:
            current_prompt += "\n\nYou MUST return valid JSON matching this schema:\n" + json.dumps(
                schema
            )

        def _call_llm() -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": current_prompt}],
                "max_tokens": self._max_tokens,
            }
            if schema is not None:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            return msg.content or ""

        if self._client is None:
            raise RuntimeError("LLM client not initialized: missing API key")

        last_error: Exception | None = None
        backoff = self._retry_delay
        for attempt in range(1, self._max_retries + 1):
            await _acquire_llm_token()
            try:
                output = await asyncio.to_thread(_call_llm)
                return output
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                if _is_transient(e):
                    await _mark_rate_limited()
                    wait = max(5.0, backoff * 3)
                else:
                    wait = backoff
                logger.warning(
                    f"LLM retry {attempt}/{self._max_retries}",
                    retry_count=attempt,
                    exception=str(e),
                )
            if attempt < self._max_retries:
                await asyncio.sleep(wait)
                backoff *= 2
        logger.error(f"LLM failed after {self._max_retries} retries", exception=str(last_error))
        raise RuntimeError(f"LLM failed after {self._max_retries} retries: {last_error}")

    async def maybe_flush(self) -> None:
        pass

    async def flush(self) -> None:
        pass

    def _flush_sync(self) -> None:
        pass

    async def json_chat(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        content: str = "",
        limit: int = 16000,
    ) -> dict[str, Any] | list[Any]:
        full = prompt
        if content:
            full = prompt + "\n\n" + content[:limit]
        raw = await self.chat(full, schema=schema)
        raw = _strip_markdown(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {} if "{" in prompt else []


def _strip_markdown(raw: str) -> str:
    """Strip <think> blocks and extract JSON from code fences.

    DeepSeek V3.2 sometimes emits reasoning blocks or conversational
    filler before the actual JSON output.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    return raw.strip()
