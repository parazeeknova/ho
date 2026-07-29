import asyncio
import json
import re
import time
from typing import Any

from generalcompute import GeneralCompute

from src.llm.config import LLMConfig

MAX_RETRIES = 5
RETRY_DELAY = 2

# Token bucket gating ALL LLM calls across every agent.
# Rate = tokens replenished per second. 1.4/s = 84/min (safe under 100 RPM).
# Max = burst capacity if the bucket has been idle.
_TOKEN_RATE = 1.4
_TOKEN_MAX = 30
_token_lock = asyncio.Lock()
_token_count = _TOKEN_MAX
_token_last = time.monotonic()

# Global rate-limit backpressure. When ANY caller gets a 429 / rate-limit
# error, we set this timestamp and drain the bucket so every OTHER caller
# waits the full penalty window before making another request.
_rate_penalty_secs = 60.0
_rate_limit_hit_at = 0.0


async def _acquire_llm_token() -> None:
    """Wait until a rate-limit token is available (token bucket)."""
    global _token_count, _token_last
    while True:
        # Honour the global penalty window if a 429 was just received.
        remaining = _rate_penalty_secs - (time.monotonic() - _rate_limit_hit_at)
        if remaining > 0:
            await asyncio.sleep(remaining)
            continue

        async with _token_lock:
            now = time.monotonic()
            elapsed = now - _token_last
            _token_last = now
            _token_count = min(_TOKEN_MAX, _token_count + elapsed * _TOKEN_RATE)
            if _token_count >= 1.0:
                _token_count -= 1.0
                return
            wait = (1.0 - _token_count) / _TOKEN_RATE
        await asyncio.sleep(wait)


async def _mark_rate_limited() -> None:
    """Called when the LLM returns a 429 or rate-limit error.
    Drains the token bucket under the lock so every concurrent caller pauses together."""
    global _rate_limit_hit_at, _token_count
    async with _token_lock:
        _rate_limit_hit_at = time.monotonic()
        _token_count = 0.0


def _is_rate_limited(error: Exception) -> bool:
    msg = str(error).lower()
    return "429" in msg or "rate limit" in msg


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
    def __init__(self, verbose: bool = True) -> None:
        self.cumulative_output_tokens = 0
        self.verbose = verbose
        cfg = LLMConfig()
        self.model = cfg.model
        self._client = GeneralCompute(api_key=cfg.api_key)

    async def aclose(self) -> None:
        pass

    async def chat(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        current_prompt = prompt
        if len(current_prompt) > 24000:
            current_prompt = current_prompt[:24000]

        if schema is not None:
            current_prompt += "\n\nYou MUST return valid JSON matching this schema:\n" + json.dumps(
                schema
            )

        def _call_llm() -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": current_prompt}],
                "max_tokens": 4096,
            }
            if schema is not None:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            return msg.content or ""

        last_error: Exception | None = None
        backoff = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            await _acquire_llm_token()
            try:
                output = await asyncio.to_thread(_call_llm)
                return output
            except Exception as e:
                last_error = e
                if _is_rate_limited(e):
                    await _mark_rate_limited()
                    # Rate-limit errors get longer, aggressive backoff
                    wait = max(5.0, backoff * 3)
                else:
                    wait = backoff
            if attempt < MAX_RETRIES:
                print(f"  [LLM retry {attempt}/{MAX_RETRIES}] {last_error}")
                await asyncio.sleep(wait)
                backoff *= 2
        raise RuntimeError(f"LLM failed after {MAX_RETRIES} retries: {last_error}")

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
