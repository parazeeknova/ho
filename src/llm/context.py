import asyncio
import json
import re
from typing import Any

from generalcompute import GeneralCompute

from src.configuration import LLMConfig, get_config
from src.logging import get_logger
from src.radar.core.governor import (
    _is_429,
    acquire_budget,
    handle_429,
    init_governor,
    release_budget,
)
from src.retry import _is_transient

logger = get_logger("llm")

_initialized = False


def _ensure_governor() -> None:
    global _initialized
    if not _initialized:
        init_governor()
        _initialized = True


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
        _ensure_governor()

    async def aclose(self) -> None:
        pass

    async def chat(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        interactive: bool = False,
    ) -> str:
        current_prompt = prompt
        if len(current_prompt) > 120000:
            current_prompt = current_prompt[:120000]

        if schema is not None:
            current_prompt += "\n\nYou MUST return valid JSON matching this schema:\n" + json.dumps(
                schema
            )

        _mt = max_tokens if max_tokens is not None else self._max_tokens
        est_tokens = len(current_prompt) // 3 + _mt

        def _call_llm() -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": current_prompt}],
                "max_tokens": _mt,
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
            await acquire_budget(est_tokens, interactive=interactive)
            try:
                output = await asyncio.to_thread(_call_llm)
                release_budget()
                return output
            except asyncio.CancelledError:
                release_budget()
                raise
            except Exception as e:
                release_budget()
                last_error = e
                if _is_429(str(e)) or _is_transient(e):
                    await handle_429()
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
        logger.error(
            f"LLM failed after {self._max_retries} retries",
            exception=str(last_error),
        )
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
        max_tokens: int | None = None,
        interactive: bool = False,
    ) -> dict[str, Any] | list[Any]:
        full = prompt
        if content:
            full = prompt + "\n\n" + content[:limit]
        raw = await self.chat(
            full,
            schema=schema,
            max_tokens=max_tokens,
            interactive=interactive,
        )
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
