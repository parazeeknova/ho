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

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same_job": {"type": "boolean"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["same_job", "confidence"],
}

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
        self._fallback_models = [m for m in (cfg.fallback_models or []) if m and m != self.model]
        self._max_retries = cfg.max_retries
        self._retry_delay = cfg.retry_delay
        self._max_tokens = cfg.max_tokens
        # Per-model failure notes from the last chat() so callers can tell the
        # user what was tried and why ("deepseek-v3.2: read timeout", ...).
        self.last_failures: list[tuple[str, str]] = []
        try:
            self._client = GeneralCompute(api_key=cfg.api_key)
        except ValueError:
            self._client = None
        _ensure_governor()

    def model_chain(self) -> list[str]:
        """Primary model + fallbacks, in the order chat() tries them."""
        return [self.model, *self._fallback_models]

    def failure_report(self) -> str:
        """Human-readable summary of what chat() last tried and what failed."""
        if not self.last_failures:
            return "no attempts recorded"
        return "; ".join(f"{m}: {err}" for m, err in self.last_failures)

    async def aclose(self) -> None:
        pass

    async def chat(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        interactive: bool = False,
        system_prompt: str | None = None,
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

        if self._client is None:
            raise RuntimeError("LLM client not initialized: missing API key")

        def _call_llm(model: str) -> str:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": current_prompt})
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": _mt,
            }
            if schema is not None:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            return msg.content or ""

        # Try each model in the chain: primary first, then fallbacks. A model
        # that is overloaded (429) or times out exhausts its retries, then we
        # move to the next model instead of failing the whole request.
        chain = [self.model, *self._fallback_models]
        last_error: Exception | None = None
        self.last_failures = []
        for model in chain:
            backoff = self._retry_delay
            for attempt in range(1, self._max_retries + 1):
                await acquire_budget(est_tokens, interactive=interactive)
                try:
                    output = await asyncio.to_thread(_call_llm, model)
                    release_budget()
                    return output
                except asyncio.CancelledError:
                    release_budget()
                    raise
                except Exception as e:
                    release_budget()
                    last_error = e
                    # 429 = provider throttled. Don't hammer the same model;
                    # move to the next one in the chain immediately.
                    if _is_429(str(e)):
                        self.last_failures.append((model, "rate-limited (429)"))
                        logger.warning(
                            f"LLM model {model} rate-limited (429); switching fallback",
                            model=model,
                        )
                        await handle_429()
                        break
                    wait = max(5.0, backoff * 3) if _is_transient(e) else backoff
                    self.last_failures.append((model, str(e)[:120]))
                    logger.warning(
                        f"LLM retry {attempt}/{self._max_retries} on {model}",
                        retry_count=attempt,
                        model=model,
                        exception=str(e),
                    )
                    if attempt < self._max_retries:
                        await asyncio.sleep(wait)
                        backoff *= 2
            else:
                # Model exhausted its retries; log and try next fallback.
                continue
            # broke out on 429 -> try next model.
            continue

        logger.error(
            "LLM failed on all models",
            models=", ".join(chain),
            exception=str(last_error),
        )
        raise RuntimeError(f"LLM failed on all models [{', '.join(chain)}]: {last_error}")

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
        system_prompt: str | None = None,
    ) -> dict[str, Any] | list[Any]:
        full = prompt
        if content:
            full = prompt + "\n\n" + content[:limit]
        raw = await self.chat(
            full,
            schema=schema,
            max_tokens=max_tokens,
            interactive=interactive,
            system_prompt=system_prompt,
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
