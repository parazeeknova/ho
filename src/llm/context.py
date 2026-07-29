import asyncio
import json
import threading
from typing import Any

from generalcompute import GeneralCompute

from src.llm.config import LLMConfig

MAX_RETRIES = 3
RETRY_DELAY = 2

DIM = "\033[2m"
ITALIC = "\033[3m"
RESET = "\033[0m"

MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "company": {"type": "string"},
        "company_description": {"type": "string"},
        "role_summary": {"type": "string"},
        "match_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "shortlist_probability": {"type": "integer", "minimum": 0, "maximum": 100},
        "matching_skills": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "jd_summary": {"type": "string"},
        "salary": {"type": ["string", "null"]},
        "posted_date": {"type": ["string", "null"]},
        "apply_link": {"type": ["string", "null"]},
        "is_undergrad_friendly": {"type": "boolean"},
        "is_remote": {"type": "boolean"},
        "location": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": ["STRONG_MATCH", "GOOD_MATCH", "WEAK_MATCH", "NO_MATCH"],
        },
    },
    "required": [
        "role",
        "company",
        "match_percent",
        "shortlist_probability",
        "matching_skills",
        "missing_skills",
        "jd_summary",
        "location",
        "is_undergrad_friendly",
        "is_remote",
        "verdict",
    ],
}

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same_job": {"type": "boolean"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["same_job", "confidence"],
}


class ContextManager:
    _global_sem: asyncio.Semaphore | None = None
    _max_llm_concurrency = 8

    @classmethod
    def set_max_concurrency(cls, n: int) -> None:
        cls._max_llm_concurrency = n
        cls._global_sem = None

    def __init__(self, verbose: bool = True) -> None:
        self.cumulative_output_tokens = 0
        self.verbose = verbose
        self._lock = threading.Lock()
        cfg = LLMConfig()
        self.model = cfg.model
        self._client = GeneralCompute(api_key=cfg.api_key)
        if ContextManager._global_sem is None:
            ContextManager._global_sem = asyncio.Semaphore(ContextManager._max_llm_concurrency)

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
            }
            if schema is not None:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            return msg.content or ""

        sem = ContextManager._global_sem
        if sem is None:
            ContextManager._global_sem = asyncio.Semaphore(ContextManager._max_llm_concurrency)
            sem = ContextManager._global_sem

        last_error: Exception | None = None
        backoff = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            async with sem:
                try:
                    output = await asyncio.to_thread(_call_llm)
                    return output
                except Exception as e:
                    last_error = e
            if attempt < MAX_RETRIES:
                print(f"  [LLM retry {attempt}/{MAX_RETRIES}] {last_error}")
                await asyncio.sleep(backoff)
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
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
    return raw.strip()
