"""Context manager: token tracking, auto-flush, retry, JSON schema enforcement.

Thread-safe: a threading.Lock serialises token-counter updates and slot erasures
so that concurrent asyncio workers and signal-handler flushes never race on the
shared KV-cache state inside llama-server.
"""

import json
import threading
import urllib.request
from typing import Any

import httpx

LLM_URL = "http://127.0.0.1:8899"
MODEL = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF:Q5_K_M"
MAX_RETRIES = 3
RETRY_DELAY = 4
TOKEN_ESTIMATE_PER_CHAR = 0.4
FLUSH_THRESHOLD = 6000

DIM = "\033[2m"
ITALIC = "\033[3m"
RESET = "\033[0m"

MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "company": {"type": "string"},
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

REVALIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "company": {"type": "string"},
        "match_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "shortlist_probability": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["role", "company", "match_percent", "shortlist_probability"],
}


def _build_payload(prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_object",
            "schema": schema,
        }
    return payload


class ContextManager:
    def __init__(self, verbose: bool = True) -> None:
        self.cumulative_output_tokens = 0
        self.verbose = verbose
        self._lock = threading.Lock()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        payload = _build_payload(prompt, schema)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.post(
                    f"{LLM_URL}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
                msg = result["choices"][0]["message"]
                output: str = msg.get("content", "")
                reasoning: str = msg.get("reasoning_content", "")

                if self.verbose and reasoning:
                    self._show_thinking(reasoning, len(output))

                tokens = int((len(output) + len(reasoning)) * TOKEN_ESTIMATE_PER_CHAR)
                with self._lock:
                    self.cumulative_output_tokens += tokens
                return output
            except Exception:
                if attempt < MAX_RETRIES:
                    print(f"  [LLM retry {attempt}/{MAX_RETRIES}]")
                    await _async_sleep(RETRY_DELAY)
        raise RuntimeError("LLM failed after all retries")

    def _show_thinking(self, reasoning: str, output_len: int) -> None:
        lines = reasoning.strip().split("\n")
        preview = "\n".join(lines[:5])
        if len(lines) > 5:
            preview += f"\n  ... ({len(lines)} lines total)"
        print(f"\n  {DIM}{ITALIC}[thinking]{RESET} {DIM}{preview}{RESET}")
        print(f"  {DIM}[response] ({output_len} chars){RESET}")

    async def maybe_flush(self) -> None:
        with self._lock:
            if self.cumulative_output_tokens < FLUSH_THRESHOLD:
                return
        await self.flush()

    async def flush(self) -> None:
        with self._lock:
            try:
                resp = await self._client.get(f"{LLM_URL}/slots")
                slots = resp.json()
                for slot in slots:
                    sid = slot.get("id")
                    if sid is not None and slot.get("state") != 0:
                        await self._client.get(
                            f"{LLM_URL}/slots/{sid}", params={"action": "erase"}
                        )
                self.cumulative_output_tokens = 0
                print("  [ctx flushed]")
            except Exception:
                self.cumulative_output_tokens = 0

    def _flush_sync(self) -> None:
        """Synchronous fallback for signal handlers — uses urllib to avoid
        needing a running event loop.  Acquires the lock non-blocking so a
        signal handler never deadlocks waiting on an in-flight async flush."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            try:
                raw = urllib.request.urlopen(f"{LLM_URL}/slots", timeout=5).read()
                slots = json.loads(raw)
                for slot in slots:
                    sid = slot.get("id")
                    if sid is not None and slot.get("state") != 0:
                        urllib.request.urlopen(
                            f"{LLM_URL}/slots/{sid}?action=erase", timeout=5
                        )
                self.cumulative_output_tokens = 0
                print("  [ctx flushed]")
            except Exception:
                self.cumulative_output_tokens = 0
        finally:
            self._lock.release()

    async def json_chat(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        content: str = "",
        limit: int = 28000,
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


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
