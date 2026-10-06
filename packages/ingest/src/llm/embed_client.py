"""Unified embedding client for the configured external provider.

Replaces the previous per-call HTTP posts to the local llama-server. Every
caller (resume indexing, persona building, corpus backfill, screener RAG)
goes through this one client so the provider endpoint, model, task-type
handling, and error policy live in a single place.

The endpoint is configured via EMBED_URL / EMBED_MODEL. An empty EMBED_URL is
an error: callers that want best-effort behaviour should use ``embed_one``,
which returns None instead of raising.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.configuration import get_config
from src.logging import get_logger

logger = get_logger("embed_client")


def embed_url() -> str:
    """The configured embed endpoint (no trailing slash). Raises if unset."""
    url = (get_config().embed.url or "").rstrip("/")
    if not url:
        raise RuntimeError("EMBED_URL is not configured")
    return url


async def embed(
    inputs: list[str],
    *,
    task_type: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    client: httpx.AsyncClient | None = None,
) -> list[list[float]]:
    """Embed one or more texts. Returns one vector per input, in order.

    Raises on transport/HTTP/parse failure. Pass a shared *client* to reuse a
    connection pool across a batch loop; otherwise a client is created and
    closed per call.
    """
    cfg = get_config().embed
    url = embed_url()
    payload: dict[str, Any] = {"model": model or cfg.model, "input": inputs}
    if task_type:
        # Providers that support task-typed embeddings (Gemini, Cohere, ...)
        # use this to optimize query vs document vectors.
        payload["task_type"] = task_type

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))
    try:
        resp = await client.post(f"{url}/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()["data"]
        return [[float(v) for v in item["embedding"]] for item in data]
    finally:
        if owns_client:
            await client.aclose()


async def embed_one(
    text: str,
    *,
    task_type: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    client: httpx.AsyncClient | None = None,
) -> list[float] | None:
    """Best-effort single-text embed: returns None instead of raising."""
    try:
        vectors = await embed(
            [text], task_type=task_type, model=model, timeout=timeout, client=client
        )
        return vectors[0] if vectors else None
    except Exception as e:
        logger.warning("Embedding lookup failed", error=str(e))
        return None
