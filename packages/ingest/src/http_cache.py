"""Shared HTTP response cache with ETag/304 conditional requests.

Backed by the ``http_cache`` Postgres table so hits survive restarts and are
shared across master/worker processes. Flow:

1. Cache hit (fresh row): issue a conditional GET with If-None-Match /
   If-Modified-Since. On 304 the cached body is returned with
   ``extensions["cached"] = True`` (a synthetic 200 httpx.Response, so
   existing ``resp.json()`` / ``resp.text`` / ``raise_for_status()`` call
   sites work unchanged).
2. 200 with an unchanged body hash: refresh the row's timestamp/etag and
   mark the response cached (no re-parse needed by callers that opt in).
3. Miss or expired row: plain GET, stored only when 2xx, small enough, and
   the request carried no Authorization header (credentials are never
   written to the cache).

The store is registered once at startup via ``set_http_cache_store``;
without it, ``cached_get`` degrades to a plain GET.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from typing import Any

import httpx
from src.configuration import get_config
from src.logging import get_logger

logger = get_logger("http_cache")

_CACHE_STORE: Any | None = None

_hits = 0
_misses = 0
_bytes_saved = 0
_bypassed = 0  # auth'd or non-cacheable requests (never cached)


def set_http_cache_store(store: Any) -> None:
    """Register the Postgres-backed store used for the cache.

    Call once at startup, right after the MemoryStore pool is created.
    """
    global _CACHE_STORE
    _CACHE_STORE = store


def get_http_cache_stats() -> dict[str, Any]:
    return {
        "hits": _hits,
        "misses": _misses,
        "bytes_saved": _bytes_saved,
        "bypassed": _bypassed,
    }


def _cache_key(method: str, url: str, params: dict | None) -> str:
    key = method + "|" + url
    if params:
        key += "|" + repr(sorted(params.items()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _body_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _has_auth(headers: dict | None) -> bool:
    if not headers:
        return False
    lowered = {k.lower() for k in headers}
    return "authorization" in lowered or "cookie" in lowered


async def _maybe_store(store: Any, url_hash: str, url: str, resp: httpx.Response, ttl: int) -> None:
    cfg = get_config().http
    if resp.status_code != 200:
        return
    body = resp.text
    if len(body.encode("utf-8")) > cfg.cache_max_body_bytes:
        return
    try:
        await store.upsert_http_cache(
            url_hash,
            url,
            200,
            resp.headers.get("etag", ""),
            resp.headers.get("last-modified", ""),
            resp.headers.get("content-type", ""),
            body,
            _body_hash(body),
            ttl,
        )
    except Exception:
        logger.debug("http_cache store write failed (best-effort)")


async def cached_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    ttl_seconds: int | None = None,
) -> httpx.Response:
    """GET *url* through the shared response cache.

    Returns an httpx.Response; hits carry ``extensions["cached"] = True``.
    Degrades to a plain GET when caching is disabled, no store is
    registered, or the request carries auth headers.
    """
    global _hits, _misses, _bytes_saved, _bypassed
    cfg = get_config().http
    store = _CACHE_STORE
    if _has_auth(headers):
        _bypassed += 1
        return await client.get(url, params=params, headers=headers)
    if not cfg.cache_enabled or store is None:
        _misses += 1
        return await client.get(url, params=params, headers=headers)

    ttl = int(ttl_seconds or cfg.cache_ttl_default)
    url_hash = _cache_key("GET", url, params)
    now = time.time()

    row = await store.get_http_cache_row(url_hash)
    if row is None or now - float(row.get("fetched_at", 0) or 0) >= ttl:
        _misses += 1
        resp = await client.get(url, params=params, headers=headers)
        await _maybe_store(store, url_hash, url, resp, ttl)
        return resp

    cond = dict(headers or {})
    if row.get("etag"):
        cond["If-None-Match"] = row["etag"]
    if row.get("last_modified"):
        cond["If-Modified-Since"] = row["last_modified"]
    resp = await client.get(url, params=params, headers=cond)

    if resp.status_code == 304:
        _hits += 1
        body = row.get("body", "")
        _bytes_saved += len(body.encode("utf-8"))
        return httpx.Response(
            200,
            text=body,
            headers={
                "content-type": row.get("content_type") or "application/json",
                "etag": row.get("etag", ""),
            },
            extensions={"cached": True},
        )
    if resp.status_code == 200:
        body_hash = _body_hash(resp.text)
        if body_hash == row.get("body_hash"):
            _hits += 1
            _bytes_saved += len(resp.text.encode("utf-8"))
            with contextlib.suppress(Exception):
                await store.update_http_cache_fetched(
                    url_hash,
                    resp.headers.get("etag", ""),
                    resp.headers.get("last-modified", ""),
                )
            return httpx.Response(
                200,
                text=resp.text,
                headers=resp.headers,
                extensions={"cached": True},
            )
        _misses += 1
        await _maybe_store(store, url_hash, url, resp, ttl)
    else:
        _misses += 1
    return resp
