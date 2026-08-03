"""Tests for the shared HTTP response cache (cached_get)."""

from __future__ import annotations

import time

import httpx
import pytest
import src.http_cache as hc
from src.configuration import get_config

HEADERS_JSON = {"content-type": "application/json"}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    hc._hits = 0
    hc._misses = 0
    hc._bytes_saved = 0
    hc._bypassed = 0
    monkeypatch.setattr(hc, "_CACHE_STORE", None)
    cfg = get_config().http
    monkeypatch.setattr(cfg, "cache_enabled", True)
    monkeypatch.setattr(cfg, "cache_ttl_default", 900)


class _FakeStore:
    """In-memory stand-in for the Postgres http_cache row methods."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.fetched_updates: list[tuple] = []

    async def get_http_cache_row(self, url_hash: str) -> dict | None:
        return self.rows.get(url_hash)

    async def upsert_http_cache(self, *args) -> None:
        (
            url_hash,
            url,
            status,
            etag,
            last_modified,
            content_type,
            body,
            body_hash,
            ttl,
        ) = args
        self.rows[url_hash] = {
            "url_hash": url_hash,
            "url": url,
            "status": status,
            "etag": etag,
            "last_modified": last_modified,
            "content_type": content_type,
            "body": body,
            "body_hash": body_hash,
            "fetched_at": time.time(),
            "ttl_seconds": ttl,
        }

    async def update_http_cache_fetched(self, url_hash: str, etag: str, lm: str) -> None:
        self.fetched_updates.append((url_hash, etag, lm))


def _client(responses: list[httpx.Response]) -> httpx.AsyncClient:
    async def _handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


async def test_miss_fetches_and_stores() -> None:
    store = _FakeStore()
    hc.set_http_cache_store(store)
    client = _client([httpx.Response(200, text='{"jobs": []}', headers=HEADERS_JSON)])
    resp = await hc.cached_get(client, "https://example.com/api")
    assert resp.status_code == 200
    assert resp.json() == {"jobs": []}
    assert resp.extensions.get("cached") is None
    assert len(store.rows) == 1
    assert hc.get_http_cache_stats()["misses"] == 1


async def test_304_returns_cached_body() -> None:
    store = _FakeStore()
    hc.set_http_cache_store(store)
    client = _client(
        [
            httpx.Response(200, text="first", headers={"etag": '"v1"', **HEADERS_JSON}),
            httpx.Response(304),
        ]
    )
    r1 = await hc.cached_get(client, "https://example.com/a")
    r2 = await hc.cached_get(client, "https://example.com/a")
    assert r1.text == "first"
    assert r2.text == "first"
    assert r2.status_code == 200
    assert r2.extensions["cached"] is True
    assert hc.get_http_cache_stats()["hits"] == 1
    assert store.rows  # body survived for replay


async def test_unchanged_body_hash_marks_cached() -> None:
    store = _FakeStore()
    hc.set_http_cache_store(store)
    client = _client(
        [
            httpx.Response(200, text="same", headers={"etag": '"v1"', **HEADERS_JSON}),
            # server ignores conditional headers, returns 200 with same body
            httpx.Response(200, text="same", headers={"etag": '"v2"', **HEADERS_JSON}),
        ]
    )
    r1 = await hc.cached_get(client, "https://example.com/b")  # noqa: F841
    r2 = await hc.cached_get(client, "https://example.com/b")
    assert r2.extensions["cached"] is True
    assert store.fetched_updates and store.fetched_updates[0][1] == '"v2"'


async def test_changed_body_is_refreshed() -> None:
    store = _FakeStore()
    hc.set_http_cache_store(store)
    client = _client(
        [
            httpx.Response(200, text="old", headers={"etag": '"v1"', **HEADERS_JSON}),
            httpx.Response(200, text="new", headers={"etag": '"v2"', **HEADERS_JSON}),
        ]
    )
    r1 = await hc.cached_get(client, "https://example.com/c")  # noqa: F841
    r2 = await hc.cached_get(client, "https://example.com/c")
    assert r2.text == "new"
    assert r2.extensions.get("cached") is None
    assert store.rows[next(iter(store.rows))]["body"] == "new"


async def test_auth_requests_never_cached() -> None:
    store = _FakeStore()
    hc.set_http_cache_store(store)
    client = _client([httpx.Response(200, text='{"ok": true}', headers=HEADERS_JSON)])
    resp = await hc.cached_get(
        client, "https://example.com/private", headers={"Authorization": "Bearer tok"}
    )
    assert resp.status_code == 200
    assert not store.rows
    assert hc.get_http_cache_stats()["bypassed"] == 1


async def test_no_store_degrades_to_plain_get() -> None:
    client = _client([httpx.Response(200, text="plain", headers=HEADERS_JSON)])
    resp = await hc.cached_get(client, "https://example.com/d")
    assert resp.text == "plain"
    assert resp.extensions.get("cached") is None


async def test_non_200_not_stored() -> None:
    store = _FakeStore()
    hc.set_http_cache_store(store)
    client = _client([httpx.Response(500, text="boom")])
    resp = await hc.cached_get(client, "https://example.com/e")
    assert resp.status_code == 500
    assert not store.rows


async def test_cached_get_retries_via_retry_helper() -> None:
    """cached_get returns real responses so raise_for_status/retry still work."""
    store = _FakeStore()
    hc.set_http_cache_store(store)
    client = _client([httpx.Response(200, text="ok", headers=HEADERS_JSON)])
    resp = await hc.cached_get(client, "https://example.com/f")
    assert resp.raise_for_status() == resp  # httpx.Response contract preserved
