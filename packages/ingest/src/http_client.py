"""Centralized HTTP client manager.

Reuses persistent httpx.AsyncClient instances with connection pooling,
keep-alive, sensible limits, and clean shutdown. No connector should
manually construct HTTP clients.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from src.configuration import HttpConfig, get_config

_CLIENTS: dict[str, httpx.AsyncClient] = {}
_CLIENTS_LOCK = asyncio.Lock()

_client_config: HttpConfig | None = None
_global_client: httpx.AsyncClient | None = None


def _make_client(
    timeout: float | None = None,
    extra_limits: dict[str, Any] | None = None,
    follow_redirects: bool = True,
) -> httpx.AsyncClient:
    cfg = _client_config or get_config().http
    t = timeout if timeout is not None else cfg.default_timeout
    limits_kw = {
        "max_keepalive_connections": cfg.max_keepalive,
        "max_connections": cfg.max_connections,
        **(extra_limits or {}),
    }
    return httpx.AsyncClient(
        timeout=httpx.Timeout(t, connect=cfg.connect_timeout),
        limits=httpx.Limits(**limits_kw),
        follow_redirects=follow_redirects,
    )


async def get_client(
    name: str = "default",
    timeout: float | None = None,
    extra_limits: dict[str, Any] | None = None,
    follow_redirects: bool = True,
) -> httpx.AsyncClient:
    """Return a named (cached) httpx.AsyncClient. Create it if missing.

    Callers should NOT close this client; use ``close_all()`` at shutdown.
    """
    global _client_config
    if _client_config is None:
        _client_config = get_config().http

    if name in _CLIENTS:
        return _CLIENTS[name]

    async with _CLIENTS_LOCK:
        if name in _CLIENTS:
            return _CLIENTS[name]
        client = _make_client(
            timeout=timeout, extra_limits=extra_limits, follow_redirects=follow_redirects
        )
        _CLIENTS[name] = client
        return client


async def close_all() -> None:
    """Gracefully close all managed httpx clients."""
    async with _CLIENTS_LOCK:
        names = list(_CLIENTS.keys())
        for name in names:
            client = _CLIENTS.pop(name, None)
            if client is not None:
                await client.aclose()


class HttpClientManager:
    """Async context manager for a local-scoped HTTP client.

    Usage::

        async with HttpClientManager(timeout=10.0) as client:
            resp = await client.get(url)
    """

    def __init__(
        self,
        timeout: float | None = None,
        extra_limits: dict[str, Any] | None = None,
        follow_redirects: bool = True,
    ) -> None:
        self._timeout = timeout
        self._extra_limits = extra_limits
        self._follow_redirects = follow_redirects
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> httpx.AsyncClient:
        self._client = _make_client(
            timeout=self._timeout,
            extra_limits=self._extra_limits,
            follow_redirects=self._follow_redirects,
        )
        return self._client

    async def __aexit__(self, *args: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
