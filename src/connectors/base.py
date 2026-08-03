"""Unified plugin-based Native Connector Framework.

Every connector must inherit from BaseConnector and define:
  - capability_discovery() -> Supported schemas and features.
  - sync_incremental(checkpoint: dict) -> Incremental stateful syncing.
  - health_report() -> Latency, status, error rates.
"""  # noqa: E501

from __future__ import annotations

import contextlib
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.http_cache import cached_get
from src.http_client import get_client
from src.logging import get_logger
from src.retry import RateLimiter, retry

logger = get_logger("connectors")

# Above this error rate (over the trailing window), _fetch refuses to hit the
# network and raises immediately: an open circuit for a misbehaving provider.
CONNECTOR_ERROR_RATE_CIRCUIT_OPEN = 0.3
CONNECTOR_CIRCUIT_WINDOW = 20

# Module-level health registry so other subsystems (scheduler, health checks)
# can consume connector state without owning connector instances.
CONNECTOR_HEALTH: dict[str, ConnectorHealth] = {}


class ConnectorUnavailableError(RuntimeError):
    """Raised when a connector's circuit breaker is open (high error rate)."""


_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",  # noqa: E501
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36",  # noqa: E501
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15",  # noqa: E501
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",  # noqa: E501
    "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0",  # noqa: E501
]


@dataclass
class ConnectorCapability:
    source_name: str
    entity_types: list[str]
    supports_enrichment: bool = True
    supports_incremental: bool = False
    max_batch_size: int = 50
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorHealth:
    source_name: str
    status: str = "unknown"
    latency_ms: float = 0.0
    error_rate: float = 0.0
    last_success: datetime | None = None
    last_error: str | None = None
    requests_total: int = 0
    requests_failed: int = 0
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredEntity:
    name: str
    url: str | None = None
    description: str = ""
    source: str = "unknown"
    entity_type: str = "company"
    extra: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    discovered_at: float = field(default_factory=time.time)


@dataclass
class SyncCheckpoint:
    cursor: str | None = None
    last_synced_at: float = 0.0
    items_processed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    source_name: str = "base"
    rate_limit_delay: float = 1.0
    _health_counter: int = 0
    _health_failures: int = 0

    def __init__(self) -> None:
        self._rate_limiter = RateLimiter(delay=self.rate_limit_delay)
        self._last_success: datetime | None = None
        self._last_error: str | None = None
        self._latency_samples: list[float] = []

    async def _fetch(self, url: str, params: dict | None = None) -> str:
        if self._circuit_open():
            self._health_counter += 1
            self._health_failures += 1
            raise ConnectorUnavailableError(
                f"{self.source_name} connector circuit open "
                f"(error rate {self._error_rate():.0%} over last "
                f"{CONNECTOR_CIRCUIT_WINDOW} requests)"
            )
        await self._rate_limiter.acquire()
        client = await get_client(f"connector_{self.source_name}", timeout=12.0)

        async def _do() -> str:
            resp = await cached_get(
                client,
                url,
                params=params,
                headers={"User-Agent": random.choice(_USER_AGENTS)},
            )
            resp.raise_for_status()
            return resp.text

        try:
            t0 = time.monotonic()
            result = await retry(_do)
            elapsed = time.monotonic() - t0
            self._latency_samples.append(elapsed)
            self._health_counter += 1
            self._last_success = datetime.now(UTC)
            self._update_health_registry()
            return result
        except Exception as e:
            self._health_counter += 1
            self._health_failures += 1
            self._last_error = str(e)
            self._update_health_registry()
            logger.warning(
                f"{self.source_name} connector fetch failed",
                connector=self.source_name,
                exception=str(e),
                extra={"url": url},
            )
            raise

    def _circuit_open(self) -> bool:
        return self._error_rate() >= CONNECTOR_ERROR_RATE_CIRCUIT_OPEN

    def _error_rate(self) -> float:
        if self._health_counter <= 0:
            return 0.0
        return self._health_failures / self._health_counter

    def _update_health_registry(self) -> None:
        with contextlib.suppress(Exception):
            CONNECTOR_HEALTH[self.source_name] = self._health_snapshot()

    # --- ABSTRACT INTERFACE ---

    @abstractmethod
    async def discover(self) -> list[DiscoveredEntity]: ...

    @abstractmethod
    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity: ...

    # --- NEW PLUGIN METHODS ---

    async def capability_discovery(self) -> ConnectorCapability:
        return ConnectorCapability(
            source_name=self.source_name,
            entity_types=["company"],
        )

    async def sync_incremental(
        self, checkpoint: dict | None = None
    ) -> tuple[list[DiscoveredEntity], dict]:
        entities = await self.discover()
        next_checkpoint = {
            "cursor": str(int(time.time())),
            "last_synced_at": time.time(),
            "items_processed": len(entities),
        }
        return entities, next_checkpoint

    def _health_snapshot(self) -> ConnectorHealth:
        recent_latency = 0.0
        if self._latency_samples:
            recent = self._latency_samples[-10:]
            recent_latency = sum(recent) / len(recent) * 1000.0
        error_rate = self._error_rate()
        return ConnectorHealth(
            source_name=self.source_name,
            status="healthy" if error_rate < 0.1 else "degraded",
            latency_ms=round(recent_latency, 1),
            error_rate=round(error_rate, 4),
            last_success=self._last_success,
            last_error=self._last_error,
            requests_total=self._health_counter,
            requests_failed=self._health_failures,
        )

    async def health_report(self) -> ConnectorHealth:
        return self._health_snapshot()

    def confidence(self, entity: DiscoveredEntity) -> float:
        return entity.confidence


def searxng_query(query: str, time_range: str | None = None) -> dict[str, str]:
    params: dict[str, str] = {"q": query, "format": "json"}
    if time_range:
        params["time_range"] = time_range
    return params
