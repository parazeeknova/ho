"""ProductHunt connector with RSS feed parsing and rate-limit wrappers."""

from __future__ import annotations

import time

import httpx

from src.configuration import get_config
from src.logging import get_logger

from .base import (
    BaseConnector,
    ConnectorCapability,
    ConnectorHealth,
    DiscoveredEntity,
    searxng_query,
)

logger = get_logger("connectors")


class ProductHuntConnector(BaseConnector):
    source_name = "producthunt"
    rate_limit_delay = get_config().rate_limit.producthunt

    async def capability_discovery(self) -> ConnectorCapability:
        return ConnectorCapability(
            source_name=self.source_name,
            entity_types=["company"],
            supports_enrichment=False,
            max_batch_size=15,
            features={
                "api": "searxng_metasearch",
                "time_scoped": True,
                "search_sources": ["producthunt.com"],
            },
        )

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        try:
            cfg = get_config().searxng
            text = await self._fetch(
                cfg.url,
                searxng_query(
                    'site:producthunt.com "launched" OR "maker" "upvotes"',
                    time_range="month",
                ),
            )
            data = httpx.Response(200, text=text).json()
            for r in data.get("results", [])[:15]:
                name = r.get("title", "").split("|")[0].strip()
                if name and "product hunt" not in name.lower():
                    entities.append(
                        DiscoveredEntity(
                            name=name,
                            url=r.get("url", ""),
                            description=r.get("content", ""),
                            source="producthunt",
                            confidence=0.35,
                        )
                    )
        except Exception as e:
            logger.warning(
                "ProductHunt connector failed",
                connector="producthunt",
                exception=str(e),
            )
        return entities

    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity:
        return entity

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

    async def health_report(self) -> ConnectorHealth:
        base = await super().health_report()
        t0 = time.monotonic()
        try:
            cfg = get_config().searxng
            await self._fetch(cfg.url, searxng_query("producthunt", time_range="week"))
            base.status = "healthy"
        except Exception:
            base.status = "degraded"
        base.latency_ms = (time.monotonic() - t0) * 1000
        return base
