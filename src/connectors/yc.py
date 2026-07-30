"""YCombinator connector using the official YC API with SearXNG fallback."""

from __future__ import annotations

import random
import time

import httpx

from src.configuration import get_config
from src.http_client import get_client
from src.logging import get_logger
from src.retry import retry

from .base import (
    _USER_AGENTS,
    BaseConnector,
    ConnectorCapability,
    ConnectorHealth,
    DiscoveredEntity,
    searxng_query,
)

logger = get_logger("connectors")

YC_API_BASE = "https://api.ycombinator.com/v0"


class YCConnector(BaseConnector):
    source_name = "yc"
    rate_limit_delay = get_config().rate_limit.yc

    async def capability_discovery(self) -> ConnectorCapability:
        return ConnectorCapability(
            source_name=self.source_name,
            entity_types=["company", "founder"],
            supports_enrichment=True,
            supports_incremental=True,
            max_batch_size=50,
            features={
                "api": "yc_rest_v0",
                "endpoints": ["/companies", "/launches", "/founders"],
                "rate_limit": "shared with yc.com frontend",
            },
        )

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        try:
            client = await get_client("connector_yc", timeout=10.0)

            async def _get() -> httpx.Response:
                return await client.get(
                    f"{YC_API_BASE}/companies",
                    params={"batch": "W25", "limit": "50"},
                    headers={"User-Agent": random.choice(_USER_AGENTS)},
                )

            resp = await retry(_get, max_retries=2)
            if resp.status_code == 200:
                for c in resp.json().get("companies", [])[:50]:
                    entities.append(
                        DiscoveredEntity(
                            name=c.get("name", ""),
                            url=f"https://www.ycombinator.com/companies/{c.get('slug', '')}",
                            description=c.get("short_description", ""),
                            source="yc_api",
                            confidence=0.8,
                            extra={
                                "batch": c.get("batch", "W25"),
                                "yc_url": f"https://www.ycombinator.com/companies/{c.get('slug', '')}",  # noqa: E501
                                "team_size": c.get("team_size", 0),
                                "location": c.get("location", ""),
                                "tags": c.get("tags", []),
                                "founded_at": c.get("created_at", ""),
                                "website": c.get("website", ""),
                                "long_description": c.get("long_description", ""),
                            },
                        )
                    )
        except Exception as e:
            logger.warning("YC native API failed, using SearXNG fallback", exception=str(e))

        if not entities:
            entities = await self._searxng_fallback()

        return entities

    async def _searxng_fallback(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        try:
            cfg = get_config().searxng
            text = await self._fetch(
                cfg.url,
                searxng_query('site:ycombinator.com/companies "founded" "team size"'),
            )
            data = httpx.Response(200, text=text).json()
            for r in data.get("results", [])[:20]:
                name = r.get("title", "").split("|")[0].strip()
                if name:
                    entities.append(
                        DiscoveredEntity(
                            name=name,
                            url=r.get("url", ""),
                            description=r.get("content", ""),
                            source="yc_searxng",
                            confidence=0.4,
                        )
                    )
        except Exception:
            pass
        return entities

    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity:
        yc_slug = entity.extra.get("yc_url", "").rstrip("/").split("/")[-1]
        if not yc_slug and entity.extra.get("tags"):
            yc_slug = entity.name.lower().replace(" ", "")
        if not yc_slug:
            return entity
        try:
            client = await get_client("connector_yc", timeout=10.0)

            async def _get() -> httpx.Response:
                return await client.get(
                    f"{YC_API_BASE}/companies/{yc_slug}",
                    headers={"User-Agent": random.choice(_USER_AGENTS)},
                )

            resp = await retry(_get, max_retries=1)
            if resp.status_code == 200:
                detail = resp.json()
                entity.description = detail.get("long_description", entity.description)
                entity.extra["team_size"] = detail.get("team_size", entity.extra.get("team_size"))
                entity.extra["founded_at"] = detail.get("created_at", "")
                entity.extra["website"] = detail.get("website", "")
                entity.confidence = max(entity.confidence, 0.85)
        except Exception:
            pass
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
            client = await get_client("connector_yc", timeout=5.0)
            await retry(
                lambda: client.get(
                    f"{YC_API_BASE}/companies",
                    params={"batch": "W25", "limit": "1"},
                    headers={"User-Agent": random.choice(_USER_AGENTS)},
                ),
                max_retries=1,
            )
            base.status = "healthy"
        except Exception:
            base.status = "degraded"
        base.latency_ms = (time.monotonic() - t0) * 1000
        return base
