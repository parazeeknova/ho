"""Venture Capital portfolio connectors (a16z, Sequoia, Accel, Benchmark)."""  # noqa: E501

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

VC_DOMAINS = [
    ("a16z", "a16z.com"),
    ("sequoia", "sequoiacap.com"),
    ("accel", "accel.com"),
    ("benchmark", "benchmark.com"),
]


class VCConnector(BaseConnector):
    source_name = "vc"
    rate_limit_delay = get_config().rate_limit.vc

    async def capability_discovery(self) -> ConnectorCapability:
        return ConnectorCapability(
            source_name=self.source_name,
            entity_types=["company", "investor"],
            supports_enrichment=False,
            max_batch_size=32,
            features={
                "vc_firms": [name for name, _ in VC_DOMAINS],
                "search_method": "searxng_metasearch",
            },
        )

    async def discover(self) -> list[DiscoveredEntity]:
        cfg = get_config().searxng
        entities: list[DiscoveredEntity] = []
        for vc_name, vc_domain in VC_DOMAINS:
            try:
                text = await self._fetch(
                    cfg.url,
                    searxng_query(f'site:{vc_domain} "portfolio" OR "companies" startup'),
                )
                data = httpx.Response(200, text=text).json()
                for r in data.get("results", [])[:8]:
                    name = r.get("title", "").split("|")[0].strip()
                    if name and len(name) > 2 and vc_domain not in name.lower():
                        entities.append(
                            DiscoveredEntity(
                                name=name,
                                url=r.get("url", ""),
                                description=r.get("content", ""),
                                source=f"vc_{vc_name}",
                                confidence=0.3,
                                extra={"vc": vc_name},
                            )
                        )
            except Exception:
                pass
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
        return base


class FounderSocialConnector(BaseConnector):
    source_name = "founder_social"
    rate_limit_delay = get_config().rate_limit.founder_social

    async def capability_discovery(self) -> ConnectorCapability:
        return ConnectorCapability(
            source_name=self.source_name,
            entity_types=["company", "hiring_post"],
            supports_enrichment=False,
            max_batch_size=15,
            features={
                "signals": ["hiring", "founder_post", "stealth"],
                "search_method": "searxng_metasearch",
            },
        )

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        try:
            cfg = get_config().searxng
            text = await self._fetch(
                cfg.url,
                searxng_query(
                    '("hiring" OR "looking for" OR "join us") ("founder" OR "CEO" OR "CTO") '
                    '("seed" OR "series a" OR "pre-seed" OR "stealth") startup',
                    time_range="week",
                ),
            )
            data = httpx.Response(200, text=text).json()
            for r in data.get("results", [])[:15]:
                title = r.get("title", "")
                content = r.get("content", "")
                name = title.split("|")[0].strip()[:80]
                if not name or len(name) < 3:
                    name = title.split("-")[0].strip()[:80]
                if name and len(name) > 2:
                    entities.append(
                        DiscoveredEntity(
                            name=name,
                            url=r.get("url", ""),
                            description=content[:200],
                            source="founder_social",
                            confidence=0.3,
                            extra={"hiring_signal": True},
                        )
                    )
        except Exception as e:
            logger.warning(
                "FounderSocial connector failed",
                connector="founder_social",
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
        return base
