"""Connector base class + concrete implementations for startup data sources."""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.configuration import get_config
from src.http_client import get_client
from src.logging import get_logger
from src.retry import RateLimiter, retry

logger = get_logger("connectors")

_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",  # noqa: E501
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36",  # noqa: E501
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15",  # noqa: E501
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",  # noqa: E501
    "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0",  # noqa: E501
]


def _searxng_query(query: str, time_range: str | None = None) -> dict[str, str]:
    params: dict[str, str] = {"q": query, "format": "json"}
    if time_range:
        params["time_range"] = time_range
    return params


@dataclass
class DiscoveredEntity:
    """What a connector discovers — a company, founder, job, etc."""

    name: str
    url: str | None = None
    description: str = ""
    source: str = "unknown"
    entity_type: str = "company"
    extra: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    discovered_at: float = field(default_factory=time.time)


class BaseConnector(ABC):
    """Abstract connector with rate limiting and retry policy."""

    source_name: str = "base"
    rate_limit_delay: float = 1.0

    def __init__(self) -> None:
        self._rate_limiter = RateLimiter(delay=self.rate_limit_delay)

    async def _fetch(self, url: str, params: dict | None = None) -> str:
        await self._rate_limiter.acquire()
        client = await get_client(f"connector_{self.source_name}", timeout=12.0)

        async def _do() -> str:
            resp = await client.get(
                url,
                params=params,
                headers={"User-Agent": random.choice(_USER_AGENTS)},
            )
            resp.raise_for_status()
            return resp.text

        try:
            return await retry(_do)
        except Exception as e:
            logger.warning(
                f"{self.source_name} connector fetch failed",
                connector=self.source_name,
                exception=str(e),
                extra={"url": url},
            )
            raise

    @abstractmethod
    async def discover(self) -> list[DiscoveredEntity]:
        """Pull new entities from this source."""
        ...

    @abstractmethod
    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity:
        """Enrich an entity with more data from this source."""
        ...

    def confidence(self, entity: DiscoveredEntity) -> float:
        return entity.confidence


# Concrete connectors


class YCConnector(BaseConnector):
    source_name = "yc"
    rate_limit_delay = get_config().rate_limit.yc

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        try:
            client = await get_client("connector_yc", timeout=10.0)
            resp = await client.get(
                "https://api.ycombinator.com/v0/companies",
                params={"batch": "W25", "limit": "50"},
                headers={"User-Agent": random.choice(_USER_AGENTS)},
            )
            if resp.status_code == 200:
                for c in resp.json().get("companies", [])[:50]:
                    entities.append(
                        DiscoveredEntity(
                            name=c.get("name", ""),
                            url=f"https://www.ycombinator.com/companies/{c.get('slug', '')}",
                            description=c.get("short_description", ""),
                            source="yc",
                            confidence=0.8,
                            extra={
                                "batch": c.get("batch", "W25"),
                                "yc_url": f"https://www.ycombinator.com/companies/{c.get('slug', '')}",  # noqa: E501
                                "team_size": c.get("team_size", 0),
                                "location": c.get("location", ""),
                                "tags": c.get("tags", []),
                            },
                        )
                    )
        except Exception as e:
            logger.warning("YC connector failed", connector="yc", exception=str(e))

        if not entities:
            try:
                cfg = get_config().searxng
                text = await self._fetch(
                    cfg.url,
                    _searxng_query('site:ycombinator.com/companies "founded" "team size"'),
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
        return entity


class ProductHuntConnector(BaseConnector):
    source_name = "producthunt"
    rate_limit_delay = get_config().rate_limit.producthunt

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        try:
            cfg = get_config().searxng
            text = await self._fetch(
                cfg.url,
                _searxng_query(
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


class GitHubConnector(BaseConnector):
    source_name = "github"
    rate_limit_delay = get_config().rate_limit.github

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        queries = [
            'site:github.com "open source" "funding" OR "backed by" startup',
            'site:github.com "we are hiring" OR "join us" "seed" OR "series a"',
        ]
        cfg = get_config().searxng
        for q in queries:
            try:
                text = await self._fetch(cfg.url, _searxng_query(q))
                data = httpx.Response(200, text=text).json()
                for r in data.get("results", [])[:10]:
                    name = r.get("title", "").split(":")[0].split("/")[-1].strip()
                    if name and len(name) > 2:
                        entities.append(
                            DiscoveredEntity(
                                name=name,
                                url=r.get("url", ""),
                                description=r.get("content", ""),
                                source="github",
                                confidence=0.3,
                            )
                        )
            except Exception:
                pass
        return entities

    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity:
        return entity


class HNConnector(BaseConnector):
    source_name = "hn"
    rate_limit_delay = get_config().rate_limit.hn

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        try:
            cfg = get_config().searxng
            text = await self._fetch(
                cfg.url,
                _searxng_query(
                    'site:news.ycombinator.com "who is hiring" startup hiring',
                    time_range="month",
                ),
            )
            data = httpx.Response(200, text=text).json()
            for r in data.get("results", [])[:15]:
                content = r.get("content", "")
                for line in content.split("|")[:3]:
                    line = line.strip()
                    if line and len(line) > 3 and "http" not in line:
                        entities.append(
                            DiscoveredEntity(
                                name=line[:80],
                                url=r.get("url", ""),
                                description=content[:200],
                                source="hn",
                                confidence=0.25,
                            )
                        )
                        break
        except Exception as e:
            logger.warning("HN connector failed", connector="hn", exception=str(e))
        return entities

    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity:
        return entity


class VCConnector(BaseConnector):
    source_name = "vc"
    rate_limit_delay = get_config().rate_limit.vc

    VC_DOMAINS = [
        ("a16z", "a16z.com"),
        ("sequoia", "sequoiacap.com"),
        ("accel", "accel.com"),
        ("benchmark", "benchmark.com"),
    ]

    async def discover(self) -> list[DiscoveredEntity]:
        cfg = get_config().searxng
        entities: list[DiscoveredEntity] = []
        for vc_name, vc_domain in self.VC_DOMAINS:
            try:
                text = await self._fetch(
                    cfg.url,
                    _searxng_query(f'site:{vc_domain} "portfolio" OR "companies" startup'),
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


class FounderSocialConnector(BaseConnector):
    source_name = "founder_social"
    rate_limit_delay = get_config().rate_limit.founder_social

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        try:
            cfg = get_config().searxng
            text = await self._fetch(
                cfg.url,
                _searxng_query(
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


# Connector registry


def all_connectors() -> list[BaseConnector]:
    return [
        YCConnector(),
        ProductHuntConnector(),
        GitHubConnector(),
        HNConnector(),
        VCConnector(),
        FounderSocialConnector(),
    ]


async def discover_all(connectors: list[BaseConnector] | None = None) -> list[DiscoveredEntity]:
    if connectors is None:
        connectors = all_connectors()
    tasks = [asyncio.create_task(c.discover()) for c in connectors]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    seen: set[str] = set()
    all_entities: list[DiscoveredEntity] = []
    for r in results:
        if isinstance(r, Exception):
            logger.exception("Connector discovery failed", exc=r)
            continue
        for e in r:
            key = e.name.lower().strip()
            if key and key not in seen:
                seen.add(key)
                all_entities.append(e)
    return all_entities
