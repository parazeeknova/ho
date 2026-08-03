"""HackerNews connector using the official Firebase REST API.

Fetches "Who is Hiring" job posts directly via:
  https://hacker-news.firebaseio.com/v0/item/{id}.json

No web scraping required — structured JSON from the official HN API.
"""  # noqa: E501

from __future__ import annotations

import asyncio
import time

import httpx
from src.configuration import get_config
from src.http_client import get_client
from src.logging import get_logger
from src.retry import retry

from .base import BaseConnector, ConnectorCapability, ConnectorHealth, DiscoveredEntity

logger = get_logger("connectors")

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
WHO_IS_HIRING_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


class HackerNewsConnector(BaseConnector):
    """Fetch 'Who is Hiring' posts from HackerNews Firebase REST API."""

    source_name = "hackernews"
    rate_limit_delay = get_config().rate_limit.hn

    async def capability_discovery(self) -> ConnectorCapability:
        return ConnectorCapability(
            source_name=self.source_name,
            entity_types=["company", "job"],
            supports_enrichment=True,
            supports_incremental=True,
            max_batch_size=100,
            features={
                "api": "firebase_rest",
                "real_time": True,
                "rate_limit_info": "500 req/h for /v0/item, unlimited for /v0/topstories",
            },
        )

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        client = await get_client("connector_hn", timeout=15.0)

        try:
            who_is_hiring_id = await self._find_who_is_hiring(client)
            if who_is_hiring_id is None:
                who_is_hiring_id = await self._find_who_is_hiring_fallback(client)
            if who_is_hiring_id is None:
                return entities

            item = await self._fetch_item(client, who_is_hiring_id)
            if item is None:
                return entities

            kids = item.get("kids", [])
            top_level_limit = min(len(kids), 200)
            comments = await asyncio.gather(
                *(self._fetch_item(client, kid) for kid in kids[:top_level_limit]),
                return_exceptions=True,
            )

            for comment in comments:
                if isinstance(comment, Exception):
                    continue
                if comment is None:
                    continue
                text = (comment.get("text", "") or "").strip()
                if not text or len(text) < 50:
                    continue
                text_lower = text.lower()
                if not any(
                    kw in text_lower
                    for kw in ("hiring", "remote", "engineer", "developer", "intern")
                ):
                    continue

                entities.append(
                    DiscoveredEntity(
                        name=f"hn_job_{comment.get('id', '')}",
                        url=f"https://news.ycombinator.com/item?id={comment.get('id', '')}",
                        description=text[:300],
                        source="hackernews_api",
                        entity_type="hiring_post",
                        confidence=0.7,
                        extra={
                            "hn_id": comment.get("id"),
                            "by": comment.get("by", ""),
                            "time": comment.get("time", 0),
                            "full_text": text,
                        },
                    )
                )
        except Exception as e:
            logger.warning(
                "HackerNews native API connector failed",
                connector="hackernews",
                exception=str(e),
            )
        return entities

    async def _find_who_is_hiring(self, client: httpx.AsyncClient) -> str | None:
        try:

            async def _get_item() -> httpx.Response:
                return await client.get(f"{HN_API_BASE}/maxitem.json")

            resp = await retry(_get_item, max_retries=2)
            max_id = int(resp.text)
            check_range = 200
            for item_id in range(max_id, max(max_id - check_range, 0), -1):
                item = await self._fetch_item(client, item_id)
                if item is None:
                    continue
                title = (item.get("title", "") or "").lower()
                if "who is hiring" in title:
                    current_month = time.strftime("%B").lower()
                    if current_month in title or any(m in title for m in WHO_IS_HIRING_MONTHS):
                        return str(item_id)
            return None
        except Exception:
            return None

    async def _find_who_is_hiring_fallback(self, client: httpx.AsyncClient) -> str | None:
        try:

            async def _get_top() -> httpx.Response:
                return await client.get(f"{HN_API_BASE}/topstories.json")

            resp = await retry(_get_top, max_retries=2)
            top_ids = resp.json()
            for item_id in top_ids[:100]:
                item = await self._fetch_item(client, item_id)
                if item is None:
                    continue
                title = (item.get("title", "") or "").lower()
                if "who is hiring" in title:
                    return str(item_id)
            return None
        except Exception:
            return None

    async def _fetch_item(self, client: httpx.AsyncClient, item_id: str | int) -> dict | None:
        try:

            async def _get() -> httpx.Response:
                return await client.get(f"{HN_API_BASE}/item/{item_id}.json")

            resp = await retry(_get, max_retries=2)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity:
        hn_id = entity.extra.get("hn_id")
        if not hn_id:
            return entity
        client = await get_client("connector_hn", timeout=10.0)
        item = await self._fetch_item(client, hn_id)
        if item and item.get("text"):
            entity.description = item["text"][:500]
            entity.extra["full_text"] = item["text"]
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
            client = await get_client("connector_hn", timeout=5.0)
            await retry(lambda: client.get(f"{HN_API_BASE}/maxitem.json"), max_retries=1)
            base.status = "healthy"
        except Exception:
            base.status = "degraded"
        base.latency_ms = (time.monotonic() - t0) * 1000
        return base
