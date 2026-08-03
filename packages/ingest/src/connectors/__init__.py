"""Unified plugin-based Native Connector Framework.

Every connector must inherit from BaseConnector and define:
  - capability_discovery() -> Returns supported schemas and features.
  - sync_incremental(checkpoint: dict) -> Incremental stateful syncing.
  - health_report() -> Returns latency, status, error rates.

Single-Class Registration: Adding a new connector only requires
creating a single adapter class inheriting from BaseConnector and
adding it to all_connectors() in this module.
"""  # noqa: E501

from __future__ import annotations

import asyncio

from src.logging import get_logger

from .base import (
    BaseConnector,
    ConnectorCapability,
    ConnectorHealth,
    DiscoveredEntity,
    SyncCheckpoint,
)
from .github import GitHubConnector
from .hackernews import HackerNewsConnector
from .producthunt import ProductHuntConnector
from .vc_founder import FounderSocialConnector, VCConnector
from .yc import YCConnector

logger = get_logger("connectors")

__all__ = [
    "BaseConnector",
    "ConnectorCapability",
    "ConnectorHealth",
    "DiscoveredEntity",
    "SyncCheckpoint",
    "GitHubConnector",
    "HackerNewsConnector",
    "ProductHuntConnector",
    "YCConnector",
    "VCConnector",
    "FounderSocialConnector",
    "all_connectors",
    "discover_all",
    "health_check_all",
]


def all_connectors() -> list[BaseConnector]:
    """Registry of all native connectors.

    Add a new connector here (single line) to register it.
    """
    return [
        YCConnector(),
        ProductHuntConnector(),
        GitHubConnector(),
        HackerNewsConnector(),
        VCConnector(),
        FounderSocialConnector(),
    ]


async def discover_all(
    connectors: list[BaseConnector] | None = None,
) -> list[DiscoveredEntity]:
    if connectors is None:
        connectors = all_connectors()
    tasks = [asyncio.create_task(c.discover()) for c in connectors]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    seen: set[str] = set()
    all_entities: list[DiscoveredEntity] = []
    for r in results:
        if isinstance(r, BaseException):
            logger.exception("Connector discovery failed", exc=r)
            continue
        for e in r:
            key = e.name.lower().strip()
            if key and key not in seen:
                seen.add(key)
                all_entities.append(e)
    return all_entities


async def health_check_all(
    connectors: list[BaseConnector] | None = None,
) -> dict[str, ConnectorHealth]:
    if connectors is None:
        connectors = all_connectors()
    tasks = [asyncio.create_task(c.health_report()) for c in connectors]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    health_map: dict[str, ConnectorHealth] = {}
    for c, r in zip(connectors, results, strict=True):
        if isinstance(r, Exception):
            health_map[c.source_name] = ConnectorHealth(
                source_name=c.source_name,
                status="error",
                last_error=str(r),
            )
        elif isinstance(r, ConnectorHealth):
            health_map[c.source_name] = r
    return health_map
