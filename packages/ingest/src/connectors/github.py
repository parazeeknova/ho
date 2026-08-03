"""GitHub connector using the official REST/GraphQL API.

Fetches active organization repos, metadata, and topic tags.
Supports token-based auth via GITHUB_TOKEN env var.
"""  # noqa: E501

from __future__ import annotations

from datetime import UTC, datetime

from src.configuration import get_config
from src.http_client import get_client
from src.logging import get_logger
from src.retry import retry

from .base import BaseConnector, ConnectorCapability, ConnectorHealth, DiscoveredEntity

logger = get_logger("connectors")

GITHUB_API_BASE = "https://api.github.com"

_STARTUP_SEARCH_TOPICS = [
    "hiring",
    "jobs",
    "careers",
    "startup",
    "seed-stage",
    "series-a",
]

_STARTUP_SEARCH_QUERIES = [
    '"we are hiring" startup',
    '"join our team" backend frontend',
    '"engineering team" seed series-a',
]


class GitHubConnector(BaseConnector):
    """Fetch active organization repos and metadata via GitHub REST/GraphQL API."""

    source_name = "github"
    rate_limit_delay = get_config().rate_limit.github

    def __init__(self) -> None:
        super().__init__()
        self._token = ""
        try:
            cfg = get_config()
            self._token = getattr(cfg, "github_token", "")
        except Exception:
            pass

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def capability_discovery(self) -> ConnectorCapability:
        return ConnectorCapability(
            source_name=self.source_name,
            entity_types=["company", "technology", "repository"],
            supports_enrichment=True,
            supports_incremental=True,
            max_batch_size=30,
            features={
                "api": "github_rest",
                "requires_token": False,
                "rate_limit": "60 req/h unauthenticated, 5000 req/h with token",
                "graphql_supported": bool(self._token),
            },
        )

    async def discover(self) -> list[DiscoveredEntity]:
        entities: list[DiscoveredEntity] = []
        client = await get_client("connector_github", timeout=15.0)

        try:
            await self._discover_trending_repos(client, entities)
        except Exception as e:
            logger.warning("GitHub trending discovery failed", exception=str(e))

        try:
            await self._discover_search_results(client, entities)
        except Exception as e:
            logger.warning("GitHub search discovery failed", exception=str(e))

        return entities

    async def _discover_trending_repos(self, client, entities: list[DiscoveredEntity]) -> None:
        recent = datetime.now(UTC).strftime("%Y-%m-%d")

        async def _get() -> dict:
            resp = await client.get(
                f"{GITHUB_API_BASE}/search/repositories",
                params={
                    "q": f"hiring OR careers created:>{recent}",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 20,
                },
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

        data = await retry(_get, max_retries=2)
        for repo in data.get("items", [])[:20]:
            owner_data = repo.get("owner", {})
            org_name = (
                owner_data.get("login", "") if owner_data.get("type") == "Organization" else ""
            )
            repo_name = repo.get("name", "")
            name = org_name or repo_name
            if not name or len(name) < 2:
                continue
            topics = repo.get("topics", [])
            entities.append(
                DiscoveredEntity(
                    name=name,
                    url=repo.get("html_url", ""),
                    description=repo.get("description", "") or "",
                    source="github_api",
                    confidence=0.35,
                    extra={
                        "repo_full_name": repo.get("full_name", ""),
                        "stars": repo.get("stargazers_count", 0),
                        "language": repo.get("language", ""),
                        "topics": topics,
                        "org_type": owner_data.get("type", ""),
                    },
                )
            )

    async def _discover_search_results(self, client, entities: list[DiscoveredEntity]) -> None:
        for query in _STARTUP_SEARCH_QUERIES:
            try:

                async def _get(q: str = query) -> dict:
                    resp = await client.get(
                        f"{GITHUB_API_BASE}/search/repositories",
                        params={
                            "q": q,
                            "sort": "updated",
                            "order": "desc",
                            "per_page": 10,
                        },
                        headers=self._headers(),
                    )
                    resp.raise_for_status()
                    return resp.json()

                data = await retry(_get, max_retries=1)
                for repo in data.get("items", [])[:10]:
                    owner = repo.get("owner", {})
                    name = owner.get("login", "")
                    if not name or len(name) < 2:
                        continue
                    topics = repo.get("topics", [])
                    entities.append(
                        DiscoveredEntity(
                            name=name,
                            url=f"https://github.com/{name}",
                            description=repo.get("description", "") or "",
                            source="github_api",
                            confidence=0.3,
                            extra={
                                "repo_full_name": repo.get("full_name", ""),
                                "stars": repo.get("stargazers_count", 0),
                                "topics": topics,
                                "query": query,
                            },
                        )
                    )
            except Exception:
                pass

    async def _get_org_repos(self, client, org_name: str) -> list[dict]:
        try:

            async def _get() -> list[dict]:
                resp = await client.get(
                    f"{GITHUB_API_BASE}/orgs/{org_name}/repos",
                    params={"sort": "pushed", "per_page": 10, "type": "public"},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()

            return await retry(_get, max_retries=1)
        except Exception:
            return []

    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity:
        repo_name = entity.extra.get("repo_full_name", "") or entity.name
        if not repo_name or "/" not in repo_name:
            org_name = entity.name
        else:
            org_name = repo_name.split("/")[0]
            entity.extra["repo_full_name"] = repo_name

        if entity.extra.get("org_type") == "Organization" or not repo_name:
            client = await get_client("connector_github", timeout=15.0)
            repos = await self._get_org_repos(client, org_name)
            if repos:
                topics_all: list[str] = []
                total_stars = 0
                for r in repos[:10]:
                    topics_all.extend(r.get("topics", []))
                    total_stars += r.get("stargazers_count", 0)
                entity.extra["repo_count"] = max(entity.extra.get("repo_count", 0), len(repos))
                entity.extra["total_stars"] = total_stars
                entity.extra["topics"] = list(set(entity.extra.get("topics", []) + topics_all))
                tags = entity.extra.get("topics", [])
                if any(t in tags for t in ("hiring", "jobs", "careers")):
                    entity.confidence = max(entity.confidence, 0.55)

        return entity

    async def sync_incremental(
        self, checkpoint: dict | None = None
    ) -> tuple[list[DiscoveredEntity], dict]:
        import time

        entities = await self.discover()
        next_checkpoint = {
            "cursor": str(int(time.time())),
            "last_synced_at": time.time(),
            "items_processed": len(entities),
        }
        return entities, next_checkpoint

    async def health_report(self) -> ConnectorHealth:
        base = await super().health_report()
        client = await get_client("connector_github", timeout=5.0)
        try:
            resp = await client.get(f"{GITHUB_API_BASE}/rate_limit", headers=self._headers())
            if resp.status_code == 200:
                rate = resp.json().get("rate", {})
                base.features = {
                    "remaining": rate.get("remaining", "?"),
                    "limit": rate.get("limit", "?"),
                }
                base.status = "healthy"
            else:
                base.status = "degraded"
        except Exception:
            base.status = "unreachable"
        return base
