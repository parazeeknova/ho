"""Entity models: graph, frontier entries with leasing, cost, utility, batching.

Deterministic ID generation for deduplication. Lease heartbeats for
long-running tasks. Utility scoring for information gain/cost tradeoffs.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    COMPANY = "company"
    FOUNDER = "founder"
    EMPLOYEE = "employee"
    INVESTOR = "investor"
    CAREER_SITE = "career_site"
    ATS = "ats"
    HIRING_POST = "hiring_post"
    FUNDING_ROUND = "funding_round"
    JOB = "job"
    TECHNOLOGY = "technology"


class EdgeType(StrEnum):
    FOUNDED_BY = "founded_by"
    WORKS_AT = "works_at"
    USES_ATS = "uses_ats"
    INVESTED_BY = "invested_by"
    HIRED_FOR = "hired_for"
    POSTED_JOB = "posted_job"
    USES_TECH = "uses_tech"
    DISCOVERED_FROM = "discovered_from"
    HAS_CAREER_SITE = "has_career_site"
    HAS_FUNDING = "has_funding"


class WorkState(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


# Confidence


class Confidence(BaseModel):
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_count: int = 0
    last_verified: datetime = Field(default_factory=lambda: datetime.now(UTC))
    first_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verification_method: str = "heuristic"


def merge_confidence(old: Confidence, new: Confidence, source_bonus: int = 1) -> Confidence:
    old.source_count += source_bonus
    old.last_verified = max(old.last_verified, new.last_verified)
    old.first_seen = min(old.first_seen, new.first_seen)
    if new.verification_method != "heuristic":
        old.verification_method = new.verification_method
    raw = old.score * old.source_count + new.score * source_bonus
    old.score = min(1.0, max(0.1, raw / max(1, old.source_count + source_bonus)))
    return old


def confidence_decay(c: Confidence, max_age_days: int = 30) -> Confidence:
    age = (datetime.now(UTC) - c.last_verified).days
    if age > max_age_days:
        c.score = max(0.1, c.score * (1.0 - (age - max_age_days) / 60.0))
    return c


# Graph


class GraphNode(BaseModel):
    id: str
    node_type: NodeType
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: Confidence = Field(default_factory=Confidence)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    active: bool = True

    @property
    def name(self) -> str:
        return str(self.data.get("name") or self.data.get("company") or self.id[:12])


class GraphEdge(BaseModel):
    source_id: str
    edge_type: EdgeType
    target_id: str
    confidence: Confidence = Field(default_factory=Confidence)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Ids


def _hash(*parts: str) -> str:
    return hashlib.md5(":".join(parts).encode()).hexdigest()[:12]


def make_work_id(agent: str, node_id: str, depth: int = 0) -> str:
    return _hash(f"work:{agent}:{node_id}:{depth}")


def make_event_id(event_type: str, node_id: str) -> str:
    return _hash(f"event:{event_type}:{node_id}")


def make_company_id(name: str) -> str:
    return _hash(f"company:{name.lower().strip()}")


def make_founder_id(name: str, company_name: str) -> str:
    return _hash(f"founder:{name.lower().strip()}:{company_name.lower().strip()}")


# Cost


@dataclass
class CostEstimate:
    llm_calls: float = 0.0
    http_requests: float = 0.0
    searxng_queries: float = 0.0
    firecrawl_scrapes: float = 0.0
    embedding_calls: float = 0.0
    db_writes: float = 0.0

    @property
    def total_apx(self) -> float:
        return (
            self.llm_calls * 5.0
            + self.http_requests * 0.5
            + self.searxng_queries * 0.3
            + self.firecrawl_scrapes * 2.0
            + self.embedding_calls * 0.1
            + self.db_writes * 0.01
        )


AGENT_COSTS: dict[str, CostEstimate] = {
    "founder_miner": CostEstimate(searxng_queries=3, llm_calls=1, db_writes=3),
    "career_site_detector": CostEstimate(http_requests=1, db_writes=1),
    "founder_social_osint": CostEstimate(searxng_queries=1, llm_calls=1),
    "employee_discovery": CostEstimate(searxng_queries=1, http_requests=1),
    "ats_crawler": CostEstimate(firecrawl_scrapes=5, db_writes=5),
    "funding_agent": CostEstimate(searxng_queries=2, llm_calls=1, db_writes=2),
    "tech_stack_agent": CostEstimate(http_requests=1, llm_calls=1),
    "graph_maintenance": CostEstimate(db_writes=1),
}

AGENT_CONCURRENCY: dict[str, int] = {
    "founder_miner": 3,
    "career_site_detector": 5,
    "founder_social_osint": 2,
    "employee_discovery": 3,
    "ats_crawler": 2,
    "funding_agent": 2,
    "tech_stack_agent": 3,
    "graph_maintenance": 5,
}

AGENT_BATCHABLE: set[str] = {
    "founder_miner",
    "career_site_detector",
    "employee_discovery",
    "graph_maintenance",
}

LEASE_TTL = 120.0
HEARTBEAT_INTERVAL = 30.0


# Frontierentry


@dataclass
class FrontierEntry:
    id: str
    agent: str
    node_id: str
    node_type: NodeType = NodeType.COMPANY
    priority: int = 50
    depth: int = 0
    state: WorkState = WorkState.PENDING
    confidence: float = 0.5
    freshness: float = field(default_factory=time.monotonic)
    retries: int = 0
    max_retries: int = 3
    lease_expires: float = 0.0
    lease_holder: int = -1
    dependencies: list[str] = field(default_factory=list)
    cost: CostEstimate = field(default_factory=CostEstimate)
    source_connector: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        if self.agent in AGENT_COSTS:
            self.cost = AGENT_COSTS[self.agent]

    @property
    def can_execute(self) -> bool:
        return self.state == WorkState.PENDING and self.retries < self.max_retries

    @property
    def lease_expired(self) -> bool:
        if self.state != WorkState.LEASED:
            return False
        return time.monotonic() > self.lease_expires

    def renew_lease(self) -> None:
        self.lease_expires = time.monotonic() + LEASE_TTL

    @property
    def expected_utility(self) -> float:
        info_gain = self.confidence * 100.0
        cost = max(self.cost.total_apx, 0.1)
        return info_gain / cost

    def recalc_priority(
        self,
        graph_confidence: float | None = None,
        has_hiring_signal: bool = False,
        has_recent_funding: bool = False,
        is_startup: bool = False,
        match_score: int = 0,
        centrality: float = 0.0,
    ) -> None:
        weight = self.expected_utility / 10.0
        score = int(min(100, weight * 60 + 20))
        if graph_confidence is not None:
            score = max(score, int(graph_confidence * 100))
        if has_hiring_signal:
            score += 20
        if has_recent_funding:
            score += 20
        if is_startup:
            score += 10
        if match_score > 0:
            score = max(score, match_score // 2)
        if centrality > 0:
            score += min(15, int(centrality * 100))
        self.priority = min(100, score)


# Adaptive Semaphore


class AdaptiveSemaphore:
    def __init__(self, limit: int) -> None:
        self._sem = asyncio.Semaphore(limit)
        self._limit = limit
        self._lock = asyncio.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    async def set_limit(self, new_limit: int) -> None:
        async with self._lock:
            delta = new_limit - self._limit
            self._limit = new_limit
            if delta > 0:
                for _ in range(delta):
                    self._sem.release()
            elif delta < 0:
                for _ in range(-delta):
                    await self._sem.acquire()

    async def __aenter__(self) -> AdaptiveSemaphore:
        await self._sem.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._sem.release()


# Batch


@dataclass
class WorkBatch:
    entries: list[FrontierEntry]
    agent: str
    batch_id: str = ""

    def __post_init__(self):
        if self.entries:
            self.agent = self.entries[0].agent
            self.batch_id = _hash("batch", self.agent, str(time.monotonic()))


HandlerType = Callable[[FrontierEntry], Awaitable[list[FrontierEntry]]]
BatchHandlerType = Callable[[WorkBatch], Awaitable[list[FrontierEntry]]]


# Events


class GraphEvent(BaseModel):
    event_type: str
    node_id: str
    node_type: NodeType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


# Metrics


@dataclass
class SchedulerMetrics:
    active_workers: int = 0
    pending_work: int = 0
    completed_work: int = 0
    retried_work: int = 0
    failed_work: int = 0
    expired_work: int = 0
    total_enqueued: int = 0
    batches_executed: int = 0
    connector_latency_ms: float = 0.0
    events_fired: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    uptime_s: float = 0.0
    cost_consumed: float = 0.0


# Graph analytics


def compute_centrality(
    node_ids: list[str], edge_dicts: list[dict], iterations: int = 10
) -> dict[str, float]:
    if not node_ids:
        return {}
    damping = 0.85
    n = len(node_ids)
    ranks: dict[str, float] = dict.fromkeys(node_ids, 1.0 / n)
    if not edge_dicts:
        return ranks
    for _ in range(iterations):
        new_ranks: dict[str, float] = dict.fromkeys(node_ids, (1.0 - damping) / n)
        for e in edge_dicts:
            src, tgt = e.get("source", ""), e.get("target", "")
            if src in ranks and tgt in ranks:
                out_deg = max(1, sum(1 for x in edge_dicts if x.get("source") == src))
                new_ranks[tgt] = new_ranks.get(tgt, 0) + damping * ranks[src] / out_deg
        ranks = new_ranks
    return ranks


# Builders


def company_node(name: str, **extra: Any) -> GraphNode:
    return GraphNode(
        id=make_company_id(name), node_type=NodeType.COMPANY, data={"name": name, **extra}
    )


def edge(source: str, etype: EdgeType, target: str, **meta: Any) -> GraphEdge:
    return GraphEdge(source_id=source, edge_type=etype, target_id=target, metadata=meta)
