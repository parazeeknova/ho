"""Entity models: nodes, edges, work items, frontier entries, metrics.

Deterministic ID generation (no UUID) so work is deduplicable.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── Node types ─────────────────────────────────────────────────────────────────


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
    OFFICE = "office"


# ── Edge types ─────────────────────────────────────────────────────────────────


class EdgeType(StrEnum):
    FOUNDED_BY = "founded_by"
    WORKS_AT = "works_at"
    USES_ATS = "uses_ats"
    INVESTED_BY = "invested_by"
    HIRED_FOR = "hired_for"
    MENTIONED_IN = "mentioned_in"
    POSTED_JOB = "posted_job"
    USES_TECH = "uses_tech"
    RELATED_TO = "related_to"
    DISCOVERED_FROM = "discovered_from"
    HAS_CAREER_SITE = "has_career_site"
    HAS_FUNDING = "has_funding"
    HAS_OFFICE = "has_office"


# ── Work item agent types ─────────────────────────────────────────────────────


class AgentType(StrEnum):
    FOUNDER_MINER = "founder_miner"
    CAREER_SITE_DETECTOR = "career_site_detector"
    FOUNDER_SOCIAL_OSINT = "founder_social_osint"
    EMPLOYEE_DISCOVERY = "employee_discovery"
    ATS_CRAWLER = "ats_crawler"
    FUNDING_AGENT = "funding_agent"
    TECH_STACK_AGENT = "tech_stack_agent"
    HIRING_SIGNAL_AGENT = "hiring_signal_agent"
    GRAPH_MAINTENANCE = "graph_maintenance"
    RELATIONSHIP_BUILDER = "relationship_builder"


# ── Confidence tracking ───────────────────────────────────────────────────────


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
    total = old.source_count + source_bonus
    old.score = min(1.0, max(0.1, raw / max(total, 1)))
    return old


def confidence_decay(c: Confidence, max_age_days: int = 30) -> Confidence:
    age = (datetime.now(UTC) - c.last_verified).days
    if age > max_age_days:
        c.score = max(0.1, c.score * (1.0 - (age - max_age_days) / 60.0))
    return c


# ── Graph nodes ────────────────────────────────────────────────────────────────


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


# ── Deterministic ID helpers ───────────────────────────────────────────────────


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


def make_job_id(company: str, role: str) -> str:
    return _hash(f"job:{company.lower().strip()}:{role.lower().strip()}")


# ── FrontierEntry ──────────────────────────────────────────────────────────────


@dataclass
class FrontierEntry:
    id: str
    agent: str
    node_id: str
    node_type: NodeType
    priority: int = 50
    depth: int = 0
    revisit_interval: float = 86400.0  # seconds
    confidence: float = 0.5
    freshness: float = field(default_factory=time.monotonic)
    retries: int = 0
    max_retries: int = 3
    crawl_budget: int = 1
    dependencies: list[str] = field(default_factory=list)
    source_connector: str = ""
    entity_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    def recalc_priority(
        self,
        graph_confidence: float | None = None,
        has_hiring_signal: bool = False,
        has_recent_funding: bool = False,
        is_startup: bool = False,
        match_score: int = 0,
    ) -> None:
        score = self.priority
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
        self.priority = min(100, score)

    @property
    def is_stale(self) -> bool:
        return time.monotonic() - self.freshness > self.revisit_interval

    @property
    def can_execute(self) -> bool:
        return self.retries < self.max_retries and self.crawl_budget > 0


# ── Graph event (publish-only) ─────────────────────────────────────────────────


class GraphEvent(BaseModel):
    event_type: str
    node_id: str
    node_type: NodeType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Metrics ────────────────────────────────────────────────────────────────────


@dataclass
class SchedulerMetrics:
    active_workers: int = 0
    pending_work: int = 0
    completed_work: int = 0
    retried_work: int = 0
    failed_work: int = 0
    total_enqueued: int = 0
    connector_latency_ms: float = 0.0
    avg_queue_wait_s: float = 0.0
    llm_calls: int = 0
    events_fired: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    uptime_s: float = 0.0


# ── Node builders ──────────────────────────────────────────────────────────────


def company_node(name: str, **extra: Any) -> GraphNode:
    return GraphNode(
        id=make_company_id(name), node_type=NodeType.COMPANY, data={"name": name, **extra}
    )


def edge(source: str, etype: EdgeType, target: str, **meta: Any) -> GraphEdge:
    return GraphEdge(source_id=source, edge_type=etype, target_id=target, metadata=meta)
