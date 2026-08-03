"""Entity models: graph, frontier entries with leasing, cost, utility, batching.

Deterministic ID generation for deduplication. Lease heartbeats for
long-running tasks. Utility scoring for information gain/cost tradeoffs.

Entity Resolution: fuzzy matching for duplicate detection with canonical
IDs and alias preservation across Neo4j and PostgreSQL.
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

from src.configuration import get_config


def _default_lease_ttl() -> float:
    return get_config().scheduler.lease_ttl


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
    edges_out: set[str] = Field(default_factory=set, exclude=True)
    edges_in: set[str] = Field(default_factory=set, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def name(self) -> str:
        return str(self.data.get("name") or self.data.get("company") or self.id[:12])

    def has_edge(self, etype: EdgeType, target_id: str) -> bool:
        key = f"{etype.value}:{target_id}"
        return key in self.edges_out

    def add_edge_out(self, etype: EdgeType, target_id: str) -> None:
        self.edges_out.add(f"{etype.value}:{target_id}")

    def add_edge_in(self, etype: EdgeType, source_id: str) -> None:
        self.edges_in.add(f"{etype.value}:{source_id}")


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


# Entity Resolution


def normalize_company_name(name: str) -> str:
    """Produce a canonical, normalized form for fuzzy-deduping company names.

    Handles: 'Stripe Inc.' -> 'stripe', 'stripe.com' -> 'stripe',
    'Stripe, Inc.' -> 'stripe'.
    """
    n = name.lower().strip()
    for suffix in (
        " inc.",
        " inc",
        ", inc.",
        ", inc",
        " ltd.",
        " ltd",
        " llc",
        " llc.",
        " corp.",
        " corp",
        " corporation",
        " co.",
        " co",
        " limited",
        " l.l.c.",
        " pvt ltd",
        " private limited",
        " pte ltd",
        " gmbh",
    ):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    if n.startswith("www."):
        n = n[4:]
    for tld in (".com", ".io", ".co", ".ai", ".dev", ".app", ".org", ".net", ".in"):
        if n.endswith(tld):
            n = n[: -len(tld)].strip()
    n = " ".join(n.split())
    return n


def make_canonical_company_id(name: str) -> str:
    """Canonical ID from normalized name for stable cross-source merging."""
    return _hash(f"company_canonical:{normalize_company_name(name)}")


def fuzzy_match_companies(
    a_name: str,
    b_name: str,
    a_url: str | None = None,
    b_url: str | None = None,
) -> tuple[bool, float]:
    """Detect duplicate company names with fuzzy matching.

    Returns (is_duplicate, similarity_score).

    Rules:
    - Exact match after normalize: score 1.0
    - Same domain: score 0.95
    - High Jaccard similarity on tokenized name: score 0.7-0.9
    """
    a_norm = normalize_company_name(a_name)
    b_norm = normalize_company_name(b_name)
    if a_norm == b_norm:
        return True, 1.0
    if a_url and b_url:
        from urllib.parse import urlparse

        try:
            a_domain = urlparse(a_url).netloc.lower().replace("www.", "")
            b_domain = urlparse(b_url).netloc.lower().replace("www.", "")
            if a_domain and b_domain and a_domain == b_domain:
                return True, 0.95
        except Exception:
            pass
    a_tokens = set(a_norm.replace("-", " ").split())
    b_tokens = set(b_norm.replace("-", " ").split())
    if a_tokens and b_tokens:
        intersection = a_tokens & b_tokens
        union = a_tokens | b_tokens
        jaccard = len(intersection) / len(union) if union else 0
        if jaccard >= 0.75:
            return True, 0.7 + jaccard * 0.3
    return False, 0.0


class PropertyProvenance(BaseModel):
    value: Any
    source: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    confidence: float = 0.5


def property_provenance(
    value: Any,
    source: str,
    confidence: float = 0.5,
) -> PropertyProvenance:
    return PropertyProvenance(value=value, source=source, confidence=confidence)


def _unwrap_prov(value: Any) -> Any:
    """Unwrap a PropertyProvenance dict back to its raw value."""
    if isinstance(value, dict) and "value" in value and "source" in value:
        return value["value"]
    return value


def resolve_entity(
    existing_aliases: list[str],
    new_alias: str,
    existing_data: dict[str, Any],
    new_data: dict[str, Any],
    new_source: str,
) -> dict[str, Any]:
    """Merge data from a new source into existing node data with provenance.

    Existing non-provenance fields are upgraded to PropertyProvenance.
    Conflicting values from different sources coexist until resolved
    by confidence scores.
    """
    merged: dict[str, Any] = dict(existing_data)
    merged.setdefault("aliases", [])
    if existing_aliases:
        for alias in existing_aliases:
            if alias not in merged["aliases"]:
                merged["aliases"].append(alias)
    if new_alias not in merged["aliases"]:
        merged["aliases"].append(new_alias)

    for key, value in new_data.items():
        if key == "aliases":
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = property_provenance(value, new_source).model_dump(mode="json")
            continue
        if isinstance(existing, dict) and "value" in existing and "source" in existing:
            if _unwrap_prov(existing["value"]) != _unwrap_prov(value):
                alt_key = f"{key}__alt_{_unwrap_prov(new_source)}"
                merged[alt_key] = property_provenance(value, new_source).model_dump(mode="json")
        else:
            merged[key] = property_provenance(value, new_source).model_dump(mode="json")

    return merged


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
    _lease_ttl: float = field(default_factory=_default_lease_ttl)

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
        self.lease_expires = time.monotonic() + self._lease_ttl

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
        pagerank: float = 0.0,
        betweenness: float = 0.0,
        relationship_density: int = 0,
        entity_uncertainty: float = 0.0,
        freshness_days: float = 0.0,
    ) -> None:
        """Recalculate priority driven by GDS graph metrics.

        Formula weights (sums to ~100):
          - PageRank influence:         25%
          - Relationship density:       20%
          - Entity uncertainty:         25%
          - Freshness bonus:            10%
          - Hiring / funding signals:   10%
          - Match score / centrality:   10%

        Higher PageRank, more missing relationships, higher uncertainty,
        and fresher data all push the priority up.
        """
        pagerank_component = min(25, pagerank * 250)
        density_component = min(20, relationship_density * 2)
        uncertainty_component = min(25, entity_uncertainty * 25)
        freshness_component = max(0, 10 - min(10, freshness_days * 0.5))

        signal_component = 0
        if has_hiring_signal:
            signal_component += 5
        if has_recent_funding:
            signal_component += 5

        residual = max(0, match_score / 10)
        if centrality > 0:
            residual += min(5, centrality * 50)
        if betweenness > 0:
            residual += min(5, betweenness * 50)

        score = int(
            pagerank_component
            + density_component
            + uncertainty_component
            + freshness_component
            + signal_component
            + residual
        )
        self.priority = min(100, max(1, score))


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
    batch_efficiency: float = 0.0
    connector_latency_ms: float = 0.0
    p95_latency_s: float = 0.0
    lease_renewals: int = 0
    events_fired: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    uptime_s: float = 0.0
    cost_consumed: float = 0.0
    expansion_events: int = 0


@dataclass
class MutationEvent:
    mutated_id: str
    node_type: NodeType
    change: str  # "node_upsert", "edge_added", "edge_type"
    edge_type: EdgeType | None = None
    related_id: str | None = None
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def event_id(self) -> str:
        parts = [self.mutated_id, self.change]
        if self.edge_type:
            parts.append(self.edge_type.value)
        if self.related_id:
            parts.append(self.related_id)
        return _hash("mutation", *parts)


# Capability Graph Definitions


@dataclass
class CapabilityPattern:
    edge_type: EdgeType
    target_type: NodeType
    required: bool = False
    min_count: int = 1
    description: str = ""


CAPABILITY_GRAPH: dict[NodeType, list[CapabilityPattern]] = {
    NodeType.COMPANY: [
        CapabilityPattern(
            EdgeType.FOUNDED_BY,
            NodeType.FOUNDER,
            required=True,
            description="Should have at least one known founder",
        ),
        CapabilityPattern(
            EdgeType.USES_ATS,
            NodeType.ATS,
            required=False,
            description="May use an ATS for hiring",
        ),
        CapabilityPattern(
            EdgeType.HAS_FUNDING,
            NodeType.FUNDING_ROUND,
            required=False,
            description="May have funding rounds",
        ),
        CapabilityPattern(
            EdgeType.HAS_CAREER_SITE,
            NodeType.CAREER_SITE,
            required=False,
            description="May have a career site",
        ),
        CapabilityPattern(
            EdgeType.USES_TECH,
            NodeType.TECHNOLOGY,
            required=False,
            description="Uses specific technologies",
        ),
        CapabilityPattern(
            EdgeType.POSTED_JOB,
            NodeType.JOB,
            required=False,
            description="Has posted job openings",
        ),
    ],
    NodeType.FOUNDER: [
        CapabilityPattern(
            EdgeType.FOUNDED_BY,
            NodeType.COMPANY,
            required=True,
            description="Associated with a company",
        ),
        CapabilityPattern(
            EdgeType.WORKS_AT,
            NodeType.COMPANY,
            required=False,
            description="May work at a specific company",
        ),
    ],
    NodeType.CAREER_SITE: [
        CapabilityPattern(
            EdgeType.POSTED_JOB,
            NodeType.JOB,
            required=True,
            description="Should have job postings",
        ),
    ],
    NodeType.FUNDING_ROUND: [
        CapabilityPattern(
            EdgeType.INVESTED_BY,
            NodeType.INVESTOR,
            required=False,
            description="May have known investors",
        ),
    ],
}


class NodeUncertaintyScore(BaseModel):
    completeness: float = 0.0
    uncertainty: float = 0.0
    staleness: float = 0.0
    total: float = 0.0


def compute_uncertainty_score(
    node: GraphNode,
    adjacency: dict[str, Any],
    capability_graph: dict | None = None,
    max_age_days: int = 30,
) -> NodeUncertaintyScore:
    """Calculate uncertainty and completeness for a node.

    - completeness: fraction of required and optional capability patterns satisfied
    - uncertainty: inverse of confidence, weighted by missing required edges
    - staleness: how stale the data is (age / max_age)
    - total: composite score suitable for ranking frontier expansion

    Lower completeness and higher uncertainty mean the node is a priority
    target for information-gathering operations.
    """
    patterns = (capability_graph or CAPABILITY_GRAPH).get(node.node_type, [])
    if not patterns:
        return NodeUncertaintyScore(completeness=1.0, uncertainty=0.0, staleness=0.0, total=0.0)

    edges_out = adjacency.get("edges_out", set())
    edges_in = adjacency.get("edges_in", set())

    satisfied = 0
    required_total = 0
    required_satisfied = 0

    for pat in patterns:
        key = f"{pat.edge_type.value}:"
        matches = sum(1 for e in edges_out if key in e) + sum(1 for e in edges_in if key in e)
        if matches >= pat.min_count:
            satisfied += 1
            if pat.required:
                required_satisfied += 1
        if pat.required:
            required_total += 1

    total_patterns = len(patterns)
    completeness = satisfied / total_patterns if total_patterns > 0 else 1.0

    required_penalty = 0.0
    if required_total > 0:
        required_penalty = (required_total - required_satisfied) / required_total * 0.5
    uncertainty = (1.0 - node.confidence.score) * (0.5 + required_penalty)
    uncertainty = min(1.0, uncertainty)

    age = (datetime.now(UTC) - node.updated_at).days
    staleness = min(1.0, age / max_age_days)

    total = (1.0 - completeness) * 0.5 + uncertainty * 0.3 + staleness * 0.2
    total = min(1.0, max(0.0, total))

    return NodeUncertaintyScore(
        completeness=completeness,
        uncertainty=uncertainty,
        staleness=staleness,
        total=total,
    )


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
