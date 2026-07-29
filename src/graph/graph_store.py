"""Graph Store — pgvector-backed entity graph with typed edges, deduplication,
confidence tracking, and recursive expansion support.

Extends the existing MemoryStore pattern but adds graph semantics:
nodes, edges, confidence scores, and decay.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from src.graph.entity import (
    Confidence,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    confidence_decay,
    merge_confidence,
)

DSN = "postgresql://postgres:postgres@localhost:5433/agent_memory"

CREATE_GRAPH_SQL = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id            TEXT PRIMARY KEY,
    node_type     TEXT NOT NULL,
    data          JSONB DEFAULT '{}'::jsonb,
    confidence    JSONB DEFAULT '{
        "score": 0.5, "source_count": 0, "verification_method": "heuristic"
    }'::jsonb,
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW(),
    active        BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS graph_edges (
    source_id     TEXT NOT NULL,
    edge_type     TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    confidence    JSONB DEFAULT '{
        "score": 0.5, "source_count": 1, "verification_method": "heuristic"
    }'::jsonb,
    metadata      JSONB DEFAULT '{}'::jsonb,
    created_at    TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (source_id, edge_type, target_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges (target_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_active ON graph_nodes (active);

CREATE TABLE IF NOT EXISTS frontier_state (
    work_id      TEXT PRIMARY KEY,
    agent        TEXT NOT NULL,
    node_id      TEXT NOT NULL,
    node_type    TEXT DEFAULT 'company',
    priority     INT DEFAULT 50,
    depth        INT DEFAULT 0,
    retries      INT DEFAULT 0,
    payload      JSONB DEFAULT '{}'::jsonb,
    created_at   TIMESTAMP DEFAULT NOW(),
    updated_at   DOUBLE PRECISION DEFAULT 0
);

CREATE TABLE IF NOT EXISTS frontier_completed (
    work_id      TEXT PRIMARY KEY,
    completed_at TIMESTAMP DEFAULT NOW()
);
"""


class GraphStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls) -> GraphStore:
        pool = await asyncpg.create_pool(DSN, min_size=2, max_size=10)
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(CREATE_GRAPH_SQL)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    # Nodes

    async def upsert_node(self, node: GraphNode) -> GraphNode:
        existing = await self.get_node(node.id)
        if existing:
            existing.data = {**existing.data, **node.data}
            existing.confidence = merge_confidence(existing.confidence, node.confidence)
            existing.updated_at = datetime.now(UTC)
            existing.active = True
            node = existing
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_nodes (id, node_type, data, confidence,
                                         created_at, updated_at, active)
                VALUES ($1,$2,$3::jsonb,$4::jsonb,$5,$6,$7)
                ON CONFLICT (id) DO UPDATE SET
                    data = graph_nodes.data || EXCLUDED.data,
                    confidence = EXCLUDED.confidence,
                    updated_at = EXCLUDED.updated_at,
                    active = EXCLUDED.active
                """,
                node.id,
                node.node_type.value,
                json.dumps(node.data),
                json.dumps(node.confidence.model_dump()),
                node.created_at,
                node.updated_at,
                node.active,
            )
        return node

    async def get_node(self, node_id: str) -> GraphNode | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM graph_nodes WHERE id = $1", node_id)
        return _row_to_node(row) if row else None

    async def get_nodes_by_type(self, node_type: NodeType, limit: int = 100) -> list[GraphNode]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM graph_nodes WHERE node_type = $1 AND active = TRUE "
                "ORDER BY updated_at DESC LIMIT $2",
                node_type.value,
                limit,
            )
        return [_row_to_node(r) for r in rows if r]

    async def search_companies(self, query: str, limit: int = 20) -> list[GraphNode]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM graph_nodes WHERE node_type = 'company' AND active = TRUE "
                "AND data->>'name' ILIKE $1 ORDER BY updated_at DESC LIMIT $2",
                f"%{query}%",
                limit,
            )
        return [_row_to_node(r) for r in rows if r]

    async def decay_stale_confidence(self, max_age_days: int = 30) -> int:
        updated = 0
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, confidence FROM graph_nodes WHERE active = TRUE")
            for r in rows:
                c = _parse_confidence(r["confidence"])
                old_score = c.score
                c = confidence_decay(c, max_age_days)
                if c.score != old_score:
                    await conn.execute(
                        "UPDATE graph_nodes SET confidence = $1::jsonb, updated_at = $2 "
                        "WHERE id = $3",
                        json.dumps(c.model_dump()),
                        datetime.now(UTC),
                        r["id"],
                    )
                    updated += 1
        return updated

    async def deactivate_node(self, node_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE graph_nodes SET active = FALSE, updated_at = $1 WHERE id = $2",
                datetime.now(UTC),
                node_id,
            )

    # Edges

    async def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        existing = await self.get_edge(edge.source_id, edge.edge_type, edge.target_id)
        if existing:
            existing.confidence = merge_confidence(existing.confidence, edge.confidence)
            existing.metadata = {**existing.metadata, **edge.metadata}
            edge = existing
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_edges (source_id, edge_type, target_id,
                                         confidence, metadata, created_at)
                VALUES ($1,$2,$3,$4::jsonb,$5::jsonb,$6)
                ON CONFLICT (source_id, edge_type, target_id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    metadata = graph_edges.metadata || EXCLUDED.metadata
                """,
                edge.source_id,
                edge.edge_type.value,
                edge.target_id,
                json.dumps(edge.confidence.model_dump()),
                json.dumps(edge.metadata),
                edge.created_at,
            )
        return edge

    async def get_edge(
        self, source_id: str, edge_type: EdgeType, target_id: str
    ) -> GraphEdge | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM graph_edges WHERE source_id=$1 AND edge_type=$2 AND target_id=$3",
                source_id,
                edge_type.value,
                target_id,
            )
        return _row_to_edge(row) if row else None

    async def get_edges_from(self, node_id: str) -> list[GraphEdge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM graph_edges WHERE source_id = $1 ORDER BY created_at DESC",
                node_id,
            )
        return [_row_to_edge(r) for r in rows if r]

    async def get_edges_to(self, node_id: str) -> list[GraphEdge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM graph_edges WHERE target_id = $1 ORDER BY created_at DESC",
                node_id,
            )
        return [_row_to_edge(r) for r in rows if r]

    async def get_neighbors(self, node_id: str) -> list[tuple[GraphEdge, GraphNode]]:
        neighbors: list[tuple[GraphEdge, GraphNode]] = []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.*, n.* FROM graph_edges e
                JOIN graph_nodes n ON n.id = e.target_id
                WHERE e.source_id = $1 AND n.active = TRUE
                ORDER BY e.created_at DESC
                LIMIT 50
                """,
                node_id,
            )
        for r in rows:
            neighbors.append((_row_to_edge(r), _row_to_node(r)))
        return neighbors

    async def get_all_edges(self, limit: int = 500) -> list[GraphEdge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM graph_edges ORDER BY created_at DESC LIMIT $1",
                limit,
            )
        return [_row_to_edge(r) for r in rows if r]


# Row-to-object helpers


def _parse_confidence(raw: Any) -> Confidence:
    if isinstance(raw, str):
        return Confidence(**json.loads(raw))
    if isinstance(raw, dict):
        return Confidence(**raw)
    return Confidence()


def _row_to_node(row: asyncpg.Record | None) -> GraphNode | None:
    if row is None:
        return None
    data = row.get("data")
    if isinstance(data, str):
        data = json.loads(data)
    return GraphNode(
        id=row["id"],
        node_type=NodeType(row["node_type"]),
        data=data or {},
        confidence=_parse_confidence(row.get("confidence")),
        created_at=row.get("created_at", datetime.now(UTC)),
        updated_at=row.get("updated_at", datetime.now(UTC)),
        active=row.get("active", True),
    )


def _row_to_edge(row: asyncpg.Record | None) -> GraphEdge | None:
    if row is None:
        return None
    meta = row.get("metadata")
    if isinstance(meta, str):
        meta = json.loads(meta)
    return GraphEdge(
        source_id=row["source_id"],
        edge_type=EdgeType(row["edge_type"]),
        target_id=row["target_id"],
        confidence=_parse_confidence(row.get("confidence")),
        metadata=meta or {},
        created_at=row.get("created_at", datetime.now(UTC)),
    )
