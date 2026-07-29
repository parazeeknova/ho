"""Graph Store — Neo4j-backed entity graph with typed edges, confidence tracking,
and Cypher-powered traversal/expansion queries.

Same public API as the old pgvector GraphStore so callers work unchanged.
Requires: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD env vars.
"""  # noqa: E501

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from src.graph.entity import (
    Confidence,
    EdgeType,
    GraphEdge,
    GraphNode,
    MutationEvent,
    NodeType,
    confidence_decay,
    merge_confidence,
)


class GraphStore:
    def __init__(self) -> None:
        self._uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._user = os.environ.get("NEO4J_USERNAME", "neo4j")
        self._pwd = os.environ.get("NEO4J_PASSWORD", "password")
        self._driver: Any = None

    @classmethod
    async def create(cls) -> GraphStore:
        store = cls()
        from neo4j import AsyncGraphDatabase

        store._driver = AsyncGraphDatabase.driver(
            store._uri,
            auth=(store._user, store._pwd),
            max_connection_lifetime=3600,
        )
        await store._ensure_indexes()
        return store

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _run(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return await result.data()

    async def _ensure_indexes(self) -> None:
        await self._run(
            "CREATE CONSTRAINT unique_node IF NOT EXISTS FOR (n:GraphNode) REQUIRE n.id IS UNIQUE"
        )
        await self._run(
            "CREATE INDEX node_type_idx IF NOT EXISTS FOR (n:GraphNode) ON (n.node_type)"
        )
        await self._run(
            "CREATE INDEX node_active_idx IF NOT EXISTS FOR (n:GraphNode) ON (n.active)"
        )

    def _node_params(self, node: GraphNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "node_type": node.node_type.value,
            "data": json.dumps(node.data),
            "confidence_score": node.confidence.score,
            "confidence_src": node.confidence.source_count,
            "confidence_mtd": node.confidence.verification_method,
            "last_verified": node.confidence.last_verified.isoformat(),
            "first_seen": node.confidence.first_seen.isoformat(),
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "active": node.active,
        }

    def _row_to_node(self, row: dict) -> GraphNode:
        data = row.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)
        src = row.get("confidence_src", 0)
        return GraphNode(
            id=row["id"],
            node_type=NodeType(row["node_type"]),
            data=data or {},
            confidence=Confidence(
                score=row.get("confidence_score", 0.5),
                source_count=src if src is not None else 0,
                last_verified=_parse_dt(row.get("last_verified")),
                first_seen=_parse_dt(row.get("first_seen")),
                verification_method=row.get("confidence_mtd", "heuristic") or "heuristic",
            ),
            created_at=_parse_dt(row.get("created_at")),
            updated_at=_parse_dt(row.get("updated_at")),
            active=row.get("active", True) is not False,
        )

    def _row_to_edge(self, row: dict) -> GraphEdge:
        return GraphEdge(
            source_id=row["source_id"],
            edge_type=EdgeType(row["edge_type"]),
            target_id=row["target_id"],
            confidence=Confidence(score=row.get("confidence_score", 0.5)),
            metadata=row.get("metadata") or {},
        )

    # Nodes

    async def upsert_node(self, node: GraphNode) -> tuple[GraphNode, list[MutationEvent]]:
        existing = await self.get_node(node.id)
        events: list[MutationEvent] = []
        if existing:
            old_len = len(existing.data)
            existing.data = {**existing.data, **node.data}
            existing.confidence = merge_confidence(existing.confidence, node.confidence)
            existing.updated_at = datetime.now(UTC)
            existing.active = True
            node = existing
            if len(existing.data) > old_len:
                events.append(
                    MutationEvent(
                        mutated_id=node.id,
                        node_type=node.node_type,
                        change="node_upsert",
                    )
                )
        else:
            events.append(
                MutationEvent(
                    mutated_id=node.id,
                    node_type=node.node_type,
                    change="node_upsert",
                )
            )
            events.append(
                MutationEvent(
                    mutated_id=node.id,
                    node_type=node.node_type,
                    change="node_created",
                )
            )
        await self._run(
            """
            MERGE (n:GraphNode {id: $id})
            SET n.node_type = $node_type, n.data = $data,
                n.confidence_score = $confidence_score, n.confidence_src = $confidence_src,
                n.confidence_mtd = $confidence_mtd, n.last_verified = $last_verified,
                n.first_seen = $first_seen, n.created_at = $created_at,
                n.updated_at = $updated_at, n.active = $active
        """,
            self._node_params(node),
        )
        return node, events

    async def get_node(self, node_id: str) -> GraphNode | None:
        rows = await self._run("MATCH (n:GraphNode {id: $id}) RETURN n", {"id": node_id})
        if rows:
            return self._row_to_node(_unpack(rows[0], "n"))
        return None

    async def get_nodes_by_type(self, node_type: NodeType, limit: int = 100) -> list[GraphNode]:
        rows = await self._run(
            "MATCH (n:GraphNode {node_type: $t, active: true}) RETURN n ORDER BY n.updated_at DESC LIMIT $l",
            {"t": node_type.value, "l": limit},
        )
        return [self._row_to_node(_unpack(r, "n")) for r in rows]

    async def search_companies(self, query: str, limit: int = 20) -> list[GraphNode]:
        rows = await self._run(
            "MATCH (n:GraphNode {node_type: 'company', active: true}) WHERE n.data CONTAINS $q RETURN n ORDER BY n.updated_at DESC LIMIT $l",
            {"q": query, "l": limit},
        )
        return [self._row_to_node(_unpack(r, "n")) for r in rows]

    async def decay_stale_confidence(self, max_age_days: int = 30) -> int:
        rows = await self._run(
            "MATCH (n:GraphNode {active: true}) WHERE datetime(n.last_verified) < datetime() - duration({days: $d}) RETURN n",
            {"d": max_age_days},
        )
        updated = 0
        for r in rows:
            n = _unpack(r, "n")
            c = Confidence(
                score=n.get("confidence_score", 0.5),
                last_verified=_parse_dt(n.get("last_verified")),
            )
            c = confidence_decay(c, max_age_days)
            await self._run(
                "MATCH (n:GraphNode {id: $id}) SET n.confidence_score = $s, n.updated_at = $now",
                {"id": n["id"], "s": c.score, "now": datetime.now(UTC).isoformat()},
            )
            updated += 1
        return updated

    async def deactivate_node(self, node_id: str) -> None:
        await self._run(
            "MATCH (n:GraphNode {id: $id}) SET n.active = false, n.updated_at = $now",
            {"id": node_id, "now": datetime.now(UTC).isoformat()},
        )

    # Edges

    async def upsert_edge(self, edge: GraphEdge) -> tuple[GraphEdge, list[MutationEvent]]:
        existing = await self.get_edge(edge.source_id, edge.edge_type, edge.target_id)
        events: list[MutationEvent] = []
        if existing:
            old_score = existing.confidence.score
            existing.confidence = merge_confidence(existing.confidence, edge.confidence)
            existing.metadata = {**existing.metadata, **edge.metadata}
            edge = existing
            if edge.confidence.score > old_score + 0.05:
                events.append(
                    MutationEvent(
                        mutated_id=edge.source_id,
                        node_type=NodeType.COMPANY,
                        change="edge_confidence_boost",
                        edge_type=edge.edge_type,
                        related_id=edge.target_id,
                    )
                )
        else:
            events.append(
                MutationEvent(
                    mutated_id=edge.source_id,
                    node_type=NodeType.COMPANY,
                    change="edge_added",
                    edge_type=edge.edge_type,
                    related_id=edge.target_id,
                )
            )
        await self._run(
            """
            MATCH (a:GraphNode {id: $src}), (b:GraphNode {id: $tgt})
            MERGE (a)-[r:RELATES {type: $etype}]->(b)
            SET r.confidence_score = $cs, r.metadata = $m
        """,
            {
                "src": edge.source_id,
                "tgt": edge.target_id,
                "etype": edge.edge_type.value,
                "cs": edge.confidence.score,
                "m": json.dumps(edge.metadata),
            },
        )
        return edge, events

    async def get_edge(
        self, source_id: str, edge_type: EdgeType, target_id: str
    ) -> GraphEdge | None:
        rows = await self._run(
            "MATCH (a:GraphNode {id: $s})-[r:RELATES {type: $t}]->(b:GraphNode {id: $tg}) RETURN a.id, r, b.id",
            {"s": source_id, "t": edge_type.value, "tg": target_id},
        )
        if rows:
            r = rows[0]
            rel = _unpack(r, "r")
            return GraphEdge(
                source_id=r["a.id"],
                edge_type=edge_type,
                target_id=r["b.id"],
                confidence=Confidence(score=rel.get("confidence_score", 0.5)),
                metadata=_parse_json(rel.get("metadata")),
            )
        return None

    async def get_edges_from(self, node_id: str) -> list[GraphEdge]:
        rows = await self._run(
            "MATCH (a:GraphNode {id: $id})-[r:RELATES]->(b:GraphNode) RETURN a.id, r, b.id",
            {"id": node_id},
        )
        return [_edge_row(r) for r in rows]

    async def get_edges_to(self, node_id: str) -> list[GraphEdge]:
        rows = await self._run(
            "MATCH (a:GraphNode)-[r:RELATES]->(b:GraphNode {id: $id}) RETURN a.id, r, b.id",
            {"id": node_id},
        )
        return [_edge_row(r) for r in rows]

    async def get_neighbors(self, node_id: str) -> list[tuple[GraphEdge, GraphNode]]:
        rows = await self._run(
            "MATCH (a:GraphNode {id: $id})-[r:RELATES]->(b:GraphNode) WHERE b.active = true RETURN a.id, r, b",
            {"id": node_id},
        )
        return [
            (
                _edge_row(r),
                self._row_to_node(_unpack(r, "b")),
            )
            for r in rows
        ]

    async def get_all_edges(self, limit: int = 500) -> list[GraphEdge]:
        rows = await self._run(
            "MATCH (a:GraphNode)-[r:RELATES]->(b:GraphNode) RETURN a.id, r, b.id LIMIT $l",
            {"l": limit},
        )
        return [_edge_row(r) for r in rows]

    async def get_node_adjacency(self, node_id: str) -> dict[str, Any]:
        out_rows = await self._run(
            "MATCH (a:GraphNode {id: $id})-[r:RELATES]->(b:GraphNode) RETURN r.type, b.id",
            {"id": node_id},
        )
        in_rows = await self._run(
            "MATCH (a:GraphNode)-[r:RELATES]->(b:GraphNode {id: $id}) RETURN r.type, a.id",
            {"id": node_id},
        )
        return {
            "edges_out": {f"{r['r.type']}:{r['b.id']}" for r in out_rows},
            "edges_in": {f"{r['r.type']}:{r['a.id']}" for r in in_rows},
        }

    async def get_local_graph(self, node_id: str, radius: int = 1) -> dict[str, Any]:
        rows = await self._run(
            f"MATCH (n:GraphNode {{id: $id}})-[r:RELATES*1..{radius}]-(m:GraphNode) RETURN n, m, r",
            {"id": node_id},
        )
        seen = {node_id}
        nodes = [await self.get_node(node_id)] if node_id else []
        edges: list[GraphEdge] = []
        for row in rows:
            m = _unpack(row, "m")
            mid = m.get("id", "")
            if mid and mid not in seen:
                seen.add(mid)
                nodes.append(self._row_to_node(m))
            rels = row.get("r", [])
            for rel in rels if isinstance(rels, list) else [rels]:
                edges.append(
                    GraphEdge(
                        source_id="",
                        edge_type=EdgeType(rel.get("type", "") or ""),
                        target_id=mid,
                        confidence=Confidence(score=rel.get("confidence_score", 0.5)),
                    )
                )
        return {"nodes": nodes, "edges": edges}


# Helpers


def _unpack(row: dict, key: str) -> dict[str, Any]:
    v = row.get(key, row)
    if hasattr(v, "_properties"):
        return dict(v._properties)  # type: ignore[union-attr]
    if isinstance(v, dict):
        return v
    return {k: row[k] for k in row if not k.startswith("_")}


def _edge_row(r: dict) -> GraphEdge:
    rel = _unpack(r, "r")
    return GraphEdge(
        source_id=r.get("a.id", ""),
        edge_type=EdgeType(rel.get("type", "") or ""),
        target_id=r.get("b.id", ""),
        confidence=Confidence(score=rel.get("confidence_score", 0.5)),
        metadata=_parse_json(rel.get("metadata")),
    )


def _parse_json(val: Any) -> dict[str, Any]:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError, TypeError:
            return {}
    if isinstance(val, dict):
        return val
    return {}


def _parse_dt(val: Any) -> datetime:
    if val is None:
        return datetime.now(UTC)
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError, TypeError:
            return datetime.now(UTC)
    return datetime.now(UTC)
