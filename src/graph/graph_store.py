"""Graph Store — Neo4j-backed entity graph with typed edges, confidence tracking,
and Cypher-powered traversal/expansion queries.

Same public API as the old pgvector GraphStore so callers work unchanged.
"""  # noqa: E501

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from src.configuration import Neo4jConfig, get_config
from src.graph.entity import (
    Confidence,
    EdgeType,
    GraphEdge,
    GraphNode,
    MutationEvent,
    NodeType,
    confidence_decay,
    fuzzy_match_companies,
    make_canonical_company_id,
    merge_confidence,
    normalize_company_name,
    resolve_entity,
)
from src.logging import get_logger

logger = get_logger("graph_store")


class GraphStore:
    def __init__(self, config: Neo4jConfig | None = None) -> None:
        cfg = config or get_config().neo4j
        self._uri = cfg.uri
        self._user = cfg.username
        self._pwd = cfg.password
        self._max_conn_lifetime = cfg.max_connection_lifetime
        self._driver: Any = None

    @classmethod
    async def create(cls, config: Neo4jConfig | None = None) -> GraphStore:
        cfg = config or get_config().neo4j
        store = cls(cfg)
        from neo4j import AsyncGraphDatabase

        store._driver = AsyncGraphDatabase.driver(
            store._uri,
            auth=(store._user, store._pwd),
            max_connection_lifetime=store._max_conn_lifetime,
        )
        await store._ensure_indexes()
        logger.info("Neo4j graph store initialized")
        return store

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j graph store closed")

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
            new_data = dict(node.data)
            new_alias = node.name
            existing.data = resolve_entity(
                existing_aliases=existing.data.get("aliases", []),
                new_alias=new_alias,
                existing_data=existing.data,
                new_data=new_data,
                new_source=new_data.get("source", "unknown"),
            )
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
            node.data.setdefault("aliases", [])
            if node.name not in node.data["aliases"]:
                node.data["aliases"].append(node.name)
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

    async def upsert_node_canonical(
        self, node: GraphNode
    ) -> tuple[GraphNode, list[MutationEvent], bool]:
        """Upsert using canonical ID resolution.

        Returns (node, events, is_new).
        Attempts fuzzy matching against existing companies and merges
        if a duplicate is found, preserving aliases.
        """
        canonical_id = make_canonical_company_id(node.name)
        events: list[MutationEvent] = []
        is_new = False

        existing = await self.get_node(node.id)
        if existing:
            existing.data["aliases"] = list(set(existing.data.get("aliases", []) + [node.name]))
            resolved, resolved_events = await self.upsert_node(
                GraphNode(
                    id=existing.id,
                    node_type=existing.node_type,
                    data=existing.data,
                    confidence=merge_confidence(existing.confidence, node.confidence),
                )
            )
            return resolved, resolved_events, False

        existing_canonical = await self.get_node(canonical_id)
        if existing_canonical:
            existing_canonical.data["aliases"] = list(
                set(existing_canonical.data.get("aliases", []) + [node.name, node.id])
            )
            resolved_data = resolve_entity(
                existing_aliases=existing_canonical.data.get("aliases", []),
                new_alias=node.name,
                existing_data=existing_canonical.data,
                new_data=node.data,
                new_source=node.data.get("source", "unknown"),
            )
            resolved, resolved_events = await self.upsert_node(
                GraphNode(
                    id=existing_canonical.id,
                    node_type=existing_canonical.node_type,
                    data=resolved_data,
                    confidence=merge_confidence(existing_canonical.confidence, node.confidence),
                )
            )
            return resolved, resolved_events, False

        companies = await self.search_companies(normalize_company_name(node.name)[:20], limit=20)
        for candidate in companies:
            if candidate.id == node.id:
                continue
            is_dup, score = fuzzy_match_companies(
                node.name,
                candidate.data.get("name", ""),
                node.data.get("url"),
                candidate.data.get("url"),
            )
            if is_dup and score > 0.85:
                candidate.data["aliases"] = list(
                    set(candidate.data.get("aliases", []) + [node.name, node.id])
                )
                candidate.data["duplicate_score"] = score
                resolved_data = resolve_entity(
                    existing_aliases=candidate.data.get("aliases", []),
                    new_alias=node.name,
                    existing_data=candidate.data,
                    new_data=node.data,
                    new_source=node.data.get("source", "unknown"),
                )
                resolved, resolved_events = await self.upsert_node(
                    GraphNode(
                        id=candidate.id,
                        node_type=candidate.node_type,
                        data=resolved_data,
                        confidence=merge_confidence(candidate.confidence, node.confidence),
                    )
                )
                return resolved, resolved_events, False

        node.id = canonical_id
        node.data["aliases"] = list(set(node.data.get("aliases", []) + [node.name, node.id]))
        is_new = True
        events.append(
            MutationEvent(
                mutated_id=canonical_id,
                node_type=node.node_type,
                change="node_created",
            )
        )
        events.append(
            MutationEvent(
                mutated_id=canonical_id,
                node_type=node.node_type,
                change="node_upsert",
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
        return node, events, is_new

    async def find_duplicates(self, min_score: float = 0.8) -> list[dict[str, Any]]:
        """Find duplicate company nodes in the graph using fuzzy matching.

        Returns list of {node_a, node_b, score} for manual or automatic resolution.
        """
        companies = await self.get_nodes_by_type(NodeType.COMPANY, limit=500)
        duplicates: list[dict[str, Any]] = []
        for i, a in enumerate(companies):
            for b in companies[i + 1 :]:
                if a.id == b.id:
                    continue
                is_dup, score = fuzzy_match_companies(
                    a.data.get("name", ""),
                    b.data.get("name", ""),
                    a.data.get("url"),
                    b.data.get("url"),
                )
                if is_dup and score >= min_score:
                    duplicates.append(
                        {
                            "node_a_id": a.id,
                            "node_b_id": b.id,
                            "node_a_name": a.data.get("name", ""),
                            "node_b_name": b.data.get("name", ""),
                            "score": score,
                        }
                    )
        return duplicates

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

    # Neo4j Graph Data Science (GDS)

    async def compute_pagerank(self) -> dict[str, float]:
        """Native Neo4j PageRank across all active Company and Founder nodes.

        Returns mapping of node_id -> PageRank score (0-1, sum ≈ 1.0).
        Falls back to Python-based PageRank if GDS library not installed.
        """
        try:
            rows = await self._run(
                """
                CALL gds.pageRank.stream('graph_view')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).id AS node_id, score
                ORDER BY score DESC
                LIMIT 200
                """
            )
            if rows:
                return {r["node_id"]: r["score"] for r in rows}
        except Exception:
            logger.debug("Neo4j GDS not available, using fallback PageRank")

        nodes = await self.get_nodes_by_type(NodeType.COMPANY, limit=200)
        edges = await self.get_all_edges(limit=1000)
        node_ids = [n.id for n in nodes]
        edge_dicts = [{"source": e.source_id, "target": e.target_id} for e in edges]
        from src.graph.entity import compute_centrality

        return compute_centrality(node_ids, edge_dicts)

    async def compute_connected_components(self) -> dict[str, int]:
        """Weakly Connected Components via Neo4j GDS.

        Returns mapping of node_id -> component_id.
        Used to find isolated subgraphs (stealth startups, closed ecosystems).
        """
        try:
            rows = await self._run(
                """
                CALL gds.wcc.stream('graph_view')
                YIELD nodeId, componentId
                RETURN gds.util.asNode(nodeId).id AS node_id, componentId
                LIMIT 500
                """
            )
            if rows:
                return {r["node_id"]: r["componentId"] for r in rows}
        except Exception:
            logger.debug("Neo4j GDS WCC not available")

        return {}

    async def compute_betweenness_centrality(self) -> dict[str, float]:
        """Betweenness Centrality via Neo4j GDS.

        Returns mapping of node_id -> betweenness score.
        High betweenness nodes are critical bridges in the graph.
        """
        try:
            rows = await self._run(
                """
                CALL gds.betweenness.stream('graph_view')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).id AS node_id, score
                ORDER BY score DESC
                LIMIT 200
                """
            )
            if rows:
                return {r["node_id"]: r["score"] for r in rows}
        except Exception:
            logger.debug("Neo4j GDS betweenness not available")

        return {}

    async def update_all_graph_metrics(self) -> dict[str, Any]:
        """Periodically execute all graph algorithms and store results
        in node data properties for use in priority scoring.

        Returns dict with counts of updated nodes per metric.
        """
        pagerank = await self.compute_pagerank()
        components = await self.compute_connected_components()
        betweenness = await self.compute_betweenness_centrality()

        for node_id, score in pagerank.items():
            await self._run(
                "MATCH (n:GraphNode {id: $id}) SET n.data = apoc.convert.setProperty(n.data, 'pagerank', $score), n.updated_at = $now",
                {"id": node_id, "score": score, "now": datetime.now(UTC).isoformat()},
            )

        for node_id, component in components.items():
            await self._run(
                "MATCH (n:GraphNode {id: $id}) SET n.data = apoc.convert.setProperty(n.data, 'wcc_component', $comp), n.updated_at = $now",
                {"id": node_id, "comp": component, "now": datetime.now(UTC).isoformat()},
            )

        for node_id, score in betweenness.items():
            await self._run(
                "MATCH (n:GraphNode {id: $id}) SET n.data = apoc.convert.setProperty(n.data, 'betweenness', $score), n.updated_at = $now",
                {"id": node_id, "score": score, "now": datetime.now(UTC).isoformat()},
            )

        logger.info(
            "Graph metrics updated",
            extra={
                "pagerank_nodes": len(pagerank),
                "wcc_components": len(set(components.values())),
                "betweenness_nodes": len(betweenness),
            },
        )
        return {
            "pagerank_nodes": len(pagerank),
            "wcc_components": len(set(components.values())),
            "betweenness_nodes": len(betweenness),
        }

    async def get_graph_metrics_for_node(self, node_id: str) -> dict[str, float]:
        """Retrieve stored graph metrics for a single node."""
        node = await self.get_node(node_id)
        if node is None:
            return {}
        return {
            "pagerank": float(node.data.get("pagerank", 0)),
            "wcc_component": int(node.data.get("wcc_component", -1)),
            "betweenness": float(node.data.get("betweenness", 0)),
            "confidence_score": node.confidence.score,
        }

    # Diagnostics

    async def node_count(self) -> int:
        rows = await self._run("MATCH (n:GraphNode) RETURN count(n) AS c")
        return rows[0]["c"] if rows else 0

    async def relationship_count(self) -> int:
        rows = await self._run("MATCH ()-[r:RELATES]->() RETURN count(r) AS c")
        return rows[0]["c"] if rows else 0


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
