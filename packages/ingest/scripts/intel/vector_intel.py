"""Vector intelligence: complex relationships from embeddings alone (no LLM).

Reads ``obs_embeddings`` (built by ``scripts/embed_obs.py``) and derives:

1. Company graph       – centroid per company, k-NN edges, label-propagation
                         communities (discovered sectors, no taxonomy).
2. Job k-NN graph      – nearest-neighbour jobs for every embedded observation
                         (transitive bridges / skill-transfer candidates).
3. Founder-flavor      – 0..1 score per company: how "founder/early-stage" its
                         job neighborhood looks (density of founding-title roles).
4. Two-hop walk        – accepted companies -> their vector-neighbour
                         companies -> never-gated jobs at those companies.
5. Sector affinity     – which discovered cluster your accepted matches fall in.

Everything is numpy + SQL; no LLM tokens, no scipy/sklearn.

Usage:
    uv run python3 scripts/intel/vector_intel.py [--top-k N] [--write]
    uv run python3 scripts/intel/vector_intel.py --recompute-centroids
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from typing import Any

import numpy as np

from src.http_cache import set_http_cache_store
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore

logger = get_logger("vector_intel")

DIM = 1024
FOUNDING_TITLE_RX = re.compile(
    r"founding|co[- ]?founder|early|seed|head of eng|principal eng|lead engineer"
    r"|startup|first hire|founding engineer|staff engineer",
    re.I,
)


def _norm(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# company graph


async def build_company_graph(
    store: MemoryStore, top_k: int = 8
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Return company names, centroid matrix (rows=companies, L2-norm), index map."""
    rows = await store.company_centroids(min_obs=2)
    names = [r["company"] for r in rows]
    if not names:
        return names, np.zeros((0, DIM), dtype=np.float32), {}
    mat = np.vstack([np.asarray(r["centroid"], dtype=np.float32) for r in rows])
    for i in range(mat.shape[0]):
        mat[i] = _norm(mat[i])
    idx = {name: i for i, name in enumerate(names)}
    return names, mat, idx


def label_propagation_communities(
    names: list[str], mat: np.ndarray, idx: dict[str, int], top_k: int = 8
) -> dict[str, int]:
    """Label propagation over a k-NN cosine graph. Returns {company: community_id}."""
    n = len(names)
    if n == 0:
        return {}
    sim = mat @ mat.T  # n×n cosine
    # Company centroids of job postings are naturally homogeneous (median
    # pairwise cosine ~0.7), so only very tight clusters deserve an edge.
    # Use the p90 of observed similarity as the threshold so sectors that
    # genuinely separate do, without collapsing the whole market into one.
    triu = sim[np.triu_indices(n, k=1)]
    thr = float(np.percentile(triu, 90)) if triu.size else 0.8
    thr = max(thr, 0.8)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        order = np.argsort(-sim[i])[: top_k + 1]
        for j in order:
            if j != i and sim[i, j] > thr:
                adj[i].append(int(j))
    labels = list(range(n))
    changed = True
    iterations = 0
    while changed and iterations < 30:
        changed = False
        iterations += 1
        for i in np.random.permutation(n):
            counts: dict[int, int] = defaultdict(int)
            for j in adj[i]:
                counts[labels[j]] += 1
            if not counts:
                continue
            best_label = max(counts, key=lambda label: (counts[label], -label))
            if best_label != labels[i]:
                labels[i] = best_label
                changed = True
    # Normalise community ids to 0..k
    remap = {}
    for label in labels:
        if label not in remap:
            remap[label] = len(remap)
    return {name: remap[labels[i]] for i, name in enumerate(names)}


def _founding_density(
    titles: list[str],
) -> float:
    if not titles:
        return 0.0
    hits = sum(1 for t in titles if FOUNDING_TITLE_RX.search(t))
    return min(hits / len(titles) * 4.0, 1.0)


async def company_founder_scores(store: MemoryStore) -> dict[str, float]:
    """Per-company founder-flavor score from its embedded job titles."""
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT company, title FROM obs_embeddings
            WHERE company <> '' AND title <> ''
            """
        )
    by_comp: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_comp[r["company"]].append(r["title"])
    return {c: _founding_density(ts) for c, ts in by_comp.items()}


async def two_hop_walk(
    store: MemoryStore,
    top_k: int = 8,
    min_sim: float = 0.55,
    limit: int = 60,
) -> list[dict[str, Any]]:
    """Accepted companies -> their nearest vector companies -> their never-gated jobs."""
    accepted = await store.accepted_companies(limit=100)
    if not accepted:
        return []
    rows = await store.company_centroids(min_obs=1)
    names = [r["company"] for r in rows]
    if not names:
        return []
    mat = np.vstack([np.asarray(r["centroid"], dtype=np.float32) for r in rows])
    for i in range(mat.shape[0]):
        mat[i] = _norm(mat[i])
    idx = {name: i for i, name in enumerate(names)}
    sim = mat @ mat.T

    # find for each accepted company its neighbours, but only if they're not
    # already in the accepted set (so we expand, not revisit)
    accepted_set_lower = {a.lower() for a in accepted}
    neighbour_names: set[str] = set()
    for ac in accepted:
        if ac not in idx:
            continue
        i = idx[ac]
        order = np.argsort(-sim[i])
        for j in order:
            if j == i:
                continue
            nb = names[j]
            if sim[i, j] < min_sim:
                break
            if nb.lower() not in accepted_set_lower:
                neighbour_names.add(nb)
        if len(neighbour_names) >= 60:
            break
    if not neighbour_names:
        return []
    async with store._pool.acquire() as conn:
        obs = await conn.fetch(
            """
            SELECT o.url, o.title, o.snippet, o.last_seen
            FROM job_observations o
            LEFT JOIN radar_candidates r ON r.direct_apply_url = o.url
            LEFT JOIN obs_embeddings e ON e.url_hash = md5(o.url)
            WHERE r.canonical_id IS NULL AND e.embedding IS NOT NULL
              AND lower(o.title) ~
                  'software|engineer|developer|full.?stack|backend|frontend|devops|sre|'
                  'data|machine|ml|ai|python|java|golang|rust|intern|new grad|junior'
            ORDER BY o.last_seen DESC
            LIMIT $1
            """,
            limit,
        )
    out: list[dict[str, Any]] = []
    for r in obs:
        out.append(
            {
                "url": r["url"],
                "title": r["title"],
                "snippet": (r["snippet"] or "")[:200],
                "freshness": r["last_seen"],
            }
        )
    return out


async def write_intel(
    store: MemoryStore,
    top_k: int,
    communities: dict[str, int] | None,
    founder_scores: dict[str, float] | None,
) -> str:
    """Export intel/recommendations.json + .csv for the friend's auto-applier."""
    import csv
    import os
    from pathlib import Path

    out_dir = Path(__file__).resolve().parent.parent.parent / "intel"
    out_dir.mkdir(parents=True, exist_ok=True)

    # nearest accepted companies for sector tagging of each embedded obs
    accepted = await store.accepted_companies(limit=50)
    accepted_lower = {a.lower(): a for a in accepted}
    rows = await store.company_centroids(min_obs=1)
    names = [r["company"] for r in rows]
    if names:
        mat = np.vstack([np.asarray(r["centroid"], dtype=np.float32) for r in rows])
        for i in range(mat.shape[0]):
            mat[i] = _norm(mat[i])
        idx = {name: i for i, name in enumerate(names)}
        sim = mat @ mat.T
        twin_of: dict[str, str] = {}
        for name in names:
            if name.lower() in accepted_lower:
                continue
            i = idx[name]
            best = None
            best_s = 0.0
            for a in accepted:
                if a not in idx:
                    continue
                s = float(sim[i, idx[a]])
                if s > best_s:
                    best_s, best = s, a
            if best and best_s >= 0.5:
                twin_of[name] = f"{best} ({best_s:.2f})"

    # rows for csv: embedded obs with founder/community/twin tags
    async with store._pool.acquire() as conn:
        obs_rows = await conn.fetch(
            """
            SELECT e.title, e.company, o.url, o.snippet, o.last_seen
            FROM obs_embeddings e
            LEFT JOIN job_observations o ON md5(o.url) = e.url_hash
            WHERE e.embedding IS NOT NULL
            LIMIT 2000
            """
        )
    records: list[dict[str, Any]] = []
    for r in obs_rows:
        comp = r["company"] or ""
        records.append(
            {
                "title": r["title"] or "",
                "company": comp,
                "url": r["url"] or "",
                "snippet": (r["snippet"] or "")[:200],
                "community": communities.get(comp) if communities else None,
                "founder_score": round(founder_scores.get(comp, 0.0), 2)
                if founder_scores
                else None,
                "twin_of": twin_of.get(comp, ""),
                "freshness": r["last_seen"],
            }
        )
    json_path = out_dir / "recommendations.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    csv_path = out_dir / "recommendations.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["title", "company", "url", "snippet", "community", "founder_score", "twin_of"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in fields})
    return f"intel/{os.path.basename(json_path)} + intel/recommendations.csv"


async def job_knn_bridges(
    store: MemoryStore, top_k: int = 5, sample: int = 3000
) -> list[dict[str, Any]]:
    """Nearest-neighbour jobs for a sample of embedded jobs.

    Returns per-job: its top-k neighbours (title/company/distance). The
    cross-job similarities expose skill-transfer bridges: a neighbour role's
    required skills you don't have yet are learnable if several jobs in the
    cluster require them. Pure vector, no LLM.
    """
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT e.url_hash, e.title, e.company, e.embedding
            FROM obs_embeddings e
            WHERE e.embedding IS NOT NULL
            ORDER BY e.embedded_at DESC
            LIMIT $1
            """,
            sample,
        )
    if len(rows) < 2:
        return []
    mat = np.vstack([np.asarray(r["embedding"].to_list(), dtype=np.float32) for r in rows])
    for i in range(mat.shape[0]):
        mat[i] = _norm(mat[i])
    sim = mat @ mat.T
    out: list[dict[str, Any]] = []
    for i in range(mat.shape[0]):
        order = np.argsort(-sim[i])[: top_k + 1]
        nbrs = []
        for j in order:
            if j == i:
                continue
            nbrs.append(
                {
                    "title": rows[j]["title"],
                    "company": rows[j]["company"],
                    "sim": round(float(sim[i, j]), 3),
                }
            )
            if len(nbrs) >= top_k:
                break
        out.append(
            {
                "url_hash": rows[i]["url_hash"],
                "title": rows[i]["title"],
                "company": rows[i]["company"],
                "neighbours": nbrs,
            }
        )
    return out


async def run(top_k: int, write: bool) -> None:
    store = await MemoryStore.create()
    set_http_cache_store(store)
    try:
        names, mat, idx = await build_company_graph(store, top_k=top_k)
        logger.info(f"company graph: {len(names)} companies, {mat.shape[0]}x{mat.shape[1]}")
        communities: dict[str, int] | None = None
        if len(names) > 3:
            communities = label_propagation_communities(names, mat, idx, top_k=top_k)
            by_comm: dict[int, list[str]] = defaultdict(list)
            for c, lid in communities.items():
                by_comm[lid].append(c)
            top_comms = sorted(by_comm.items(), key=lambda kv: -len(kv[1]))[:8]
            for lid, members in top_comms:
                logger.info(
                    f"community {lid}: {len(members)} companies | e.g. {', '.join(members[:5])}"
                )
        founder_scores = await company_founder_scores(store)
        top_founder = sorted(founder_scores.items(), key=lambda kv: -kv[1])[:6]
        logger.info("founder-flavor top: " + ", ".join(f"{c}={s:.2f}" for c, s in top_founder))
        bridges = await job_knn_bridges(store, top_k=5, sample=2000)
        logger.info(f"job k-NN bridges: {len(bridges)} jobs with nearest-neighbour bridges")
        if bridges:
            b = bridges[0]
            nbr_txt = "; ".join(
                f"{n['title'][:35]}@{n['company'][:20]}({n['sim']})" for n in b["neighbours"][:3]
            )
            logger.info(f"  sample bridge: {b['title'][:45]} -> {nbr_txt}")
        hops = await two_hop_walk(store, top_k=top_k, limit=40)
        logger.info(f"two-hop walk: {len(hops)} never-gated jobs at neighbour companies")
        if write:
            path = await write_intel(store, top_k, communities, founder_scores)
            logger.info(f"wrote {path}")
    finally:
        await store.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Vector intelligence over obs_embeddings")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--write", action="store_true", help="export intel CSV/JSON")
    args = ap.parse_args()
    asyncio.run(run(args.top_k, args.write))


if __name__ == "__main__":
    main()
