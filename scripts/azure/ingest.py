"""Local ingest loop: pull Azure blob blobs into local Postgres.

Connects to the same Azure storage the VM worker uploads to, downloads
every `obs/*.jsonl` and `companies/*.jsonl` blob once (tracked in a
Postgres marker table), and writes rows into job_observations and
companies_index. Run from the project root:

    AZURE_STORAGE_ACCOUNT=... AZURE_STORAGE_KEY=... uv run --with azure-storage-blob \
        python3 scripts/azure_ingest.py

The orchestrator sweeps job_observations on its own schedule, so ingested
postings flow through gating/matching exactly like local discoveries.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from azure.storage.blob import BlobServiceClient

from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore
from src.radar.core.models import JobObservation

logger = get_logger("azure_ingest")


def _posting_id(obs: JobObservation) -> str:
    return obs.canonical_url_hash()


def _raw_json_value(raw_markdown: str) -> str:
    """Store the ATS item JSON into the jsonb raw_json column.

    raw_markdown is the full JSON of the posting as captured by the
    crawl worker. If it parses as JSON keep it as an object (jsonb),
    otherwise fall back to an empty object so the gate can still use it.
    """
    try:
        parsed = json.loads(raw_markdown) if raw_markdown else None
    except Exception:
        parsed = None
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed)
    return "{}"


async def _ensure_tables() -> MemoryStore:
    store = await MemoryStore.create()
    async with store._pool.acquire() as conn:
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS companies_index (
                slug TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                careers_url TEXT DEFAULT '',
                name TEXT DEFAULT '',
                location TEXT DEFAULT '',
                job_count INTEGER DEFAULT 0,
                first_seen DOUBLE PRECISION DEFAULT 0,
                last_seen DOUBLE PRECISION DEFAULT 0
            )"""
        )
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS azure_ingest_marker (
                blob TEXT PRIMARY KEY,
                ingested_at DOUBLE PRECISION NOT NULL
            )"""
        )
    return store


async def _persist_observation(store, obs: JobObservation) -> None:
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO job_observations (url_hash, url, source, title, snippet,
                    first_seen, last_seen, freshness_lane, direct_posting_verified, raw_json)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT (url_hash) DO UPDATE SET
                    last_seen = EXCLUDED.last_seen,
                    raw_json = CASE
                        WHEN EXCLUDED.raw_json <> '{}'::jsonb
                        THEN EXCLUDED.raw_json
                        ELSE job_observations.raw_json
                    END""",
                _posting_id(obs),
                obs.url,
                obs.source,
                obs.title or "",
                obs.snippet or "",
                obs.observed_at,
                obs.observed_at,
                "review",
                not obs.source.startswith("github_index:"),
                _raw_json_value(obs.raw_markdown),
            )
    except Exception as exc:
        logger.warning(f"persist observation {obs.url[:60]}: {exc}")


async def _ingest(store) -> None:
    conn_str = (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={os.environ['AZURE_STORAGE_ACCOUNT']};"
        f"AccountKey={os.environ['AZURE_STORAGE_KEY']};"
        "EndpointSuffix=core.windows.net"
    )
    container = os.environ.get("AZURE_CONTAINER", "radar-index")
    svc = BlobServiceClient.from_connection_string(conn_str)
    cc = svc.get_container_client(container)

    async with store._pool.acquire() as conn:
        done = {r["blob"] for r in await conn.fetch("SELECT blob FROM azure_ingest_marker")}

    obs_rows = 0
    comp_rows = 0
    for prefix, handler in (("obs/", "obs"), ("companies/", "companies")):
        for blob in cc.list_blobs(name_starts_with=prefix):
            if blob.name in done:
                continue
            try:
                data = cc.get_blob_client(blob.name).download_blob().readall()
                records = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
                if handler == "obs":
                    for rec in records:
                        obs = JobObservation(
                            url=rec.get("url", ""),
                            source=rec.get("source", "azure"),
                            title=rec.get("title", ""),
                            snippet=rec.get("snippet", ""),
                            raw_markdown=rec.get("raw_markdown", ""),
                            observed_at=rec.get("observed_at", time.time()),
                            source_freshness_evidence=rec.get("source_freshness_evidence"),
                        )
                        if obs.url.startswith("http"):
                            await _persist_observation(store, obs)
                            obs_rows += 1
                else:
                    for rec in records:
                        await _persist_company(store, rec)
                        comp_rows += 1
                async with store._pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO azure_ingest_marker (blob, ingested_at) VALUES ($1, $2)",
                        blob.name,
                        time.time(),
                    )
            except Exception as exc:
                logger.warning(f"ingest {blob.name}: {exc}")
    if obs_rows or comp_rows:
        logger.info(f"Ingested {obs_rows} observations, {comp_rows} company rows")


async def _persist_company(store, rec: dict) -> None:
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO companies_index (slug, platform, careers_url, name, location,
                    job_count, first_seen, last_seen)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (slug) DO UPDATE SET
                    platform = EXCLUDED.platform,
                    careers_url = EXCLUDED.careers_url,
                    name = EXCLUDED.name,
                    location = EXCLUDED.location,
                    job_count = GREATEST(companies_index.job_count, EXCLUDED.job_count),
                    last_seen = EXCLUDED.last_seen""",
                rec.get("slug", ""),
                rec.get("platform", ""),
                rec.get("careers_url", ""),
                rec.get("name", ""),
                rec.get("location", ""),
                rec.get("job_count", 0),
                rec.get("first_seen", time.time()),
                rec.get("last_seen", time.time()),
            )
    except Exception as exc:
        logger.warning(f"persist company {rec.get('slug')}: {exc}")


async def main() -> None:
    while True:
        try:
            store = await _ensure_tables()
            break
        except Exception as exc:
            logger.warning(f"db not ready: {exc}; retrying in 15s")
            await asyncio.sleep(15)
    while True:
        try:
            await _ingest(store)
        except Exception as exc:
            logger.warning(f"ingest cycle failed: {exc}")
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
