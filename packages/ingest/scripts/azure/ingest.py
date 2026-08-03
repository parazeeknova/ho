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
from pathlib import Path

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
    if raw_markdown:
        # Null bytes are valid in JSON strings but illegal in Postgres
        # jsonb - strip them so the bulk COPY never chokes.
        raw_markdown = raw_markdown.replace("\\u0000", "").replace("\x00", "")
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


def _clean_text(val: str | None) -> str:
    """Strip characters Postgres text/jsonb rejects (null bytes etc.)."""
    if not val:
        return ""
    return val.replace("\x00", "").replace("\\u0000", "")


async def _bulk_persist_observations(store, obs_list: list[JobObservation]) -> int:
    """Bulk-insert observations via COPY into a temp table + merge.

    Asyncpg's COPY is ~50x faster than per-row INSERTs, which matters when
    a single hourly blob holds 200k+ postings.
    """
    if not obs_list:
        return 0
    rows: list[tuple] = []
    for obs in obs_list:
        if not obs.url.startswith("http"):
            continue
        rows.append(
            (
                _posting_id(obs),
                obs.url,
                _clean_text(obs.source),
                _clean_text(obs.title),
                _clean_text(obs.snippet),
                obs.observed_at,
                obs.observed_at,
                "review",
                not obs.source.startswith("github_index:"),
                _raw_json_value(obs.raw_markdown),
            )
        )
    if not rows:
        return 0
    async with store._pool.acquire() as conn:
        await conn.execute(
            """CREATE TEMP TABLE IF NOT EXISTS _ingest_obs (
                url_hash TEXT, url TEXT, source TEXT, title TEXT, snippet TEXT,
                first_seen DOUBLE PRECISION, last_seen DOUBLE PRECISION,
                freshness_lane TEXT, direct_posting_verified BOOLEAN, raw_json TEXT
            )"""
        )
        await conn.execute("TRUNCATE _ingest_obs")
        await conn.copy_records_to_table(
            "_ingest_obs",
            records=rows,
            columns=[
                "url_hash",
                "url",
                "source",
                "title",
                "snippet",
                "first_seen",
                "last_seen",
                "freshness_lane",
                "direct_posting_verified",
                "raw_json",
            ],
        )
        await conn.execute(
            """INSERT INTO job_observations (url_hash, url, source, title, snippet,
                first_seen, last_seen, freshness_lane, direct_posting_verified, raw_json)
            SELECT url_hash, url, source, title, snippet,
                first_seen, last_seen, freshness_lane, direct_posting_verified,
                raw_json::jsonb
            FROM _ingest_obs
            ON CONFLICT (url_hash) DO UPDATE SET
                last_seen = EXCLUDED.last_seen,
                raw_json = CASE
                    WHEN EXCLUDED.raw_json <> '{}'::jsonb
                    THEN EXCLUDED.raw_json
                    ELSE job_observations.raw_json
                END"""
        )
    return len(rows)


def _local_blobs(root: Path, prefix: str) -> list[tuple[str, Path]]:
    """Local mode: list (name, path) JSONL files under root for a prefix."""
    out: list[tuple[str, Path]] = []
    base = root / prefix
    if not base.exists():
        return out
    for p in sorted(base.glob("*.jsonl")):
        out.append((f"{prefix}{p.name}", p))
    return out


async def _ingest(store) -> None:
    local_root = os.environ.get("CRAWL_OUT", "").strip()
    cc = None
    if local_root:
        root = Path(local_root)
        root.mkdir(parents=True, exist_ok=True)
    else:
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
    osint_rows = 0
    for prefix, handler in (
        ("obs/", "obs"),
        ("companies/", "companies"),
        ("founders/", "osint"),
        ("signals/", "osint"),
    ):
        if local_root:
            blobs: list[tuple[str, Path | None]] = [
                (name, p) for name, p in _local_blobs(root, prefix)
            ]
        else:
            blobs = [(b.name, None) for b in cc.list_blobs(name_starts_with=prefix)]  # type: ignore[union-attr]
        for name, path in blobs:
            if name in done:
                continue
            try:
                if local_root:
                    data = path.read_bytes()  # type: ignore[union-attr]
                else:
                    data = cc.get_blob_client(name).download_blob().readall()  # type: ignore[union-attr]
                records = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
                if handler == "obs":
                    batch: list[JobObservation] = []
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
                            batch.append(obs)
                    obs_rows += await _bulk_persist_observations(store, batch)
                elif handler == "companies":
                    batch_comp: list[dict] = []
                    for rec in records:
                        batch_comp.append(rec)
                    comp_rows += await _bulk_persist_companies(store, batch_comp)
                else:  # founders/ + signals/ -> company_osint
                    osint_rows += await _persist_osint_records(store, records)
                async with store._pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO azure_ingest_marker (blob, ingested_at) VALUES ($1, $2)",
                        name,
                        time.time(),
                    )
            except Exception as exc:
                logger.warning(f"ingest {name}: {exc}")
    if obs_rows or comp_rows or osint_rows:
        logger.info(
            f"Ingested {obs_rows} observations, {comp_rows} company rows, {osint_rows} osint rows"
        )


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


async def _bulk_persist_companies(store, rec_list: list[dict]) -> int:
    if not rec_list:
        return 0
    now = time.time()
    rows = [
        (
            r.get("slug", ""),
            r.get("platform", ""),
            r.get("careers_url", ""),
            r.get("name", ""),
            r.get("location", ""),
            r.get("job_count", 0),
            r.get("first_seen", now),
            r.get("last_seen", now),
        )
        for r in rec_list
    ]
    async with store._pool.acquire() as conn:
        await conn.execute("CREATE TEMP TABLE IF NOT EXISTS _ingest_comp (LIKE companies_index)")
        await conn.execute("TRUNCATE _ingest_comp")
        await conn.copy_records_to_table(
            "_ingest_comp",
            records=rows,
            columns=[
                "slug",
                "platform",
                "careers_url",
                "name",
                "location",
                "job_count",
                "first_seen",
                "last_seen",
            ],
        )
        await conn.execute(
            """INSERT INTO companies_index (slug, platform, careers_url, name, location,
                job_count, first_seen, last_seen)
            SELECT slug, platform, careers_url, name, location,
                job_count, first_seen, last_seen
            FROM _ingest_comp
            ON CONFLICT (slug) DO UPDATE SET
                platform = EXCLUDED.platform,
                careers_url = EXCLUDED.careers_url,
                name = EXCLUDED.name,
                location = EXCLUDED.location,
                job_count = GREATEST(companies_index.job_count, EXCLUDED.job_count),
                last_seen = EXCLUDED.last_seen"""
        )
    return len(rows)


async def _persist_osint_records(store, records: list[dict]) -> int:
    """Merge founder/funding/signal records into company_osint.

    Each record carries a ``company`` (or ``slug``) key; we merge the record
    into the company's existing OSINT payload under a ``kind`` namespace so
    founder + funding + signal data from different workers accumulate.
    """
    merged: dict[str, dict] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        company = (rec.get("company") or rec.get("slug") or "").strip()
        if not company:
            continue
        kind = "founders" if rec.get("name") and rec.get("title") else "signals"
        if company not in merged:
            merged[company] = {"founders": [], "signals": []}
        payload = {k: v for k, v in rec.items() if k not in ("company", "slug")}
        merged[company][kind].append(payload)
    if not merged:
        return 0
    async with store._pool.acquire() as conn:
        for company, data in merged.items():
            await conn.execute(
                """INSERT INTO company_osint (company, data, cached_at, expires_at)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (company) DO UPDATE SET
                       data = company_osint.data || $2,
                       cached_at = GREATEST(company_osint.cached_at, $3),
                       expires_at = GREATEST(company_osint.expires_at, $4)""",
                company,
                json.dumps(data),
                time.time(),
                time.time() + 30 * 86400,
            )
    return len(merged)


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
