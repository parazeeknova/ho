"""Exit 0 if there is unembedded corpus work for embed_obs.py, 1 otherwise.

Used by the watchdog to decide whether to relaunch the embedding backfill
without shelling out to psql.
"""

from __future__ import annotations

import asyncio

from src.memory.pgvector_store import MemoryStore


async def _main() -> int:
    try:
        s = await MemoryStore.create()
        try:
            async with s._pool.acquire() as c:
                n = await c.fetchval(
                    "SELECT count(*) FROM job_observations o "
                    "LEFT JOIN obs_embeddings e ON e.url_hash = md5(o.url) "
                    "WHERE e.url_hash IS NULL LIMIT 1"
                )
        finally:
            await s.close()
        return 0 if n else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
