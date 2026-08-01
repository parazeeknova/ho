"""Overnight health monitor: checks the radar pipeline + Azure ingest every N
seconds and logs a compact health line. Does NOT fix anything; it's the eyes
for the night shift. Run detached:
    setsid nohup env PYTHONPATH=$PWD uv run python3 scripts/overnight_monitor.py > logs/monitor.out 2>&1 &
"""

from __future__ import annotations

import asyncio
import os
import time

from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore

logger = get_logger("overnight_monitor")
CHECK_EVERY = 240  # seconds
REPORT_EVERY = 20  # checks between full reports


def count_of(pattern: str) -> int:
    import subprocess

    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:
        return -1


async def check_once(store: MemoryStore) -> dict:
    async with store._pool.acquire() as c:
        return {
            "embedded": await c.fetchval("SELECT count(*) FROM obs_embeddings"),
            "accepted": await c.fetchval(
                "SELECT count(*) FROM radar_candidates WHERE eligibility='accepted'"
            ),
            "near_miss": await c.fetchval(
                "SELECT count(*) FROM radar_candidates WHERE eligibility='near_miss'"
            ),
            "queue_pending": await c.fetchval(
                "SELECT count(*) FROM llm_queue WHERE status IN ('pending','processing')"
            ),
            "obs": await c.fetchval("SELECT count(*) FROM job_observations"),
        }


async def main() -> None:
    store = await MemoryStore.create()
    ticks = 0
    last_report = 0
    while True:
        try:
            tick = time.monotonic()
            orch = count_of(r"radar[.]engine[.]orchestrator")
            ingest = count_of(r"scripts/azure/ingest[.]py")
            embed = count_of(r"scripts/embed_obs")
            wd = count_of(r"scripts/watchdog")
            stats = await check_once(store)
            line = (
                f"orch={orch} ingest={ingest} embed={embed} watchdog={wd} "
                f"accepted={stats['accepted']} near_miss={stats['near_miss']} "
                f"queue={stats['queue_pending']} embedded={stats['embedded']} "
                f"obs={stats['obs']}"
            )
            if ticks % REPORT_EVERY == 0 or stats["queue_pending"] > 500:
                logger.info("MONITOR " + line)
            elif stats["queue_pending"] % 50 == 0:
                logger.info("MONITOR " + line)
            if stats["accepted"] != last_report:
                logger.info(f"MONITOR accepted delta: {stats['accepted']} (was {last_report})")
                last_report = stats["accepted"]
        except Exception as e:
            logger.warning(f"monitor tick error: {e}")
        ticks += 1
        await asyncio.sleep(CHECK_EVERY)


if __name__ == "__main__":
    asyncio.run(main())
