"""Local analytics: storage + container + DB numbers for the ho project.

Reports:
  - Disk usage of the project, azure_dump, intel, logs
  - Container list + status + volume mounts + memory
  - Named podman volumes used by the firecrawl stack (size on disk)
  - Database row counts (observations, candidates, companies, embeddings)
  - Process counts for the pipeline workers

Run:
    uv run python scripts/analytics.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore

logger = get_logger("analytics")

PROJECT = Path(__file__).resolve().parent.parent

CONTAINER_VOLUME_MAP = {
    "firecrawl_agent-memory-db_1": "firecrawl_agent_memory_data",
    "firecrawl_neo4j_1": "firecrawl_neo4j_data",
    "firecrawl_nuq-postgres_1": None,  # anonymous volume
    "firecrawl_redis_1": None,
    "firecrawl_searxng_1": None,
    "firecrawl_rabbitmq_1": None,
}


def _sh(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        r = subprocess.run(
            ["du", "-sk", str(p)], capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            # du -sk gives KiB
            return int(r.stdout.split()[0]) * 1024
    except Exception:
        pass
    return 0


def _fmt(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _process_count(pattern: str) -> int:
    out = _sh(f"ps -eo cmd | grep -E '{pattern}' | grep -v grep | grep -v 'uv ' | wc -l")
    try:
        return int(out or "0")
    except ValueError:
        return 0


def _containers() -> list[dict]:
    out = _sh("podman ps --format '{{.Names}}\\t{{.Status}}'")
    containers: list[dict] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        containers.append({"name": parts[0], "status": parts[1] if len(parts) > 1 else ""})
    return containers


def _volume_sizes() -> dict[str, str]:
    """Size named volumes. Postgres data lives in a mountpoint junction, so
    measure inside the mounting container with `du` for accuracy."""
    # container -> (volume, path-in-container)
    known: list[tuple[str, str, str]] = [
        ("firecrawl_agent-memory-db_1", "firecrawl_agent_memory_data", "/var/lib/postgresql"),
        ("firecrawl_neo4j_1", "firecrawl_neo4j_data", "/data"),
        ("firecrawl_neo4j_1", "firecrawl_neo4j_logs", "/logs"),
    ]
    sizes: dict[str, str] = {}
    for container, vol, dest in known:
        out = _sh(f"podman exec {container} du -sk {dest} 2>/dev/null")
        if out.strip():
            try:
                sizes[vol] = _fmt(int(out.split()[0]) * 1024)
                continue
            except (ValueError, IndexError):
                pass
        # fallback to raw path
        p = Path(
            f"/home/parazeeknova/.local/share/containers/storage/volumes/{vol}/_data"
        )
        if p.exists():
            sizes[vol] = _fmt(_dir_size(p))
    return sizes


async def _db_counts() -> dict[str, int]:
    try:
        s = await MemoryStore.create()
        async with s._pool.acquire() as c:
            return {
                "observations": await c.fetchval("SELECT count(*) FROM job_observations") or 0,
                "candidates": await c.fetchval("SELECT count(*) FROM radar_candidates") or 0,
                "accepted": await c.fetchval(
                    "SELECT count(*) FROM radar_candidates WHERE eligibility='accepted'"
                )
                or 0,
                "near_miss": await c.fetchval(
                    "SELECT count(*) FROM radar_candidates WHERE eligibility='near_miss'"
                )
                or 0,
                "embeddings": await c.fetchval("SELECT count(*) FROM obs_embeddings") or 0,
                "companies": await c.fetchval("SELECT count(*) FROM companies_index") or 0,
                "osint": await c.fetchval("SELECT count(*) FROM company_osint") or 0,
            }
    except Exception:
        return {}
    finally:
        try:
            await s.close()
        except Exception:
            pass


def main() -> None:
    print("=" * 60)
    print("  HO LOCAL ANALYTICS")
    print("=" * 60)

    print("\n-- Disk Usage --")
    for name, path in (
        ("project (excluding venv/dump)", PROJECT),
        ("azure_dump", PROJECT / "azure_dump"),
        ("intel", PROJECT / "intel"),
        ("logs", PROJECT / "logs"),
    ):
        size = 0
        if name.startswith("project"):
            size = _dir_size(PROJECT)
            for excl in (PROJECT / "azure_dump", PROJECT / ".venv", PROJECT / ".devenv", PROJECT / "checkpoints"):
                size -= _dir_size(excl)
        else:
            size = _dir_size(path)
        print(f"  {name}: {_fmt(size)}")

    print("\n-- Containers --")
    containers = _containers()
    if containers:
        for c in containers:
            print(f"  {c['name']}: {c['status']}")
    else:
        print("  (no running containers)")

    print("\n-- Named Volumes (size on disk) --")
    vsizes = _volume_sizes()
    for vol in ("firecrawl_agent_memory_data", "firecrawl_neo4j_data", "firecrawl_neo4j_logs"):
        if vol in vsizes:
            print(f"  {vol}: {vsizes[vol]}")

    print("\n-- Process Counts --")
    for pat, label in (
        ("src[.]radar[.]engine[.]orchestrator", "orchestrators"),
        ("scripts/run[.]py", "run.py"),
        ("azure/ingest[.]py", "azure ingest"),
        ("intel_loop[.]py", "intel loop"),
        ("smart_intel_loop[.]py", "smart intel loop"),
    ):
        print(f"  {label}: {_process_count(pat)}")

    print("\n-- Database --")
    counts = asyncio.run(_db_counts())
    if counts:
        for k, v in counts.items():
            print(f"  {k}: {v:,}")
    else:
        print("  (database unavailable)")

    print("\n-- System --")
    mem = _sh("free -h | head -2 | tail -1")
    if mem:
        parts = mem.split()
        print(f"  memory used: {parts[2]} / {parts[1]}")
    disk = shutil.disk_usage(str(PROJECT))
    print(f"  disk free: {_fmt(disk.free)} / {_fmt(disk.total)}")


if __name__ == "__main__":
    main()
