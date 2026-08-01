#!/usr/bin/env python3
"""Daily versioned backups of Postgres + Neo4j to an R2 bucket.

Python edition, per royal decree.

  pg:     pg_dump -Fc (custom format) piped from inside the container
  neo4j:  container briefly stopped, neo4j-admin database dump, restarted

Keeps the last BACKUP_KEEP dailies in <remote>/daily/<ts>/. Logs to
logs/backup.log. While running, holds logs/backup.lock so the watchdog
does not fight the intentional neo4j stop.

Credentials: scripts/.r2.env (git-ignored) with
  R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
or the environment variable R2_REMOTE for testing (e.g. "ho-test:backups").

Usage:
    python3 scripts/backup.py            # backup to R2
    python3 scripts/backup.py r2:.../daily/20260101-000000   # custom dest
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT / "logs" / "backup.log"
LOCK_PATH = PROJECT / "logs" / "backup.lock"
STAGING_ROOT = Path("/tmp/ho-backups")
KEEP = int(os.environ.get("BACKUP_KEEP", "10"))

PG_CONTAINER = "firecrawl_agent-memory-db_1"
NEO4J_CONTAINER = "firecrawl_neo4j_1"
NEO4J_VOLUME = "firecrawl_neo4j_data"
NEO4J_IMAGE = "neo4j:community-ubi10"

FAILED = False
NEO4J_COUNTS: dict[str, int] = {}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def container_running(name: str) -> bool:
    r = sh(["podman", "ps", "--format", "{{.Names}}"])
    return any(line.strip() == name for line in r.stdout.splitlines())


def load_r2_env() -> tuple[str, list[str]]:
    env_file = PROJECT / "scripts" / ".r2.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
    remote = os.environ.get("R2_REMOTE")
    flags: list[str] = []
    if not remote:
        for key in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
            if not os.environ.get(key):
                log(f"ERROR: {key} not set (fill scripts/.r2.env or set R2_REMOTE)")
                sys.exit(1)
        os.environ.setdefault("RCLONE_CONFIG_R2_TYPE", "s3")
        os.environ.setdefault("RCLONE_CONFIG_R2_PROVIDER", "Cloudflare")
        os.environ.setdefault("RCLONE_CONFIG_R2_ENDPOINT", os.environ["R2_ENDPOINT"])
        os.environ.setdefault("RCLONE_CONFIG_R2_ACCESS_KEY_ID", os.environ["R2_ACCESS_KEY_ID"])
        os.environ.setdefault(
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", os.environ["R2_SECRET_ACCESS_KEY"]
        )
        os.environ.setdefault("RCLONE_CONFIG_R2_NO_CHECK_BUCKET", "true")
        flags = ["--s3-no-check-bucket"]
        remote = f"r2:{os.environ['R2_BUCKET']}"
    return remote, flags


def pg_dump(staging: Path) -> None:
    global FAILED
    if not container_running(PG_CONTAINER):
        log("WARN: agent-memory-db not running, skipping pg dump")
        FAILED = True
        return
    out = staging / "pg_agent_memory.dump"
    try:
        with out.open("wb") as fh:
            r = subprocess.run(
                [
                    "podman",
                    "exec",
                    PG_CONTAINER,
                    "pg_dump",
                    "-Fc",
                    "-U",
                    "postgres",
                    "-d",
                    "agent_memory",
                ],
                stdout=fh,
                stderr=subprocess.PIPE,
            )
        if r.returncode != 0:
            log(f"ERROR: pg_dump failed: {r.stderr.decode()[:300]}")
            FAILED = True
            return
        with out.open("rb") as src, gzip.open(str(out) + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        out.unlink()
        gz_path = out.parent / (out.name + ".gz")
        log(f"pg dump: {gz_path} ({gz_path.stat().st_size // 1024} KiB)")
    except Exception as exc:
        log(f"ERROR: pg_dump exception: {exc}")
        FAILED = True


def neo4j_stop_with_retry() -> bool:
    """Stop neo4j for an offline dump. Podman may need a SIGKILL after
    the 10s grace (neo4j takes a while), and transient states can fail
    the first attempt, so retry once and surface the real error."""
    for attempt in (1, 2):
        r = sh(["podman", "stop", NEO4J_CONTAINER])
        if r.returncode == 0:
            return True
        log(f"podman stop attempt {attempt} failed: {r.stderr.strip()[-200:]}")
        time.sleep(3)
    return False


def neo4j_dump(staging: Path) -> None:
    global FAILED, NEO4J_COUNTS
    if not container_running(NEO4J_CONTAINER):
        log("WARN: neo4j not running, skipping neo4j dump")
        FAILED = True
        return
    NEO4J_COUNTS = neo4j_counts()
    log("stopping neo4j for consistent dump...")
    if not neo4j_stop_with_retry():
        log("ERROR: could not stop neo4j")
        FAILED = True
        return
    try:
        # The helper container's neo4j user must be able to write /out.
        staging.chmod(0o777)
        dump_dir = staging / "neo4j.dump"
        r = sh(
            [
                "podman",
                "run",
                "--rm",
                "-v",
                f"{NEO4J_VOLUME}:/data",
                "-v",
                f"{staging}:/out",
                NEO4J_IMAGE,
                "neo4j-admin",
                "database",
                "dump",
                "neo4j",
                "--to-path=/out",
            ],
            timeout=900,
        )
        if r.returncode != 0:
            log(f"ERROR: neo4j dump failed: {r.stderr[-300:]}")
            FAILED = True
            return
        gz = staging / "neo4j_neo4j.tar.gz"
        with tarfile.open(gz, "w:gz") as tar:
            tar.add(dump_dir, arcname="neo4j.dump")
        dump_dir.unlink()
        log(f"neo4j dump: {gz} ({gz.stat().st_size // 1024} KiB)")
    except subprocess.TimeoutExpired:
        log("ERROR: neo4j dump timed out")
        FAILED = True
    finally:
        sh(["podman", "start", NEO4J_CONTAINER])
        log("neo4j restarted")


def pg_counts() -> dict[str, int]:
    """Row counts from the live Postgres database for metadata.json."""
    if not container_running(PG_CONTAINER):
        return {}
    try:
        r = subprocess.run(
            [
                "podman",
                "exec",
                PG_CONTAINER,
                "psql",
                "-U",
                "postgres",
                "-d",
                "agent_memory",
                "-t",
                "-A",
                "-c",
                (
                    "SELECT 'observations=' || count(*) FROM job_observations "
                    "UNION ALL SELECT 'candidates=' || count(*) FROM radar_candidates "
                    "UNION ALL SELECT 'accepted=' || count(*) FROM radar_candidates "
                    "WHERE eligibility='accepted' "
                    "UNION ALL SELECT 'sources=' || count(*) FROM source_checkpoints "
                    "UNION ALL SELECT 'notified=' || count(*) FROM telegram_notified_jobs"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out: dict[str, int] = {}
        for line in r.stdout.splitlines():
            key, _, val = line.partition("=")
            if val.strip().isdigit():
                out[key.strip()] = int(val)
        return out
    except Exception:
        return {}


def neo4j_counts() -> dict[str, int]:
    """Node/edge counts from the live Neo4j database for metadata.json."""
    if not container_running(NEO4J_CONTAINER):
        return {}
    try:
        r = subprocess.run(
            [
                "podman",
                "exec",
                NEO4J_CONTAINER,
                "cypher-shell",
                "-u",
                "neo4j",
                "-p",
                "password",
                "MATCH (n) RETURN count(n) AS nodes; MATCH ()-[r]->() RETURN count(r) AS edges;",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        digits = [int(t) for t in r.stdout.split() if t.strip().isdigit()]
        return {"nodes": digits[0], "edges": digits[1]} if len(digits) >= 2 else {}
    except Exception:
        return {}


def write_metadata(staging: Path) -> None:
    """Write metadata.json (object counts, sizes, timestamps) into the snapshot."""
    meta: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": 1,
        "retention": KEEP,
        "pg": pg_counts(),
        "neo4j": NEO4J_COUNTS,
        "files": {},
    }
    for f in sorted(staging.iterdir()):
        if f.is_file():
            meta["files"][f.name] = f.stat().st_size
    try:
        (staging / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        log(f"metadata: {json.dumps(meta['pg'])} pg, {json.dumps(meta['neo4j'])} neo4j")
    except Exception as exc:
        log(f"WARN: metadata write failed: {exc}")


def upload(staging: Path, dest: str, flags: list[str]) -> None:
    global FAILED
    r = sh(["rclone", "copyto", str(staging), dest, *flags], timeout=3600)
    if r.returncode != 0:
        log(f"ERROR: rclone upload failed: {r.stderr[-300:]}")
        FAILED = True
        return
    log(f"uploaded → {dest}")


def prune(remote: str, flags: list[str]) -> None:
    r = sh(["rclone", "lsf", f"{remote}/daily/", "--dirs-only", *flags])
    snaps = sorted(
        (line.strip().rstrip("/") for line in r.stdout.splitlines()),
        reverse=True,
    )
    for old in snaps[KEEP:]:
        if not old or not old.replace("-", "").isdigit():
            continue
        log(f"pruning old backup: {old}")
        sh(["rclone", "purge", f"{remote}/daily/{old}", *flags])


def main() -> None:
    global FAILED
    remote, flags = load_r2_env()
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = sys.argv[1] if len(sys.argv) > 1 else f"{remote}/daily/{ts}"
    staging = STAGING_ROOT / ts
    staging.mkdir(parents=True, exist_ok=True)

    LOCK_PATH.parent.mkdir(exist_ok=True)
    LOCK_PATH.touch()
    try:
        log(f"=== backup {ts} → {dest} ===")
        pg_dump(staging)
        neo4j_dump(staging)
        if not FAILED:
            write_metadata(staging)
            upload(staging, dest, flags)
        prune(remote, flags)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        LOCK_PATH.unlink(missing_ok=True)

    log("backup complete ✓" if not FAILED else "backup completed WITH ERRORS ✗")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
