#!/usr/bin/env python3
"""Restore local Postgres/Neo4j volumes from the latest R2 backup.

Python edition, per royal decree. Called explicitly for disaster
recovery, or automatically by run.py at pipeline startup with
--if-empty: when a volume is found fresh (no PG_VERSION / databases
dir) and a cloud snapshot exists, it is pulled and restored.

Usage:
    python3 scripts/restore.py            # restore both from latest
    python3 scripts/restore.py pg         # postgres only
    python3 scripts/restore.py neo4j      # neo4j only
    python3 scripts/restore.py --if-empty # act only when a volume is fresh
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT / "logs" / "backup.log"
STAGING_ROOT = Path("/tmp/ho-restore")

PG_CONTAINER = "firecrawl_agent-memory-db_1"
NEO4J_CONTAINER = "firecrawl_neo4j_1"
NEO4J_VOLUME = "firecrawl_neo4j_data"
NEO4J_IMAGE = "neo4j:community-ubi10"


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


def volume_mountpoint(name: str) -> Path | None:
    r = sh(["podman", "volume", "inspect", name, "--format", "{{.Mountpoint}}"])
    p = r.stdout.strip()
    return Path(p) if p else None


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
                log(f"restore: {key} not set, skipping")
                sys.exit(0)
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


def volume_fresh(mount: Path | None, marker: str) -> bool:
    """True when the volume has no database data.

    Postgres 18 keeps data under <major>/<cluster>/ owned by an
    unreadable container uid, so for pg a volume counts as initialized
    when its root holds any entry at all. Neo4j exposes a databases dir.
    """
    if mount is None or not mount.exists():
        return True
    if marker == "PG_VERSION":
        return not any(mount.iterdir())
    return not (mount / marker).exists()


def latest_snapshot(remote: str, flags: list[str]) -> str | None:
    r = sh(["rclone", "lsf", f"{remote}/daily/", "--dirs-only", *flags])
    snaps = [line.strip().rstrip("/") for line in r.stdout.splitlines()]
    snaps = [s for s in snaps if s and s.replace("-", "").isdigit()]
    return sorted(snaps, reverse=True)[0] if snaps else None


def restore_pg(staging: Path) -> bool:
    dump_gz = staging / "pg_agent_memory.dump.gz"
    if not dump_gz.exists():
        log("restore: pg dump not in snapshot, skipping postgres")
        return False
    log("restoring postgres...")
    try:
        with gzip.open(dump_gz, "rb") as src:
            r = subprocess.run(
                [
                    "podman",
                    "exec",
                    "-i",
                    PG_CONTAINER,
                    "pg_restore",
                    "-Fc",
                    "-U",
                    "postgres",
                    "-d",
                    "agent_memory",
                    "--clean",
                    "--if-exists",
                ],
                stdin=src,
                capture_output=True,
                timeout=1800,
            )
        if r.returncode == 0:
            log("postgres restored ✓")
            return True
        log(f"ERROR: pg_restore failed: {r.stderr[-300:]}")
        return False
    except Exception as exc:
        log(f"ERROR: pg_restore exception: {exc}")
        return False


def restore_neo4j(staging: Path) -> bool:
    dump_gz = staging / "neo4j_neo4j.tar.gz"
    if not dump_gz.exists():
        log("restore: neo4j dump not in snapshot, skipping neo4j")
        return False
    log("restoring neo4j...")
    sh(["podman", "stop", NEO4J_CONTAINER])
    try:
        with tarfile.open(dump_gz, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        r = sh(
            [
                "podman",
                "run",
                "--rm",
                "-v",
                f"{NEO4J_VOLUME}:/data",
                "-v",
                f"{staging}:/in",
                NEO4J_IMAGE,
                "neo4j-admin",
                "database",
                "load",
                "neo4j",
                "--from-path=/in",
                "--overwrite-destination=true",
            ],
            timeout=1800,
        )
        if r.returncode == 0:
            log("neo4j restored ✓")
            return True
        log(f"ERROR: neo4j load failed: {r.stderr[-300:]}")
        return False
    finally:
        sh(["podman", "start", NEO4J_CONTAINER])


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    pg_mount = volume_mountpoint("firecrawl_agent_memory_data")
    neo_mount = volume_mountpoint(NEO4J_VOLUME)
    pg_fresh = volume_fresh(pg_mount, "PG_VERSION")
    neo_fresh = volume_fresh(neo_mount, "databases")

    if mode == "--if-empty":
        if not pg_fresh and not neo_fresh:
            log("restore: volumes have data, nothing to do")
            return
        msg = f"restore: empty volumes detected (pg={pg_fresh} neo4j={neo_fresh})"
        log(f"{msg}, restoring from cloud")
        mode = "all"

    remote, flags = load_r2_env()
    snap = latest_snapshot(remote, flags)
    if snap is None:
        log(f"restore: no snapshots in {remote}/daily/, nothing to restore")
        return

    staging = STAGING_ROOT / snap
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    log(f"restore: pulling latest snapshot {snap} from cloud...")
    r = sh(["rclone", "copy", f"{remote}/daily/{snap}", str(staging), *flags], timeout=1800)
    if r.returncode != 0:
        log(f"ERROR: rclone copy failed: {r.stderr[-300:]}")
        sys.exit(1)

    if mode in ("all", "pg") and pg_fresh:
        restore_pg(staging)
    if mode in ("all", "neo4j") and neo_fresh:
        restore_neo4j(staging)

    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
