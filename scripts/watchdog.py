#!/usr/bin/env python3
"""Self-healing watchdog for the ho pipeline.

Runs every CHECK_INTERVAL seconds and repairs any component that died:
  * llama-server embedding service (:8900)          -> relaunch scripts/serve.py
  * firecrawl containers (8)                        -> podman start / compose up
  * pipeline supervisor (scripts/run.py)            -> full relaunch if dead
  * azure ingest loop (scripts/azure/ingest.py)     -> relaunch with stored creds

Each component has a per-item cooldown so crash-loops do not thrash.
Logs to logs/watchdog.log. Single-instance via lockfile.

Usage:
    nohup uv run python3 scripts/watchdog.py > /dev/null 2>&1 &
    or: systemctl --user enable --now ho-watchdog.service
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT / "logs" / "watchdog.log"
LOCK_PATH = PROJECT / "logs" / "watchdog.lock"
CHECK_INTERVAL = 60
COOLDOWN = 180  # per-component minimum seconds between heal attempts
PIPELINE_COOLDOWN = 300  # run.py can take >3min to come up; avoid double relaunch

EMBED_HEALTH = "http://127.0.0.1:8900/health"
CONTAINERS = [
    "firecrawl_redis_1",
    "firecrawl_nuq-postgres_1",
    "firecrawl_searxng_1",
    "firecrawl_neo4j_1",
    "firecrawl_agent-memory-db_1",
    "firecrawl_rabbitmq_1",
    "firecrawl_playwright-service_1",
    "firecrawl_api_1",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    line = f'{{"timestamp": "{ts}", "level": "INFO", "message": "{msg}", "logger": "watchdog"}}\n'
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(line)
    except Exception:
        pass
    print(f"[watchdog] {msg}", flush=True)


def sh(cmd: str, timeout: int = 20) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def run_detached(cmd: str) -> None:
    """Start a long-lived process fully detached from this shell."""
    subprocess.Popen(
        cmd,
        shell=True,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def count_of(pidfile_pattern: str) -> int:
    code, out = sh(f"pgrep -f '{pidfile_pattern}' | wc -l")
    m = re.search(r"\d+", out or "0")
    return int(m.group()) if m else 0


class Watchdog:
    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def due(self, key: str, cooldown: float | None = None) -> bool:
        now = time.monotonic()
        cd = cooldown if cooldown is not None else COOLDOWN
        if now - self._last.get(key, -cd) < cd:
            return False
        self._last[key] = now
        return True

    def heal_embed(self) -> None:
        code, _ = sh(f"curl -s -o /dev/null -w '%{{http_code}}' {EMBED_HEALTH}")
        if code == 0 and "200" in _:
            return
        log("embedding server down, relaunching serve.py")
        sh("pkill -f 'scripts/serve.py' 2>/dev/null; pkill -x llama-server 2>/dev/null")
        time.sleep(2)
        run_detached(
            f"cd {PROJECT} && nohup uv run python3 scripts/serve.py "
            f"> {PROJECT / 'logs' / 'serve.out'} 2>&1 &"
        )

    def heal_containers(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        # A backup holds neo4j stopped on purpose; do not fight it.
        if (PROJECT / "logs" / "backup.lock").exists():
            return statuses
        for name in CONTAINERS:
            code, out = sh(f"podman ps -a --filter name={name} --format '{{{{.Status}}}}'")
            status = (out or "missing").split()[0] if out else "missing"
            statuses[name] = status
            if status in ("missing", "Exited"):
                if not self.due(f"container:{name}"):
                    continue
                if status == "Exited":
                    rc, err = sh(f"podman start {name}")
                    log(f"restarted container {name}: rc={rc} {err[:80]}")
                else:
                    log(f"container {name} missing, running compose up")
                    svc = name.replace("firecrawl_", "")
                    sh(f"docker compose -f {PROJECT / 'docker-compose.yaml'} up -d {svc}")
        return statuses

    def heal_pipeline(self, container_statuses: dict[str, str]) -> None:
        alive = count_of(r"scripts/run\.py") + count_of(r"radar[.]engine[.]orchestrator")
        if alive > 0:
            return
        if not self.due("pipeline", PIPELINE_COOLDOWN):
            return
        critical_down = [n for n, s in container_statuses.items() if s in ("missing", "Exited")]
        down = critical_down or "none"
        log(f"pipeline supervisor dead, relaunching run.py (containers down: {down})")
        run_detached(
            f"cd {PROJECT} && nohup uv run python3 scripts/run.py "
            f"> {PROJECT / 'logs' / 'run.out'} 2>&1 &"
        )

    def heal_ingest(self) -> None:
        alive = count_of(r"scripts/azure/ingest.py")
        if alive > 0:
            return
        if not self.due("ingest"):
            return
        log("azure ingest dead, relaunching")
        creds = PROJECT / "scripts" / ".watchdog.env"
        source = f"set -a; . {creds}; set +a;" if creds.exists() else ""
        run_detached(
            f"cd {PROJECT} && {source} PYTHONPATH={PROJECT} nohup uv run --with azure-storage-blob "
            f"python3 scripts/azure/ingest.py >> {PROJECT / 'logs' / 'ingest.out'} 2>&1 &"
        )

    def cycle(self) -> None:
        try:
            self.heal_embed()
            statuses = self.heal_containers()
            self.heal_pipeline(statuses)
            self.heal_ingest()
        except Exception as exc:
            log(f"cycle error: {exc}")


def acquire_lock() -> bool:
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)
            return False
        except ValueError, ProcessLookupError:
            pass
    LOCK_PATH.write_text(str(os.getpid()))
    return True


def main() -> None:
    if not acquire_lock():
        print("watchdog already running", flush=True)
        sys.exit(0)
    log(f"watchdog started (interval={CHECK_INTERVAL}s)")
    wd = Watchdog()
    time.sleep(15)  # let a mid-start embedding server finish loading before first check
    wd.cycle()
    while True:
        time.sleep(CHECK_INTERVAL)
        wd.cycle()


if __name__ == "__main__":
    main()
