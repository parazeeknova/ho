#!/usr/bin/env python3
"""Full pipeline launcher: start all services, wait for health, run orchestrator.

Usage:
    make run          → full pipeline
    python scripts/run.py --no-pipeline  → infrastructure only (same as make dev)
"""

import argparse
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DOCKER_COMPOSE = f"docker compose -f {PROJECT}/docker-compose.yaml"


def run(cmd: str, silent: bool = True) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=silent, text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr)[:500]
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def check_http(url: str) -> bool:
    with contextlib.suppress(Exception):
        import urllib.request

        urllib.request.urlopen(url, timeout=3)
        return True
    return False


def check_port(host: str, port: int) -> bool:
    with contextlib.suppress(Exception), socket.create_connection((host, port), timeout=2):
        return True
    return False


def container_running(name: str) -> bool:
    code, _ = run(
        f"podman ps --filter name='{name}' --filter status=running --format '{{{{.Names}}}}'"
    )
    return code == 0


def print_status(who: str, status: str, detail: str = "") -> None:
    icon = {"OK": "✓", "WAIT": "⏳", "FAIL": "✗"}.get(status, " ")
    line = f"  {icon} {who:<25}"
    if detail:
        line += f"  {detail}"
    print(line, flush=True)


def start_infrastructure() -> None:
    """Start all containers and wait for health."""
    print("\n── Starting infrastructure ──\n")

    # Cleanup stale
    run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
    run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
    time.sleep(1)

    # Launch core infra
    run(
        f"{DOCKER_COMPOSE} up -d redis playwright-service "
        "nuq-postgres searxng neo4j agent-memory-db",
        silent=False,
    )

    # Launch rabbitmq
    run(
        "podman run -d --name firecrawl_rabbitmq_1 "
        "--network firecrawl_default --network-alias rabbitmq "
        "--restart unless-stopped "
        "--entrypoint /bin/bash rabbitmq:3-management "
        '-c "rm -f /var/lib/rabbitmq/.erlang.cookie; '
        'exec docker-entrypoint.sh rabbitmq-server"',
        silent=True,
    )

    # Wait for rabbitmq
    print_status("rabbitmq", "WAIT")
    for _ in range(15):
        time.sleep(2)
        if container_running("firecrawl_rabbitmq"):
            code, _ = run(
                "podman exec firecrawl_rabbitmq_1 rabbitmqctl await_startup 2>/dev/null",
                silent=True,
            )
            if code == 0:
                print_status("rabbitmq", "OK", "running")
                break
    else:
        print_status("rabbitmq", "FAIL", "timeout")

    # Launch API
    print_status("firecrawl api", "WAIT")
    run(f"{DOCKER_COMPOSE} up -d api", silent=True)
    for _ in range(15):
        time.sleep(2)
        if check_http("http://localhost:3002/health") or check_port("localhost", 3002):
            print_status("firecrawl api", "OK", ":3002")
            break
    else:
        print_status("firecrawl api", "FAIL", "timeout")

    # Check remaining services
    checks = [
        ("redis", lambda: container_running("firecrawl_redis"), ":6379"),
        ("playwright", lambda: container_running("firecrawl_playwright"), ":3000"),
        ("nuq-postgres", lambda: container_running("firecrawl_nuq-postgres"), ":5432"),
        ("searxng", lambda: check_http("http://localhost:8080"), ":8080"),
        ("neo4j", lambda: check_port("localhost", 7687), ":7687"),
        ("agent-memory-db", lambda: check_port("localhost", 5433), ":5433"),
    ]
    for name, check_fn, port in checks:
        for _ in range(10):
            if check_fn():
                print_status(name, "OK", port)
                break
            time.sleep(1)
        else:
            print_status(name, "FAIL", port)

    print("\n── All services started ──\n")


def run_pipeline() -> int:
    """Launch the orchestrator pipeline. Returns exit code."""
    env = os.environ.copy()
    env.setdefault("OVERNIGHT_LOOP", "true")
    log_dir = PROJECT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "run.log"

    print("── Starting pipeline ──\n")
    sys.stdout.flush()

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.radar.orchestrator"],
        cwd=str(PROJECT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _on_signal(sig, _frame):
        proc.send_signal(sig)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    with open(log_path, "a") as log_file:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)

    proc.wait()
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="ho pipeline launcher")
    parser.add_argument(
        "--no-pipeline", action="store_true", help="Start infra only, don't run pipeline"
    )
    parser.add_argument("--no-health", action="store_true", help="Skip health checks before infra")
    args = parser.parse_args()

    if not args.no_health:
        print("── Running health checks ──")
        r = subprocess.run([sys.executable, f"{PROJECT}/scripts/health.py"], cwd=str(PROJECT))
        if r.returncode != 0:
            print("Health checks failed. Continuing anyway...")

    start_infrastructure()

    if args.no_pipeline:
        print("\nInfrastructure running. Pipeline not started (--no-pipeline).")
        print("Press Ctrl+C to stop all services.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
            run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
        return

    code = run_pipeline()
    sys.exit(code)


if __name__ == "__main__":
    main()
