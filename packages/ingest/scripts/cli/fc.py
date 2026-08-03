"""Firecrawl stack + Neo4j container manager.

Consolidates the old Makefile bash: fc-up / fc-down / fc-logs / fc status /
dev-down / clean-volumes / tor-up / graph-up / graph-stop / graph-reset /
graph-shell.

Usage:
    uv run python scripts/cli/fc.py                            # up (default)
    uv run python scripts/cli/fc.py down
    uv run python scripts/cli/fc.py logs
    uv run python scripts/cli/fc.py status
    uv run python scripts/cli/fc.py clean
    uv run python scripts/cli/fc.py dev-down
    uv run python scripts/cli/fc.py tor-up
    uv run python scripts/cli/fc.py graph-up|graph-stop|graph-reset|graph-shell
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(PROJECT / "docker-compose.yaml")]

RABBITMQ = "firecrawl_rabbitmq_1"
CORE = ["redis", "playwright-service", "nuq-postgres", "searxng", "neo4j"]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT, check=check)


def _podman_rm(name: str) -> None:
    _run(["podman", "rm", "-f", name], check=False)


def up() -> None:
    _run([*COMPOSE, "up", "-d", *CORE])
    # RabbitMQ needs a fresh .erlang.cookie to start cleanly.
    _podman_rm(RABBITMQ)
    _run(
        [
            "podman",
            "run",
            "-d",
            "--name",
            RABBITMQ,
            "--network",
            "firecrawl_default",
            "--network-alias",
            "rabbitmq",
            "--restart",
            "no",
            "--entrypoint",
            "/bin/bash",
            "rabbitmq:3-management",
            "-c",
            "rm -f /var/lib/rabbitmq/.erlang.cookie; exec docker-entrypoint.sh rabbitmq-server",
        ]
    )
    print("Waiting for rabbitmq...")
    for _ in range(10):
        ok = _run(["podman", "exec", RABBITMQ, "rabbitmqctl", "await_startup"], check=False)
        if ok.returncode == 0:
            break
        time.sleep(2)
    _run([*COMPOSE, "up", "-d", "api"])


def down() -> None:
    _run([*COMPOSE, "down"])
    _podman_rm(RABBITMQ)


def logs() -> None:
    _run([*COMPOSE, "logs", "-f"])


def status() -> None:
    _run([*COMPOSE, "ps"])


def clean() -> None:
    _run([*COMPOSE, "down", "-v"])
    _podman_rm(RABBITMQ)
    storage = PROJECT / "storage"
    if storage.exists():
        import shutil

        shutil.rmtree(storage)
        print("removed storage/")
    print("All container volumes and local storage cleared.")


def dev_down() -> None:
    down()
    subprocess.run(["killall", "llama-server"], check=False)


def tor_up() -> None:
    _run([*COMPOSE, "up", "-d", "torproxy"])


def graph_up() -> None:
    _run([*COMPOSE, "up", "-d", "neo4j"])


def graph_stop() -> None:
    _run([*COMPOSE, "stop", "neo4j"])


def graph_reset() -> None:
    _run([*COMPOSE, "down", "-v", "neo4j"])
    _run([*COMPOSE, "up", "-d", "neo4j"])


def graph_shell() -> None:
    _run([*COMPOSE, "exec", "neo4j", "cypher-shell", "-u", "neo4j", "-p", "password"])


ACTIONS = {
    "up": up,
    "down": down,
    "logs": logs,
    "status": status,
    "clean": clean,
    "dev-down": dev_down,
    "tor-up": tor_up,
    "graph-up": graph_up,
    "graph-stop": graph_stop,
    "graph-reset": graph_reset,
    "graph-shell": graph_shell,
}


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "up"
    fn = ACTIONS.get(action)
    if fn is None:
        print(f"unknown action '{action}'; choices: {', '.join(ACTIONS)}")
        return 1
    fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
