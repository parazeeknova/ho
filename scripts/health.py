#!/usr/bin/env python3
"""Health checks for all services in the ho pipeline."""

import contextlib
import subprocess
import sys
import urllib.request
from collections.abc import Callable

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
fails = 0


def check(name: str, fn: Callable[[], bool]) -> None:
    global fails
    ok = False
    with contextlib.suppress(Exception):
        ok = fn()
    if ok:
        print(f"  {GREEN}OK{RESET}   {name}")
    else:
        print(f"  {RED}DOWN{RESET} {name}")
        fails += 1


def http_ok(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def container_running(pattern: str) -> bool:
    try:
        r = subprocess.run(
            [
                "podman", "ps", "--filter", f"name={pattern}",
                "--filter", "status=running", "--format", "{{.Names}}",
            ],
            capture_output=True, text=True, timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def pg_ready(host: str, port: int, db: str) -> bool:
    try:
        r = subprocess.run(
            ["pg_isready", "-h", host, "-p", str(port), "-U", "postgres", "-d", db, "-q"],
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


print("LLM")
check("llama-server :8899",      lambda: http_ok("http://localhost:8899/health"))
check("llama-server :8900",      lambda: http_ok("http://localhost:8900/health"))

print()
print("Firecrawl")
check("api              :3002",  lambda: http_ok("http://localhost:3002"))
check("redis",                   lambda: container_running("firecrawl_redis"))
check("rabbitmq",               lambda: container_running("firecrawl_rabbitmq"))
check("playwright",             lambda: container_running("firecrawl_playwright"))
check("nuq-postgres",           lambda: container_running("firecrawl_nuq-postgres"))

print()
print("Agent Memory")
check("agent-memory-db",        lambda: container_running("firecrawl_agent-memory-db"))
check("pgvector :5433",         lambda: pg_ready("localhost", 5433, "agent_memory"))

print()
print("Metasearch")
check("searxng          :8080", lambda: http_ok("http://localhost:8080"))

sys.exit(fails)
