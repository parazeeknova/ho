#!/usr/bin/env python3
"""ho — unified CLI for the whole project.

One entry point for every operation; subcommands dispatch to the existing
scripts so you never guess a `bun run` name again.

Usage:
    ho run [--dry-run] [--radar-workers N] [--max-minutes M]
    ho initm [--grill] [--no-resume] [--resume-url URL]
    ho intel [--top N] [--discord]
    ho backup [--vol NAME]        | ho backup list | ho backup restore [FILE]
    ho health
    ho check                      # format + lint + typecheck (py + node)
    ho test                       # pytest + node tests
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INGEST = REPO / "packages" / "ingest"
AUTOFILL = REPO / "packages" / "autofill"
PYTHONPATH = ":".join([str(REPO), str(INGEST), str(AUTOFILL), os.environ.get("PYTHONPATH", "")])


def _run(script: Path, args: list[str]) -> int:
    env = dict(os.environ, PYTHONPATH=PYTHONPATH)
    print(f"[ho] {script.name} {' '.join(args)}", flush=True)
    return subprocess.call([sys.executable, str(script), *args], env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="start the full pipeline (containers, radar, autofill)")
    sub.add_parser("status", help="print all pipeline counters and exit")
    sub.add_parser("initm", help="set up resume + persona memory")
    sub.add_parser("intel", help="market / skill-arbitrage report")
    sub.add_parser("backup", help="checkpoint backup / restore")
    sub.add_parser("health", help="service health checks")
    sub.add_parser("check", help="format + lint + typecheck across py + node")
    sub.add_parser("test", help="run all tests (pytest + node)")

    args, extra = ap.parse_known_args()

    if args.cmd == "run":
        return _run(INGEST / "scripts" / "run_all.py", extra)
    if args.cmd == "status":
        return _run(INGEST / "scripts" / "run_all.py", ["--status"] + extra)
    if args.cmd == "initm":
        return _run(AUTOFILL / "scripts" / "init_memory.py", extra)
    if args.cmd == "intel":
        return _run(INGEST / "scripts" / "intel" / "radar_intel.py", extra)
    if args.cmd == "backup":
        if extra and extra[0] in ("list", "restore"):
            return _run(INGEST / "scripts" / "backup" / "checkpoint_backup.py", extra)
        return _run(INGEST / "scripts" / "backup" / "checkpoint_backup.py", extra)
    if args.cmd == "health":
        return _run(INGEST / "scripts" / "tools" / "health.py", extra)
    if args.cmd == "check":
        for c in [
            "uv run ruff format .",
            "uv run ruff check . --fix",
            "uv run mypy",
            "bun run --filter 'autofill-node*' lint",
            "bun run --filter 'autofill-node*' format:check",
            "bun run --filter 'autofill-node*' typecheck",
        ]:
            if subprocess.call(c, shell=True) != 0:
                return 1
        return 0
    if args.cmd == "test":
        for c in [
            "uv run python -m pytest . -v --ignore=refs",
            "bun run --filter 'autofill-node*' test",
        ]:
            if subprocess.call(c, shell=True) != 0:
                return 1
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
