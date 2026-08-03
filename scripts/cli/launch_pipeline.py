"""Robust detached launcher for the ho pipeline.

Survives shell exit: starts the pipeline (scripts/run.py) with the
high LLM-budget env, writes the PID to logs/pipeline.pid, and supports
stop/status.

Usage:
    uv run python scripts/cli/launch_pipeline.py            # start (default)
    uv run python scripts/cli/launch_pipeline.py --stop     # stop
    uv run python scripts/cli/launch_pipeline.py --status   # is it alive?
    uv run python scripts/cli/launch_pipeline.py --restart
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT / "logs" / "run_pipeline.log"
PID_FILE = PROJECT / "logs" / "pipeline.pid"

_ENV_OVERRIDES = {
    "LLM_QUEUE_RPM": "240",
    "LLM_QUEUE_MAX_IN_FLIGHT": "30",
    "LLM_QUEUE_TPM": "400000",
    "LLM_BUDGET_RADAR_RPM": "240",
    "LLM_BUDGET_RADAR_TPM": "400000",
}


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _status() -> int:
    pid = _read_pid()
    if _alive(pid):
        print(f"pipeline running (pid {pid})")
        return 0
    print("pipeline not running")
    return 1


def _stop() -> int:
    import contextlib

    pid = _read_pid()
    if _alive(pid):
        print(f"stopping pipeline (pid {pid})...")
        os.kill(pid, signal.SIGTERM)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)
        print("stopped")
        return 0
    print("pipeline not running")
    return 0


def _start() -> int:
    if _alive(_read_pid()):
        print(f"pipeline already running (pid {_read_pid()})")
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT)
    env.update(_ENV_OVERRIDES)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("ab") as log:
        proc = subprocess.Popen(
            ["uv", "run", "python", "scripts/run.py"],
            cwd=PROJECT,
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    PID_FILE.write_text(f"{proc.pid}\n")
    print(f"pipeline started (pid {proc.pid}) -> logs/run_pipeline.log")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Detached ho pipeline launcher")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--start", action="store_true", help="start the pipeline (default)")
    group.add_argument("--stop", action="store_true", help="stop a running pipeline")
    group.add_argument("--status", action="store_true", help="is the pipeline alive?")
    group.add_argument("--restart", action="store_true", help="stop, then start")
    args = ap.parse_args()

    if args.stop or args.restart:
        _stop()
    if args.start or args.restart or not (args.stop or args.status):
        return _start()
    if args.status:
        return _status()
    return _start()


if __name__ == "__main__":
    sys.exit(main())
