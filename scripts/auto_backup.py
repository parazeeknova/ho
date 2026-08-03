"""Per-sweep auto-backup: snapshot volumes after each sweep, keep latest 10.

Wraps checkpoint_backup.py and prunes old snapshots so at most KEEP remain.
Called by the orchestrator after each sweep completes (or manually).

Run:
    uv run python scripts/auto_backup.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT / "checkpoints"
KEEP = int(__import__("os").environ.get("AUTO_BACKUP_KEEP", "10"))


def _sh(cmd: str, timeout: int = 1800) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def _prune(keep: int = KEEP) -> list[Path]:
    """Delete oldest checkpoint dirs beyond `keep`. Returns removed paths."""
    dirs = sorted([d for d in CHECKPOINT_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    removed: list[Path] = []
    while len(dirs) > keep:
        victim = dirs.pop(0)
        _sh(f"rm -rf {victim}")
        removed.append(victim)
    return removed


def main() -> int:
    # Snapshot using the same volume-export path as checkpoint_backup.
    rc, out = _sh(
        f"cd {PROJECT} && PYTHONPATH={PROJECT} "
        f"uv run python scripts/checkpoint_backup.py",
        timeout=1800,
    )
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    if rc != 0:
        print(f"[{ts}] auto-backup FAILED: {out[-300:]}", flush=True)
        return 1
    removed = _prune()
    summary = {"ts": ts, "kept": len(list(CHECKPOINT_DIR.iterdir())), "pruned": [p.name for p in removed]}
    print(f"[{ts}] auto-backup done; pruned {len(removed)} -> kept {summary['kept']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
