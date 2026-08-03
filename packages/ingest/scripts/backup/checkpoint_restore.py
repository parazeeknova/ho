"""Restore a local volume checkpoint (tarball) back into podman.

Given a checkpoint directory (default: latest under ``checkpoints/``), for
each volume tarball it:
  1. finds the target podman volume (create if missing)
  2. stops/removes the running container that mounts it (so files aren't in use)
  3. wipes the volume's _data and extracts the tarball into it
  4. restarts the container

Use with care: this overwrites the live volume.

Run:
    uv run python scripts/backup/checkpoint_restore.py                # latest
    uv run python scripts/backup/checkpoint_restore.py --dir checkpoints/20260802-123456
    uv run python scripts/backup/checkpoint_restore.py \
        --vol firecrawl_agent_memory_data  # latest, one vol
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT / "checkpoints"

# Which container mounts a given volume (so we can restart it safely).
VOLUME_TO_CONTAINER = {
    "firecrawl_agent_memory_data": "firecrawl_agent-memory-db_1",
    "firecrawl_neo4j_data": "firecrawl_neo4j_1",
    "firecrawl_neo4j_logs": "firecrawl_neo4j_1",
}


def _sh(cmd: str, timeout: int = 600) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def _volume_data_path(vol: str) -> Path:
    return Path.home() / ".local/share/containers/storage/volumes" / vol / "_data"


def _ensure_volume(vol: str) -> bool:
    rc, out = _sh("podman volume ls --format '{{.Name}}'")
    if vol in out.split():
        return True
    rc, err = _sh(f"podman volume create {vol}")
    return rc == 0


def _find_latest() -> Path:
    dirs = sorted([d for d in CHECKPOINT_DIR.iterdir() if d.is_dir()], reverse=True)
    if not dirs:
        print(f"No checkpoints found under {CHECKPOINT_DIR}")
        sys.exit(1)
    return dirs[0]


def _restore_one(vol: str, tar_path: Path) -> bool:
    print(f"  restoring {vol} from {tar_path.name}")
    data_path = _volume_data_path(vol)
    if not _ensure_volume(vol):
        print(f"    could not ensure volume {vol}")
        return False

    container = VOLUME_TO_CONTAINER.get(vol)
    if container:
        # Stop the container so the volume isn't in use, but never remove it.
        _sh(f"podman stop {container} 2>/dev/null")

    if data_path.exists():
        shutil.rmtree(data_path, ignore_errors=True)
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            # If the archive has a single top-level "volume_data", extract into _data
            members = tf.getnames()
            root = members[0].split("/")[0] if members else "volume_data"
            if root == "volume_data" and len({m.split("/")[0] for m in members}) == 1:
                tf.extractall(data_path)
            else:
                tf.extractall(data_path)
    except Exception as e:
        print(f"    extract failed: {e}")
        return False

    # podman volume export produces a root-level layout; normalize the
    # "volume_data" wrapper if the fallback tar used it.
    wrapped = data_path / "volume_data"
    if wrapped.is_dir() and wrapped != data_path:
        for item in wrapped.iterdir():
            shutil.move(str(item), str(data_path))
        wrapped.rmdir()

    if container:
        _sh(f"podman start {container} 2>/dev/null")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None, help="checkpoint dir")
    ap.add_argument("--vol", default=None, help="restore only this volume")
    args = ap.parse_args()

    ckpt = args.dir or _find_latest()
    manifest_path = ckpt / "_manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text()).get("volumes", [])

    if not manifest:
        tars = sorted(ckpt.glob("*.tar.gz"))
        manifest = [{"volume": t.name.replace(".tar.gz", ""), "file": t.name} for t in tars]

    if args.vol:
        manifest = [m for m in manifest if m["volume"] == args.vol]

    print(f"Restoring {len(manifest)} volumes from {ckpt}")
    ok = 0
    for m in manifest:
        tar_path = ckpt / m["file"]
        if tar_path.exists() and _restore_one(m["volume"], tar_path):
            ok += 1
        else:
            print(f"  FAIL {m['volume']}")
    print(f"Done: restored {ok}/{len(manifest)} volumes")


if __name__ == "__main__":
    main()
