"""Local checkpoint backup: snapshot named container volumes to disk.

Creates timestamped tarballs of the firecrawl stack's named volumes under
``checkpoints/<timestamp>/``. Postgres + Neo4j are the critical ones; the
script snapshots any volume passed (default = the ho stack's named volumes).

Uses ``podman volume export`` where available (tar stream of the volume), with
a raw-copy fallback via ``cp -a`` of the volume's ``_data`` directory.

Run:
    uv run python scripts/backup/checkpoint_backup.py            # all ho volumes
    uv run python scripts/backup/checkpoint_backup.py --vol firecrawl_agent_memory_data
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT / "checkpoints"

# Default ho-stack named volumes.
HO_VOLUMES = [
    "firecrawl_agent_memory_data",
    "firecrawl_neo4j_data",
    "firecrawl_neo4j_logs",
]


def _sh(cmd: str, timeout: int = 600) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def _volume_sources() -> dict[str, str]:
    """Map volume name -> its _data directory path on disk."""
    out = _sh("podman volume ls --format '{{.Name}}'")[1]
    base = Path.home() / ".local/share/containers/storage/volumes"
    mapping: dict[str, str] = {}
    for vol in out.splitlines():
        vol = vol.strip()
        if not vol:
            continue
        p = base / vol / "_data"
        if p.exists():
            mapping[vol] = str(p)
    return mapping


def _volume_export(vol: str, dest_tar: Path) -> bool:
    """Export + gzip the volume via `podman volume export`. True on success."""
    rc, _ = _sh(
        f"podman volume export {vol} | gzip -c > {dest_tar}",
        timeout=1800,
    )
    return rc == 0 and dest_tar.exists() and dest_tar.stat().st_size > 0


def _tar_dir(src: str, dest_tar: Path) -> bool:
    """Fallback: tar the volume's _data directory."""
    try:
        with tarfile.open(dest_tar, "w:gz") as tf:
            tf.add(src, arcname="volume_data")
        return dest_tar.exists() and dest_tar.stat().st_size > 0
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", action="append", default=[], help="volume name (repeatable)")
    args = ap.parse_args()

    volumes = args.vol or HO_VOLUMES
    sources = _volume_sources()
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = CHECKPOINT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    print(f"Checkpointing {len(volumes)} volumes -> {out_dir}")

    for vol in volumes:
        if vol not in sources:
            print(f"  SKIP {vol}: not a local podman volume")
            continue
        dest_tar = out_dir / f"{vol}.tar.gz"
        ok = _volume_export(vol, dest_tar)
        method = "export" if ok else None
        if not ok:
            print(f"  export failed for {vol}, falling back to tar of _data")
            ok = _tar_dir(sources[vol], dest_tar)
            method = "tar" if ok else None
        size = dest_tar.stat().st_size if dest_tar.exists() else 0
        if ok:
            print(f"  OK {vol}: {size / 1e6:.1f} MB ({method})")
            manifest.append({"volume": vol, "file": dest_tar.name, "size": size, "method": method})
        else:
            print(f"  FAIL {vol}")

    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "volumes": manifest,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"Done: {len(manifest)} volumes -> {out_dir}")


if __name__ == "__main__":
    main()
