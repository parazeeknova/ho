"""Robust full download of ALL blobs from the Azure relic container.

Streams each blob to disk (no full in-memory copy), skips already-downloaded
blobs, and logs progress. Run detached; re-run to resume.

Usage:
    uv run --with azure-storage-blob python3 scripts/azure/full_pull.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from azure.storage.blob import BlobServiceClient

OUT_ROOT = Path(__file__).resolve().parent.parent.parent / "azure_dump"


def main() -> None:
    account = os.environ["AZURE_STORAGE_ACCOUNT"]
    key = os.environ["AZURE_STORAGE_KEY"]
    container = os.environ.get("AZURE_CONTAINER", "radar-index")
    conn = (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={account};AccountKey={key};EndpointSuffix=core.windows.net"
    )
    svc = BlobServiceClient.from_connection_string(conn)
    cc = svc.get_container_client(container)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    blobs = list(cc.list_blobs())
    total = len(blobs)
    print(f"Found {total} blobs to download", flush=True)

    manifest_path = OUT_ROOT / "_manifest.json"
    done: set[str] = set()
    if manifest_path.exists():
        try:
            for m in json.loads(manifest_path.read_text()):
                if (OUT_ROOT / m["name"]).exists() and (OUT_ROOT / m["name"]).stat().st_size == m[
                    "size"
                ]:
                    done.add(m["name"])
        except Exception:
            pass

    pending = [b for b in blobs if b.name not in done]
    print(f"{len(done)} already done; downloading {len(pending)}", flush=True)

    manifest: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = []
    seen_names = {m["name"] for m in manifest}

    for i, b in enumerate(pending, 1):
        try:
            dest = OUT_ROOT / b.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            bc = cc.get_blob_client(b.name)
            stream = bc.download_blob().chunks()
            size = 0
            with open(dest, "wb") as f:
                for chunk in stream:
                    f.write(chunk)
                    size += len(chunk)
            if b.name not in seen_names:
                manifest.append({"name": b.name, "size": size, "ts": b.last_modified.isoformat()})
                seen_names.add(b.name)
            with open(manifest_path, "w") as mf:
                json.dump(manifest, mf, indent=2)
            if i % 5 == 0 or size > 50_000_000:
                print(
                    f"  [{time.strftime('%H:%M:%S')}] "
                    f"{i}/{len(pending)} {b.name} ({size / 1e6:.1f} MB)",
                    flush=True,
                )
        except Exception as exc:
            print(f"  ERROR {b.name}: {exc}", flush=True)

    total_bytes = sum(m["size"] for m in manifest)
    print(
        f"DONE: {len(manifest)}/{total} blobs, {total_bytes / 1e6:.1f} MB -> {OUT_ROOT}", flush=True
    )


if __name__ == "__main__":
    sys.exit(main())
