"""Shared helpers for Azure-relic intelligence workers.

Reusable across any VPS/VM that runs ho's relic workers. Provides:
  - connection-string / container setup from env
  - write-once unique-named blob upload (``{prefix}/{hour}_{seq}.jsonl``)
  - checkpointed read of the newest companies/obs blobs so each worker
    consumes the freshest company list without re-fetching state.

Config (env): AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER
(default ``radar-index``). Run with ``--with azure-storage-blob``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from azure.storage.blob import BlobServiceClient

DEFAULT_CONTAINER = "radar-index"


def container_client() -> Any:
    """Return an Azure container client built from env credentials."""
    account = os.environ.get("AZURE_STORAGE_ACCOUNT")
    key = os.environ.get("AZURE_STORAGE_KEY")
    if not account or not key:
        raise RuntimeError("AZURE_STORAGE_ACCOUNT/AZURE_STORAGE_KEY not set")
    conn = (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={account};AccountKey={key};EndpointSuffix=core.windows.net"
    )
    svc = BlobServiceClient.from_connection_string(conn)
    return svc.get_container_client(os.environ.get("AZURE_CONTAINER", DEFAULT_CONTAINER))


def upload_records(prefix: str, records: list[dict[str, Any]], cc: Any = None) -> str | None:
    """Upload a list of JSON records as a write-once blob. Returns blob name."""
    if not records:
        return None
    cc = cc or container_client()
    body = "\n".join(json.dumps(r) for r in records).encode()
    hour = int(time.time() // 3600)
    seq = int(time.time() * 1000) % 100000
    blob_name = f"{prefix}/{hour}_{seq}.jsonl"
    cc.get_blob_client(blob_name).upload_blob(body, overwrite=False)
    return blob_name


def newest_blob(prefix: str, cc: Any = None) -> dict[str, Any] | None:
    """Return the newest blob under a prefix as parsed records."""
    cc = cc or container_client()
    blobs = [b for b in cc.list_blobs() if b.name.startswith(prefix) and b.name.endswith(".jsonl")]
    if not blobs:
        return None
    newest = max(blobs, key=lambda b: b.last_modified)
    data = cc.get_blob_client(newest.name).download_blob().readall().decode()
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    return {"name": newest.name, "records": records}


def read_blob(name: str, cc: Any = None) -> list[dict[str, Any]]:
    cc = cc or container_client()
    data = cc.get_blob_client(name).download_blob().readall().decode()
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def list_blobs(prefix: str, cc: Any = None) -> list[str]:
    cc = cc or container_client()
    return [b.name for b in cc.list_blobs(name_starts_with=prefix)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)
