"""Purge stale/fake entries from the Qdrant jobs ledger.

Usage:
    uv run python scripts/cleanup_qdrant.py
"""
import hashlib
import os

from qdrant_client import QdrantClient

FAKE_KEYS = [
    "techco:backendengineer",
]


def main() -> None:
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=3)
        client.get_collections()
        print("Connected to Qdrant at localhost:6333")
    except Exception as e:
        print(f"localhost:6333 failed ({e}), falling back to local disk")
        storage_dir = os.path.join(os.getcwd(), "storage", "qdrant")
        client = QdrantClient(path=storage_dir)

    for key in FAKE_KEYS:
        point_id = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:12], 16)
        try:
            client.delete("jobs_ledger", points_selector=[point_id])
            print(f"  Deleted point {point_id} ({key})")
        except Exception as e:
            print(f"  Delete {key} failed: {e}")

    records, _ = client.scroll(
        collection_name="jobs_ledger", limit=200, with_payload=True
    )
    for r in records:
        p = r.payload
        if p:
            company = p.get("company", "?")
            role = p.get("role", "?")
            print(f"  {r.id}: {company} / {role} ({p.get('match_percent', 0)}%)")
    print(f"Total: {len(records)} records")


if __name__ == "__main__":
    main()
