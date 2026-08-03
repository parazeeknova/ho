#!/usr/bin/env python3
"""Snapshot the autofill queue into a durable batch tracking ledger.

Writes one JSON line per (job_id, status, error, updated_at) into
``logs/batch-tracking.jsonl`` so a job's retry history (pending -> failed ->
pending -> skipped, etc.) is preserved rather than overwritten. Idempotent:
re-running only appends rows whose state actually changed, so it can be called
after every job and once more at the end of the batch for reconciliation.

Also prints a live summary of submitted / skipped / failed / deferred / pending
runs, plus the failure reasons — the tracking view used while walking the
100-job batch.

Usage:
    uv run python scripts/batch_track.py            # snapshot + summary
    uv run python scripts/batch_track.py --quiet    # snapshot only, no table
    uv run python scripts/batch_track.py --last 20  # print last N ledger lines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from autofill.db import AutofillDB
from dotenv import load_dotenv

load_dotenv()

PROJECT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT / "logs" / "batch-tracking.jsonl"


def _dedupe_key(entry: dict) -> tuple[str, str, str, str]:
    return (
        str(entry.get("job_id", "")),
        str(entry.get("status", "")),
        str(entry.get("error", "")),
        str(entry.get("updated_at", "")),
    )


def _load_existing() -> list[dict]:
    if not LEDGER.exists():
        return []
    out: list[dict] = []
    for raw in LEDGER.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


def snapshot() -> list[dict]:
    """Append changed queue rows to the ledger. Returns all ledger entries."""
    existing = _load_existing()
    seen = {_dedupe_key(e) for e in existing}

    async def _snapshot() -> tuple[list[dict], str]:
        import datetime as _dt

        db = await AutofillDB.create()
        try:
            rows = await db.get_all_jobs()
        finally:
            await db.close()
        now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        return rows, now

    rows, now = asyncio.run(_snapshot())

    added = 0
    for row in rows:
        entry = {
            "snapshot_at": now,
            "job_id": row.get("job_id"),
            "apply_link": row.get("apply_link"),
            "company": row.get("company"),
            "role": row.get("role"),
            "apply_mode": row.get("apply_mode"),
            "status": row.get("status"),
            "retries": row.get("retries"),
            "error": row.get("error"),
            "updated_at": row.get("updated_at"),
        }
        if _dedupe_key(entry) in seen:
            continue
        existing.append(entry)
        seen.add(_dedupe_key(entry))
        added += 1

    if added:
        LEDGER.parent.mkdir(exist_ok=True)
        LEDGER.write_text("\n".join(json.dumps(e, default=str) for e in existing) + "\n")
    return existing


def _summary(entries: list[dict]) -> None:
    """Print the most recent ledger state per job plus a status tally."""
    latest: dict[str, dict] = {}
    for e in entries:
        latest[str(e.get("job_id", ""))] = e

    if not latest:
        print("No tracked jobs yet.")
        return

    tally: dict[str, int] = {}
    failed: list[tuple[str, str, str]] = []
    for e in latest.values():
        status = str(e.get("status", "?"))
        tally[status] = tally.get(status, 0) + 1
        if status == "failed":
            failed.append(
                (
                    str(e.get("job_id", "?")),
                    str(e.get("company") or e.get("apply_link") or "?"),
                    str(e.get("error", "")),
                )
            )

    print(f"\n=== Batch tracking ({len(latest)} jobs) ===")
    order = ["submitted", "skipped", "awaiting_review", "deferred", "failed", "pending", "filling"]
    for status in order:
        if status in tally:
            print(f"  {status:<16} {tally[status]}")
    for status, count in sorted(tally.items()):
        if status not in order:
            print(f"  {status:<16} {count}")

    if failed:
        print("\n--- Failed runs (latest error) ---")
        for job_id, label, err in failed:
            print(f"  {job_id}  {label}\n      {err}")
    print()


def _print_last(entries: list[dict], n: int) -> None:
    for e in entries[-n:]:
        print(json.dumps(e, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Autofill batch tracking ledger")
    parser.add_argument("--quiet", action="store_true", help="snapshot only, no summary")
    parser.add_argument("--last", type=int, default=0, help="print last N ledger lines")
    args = parser.parse_args()

    entries = snapshot()

    if args.last > 0:
        _print_last(entries, args.last)
    elif not args.quiet:
        _summary(entries)


if __name__ == "__main__":
    main()
