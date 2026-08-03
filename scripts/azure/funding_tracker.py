"""Azure relic worker: track funding rounds + hiring signals per company.

Runs on the Azure VM (or any VPS). Reads the newest ``companies/`` blob,
and for each company queries token-free web sources for funding rounds
(amount, stage, lead investors, date) and hiring-expansion signals
(recent job posts, product launches). Uploads results as
``signals/{hour}_{seq}.jsonl``.

The local machine's ingest consumes ``signals/`` blobs into
``company_osint``, powering the "just raised / hiring now" alert tier.

Run:
    uv run --with azure-storage-blob python3 scripts/azure/funding_tracker.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

import httpx
from azure_intel_common import container_client, log, newest_blob, upload_records

_MONEY_RE = re.compile(
    r"[$€£]\s?([0-9]+(?:\.[0-9]+)?)\s?([MBK])(?:\s?(?:million|billion|k))?", re.I
)
_STAGE_RE = re.compile(r"\b(pre-?seed|seed|series\s*[a-z]|venture|growth|late-stage)\b", re.I)
_ROUND_RE = re.compile(r"\b(raised|announces?|secured|closes?|led)\s", re.I)
_DATE_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+[0-9]{1,2}(?:,\s*[0-9]{4})?\b",
    re.I,
)


def _searxng_search(q: str) -> list[str]:
    url = os.environ.get("SEARXNG_URL", "http://localhost:8080/search")
    try:
        r = httpx.get(url, params={"q": q, "format": "json"}, timeout=8.0)
        if r.status_code == 200:
            return [
                f"{x.get('title', '')} {x.get('content', '')}" for x in r.json().get("results", [])
            ]
    except Exception:
        pass
    return []


def _parse_funding(snippets: list[str], company: str) -> dict[str, Any]:
    text = " ".join(snippets)
    out: dict[str, Any] = {"company": company, "ts": time.time()}
    m = _MONEY_RE.search(text)
    if m:
        amount = float(m.group(1))
        mult = {"M": 1e6, "B": 1e9, "K": 1e3}.get(m.group(2).upper(), 1e6)
        out["amount_usd"] = amount * mult
    s = _STAGE_RE.search(text)
    if s:
        out["stage"] = s.group(1).lower()
    if _ROUND_RE.search(text):
        out["funding_event"] = True
    d = _DATE_RE.search(text)
    if d:
        out["date"] = d.group(0)
    # Lead investor: "led by X" / "with participation from Y"
    led = re.search(r"\bled\s+by\s+([A-Z][A-Za-z0-9 .&]+?)(?:[,.]|and|to|\s)", text)
    if led:
        out["lead_investor"] = led.group(1).strip()
    if len(out) <= 2:
        return {}
    return out


async def run(limit: int = 300) -> None:
    cc = container_client()
    companies = newest_blob("companies/", cc)
    if not companies:
        log("no companies blob yet")
        return
    total = 0
    done: set[str] = set()
    for rec in companies["records"]:
        slug = (rec.get("slug") or "").strip()
        if not slug or slug in done:
            continue
        snippets = _searxng_search(f'"{slug}" funding raised OR seed OR "series"')
        funding = _parse_funding(snippets, slug)
        if funding:
            name = upload_records("signals", [funding], cc)
            log(f"  {slug}: {json.dumps(funding)[:120]} -> {name}")
            total += 1
            done.add(slug)
        await asyncio.sleep(0.5)
        if total >= limit:
            break
    log(f"done: uploaded {total} funding/signal records")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    asyncio.run(run(args.limit))


if __name__ == "__main__":
    main()
