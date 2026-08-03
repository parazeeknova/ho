"""Azure relic worker: mine founder intelligence for known companies.

Runs on the Azure VM (or any VPS) alongside crawl_worker.py. Reads the
newest ``companies/`` blob, mines founder details (names, titles, LinkedIn,
github, email) for each company via token-free web search, and uploads the
results as ``founders/{hour}_{seq}.jsonl``.

The local machine's ingest consumes ``founders/`` blobs into the
``company_osint`` table, feeding warm-intro paths and outreach.

Run:
    uv run --with azure-storage-blob python3 scripts/azure/founder_miner.py
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx
from azure_intel_common import container_client, log, newest_blob, upload_records

_LINKEDIN_RE = re.compile(r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_%.\-]+")
_GITHUB_RE = re.compile(r"https?://github\.com/[A-Za-z0-9_\-]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Companies we've already enriched (in-memory; checkpoints keep it small).
_DONE: set[str] = set()


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


def _extract(snippets: list[str], company: str) -> list[dict[str, Any]]:
    """Lightweight founder extraction from search snippets (token-free).

    Only accepts a proper "First Last" name (two capitalised words) that
    directly precedes a founder/exec title. Loose matches like "board of
    dire" or "by creative dire" are rejected by requiring the title to be an
    exact keyword and the name to be exactly two clean words.
    """
    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    name_re = re.compile(
        r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\s*[-–,|]?\s*"
        r"(co-?founder|founder|ceo|cto|cmo|cpo|chief\s+\w+|"
        r"president|coo|chief executive|vice president)\b",
        re.I,
    )
    for s in snippets:
        text = re.sub(r"\s+", " ", s)
        for m in name_re.finditer(text):
            first, last = m.group(1), m.group(2)
            name = f"{first} {last}"
            # Reject fragments that are actually the tail of a longer phrase.
            # "Board of Dire" fails because "Dire" is not a capital word; but
            # guard against accidental article/preposition endings.
            if last.lower() in ("the", "and", "of", "for", "in", "by", "dire", "executive"):
                continue
            if len(name) < 6 or len(name) > 40:
                continue
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            founder: dict[str, Any] = {
                "name": name,
                "title": m.group(3).title(),
                "company": company,
            }
            ln = _LINKEDIN_RE.search(text)
            if ln:
                founder["linkedin_url"] = ln.group(0).rstrip(".,)")
            gh = _GITHUB_RE.search(text)
            if gh:
                founder["github_url"] = gh.group(0).rstrip(".,)")
            em = _EMAIL_RE.search(text)
            if em and em.group(0).split("@")[1] not in ("example.com", "email.com"):
                founder["email"] = em.group(0)
            out.append(founder)
    return out


async def mine_one(company: str) -> list[dict[str, Any]]:
    qs = [
        f'"{company}" founder OR co-founder OR CEO',
        f'"{company}" "co-founder" linkedin',
    ]
    snippets: list[str] = []
    for q in qs:
        snippets.extend(_searxng_search(q))
    return _extract(snippets, company)


async def run(limit: int = 300) -> None:
    cc = container_client()
    companies = newest_blob("companies/", cc)
    if not companies:
        log("no companies blob yet")
        return
    total = 0
    for rec in companies["records"]:
        slug = (rec.get("slug") or "").strip()
        if not slug or slug in _DONE:
            continue
        founders = await mine_one(slug)
        if founders:
            total += len(founders)
            name = upload_records("founders", founders, cc)
            log(f"  {slug}: {len(founders)} founders -> {name}")
            _DONE.add(slug)
        await asyncio.sleep(0.5)
        if total >= limit:
            break
    log(f"done: uploaded {total} founder records")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    asyncio.run(run(args.limit))


if __name__ == "__main__":
    main()
