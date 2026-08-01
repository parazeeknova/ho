"""Azure index worker: mass-crawl worldwide companies + job postings.

Runs on an Azure VM. Token-free sources:
  1. ATS boards (Greenhouse / Lever / Ashby) for every slug harvested from
     community GitHub READMEs - full index, no title/freshness filter.
  2. workatastartup.com/jobs paginated listings.
  3. Hacker News "Who is hiring" threads via the Algolia items API.

Everything is buffered and uploaded hourly to Azure Blob Storage
(container `radar-index`): `obs/{epoch_hour}.jsonl` for postings and
`companies/{epoch_hour}.jsonl` for the company index. The local machine
runs scripts/azure_ingest.py to pull the blobs into Postgres.

Config comes from /opt/radar-worker/config.env (AZURE_STORAGE_ACCOUNT,
AZURE_STORAGE_KEY, AZURE_CONTAINER).
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
from typing import Any

from azure.storage.blob import BlobServiceClient
from httpx import AsyncClient

from src.logging import get_logger

logger = get_logger("azure_crawl_worker")

GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
LEVER_API = "https://api.lever.co/v0/postings/{slug}"
ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board/{slug}"
WORKABLE_API = "https://apply.workable.com/api/v1/widget/accounts/{slug}"

CDX_ENDPOINT = "https://index.commoncrawl.org"

_CDX_DOMAINS = {
    "greenhouse": "boards.greenhouse.io/*",
    "lever": "jobs.lever.co/*",
    "ashby": "jobs.ashbyhq.com/*",
    "workable": "apply.workable.com/*",
}

_README_URLS = [
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/README.md",
    "https://raw.githubusercontent.com/LorenzoLaCorte/european-tech-internships-2026/main/README.md",
    "https://raw.githubusercontent.com/DereC4/internships-and-newgrad/main/README.md",
    "https://raw.githubusercontent.com/pittcsc/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/vanshb03/Summer2027-Internships/main/README.md",
    "https://raw.githubusercontent.com/cvrve/Summer2025-Internships/main/README.md",
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2025-Internships/dev/README.md",
    "https://raw.githubusercontent.com/Coding-Crashkurse/Summer-2026-Internships/main/README.md",
    "https://raw.githubusercontent.com/AlanChen4/Summer-2026-SWE-Internships/main/README.md",
]

_ATS_SLUG_RE = re.compile(
    r"https?://(?:boards\.greenhouse\.io|jobs\.lever\.co|jobs\.ashbyhq\.com|"
    r"apply\.workable\.com)/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)

_WAS_JOB_RE = re.compile(r'<a[^>]*href="(/jobs/\d+)"[^>]*>([^<]+)</a>', re.IGNORECASE)
_WAS_COMPANY_LINK_RE = re.compile(
    r'<a[^>]*href="(/companies/[a-z0-9-]+)"[^>]*>(.*?)</a>', re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


class Indexer:
    def __init__(self) -> None:
        conn_str = (
            "DefaultEndpointsProtocol=https;"
            f"AccountName={os.environ['AZURE_STORAGE_ACCOUNT']};"
            f"AccountKey={os.environ['AZURE_STORAGE_KEY']};"
            "EndpointSuffix=core.windows.net"
        )
        self.container = os.environ.get("AZURE_CONTAINER", "radar-index")
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container_client = self.client.get_container_client(self.container)
        self.obs: list[dict[str, Any]] = []
        self.companies: dict[str, dict[str, Any]] = {}
        self.seen_jobs: set[str] = set()
        self.slugs: dict[str, set[str]] = {
            "greenhouse": set(),
            "lever": set(),
            "ashby": set(),
            "workable": set(),
        }
        self.lock = asyncio.Lock()

    # ── state ────────────────────────────────────────────────────────

    async def load_state(self) -> None:
        try:
            blob = self.container_client.get_blob_client("state/ats_slugs.json")
            data = json.loads(blob.download_blob().readall())
            for k in self.slugs:
                self.slugs[k].update(data.get(k, []))
            logger.info(f"Loaded slug state: {sum(len(v) for v in self.slugs.values())} slugs")
        except Exception:
            logger.info("No slug state blob yet")

    async def save_state(self) -> None:
        data = {k: sorted(v) for k, v in self.slugs.items()}
        blob = self.container_client.get_blob_client("state/ats_slugs.json")
        blob.upload_blob(json.dumps(data).encode(), overwrite=True)

    async def flush(self) -> None:
        if not self.obs and not self.companies:
            return
        hour = int(time.time() // 3600)
        async with self.lock:
            if self.obs:
                body = "\n".join(json.dumps(o) for o in self.obs).encode()
                self.container_client.get_blob_client(f"obs/{hour}.jsonl").upload_blob(
                    body, overwrite=True
                )
                logger.info(f"Uploaded {len(self.obs)} observations to obs/{hour}.jsonl")
                self.obs = []
            if self.companies:
                body = "\n".join(json.dumps(c) for c in self.companies.values()).encode()
                self.container_client.get_blob_client(f"companies/{hour}.jsonl").upload_blob(
                    body, overwrite=True
                )
                logger.info(f"Uploaded {len(self.companies)} companies to companies/{hour}.jsonl")
            await self.save_state()

    def add_obs(self, obs: dict[str, Any]) -> None:
        key = obs["url"]
        if key in self.seen_jobs:
            return
        self.seen_jobs.add(key)
        self.obs.append(obs)

    def add_company(
        self,
        slug: str,
        platform: str,
        careers_url: str = "",
        name: str = "",
        location: str = "",
        job_count: int = 0,
    ) -> None:
        slug = slug.lower().strip()
        if not slug:
            return
        cur = self.companies.get(slug)
        if cur is None:
            cur = {
                "slug": slug,
                "platform": platform,
                "careers_url": careers_url,
                "name": name,
                "location": location,
                "job_count": job_count,
                "first_seen": time.time(),
                "last_seen": time.time(),
            }
            self.companies[slug] = cur
        else:
            cur["last_seen"] = time.time()
            cur["job_count"] = max(cur["job_count"], job_count)
            if name:
                cur["name"] = name

    # ── sources ──────────────────────────────────────────────────────

    async def harvest_slugs_from_cdx(self, client: AsyncClient) -> None:
        """Harvest tens of thousands of ATS board slugs from Common Crawl's CDX index.

        Crawl coverage varies wildly by collection (2025-43 has huge lever
        depth, 2026-30 has almost none), so paginate the most recent N
        collections per platform until the slug set plateaus.
        """
        try:
            coll = await client.get(f"{CDX_ENDPOINT}/collinfo.json", timeout=20.0)
            candidates = [c["id"] for c in coll.json()]
            for cid in list(candidates):
                try:
                    if not re.match(r"^CC-MAIN-202[56]-", cid):
                        candidates.remove(cid)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"cdx collinfo: {exc}")
            return
        if not candidates:
            logger.warning("cdx: no 2025/2026 collections found")
            return
        candidates = sorted(candidates)[-10:]
        logger.info(f"cdx: collections {candidates[0]}..{candidates[-1]} ({len(candidates)})")
        host_re = re.compile(r"https?://[^/]+/([^/?#]+)", re.IGNORECASE)
        junk = {
            "404",
            "about",
            "accounts",
            "admin",
            "api",
            "apply",
            "auth",
            "blog",
            "board",
            "boards",
            "careers",
            "contact",
            "embed",
            "favicon.ico",
            "feed",
            "index",
            "jobs",
            "legal",
            "login",
            "privacy",
            "robots",
            "robots.txt",
            "rss",
            "search",
            "sitemap",
            "sitemap.xml",
            "signin",
            "team",
            "terms",
            "v1",
            "wp-content",
            "www",
        }
        for platform, pattern in _CDX_DOMAINS.items():
            before = len(self.slugs[platform])
            for cid in candidates:
                page = 0
                errors = 0
                new_since_last_coll = 0
                clean = False
                while page < 250 and errors < 8:
                    try:
                        resp = await client.get(
                            f"{CDX_ENDPOINT}/{cid}-index",
                            params={
                                "url": pattern,
                                "output": "json",
                                "pageSize": 100,
                                "page": page,
                            },
                            timeout=90.0,
                        )
                        if resp.status_code != 200:
                            errors += 1
                            await asyncio.sleep(4)
                            continue
                        lines = [ln for ln in resp.text.splitlines() if ln.strip()]
                        if not lines:
                            clean = True
                            break
                        added = 0
                        for line in lines:
                            try:
                                rec = json.loads(line)
                                url = rec.get("url", "")
                                m = host_re.match(url)
                                if not m:
                                    continue
                                s = m.group(1).strip().lower()
                                if not s or s in junk or "." in s:
                                    continue
                                if s not in self.slugs[platform]:
                                    self.slugs[platform].add(s)
                                    added += 1
                            except Exception:
                                continue
                        new_since_last_coll += added
                        page += 1
                        await asyncio.sleep(1.5)
                    except Exception:
                        errors += 1
                        await asyncio.sleep(4)
                logger.info(
                    f"cdx {platform} {cid}: pages={page} errors={errors} "
                    f"clean={clean} +{new_since_last_coll} (total {len(self.slugs[platform])})"
                )
                if clean and new_since_last_coll == 0:
                    break
            logger.info(f"cdx {platform}: +{len(self.slugs[platform]) - before} slugs in harvest")
        total = sum(len(v) for v in self.slugs.values())
        logger.info(
            f"CDX harvest done: greenhouse={len(self.slugs['greenhouse'])} "
            f"lever={len(self.slugs['lever'])} ashby={len(self.slugs['ashby'])} "
            f"workable={len(self.slugs['workable'])} (total {total})"
        )

    async def harvest_slugs(self, client: AsyncClient) -> None:
        for url in _README_URLS:
            try:
                resp = await client.get(url, headers={"User-Agent": UA}, timeout=20.0)
                if resp.status_code != 200:
                    continue
                text = resp.text.lower()
                matches = _ATS_SLUG_RE.findall(resp.text)
                gh, lev, ash = (
                    "boards.greenhouse.io" in text,
                    "lever" in text,
                    "ashbyhq.com" in text,
                )
                for slug in matches:
                    slug = slug.lower().rstrip("/")
                    if slug in ("embed", "v1", "api", "www", "board"):
                        continue
                    if gh:
                        self.slugs["greenhouse"].add(slug)
                    if lev:
                        self.slugs["lever"].add(slug)
                    if ash:
                        self.slugs["ashby"].add(slug)
            except Exception as exc:
                logger.debug(f"harvest {url}: {exc}")
        logger.info(
            "Slug harvest complete: "
            f"greenhouse={len(self.slugs['greenhouse'])} lever={len(self.slugs['lever'])} "
            f"ashby={len(self.slugs['ashby'])}"
        )

    async def poll_ats(self, client: AsyncClient) -> None:
        sem = asyncio.Semaphore(30)
        jobs = list(self.slugs.items())

        async def _fetch(platform: str, slug: str) -> None:
            async with sem:
                try:
                    if platform == "greenhouse":
                        resp = await client.get(GREENHOUSE_API.format(slug=slug), timeout=10.0)
                        if resp.status_code != 200:
                            return
                        data = resp.json()
                        for item in data.get("jobs", []):
                            url = item.get("absolute_url") or (
                                f"https://boards.greenhouse.io/{slug}/jobs/{item['id']}"
                            )
                            self.add_obs(
                                {
                                    "url": url,
                                    "source": "greenhouse",
                                    "title": item.get("title", ""),
                                    "snippet": (
                                        f"{item.get('company_name', '')} | "
                                        f"{item.get('location', {}).get('name', '')}"
                                    ),
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "greenhouse",
                            careers_url=f"https://boards.greenhouse.io/{slug}",
                            name=data.get("name", ""),
                            job_count=len(data.get("jobs", [])),
                        )
                    elif platform == "lever":
                        resp = await client.get(LEVER_API.format(slug=slug), timeout=10.0)
                        if resp.status_code != 200:
                            return
                        for item in resp.json():
                            self.add_obs(
                                {
                                    "url": item.get("hostedUrl", ""),
                                    "source": "lever",
                                    "title": item.get("text", ""),
                                    "snippet": (
                                        f"{item.get('categories', {}).get('team', '')} | "
                                        f"{item.get('categories', {}).get('location', '')}"
                                    ),
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "lever",
                            careers_url=f"https://jobs.lever.co/{slug}",
                            name=slug,
                            job_count=len(resp.json()),
                        )
                    elif platform == "ashby":
                        resp = await client.get(ASHBY_API.format(slug=slug), timeout=10.0)
                        if resp.status_code != 200:
                            return
                        data = resp.json()
                        for item in data.get("jobs", []):
                            self.add_obs(
                                {
                                    "url": item.get("jobUrl", ""),
                                    "source": "ashby",
                                    "title": item.get("title", ""),
                                    "snippet": f"{data.get('name', '')} | {item.get('location', '')}",  # noqa: E501
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "ashby",
                            careers_url=f"https://jobs.ashbyhq.com/{slug}",
                            name=data.get("name", ""),
                            job_count=len(data.get("jobs", [])),
                        )
                    elif platform == "workable":
                        resp = await client.get(WORKABLE_API.format(slug=slug), timeout=10.0)
                        if resp.status_code != 200:
                            return
                        data = resp.json()
                        for item in data.get("jobs", []):
                            self.add_obs(
                                {
                                    "url": item.get("url", "") or item.get("absolute_url", ""),
                                    "source": "workable",
                                    "title": item.get("title", ""),
                                    "snippet": f"{data.get('name', '')} | {item.get('city', '')}",
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "workable",
                            careers_url=f"https://apply.workable.com/{slug}/",
                            name=data.get("name", ""),
                            job_count=len(data.get("jobs", [])),
                        )
                except Exception as exc:
                    logger.debug(f"ats {platform}/{slug}: {exc}")

        tasks = [_fetch(p, s) for p, slugs in jobs for s in slugs]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"ATS poll done: {len(self.seen_jobs)} unique jobs so far")

    async def poll_workatastartup(self, client: AsyncClient) -> None:
        page = 1
        while page <= 60:
            try:
                resp = await client.get(
                    f"https://www.workatastartup.com/jobs?page={page}",
                    headers={"User-Agent": UA},
                    timeout=20.0,
                )
                if resp.status_code != 200:
                    break
                text = resp.text
                found = 0
                for href, label in _WAS_JOB_RE.findall(text):
                    url = f"https://www.workatastartup.com{href}"
                    self.add_obs(
                        {
                            "url": url,
                            "source": "workatastartup",
                            "title": html.unescape(label.strip()),
                            "snippet": "",
                            "raw_markdown": "",
                            "observed_at": time.time(),
                            "source_freshness_evidence": "was index",
                        }
                    )
                    found += 1
                for href, label in _WAS_COMPANY_LINK_RE.findall(text):
                    label = html.unescape(_TAG_RE.sub("", label)).strip()
                    self.add_company(
                        href.split("/")[-1],
                        "workatastartup",
                        careers_url=f"https://www.workatastartup.com{href}",
                        name=label,
                    )
                logger.info(f"workatastartup page {page}: {found} jobs")
                if found == 0:
                    break
                page += 1
                await asyncio.sleep(0.6)
            except Exception as exc:
                logger.warning(f"workatastartup page {page}: {exc}")
                break

    async def poll_hn(self, client: AsyncClient) -> None:
        try:
            resp = await client.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": "Who is hiring", "tags": "story", "hitsPerPage": 3},
                timeout=20.0,
            )
            hits = resp.json().get("hits", [])
            if not hits:
                return
            story = hits[0]
            item_resp = await client.get(
                f"https://hn.algolia.com/api/v1/items/{story['objectID']}", timeout=30.0
            )
            for child in item_resp.json().get("children", []):
                text = (child.get("text") or "").strip()
                if not text:
                    continue
                link = ""
                m = re.search(r'href="(https?://[^"]+)"', text)
                if m:
                    link = m.group(1)
                self.add_obs(
                    {
                        "url": link or f"https://news.ycombinator.com/item?id={child['id']}",
                        "source": "hn_whoishiring",
                        "title": "",
                        "snippet": html.unescape(_TAG_RE.sub("", text))[:400],
                        "raw_markdown": text,
                        "observed_at": time.time(),
                        "source_freshness_evidence": f"hn story {story['objectID']}",
                    }
                )
            logger.info(f"HN thread {story['objectID']}: parsed comments")
        except Exception as exc:
            logger.warning(f"hn: {exc}")

    async def poll_hn_history(self, client: AsyncClient) -> None:
        """Pull the last 6 'Who is hiring' threads (monthly history)."""
        try:
            resp = await client.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": "Who is hiring", "tags": "story", "hitsPerPage": 6},
                timeout=20.0,
            )
            hits = resp.json().get("hits", [])
            seen = set()
            for story in hits:
                oid = story["objectID"]
                if oid in seen:
                    continue
                seen.add(oid)
                try:
                    item_resp = await client.get(
                        f"https://hn.algolia.com/api/v1/items/{oid}", timeout=30.0
                    )
                    for child in item_resp.json().get("children", []):
                        text = (child.get("text") or "").strip()
                        if not text:
                            continue
                        link = ""
                        m = re.search(r'href="(https?://[^"]+)"', text)
                        if m:
                            link = m.group(1)
                        self.add_obs(
                            {
                                "url": link
                                or f"https://news.ycombinator.com/item?id={child['id']}",
                                "source": "hn_whoishiring",
                                "title": "",
                                "snippet": html.unescape(_TAG_RE.sub("", text))[:400],
                                "raw_markdown": text,
                                "observed_at": time.time(),
                                "source_freshness_evidence": f"hn story {oid}",
                            }
                        )
                except Exception as exc:
                    logger.debug(f"hn history {oid}: {exc}")
            logger.info(f"HN history: parsed {len(seen)} threads")
        except Exception as exc:
            logger.warning(f"hn history: {exc}")

    async def poll_remotive(self, client: AsyncClient) -> None:
        """Remotive public API - token-free JSON job feed."""
        try:
            resp = await client.get("https://remotive.com/api/remote-jobs", timeout=30.0)
            if resp.status_code != 200:
                return
            for item in resp.json().get("jobs", []):
                self.add_obs(
                    {
                        "url": item.get("url", ""),
                        "source": "remotive",
                        "title": item.get("title", ""),
                        "snippet": (
                            f"{item.get('company_name', '')} | "
                            f"{item.get('candidate_required_location', '')}"
                        ),
                        "raw_markdown": json.dumps(item),
                        "observed_at": time.time(),
                        "source_freshness_evidence": "remotive api",
                    }
                )
                self.add_company(
                    str(item.get("company_name", "")).lower().replace(" ", "-"),
                    "remotive",
                    careers_url=f"https://remotive.com/company/{item.get('company_name', '')}",
                    name=item.get("company_name", ""),
                    location=item.get("candidate_required_location", ""),
                )
            logger.info("Remotive poll done")
        except Exception as exc:
            logger.warning(f"remotive: {exc}")

    async def poll_arbeitnow(self, client: AsyncClient) -> None:
        """Arbeitnow public API - token-free JSON job feed (worldwide)."""
        try:
            resp = await client.get("https://www.arbeitnow.com/api/job-board-api", timeout=30.0)
            if resp.status_code != 200:
                return
            for item in resp.json().get("data", []):
                self.add_obs(
                    {
                        "url": item.get("url", ""),
                        "source": "arbeitnow",
                        "title": item.get("title", ""),
                        "snippet": (f"{item.get('company_name', '')} | {item.get('location', '')}"),
                        "raw_markdown": json.dumps(item),
                        "observed_at": time.time(),
                        "source_freshness_evidence": "arbeitnow api",
                    }
                )
                self.add_company(
                    str(item.get("company_name", "")).lower().replace(" ", "-"),
                    "arbeitnow",
                    careers_url=item.get("company_url", ""),
                    name=item.get("company_name", ""),
                    location=item.get("location", ""),
                )
            logger.info("Arbeitnow poll done")
        except Exception as exc:
            logger.warning(f"arbeitnow: {exc}")

    # ── main loop ────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.load_state()
        last_ats = 0.0
        last_cdx = 0.0
        last_was = 0.0
        last_hn = 0.0
        last_hn_hist = 0.0
        last_remotive = 0.0
        last_arbeitnow = 0.0
        while True:
            async with AsyncClient(
                headers={"User-Agent": UA}, timeout=15.0, follow_redirects=True
            ) as client:
                now = time.monotonic()
                if now - last_ats > 7200 or last_ats == 0:
                    await self.harvest_slugs(client)
                    await self.poll_ats(client)
                    last_ats = time.monotonic()
                if now - last_cdx > 21600 or last_cdx == 0:
                    await self.harvest_slugs_from_cdx(client)
                    last_cdx = time.monotonic()
                if now - last_was > 3600 or last_was == 0:
                    await self.poll_workatastartup(client)
                    last_was = time.monotonic()
                if now - last_hn > 3600 or last_hn == 0:
                    await self.poll_hn(client)
                    last_hn = time.monotonic()
                if now - last_hn_hist > 21600 or last_hn_hist == 0:
                    await self.poll_hn_history(client)
                    last_hn_hist = time.monotonic()
                if now - last_remotive > 3600 or last_remotive == 0:
                    await self.poll_remotive(client)
                    last_remotive = time.monotonic()
                if now - last_arbeitnow > 3600 or last_arbeitnow == 0:
                    await self.poll_arbeitnow(client)
                    last_arbeitnow = time.monotonic()
            await self.flush()
            await asyncio.sleep(600)


async def main() -> None:
    idx = Indexer()
    try:
        await idx.run()
    finally:
        await idx.flush()


if __name__ == "__main__":
    asyncio.run(main())
