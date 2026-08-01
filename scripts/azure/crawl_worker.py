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

_CDX_DOMAINS: dict[str, tuple[str, str]] = {
    "greenhouse": ("boards.greenhouse.io/*", r"boards\.greenhouse\.io/([a-z0-9_-]+)"),
    "lever": ("jobs.lever.co/*", r"jobs\.lever\.co/([a-z0-9_-]+)"),
    "ashby": ("jobs.ashbyhq.com/*", r"jobs\.ashbyhq\.com/([a-z0-9_-]+)"),
    "workable": ("apply.workable.com/*", r"apply\.workable\.com/([a-z0-9_-]+)"),
    "smartrecruiters": (
        "jobs.smartrecruiters.com/*",
        r"jobs\.smartrecruiters\.com/([a-zA-Z0-9_-]+)",
    ),
    "teamtailor": ("*.teamtailor.com/jobs/*", r"([a-z0-9-]+)\.teamtailor\.com"),
    "recruitee": ("*.recruitee.com/*", r"([a-z0-9-]+)\.recruitee\.com"),
    "comeet": ("www.comeet.com/jobs/*", r"comeet\.com/jobs/([a-zA-Z0-9_-]+)"),
    "eightfold": ("*.eightfold.ai/*", r"([a-z0-9-]+)\.eightfold\.ai"),
}

# Discovery-only ATS families: we harvest their board slugs from the CDX
# index (feeding the company corpus and future pollers) even though no
# public JSON API exists to poll them yet.
_CDX_DISCOVERY: dict[str, tuple[str, str]] = {
    "workday": ("*.myworkdayjobs.com/*", r"([a-z0-9-]+)\.wd\d?\.myworkdayjobs\.com"),
    "icims": ("*.icims.com/*", r"([a-z0-9-]+)\.icims\.com"),
    "jazzhr": ("*.jazzhr.com/*", r"([a-z0-9-]+)\.jazzhr\.com"),
    "bamboohr": ("*.bamboohr.com/jobs/*", r"([a-z0-9-]+)\.bamboohr\.com"),
    "jobvite": ("jobs.jobvite.com/*", r"jobs\.jobvite\.com/([a-zA-Z0-9_-]+)"),
    "bullhorn": ("*.bullhornstaffing.com/*", r"([a-z0-9-]+)\.bullhornstaffing\.com"),
    "successfactors": (
        "*.successfactors.*/*",
        r"([a-z0-9-]+)\.(?:successfactors\.(?:eu|com)|sapsf\.com)",
    ),
    "taleo": ("*.taleo.net/*", r"([a-z0-9-]+)\.taleo\.net"),
    "ukg": ("*.ultipro.com/*", r"([a-z0-9-]+)\.ultipro\.com"),
    "oraclehcm": ("*.oraclecloud.com/hcmUI/*", r"([a-z0-9-]+)\.oraclecloud\.com"),
    "peopleadmin": ("*.peopleadmin.com/*", r"([a-z0-9-]+)\.peopleadmin\.com"),
    "phenom": ("*.phenompeople.com/*", r"([a-z0-9-]+)\.phenompeople\.com"),
    "cornerstone": ("*.cornerstoneondemand.com/*", r"([a-z0-9-]+)\.cornerstoneondemand\.com"),
    "plum": ("*.plum.io/*", r"([a-z0-9-]+)\.plum\.io"),
    "sapling": ("*.saplinghr.com/*", r"([a-z0-9-]+)\.saplinghr\.com"),
    "bamboohr_2": ("*.bamboohr.com/*", r"([a-z0-9-]+)\.bamboohr\.com"),
    "adp": ("*.adp.com/jobs/*", r"([a-z0-9-]+)\.adp\.com"),
    "successfactors_2": ("*.sapsf.com/*", r"([a-z0-9-]+)\.sapsf\.com"),
    "cerner": ("*.cerner.com/*", r"([a-z0-9-]+)\.cerner\.com"),
    "dice": ("*.dice.com/jobs/*", r"([a-z0-9-]+)\.dice\.com"),
}

_ATS_API: dict[str, str] = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
    "lever": "https://api.lever.co/v0/postings/{slug}?mode=json",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{slug}",
    "workable": "https://apply.workable.com/api/v1/widget/accounts/{slug}",
    "smartrecruiters": ("https://api.smartrecruiters.com/v1/companies/{slug}/postings?limit=100"),
    "teamtailor": "https://{slug}.teamtailor.com/jobs.json",
    "recruitee": "https://{slug}.recruitee.com/api/offers/",
    "comeet": "https://www.comeet.com/api/v1/jobs/{slug}",
    "eightfold": ("https://{slug}.eightfold.ai/api/apply/v2/jobs?domain={slug}&num=100"),
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


_SUFFIXES = (
    "inc",
    "llc",
    "labs",
    "hq",
    "co",
    "corp",
    "corporation",
    "technologies",
    "technology",
    "systems",
    "software",
    "group",
    "holdings",
    "ventures",
    "partners",
    "capital",
    "cloud",
    "digital",
    "platform",
    "networks",
    "media",
    "studio",
    "works",
    "ai",
)

_ATS_PROBES = _ATS_API

_FRESHER_RE = re.compile(
    r"\b(?:intern(?:ship)?|interns|graduate|graduates?|grad|entry[- ]level|"
    r"early[- ]career|junior|new[- ]grad|fresher|freshers?|trainee|"
    r"apprentice|campus|co-?op|student|202[2-6]\s*[bB]atch|"
    r"recent[- ]graduate|undergraduate|graduate[- ]program)\b",
    re.IGNORECASE,
)

_SENIOR_RE = re.compile(
    r"\b(?:senior|staff|principal|lead|head|director|vp|manager|architect|"
    r"sr\.?|experienced|10\+|15\+)\b",
    re.IGNORECASE,
)


def is_fresher_role(title: str, snippet: str = "") -> bool:
    """Entry-level classifier: fresher signals beat senior signals."""
    text = f"{title} {snippet}"
    if _SENIOR_RE.search(text) and not _FRESHER_RE.search(title):
        return False
    return bool(_FRESHER_RE.search(text))


def _slug_variants(name: str) -> list[str]:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        return []
    variants = {slug}
    for suffix in _SUFFIXES:
        if slug.endswith(f"-{suffix}") and len(slug) > len(suffix) + 2:
            variants.add(slug[: -(len(suffix) + 1)])
    return list(variants)


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
            "smartrecruiters": set(),
            "teamtailor": set(),
            "recruitee": set(),
            "comeet": set(),
            "eightfold": set(),
            "discovery": set(),
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

    async def save_checkpoint(self) -> None:
        """Write a dated state snapshot so progress survives the relic."""
        try:
            data = {k: sorted(v) for k, v in self.slugs.items()}
            meta = {
                "ts": int(time.time()),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "companies": len(self.companies),
                "seen_jobs": len(self.seen_jobs),
                "slugs": data,
            }
            name = time.strftime("state/checkpoints/%Y%m%d-%H%M%S.json")
            self.container_client.get_blob_client(name).upload_blob(
                json.dumps(meta).encode(), overwrite=False
            )
            logger.info(f"checkpoint saved: {name}")
        except Exception as exc:
            logger.warning(f"checkpoint save failed: {exc}")

    async def flush(self) -> None:
        if not self.obs and not self.companies:
            return
        hour = int(time.time() // 3600)
        async with self.lock:
            if self.obs:
                body = "\n".join(json.dumps(o) for o in self.obs).encode()
                blob_name = f"obs/{hour}.jsonl"
                blob_client = self.container_client.get_blob_client(blob_name)
                try:
                    blob_client.create_append_blob()
                    blob_client.append_block(body)
                except Exception:
                    # Existing block blob (legacy) or race: write a unique
                    # append blob so nothing is lost this cycle.
                    seq = int(time.time() * 1000) % 100000
                    alt = f"obs/{hour}_{seq}.jsonl"
                    alt_client = self.container_client.get_blob_client(alt)
                    alt_client.create_append_blob()
                    alt_client.append_block(body)
                    blob_name = alt
                freshers = [o for o in self.obs if o.get("fresher")]
                if freshers:
                    fbody = "\n".join(json.dumps(o) for o in freshers).encode()
                    fblob = f"freshers/{hour}.jsonl"
                    f_client = self.container_client.get_blob_client(fblob)
                    try:
                        f_client.create_append_blob()
                        f_client.append_block(fbody)
                    except Exception:
                        fseq = int(time.time() * 1000) % 100000
                        fblob = f"freshers/{hour}_{fseq}.jsonl"
                        alt_f = self.container_client.get_blob_client(fblob)
                        alt_f.create_append_blob()
                        alt_f.append_block(fbody)
                    logger.info(
                        f"Uploaded {len(freshers)} fresher observations to freshers/{fblob}"
                    )
                logger.info(f"Appended {len(self.obs)} observations to obs/{blob_name}")
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
        obs["fresher"] = is_fresher_role(obs.get("title", ""), obs.get("snippet", ""))
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
        candidates = sorted(candidates)
        # Walk the whole 2024-2026 crawl history, not just the newest dozen -
        # older collections surface slugs newer ones never captured.
        candidates = [c for c in candidates if re.match(r"^CC-MAIN-202[456]-", c)]
        logger.info(f"cdx: collections {candidates[0]}..{candidates[-1]} ({len(candidates)})")
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

        async def walk_one(platform: str, cid: str) -> None:
            spec = _CDX_DOMAINS.get(platform) or _CDX_DISCOVERY.get(platform)
            if spec is None:
                return
            pattern, slug_re = spec
            slug_re = re.compile(slug_re, re.IGNORECASE)
            target = "discovery" if platform in _CDX_DISCOVERY else platform
            page = 0
            errors = 0
            new_since_last_coll = 0
            clean = False
            while page < 250 and errors < 15:
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
                        await asyncio.sleep(2)
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
                            m = slug_re.search(url)
                            if not m:
                                continue
                            s = m.group(1).strip().lower()
                            if not s or s in junk or "." in s:
                                continue
                            if s not in self.slugs[target]:
                                self.slugs[target].add(s)
                                added += 1
                        except Exception:
                            continue
                    new_since_last_coll += added
                    page += 1
                    await asyncio.sleep(0.2)
                except Exception:
                    errors += 1
                    await asyncio.sleep(2)
            logger.info(
                f"cdx {platform} {cid}: pages={page} errors={errors} "
                f"clean={clean} +{new_since_last_coll} (total {len(self.slugs[target])})"
            )

        sem = asyncio.Semaphore(32)

        async def _gated(p: str, c: str) -> None:
            async with sem:
                await walk_one(p, c)

        tasks = [_gated(p, c) for p in (*_CDX_DOMAINS, *_CDX_DISCOVERY) for c in candidates]
        await asyncio.gather(*tasks)
        parts = " ".join(
            f"{p}={len(self.slugs.get('discovery' if p in _CDX_DISCOVERY else p, set()))}"
            for p in (*_CDX_DOMAINS, *_CDX_DISCOVERY)
        )
        discovery = len(self.slugs.get("discovery", set()))
        total = sum(len(v) for v in self.slugs.values())
        logger.info(f"CDX harvest done: {parts} discovery={discovery} (total {total})")

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
        sem = asyncio.Semaphore(400)
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
                    elif platform == "smartrecruiters":
                        resp = await client.get(
                            _ATS_API["smartrecruiters"].format(slug=slug), timeout=10.0
                        )
                        if resp.status_code != 200:
                            return
                        data = resp.json()
                        items = data.get("postings", data) if isinstance(data, dict) else data
                        for item in items:
                            self.add_obs(
                                {
                                    "url": item.get("jobUrl", ""),
                                    "source": "smartrecruiters",
                                    "title": item.get("name", ""),
                                    "snippet": item.get("location", ""),
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "smartrecruiters",
                            careers_url=f"https://jobs.smartrecruiters.com/{slug}",
                            name=slug,
                            job_count=len(items),
                        )
                    elif platform == "teamtailor":
                        resp = await client.get(
                            _ATS_API["teamtailor"].format(slug=slug), timeout=10.0
                        )
                        if resp.status_code != 200:
                            return
                        for item in resp.json():
                            self.add_obs(
                                {
                                    "url": item.get("url") or "",
                                    "source": "teamtailor",
                                    "title": item.get("title", ""),
                                    "snippet": item.get("location", ""),
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "teamtailor",
                            careers_url=f"https://{slug}.teamtailor.com",
                            name=slug,
                            job_count=len(resp.json()),
                        )
                    elif platform == "recruitee":
                        resp = await client.get(
                            _ATS_API["recruitee"].format(slug=slug), timeout=10.0
                        )
                        if resp.status_code != 200:
                            return
                        items = resp.json().get("offers", [])
                        for item in items:
                            self.add_obs(
                                {
                                    "url": item.get("careers_url", ""),
                                    "source": "recruitee",
                                    "title": item.get("title", ""),
                                    "snippet": item.get("location", ""),
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "recruitee",
                            careers_url=f"https://{slug}.recruitee.com",
                            name=slug,
                            job_count=len(items),
                        )
                    elif platform == "comeet":
                        resp = await client.get(_ATS_API["comeet"].format(slug=slug), timeout=10.0)
                        if resp.status_code != 200:
                            return
                        data = resp.json()
                        items = data.get("jobs", data) if isinstance(data, dict) else data
                        for item in items:
                            self.add_obs(
                                {
                                    "url": item.get("url", ""),
                                    "source": "comeet",
                                    "title": item.get("title", ""),
                                    "snippet": item.get("location", ""),
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "comeet",
                            careers_url=f"https://www.comeet.com/jobs/{slug}",
                            name=slug,
                            job_count=len(items),
                        )
                    elif platform == "eightfold":
                        resp = await client.get(
                            _ATS_API["eightfold"].format(slug=slug), timeout=10.0
                        )
                        if resp.status_code != 200:
                            return
                        items = resp.json().get("positions", [])
                        for item in items:
                            self.add_obs(
                                {
                                    "url": item.get("canonicalPositionUrl", ""),
                                    "source": "eightfold",
                                    "title": item.get("name", ""),
                                    "snippet": item.get("location", ""),
                                    "raw_markdown": json.dumps(item),
                                    "observed_at": time.time(),
                                    "source_freshness_evidence": "ats json",
                                }
                            )
                        self.add_company(
                            slug,
                            "eightfold",
                            careers_url=f"https://{slug}.eightfold.ai/careers",
                            name=slug,
                            job_count=len(items),
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

    async def poll_remoteok(self, client: AsyncClient) -> None:
        """RemoteOK public API - large token-free remote job feed."""
        try:
            resp = await client.get("https://remoteok.com/api", timeout=30.0)
            if resp.status_code != 200:
                return
            for item in resp.json():
                if not isinstance(item, dict) or not item.get("url"):
                    continue
                self.add_obs(
                    {
                        "url": item.get("url", ""),
                        "source": "remoteok",
                        "title": item.get("position", ""),
                        "snippet": f"{item.get('company', '')} | Remote",
                        "raw_markdown": json.dumps(item),
                        "observed_at": time.time(),
                        "source_freshness_evidence": "remoteok api",
                    }
                )
                self.add_company(
                    str(item.get("company", "")).lower().replace(" ", "-"),
                    "remoteok",
                    careers_url=item.get("company_url", ""),
                    name=item.get("company", ""),
                )
            logger.info("RemoteOK poll done")
        except Exception as exc:
            logger.warning(f"remoteok: {exc}")

    async def poll_himalayas(self, client: AsyncClient) -> None:
        """Himalayas public API - curated remote tech jobs."""
        try:
            resp = await client.get(
                "https://himalayas.app/jobs/api",
                headers={"Accept": "application/json"},
                timeout=30.0,
            )
            if resp.status_code != 200:
                logger.info(f"himalayas: blocked ({resp.status_code})")
                return
            for item in resp.json().get("jobs", []):
                self.add_obs(
                    {
                        "url": item.get("url", ""),
                        "source": "himalayas",
                        "title": item.get("title", ""),
                        "snippet": f"{item.get('company', {}).get('name', '')} | Remote",
                        "raw_markdown": json.dumps(item),
                        "observed_at": time.time(),
                        "source_freshness_evidence": "himalayas api",
                    }
                )
                cname = (item.get("company") or {}).get("name", "")
                if cname:
                    self.add_company(
                        str(cname).lower().replace(" ", "-"),
                        "himalayas",
                        name=cname,
                    )
            logger.info("Himalayas poll done")
        except Exception as exc:
            logger.warning(f"himalayas: {exc}")

    async def poll_jobicy(self, client: AsyncClient) -> None:
        """Jobicy public API - remote job feed (no params needed)."""
        try:
            resp = await client.get(
                "https://jobicy.com/api/v2/remote-jobs",
                timeout=30.0,
            )
            if resp.status_code != 200:
                return
            for item in resp.json().get("jobs", []):
                self.add_obs(
                    {
                        "url": item.get("url", ""),
                        "source": "jobicy",
                        "title": item.get("jobTitle", ""),
                        "snippet": (
                            f"{item.get('companyName', '')} | "
                            f"{item.get('jobGeo', '')}"
                        ),
                        "raw_markdown": json.dumps(item),
                        "observed_at": time.time(),
                        "source_freshness_evidence": "jobicy api",
                    }
                )
                cname = item.get("companyName", "")
                if cname:
                    self.add_company(
                        str(cname).lower().replace(" ", "-"),
                        "jobicy",
                        name=cname,
                    )
            logger.info("Jobicy poll done")
        except Exception as exc:
            logger.warning(f"jobicy: {exc}")

    # ── main loop ────────────────────────────────────────────────────

    async def resolve_directory_slugs(self, client: AsyncClient) -> None:
        """Probe the Chad directory's companies for live ATS board slugs.

        Every company name is slugified (plus suffix-stripped variants)
        and probed against the public JSON endpoints of greenhouse, lever,
        ashby and workable. YC batches 2021-2026 are probed first - they
        are the most likely to still be operating with a live board.
        """
        try:
            blob = self.container_client.get_blob_client("directory/companies.jsonl")
            data = blob.download_blob().readall()
            companies = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        except Exception as exc:
            logger.info(f"directory: no companies.jsonl yet ({exc})")
            return

        def _key(c: dict[str, Any]) -> tuple[int, int]:
            srcs = c.get("sources", [])
            try:
                year = int(c.get("year") or 0)
            except (TypeError, ValueError):
                year = 0
            if "yc" in srcs and 2021 <= year <= 2026:
                priority = 0
            elif "yc" in srcs:
                priority = 1
            else:
                priority = 2
            return (priority, -year if year else 0)

        companies.sort(key=_key)
        known = {s for v in self.slugs.values() for s in v}
        targets: list[tuple[str, str]] = []
        for c in companies:
            for variant in _slug_variants(c.get("name", "")):
                if variant in known:
                    continue
                for platform in _ATS_PROBES:
                    targets.append((platform, variant))
        targets = list(dict.fromkeys(targets))
        logger.info(
            f"directory: {len(companies)} companies, {len(targets)} probes "
            f"(known slugs {len(known)})"
        )
        if not targets:
            return

        sem = asyncio.Semaphore(300)
        found: dict[str, set[str]] = {k: set() for k in self.slugs}
        probed = 0

        async def probe(platform: str, slug: str) -> None:
            nonlocal probed
            async with sem:
                url = _ATS_PROBES[platform].format(slug=slug)
                try:
                    r = await client.get(url)
                    ctype = r.headers.get("content-type", "")
                    if r.status_code == 200 and "application/json" in ctype:
                        found[platform].add(slug)
                except Exception:
                    pass
                probed += 1
                if probed % 500 == 0:
                    logger.info(f"directory probe: {probed}/{len(targets)}")

        wave = 2000
        for start in range(0, len(targets), wave):
            chunk = targets[start : start + wave]
            await asyncio.gather(*(probe(p, s) for p, s in chunk))
            async with self.lock:
                for p, slugs in found.items():
                    self.slugs[p].update(slugs)
            await self.save_state()
            new = sum(len(v) for v in found.values())
            total = sum(len(v) for v in self.slugs.values())
            logger.info(f"directory wave {start // wave + 1}: +{new} new slugs (total {total})")

    async def crawl_directory_sitemaps(self, client: AsyncClient) -> None:
        """Live discovery: robots.txt + sitemap crawl for directory companies.

        Resolves candidate domains (slug + .com), reads robots.txt, then
        scans the referenced sitemaps for ATS board URLs, feeding new
        slugs into the store. This is the freshest possible signal - the
        boards' own sitemaps describe what exists today.
        """
        import socket as _socket

        try:
            blob = self.container_client.get_blob_client("directory/companies.jsonl")
            data = blob.download_blob().readall()
            companies = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        except Exception as exc:
            logger.info(f"sitemaps: no directory blob yet ({exc})")
            return

        domains: list[str] = []
        seen_domains: set[str] = set()
        for c in companies:
            for variant in _slug_variants(c.get("name", "")):
                for tld in (".com", ".io", ".co"):
                    domain = f"{variant}{tld}"
                    if domain not in seen_domains:
                        seen_domains.add(domain)
                        domains.append(domain)
        logger.info(f"sitemaps: {len(companies)} companies, {len(domains)} candidate domains")

        resolved: set[str] = set()
        sem_dns = asyncio.Semaphore(128)
        dns_done = 0

        async def check_dns(domain: str) -> None:
            nonlocal dns_done
            async with sem_dns:
                try:
                    await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None, _socket.getaddrinfo, domain, 443
                        ),
                        timeout=2.0,
                    )
                    resolved.add(domain)
                except Exception:
                    pass
                dns_done += 1
                if dns_done % 3000 == 0:
                    logger.info(
                        f"sitemaps dns: {dns_done}/{len(domains)} (resolved {len(resolved)})"
                    )

        await asyncio.gather(*(check_dns(d) for d in domains))
        logger.info(f"sitemaps: {len(resolved)}/{len(domains)} domains resolved")

        slug_re_cache = {
            p: re.compile(rx, re.IGNORECASE)
            for p, (_, rx) in (*_CDX_DOMAINS.items(), *_CDX_DISCOVERY.items())
        }

        async def crawl_domain(domain: str) -> None:
            robots_url = f"https://{domain}/robots.txt"
            sitemaps: list[str] = []
            try:
                r = await client.get(robots_url, timeout=8.0)
                if r.status_code == 200:
                    for line in r.text.splitlines()[:200]:
                        if line.lower().startswith("sitemap:"):
                            url = line.split(":", 1)[1].strip()
                            if url.startswith("http"):
                                sitemaps.append(url)
                            if len(sitemaps) >= 3:
                                break
            except Exception:
                pass
            for sm_url in sitemaps[:2]:
                try:
                    r = await client.get(sm_url, timeout=15.0)
                    if r.status_code != 200:
                        continue
                    text = r.text[:500_000]
                    for platform, sre in slug_re_cache.items():
                        target = "discovery" if platform in _CDX_DISCOVERY else platform
                        for m in sre.finditer(text):
                            s = m.group(1).strip().lower()
                            if s and "." not in s and s not in self.slugs[target]:
                                self.slugs[target].add(s)
                except Exception:
                    continue

        sem = asyncio.Semaphore(300)

        async def _gated(domain: str) -> None:
            async with sem:
                await crawl_domain(domain)

        await asyncio.gather(*(_gated(d) for d in sorted(resolved)))
        parts = " ".join(
            f"{p}={len(self.slugs.get('discovery' if p in _CDX_DISCOVERY else p, set()))}"
            for p in (*_CDX_DOMAINS, *_CDX_DISCOVERY)
        )
        total = sum(len(v) for v in self.slugs.values())
        logger.info(f"sitemaps done: {parts} (total {total})")

    async def run(self) -> None:
        await self.load_state()
        last_ats = 0.0
        last_cdx = 0.0
        last_dir = 0.0
        last_was = 0.0
        last_hn = 0.0
        last_hn_hist = 0.0
        last_remotive = 0.0
        last_arbeitnow = 0.0
        last_remoteok = 0.0
        last_checkpoint = 0.0

        # Independent checkpoint loop: snapshots progress every 10 minutes
        # even while a long CDX walk holds the main cycle.
        async def _checkpoint_loop() -> None:
            while True:
                await asyncio.sleep(600)
                await self.save_checkpoint()

        asyncio.create_task(_checkpoint_loop())
        while True:
            async with AsyncClient(
                headers={"User-Agent": UA}, timeout=15.0, follow_redirects=True
            ) as client:
                now = time.monotonic()
                if now - last_dir > 3600 or last_dir == 0:
                    await self.resolve_directory_slugs(client)
                    await self.crawl_directory_sitemaps(client)
                    last_dir = time.monotonic()
                if now - last_ats > 1800 or last_ats == 0:
                    await self.harvest_slugs(client)
                    await self.poll_ats(client)
                    last_ats = time.monotonic()
                if now - last_cdx > 900 or last_cdx == 0:
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
                if now - last_remoteok > 1800 or last_remoteok == 0:
                    await self.poll_remoteok(client)
                    await self.poll_himalayas(client)
                    await self.poll_jobicy(client)
                    last_remoteok = time.monotonic()
            await self.flush()
            if time.monotonic() - last_checkpoint > 600 or last_checkpoint == 0:
                await self.save_checkpoint()
                last_checkpoint = time.monotonic()
            await asyncio.sleep(60)


async def main() -> None:
    idx = Indexer()
    try:
        await idx.run()
    finally:
        await idx.flush()


if __name__ == "__main__":
    asyncio.run(main())
