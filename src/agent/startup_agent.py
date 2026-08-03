"""StartupAgent: OSINT-grade company intelligence via SearXNG dorks + LLM extraction.

Extracts founder details (name, title, LinkedIn, GitHub, email), funding rounds
(amount, lead investors, date), and technical signals.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import Any

from rich.console import Console

from src.agent.email_triangulation import triangulate_founder_email
from src.configuration import get_config
from src.http_client import get_client
from src.llm.context import ContextManager
from src.logging import get_logger

console = Console()

_PLACEHOLDER_COMPANY_RX = re.compile(
    r"not.?specified|unknown|n[./]?a\.?|tbd|placeholder|no.?company|"
    r"^company$|^job listing$|^-+$|well.?known",
    re.I,
)

# Discovery adapters sometimes register article headlines / newsletter titles
# as "companies" (HN titles, news blurbs). OSINT on those names is pure noise:
# the search returns unrelated top stories and the LLM attributes them to the
# placeholder. A real company name is short, title-cased and noun-led.
_HEADLINE_START_RX = re.compile(
    r"^(what|how|why|when|where|who|after|before|he|she|they|we|this|that|"
    r"the|a|an|in|on|at|is|are|was|were|leaves|leaving|quits|quitting|"
    r"raises|raised|builds|building|launches|launched|sells|selling|"
    r"moves|moving|breaks|breaks|turns|turning|from|inside|meet|why)\b",
    re.I,
)


def _is_plausible_company_name(company: str) -> bool:
    """True only for names that plausibly identify a real company.

    Rejects placeholders and sentence-like discovery artifacts (long
    headline strings, verb-led phrases) so OSINT never runs against a
    name that cannot resolve to a real business.
    """
    name = (company or "").strip()
    if not name or _PLACEHOLDER_COMPANY_RX.search(name):
        return False
    if len(name) > 45:
        return False
    words = name.split()
    if len(words) >= 6:
        return False
    return not bool(_HEADLINE_START_RX.match(name))


def _company_domain(company: str) -> str:
    """Best-effort domain for a company, or '' when the name is a placeholder.

    Prevents fabricating emails like ``careers@notspecifiedinjoblisting.com``
    when the OSINT layer could not identify a real company.
    """
    name = (company or "").strip()
    if not _is_plausible_company_name(name):
        return ""
    return name.lower().replace(" ", "").strip() + ".com"


# Cross-company signal pollution guard: SearXNG returns the same generic top
# stories for many queries, so the LLM can attribute the same signal (e.g.
# "Open-sourced Kimi-K3 on GitHub") to unrelated companies. A normalized
# signal already seen for a different company is dropped.
_SIGNAL_LOG: list[tuple[str, frozenset[str]]] = []


def _signal_tokens(sig: str) -> frozenset[str]:
    toks = {t for t in re.sub(r"[^a-z0-9 ]", " ", sig.lower()).split() if len(t) > 2}
    return frozenset(toks)


def _signal_is_pollution(sig: str, company_key: str) -> bool:
    """True when the same signal text was already recorded for another company."""
    toks = _signal_tokens(sig)
    if not toks:
        return False
    for seen_company, seen_toks in _SIGNAL_LOG:
        if seen_company == company_key:
            continue
        overlap = len(toks & seen_toks) / min(len(toks), len(seen_toks))
        if overlap >= 0.6:
            return True
    _SIGNAL_LOG.append((company_key, toks))
    if len(_SIGNAL_LOG) > 500:
        _SIGNAL_LOG[:200] = []
    return False


logger = get_logger("startup_agent")

FOUNDER_POST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "founder_posts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "post_url": {"type": "string"},
                    "founder_name": {"type": "string"},
                    "intent": {"type": "string"},
                },
                "required": ["post_url", "founder_name", "intent"],
            },
        },
    },
    "required": ["founder_posts"],
}


_searxng_sem = asyncio.Semaphore(get_config().searxng.semaphore)


async def _searxng_search(query: str, time_range: str | None = None) -> list[str]:
    """Execute search query against local SearXNG."""
    params: dict[str, str] = {
        "q": query,
        "format": "json",
        "engines": "bing,bing news,github",
    }
    if time_range:
        params["time_range"] = time_range
    cfg = get_config().searxng
    async with _searxng_sem:
        try:
            client = await get_client("startup_agent", timeout=cfg.timeout)
            resp = await client.get(
                cfg.url,
                params=params,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                return [
                    f"{r.get('title', '')}: {r.get('content', '')} ({r.get('url', '')})"
                    for r in results[:5]
                    if r.get("content") or r.get("title")
                ]
        except Exception as e:
            logger.debug(
                "SearXNG query failed",
                source="searxng",
                exception=str(e),
                extra={"query": query[:60]},
            )
    return []


FOUNDER_SCHEMA = {
    "type": "object",
    "properties": {
        "is_startup": {"type": "boolean"},
        "founders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "linkedin_url": {"type": ["string", "null"]},
                    "github_url": {"type": ["string", "null"]},
                    "email": {"type": ["string", "null"]},
                },
                "required": ["name"],
            },
        },
        "funding_info": {
            "type": "object",
            "properties": {
                "round": {"type": ["string", "null"]},
                "amount_raised": {"type": ["string", "null"]},
                "lead_investors": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "date_announced": {"type": ["string", "null"]},
            },
            "required": [],
        },
        "osint_signals": {
            "type": "array",
            "items": {"type": "string"},
        },
        "funding_stage": {"type": ["string", "null"]},
        "founder_socials": {
            "type": "array",
            "items": {"type": "string"},
        },
        "company_news": {"type": ["string", "null"]},
    },
    "required": ["is_startup", "founders"],
}


_LINKEDIN_RE = re.compile(r"https?://(?:[\w-]+\.)*linkedin\.com/in/[A-Za-z0-9_\-%]+")


class StartupAgent:
    """Agent that researches startup founders, funding, and outreach info."""

    _ENTERPRISE_DOMAINS = frozenset(
        {
            "google",
            "microsoft",
            "amazon",
            "apple",
            "meta",
            "netflix",
            "ibm",
            "oracle",
            "salesforce",
            "adobe",
            "cisco",
            "intel",
            "nvidia",
            "amd",
            "sap",
            "servicenow",
            "atlassian",
            "uber",
            "spotify",
            "stripe",
            "twitter",
            "reddit",
            "roblox",
            "snap",
            "lyft",
            "instacart",
            "doordash",
            "palantir",
            "cloudflare",
        }
    )
    _FUNDING_KW_RE = (
        r"\b(?:seed|pre-?seed|series\s+[a-c]|vc-?backed|"
        r"y\s*combinator|techstars|accelerator|incubator|"
        r"raised\s+\$?\d)"
    )

    def __init__(self, ctx: ContextManager, store: Any = None) -> None:
        self.ctx = ctx
        self.store = store

    @staticmethod
    def _should_skip_llm(job: dict[str, Any]) -> str | None:
        """Return a skip reason if LLM analysis is a waste, else None."""
        company = str(job.get("company", "")).lower().strip()
        if not company or company in ("n/a", "unknown"):
            return "no company"
        if company in StartupAgent._ENTERPRISE_DOMAINS:
            return "enterprise"
        return None

    # noqa: E501
    @staticmethod
    def _priority_score(job: dict[str, Any]) -> int:
        """Score 0-100 for how 'LLM-worthy' this company is."""
        score = 0
        company = str(job.get("company", "")).lower()
        role = str(job.get("role", "")).lower()
        desc = str(job.get("company_description", "")).lower()
        jd = str(job.get("jd_summary", "")).lower()
        verdict = str(job.get("verdict", "")).upper()
        combined = f"{company} {role} {desc} {jd}"

        match = int(job.get("match_percent", 0))
        score += min(40, match // 2)

        if verdict in ("STRONG_MATCH", "GOOD_MATCH"):
            score += 20
        elif verdict == "WEAK_MATCH":
            score += 10

        for kw in (
            "startup",
            "early-stage",
            "seed",
            "series a",
            "series b",
            "y combinator",
            "yc-backed",
            "accelerator",
            "pre-seed",
            "stealth",
            "founded",
            "backed by",
        ):
            if kw in combined:
                score += 5

        if re.search(StartupAgent._FUNDING_KW_RE, combined):
            score += 10

        if job.get("is_startup"):
            score += 10

        founders = job.get("founders", [])
        if founders and isinstance(founders, list):
            if isinstance(founders[0], dict):
                score += 15
            else:
                score += 8

        if job.get("funding_stage") and job["funding_stage"] != "N/A":
            score += 10

        if job.get("founder_posts"):
            score += 15

        source = str(job.get("source", "")).lower()
        if source in ("yc", "discovered", "searxng"):
            score += 10
        if source == "linkedin_guest":
            score += 5

        link = str(job.get("apply_link", ""))
        if any(
            pat in link
            for pat in (
                "greenhouse",
                "lever.co",
                "ashbyhq",
                "workable",
                "myworkdayjobs",
                "smartrecruiters",
                "rippling",
            )
        ):
            score += 8

        return min(100, score)

    _OSINT_CACHE_KEYS = (
        "is_startup",
        "founders",
        "funding_info",
        "funding_stage",
        "founder_socials",
        "company_news",
        "osint_signals",
    )

    async def analyze_startup(self, job: dict[str, Any]) -> dict[str, Any]:
        """Research company founders, funding stage, socials, and recent news.

        Wrapper with a per-company OSINT cache: the same company is
        analyzed again whenever another of its roles gets accepted, so
        this avoids repeated SearXNG/Wikipedia/LLM work and source
        hammering.
        """
        company = str(job.get("company") or "").strip()
        if not _is_plausible_company_name(company):
            return job
        if self.store is not None:
            cached = await self._get_cached_osint(company)
            if cached:
                job.update(cached)
                return job
        result = await self._analyze_startup_uncached(job)
        if self.store is not None:
            await self._put_cached_osint(company, result)
        return result

    async def _get_cached_osint(self, company: str) -> dict[str, Any] | None:
        if self.store is None:
            return None
        try:
            data = await self.store.get_company_osint(company)
        except Exception:
            return None
        if not data:
            return None
        return {k: data.get(k) for k in self._OSINT_CACHE_KEYS if k in data}

    async def _put_cached_osint(self, company: str, job: dict[str, Any]) -> None:
        if self.store is None:
            return
        payload = {k: job.get(k) for k in self._OSINT_CACHE_KEYS if job.get(k) is not None}
        if not payload:
            return
        # A result with no founders, funding or news is a degraded run
        # (e.g. rate-limited sources); cache it briefly so the next sweep
        # retries rather than serving emptiness for the full week.
        substance = any(
            payload.get(k)
            for k in (
                "founders",
                "funding_info",
                "funding_stage",
                "founder_socials",
                "company_news",
                "osint_signals",
            )
        )
        with contextlib.suppress(Exception):
            await self.store.put_company_osint(company, payload, degraded=not substance)
        if substance and (payload.get("funding_info") or payload.get("funding_stage")):
            await self._record_funding_evidence(company, payload)

    async def _record_funding_evidence(self, company: str, payload: dict[str, Any]) -> None:
        """A funding round is a strong hiring precursor: log it as evidence."""
        try:
            from src.graph.entity import make_company_id

            fi = payload.get("funding_info") or {}
            if not isinstance(fi, dict):
                fi = {}
            await self.store.record_evidence(
                make_company_id(company),
                claim="funding_round",
                source="startup_osint",
                company_name=company,
                evidence_type="funding",
                weight=0.25,
                ref_url=str(fi.get("url") or ""),
            )
        except Exception:
            pass

    async def _finalize_founders(self, job: dict[str, Any], company: str, domain: str) -> None:
        """Email triangulation + real LinkedIn profile search for founders.

        Both run only when search sources genuinely return data; nothing
        is fabricated (a guessed URL is worse than a missing one).

        The cost-aware ``deep_enrich`` gate (set by the orchestrator for
        strong matches or multi-role companies) skips the expensive email
        triangulation and LinkedIn resolution for low-value candidates.
        """
        founders = job.get("founders") or []
        if not founders or not isinstance(founders[0], dict):
            return
        deep = bool(job.get("deep_enrich", True))
        resolved_linkedin = 0
        for f in founders:
            if not isinstance(f, dict) or not f.get("name"):
                continue
            if deep and not f.get("email"):
                tri = await triangulate_founder_email(f["name"], domain)
                if tri:
                    f["email"] = tri["email"]
                    f["email_triangulated"] = True
            if deep and not f.get("linkedin_url") and resolved_linkedin < 3:
                url = await self._resolve_linkedin(str(f["name"]), company)
                if url:
                    f["linkedin_url"] = url
                    resolved_linkedin += 1
                    socials = job.get("founder_socials") or []
                    if url not in socials:
                        socials.append(url)
                        job["founder_socials"] = socials

    async def _resolve_linkedin(self, name: str, company: str) -> str | None:
        """Find a founder's real LinkedIn profile URL via search.

        Wikipedia only yields names; SearXNG/Bing rarely surfaces
        linkedin.com pages in the generic company query, so search for
        the person explicitly. Never fabricates a URL.
        """
        for q in (f'"{name}" "{company}" linkedin', f'"{name}" linkedin'):
            snippets = await _searxng_search(q)
            for snippet in snippets:
                for match in _LINKEDIN_RE.finditer(snippet):
                    url = match.group(0)
                    if url.endswith((".", ",", ")")):
                        url = url[:-1]
                    if url:
                        return url
        return None

    async def _analyze_startup_uncached(self, job: dict[str, Any]) -> dict[str, Any]:
        """The uncached search + extraction pipeline."""
        company = str(job.get("company") or "").strip()
        if not company or company in ("N/A", "Unknown", "Company"):
            return job

        queries = [
            f'"{company}" founder OR CEO site:linkedin.com/in/ OR site:github.com',
            f'"{company}" email "@{company.lower().replace(" ", "")}.com" contact founder',
            f'"{company}" "Seed" OR "Pre-seed" OR "Series A" funding raised investors "TechCrunch" OR "Crunchbase"',  # noqa: E501
        ]
        results_list = await asyncio.gather(*(_searxng_search(q) for q in queries))
        combined_snippets = "\n".join(snippet for sublist in results_list for snippet in sublist)

        # Founder names straight from the Wikipedia infobox; SearXNG/Bing
        # rarely surfaces founder pages, so this is the primary founder source.
        from src.search.wikipedia import get_wikipedia_founders

        wiki_founders = await get_wikipedia_founders(company)

        if not combined_snippets:
            if wiki_founders:
                job["founders"] = wiki_founders
            if job.get("founders"):
                domain = _company_domain(company)
                if domain:
                    await self._finalize_founders(job, company, domain)
            return job

        prompt = (
            f"Analyze web OSINT data for company '{company}':\n\n"
            f"{combined_snippets[:40000]}\n\n"
            "Extract the following structured intelligence:\n\n"
            "1. is_startup: boolean (true if startup/venture-backed, false if enterprise).\n\n"
            "2. founders: array of founder objects with these keys:\n"
            "   - name (string): full name.\n"
            "   - title (string): e.g. 'CEO', 'CTO', 'Co-founder'.\n"
            "   - linkedin_url (string|null): full https://www.linkedin.com/in/... URL.\n"
            "   - github_url (string|null): full https://github.com/... URL.\n"
            "   - email (string|null): ONLY if EXPLICITLY LISTED in the source material.\n"
            "     Do NOT guess or invent email addresses. Return null if no publicly\n"
            "     listed email was found.\n\n"
            "3. funding_info: object with keys:\n"
            "   - round (string|null): 'Pre-Seed', 'Seed', 'Series A', 'Series B', etc.\n"
            "   - amount_raised (string|null): e.g. '$3.5M', '$25M'.\n"
            "   - lead_investors (array of strings): VC names.\n"
            "   - date_announced (string|null): e.g. '2024-03'.\n\n"
            "4. osint_signals: array of 1-2 strings with recent tech blog posts,\n"
            "   active GitHub orgs, product launches, or notable milestones.\n"
            "   Example: 'Open-sourced core SDK on GitHub (2025)', "
            "'Raised Series A led by a16z (Mar 2025)'.\n\n"
            "5. funding_stage (string|null): legacy field, same as funding_info.round.\n"
            "6. founder_socials (array of strings): legacy field with LinkedIn/X URLs.\n"
            "7. company_news (string|null): one-sentence recent news summary.\n\n"
            "CRITICAL RULES:\n"
            f"- Everything you extract MUST be specifically about '{company}'.\n"
            "  If a search result concerns a DIFFERENT company (another firm's\n"
            "  product, funding, founder or news), discard it entirely - do not\n"
            "  fold it into any field.\n"
            "- All URLs MUST be valid https:// links. Return null if no valid URL found.\n"
            "- Missing fields must be explicit null, never invented.\n"
            "- NEVER guess or fabricate email addresses. Return null for email unless\n"
            "  the email was explicitly present in the search results.\n"
            "- Return valid JSON matching the exact schema."
        )

        extracted = await self.ctx.json_chat(prompt, schema=FOUNDER_SCHEMA)
        if not isinstance(extracted, dict):
            return job

        job["is_startup"] = bool(extracted.get("is_startup", False))

        if extracted.get("funding_stage"):
            job["funding_stage"] = str(extracted["funding_stage"])
        if extracted.get("company_news"):
            job["company_news"] = str(extracted["company_news"])

        raw_founders = extracted.get("founders", [])
        if raw_founders and isinstance(raw_founders[0], dict):
            job["founders"] = raw_founders
            job["founder_socials"] = [
                f.get("linkedin_url")
                for f in raw_founders
                if isinstance(f, dict) and f.get("linkedin_url")
            ]
        elif raw_founders and isinstance(raw_founders[0], str):
            if isinstance(raw_founders, str):
                try:
                    parsed = json.loads(raw_founders)
                    if isinstance(parsed, list):
                        job["founders"] = [{"name": n} for n in parsed if isinstance(n, str)]
                    else:
                        job["founders"] = [{"name": raw_founders}]
                except Exception:
                    job["founders"] = [{"name": raw_founders}]
            else:
                job["founders"] = [{"name": n} for n in raw_founders]
            if extracted.get("founder_socials"):
                job["founder_socials"] = extracted["founder_socials"]
        else:
            if extracted.get("founder_socials"):
                job["founder_socials"] = extracted["founder_socials"]

        if not job.get("founders") and wiki_founders:
            job["founders"] = wiki_founders

        # Email triangulation fallback + LinkedIn profile resolution for
        # founders missing direct contact data. Only when the company name
        # resolves to a plausible domain (never placeholder names).
        domain = _company_domain(company)
        if domain:
            await self._finalize_founders(job, company, domain)

        fi = extracted.get("funding_info")
        if isinstance(fi, dict) and any(fi.values()):
            job["funding_info"] = fi
            if not job.get("funding_stage") and fi.get("round"):
                parts = [fi["round"]]
                if fi.get("amount_raised"):
                    parts.append(f"({fi['amount_raised']})")
                if fi.get("lead_investors"):
                    parts.append("led by " + ", ".join(fi["lead_investors"]))
                job["funding_stage"] = " ".join(parts)

        signals = extracted.get("osint_signals")
        if isinstance(signals, list):
            kept = []
            for s in signals[:2]:
                sig = str(s)
                if _signal_is_pollution(sig, company.lower()):
                    logger.info(
                        f"OSINT signal dropped as cross-company pollution for {company}",
                        signal=sig[:80],
                    )
                    continue
                kept.append(sig)
            job["osint_signals"] = kept

        if job["is_startup"]:
            job["match_percent"] = min(99, job.get("match_percent", 0) + 10)
            job["shortlist_probability"] = min(95, job.get("shortlist_probability", 0) + 10)

        return job

    async def mine_founder_posts(
        self, company: str, roles: list[str] | None = None
    ) -> list[dict[str, str]]:
        """Search for recent LinkedIn posts where the company's founder/CEO/CTO
        is actively saying 'I am hiring' or 'DM me' or 'looking for'.

        Returns a list of dicts with keys: post_url, founder_name, intent.
        """
        if not company:
            return []

        role_part = ""
        if roles:
            role_part = " AND (" + " OR ".join(f'"{r}"' for r in roles[:3]) + ")"

        query = (
            f"site:linkedin.com/posts/ OR site:linkedin.com/feed/update/ "
            f'"{company}" AND ("hiring" OR "looking for" OR "DM me") '
            f'AND ("founder" OR "CEO" OR "CTO")'
            f"{role_part}"
        )

        snippets = await _searxng_search(query, time_range="day")
        if not snippets:
            return []

        prompt = (
            f"Search results for founder hiring posts at '{company}':\n\n"
            f"{chr(10).join(snippets[:3])}\n\n"
            "Extract ONLY LinkedIn posts where a founder/CEO/CTO says "
            "they are actively hiring. Return:\n"
            "- post_url: the LinkedIn post URL (must be a valid https:// linkedin.com URL)\n"
            "- founder_name: the person's name who posted\n"
            "- intent: 1-sentence summary of what role they are hiring for\n\n"
            "Return valid JSON matching the schema. Empty array if no real hiring posts found."
        )

        try:
            result = await self.ctx.json_chat(prompt, schema=FOUNDER_POST_SCHEMA)
        except Exception as e:
            logger.warning("Founder posts LLM failed", entity=company, exception=str(e))
            return []

        if not isinstance(result, dict):
            return []

        posts = result.get("founder_posts", [])
        if not isinstance(posts, list):
            return []

        valid = []
        for p in posts[:3]:
            if isinstance(p, dict) and p.get("post_url", "").startswith("http"):
                valid.append(
                    {
                        "post_url": str(p["post_url"]),
                        "founder_name": str(p.get("founder_name", "Unknown")),
                        "intent": str(p.get("intent", "")),
                    }
                )
        return valid

    async def batch_analyze_startups(
        self, jobs: list[dict[str, Any]], concurrency: int = 8
    ) -> list[dict[str, Any]]:
        """Priority-scheduled startup analysis.

        Deterministic checks run first (zero LLM cost). Only companies
        that score above the threshold get LLM analysis. The global
        token bucket is conserved for high-signal opportunities.
        """
        if not jobs:
            return []

        scored: list[tuple[int, int, dict[str, Any]]] = []
        pass_through: list[dict[str, Any]] = []

        for idx, j in enumerate(jobs):
            skip = self._should_skip_llm(j)
            if skip:
                pass_through.append(j)
                continue

            score = self._priority_score(j)
            if score < 30:
                pass_through.append(j)
                continue

            scored.append((score, idx, j))

        if not scored:
            return jobs

        scored.sort(key=lambda x: x[0], reverse=True)

        logger.info(f"StartupAgent: {len(scored)}/{len(jobs)} selected for OSINT")

        sem = asyncio.Semaphore(concurrency)
        result_map: dict[int, dict[str, Any]] = {}

        async def _worker(idx: int, j: dict[str, Any]) -> None:
            async with sem:
                try:
                    result_map[idx] = await self.analyze_startup(j)
                except Exception:
                    result_map[idx] = j

        tasks = [_worker(idx, j) for _, idx, j in scored]
        await asyncio.gather(*tasks)

        output = jobs[:]
        for _, idx, _ in scored:
            if idx in result_map:
                output[idx] = result_map[idx]

        return output
