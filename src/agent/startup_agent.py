"""StartupAgent: OSINT-grade company intelligence via SearXNG dorks + LLM extraction.

Extracts founder details (name, title, LinkedIn, GitHub, email), funding rounds
(amount, lead investors, date), and technical signals.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from rich.console import Console

from src.agent.email_triangulation import triangulate_founder_email
from src.configuration import get_config
from src.llm.context import ContextManager
from src.logging import get_logger

console = Console()
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
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
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

    def __init__(self, ctx: ContextManager) -> None:
        self.ctx = ctx

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

    async def analyze_startup(self, job: dict[str, Any]) -> dict[str, Any]:
        """Research company founders, funding stage, socials, and recent news."""
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

        if not combined_snippets:
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
            job["founders"] = [{"name": n} for n in raw_founders]
            if extracted.get("founder_socials"):
                job["founder_socials"] = extracted["founder_socials"]
        else:
            if extracted.get("founder_socials"):
                job["founder_socials"] = extracted["founder_socials"]

        # Email triangulation fallback for founders missing direct email
        domain = company.lower().replace(" ", "").strip() + ".com"
        for f in job.get("founders", []):
            if isinstance(f, dict) and f.get("name") and not f.get("email"):
                tri = await triangulate_founder_email(f["name"], domain)
                if tri:
                    f["email"] = tri["email"]
                    f["email_triangulated"] = True

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
            job["osint_signals"] = [str(s) for s in signals[:2]]

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
