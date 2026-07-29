"""StartupAgent: OSINT-grade company intelligence via SearXNG dorks + LLM extraction.

Extracts founder details (name, title, LinkedIn, GitHub, email), funding rounds
(amount, lead investors, date), and technical signals.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from src.llm.context import ContextManager

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


async def _searxng_search(query: str) -> list[str]:
    """Execute search query against local SearXNG."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "http://localhost:8080/search",
                params={"q": query, "format": "json", "time_range": "day"},
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                return [
                    f"{r.get('title', '')}: {r.get('content', '')} ({r.get('url', '')})"
                    for r in results[:5]
                    if r.get("content") or r.get("title")
                ]
    except Exception as e:
        print(f"  [dim]SearXNG query '{query[:60]}': {e}[/dim]")
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

    def __init__(self, ctx: ContextManager) -> None:
        self.ctx = ctx

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
            f"{combined_snippets[:8000]}\n\n"
            "Extract the following structured intelligence:\n\n"
            "1. is_startup: boolean (true if startup/venture-backed, false if enterprise).\n\n"
            "2. founders: array of founder objects with these keys:\n"
            "   - name (string): full name.\n"
            "   - title (string): e.g. 'CEO', 'CTO', 'Co-founder'.\n"
            "   - linkedin_url (string|null): full https://www.linkedin.com/in/... URL.\n"
            "   - github_url (string|null): full https://github.com/... URL.\n"
            "   - email (string|null): if not found, aggressively guess using format\n"
            "     first@company.com or first.last@company.com based on founder name.\n\n"
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
            "- Emails may be guessed from name+domain pattern, mark as 'guessed' if so.\n"
            "- Return valid JSON matching the exact schema."
        )

        extracted = await self.ctx.json_chat(prompt, schema=FOUNDER_SCHEMA)
        if not isinstance(extracted, dict):
            return job

        # --- Top-level fields ---
        job["is_startup"] = bool(extracted.get("is_startup", False))

        if extracted.get("funding_stage"):
            job["funding_stage"] = str(extracted["funding_stage"])
        if extracted.get("company_news"):
            job["company_news"] = str(extracted["company_news"])

        # --- Founders: support both old (list[str]) and new (list[dict]) ---
        raw_founders = extracted.get("founders", [])
        if raw_founders and isinstance(raw_founders[0], dict):
            # New nested schema — store full objects
            job["founders"] = raw_founders
            # Legacy string list for backward compat
            job["founder_socials"] = [
                f.get("linkedin_url")
                for f in raw_founders
                if isinstance(f, dict) and f.get("linkedin_url")
            ]
        elif raw_founders and isinstance(raw_founders[0], str):
            # Legacy flat list
            job["founders"] = [{"name": n} for n in raw_founders]
            if extracted.get("founder_socials"):
                job["founder_socials"] = extracted["founder_socials"]
        else:
            if extracted.get("founder_socials"):
                job["founder_socials"] = extracted["founder_socials"]

        # --- Funding info (new nested object) ---
        fi = extracted.get("funding_info")
        if isinstance(fi, dict) and any(fi.values()):
            job["funding_info"] = fi
            # Legacy string fallback from nested
            if not job.get("funding_stage") and fi.get("round"):
                parts = [fi["round"]]
                if fi.get("amount_raised"):
                    parts.append(f"({fi['amount_raised']})")
                if fi.get("lead_investors"):
                    parts.append("led by " + ", ".join(fi["lead_investors"]))
                job["funding_stage"] = " ".join(parts)

        # --- OSINT signals ---
        signals = extracted.get("osint_signals")
        if isinstance(signals, list):
            job["osint_signals"] = [str(s) for s in signals[:2]]

        # --- Startup scoring boost ---
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

        snippets = await _searxng_search(query)
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
            print(f"  [dim]Founder posts LLM failed for {company}: {e}[/dim]")
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
        """Parallel analysis of startup intelligence for candidate jobs."""
        if not jobs:
            return []

        sem = asyncio.Semaphore(concurrency)

        async def _worker(j: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                try:
                    return await self.analyze_startup(j)
                except Exception:
                    return j

        return await asyncio.gather(*(_worker(j) for j in jobs))
