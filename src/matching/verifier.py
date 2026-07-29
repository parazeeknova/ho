import asyncio

import httpx
from firecrawl import FirecrawlApp

from src.llm.context import VERIFY_SCHEMA, ContextManager

FIRECRAWL_URL = "http://127.0.0.1:3002"

VERIFY_PROMPT = """Compare two scraped job listings. Are they the same job?
Original: {original}
Alternate: {alternate}
Return valid JSON matching the required schema."""


async def _scrape_alternate(role: str, company: str, original_url: str) -> str:
    """Async Firecrawl call to verify job existence across alternate sources."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{FIRECRAWL_URL}/v1/search",
                json={"query": f"{role} {company} job"},
            )
            if resp.status_code != 200:
                return ""
            data = resp.json().get("data", []) or []
            if not isinstance(data, list) or not data:
                return ""

            alt_url = None
            for r in data:
                u = r.get("url", "")
                if u and u != original_url and u.startswith("http"):
                    alt_url = u
                    break
            if not alt_url:
                return ""

            scrape_resp = await client.post(
                f"{FIRECRAWL_URL}/v1/scrape",
                json={"url": alt_url, "formats": ["markdown"]},
            )
            if scrape_resp.status_code == 200:
                alt_content = (scrape_resp.json().get("data") or {}).get("markdown", "") or ""
                return alt_content if len(alt_content) >= 100 else ""
    except Exception:
        pass
    return ""


async def _verify_one(
    app: FirecrawlApp,
    role: str,
    company: str,
    original_url: str,
    ctx: ContextManager,
    sem: asyncio.Semaphore,
) -> bool:
    async with sem:
        alt_content = await _scrape_alternate(role, company, original_url)
        if not alt_content:
            return True

        prompt = VERIFY_PROMPT.replace("{original}", f"{role} @ {company} [{original_url}]")
        prompt = prompt.replace("{alternate}", alt_content[:3000])

        result = await ctx.json_chat(prompt, schema=VERIFY_SCHEMA)
        if isinstance(result, dict):
            confidence = result.get("confidence", 50)
            return confidence >= 30
        return True


async def verify_jobs(
    app: FirecrawlApp,
    jobs: list[dict],
    ctx: ContextManager,
    concurrency: int = 4,
) -> list[dict]:
    if not jobs:
        return []

    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for j in jobs:
        role = j.get("role", "?")
        company = j.get("company", "?")
        url = j.get("source_url", "")
        tasks.append(_verify_one(app, role, company, url, ctx, sem))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    verified = []
    for j, result in zip(jobs, results, strict=True):
        if isinstance(result, BaseException) or result is True:
            verified.append(j)
        else:
            print(f"    [red]✗ FAILED: {j.get('role')} @ {j.get('company')}[/red]")

    return verified
