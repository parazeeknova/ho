"""Job verifier: cross-check listings via alternate source with JSON schema enforcement."""

import asyncio

from firecrawl import FirecrawlApp

from src.llm.context import VERIFY_SCHEMA, ContextManager

VERIFY_PROMPT = """Compare two scraped job listings. Are they the same job?
Original: {original}
Alternate: {alternate}
Return valid JSON matching the required schema."""


async def _verify_one(
    app: FirecrawlApp,
    role: str,
    company: str,
    original_url: str,
    ctx: ContextManager,
    sem: asyncio.Semaphore,
) -> bool:
    async with sem:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _verify_sync, app, role, company, original_url, ctx)


def _verify_sync(
    app: FirecrawlApp,
    role: str,
    company: str,
    original_url: str,
    ctx: ContextManager,
) -> bool:
    try:
        alt_results = app.search(f"{role} {company} job")
        data = getattr(alt_results, "web", []) or []
        if not isinstance(data, list) or not data:
            return True

        alt_url = None
        for r in data:
            u = getattr(r, "url", "")
            if u and u != original_url and u.startswith("http"):
                alt_url = u
                break

        if not alt_url:
            return True

        alt_result = app.scrape_url(alt_url, formats=["markdown"])
        alt_content = getattr(alt_result, "markdown", "") or ""
        if len(alt_content) < 100:
            return True

        prompt = VERIFY_PROMPT.replace("{original}", f"{role} @ {company} [{original_url}]")
        prompt = prompt.replace("{alternate}", alt_content[:3000])

        result = ctx.json_chat(prompt, schema=VERIFY_SCHEMA)
        if isinstance(result, dict):
            confidence = result.get("confidence", 50)
            return confidence >= 30

        return True
    except Exception:
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
