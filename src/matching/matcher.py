"""Job matcher: concurrent RAG-powered semantic matching with JSON schema enforcement."""

import asyncio

from src.llm.context import MATCH_SCHEMA

MATCH_PROMPT = (
    "You are a job-resume matching engine. Compare this job description "
    "against the RELEVANT RESUME SNIPPETS below (note: these are only the "
    "most relevant parts of the resume, not the full resume).\n\n"
    "Relevant resume snippets:\n{relevant_chunks}\n\n"
    "Full job listing:\n{job_description}\n\n"
    "CRITICAL RULE: If the full job listing below is a company homepage, "
    "a job search directory, an error page, or a list of multiple different "
    "jobs rather than ONE SINGLE specific job posting, you MUST set "
    "match_percent to 0 and verdict to NO_MATCH.\n"
    "Be conservative with scores. STRONG_MATCH only if genuine skill alignment. "
    "Return valid JSON matching the required schema."
)


async def _match_one(
    job: dict,
    rag,
    ctx,
    sem: asyncio.Semaphore,
) -> dict | None:
    loop = asyncio.get_running_loop()
    jd_text = job.get("markdown", job.get("snippet", ""))
    if len(jd_text) < 40:
        return None

    async with sem:
        retrieved = await loop.run_in_executor(None, rag.retrieve, jd_text, 8)
        relevant = "\n".join(
            f"[{chunk_id}] {text}" for chunk_id, text, score in retrieved if score > 0.25
        )
        if not relevant:
            relevant = jd_text[:500]

        prompt = MATCH_PROMPT.replace("{relevant_chunks}", relevant[:3000])
        prompt = prompt.replace("{job_description}", jd_text[:5000])

        result = await ctx.json_chat(prompt, MATCH_SCHEMA)

        if not isinstance(result, dict) or "match_percent" not in result:
            return None

        required = {"role", "company", "match_percent", "shortlist_probability"}
        if not required.issubset(result.keys()):
            return None

        role_lower = str(result.get("role", "")).lower()
        _echo_kws = (
            "matching engine",
            "job search",
            "analysis of",
            "resume matcher",
            "scoring engine",
            "job matcher",
            "match scorer",
            "ranking system",
        )
        if any(kw in role_lower for kw in _echo_kws):
            fallback_role = (job.get("title") or "").strip()
            if fallback_role:
                result["role"] = fallback_role
            else:
                return None

        result["match_percent"] = int(result["match_percent"])
        result["shortlist_probability"] = int(result["shortlist_probability"])
        return result


async def batch_match(
    jobs: list[dict],
    rag,
    ctx,
    concurrency: int = 4,
) -> list[dict]:
    if not jobs:
        return []

    sem = asyncio.Semaphore(concurrency)
    tasks = []

    for job in jobs:
        title = job.get("title", job.get("url", ""))[:60]
        print(f"  [match] {title}")
        tasks.append(_match_one(job, rag, ctx, sem))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    scored: list[dict] = []
    for job, result in zip(jobs, results, strict=True):
        if isinstance(result, BaseException):
            print(f"    match failed: {result}")
            continue
        if result is not None and result["match_percent"] >= 40:
            result["source_url"] = job.get("url", job.get("source_url", ""))
            scored.append(result)
            pct = result.get("match_percent", "?")
            verdict = result.get("verdict", "?")
            role = result.get("role", "?")
            company = result.get("company", "?")
            print(f"    {pct}% | {verdict} | {role} @ {company}")

    scored.sort(key=lambda j: j["match_percent"], reverse=True)
    return scored
