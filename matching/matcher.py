"""Job matcher: RAG-powered semantic matching of JD against resume."""

import time

MATCH_PROMPT = (
    "You are a job-resume matching engine. Compare this job description "
    "against the RELEVANT RESUME SNIPPETS below (note: these are only the "
    "most relevant parts of the resume, not the full resume).\n\n"
    "Relevant resume snippets:\n{relevant_chunks}\n\n"
    "Full job listing:\n{job_description}\n\n"
    "Output ONLY valid JSON:\n"
    '{{\n  "role": "job title",\n  "company": "company name",\n'
    '  "match_percent": 0-100,\n  "shortlist_probability": 0-100,\n'
    '  "matching_skills": ["skill"],\n  "missing_skills": ["skill"],\n'
    '  "jd_summary": "one line",\n  "salary": "salary or null",\n'
    '  "posted_date": "ISO date or null",\n'
    '  "apply_link": "url or null",\n'
    '  "is_undergrad_friendly": true/false,\n'
    '  "is_remote": true/false,\n  "location": "string",\n'
    '  "verdict": "STRONG_MATCH"/"GOOD_MATCH"/"WEAK_MATCH"/"NO_MATCH"\n}}\n\n'
    "Be conservative with scores. STRONG_MATCH only if genuine skill alignment. "
    "Return ONLY JSON. No markdown."
)


def match_job(job_text: str, relevant_chunks: str, ctx) -> dict | None:
    ctx.maybe_flush()

    prompt = MATCH_PROMPT.replace("{relevant_chunks}", relevant_chunks[:3000])
    prompt = prompt.replace("{job_description}", job_text[:5000])

    result = ctx.json_chat(prompt)
    if not isinstance(result, dict) or "match_percent" not in result:
        return None

    required = {"role", "company", "match_percent", "shortlist_probability"}
    if not required.issubset(result.keys()):
        return None

    result["match_percent"] = int(result["match_percent"])
    result["shortlist_probability"] = int(result["shortlist_probability"])

    return result


def batch_match(jobs: list[dict], rag, ctx) -> list[dict]:
    scored = []
    for i, job in enumerate(jobs):
        jd_text = job.get("markdown", job.get("snippet", ""))
        title = job.get("title", job.get("url", ""))[:60]
        if len(jd_text) < 100:
            continue

        # RAG: retrieve most relevant resume chunks for this JD
        retrieved = rag.retrieve(jd_text, top_k=8)
        relevant = "\n".join(
            f"[{chunk_id}] {text}" for chunk_id, text, score in retrieved if score > 0.01
        )
        if not relevant:
            relevant = jd_text[:500]  # fallback

        print(f"  [{i + 1}/{len(jobs)}] Matching: {title}")
        try:
            result = match_job(jd_text, relevant, ctx)
            if result and result["match_percent"] >= 40:
                result["source_url"] = job.get("url", job.get("source_url", ""))
                scored.append(result)
                pct = result["match_percent"]
                verdict = result["verdict"]
                role = result.get("role", "?")
                company = result.get("company", "?")
                print(f"    {pct}% | {verdict} | {role} @ {company}")
        except Exception as e:
            print(f"    failed: {e}")
            continue
        time.sleep(0.2)

    scored.sort(key=lambda j: j["match_percent"], reverse=True)
    return scored
