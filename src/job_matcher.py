"""Job matcher: LLM-powered semantic matching of JD against resume."""

import time

MATCH_PROMPT = (
    "You are a job-resume matching engine. Compare this job description "
    "against the candidate's resume.\n\n"
    "Resume sections:\n{resume_summary}\n\n"
    "Job listing:\n{job_description}\n\n"
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


def match_job(job_text: str, resume_summary: str, ctx) -> dict | None:
    ctx.maybe_flush()

    prompt = MATCH_PROMPT.replace("{resume_summary}", resume_summary[:3000])
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


def batch_match(jobs: list[dict], resume_summary: str, ctx) -> list[dict]:
    scored = []
    for i, job in enumerate(jobs):
        jd_text = job.get("markdown", job.get("snippet", ""))
        title = job.get("title", job.get("url", ""))[:60]
        if len(jd_text) < 100:
            continue

        print(f"  [{i + 1}/{len(jobs)}] Matching: {title}")
        try:
            result = match_job(jd_text, resume_summary, ctx)
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
