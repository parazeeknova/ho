import json
import sys
import time
import urllib.request

LLM_URL = "http://127.0.0.1:8899"
FC_URL = "http://127.0.0.1:3002"
MODEL = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF:Q5_K_M"
MAX_RETRIES = 3
RETRY_DELAY = 4
TARGET_JOBS = 10
CONTEXT_LIMIT = 28000


def clear_llm_context() -> None:
    """Erase all cached KV slots in llama-server to free context."""
    try:
        slots = json.loads(urllib.request.urlopen(f"{LLM_URL}/slots", timeout=5).read())
        for slot in slots:
            sid = slot.get("id")
            if sid is not None and slot.get("state") != 0:
                urllib.request.urlopen(f"{LLM_URL}/slots/{sid}?action=erase", timeout=5)
    except Exception:
        pass  # server may not be running, or older version


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def chat(prompt: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _post(
                f"{LLM_URL}/v1/chat/completions",
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            return result["choices"][0]["message"]["content"]
        except Exception:
            if attempt < MAX_RETRIES:
                print(f"  [LLM retry {attempt}/{MAX_RETRIES}]")
                time.sleep(RETRY_DELAY)
    raise RuntimeError("LLM failed")


def clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
    return raw.strip()


def llm_json(prompt: str, content: str = "") -> dict | list:
    full = prompt
    if content:
        if len(content) > CONTEXT_LIMIT:
            content = content[:CONTEXT_LIMIT]
        full = prompt + "\n\n" + content
    raw = chat(full)
    try:
        return json.loads(clean_json(raw))
    except json.JSONDecodeError:
        return {} if "{" in prompt else []


def scrape(url: str) -> str:
    return _post(f"{FC_URL}/v1/scrape", {"url": url, "formats": ["markdown"]})["data"]["markdown"]


def search(query: str) -> list[dict]:
    return _post(f"{FC_URL}/v1/search", {"query": query}).get("data", [])


def map_urls(url: str) -> list[str]:
    return _post(f"{FC_URL}/v1/map", {"url": url}).get("links", [])


def crawl(url: str, max_pages: int = 10) -> list[dict]:
    resp = _post(
        f"{FC_URL}/v1/crawl",
        {
            "url": url,
            "maxPages": max_pages,
            "scrapeOptions": {"formats": ["markdown"]},
        },
    )
    job_id = resp["id"]
    while True:
        status = _post(f"{FC_URL}/v1/crawl/{job_id}", {})
        s = status.get("status")
        if s == "completed":
            return status.get("data", [])
        if s == "failed":
            return []
        time.sleep(2)


def chunk_text(text: str, chunk_size: int = 6000) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# ── LLM-powered prompts ──

QUERY_GEN = (
    "Generate 10 diverse natural-language search queries to find undergrad/intern/entry "
    'remote jobs for "{position}". Use plain English phrases people actually search, '
    "NOT advanced operators. Mix: different company types (AI startups, fintech, SaaS, "
    "consulting), different locations (remote USA, remote India, remote Europe, remote "
    "worldwide), different title wordings (intern, junior, associate, entry-level, new grad, "
    "campus hire, university grad). Include 2 queries in non-English languages (Japanese, "
    "Korean, Spanish) for global roles. Return ONLY a JSON array of 10 strings. No markdown."
)


JOB_LIST_EXTRACT = (
    "This is a career/job listing page. Extract EVERY job posting into a JSON array. "
    'Each job: {{"company":"...","role":"...","location":"...","salary":null,'
    '"apply_link":"...","description":"...","is_undergrad":true/false,'
    '"is_remote":true/false,"matches":true/false}}. '
    'Set "matches" to true ONLY if this role genuinely matches the position '
    '"{position}" or is a close equivalent (e.g. SDE → software engineer, '
    "web dev → fullstack, data analyst → data science). Be strict: don't match "
    "unrelated roles (Android dev is NOT fullstack, DevOps is NOT software engineer). "
    "Return ONLY the JSON array. No markdown."
)

SINGLE_EXTRACT = (
    "Extract this job listing. Return ONLY valid JSON: "
    '{{"company":"...","role":"...","location":"...","salary":null,'
    '"apply_link":"...","description":"one sentence",'
    '"is_undergrad":true/false,"is_remote":true/false,'
    '"matches":true/false}}. '
    'Set "matches" to true ONLY if this role is genuinely a "{position}" '
    "role or close equivalent (e.g. SDE → software engineer, web dev → fullstack, "
    "data analyst → data science). Be strict: don't match unrelated roles. "
    "Return ONLY JSON. No markdown."
)

DEDUP_RANK = (
    "Given these {count} job listings for undergrad {position} roles, "
    "remove duplicates (same role at same company), then rank by relevance "
    "(most junior/undergrad-friendly first). Return top {target} as a JSON array. "
    "Keep all original fields. No markdown."
)

CAREER_URL_GEN = (
    "From these search results about {position} jobs, identify companies actively "
    "hiring. Return a JSON array of their career/jobs page URLs. "
    "Only return real company career pages. Return ONLY JSON array. No markdown."
)

QUERY_REGEN = (
    "We need more undergrad {position} roles. Previous attempts found companies "
    "like: {companies}. Generate 8 completely different natural-language search "
    "queries targeting companies NOT already found. Use plain English, no site: "
    "operators. Try niche platforms (Y Combinator jobs, remoteok, weworkremotely, "
    "hackernews whoishiring), specific tech stacks (React intern, Python entry), "
    "or geographic areas (Bangalore entry, Berlin junior). "
    "Return ONLY JSON array. No markdown."
)


def is_good(job: dict) -> bool:
    if not job.get("matches", False):
        return False
    title = (job.get("role", "") + " " + job.get("description", "")).lower()
    bad = [
        "senior",
        "staff engineer",
        "principal",
        "director",
        "vp ",
        "5+ year",
        "7+ year",
        "10+ year",
        "lead ",
        "architect",
        "manager",
        "head of",
        "sr.",
        "sr ",
    ]
    for b in bad:
        if b in title:
            return False
    good = [
        "intern",
        "undergrad",
        "new grad",
        "entry",
        "junior",
        "associate",
        "campus",
        "university",
        "fresher",
        "0-1",
        "0-2",
        "graduate",
        "trainee",
        "early career",
        "co-op",
        "coop",
    ]
    for g in good:
        if g in title:
            return True
    return job.get("is_undergrad", False)


def salary_filter(job: dict) -> bool:
    sal = str(job.get("salary", ""))
    if not sal or sal == "None" or sal == "null":
        return True
    loc = (job.get("location", "") + " " + job.get("description", "")).lower()
    india_kw = [
        "india",
        "bangalore",
        "mumbai",
        "delhi",
        "hyderabad",
        "pune",
        "chennai",
        "gurgaon",
        "noida",
        "bengaluru",
        "gurugram",
    ]
    if any(k in loc for k in india_kw):
        nums = [int(s) for s in sal.replace(",", "").split() if s.replace(",", "").isdigit()]
        for n in nums:
            if n < 80000:
                return False
    return True


def extract_jobs_from_page(content: str, position: str) -> list[dict]:
    """Smart extraction: tries batch extract first, falls back to chunking."""
    if len(content) < 200:
        return []

    p = position

    # Try extracting all jobs at once from a career page
    if len(content) > 1500:
        result = llm_json(JOB_LIST_EXTRACT.replace("{position}", p), content[:CONTEXT_LIMIT])
        if isinstance(result, list) and len(result) > 0:
            return result

    # Chunk large pages and extract per chunk
    if len(content) > 8000:
        jobs = []
        for chunk in chunk_text(content, 6000):
            result = llm_json(JOB_LIST_EXTRACT.replace("{position}", p), chunk)
            if isinstance(result, list):
                jobs.extend(result)
            time.sleep(0.3)
        if jobs:
            return jobs

    # Single extraction fallback
    result = llm_json(SINGLE_EXTRACT.replace("{position}", p), content[:6000])
    if isinstance(result, dict) and result.get("role"):
        return [result]
    return []


def main() -> None:
    position = input("Position: ").strip()
    if not position:
        sys.exit(1)

    print("Clearing LLM context...")
    clear_llm_context()

    seen_urls = set()
    all_jobs = []
    seen_companies = set()
    round_num = 0

    while len(all_jobs) < TARGET_JOBS and round_num < 8:
        round_num += 1
        print(f"\n{'═' * 60}")
        print(
            f"  ROUND {round_num}  │  Jobs: {len(all_jobs)}/{TARGET_JOBS}"
            f"  │  URLs seen: {len(seen_urls)}"
        )
        print(f"{'═' * 60}")

        # ── 1. LLM generates queries ──
        if round_num == 1:
            prompt = QUERY_GEN.replace("{position}", position)
        else:
            companies_str = ", ".join(list(seen_companies)[-10:]) or "none"
            prompt = QUERY_REGEN.replace("{position}", position).replace(
                "{companies}", companies_str
            )
        queries = llm_json(prompt)
        if not isinstance(queries, list) or not queries:
            print("  Failed to generate queries, retrying...")
            queries = llm_json(prompt)
        queries = queries[:10]
        for i, q in enumerate(queries, 1):
            print(f"  Q{i}: {q}")

        # ── 2. Execute all searches in parallel ──
        all_results = []
        for q in queries:
            try:
                results = search(q)
                all_results.extend(results)
                time.sleep(0.3)
            except Exception:
                pass

        # ── 3. Collect all unique URLs ──
        new_urls = [
            r.get("url")
            for r in all_results
            if r.get("url")
            and isinstance(r.get("url"), str)
            and r["url"].startswith("http")
            and r["url"] not in seen_urls
        ]
        new_urls = list(dict.fromkeys(new_urls))
        print(f"\n  {len(new_urls)} new URLs from {len(all_results)} results")

        # ── 4. Scrape + extract ──
        for url in new_urls[:25]:
            seen_urls.add(url)
            try:
                md = scrape(url)
                if not md or len(md) < 100:
                    continue
                extracted = extract_jobs_from_page(md, position)
                for job in extracted:
                    if not isinstance(job, dict):
                        continue
                    job["source_url"] = url
                    if is_good(job) and salary_filter(job):
                        c = job.get("company", "")
                        if c and c not in seen_companies:
                            seen_companies.add(c)
                            all_jobs.append(job)
                            print(f"    ✓ {job.get('role')} @ {c} [{job.get('location', '?')}]")
                time.sleep(0.3)
            except Exception:
                pass
        if len(all_jobs) >= TARGET_JOBS:
            break

        # ── 5. LLM discovers company career pages ──
        buf_text = "\n".join(
            f"{r.get('title', '')} -> {r.get('url', '')}" for r in all_results if r.get("url")
        )
        career_urls = llm_json(
            CAREER_URL_GEN.replace("{position}", position), buf_text[:CONTEXT_LIMIT]
        )
        if not isinstance(career_urls, list):
            career_urls = []
        career_urls = [
            u
            for u in career_urls
            if isinstance(u, str) and u.startswith("http") and u not in seen_urls
        ][:8]
        print(f"\n  LLM discovered {len(career_urls)} career pages")

        # ── 6. Map + crawl each career domain ──
        for career_url in career_urls:
            domain = "/".join(career_url.split("/")[:3])
            seen_urls.add(domain)
            seen_urls.add(career_url)

            # Map for job-related URLs
            try:
                links = map_urls(domain)
                job_links = [
                    link
                    for link in links
                    if any(
                        k in link.lower()
                        for k in ["job", "career", "position", "role", "apply", "open"]
                    )
                ]
                for link in job_links[:8]:
                    if link in seen_urls:
                        continue
                    seen_urls.add(link)
                    try:
                        md = scrape(link)
                        for job in extract_jobs_from_page(md, position):
                            job["source_url"] = link
                            if is_good(job) and salary_filter(job):
                                c = job.get("company", "")
                                if c not in seen_companies:
                                    seen_companies.add(c)
                                    all_jobs.append(job)
                                    print(f"    ✓ {job.get('role')} @ {c}")
                        time.sleep(0.3)
                    except Exception:
                        pass
            except Exception:
                pass

            # Crawl domain
            try:
                pages = crawl(domain, max_pages=8)
                for p in pages:
                    md = p.get("markdown", "")
                    url = p.get("metadata", {}).get("url", "")
                    if md and len(md) > 100 and url not in seen_urls:
                        seen_urls.add(url)
                        for job in extract_jobs_from_page(md, position):
                            job["source_url"] = url
                            if is_good(job) and salary_filter(job):
                                c = job.get("company", "")
                                if c not in seen_companies:
                                    seen_companies.add(c)
                                    all_jobs.append(job)
                                    print(f"    ✓ {job.get('role')} @ {c}")
            except Exception:
                pass

        print(f"\n  Round {round_num} done: {len(all_jobs)} jobs, {len(seen_companies)} companies")

    # ── 7. LLM dedup + rank final results ──
    if len(all_jobs) > TARGET_JOBS:
        jobs_json = json.dumps(all_jobs, ensure_ascii=False)
        ranked = llm_json(
            DEDUP_RANK.replace("{count}", str(len(all_jobs)))
            .replace("{position}", position)
            .replace("{target}", str(TARGET_JOBS)),
            jobs_json[:CONTEXT_LIMIT],
        )
        if isinstance(ranked, list) and len(ranked) > 0:
            all_jobs = ranked

    all_jobs = all_jobs[:TARGET_JOBS]

    # ── 8. Final display ──
    print(f"\n{'═' * 60}")
    print(f"  TOP {len(all_jobs)} {position.upper()} ROLES")
    print("  Undergrad · Remote · Latest")
    print(f"{'═' * 60}")

    for i, j in enumerate(all_jobs, 1):
        print(f"\n  {i}. {j.get('role')} @ {j.get('company')}")
        print(f"     Location:  {j.get('location', 'N/A')}")
        sal = j.get("salary")
        if sal and str(sal) not in ("None", "null", ""):
            print(f"     Salary:    {sal}")
        print(f"     {j.get('description', '')}")
        link = j.get("apply_link") or j.get("source_url", "")
        if link:
            print(f"     Apply:     {link}")

    if not all_jobs:
        print("\n  No matching roles found.")
        print(f"  Tried {len(seen_urls)} URLs across {round_num} rounds.")


if __name__ == "__main__":
    main()
