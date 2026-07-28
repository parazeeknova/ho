"""Pipeline orchestrator: resume → search → match → verify → output."""

import json
import time
import urllib.request
from datetime import UTC

from src.context_manager import ContextManager
from src.job_matcher import batch_match
from src.job_searcher import scrape_github_indexes, search_web
from src.job_verifier import verify_job
from src.output_writer import write_md
from src.resume_loader import load_resume

FC_URL = "http://127.0.0.1:3002"
TARGET = 15


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def scrape(url: str) -> str:
    return _post(f"{FC_URL}/v1/scrape", {"url": url, "formats": ["markdown"]})["data"]["markdown"]


def extract_jobs_from_index(markdown: str, ctx) -> list[dict]:
    """Use LLM to extract all job listings from a GitHub index markdown."""
    prompt = (
        "Extract ALL job/internship listings from this markdown. "
        "Return a JSON array. Each entry: "
        '{{"company":"...","role":"...","location":"...",'
        '"apply_link":"...","posted":"..."}}. '
        "Be exhaustive — extract every single row/listing. "
        "Return ONLY JSON array."
    )
    result = ctx.json_chat(prompt, markdown, limit=20000)
    return result if isinstance(result, list) else []


def filter_recent(jobs: list[dict], max_days: int = 7) -> list[dict]:
    """Keep only jobs posted within max_days. If no date, keep it."""
    from datetime import datetime

    filtered = []
    now = datetime.now(UTC)
    for j in jobs:
        date_str = j.get("posted_date") or j.get("posted")
        if not date_str:
            filtered.append(j)
            continue
        try:
            dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            if (now - dt).days <= max_days:
                filtered.append(j)
        except ValueError, TypeError:
            filtered.append(j)
    return filtered


def run() -> None:
    ctx = ContextManager()
    ctx.flush()  # clean start

    # ── 1. Load resume ──
    print("=" * 60)
    print("  PHASE 1: Load Resume")
    print("=" * 60)
    full_text, chunks = load_resume()
    resume_summary = (
        f"SKILLS:\n{chunks.get('skills', '')[:2000]}\n\n"
        f"EXPERIENCE:\n{chunks.get('experience', '')[:2000]}\n\n"
        f"EDUCATION:\n{chunks.get('education', '')[:1000]}\n\n"
        f"PROJECTS:\n{chunks.get('projects', '')[:1500]}"
    )

    # ── 2. Scrape GitHub indexes ──
    print("\n" + "=" * 60)
    print("  PHASE 2: Scrape GitHub Internship Indexes")
    print("=" * 60)
    indexes = scrape_github_indexes()
    index_jobs = []
    for idx in indexes:
        extracted = extract_jobs_from_index(idx["markdown"], ctx)
        index_jobs.extend(extracted)
        print(f"  Extracted {len(extracted)} listings from {idx['source'].split('/')[-1]}")
    print(f"  Total from indexes: {len(index_jobs)}")

    # ── 3. Web search for additional jobs ──
    print("\n" + "=" * 60)
    print("  PHASE 3: Web Search")
    print("=" * 60)
    position = (
        ctx.chat(
            "Based on this resume, what is the single best job title to search for? "
            "Return just the title, nothing else.\n\n" + resume_summary[:2000]
        )
        .strip()
        .strip('"')
    )
    print(f"  Target position: {position}")

    web_results = search_web(position, ctx)
    web_jobs = []
    for wr in web_results[:20]:
        url = wr.get("url", "")
        if not url:
            continue
        try:
            print(f"  Scraping: {wr.get('title', url)[:60]}")
            md = scrape(url)
            if md and len(md) > 200:
                web_jobs.append({"markdown": md, "url": url, "title": wr.get("title", "")})
            time.sleep(0.5)
        except Exception as e:
            print(f"    failed: {e}")
    print(f"  Scraped {len(web_jobs)} web listings")

    # ── 4. Match jobs against resume ──
    print("\n" + "=" * 60)
    print("  PHASE 4: Match Jobs Against Resume")
    print("=" * 60)

    all_jobs_to_match = [
        {
            "markdown": j.get("markdown"),
            "url": j.get("source_url", j.get("url", "")),
            "title": j.get("role", j.get("title", "")),
        }
        for j in index_jobs[:50]
    ]
    all_jobs_to_match.extend(web_jobs)

    scored = batch_match(all_jobs_to_match, resume_summary, ctx)
    scored = filter_recent(scored, max_days=7)
    scored = scored[:TARGET]
    print(f"\n  Kept {len(scored)} jobs (matched + recent)")

    # ── 5. Cross-verify ──
    print("\n" + "=" * 60)
    print("  PHASE 5: Cross-Verify Top Matches")
    print("=" * 60)
    verified = []
    for j in scored[:TARGET]:
        role = j.get("role", "?")
        company = j.get("company", "?")
        url = j.get("source_url", "")
        print(f"  Verifying: {role} @ {company}")
        if verify_job(role, company, url, ctx):
            verified.append(j)
            print("    verified")
        else:
            print("    FAILED verification — dropped")
    print(f"  {len(verified)}/{len(scored)} passed verification")

    # ── 6. Output ──
    print("\n" + "=" * 60)
    print("  PHASE 6: Generate Output")
    print("=" * 60)
    write_md(verified)
    ctx.flush()

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    run()
