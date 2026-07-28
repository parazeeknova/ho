"""Pipeline: resume → search → MQ → RAG match → revalidate → verify → output."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from firecrawl import FirecrawlApp

from llm.context import ContextManager
from matching.matcher import batch_match
from matching.verifier import verify_job
from output.writer import write_md
from pipeline.queue import JobPipeline, QueuedJob
from rag.engine import build_rag_from_chunks
from rag.loader import load_resume
from search.searcher import GITHUB_INDEXES, search_web

TARGET = 15
MAX_SCRAPE_WORKERS = 6
MATCH_BATCH_SIZE = 5


def extract_jobs_from_index(markdown: str, ctx: ContextManager) -> list[dict]:
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


def scrape_index_worker(url: str, app: FirecrawlApp, pipeline: JobPipeline) -> None:
    try:
        result = app.scrape_url(url, params={"formats": ["markdown"]})
        md = result.get("markdown", result.get("data", {}).get("markdown", ""))
        if md:
            pipeline.push(QueuedJob(markdown=md, url=url, title=f"INDEX:{url.split('/')[-1]}"))
    except Exception as e:
        print(f"  [err] index {url}: {e}")


def scrape_url_worker(item: dict, app: FirecrawlApp, pipeline: JobPipeline) -> None:
    url = item.get("url", "")
    if not url:
        return
    try:
        result = app.scrape_url(url, params={"formats": ["markdown"]})
        md = result.get("markdown", result.get("data", {}).get("markdown", ""))
        if md and len(md) > 100:
            pipeline.push(QueuedJob(markdown=md, url=url, title=item.get("title", "")))
    except Exception as e:
        print(f"  [err] scrape {url}: {e}")


def consumer_loop(pipeline: JobPipeline, rag, ctx: ContextManager) -> list[dict]:
    matched: list[dict] = []
    index_jobs: list[dict] = []
    web_buf: list[QueuedJob] = []

    while True:
        job = pipeline.pop(timeout=2)
        if job is None:
            if pipeline.is_done:
                break
            continue

        if job.title.startswith("INDEX:"):
            if len(job.markdown) > 500:
                extracted = extract_jobs_from_index(job.markdown, ctx)
                index_jobs.extend(extracted)
                print(f"  [consumer] {len(extracted)} from {job.title}")
        else:
            web_buf.append(job)

        if len(web_buf) >= MATCH_BATCH_SIZE:
            batch = [{"markdown": j.markdown, "url": j.url, "title": j.title} for j in web_buf]
            scored = batch_match(batch, rag, ctx)
            matched.extend(scored)
            for _ in web_buf:
                pipeline.task_done()
            web_buf.clear()

        print(f"  [{pipeline.log_status()}]", end="\r")

    # Flush remaining
    if index_jobs:
        idx_batch = [
            {
                "markdown": "",
                "url": j.get("apply_link", ""),
                "title": j.get("role", ""),
                "snippet": str(j),
            }
            for j in index_jobs[:50]
        ]
        scored = batch_match(idx_batch, rag, ctx)
        matched.extend(scored)

    if web_buf:
        batch = [{"markdown": j.markdown, "url": j.url, "title": j.title} for j in web_buf]
        scored = batch_match(batch, rag, ctx)
        matched.extend(scored)
        for _ in web_buf:
            pipeline.task_done()

    print()
    return matched


def run() -> None:
    ctx = ContextManager()
    ctx.flush()
    app = FirecrawlApp(api_key="sk-no-auth", api_url="http://127.0.0.1:3002")
    pipeline = JobPipeline()

    print("=" * 60)
    print("  PHASE 1: Load Resume + Build RAG Index")
    print("=" * 60)
    full_text, chunks = load_resume()
    rag = build_rag_from_chunks(chunks)
    print(f"  Indexed {len(rag.doc_texts)} chunks")

    print("\n" + "=" * 60)
    print("  PHASE 2: Producer/Consumer (scrape → MQ → LLM match)")
    print("=" * 60)

    matched_result: list[dict] = []

    def _consumer_target() -> None:
        nonlocal matched_result
        matched_result = consumer_loop(pipeline, rag, ctx)

    consumer = threading.Thread(target=_consumer_target, daemon=True)
    consumer.start()

    position = (
        ctx.chat(
            "Based on this resume, what is the single best job title "
            "to search for? Return just the title, nothing else.\n\n" + full_text[:2000]
        )
        .strip()
        .strip('"')
    )
    print(f"  Target position: {position}")

    with ThreadPoolExecutor(max_workers=MAX_SCRAPE_WORKERS) as executor:
        futures = []
        for url in GITHUB_INDEXES:
            futures.append(executor.submit(scrape_index_worker, url, app, pipeline))
        web_hits = search_web(app, position, ctx)
        for hit in web_hits:
            futures.append(executor.submit(scrape_url_worker, hit, app, pipeline))
        print(f"  {len(futures)} tasks running, consumer draining...")
        for f in as_completed(futures):
            f.result()

    print("  Producers done. Signalling stop...")
    pipeline.signal_done()
    consumer.join(timeout=300)

    print("\n" + "=" * 60)
    print("  PHASE 3: RAG Revalidation")
    print("=" * 60)
    validated = []
    for j in matched_result:
        v = rag.revalidate(j, ctx)
        if v and v.get("match_percent", 0) >= 30:
            validated.append(v)
            pct = v["match_percent"]
            role = v.get("role", "?")
            company = v.get("company", "?")
            tag = "[reval]" if v.get("_revalidated") else "[kept]"
            print(f"  {tag} {pct}% | {role} @ {company}")
    dropped = len(matched_result) - len(validated)
    print(f"  Kept {len(validated)}, dropped {dropped}")

    print("\n" + "=" * 60)
    print("  PHASE 4: Filter + Cross-Verify")
    print("=" * 60)
    scored = filter_recent(validated, max_days=7)
    scored.sort(key=lambda j: j["match_percent"], reverse=True)
    scored = scored[:TARGET]

    verified = []
    for j in scored[:TARGET]:
        role = j.get("role", "?")
        company = j.get("company", "?")
        url = j.get("source_url", "")
        print(f"  Verifying: {role} @ {company}")
        if verify_job(app, role, company, url, ctx):
            verified.append(j)
            print("    verified")
        else:
            print("    FAILED — dropped")

    print("\n" + "=" * 60)
    print("  PHASE 5: Generate Output")
    print("=" * 60)
    write_md(verified)
    ctx.flush()

    print(f"\n  Redis queue: {pipeline.pending} items remaining")
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    run()
