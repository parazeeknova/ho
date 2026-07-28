"""Job searcher: GitHub internship indexes + web search via Firecrawl SDK."""

import time

from firecrawl import FirecrawlApp

from src.llm.context import ContextManager

GITHUB_INDEXES = [
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/vanshb03/Summer2027-Internships/main/README.md",
    "https://raw.githubusercontent.com/LorenzoLaCorte/european-tech-internships-2026/main/README.md",
    "https://raw.githubusercontent.com/zapplyjobs/Research-Internships-for-Undergraduates/main/README.md",
    "https://raw.githubusercontent.com/DereC4/internships-and-newgrad/main/README.md",
]


def scrape_github_indexes(app: FirecrawlApp) -> list[dict[str, str]]:
    jobs = []
    for url in GITHUB_INDEXES:
        print(f"  Scraping index: {url.split('/')[-3]}/{url.split('/')[-2]}")
        try:
            result = app.scrape_url(url, formats=["markdown"])
            md = getattr(result, "markdown", "") or ""
            jobs.append({"source": url, "markdown": md, "type": "github_index"})
            print(f"    {len(md)} chars")
        except Exception as e:
            print(f"    failed: {e}")
        time.sleep(1)
    return jobs


def search_web(app: FirecrawlApp, position: str, ctx: ContextManager) -> list[dict[str, str]]:
    query_prompt = (
        f"Generate 8 diverse natural-language search queries to find undergrad/"
        f"intern/entry-level remote jobs for: {position}. Target job boards that "
        f"are easy to scrape: GitHub READMEs, Wellfound, Y Combinator jobs, "
        f"company career pages (greenhouse.io, lever.co, ashbyhq.com, workable.com), "
        f"Remotive, WeWorkRemotely, RemoteOK. Avoid indeed, glassdoor, ziprecruiter, "
        f"upwork. Return ONLY a JSON array of 8 strings. No markdown."
    )
    queries = ctx.json_chat(query_prompt)
    if not isinstance(queries, list):
        queries = [f"{position} intern remote", f"entry level {position} remote"]

    results = []
    for q in queries[:8]:
        try:
            search_results = app.search(q)
            data = getattr(search_results, "web", []) or []
            if isinstance(data, list):
                for r in data[:5]:
                    url = getattr(r, "url", "")
                    if url and url.startswith("http"):
                        results.append(
                            {
                                "url": url,
                                "title": getattr(r, "title", "") or "",
                                "type": "web_search",
                            }
                        )
            time.sleep(0.5)
        except Exception:
            pass

    print(f"  Web search: {len(results)} URLs from {len(queries)} queries")
    return results


def scrape_urls(app: FirecrawlApp, urls: list[dict]) -> list[dict[str, str]]:
    jobs = []
    for item in urls:
        url = item.get("url", "")
        if not url:
            continue
        try:
            result = app.scrape_url(url, formats=["markdown"])
            md = getattr(result, "markdown", "") or ""
            if md and len(md) > 100:
                jobs.append({"markdown": md, "url": url, "title": item.get("title", "")})
            time.sleep(0.5)
        except Exception:
            pass
    return jobs
