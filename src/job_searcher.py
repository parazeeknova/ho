"""Job searcher: GitHub internship indexes + web search via firecrawl."""

import json
import time
import urllib.request

FC_URL = "http://127.0.0.1:3002"

GITHUB_INDEXES = [
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/vanshb03/Summer2027-Internships/main/README.md",
    "https://raw.githubusercontent.com/LorenzoLaCorte/european-tech-internships-2026/main/README.md",
    "https://raw.githubusercontent.com/zapplyjobs/Research-Internships-for-Undergraduates/main/README.md",
    "https://raw.githubusercontent.com/DereC4/internships-and-newgrad/main/README.md",
]


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def scrape(url: str) -> str:
    result = _post(f"{FC_URL}/v1/scrape", {"url": url, "formats": ["markdown"]})
    return result["data"]["markdown"]


def search(query: str) -> list[dict]:
    return _post(f"{FC_URL}/v1/search", {"query": query}).get("data", [])


def scrape_github_indexes() -> list[dict[str, str]]:
    """Scrape all GitHub internship indexes for raw job listings."""
    jobs = []
    for url in GITHUB_INDEXES:
        print(f"  Scraping index: {url.split('/')[-3]}/{url.split('/')[-2]}")
        try:
            md = scrape(url)
            jobs.append({"source": url, "markdown": md, "type": "github_index"})
            print(f"    {len(md)} chars")
        except Exception as e:
            print(f"    failed: {e}")
        time.sleep(1)
    return jobs


def search_web(position: str, ctx) -> list[dict[str, str]]:
    """Generate queries from resume + position, search the web."""
    query_prompt = (
        f"Generate 8 diverse natural-language search queries to find undergrad/"
        f"intern/entry-level remote jobs for: {position}. Use plain English "
        f"phrases. Target varied platforms and company types. "
        f"Return ONLY a JSON array of 8 strings. No markdown."
    )
    queries = ctx.json_chat(query_prompt)
    if not isinstance(queries, list):
        queries = [f"{position} intern remote", f"entry level {position} remote"]

    results = []
    for q in queries[:8]:
        try:
            for r in search(q)[:5]:
                url = r.get("url", "")
                if url and url.startswith("http"):
                    results.append({"url": url, "title": r.get("title", ""), "type": "web_search"})
            time.sleep(0.5)
        except Exception:
            pass

    print(f"  Web search: {len(results)} URLs from {len(queries)} queries")
    return results
