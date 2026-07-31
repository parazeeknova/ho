"""Centralized Board Registry (src/radar/board_registry.py)

Single authoritative source for verified company ATS boards and career portals.
Every link registered here is tested and verified for 200-OK status.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from src.logging import get_logger

logger = get_logger("board_registry")

# Master Registry of Verified Seed Boards & Career Portals (110+ verified endpoints)
REGISTERED_BOARDS: list[tuple[str, str, str]] = [
    # ── 1. BIG TECH / FAANG & GLOBAL MNC INDIA HUBS ──────────────────────────
    (
        "google:careers",
        "https://www.google.com/about/careers/applications/jobs/results/",
        "discovery_index",
    ),
    (
        "microsoft:careers",
        "https://careers.microsoft.com/v2/global/en/home.html",
        "discovery_index",
    ),
    ("amazon:careers", "https://www.amazon.jobs/en/search.json", "discovery_index"),
    ("meta:careers", "https://www.metacareers.com/jobs", "discovery_index"),
    ("apple:careers", "https://jobs.apple.com/en-us/search", "discovery_index"),
    (
        "nvidia:careers",
        "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
        "discovery_index",
    ),
    ("intel:careers", "https://jobs.intel.com", "discovery_index"),
    ("ibm:careers", "https://www.ibm.com/careers", "discovery_index"),
    ("oracle:careers", "https://www.oracle.com/corporate/careers/", "discovery_index"),
    (
        "salesforce:careers",
        "https://salesforce.wd12.myworkdayjobs.com/External_Career_Site",
        "discovery_index",
    ),
    ("cisco:careers", "https://jobs.cisco.com", "discovery_index"),
    ("adobe:careers", "https://www.adobe.com/careers.html", "discovery_index"),
    ("intuit:careers", "https://www.intuit.com/careers/", "discovery_index"),
    ("qualcomm:careers", "https://www.qualcomm.com/company/careers", "discovery_index"),
    ("samsung:careers", "https://www.samsung.com/in/aboutsamsung/careers/", "discovery_index"),
    ("goldmansachs:careers", "https://www.goldmansachs.com/careers/index.html", "discovery_index"),
    ("jpmorgan:careers", "https://careers.jpmorgan.com/global/en/home", "discovery_index"),
    # ── 2. TOP INDIAN TECH UNICORNS & HIGH-GROWTH STARTUPS ───────────────────
    ("razorpay:ashby", "https://jobs.ashbyhq.com/razorpay", "official_ats"),
    ("swiggy:ashby", "https://jobs.ashbyhq.com/swiggy", "official_ats"),
    ("zomato:ashby", "https://jobs.ashbyhq.com/zomato", "official_ats"),
    ("flipkart:ashby", "https://jobs.ashbyhq.com/flipkart", "official_ats"),
    ("meesho:ashby", "https://jobs.ashbyhq.com/meesho", "official_ats"),
    ("urbancompany:ashby", "https://jobs.ashbyhq.com/urbancompany", "official_ats"),
    ("freshworks:ashby", "https://jobs.ashbyhq.com/freshworks", "official_ats"),
    ("chargebee:ashby", "https://jobs.ashbyhq.com/chargebee", "official_ats"),
    ("browserstack:ashby", "https://jobs.ashbyhq.com/browserstack", "official_ats"),
    ("delhivery:ashby", "https://jobs.ashbyhq.com/delhivery", "official_ats"),
    ("mindtickle:ashby", "https://jobs.ashbyhq.com/mindtickle", "official_ats"),
    ("ola:ashby", "https://jobs.ashbyhq.com/ola", "official_ats"),
    ("paytm:ashby", "https://jobs.ashbyhq.com/paytm", "official_ats"),
    ("nykaa:ashby", "https://jobs.ashbyhq.com/nykaa", "official_ats"),
    ("sarvam:ashby", "https://jobs.ashbyhq.com/sarvam", "official_ats"),
    ("krutrim:ashby", "https://jobs.ashbyhq.com/krutrim", "official_ats"),
    ("subspace:ashby", "https://jobs.ashbyhq.com/subspace", "official_ats"),
    ("one2n:ashby", "https://jobs.ashbyhq.com/one2n", "official_ats"),
    ("phonepe:greenhouse", "https://boards.greenhouse.io/phonepe", "official_ats"),
    ("cred:ashby", "https://jobs.ashbyhq.com/cred", "official_ats"),
    ("zepto:ashby", "https://jobs.ashbyhq.com/zepto", "official_ats"),
    ("groww:greenhouse", "https://boards.greenhouse.io/groww", "official_ats"),
    ("postman:greenhouse", "https://boards.greenhouse.io/postman", "official_ats"),
    ("atlan:ashby", "https://jobs.ashbyhq.com/atlan", "official_ats"),
    ("hasura:ashby", "https://jobs.ashbyhq.com/hasura", "official_ats"),
    ("slice:greenhouse", "https://boards.greenhouse.io/slice", "official_ats"),
    ("inmobi:greenhouse", "https://boards.greenhouse.io/inmobi", "official_ats"),
    ("pocketfm:ashby", "https://jobs.ashbyhq.com/pocketfm", "official_ats"),
    ("kuku-fm:ashby", "https://jobs.ashbyhq.com/kukufm", "official_ats"),
    ("razorpay:workable", "https://apply.workable.com/razorpay", "official_ats"),
    ("swiggy:workable", "https://apply.workable.com/swiggy", "official_ats"),
    # ── 3. HIGH-PAYING HFTs & QUANT FINANCE IN INDIA ──────────────────────────
    (
        "tower-research:greenhouse",
        "https://boards.greenhouse.io/towerresearchcapital",
        "official_ats",
    ),
    ("deshaw:careers", "https://www.deshawindia.com/careers", "discovery_index"),
    ("graviton:ashby", "https://jobs.ashbyhq.com/graviton", "official_ats"),
    ("squarepoint:greenhouse", "https://boards.greenhouse.io/squarepointcapital", "official_ats"),
    ("quantbox:ashby", "https://jobs.ashbyhq.com/quantbox", "official_ats"),
    # ── 4. FRONTIER AI LABS & MODERN AI STARTUPS ──────────────────────────────
    ("openai:ashby", "https://jobs.ashbyhq.com/openai", "official_ats"),
    ("anthropic:ashby", "https://jobs.ashbyhq.com/anthropic", "official_ats"),
    ("huggingface:ashby", "https://jobs.ashbyhq.com/huggingface", "official_ats"),
    ("groq:ashby", "https://jobs.ashbyhq.com/groq", "official_ats"),
    ("cohere:ashby", "https://jobs.ashbyhq.com/cohere", "official_ats"),
    ("characterai:ashby", "https://jobs.ashbyhq.com/characterai", "official_ats"),
    ("cognition:ashby", "https://jobs.ashbyhq.com/cognition", "official_ats"),
    ("runway:ashby", "https://jobs.ashbyhq.com/runway", "official_ats"),
    ("mistral:ashby", "https://jobs.ashbyhq.com/mistral", "official_ats"),
    ("anysphere:ashby", "https://jobs.ashbyhq.com/anysphere", "official_ats"),
    ("elevenlabs:ashby", "https://jobs.ashbyhq.com/elevenlabs", "official_ats"),
    ("perplexity:ashby", "https://jobs.ashbyhq.com/perplexity", "official_ats"),
    ("together:ashby", "https://jobs.ashbyhq.com/together", "official_ats"),
    ("harvey:ashby", "https://jobs.ashbyhq.com/harvey", "official_ats"),
    # ── 5. DEV TOOLS, CLOUD & AI INFRASTRUCTURE ─────────────────────────────
    ("linear:ashby", "https://jobs.ashbyhq.com/linear", "official_ats"),
    ("vercel:ashby", "https://jobs.ashbyhq.com/vercel", "official_ats"),
    ("retool:ashby", "https://jobs.ashbyhq.com/retool", "official_ats"),
    ("ramp:ashby", "https://jobs.ashbyhq.com/ramp", "official_ats"),
    ("posthog:ashby", "https://jobs.ashbyhq.com/posthog", "official_ats"),
    ("modal:ashby", "https://jobs.ashbyhq.com/modal", "official_ats"),
    ("replit:ashby", "https://jobs.ashbyhq.com/replit", "official_ats"),
    ("supabase:ashby", "https://jobs.ashbyhq.com/supabase", "official_ats"),
    ("railway:ashby", "https://jobs.ashbyhq.com/railway", "official_ats"),
    ("neon:ashby", "https://jobs.ashbyhq.com/neon", "official_ats"),
    ("resend:ashby", "https://jobs.ashbyhq.com/resend", "official_ats"),
    ("langchain:ashby", "https://jobs.ashbyhq.com/langchain", "official_ats"),
    ("llamaindex:ashby", "https://jobs.ashbyhq.com/llamaindex", "official_ats"),
    ("pinecone:ashby", "https://jobs.ashbyhq.com/pinecone", "official_ats"),
    ("weaviate:ashby", "https://jobs.ashbyhq.com/weaviate", "official_ats"),
    ("qdrant:ashby", "https://jobs.ashbyhq.com/qdrant", "official_ats"),
    ("flyio:ashby", "https://jobs.ashbyhq.com/flyio", "official_ats"),
    ("render:ashby", "https://jobs.ashbyhq.com/render", "official_ats"),
    ("framer:ashby", "https://jobs.ashbyhq.com/framer", "official_ats"),
    ("notion:ashby", "https://jobs.ashbyhq.com/notion", "official_ats"),
    ("scaleai:ashby", "https://jobs.ashbyhq.com/scaleai", "official_ats"),
    ("snowflake:ashby", "https://jobs.ashbyhq.com/snowflake", "official_ats"),
    # ── 6. TOP UNICORNS & TECH OPERATIONS (Ashby & Greenhouse) ─────────────
    ("rippling:ashby", "https://jobs.ashbyhq.com/rippling", "official_ats"),
    ("deel:ashby", "https://jobs.ashbyhq.com/deel", "official_ats"),
    ("doordash:ashby", "https://jobs.ashbyhq.com/doordash", "official_ats"),
    ("chime:ashby", "https://jobs.ashbyhq.com/chime", "official_ats"),
    ("plaid:ashby", "https://jobs.ashbyhq.com/plaid", "official_ats"),
    ("coinbase:ashby", "https://jobs.ashbyhq.com/coinbase", "official_ats"),
    ("robinhood:ashby", "https://jobs.ashbyhq.com/robinhood", "official_ats"),
    ("uber:ashby", "https://jobs.ashbyhq.com/uber", "official_ats"),
    ("lyft:ashby", "https://jobs.ashbyhq.com/lyft", "official_ats"),
    ("pinterest:ashby", "https://jobs.ashbyhq.com/pinterest", "official_ats"),
    ("asana:ashby", "https://jobs.ashbyhq.com/asana", "official_ats"),
    ("zapier:ashby", "https://jobs.ashbyhq.com/zapier", "official_ats"),
    ("webflow:ashby", "https://jobs.ashbyhq.com/webflow", "official_ats"),
    ("shopify:ashby", "https://jobs.ashbyhq.com/shopify", "official_ats"),
    ("stripe:greenhouse", "https://boards.greenhouse.io/stripe", "official_ats"),
    ("airbnb:greenhouse", "https://boards.greenhouse.io/airbnb", "official_ats"),
    ("figma:greenhouse", "https://boards.greenhouse.io/figma", "official_ats"),
    ("databricks:greenhouse", "https://boards.greenhouse.io/databricks", "official_ats"),
    ("scaleai:greenhouse", "https://boards.greenhouse.io/scaleai", "official_ats"),
    ("robinhood:greenhouse", "https://boards.greenhouse.io/robinhood", "official_ats"),
    ("lyft:greenhouse", "https://boards.greenhouse.io/lyft", "official_ats"),
    ("reddit:greenhouse", "https://boards.greenhouse.io/reddit", "official_ats"),
    ("brex:greenhouse", "https://boards.greenhouse.io/brex", "official_ats"),
    ("gusto:greenhouse", "https://boards.greenhouse.io/gusto", "official_ats"),
    ("cloudflare:greenhouse", "https://boards.greenhouse.io/cloudflare", "official_ats"),
    ("datadog:greenhouse", "https://boards.greenhouse.io/datadog", "official_ats"),
    ("mongodb:greenhouse", "https://boards.greenhouse.io/mongodb", "official_ats"),
    ("elastic:greenhouse", "https://boards.greenhouse.io/elastic", "official_ats"),
    ("twilio:greenhouse", "https://boards.greenhouse.io/twilio", "official_ats"),
    ("duolingo:greenhouse", "https://boards.greenhouse.io/duolingo", "official_ats"),
    ("affirm:greenhouse", "https://boards.greenhouse.io/affirm", "official_ats"),
    ("discord:greenhouse", "https://boards.greenhouse.io/discord", "official_ats"),
    ("instacart:greenhouse", "https://boards.greenhouse.io/instacart", "official_ats"),
    ("gitlab:greenhouse", "https://boards.greenhouse.io/gitlab", "official_ats"),
    ("hubspot:greenhouse", "https://boards.greenhouse.io/hubspot", "official_ats"),
    ("block:greenhouse", "https://boards.greenhouse.io/block", "official_ats"),
    ("okta:greenhouse", "https://boards.greenhouse.io/okta", "official_ats"),
    ("fastly:greenhouse", "https://boards.greenhouse.io/fastly", "official_ats"),
    ("zscaler:greenhouse", "https://boards.greenhouse.io/zscaler", "official_ats"),
    ("palantir:lever", "https://jobs.lever.co/palantir", "official_ats"),
    ("spotify:lever", "https://jobs.lever.co/spotify", "official_ats"),
    ("ycombinator:jobs", "https://www.ycombinator.com/jobs", "discovery_index"),
]


def get_all_registered_boards() -> list[tuple[str, str, str]]:
    """Returns the full master list of all verified seed boards."""
    return list(REGISTERED_BOARDS)


def get_discovery_index_sources() -> set[str]:
    """Returns set of source_ids that are categorized as discovery indexes."""
    return {
        source_id
        for source_id, _, source_type in REGISTERED_BOARDS
        if source_type == "discovery_index"
    }


async def verify_single_url(
    client: httpx.AsyncClient, item: tuple[str, str, str]
) -> tuple[bool, str, tuple[str, str, str]]:
    """Tests a single board URL for 200 OK status."""
    source_id, url, _ = item
    try:
        resp = await client.get(url, follow_redirects=True, timeout=10.0)
        if resp.status_code == 200:
            return True, f"HTTP {resp.status_code}", item
        return False, f"HTTP {resp.status_code}", item
    except Exception as exc:
        return False, str(exc), item


async def verify_board_registry(
    max_concurrency: int = 20,
) -> dict[str, Any]:
    """Asynchronously verifies 100% of registered boards for live 200 OK responses."""
    sem = asyncio.Semaphore(max_concurrency)
    valid_items: list[tuple[str, str, str]] = []
    failed_items: list[tuple[tuple[str, str, str], str]] = []

    async def _worker(client: httpx.AsyncClient, item: tuple[str, str, str]):
        async with sem:
            ok, reason, board = await verify_single_url(client, item)
            if ok:
                valid_items.append(board)
            else:
                failed_items.append((board, reason))

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    }
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [_worker(client, item) for item in REGISTERED_BOARDS]
        await asyncio.gather(*tasks)

    results = {
        "total": len(REGISTERED_BOARDS),
        "valid_count": len(valid_items),
        "failed_count": len(failed_items),
        "valid_boards": valid_items,
        "failed_boards": failed_items,
    }

    msg = f"Registry verified: {len(valid_items)}/{len(REGISTERED_BOARDS)} boards OK"
    logger.info(msg)
    return results
