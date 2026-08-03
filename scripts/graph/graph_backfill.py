"""Graph backfill: populate Neo4j companies + founders from the index.

Reads every company in companies_index plus every ATS slug found in
job_observations, upserts company GraphNodes, and runs the pipeline's own
_founder_miner (Wikipedia founders + StartupAgent OSINT + FOUNDED_BY edges)
for each. Founder cold-DM cards go to Telegram for founders with emails,
capped so the backfill never spams.

Run:  uv run python3 scripts/graph_backfill.py
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from src.agent.startup_agent import StartupAgent
from src.agent.telegram_agent import TelegramAgent
from src.graph.entity import FrontierEntry, NodeType, make_work_id
from src.graph.entity import company_node as _cn
from src.graph.event_bus import EventBus
from src.graph.graph_store import GraphStore
from src.llm.context import ContextManager
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore
from src.radar.engine.orchestrator import _founder_miner

logger = get_logger("graph_backfill")

_ATS_SLUG_RE = re.compile(
    r"https?://(?:boards\.greenhouse\.io|jobs\.lever\.co|jobs\.ashbyhq\.com|"
    r"apply\.workable\.com)/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)

MAX_CARDS = 20
_cards_sent = 0
_seen_companies: set[str] = set()


async def _sub_founder(event: Any) -> list[Any]:
    global _cards_sent
    if _cards_sent >= MAX_CARDS:
        return []
    d = event.payload
    email = d.get("email")
    linkedin = d.get("linkedin_url")
    if not (email or linkedin):
        return []
    _cards_sent += 1
    try:
        ta = TelegramAgent()
        if ta.is_configured:
            await ta.send_categorized_alert(
                "outreach",
                {
                    "role": f"Founder: {d.get('name', '?')}",
                    "company": d.get("company", ""),
                    "apply_link": linkedin or f"mailto:{email}",
                    "founders": [
                        {"name": d.get("name", "?"), "email": email, "linkedin_url": linkedin}
                    ],
                    "funding_stage": d.get("funding_stage", ""),
                },
                dedup_key=f"founder:{event.node_id}",
            )
    except Exception as exc:
        logger.warning(f"founder card failed: {exc}")
    return []


async def _company_list(store) -> list[tuple[str, int]]:
    companies: dict[str, int] = {}
    async with store._pool.acquire() as conn:
        rows = await conn.fetch("SELECT slug, job_count FROM companies_index WHERE slug != ''")
        for r in rows:
            companies[r["slug"]] = max(companies.get(r["slug"], 0), int(r["job_count"] or 0))
        url_rows = await conn.fetch(
            """SELECT DISTINCT url FROM job_observations
               WHERE url LIKE '%%greenhouse.io%%' OR url LIKE '%%lever.co%%'
                  OR url LIKE '%%ashbyhq.com%%' OR url LIKE '%%workable.com%%'"""
        )
    for r in url_rows:
        for slug in _ATS_SLUG_RE.findall(str(r["url"])):
            slug = slug.lower().rstrip("/")
            if slug in ("embed", "v1", "api", "www", "board"):
                continue
            companies.setdefault(slug, 0)
            companies[slug] = companies.get(slug, 0) + 1
    return sorted(companies.items(), key=lambda kv: -kv[1])


async def main() -> None:
    store = await MemoryStore.create()
    graph = await GraphStore.create()
    bus = EventBus()
    bus.subscribe("founder_discovered", _sub_founder)
    ctx = ContextManager()
    sa = StartupAgent(ctx, store=store)

    companies = await _company_list(store)
    logger.info(f"Backfill: {len(companies)} companies to process")

    processed = 0
    errors = 0
    t0 = time.time()
    for company, _count in companies:
        if not company or company.lower() in _seen_companies:
            continue
        _seen_companies.add(company.lower())
        try:
            node = await graph.get_node(company)
            if node is None:
                node = _cn(company, source="radar_backfill")
                node, _ = await graph.upsert_node(node)
            entry = FrontierEntry(
                id=make_work_id("founder_miner", node.id),
                agent="founder_miner",
                node_id=node.id,
                node_type=NodeType.COMPANY,
                priority=70,
                depth=1,
                payload={"company": company},
            )
            await _founder_miner(entry, graph, bus, sa)
            processed += 1
            if processed % 50 == 0:
                el = time.time() - t0
                remaining = len(companies) - processed
                logger.info(
                    f"Backfill progress: {processed}/{len(companies)} "
                    f"({el:.0f}s, {el / max(processed, 1) * remaining / 60:.0f} min left)"
                )
        except Exception as exc:
            errors += 1
            logger.warning(f"backfill {company}: {exc}")
    logger.info(f"Backfill done: {processed} companies, {errors} errors, {_cards_sent} cards sent")


if __name__ == "__main__":
    asyncio.run(main())
