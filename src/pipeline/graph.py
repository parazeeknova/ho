"""Multi-agent graph topology for job matching with self-correction.

Pipeline (sequential state machine):

  Node 1  –  Context Builder   (deterministic: dedup + RAG retrieval)
  Node 2  –  Matcher Agent     (LLM: gemma-4-31B-it with structured output)
  Node 3  –  Critic Agent      (LLM + hard-constraint rules)
  Edge    –  Self-correction    (route back to Matcher if needed, max 2 retries)
  Node 4  –  Memory Saver      (persist to pgvector)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import Any, TypedDict, cast

import httpx

from src.configuration import get_config
from src.llm.config import build_embed_query
from src.llm.context import ContextManager
from src.llm.schemas import CriticReview, JobMatch, canonicalize_markdown
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore
from src.search.searcher import harvest_and_save_domains

logger = get_logger("pipeline_graph")

MAX_CORRECTION_LOOPS = 2

_graph_retry_queue: list[dict[str, str]] = []


def drain_retry_queue() -> list[dict[str, str]]:
    jobs = list(_graph_retry_queue)
    _graph_retry_queue.clear()
    return jobs


def _queue_for_retry(state: GraphState) -> None:
    _graph_retry_queue.append(
        {
            "markdown": state["markdown"],
            "url": state["url"],
            "title": state.get("title", ""),
        }
    )


MATCHER_PROMPT = """\
You are a job-resume matching engine. Compare this job description against the \
RELEVANT RESUME SNIPPETS below (these are only the most relevant parts of the \
resume, not the full resume).

Relevant resume snippets:
{relevant_chunks}

Full job listing:
{job_description}

CRITICAL RULES:
- Extract all metadata fields: role title, company name, \
company_description (1-2 sentence company overview), role_summary (1-2 sentence role overview), \
location (city/country or "Remote"), salary (if mentioned), posted_date, and apply_link.
- If the text is a company homepage, job directory, error page, or lists \
multiple different jobs instead of ONE SINGLE posting, set match_percent=0 and \
verdict=NO_MATCH.
- The candidate is early-career / new-grad / intern based in India. Accept \
remote roles worldwide, onsite roles in India, and roles with visa sponsorship.
- If salary is specified below 70K INR/month (or equivalent), set match_percent=0 \
and verdict=NO_MATCH.
- If the job description explicitly states the posting is older than 24 hours \
or more than 1 day old, set match_percent=0 and verdict=NO_MATCH.
- If the job requires 5+ years of experience or is titled Senior/Staff/Lead/\
Principal/Manager/Director/Architect, set match_percent=0 and verdict=NO_MATCH.
- Be conservative. STRONG_MATCH only with genuine skill alignment.
- Return valid JSON matching the required schema.
"""

CRITIC_PROMPT = """\
You are a strict job-match auditor. Review this match result and check it \
against hard constraints.

Match result:
{result}

Job description:
{job_description}

HARD CONSTRAINTS — fail the match if ANY of these are true:
1. The role title contains Senior, Staff, Lead, Principal, Manager, Director, \
Architect, VP, or "5+ years", "7+ years", "10+ years" of experience.
2. The salary (if present) is below 70K INR per month or equivalent.
3. The match_percent is unreasonably high given the skills gap or JD content.
4. The JD text is clearly a landing page, directory, or error page rather than \
a real job posting.
5. The JD explicitly mentions it was posted more than 24 hours ago (or >1 day \
ago, or a date older than yesterday).

Return a JSON object with:
- passed: bool (true if the match passes all hard constraints)
- critique_reason: string (explain what failed, or "All checks passed")
- requires_rescore: bool (true if the matcher should retry with this feedback)
"""  # noqa: E501


class GraphState(TypedDict, total=False):
    markdown: str
    url: str
    title: str
    skip: bool
    match: dict[str, Any] | None
    critique: dict[str, Any] | None
    retries: int


async def _embed_query(client: httpx.AsyncClient, text: str) -> list[float]:
    cfg = get_config().embed
    prefixed = build_embed_query(text)
    resp = await client.post(
        f"{cfg.url}/embeddings",
        json={"model": cfg.model, "input": prefixed},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


def _apply_hard_constraints(match: dict[str, Any]) -> CriticReview:
    """Deterministic pre-check before calling the LLM Critic.

    Uses strict regex word boundaries (\\b) for title-level keywords
    applied ONLY to the role field, so 'reports to the Engineering Manager'
    in the JD body doesn't filter entry-level jobs.
    """
    role = str(match.get("role", "")).lower()
    jd_summary = str(match.get("jd_summary", "")).lower()

    _non_tech_pats = [
        r"\bcontent\s+creator\b",
        r"\bhost\s+live\b",
        r"\bsales\s+provider\b",
        r"\bsales\s+executive\b",
        r"\bsales\s+representative\b",
        r"\bproperty\s+development\b",
        r"\baccount\s+executive\b",
        r"\bmarketing\b",
        r"\brecruiter\b",
        r"\bcustomer\s+service\b",
        r"\bcustomer\s+support\b",
        r"\btelemarketing\b",
        r"\bsocial\s+media\b",
        r"\badministrative\s+assistant\b",
        r"\bstore\s+manager\b",
        r"\bcashier\b",
        r"\bdriver\b",
    ]
    combined = role + " " + jd_summary
    for pat in _non_tech_pats:
        if re.search(pat, combined):
            return CriticReview(
                passed=False,
                critique_reason="Role is non-technical — hard-constraint violation.",
                requires_rescore=False,
            )

    _title_pats = [
        r"\bsenior\b",
        r"\bsr\.?\b",
        r"\bstaff\b",
        r"\bmanager\b",
        r"\bdirector\b",
        r"\bvp\b",
        r"\bvice\s+president\b",
        r"\bhead\s+of\b",
        r"\barchitect\b",
        r"\bprincipal\b",
    ]
    for pat in _title_pats:
        if re.search(pat, role):
            return CriticReview(
                passed=False,
                critique_reason="Role title contains senior/leadership keyword — hard-constraint.",
                requires_rescore=False,
            )

    _exp_pats = [
        r"\b5\+?\s*years?\b",
        r"\b7\+?\s*years?\b",
        r"\b10\+?\s*years?\b",
        r"\b\d{2,}\s*\+\s*years?\b",
        r"\bph\.?d\b",
        r"\bdoctorate\b",
        r"\bpostdoc\b",
    ]
    for pat in _exp_pats:
        if re.search(pat, combined):
            return CriticReview(
                passed=False,
                critique_reason="JD requires 5+ years or PhD — hard-constraint violation.",
                requires_rescore=False,
            )

    return CriticReview(passed=True, critique_reason="Pre-checks passed", requires_rescore=False)


# Node implementations


async def node_context_builder(
    state: GraphState,
    store: MemoryStore,
    client: httpx.AsyncClient,
) -> GraphState:
    """Node 1: Deduplicate URL and build context from RAG memory."""
    url = state["url"]
    md = state["markdown"]

    cleaned = canonicalize_markdown(md, url)
    if cleaned is None:
        state["skip"] = True
        return state

    if await store.is_url_processed(url):
        state["skip"] = True
        return state

    state["skip"] = False
    state["markdown"] = cleaned

    with contextlib.suppress(Exception):
        await harvest_and_save_domains([url], store)

    try:
        query_emb = await _embed_query(client, cleaned[:8000])
        chunks = await store.search_similar_chunks(query_emb, top_k=5)
        state["_rag_chunks"] = chunks  # type: ignore[typeddict-unknown-key]
    except Exception:
        state["_rag_chunks"] = []  # type: ignore[typeddict-unknown-key]

    return state


async def node_matcher(
    state: GraphState,
    ctx: ContextManager,
    critique_feedback: str = "",
) -> GraphState:
    """Node 2: Call the LLM Matcher Agent."""
    chunks = cast(list[dict[str, Any]], state.get("_rag_chunks", []))
    jd_text = state["markdown"]

    if len(jd_text) > 80000:
        jd_text = jd_text[:40000] + "\n...\n" + jd_text[-40000:]

    relevant = (
        "\n".join(
            f"[{c['section']}] {c['content']}"
            for c in chunks  # type: ignore[index]
        )
        if chunks
        else "No relevant resume chunks found."
    )

    prompt = MATCHER_PROMPT.replace("{relevant_chunks}", relevant[:3000])
    prompt = prompt.replace("{job_description}", jd_text)

    if critique_feedback:
        prompt += (
            "\n\nPREVIOUS ATTEMPT WAS REJECTED by the critic. "
            f"Feedback: {critique_feedback}\n"
            "Please rescore carefully, addressing the issues raised."
        )

    try:
        result = await ctx.json_chat(prompt, JobMatch.model_json_schema())
    except Exception as e:
        logger.exception(
            "LLM failed for job",
            exc=e,
            source=state.get("title", state.get("url", "?")),
        )
        _queue_for_retry(state)
        state["match"] = None
        return state

    if not isinstance(result, dict) or "match_percent" not in result:
        state["match"] = None
        return state

    required = {"role", "company", "match_percent", "shortlist_probability"}
    if not required.issubset(result.keys()):
        state["match"] = None
        return state

    try:
        JobMatch.model_validate(result)
    except Exception:
        state["match"] = None
        return state

    raw_link = result.get("apply_link")
    if not raw_link or not str(raw_link).startswith("http"):
        result["apply_link"] = state["url"]
    result["url"] = state["url"]
    result["source_url"] = state["url"]

    state["match"] = result
    return state


async def node_critic(
    state: GraphState,
    ctx: ContextManager,
) -> GraphState:
    """Node 3: Verify match against hard constraints, escalate to LLM if needed."""
    match = state.get("match")
    if match is None:
        state["critique"] = {
            "passed": False,
            "critique_reason": "No match result to critique",
            "requires_rescore": False,
        }
        return state

    hard = _apply_hard_constraints(match)
    if not hard.passed:
        state["critique"] = hard.model_dump()
        return state

    prompt = CRITIC_PROMPT.replace("{result}", json.dumps(match, indent=2))
    prompt = prompt.replace("{job_description}", state["markdown"][:80000])

    result = await ctx.json_chat(prompt, CriticReview.model_json_schema())

    if isinstance(result, dict):
        state["critique"] = result
    else:
        state["critique"] = {
            "passed": False,
            "critique_reason": "Critic LLM returned unparseable output",
            "requires_rescore": False,
        }

    return state


async def node_memory_saver(
    state: GraphState,
    store: MemoryStore,
) -> GraphState:
    """Node 4: Persist final result to pgvector ledger."""
    match = state.get("match")
    if match is None:
        return state

    match["url"] = state["url"]
    match.setdefault("source_url", state["url"])
    if not match.get("apply_link"):
        match["apply_link"] = state["url"]

    await store.save_job_result(match)

    return state


# Graph runner


async def run_graph(
    state: GraphState,
    store: MemoryStore,
    embed_client: httpx.AsyncClient,
    ctx: ContextManager,
) -> dict[str, Any] | None:
    """Execute the full multi-agent pipeline on a single JD.

    Returns the final ``JobMatch`` dict, or ``None`` if rejected / already
    processed.
    """
    state.setdefault("retries", 0)

    state = await node_context_builder(state, store, embed_client)
    if state.get("skip", False):
        return None

    loop_count = 0
    critique_feedback = ""
    while True:
        state = await node_matcher(state, ctx, critique_feedback)
        if state.get("match") is None:
            return None

        state = await node_critic(state, ctx)
        critique = CriticReview.model_validate(state["critique"] or {})

        if critique.passed:
            break

        if not critique.requires_rescore or loop_count >= MAX_CORRECTION_LOOPS:
            state["match"] = None
            return None

        loop_count += 1
        critique_feedback = critique.critique_reason

    state = await node_memory_saver(state, store)

    return state.get("match")


async def run_batch(
    jd_batch: list[dict[str, str]],
    store: MemoryStore,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Run the graph pipeline concurrently over a batch of JDs.

    Each item must be a dict with keys ``markdown``, ``url``, and optionally
    ``title``.  A single shared ``httpx.AsyncClient`` and ``ContextManager``
    are used across the entire batch.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    ctx = ContextManager()
    cfg = get_config()

    async def _one(jd: dict[str, str]) -> None:
        async with sem:
            result = await run_graph(
                {
                    "markdown": jd["markdown"],
                    "url": jd["url"],
                    "title": jd.get("title", ""),
                    "skip": False,
                    "match": None,
                    "critique": None,
                    "retries": 0,
                },
                store,
                embed_client,
                ctx,
            )
            if result is not None:
                results.append(result)
                pct = result.get("match_percent", "?")
                verdict = result.get("verdict", "?")
                role = result.get("role", "?")
                company = result.get("company", "?")
                logger.info(
                    "Job matched",
                    extra={
                        "match_percent": pct,
                        "verdict": verdict,
                        "role": role,
                        "company": company,
                    },
                )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.embed.timeout, connect=10.0),
        limits=httpx.Limits(
            max_keepalive_connections=concurrency,
            max_connections=concurrency * 2,
        ),
    ) as embed_client:
        tasks = [asyncio.create_task(_one(jd)) for jd in jd_batch]
        await asyncio.gather(*tasks, return_exceptions=True)

    await ctx.aclose()

    results.sort(key=lambda j: j.get("match_percent", 0), reverse=True)
    return results
