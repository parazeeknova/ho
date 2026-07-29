"""Multi-agent graph topology for job matching with self-correction.

Pipeline (sequential state machine):

  Node 1  –  Context Builder   (deterministic: dedup + RAG retrieval)
  Node 2  –  Matcher Agent     (LLM: Qwen3.5-4B with structured output)
  Node 3  –  Critic Agent      (LLM + hard-constraint rules)
  Edge    –  Self-correction    (route back to Matcher if needed, max 2 retries)
  Node 4  –  Memory Saver      (persist to pgvector + jobs.md)

The pipeline is a single async function ``run_graph(...)`` that walks the
nodes and returns the final ``JobMatch`` result (or ``None`` when the URL
was already processed or the job fails all gates).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypedDict, cast

import httpx

from src.llm.config import build_embed_query
from src.llm.schemas import CriticReview, JobMatch, canonicalize_markdown
from src.memory.pgvector_store import MemoryStore

# ── LLM endpoints ──────────────────────────────────────────────────────────

LLM_URL = "http://127.0.0.1:8899"
LLM_MODEL = "Qwen/Qwen3.5-4B"
MAX_RETRIES = 3
RETRY_DELAY = 4

EMBED_URL = "http://127.0.0.1:8900"

MAX_CORRECTION_LOOPS = 2

MATCHER_PROMPT = """\
You are a job-resume matching engine. Compare this job description against the \
RELEVANT RESUME SNIPPETS below (these are only the most relevant parts of the \
resume, not the full resume).

Relevant resume snippets:
{relevant_chunks}

Full job listing:
{job_description}

CRITICAL RULES:
- If the text is a company homepage, job directory, error page, or lists \
multiple different jobs instead of ONE SINGLE posting, set match_percent=0 and \
verdict=NO_MATCH.
- The candidate is early-career / new-grad / intern based in India. Accept \
remote roles worldwide, onsite roles in India, and roles with visa sponsorship.
- If salary is specified below 70K INR/month (or equivalent), set match_percent=0 \
and verdict=NO_MATCH.
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

Return a JSON object with:
- passed: bool (true if the match passes all hard constraints)
- critique_reason: string (explain what failed, or "All checks passed")
- requires_rescore: bool (true if the matcher should retry with this feedback)
"""  # noqa: E501


class GraphState(TypedDict, total=False):
    markdown: str
    url: str
    title: str
    match: dict[str, Any] | None
    critique: dict[str, Any] | None
    retries: int


async def _llm_chat(
    client: httpx.AsyncClient,
    prompt: str,
    schema: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_object",
            "schema": schema,
        }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.post(f"{LLM_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"].get("content", "")
        except Exception:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
    raise RuntimeError("LLM failed after all retries")


async def _embed_query(client: httpx.AsyncClient, text: str) -> list[float]:
    """Generate an embedding for a retrieval query using the embedding model."""
    prefixed = build_embed_query(text)
    resp = await client.post(
        f"{EMBED_URL}/v1/embeddings",
        json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": prefixed},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


async def _embed_chunks(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for resume chunks (no instruction prefix)."""
    resp = await client.post(
        f"{EMBED_URL}/v1/embeddings",
        json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": texts},
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


# ── Schema for structured LLM output ───────────────────────────────────────

MATCH_SCHEMA_GRAPH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "company": {"type": "string"},
        "match_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "shortlist_probability": {"type": "integer", "minimum": 0, "maximum": 100},
        "matching_skills": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "jd_summary": {"type": "string"},
        "salary": {"type": ["string", "null"]},
        "location": {"type": "string"},
        "is_remote": {"type": "boolean"},
        "verdict": {
            "type": "string",
            "enum": ["STRONG_MATCH", "GOOD_MATCH", "WEAK_MATCH", "NO_MATCH"],
        },
    },
    "required": [
        "role",
        "company",
        "match_percent",
        "shortlist_probability",
        "matching_skills",
        "missing_skills",
        "jd_summary",
        "location",
        "is_remote",
        "verdict",
    ],
}

CRITIC_SCHEMA_GRAPH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "critique_reason": {"type": "string"},
        "requires_rescore": {"type": "boolean"},
    },
    "required": ["passed", "critique_reason", "requires_rescore"],
}


def _strip_markdown_code(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
    return raw.strip()


def _apply_hard_constraints(match: dict[str, Any]) -> CriticReview:
    """Deterministic pre-check before calling the LLM Critic."""
    role = str(match.get("role", "")).lower()
    jd_summary = str(match.get("jd_summary", "")).lower()
    text = role + " " + jd_summary

    senior_kws = (
        "senior",
        "sr.",
        "staff ",
        "lead ",
        "principal",
        "architect",
        "manager",
        "director",
        "head of",
        "vp ",
        "vice president",
        "5+ year",
        "7+ year",
        "10+ year",
    )
    if any(kw in text for kw in senior_kws):
        return CriticReview(
            passed=False,
            critique_reason="Role or JD contains senior/leadership keywords — "
            "hard-constraint violation.",
            requires_rescore=False,
        )

    return CriticReview(passed=True, critique_reason="Pre-checks passed", requires_rescore=False)


# ── Node implementations ───────────────────────────────────────────────────


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
        state["match"] = None
        return state

    if await store.is_url_processed(url):
        state["match"] = None
        return state

    state["markdown"] = cleaned

    try:
        query_emb = await _embed_query(client, cleaned[:8000])
        chunks = await store.search_similar_chunks(query_emb, top_k=5)
        state["_rag_chunks"] = chunks  # type: ignore[typeddict-unknown-key]
    except Exception:
        state["_rag_chunks"] = []  # type: ignore[typeddict-unknown-key]

    return state


async def node_matcher(
    state: GraphState,
    client: httpx.AsyncClient,
    critique_feedback: str = "",
) -> GraphState:
    """Node 2: Call the LLM Matcher Agent."""
    chunks = cast(list[dict[str, Any]], state.get("_rag_chunks", []))
    jd_text = state["markdown"]

    relevant = (
        "\n".join(
            f"[{c['section']}] {c['content']}"
            for c in chunks  # type: ignore[index]
        )
        if chunks
        else "No relevant resume chunks found."
    )

    prompt = MATCHER_PROMPT.replace("{relevant_chunks}", relevant[:3000])
    prompt = prompt.replace("{job_description}", jd_text[:5000])

    if critique_feedback:
        prompt += (
            "\n\nPREVIOUS ATTEMPT WAS REJECTED by the critic. "
            f"Feedback: {critique_feedback}\n"
            "Please rescore carefully, addressing the issues raised."
        )

    raw = await _llm_chat(client, prompt, MATCH_SCHEMA_GRAPH)
    raw = _strip_markdown_code(raw)
    try:
        match = json.loads(raw)
    except json.JSONDecodeError:
        state["match"] = None
        return state

    if not isinstance(match, dict) or "match_percent" not in match:
        state["match"] = None
        return state

    required = {"role", "company", "match_percent", "shortlist_probability"}
    if not required.issubset(match.keys()):
        state["match"] = None
        return state

    try:
        JobMatch.model_validate(match)
    except Exception:
        state["match"] = None
        return state

    state["match"] = match
    return state


async def node_critic(
    state: GraphState,
    client: httpx.AsyncClient,
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
    prompt = prompt.replace("{job_description}", state["markdown"][:3000])

    raw = await _llm_chat(client, prompt, CRITIC_SCHEMA_GRAPH)
    raw = _strip_markdown_code(raw)
    try:
        critique = json.loads(raw)
        state["critique"] = critique
    except json.JSONDecodeError:
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
    """Node 4: Persist final result to pgvector ledger and jobs.md."""
    match = state.get("match")
    if match is None:
        return state

    match["url"] = state["url"]

    await store.save_job_result(match)

    _append_to_jobs_md(match)

    return state


def _append_to_jobs_md(match: dict[str, Any]) -> None:
    """Append a single match to jobs.md markdown table."""
    import os
    from datetime import UTC, datetime

    path = os.path.join(os.path.dirname(__file__), "..", "..", "jobs.md")

    try:
        with open(path) as fh:
            existing = fh.read()
    except FileNotFoundError:
        existing = ""

    lines: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    in_table = False

    for line in existing.splitlines():
        if line.startswith("| # | Role"):
            in_table = True
        elif in_table and line.startswith("|") and line.count("|") >= 8:
            lines.append(line)
        elif in_table and not line.startswith("|"):
            in_table = False

    role = match.get("role", "Unknown")
    company = match.get("company", "Unknown")
    match_pct = match.get("match_percent", 0)
    shortlist = match.get("shortlist_probability", 0)
    salary = match.get("salary") or "-"
    location = match.get("location", "-")
    apply_link = match.get("url", "")

    idx = len(lines) + 1
    new_row = (
        f"| {idx} | {role} | {company} | {match_pct}% | {shortlist}% "
        f"| {salary} | {now} | {location} | [Apply]({apply_link}) |"
    )
    lines.append(new_row)

    header = (
        "# Job Matches\n\n"
        f"Generated: {now}\n\n"
        "| # | Role | Company | JD Match | Shortlist% | Salary | Posted | Location | Apply |\n"
        "|---|------|---------|----------|------------|--------|--------|----------|-------|\n"
    )
    with open(path, "w") as fh:
        fh.write(header + "\n".join(lines) + "\n\n")


# ── Graph runner ────────────────────────────────────────────────────────────


async def run_graph(
    state: GraphState,
    store: MemoryStore,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any] | None:
    """Execute the full multi-agent pipeline on a single JD and return the
    final ``JobMatch`` dict, or ``None`` if rejected / already processed.

    Parameters
    ----------
    state:
        Must include at least ``markdown``, ``url``, and optionally ``title``.
    store:
        An initialised ``MemoryStore`` connected to the agent-memory db.
    client:
        A shared ``httpx.AsyncClient``.  One is created internally if not
        supplied (useful for single-shot invocation).
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    assert client is not None

    try:
        state.setdefault("retries", 0)

        # Node 1: Context Builder
        state = await node_context_builder(state, store, client)
        if state.get("match") is None and state.get("_rag_chunks") == []:  # type: ignore[comparison-overlap]
            return None

        # Node 2 + 3 with self-correction loop
        loop_count = 0
        critique_feedback = ""
        while True:
            state = await node_matcher(state, client, critique_feedback)
            if state.get("match") is None:
                return None

            state = await node_critic(state, client)
            critique = CriticReview.model_validate(state["critique"] or {})

            if critique.passed:
                break

            if not critique.requires_rescore or loop_count >= MAX_CORRECTION_LOOPS:
                state["match"] = None
                return None

            loop_count += 1
            critique_feedback = critique.critique_reason

        # Node 4: Memory Saver
        state = await node_memory_saver(state, store)

        return state.get("match")
    finally:
        if own_client and client is not None:
            await client.aclose()


async def run_batch(
    jd_batch: list[dict[str, str]],
    store: MemoryStore,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Run the graph pipeline concurrently over a batch of JDs.

    Each item must be a dict with keys ``markdown``, ``url``, and optionally
    ``title``.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async def _one(jd: dict[str, str]) -> None:
        async with (
            sem,
            httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=1, max_connections=1),
            ) as client,
        ):
            result = await run_graph(
                {
                    "markdown": jd["markdown"],
                    "url": jd["url"],
                    "title": jd.get("title", ""),
                    "match": None,
                    "critique": None,
                    "retries": 0,
                },
                store,
                client,
            )
            if result is not None:
                results.append(result)
                pct = result.get("match_percent", "?")
                verdict = result.get("verdict", "?")
                role = result.get("role", "?")
                company = result.get("company", "?")
                print(f"    {pct}% | {verdict} | {role} @ {company}")

    tasks = [asyncio.create_task(_one(jd)) for jd in jd_batch]
    await asyncio.gather(*tasks, return_exceptions=True)

    results.sort(key=lambda j: j.get("match_percent", 0), reverse=True)
    return results
