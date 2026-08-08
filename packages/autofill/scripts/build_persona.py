#!/usr/bin/env python3
"""Build the candidate persona from persona.json and index it into pgvector.

Renders each grilled Q&A into a retrievable chunk, embeds via the local
embedding server (EMBED_URL), indexes into ``persona_embeddings``, and stores
the resume summary inside ``persona.json`` (``resume_summary``) so the single
persona file doubles as the radar matcher's grounding text.

Usage:
    uv run python scripts/build_persona.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # packages/autofill
REPO = ROOT.parent.parent  # repo root
for _p in (REPO, REPO / "packages" / "ingest", REPO / "packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv()
os.environ["LOG_LEVEL"] = "WARNING"  # quiet JSON log spam in setup scripts

from src.configuration import get_config  # noqa: E402
from src.logging import get_logger  # noqa: E402
from src.memory.pgvector_store import MemoryStore  # noqa: E402

logger = get_logger("build_persona")

PERSONA_JSON = REPO / "data" / "persona.json"  # repo root

# Resume sections to fold into the persona.json resume_summary field.
# NOTE: "projects" is critical — the resume PDF's biggest section. It was
# missing here, so every project bullet (Asocialmedia, Verso, Lumen, Chorus)
# was silently dropped from the matcher grounding, leaving the LLM scoring
# against a summary that described none of the candidate's actual work.
_RESUME_SECTIONS = ("header", "skills", "experience", "projects", "education", "achievements")


def load_persona(path: Path = PERSONA_JSON) -> list[dict[str, str]]:
    if not path.exists():
        sys.exit(
            f"persona.json not found at {path}. Run the grilling wizard first:\n"
            "    uv run python scripts/grill_persona.py"
        )
    data = json.loads(path.read_text())
    return data.get("answers", [])


def load_identity(path: Path = PERSONA_JSON) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"persona.json not found at {path}. Run the grilling wizard first.")
    data = json.loads(path.read_text())
    return data.get("identity", {})


def render_chunks(
    answers: list[dict[str, str]], identity: dict[str, str] | None = None
) -> list[dict[str, str]]:
    """Turn each Q&A (and identity field) into a retrievable text chunk."""
    chunks: list[dict[str, str]] = []
    for a in answers:
        content = f"Q: {a['question']}\nA: {a['answer']}"
        chunks.append(
            {
                "category": a["category"],
                "question": a["question"],
                "answer": a["answer"],
                "content": content,
            }
        )
    for field, value in (identity or {}).items():
        if not value:
            continue
        question = f"What is the candidate's {field}?"
        chunks.append(
            {
                "category": "identity",
                "question": question,
                "answer": str(value),
                "content": f"Q: {question}\nA: {value}",
            }
        )
    return chunks


def _clean(text: str) -> str:
    # Collapse markdown-table artifacts and long runs of whitespace. Table
    # separator rows are dropped, cell pipes are flattened inside a line, and
    # dashed separator runs are collapsed (resume PDFs convert to tables).
    lines = [
        ln.strip(" |")
        for ln in text.splitlines()
        if ln.strip(" |") and not set(ln.strip()) <= set("-|")
    ]
    joined = " ".join(lines)
    joined = re.sub(r"\s*\|\s*", " ", joined)
    joined = re.sub(r"\s*-{3,}\s*", " ", joined)
    return re.sub(r"\s{2,}", " ", joined).strip()


async def resume_summary(store: MemoryStore) -> str:
    """Fetch resume_embeddings and produce a compact, de-duplicated summary.

    Retrieves chunks by SECTION rather than one semantic query, so every
    project/experience bullet is captured — a single "resume summary" query
    only surfaces chunks semantically close to that phrase (the heading and
    intro), silently dropping the specific project bullets the matcher needs.
    """
    cfg = get_config().embed
    parts: list[str] = []
    seen: set[str] = set()
    # Query per section so the retrieval covers all of them, not just the
    # chunks closest to "resume summary".
    section_queries = {
        "header": "candidate contact information name",
        "skills": "candidate technical skills technologies",
        "experience": "candidate work experience roles employers",
        "projects": "candidate projects built products",
        "education": "candidate education university degree",
        "achievements": "candidate achievements awards publications hackathons",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        for sec, query in section_queries.items():
            resp = await client.post(
                f"{cfg.url}/embeddings",
                json={"model": cfg.model, "input": [query]},
            )
            resp.raise_for_status()
            emb = resp.json()["data"][0]["embedding"]
            for r in await store.search_similar_chunks(emb, top_k=20):
                rsec = r["section"]
                if rsec != sec or rsec not in _RESUME_SECTIONS:
                    continue
                cleaned = _clean(r["content"])
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                if rsec == "skills":
                    parts.append(f"- Skills: {cleaned}")
                elif rsec == "experience":
                    parts.append(f"- Experience: {cleaned}")
                elif rsec == "education":
                    parts.append(f"- Education: {cleaned}")
                elif rsec == "achievements":
                    parts.append(f"- Achievements: {cleaned}")
                else:
                    parts.append(f"- {cleaned}")
    return "\n".join(parts[:60])


async def embed_chunks(
    chunks: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Embed each chunk content via the local embedding server."""
    cfg = get_config().embed
    records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
    ) as client:
        for chunk in chunks:
            resp = await client.post(
                f"{cfg.url}/embeddings",
                json={"model": cfg.model, "input": [chunk["content"]]},
            )
            resp.raise_for_status()
            records.append(
                {
                    "category": chunk["category"],
                    "question": chunk["question"],
                    "answer": chunk["answer"],
                    "content": chunk["content"],
                    "embedding": resp.json()["data"][0]["embedding"],
                }
            )
    return records


async def main() -> None:
    import ux

    ux.chip("info", "PERSONA BUILDER  -  persona.json -> persona_embeddings + resume_summary")
    answers = load_persona()
    identity = load_identity()
    chunks = render_chunks(answers, identity)
    ux.chip("info", f"Loaded {len(answers)} answers + {len(identity)} identity fields")

    store = await MemoryStore.create()
    try:
        with ux.console.status("Fetching resume summary...", spinner="dots"):
            resume = await resume_summary(store)

        persona = json.loads(PERSONA_JSON.read_text())
        persona["resume_summary"] = resume
        tmp = PERSONA_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(persona, indent=2) + "\n")
        os.replace(tmp, PERSONA_JSON)
        ux.chip("ok", f"Stored resume_summary in {PERSONA_JSON} ({len(resume)} chars)")

        records = await embed_chunks(chunks)
        await store.clear_persona()
        await store.index_persona_chunks(records)
        count = await store.persona_chunk_count()
        ux.chip("ok", f"Indexed {count} persona chunks into persona_embeddings")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
