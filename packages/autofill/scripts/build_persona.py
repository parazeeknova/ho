#!/usr/bin/env python3
"""Build the candidate persona from persona.json and index it into pgvector.

Renders each grilled Q&A into a retrievable chunk, embeds via the local
embedding server (EMBED_URL), indexes into ``persona_embeddings``, and writes
a clean ``persona.txt`` combining the persona answers with the resume summary.

Usage:
    uv run python scripts/build_persona.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # packages/autofill
REPO = ROOT.parent.parent  # repo root
for _p in (REPO, REPO / "packages" / "ingest", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv()

from src.configuration import get_config  # noqa: E402
from src.logging import get_logger  # noqa: E402
from src.memory.pgvector_store import MemoryStore  # noqa: E402

logger = get_logger("build_persona")

PERSONA_JSON = ROOT / "persona.json"
PERSONA_TXT = ROOT / "persona.txt"

# Resume sections to fold into persona.txt so the radar matcher keeps grounding.
_RESUME_SECTIONS = ("header", "skills", "experience", "education", "achievements")


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
    # Collapse markdown-table artifacts and long runs of whitespace.
    lines = [
        ln.strip(" |")
        for ln in text.splitlines()
        if ln.strip(" |") and not set(ln.strip()) <= set("-|")
    ]
    return " ".join(lines).strip()


async def resume_summary(store: MemoryStore) -> str:
    """Fetch resume_embeddings and produce a compact, de-duplicated summary."""
    cfg = get_config().embed
    parts: list[str] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        resp = await client.post(
            f"{cfg.url}/embeddings",
            json={"model": cfg.model, "input": ["resume summary"]},
        )
        resp.raise_for_status()
        emb = resp.json()["data"][0]["embedding"]
    seen: set[str] = set()
    for r in await store.search_similar_chunks(emb, top_k=40):
        sec = r["section"]
        if sec not in _RESUME_SECTIONS:
            continue
        cleaned = _clean(r["content"])
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        if sec == "skills":
            parts.append(f"- Skills: {cleaned}")
        elif sec == "experience":
            parts.append(f"- Experience: {cleaned}")
        elif sec == "education":
            parts.append(f"- Education: {cleaned}")
        elif sec == "achievements":
            parts.append(f"- Achievements: {cleaned}")
        else:
            parts.append(f"- {cleaned}")
    return "\n".join(parts[:40])


def render_persona_txt(
    answers: list[dict[str, str]], resume: str, identity: dict[str, str] | None = None
) -> str:
    blocks = ["Candidate Profile:"]
    for field, value in (identity or {}).items():
        if value:
            blocks.append(f"- {field}: {value}")
    for a in answers:
        blocks.append(f"- {a['question']}: {a['answer']}")
    if resume:
        blocks.append("")
        blocks.append("From Resume:")
        blocks.append(resume)
    return "\n".join(blocks) + "\n"


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

    ux.banner("PERSONA BUILDER", "persona.json  ->  persona_embeddings + persona.txt")
    answers = load_persona()
    identity = load_identity()
    chunks = render_chunks(answers, identity)
    ux.chip("info", f"Loaded {len(answers)} answers + {len(identity)} identity fields")

    store = await MemoryStore.create()
    try:
        with ux.console.status("Fetching resume summary...", spinner="dots"):
            resume = await resume_summary(store)

        persona_txt = render_persona_txt(answers, resume, identity)
        PERSONA_TXT.write_text(persona_txt)
        ux.chip("ok", f"Wrote {PERSONA_TXT} ({len(persona_txt)} chars)")

        records = await embed_chunks(chunks)
        await store.clear_persona()
        await store.index_persona_chunks(records)
        count = await store.persona_chunk_count()
        ux.chip("ok", f"Indexed {count} persona chunks into persona_embeddings")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
