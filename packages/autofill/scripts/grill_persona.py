#!/usr/bin/env python3
"""Interactive wizard that builds the candidate persona and indexes it.

Asks for identity/contact fields and the grilled personal Q&A (with the
existing persona.json values shown as defaults), writes persona.json
atomically, then rebuilds the persona memory (persona_embeddings +
resume_summary) via build_persona.py.

On a fresh setup (no persona.json) the identity fields are pre-filled from
the indexed resume header when available, and saved values that differ from
resume extraction are flagged as warnings.

Usage:
    uv run python scripts/grill_persona.py
    uv run python scripts/grill_persona.py --no-build   # only write persona.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # packages/autofill
REPO = ROOT.parent.parent  # repo root
for _p in (REPO, REPO / "packages" / "ingest", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ux  # noqa: E402
from rich.prompt import Prompt  # noqa: E402
from src.logging import get_logger  # noqa: E402

logger = get_logger("grill_persona")

PERSONA_JSON = ROOT / "data" / "persona.json"

IDENTITY_FIELDS = (
    "firstName",
    "lastName",
    "email",
    "phone",
    "linkedin",
    "github",
    "website",
)

WIZARD_QUESTIONS: list[tuple[str, str]] = [
    ("current_location", "Where are you currently based?"),
    ("work_model", "How do you prefer to work - remote, hybrid, or onsite?"),
    ("relocation", "Are you open to relocating? Any regions you'd avoid?"),
    ("nationality", "What is your nationality?"),
    ("work_authorization", "Are you legally authorized to work in India?"),
    ("work_authorization", "Are you legally authorized to work in the United States?"),
    ("visa_sponsorship", "Would you need visa sponsorship for roles abroad?"),
    ("current_compensation", "What is your current compensation? (or say N/A if not earning)"),
    ("expected_compensation", "What are your salary expectations? (include currency)"),
    ("equity", "Do you currently hold any equity, RSUs, or ESOPs?"),
    ("availability", "How soon can you start if selected?"),
    ("working_hours", "How many hours per week are you available?"),
]


def load_existing() -> dict:
    if PERSONA_JSON.exists():
        try:
            return json.loads(PERSONA_JSON.read_text())
        except json.JSONDecodeError:
            ux.chip("warn", f"{PERSONA_JSON} is corrupt; starting fresh.")
    return {"name": "", "version": 1, "identity": {}, "answers": []}


def _ask(label: str, default: str = "") -> str:
    try:
        value = Prompt.ask(f"  {label}", default=default) if default else Prompt.ask(f"  {label}")
    except EOFError, KeyboardInterrupt:
        return default
    return value.strip()


async def _resume_defaults() -> dict[str, str]:
    """Extract identity defaults from the indexed resume header, if any."""
    try:
        from autofill.profile import _regex_extract
        from autofill.rag import _embed_text
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        try:
            if not await store.chunk_count():
                return {}
            emb = await _embed_text("candidate contact information resume header")
            if not emb:
                return {}
            rows = await store.search_similar_chunks(emb, top_k=8)
            parts = [
                r["content"].strip()
                for r in rows
                if r.get("section") == "header" and r.get("content", "").strip()
            ]
            if not parts:
                return {}
            return _regex_extract(" | ".join(parts))
        finally:
            await store.close()
    except Exception as e:
        logger.warning("Resume prefill unavailable; falling back to blank defaults", error=str(e))
        return {}


def _norm(value: str) -> str:
    v = value.strip().lower()
    for prefix in ("https://", "http://", "www."):
        if v.startswith(prefix):
            v = v[len(prefix) :]
    return v


def identity_mismatches(saved: dict, resume: dict) -> list[tuple[str, str, str]]:
    """Return (field, saved, resume) for saved values differing from resume extraction."""
    out: list[tuple[str, str, str]] = []
    for field, resume_value in resume.items():
        saved_value = (saved.get(field) or "").strip()
        if saved_value and _norm(saved_value) != _norm(resume_value):
            out.append((field, saved_value, resume_value))
    return out


def grill_identity(data: dict, resume: dict) -> dict:
    identity = dict(data.get("identity", {}))
    ux.section(1, 3, "Identity & Contact", "Enter keeps the current value, if any")
    for field in IDENTITY_FIELDS:
        existing = identity.get(field, "")
        default = existing or resume.get(field, "")
        label = f"{field} (from resume)" if default and not existing else field
        value = _ask(label, default)
        if value:
            identity[field] = value
    name = _ask("full name", data.get("name", "") or identity.get("firstName", ""))
    if name:
        data["name"] = name
    data["identity"] = identity
    return data


def _previous_answer(
    category: str,
    question: str,
    by_question: dict[str, dict],
    by_category: dict[str, list[dict]],
) -> dict:
    """Find the existing answer to use as a default.

    Matches the exact question text first; falls back to the category when
    the wizard phrasing differs from the stored question (and the category
    maps to a single stored answer).
    """
    if question in by_question:
        return by_question[question]
    matches = by_category.get(category, [])
    if len(matches) == 1:
        return matches[0]
    return {}


def grill_answers(data: dict) -> dict:
    stored = data.get("answers", [])
    by_question = {a["question"]: a for a in stored}
    by_category: dict[str, list[dict]] = {}
    for a in stored:
        by_category.setdefault(a["category"], []).append(a)

    ux.section(2, 3, "Personal Q&A", "Free-form answers; dropdowns get closest option")
    answers: list[dict] = []
    seen: set[str] = set()
    for category, question in WIZARD_QUESTIONS:
        prev = _previous_answer(category, question, by_question, by_category)
        answer = _ask(question, prev.get("answer", ""))
        if not answer:
            continue
        final_question = (
            prev.get("question", question) if answer == prev.get("answer", "") else question
        )
        if final_question in seen:
            continue
        answers.append({"category": category, "question": final_question, "answer": answer})
        seen.add(final_question)

    ux.section(3, 3, "Extra Q&A", "optional - skip with Enter")
    while True:
        question = _ask("question (Enter to finish)")
        if not question:
            break
        answer = _ask("answer")
        if answer:
            answers.append({"category": "general", "question": question, "answer": answer})

    data["answers"] = answers
    return data


def save_persona(data: dict) -> None:
    data["version"] = int(data.get("version", 1)) + 1
    tmp = PERSONA_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, PERSONA_JSON)
    ux.chip("ok", f"Wrote {PERSONA_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the candidate persona interactively.")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Write persona.json without rebuilding memory",
    )
    args = parser.parse_args()

    ux.banner("CANDIDATE PERSONA GRILL", "interactive wizard  ·  identity + personal Q&A")
    data = load_existing()
    resume = asyncio.run(_resume_defaults())
    if resume:
        ux.chip("info", f"Prefilling identity defaults from resume ({len(resume)} fields)")
    data = grill_identity(data, resume)
    data = grill_answers(data)
    save_persona(data)

    for field, saved_value, resume_value in identity_mismatches(data.get("identity", {}), resume):
        ux.chip(
            "warn",
            f"identity.{field} '{saved_value}' differs from resume '{resume_value}'",
        )

    if not data["answers"] and not data["identity"]:
        ux.chip("info", "Nothing entered; persona.json left as-is.")
        return

    if args.no_build:
        ux.chip(
            "info",
            "Skipping memory build (--no-build). "
            "Run `uv run python scripts/build_persona.py` later.",
        )
        return

    build = ROOT / "scripts" / "build_persona.py"
    ux.divider()
    with ux.console.status("Rebuilding persona memory...", spinner="dots"):
        result = subprocess.run([sys.executable, str(build)], cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
