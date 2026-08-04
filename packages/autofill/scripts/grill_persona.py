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
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # packages/autofill
REPO = ROOT.parent.parent  # repo root
for _p in (REPO, REPO / "packages" / "ingest", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ux  # noqa: E402

os.environ["LOG_LEVEL"] = "WARNING"  # quiet JSON log spam in setup scripts
from rich.prompt import Prompt  # noqa: E402
from src.logging import get_logger  # noqa: E402

logger = get_logger("grill_persona")

PERSONA_JSON = ROOT / "data" / "persona.json"

CONTACT_FIELDS = ("email", "phone", "linkedin", "github", "website", "twitter")
LINK_FIELDS = {"linkedin", "github", "website", "twitter"}

WIZARD_QUESTIONS: list[tuple[str, str]] = [
    ("current_location", "Where are you currently based?"),
    (
        "work_model",
        "How do you prefer to work? (remote / hybrid / onsite)",
    ),
    ("relocation", "Are you open to relocating? Any regions you'd avoid?"),
    ("nationality", "What is your nationality?"),
    ("work_authorization", "Are you legally authorized to work in India?"),
    ("work_authorization", "Are you legally authorized to work in the United States?"),
    ("visa_sponsorship", "Would you need visa sponsorship for roles abroad?"),
    (
        "gender_identity",
        "What is your gender? (or say Prefer not to answer)",
    ),
    ("pronouns", "What are your pronouns? (or say N/A)"),
    (
        "sexual_orientation",
        "What is your sexual orientation? "
        "(heterosexual / homosexual / bisexual / pansexual / asexual / prefer not to answer)",
    ),
    (
        "ethnicity",
        "What is your ethnicity / race? (or say Prefer not to answer)",
    ),
    (
        "disability",
        "Do you have a disability? (Yes / No / Prefer not to answer)",
    ),
    (
        "veteran_status",
        "Are you a veteran of any armed forces? (Yes / No / Prefer not to answer)",
    ),
    (
        "employee_relation",
        "Are you related to anyone employed at the company? (Yes / No)",
    ),
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


_YES_ALIASES = {"y", "yes", "yeah", "yep", "sure"}
_NO_ALIASES = {"n", "no", "nope", "nah"}
_PREFER_NOT_ALIASES = {
    "prefer not to answer",
    "prefer not to say",
    "prefer not",
    "prefer not say",
    "decline",
    "decline to answer",
    "decline to state",
    "skip",
    "pnta",
    "-",
}
_NA_ALIASES = {"n/a", "na", "not applicable", "none", "--"}


def _normalize_answer(value: str, category: str = "") -> str:
    """Sanitize a grilled answer before it lands in persona.json.

    Collapses internal whitespace and maps common phrasings to canonical
    values: yes/no aliases -> "Yes"/"No", "prefer not to say" family ->
    "Prefer not to answer", "n/a" family -> "N/A". Free-text answers
    (compensation, availability, working hours) pass through cleanly.
    """
    v = re.sub(r"\s+", " ", (value or "")).strip()
    if not v:
        return v
    low = v.lower().strip(" .")
    if low in _PREFER_NOT_ALIASES:
        return "Prefer not to answer"
    if low in _YES_ALIASES:
        return "Yes"
    if low in _NO_ALIASES:
        return "No"
    if low in _NA_ALIASES:
        return "N/A"
    return v


async def _resume_defaults() -> dict[str, str]:
    """Extract identity defaults from the indexed resume header, if any."""
    try:
        from autofill.profile import _regex_extract
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        try:
            if not await store.chunk_count():
                return {}
            # Fetch the header section directly: an embedding-similarity search
            # ranks the sparse name row out of the top hits, but the name must
            # never be missed.
            async with store._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT content FROM resume_embeddings
                    WHERE section = 'header'
                    ORDER BY id
                    LIMIT 30
                    """
                )
            parts = [r["content"].strip() for r in rows if r.get("content", "").strip()]
            if not parts:
                return {}
            joined = "\n".join(parts)
            found = _regex_extract(joined)
            found["name"] = _extract_name(joined)
            return found
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


def _sanitize_link(value: str) -> str:
    """Normalize a contact link to an absolute https:// URL.

    The TS ProfileSchema validates links as URLs, and a bare
    ``linkedin.com/in/...`` fails ``z.string().url()`` — so anything without
    a scheme gets ``https://`` prepended (the old value is already absolute).
    """
    v = value.strip()
    if v and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", v):
        v = "https://" + v
    return v


def _extract_name(text: str) -> str:
    """Pull the full name from the resume header.

    Scans for a name-shaped line: 2+ alphabetic words (hyphens/apostrophes
    allowed, no digits/dots/slashes/@), first word capitalised — so the
    contact row, tagline and bullet lines never match. Handles both plain
    (``Harsh Sahu``) and markdown-table (``| Harsh | Sahu |``) header rows.
    """
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        words = " ".join(cells).split()
        if len(words) < 2:
            continue
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", w) for w in words[:3]):
            continue
        if not (words[0][0].isupper() and words[1][0].isupper()):
            continue
        return " ".join(words[:3]).strip()
    return ""


def identity_mismatches(saved: dict, resume: dict) -> list[tuple[str, str, str]]:
    """Return (field, saved, resume) for saved values differing from resume extraction."""
    out: list[tuple[str, str, str]] = []
    for field, resume_value in resume.items():
        saved_value = (saved.get(field) or "").strip()
        if saved_value and _norm(saved_value) != _norm(resume_value):
            out.append((field, saved_value, resume_value))
    return out


def grill_identity(data: dict, resume: dict, ask_all: bool = False) -> dict:
    identity = dict(data.get("identity", {}))
    ux.section(1, 3, "Identity & Contact", "Enter keeps the current value, if any")

    # Full name comes from the resume (or the existing persona) — never ask
    # for first/last name separately unless nothing is known at all.
    existing_name = (
        data.get("name", "")
        or " ".join(
            filter(None, (identity.get("firstName", ""), identity.get("lastName", "")))
        ).strip()
    )
    resume_name = resume.get("name", "")
    name = ""
    if ask_all or not (existing_name or resume_name):
        default_name = existing_name or resume_name
        label = "full name (from resume)" if resume_name and not existing_name else "full name"
        name = _ask(label, default_name)
    if name:
        name = _normalize_answer(name)
        data["name"] = name
        parts = name.split(None, 1)
        identity["firstName"] = parts[0]
        identity["lastName"] = parts[1] if len(parts) > 1 else ""

    for field in CONTACT_FIELDS:
        existing = identity.get(field, "")
        if existing and not ask_all:
            continue
        default = existing or resume.get(field, "")
        label = f"{field} (from resume)" if default and not existing else field
        value = _ask(label, default)
        if value:
            if field in LINK_FIELDS:
                value = _sanitize_link(value)
            elif field == "email":
                value = _normalize_answer(value).lower()
            else:
                value = _normalize_answer(value)
            identity[field] = value

    # Nothing known about the name anywhere: ask explicitly.
    if not name and not identity.get("firstName"):
        first = _ask("first name")
        if first:
            identity["firstName"] = first
        last = _ask("last name")
        if last:
            identity["lastName"] = last

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


def grill_answers(data: dict, ask_all: bool = False) -> dict:
    stored = data.get("answers", [])
    by_question = {a["question"]: a for a in stored}
    by_category: dict[str, list[dict]] = {}
    for a in stored:
        by_category.setdefault(a["category"], []).append(a)

    ux.section(2, 3, "Personal Q&A", "Free-form answers; dropdowns get closest option")
    answers: list[dict] = []
    kept: list[dict] = []
    seen: set[str] = set()
    skipped = 0
    for category, question in WIZARD_QUESTIONS:
        # Re-grills only ask what is still missing: a question is skipped when
        # it already has an answer (or its single-question category does),
        # unless --all forces a full re-ask. Skipped questions keep their
        # stored entries — the saved list must NEVER shrink from a skip.
        multi = sum(1 for c, _ in WIZARD_QUESTIONS if c == category) > 1
        already = question in by_question or (not multi and bool(by_category.get(category)))
        if already and not ask_all:
            skipped += 1
            entry = by_question.get(question) or (by_category.get(category) or [None])[0]
            if entry and entry not in kept:
                kept.append(entry)
            continue
        prev = _previous_answer(category, question, by_question, by_category)
        answer = _ask(question, prev.get("answer", ""))
        if not answer:
            continue
        answer = _normalize_answer(answer, category)
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

    wizard_cats = {c for c, _ in WIZARD_QUESTIONS}
    extra_kept = [a for a in stored if a.get("category") not in wizard_cats and a not in kept]
    data["answers"] = kept + extra_kept + answers
    if skipped and not ask_all:
        ux.chip("info", f"Skipped {skipped} already-answered question(s) (--all to re-ask)")
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-ask every question even when already answered",
    )
    args = parser.parse_args()

    ux.banner("CANDIDATE PERSONA GRILL", "interactive wizard  ·  identity + personal Q&A")
    data = load_existing()
    resume = asyncio.run(_resume_defaults())
    if resume:
        ux.chip("info", f"Prefilling identity defaults from resume ({len(resume)} fields)")
    data = grill_identity(data, resume, ask_all=args.all)
    data = grill_answers(data, ask_all=args.all)
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
