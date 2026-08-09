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
from typing import Any

ROOT = Path(__file__).resolve().parent.parent  # packages/autofill
REPO = ROOT.parent.parent  # repo root
for _p in (REPO, REPO / "packages" / "ingest", REPO / "packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ux  # noqa: E402

os.environ["LOG_LEVEL"] = "WARNING"  # quiet JSON log spam in setup scripts
from rich.prompt import Prompt  # noqa: E402
from src.logging import get_logger  # noqa: E402

logger = get_logger("grill_persona")

PERSONA_JSON = REPO / "data" / "persona.json"  # repo root

CONTACT_FIELDS = ("email", "phone", "linkedin", "github", "website", "twitter")
LINK_FIELDS = {"linkedin", "github", "website", "twitter"}

# Core application-form questions every candidate must answer. The dynamic
# generator (generate_dynamic_questions) appends candidate-specific ones on
# top of these.
# Best-effort LLM follow-up questions: the interactive grill must never block
# minutes on a slow/overloaded model just to invent optional questions.
_DYN_QUESTION_TIMEOUT_S = 90.0

# Auto-prefill sources populated in main() before the grill runs: identity-like
# facts extracted from the resume and the full resume+portfolio text blob.
_AUTO_RESUME: dict[str, str] = {}
_AUTO_RESUME_CTX: str = ""

CORE_QUESTIONS: list[tuple[str, str]] = [
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
    (
        "education",
        "What is your highest level of education (degree + field of study)?",
    ),
    (
        "education",
        "What university / college did you attend, and when do/did you graduate?",
    ),
    (
        "ai_experience",
        "Have you built or integrated any AI / LLM APIs (such as OpenAI, "
        "Claude, etc.) into an application?",
    ),
    (
        "ai_experience",
        "Which backend language are you strongest in?",
    ),
    (
        "projects",
        "Briefly describe a recent project you're most proud of.",
    ),
]


def load_existing() -> dict:
    if PERSONA_JSON.exists():
        try:
            return json.loads(PERSONA_JSON.read_text())
        except json.JSONDecodeError:
            ux.chip("warn", f"{PERSONA_JSON} is corrupt; starting fresh.")
    return {"name": "", "version": 1, "identity": {}, "answers": []}


def _ask(label: str, default: str = "") -> str:
    # LLM-generated questions sometimes end in '.', '?' or '!' — strip trailing
    # sentence punctuation so "proud of." + the appended ": " never renders as
    # "proud of.: ". Static labels never end in punctuation, so this is safe.
    label = re.sub(r"[.!?]+\s*$", "", label).strip()
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


async def _resume_context() -> str:
    """Full resume + portfolio context (all sections) for grounding the dynamic
    question generator and auto-prefilling answers. Returns '' if unavailable."""
    try:
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        try:
            if not await store.chunk_count():
                return ""
            async with store._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT section, content FROM resume_embeddings
                    WHERE section IN ('header', 'skills', 'projects', 'experience',
                                      'education', 'portfolio')
                    ORDER BY id
                    """
                )
            blocks: list[str] = []
            current: dict[str, list[str]] = {}
            order = ["header", "skills", "experience", "education", "projects", "portfolio"]
            for r in rows:
                sec = str(r.get("section") or "header")
                txt = str(r.get("content") or "").strip()
                if txt:
                    current.setdefault(sec, []).append(txt)
            for sec in order:
                if current.get(sec):
                    blocks.append(f"=== {sec.upper()} ===\n" + "\n".join(current[sec]))
            return "\n\n".join(blocks)
        finally:
            await store.close()
    except Exception as e:
        logger.warning("Resume context unavailable", error=str(e))
        return ""


async def _resume_defaults() -> dict[str, str]:
    """Extract identity defaults from the indexed resume header, if any."""
    try:
        from autofill.src.screener.profile import _regex_extract
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


_DYN_QUESTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "question": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["category", "question"],
            },
        }
    },
    "required": ["questions"],
}


async def generate_dynamic_questions(
    ctx: Any,
    resume_summary: str,
    existing: list[dict],
    identity: dict,
    resume_context: str = "",
    target: int = 6,
) -> list[tuple[str, str]]:
    """Generate candidate-tailored follow-up questions from the LLM.

    Feeds the full resume + portfolio context, already-answered persona Q&A and
    identity to the model so it only proposes questions that are genuinely
    unanswered — it must NOT repeat core questions, already-answered ones, or
    anything the resume/portfolio already covers. Returns (category, question)
    tuples merged on top of CORE_QUESTIONS. Returns [] on any failure.
    """
    if ctx is None:
        return []
    known = []
    for a in existing[:30]:
        q = (a.get("question") or "").strip()
        ans = (a.get("answer") or "").strip()
        if q and ans:
            known.append(f"- {q}: {ans}")
    known_text = "\n".join(known) if known else "(nothing answered yet)"
    ident_text = ", ".join(f"{k}: {v}" for k, v in (identity or {}).items() if v)
    core_text = "\n".join(f"- {q}" for _, q in CORE_QUESTIONS)
    resume_blob = resume_context or resume_summary or ""
    prompt = (
        "You are building a job-application profile for a candidate. "
        "Generate a list of SHORT, high-value screening questions that an "
        "application form or recruiter would STILL need answered.\n\n"
        f"KNOWN IDENTITY: {ident_text}\n\n"
        f"ALREADY ANSWERED:\n{known_text}\n\n"
        f"ALREADY ASKED (core questions — DO NOT re-ask or paraphrase these):\n{core_text}\n\n"
        f"RESUME + PORTFOLIO:\n{(resume_blob or '(none yet)')[:6000]}\n\n"
        "Rules:\n"
        "- Each question must be answerable in one short line.\n"
        "- ONLY ask about things NOT already in the resume, portfolio, identity, "
        "already-answered list, or the core questions above. If the resume says "
        "they know Python/Go, do NOT ask 'what languages do you know?'.\n"
        "- Do NOT ask for contact info (email/phone/linkedin/website) or anything "
        "in the identity section.\n"
        "- Prefer genuinely unknown nuance: exact years of experience, notable "
        "project details, preferred tools, notice period, salary (if absent), "
        "work/office constraints, visa nuance (if absent).\n"
        "- Return at most {target} questions.\n"
        f"- Return exactly {target} questions as JSON with keys category and question; "
        "category should be a short lowercase slug like 'skills' or 'projects'.\n"
    )
    try:
        raw = await ctx.chat(prompt, schema=_DYN_QUESTION_SCHEMA, max_tokens=1200)
        import json as _json

        parsed = _json.loads(raw)
        questions = parsed.get("questions") or []
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        # Semantic block-list: token overlap against core + known questions so
        # the LLM's paraphrased duplicates are filtered, not just exact matches.
        blocked = _question_blocklist(CORE_QUESTIONS, existing)
        for item in questions[:target]:
            category = str(item.get("category") or "general").strip().lower()
            question = re.sub(r"\s+", " ", str(item.get("question") or "").strip()).strip(" ")
            # Normalize trailing sentence punctuation so "proud of." and
            # "proud of" dedupe and match cleanly on re-grill.
            question = re.sub(r"[.!?]+\s*$", "", question)
            key = question.lower().strip()
            if not question or key in seen:
                continue
            if _overlaps_blocklist(question, blocked):
                continue
            seen.add(key)
            out.append((category, question))
        return out
    except Exception as e:
        logger.warning(
            "Dynamic question generation failed; using static question set", error=str(e)
        )
        return []


def _question_blocklist(
    core: list[tuple[str, str]], existing: list[dict]
) -> list[tuple[str, set[str]]]:
    """(canonical phrase, significant tokens) for every question that must not
    be re-asked. Used to catch the LLM's paraphrased duplicates."""
    phrases: list[str] = [q for _, q in core]
    phrases += [str(a.get("question") or "") for a in existing]
    out: list[tuple[str, set[str]]] = []
    for p in phrases:
        tokens = _significant_tokens(p)
        if tokens:
            out.append((p, tokens))
    return out


def _significant_tokens(text: str) -> set[str]:
    """Lowercased, stopword-filtered, lightly-stemmed tokens for overlap
    matching ("relocating"/"relocation"/"relocat" all reduce to 'relocat')."""
    stops = {
        "what",
        "is",
        "are",
        "do",
        "you",
        "your",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "or",
        "and",
        "does",
        "would",
        "not",
        "how",
        "much",
        "many",
        "any",
        "when",
        "where",
        "if",
        "currently",
        "prefer",
        "preferred",
        "include",
        "note",
        "say",
        "answer",
        "with",
        "from",
        "this",
        "that",
        "have",
        "has",
        "had",
    }
    tokens = {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in stops}
    # Light suffix stem so grammatical variants collapse to one token.
    stemmed: set[str] = set()
    for t in tokens:
        for suf in ("ing", "ion", "ions", "ation", "s", "es", "ed"):
            if t.endswith(suf) and len(t) - len(suf) >= 4:
                t = t[: -len(suf)]
                break
        stemmed.add(t)
    return stemmed


def _overlaps_blocklist(question: str, blocked: list[tuple[str, set[str]]]) -> bool:
    """True when the question shares >=2 significant tokens with a blocked one
    (catching 'salary expectations in USD' vs 'what are your salary
    expectations'), unless the question adds a genuinely distinct qualifier."""
    q_tokens = _significant_tokens(question)
    if not q_tokens:
        return False
    for _, b_tokens in blocked:
        shared = q_tokens & b_tokens
        if len(shared) >= 2:
            return True
    return False


def _finish_question_gen_thread(
    result: dict[str, object], ctx: Any, extra_budget: float = 20.0
) -> list[tuple[str, str]]:
    """Await the background thread's question generation with a bounded extra
    wait. Returns the merged question set, or the static core set on timeout/
    error — the wizard must never block long on optional questions."""
    import time as _time

    if result.get("done"):
        qs = result.get("questions")
        return list(qs) if isinstance(qs, list) and qs else list(CORE_QUESTIONS)
    deadline = _time.time() + extra_budget
    while _time.time() < deadline:
        if result.get("done"):
            qs = result.get("questions")
            return list(qs) if isinstance(qs, list) and qs else list(CORE_QUESTIONS)
        _time.sleep(0.2)
    _report_llm_fallback(ctx, "timed out", f"after {int(extra_budget)}s more")
    return list(CORE_QUESTIONS)


def _report_llm_fallback(ctx: Any, reason: str, detail: str = "") -> None:
    """Print a clean [ho] line telling the user which models were tried and why
    the optional LLM follow-up questions fell back to the static set."""
    chain = " -> ".join(ctx.model_chain()) if ctx is not None else "(no LLM client)"
    tried = f"tried [{chain}]"
    if ctx is not None:
        try:
            rep = ctx.failure_report()
            if rep and rep != "no attempts recorded":
                tried = f"tried [{chain}]: {rep}"
        except Exception:
            pass
    msg = f"LLM follow-up questions {reason} {detail}; using static set".strip()
    logger.warning(msg, models=chain, reason=reason)
    ux.chip("warn", msg)
    ux.chip("info", tried)


async def build_question_set(
    ctx: Any,
    resume_summary: str,
    existing: list[dict],
    identity: dict,
    resume_context: str = "",
) -> list[tuple[str, str]]:
    """Merge the core application-form questions with LLM-generated ones.

    Dynamic questions whose category duplicates a core one are skipped, and
    anything the candidate already answered stays the wizard's job to skip —
    so this never shrinks the existing answer list.

    The LLM call is best-effort and BOUNDED: the interactive grill must never
    block minutes on a slow/overloaded model just to invent optional follow-up
    questions — on any timeout/error it falls back to the static core set.
    """
    try:
        dynamic = await asyncio.wait_for(
            generate_dynamic_questions(
                ctx, resume_summary, existing, identity, resume_context=resume_context
            ),
            timeout=_DYN_QUESTION_TIMEOUT_S,
        )
    except TimeoutError:
        _report_llm_fallback(ctx, "timed out", f"after {int(_DYN_QUESTION_TIMEOUT_S)}s")
        dynamic = []
    except Exception as exc:
        _report_llm_fallback(ctx, "failed", str(exc))
        dynamic = []
    core_cats = {c for c, _ in CORE_QUESTIONS}
    merged = list(CORE_QUESTIONS)
    seen_questions = {q for _, q in merged}
    for category, question in dynamic:
        if category in core_cats:
            continue
        if question in seen_questions:
            continue
        seen_questions.add(question)
        merged.append((category, question))
    return merged


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


def _extract_resume_fact(key: str, resume_ctx: str) -> str:
    """Extract a short fact from the resume/portfolio text blob by section."""
    if not resume_ctx:
        return ""
    low = key.lower()
    if low == "skills":
        m = re.search(r"=== SKILLS ===\n(.*?)(?:\n=== |\Z)", resume_ctx, re.S)
        if m:
            return " ".join(m.group(1).split())[:300]
    elif low in ("location", "city"):
        m = re.search(
            r"([A-Z][A-Za-z ,.-]+(?:India|IN|US|USA|UK|Canada|Germany|Remote)?)",
            resume_ctx,
        )
        if m:
            return m.group(1)[:80]
    elif low in ("projects",):
        m = re.search(r"=== PROJECTS ===\n(.*?)(?:\n=== |\Z)", resume_ctx, re.S)
        if m:
            return " ".join(m.group(1).split())[:300]
    elif low in ("education", "university", "degree"):
        m = re.search(r"=== EDUCATION ===\n(.*?)(?:\n=== |\Z)", resume_ctx, re.S)
        if m:
            return " ".join(m.group(1).split())[:300]
    return ""


def _auto_answer(question: str, resume: dict[str, str], resume_ctx: str = "") -> str:
    """Best-effort answer for a grill question derived from resume sections /
    portfolio text, so the user mostly confirms instead of typing.

    Matches the question's significant tokens against resume/portfolio content
    and returns the most relevant short extract. Returns '' when nothing maps.
    """
    q_tokens = _significant_tokens(question)
    if not q_tokens:
        return ""

    # Map the question's topic to the resume section that most likely answers it.
    low = question.lower()
    if "location" in low or "based" in low or "relocat" in low:
        loc = resume.get("location") or resume.get("city") or ""
        return loc
    if "salary" in low or "compensation" in low or "comp" in low:
        return resume.get("expected_compensation") or ""
    if "visa" in low or "sponsor" in low or "authorized" in low:
        return resume.get("work_authorization") or ""
    if "skill" in low or "tech" in low or "stack" in low or "tools" in low:
        return resume.get("skills") or ""
    if "hours" in low or "week" in low:
        return resume.get("hours_per_week") or ""
    if "start" in low or "notice" in low or "available" in low:
        return resume.get("availability") or ""
    if "project" in low or "proud" in low:
        return resume.get("projects") or ""
    if "education" in low or "university" in low or "degree" in low or "graduat" in low:
        return resume.get("education") or ""
    if "ai" in low or "llm" in low or "openai" in low or "claude" in low or "backend" in low:
        return resume.get("ai_experience") or resume.get("skills") or ""

    # Fallback: token-overlap against resume_ctx, return the tightest extract.
    if resume_ctx:
        best = ""
        best_score = 0
        for chunk in _split_ctx_sentences(resume_ctx):
            lt = chunk.strip()
            if not lt or len(lt) > 260:
                continue
            ltokens = _significant_tokens(lt)
            shared = q_tokens & ltokens
            if len(shared) > best_score:
                best_score = len(shared)
                best = lt
        if best_score >= 2:
            return best[:240]
    return ""


def _split_ctx_sentences(text: str) -> list[str]:
    """Split resume/portfolio context into sentence-ish chunks for overlap."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def grill_answers(data: dict, ask_all: bool = False, questions: list | None = None) -> dict:
    if questions is None:
        questions = CORE_QUESTIONS
    stored = data.get("answers", [])
    by_question = {a["question"]: a for a in stored}
    by_category: dict[str, list[dict]] = {}
    for a in stored:
        by_category.setdefault(a["category"], []).append(a)

    ux.section(2, 3, "Personal Q&A", "Enter keeps the prefilled answer; type to change")
    answers: list[dict] = []
    kept: list[dict] = []
    seen: set[str] = set()
    skipped = 0
    for category, question in questions:
        # Re-grills only ask what is still missing: a question is skipped when
        # it already has an answer (or its single-question category does),
        # unless --all forces a full re-ask. Skipped questions keep their
        # stored entries — the saved list must NEVER shrink from a skip.
        multi = sum(1 for c, _ in questions if c == category) > 1
        already = question in by_question or (not multi and bool(by_category.get(category)))
        if already and not ask_all:
            skipped += 1
            entry = by_question.get(question)
            if entry is None:
                cat_entries = by_category.get(category) or []
                entry = cat_entries[0] if cat_entries else None
            if entry is not None and entry not in kept:
                kept.append(entry)
            continue
        prev = _previous_answer(category, question, by_question, by_category)
        # Auto-prefill from resume/portfolio when there's no stored answer yet.
        default = prev.get("answer", "") or _auto_answer(question, _AUTO_RESUME, _AUTO_RESUME_CTX)
        answer = _ask(question, default)
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

    wizard_cats = {c for c, _ in questions}
    extra_kept = [a for a in stored if a.get("category") not in wizard_cats and a not in kept]
    data["answers"] = kept + extra_kept + answers
    if skipped and not ask_all:
        ux.chip("info", f"Skipped {skipped} already-answered question(s) (--all to re-ask)")
    return data


def save_persona(data: dict) -> None:
    data["version"] = int(data.get("version", 1)) + 1
    PERSONA_JSON.parent.mkdir(parents=True, exist_ok=True)
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

    # Full resume + portfolio context so generated questions are grounded in
    # what the candidate has already said (avoids dumb repeats).
    resume_ctx = asyncio.run(_resume_context())
    if resume_ctx:
        ux.chip("info", f"Loaded {len(resume_ctx)} chars of resume + portfolio context")

    # Dynamic question set: core application-form questions + LLM-generated
    # candidate-specific follow-ups. Falls back to core-only on any LLM issue.
    ctx = None
    try:
        from src.llm.context import ContextManager

        ctx = ContextManager()
    except Exception:
        ctx = None

    # Populate auto-prefill sources: identity-ish facts + full resume text so
    # each question shows a best-guess default (Enter keeps, type to change).
    global _AUTO_RESUME, _AUTO_RESUME_CTX
    _AUTO_RESUME = dict(resume)
    for key in (
        "skills",
        "location",
        "expected_compensation",
        "work_authorization",
        "hours_per_week",
        "availability",
        "projects",
        "education",
        "ai_experience",
        "city",
    ):
        _AUTO_RESUME.setdefault(key, _extract_resume_fact(key, resume_ctx))
    _AUTO_RESUME_CTX = resume_ctx

    # Dynamic question generation is best-effort and can take a while on a
    # slow/overloaded model. Run it in a background THREAD while the user
    # answers the identity section (the interactive flow is sync/blocking, so
    # an asyncio task would be frozen by the prompts). Await the thread result
    # right before personal Q&A with a short budget — the static core set is the
    # instant fallback, so the wizard never blocks 90s on optional questions.
    import threading as _threading

    q_gen_result: dict[str, object] = {"questions": None, "done": False}

    def _gen_in_thread() -> None:
        try:
            qs = asyncio.run(
                build_question_set(
                    ctx,
                    str(data.get("resume_summary") or ""),
                    data.get("answers", []),
                    data.get("identity", {}),
                    resume_context=resume_ctx,
                )
            )
            q_gen_result["questions"] = qs
        except Exception as exc:  # noqa: BLE001
            q_gen_result["questions"] = list(CORE_QUESTIONS)
            _report_llm_fallback(ctx, "failed", str(exc))
        finally:
            q_gen_result["done"] = True

    gen_thread = _threading.Thread(target=_gen_in_thread, daemon=True)
    gen_thread.start()

    data = grill_identity(data, resume, ask_all=args.all)

    questions = _finish_question_gen_thread(q_gen_result, ctx)
    if len(questions) > len(CORE_QUESTIONS):
        ux.chip(
            "info",
            f"Grilling {len(questions)} questions "
            f"({len(questions) - len(CORE_QUESTIONS)} generated from your resume).",
        )

    data = grill_answers(data, ask_all=args.all, questions=questions)
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
    try:
        main()
    except KeyboardInterrupt:
        import sys as _sys

        print("\n[ho] Quit persona wizard? (y/N) ", flush=True)
        try:
            ans = input().strip().lower()
        except KeyboardInterrupt, EOFError:
            ans = "y"
        if ans in ("y", "yes"):
            print("[ho] Exiting. No changes saved unless a question was answered.", flush=True)
        else:
            print("[ho] Continuing wizard...", flush=True)
            main()
        _sys.exit(0)
