# ruff: noqa: N815  (camelCase fields mirror the Node/TS payload keys)
"""Candidate profile used by the autofill pipeline.

Identity fields are resolved deterministically from the persona RAG store
(persona_embeddings identity chunks), with regex fallbacks over the resume
header and a direct persona.json read as the last real-data fallback.
No LLM calls happen here; the resolution is fully deterministic.
"""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from src.logging import get_logger

logger = get_logger("autofill.src.screener.profile")

PERSONA_MATCH_THRESHOLD = 0.6

_IDENTITY_FIELDS = (
    "firstName",
    "lastName",
    "email",
    "phone",
    "linkedin",
    "github",
    "website",
    "twitter",
    "preferredName",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?[\d][\d\s().-]{7,}\d")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w.-]+", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+", re.I)
_TWITTER_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:x|twitter)\.com/[A-Za-z0-9_]+", re.I)
_WEBSITE_RE = re.compile(r"(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:/[\w./-]*)?", re.I)


class Profile(BaseModel):
    firstName: str = Field(default="John", alias="first_name")
    lastName: str = Field(default="Doe", alias="last_name")
    email: str = Field(default="john.doe@example.com")
    phone: str = Field(default="+1234567890")
    linkedin: str | None = Field(default="https://linkedin.com/in/johndoe")
    github: str | None = Field(default="https://github.com/johndoe")
    website: str | None = Field(default="https://johndoe.dev")
    twitter: str | None = Field(default=None, alias="twitter")
    preferredName: str | None = Field(default=None, alias="preferred_name")
    location: str | None = Field(default=None, alias="location")
    school: str | None = Field(default=None, alias="school")
    university: str | None = Field(default=None, alias="university")
    resumePath: str | None = Field(default=None, alias="resume_path")
    customAnswers: dict[str, str] = Field(default_factory=dict, alias="custom_answers")

    model_config = {"populate_by_name": True, "serialize_by_alias": False}


def _load_persona_json() -> dict[str, Any]:
    """Locate and load data/persona.json at the repo root.

    ``profile.py`` lives at ``packages/autofill/src/screener/`` so the
    repo root is FOUR parents up (screener -> src -> autofill -> packages ->
    root).
    A CWD-relative lookup is also tried first (covers running from the repo
    root). Previously the path resolved to ``packages/data/`` (wrong), so the
    persona was never loaded — identity fell back to semantic-search garbage
    and ``customAnswers`` stayed empty.
    """
    import os

    base = Path(__file__).resolve().parents[4]  # repo root
    candidates = (
        Path.cwd() / "data" / "persona.json",
        base / "data" / "persona.json",
        Path(os.environ.get("CANDIDATE_PERSONA_FILE", "")),
    )
    for path in candidates:
        if not str(path):
            continue
        try:
            with open(path) as f:
                return json.load(f)
        except OSError:
            continue
    return {}


def _regex_extract(text: str) -> dict[str, str]:
    """Best-effort deterministic extraction of contact fields from raw text."""
    found: dict[str, str] = {}
    m = _EMAIL_RE.search(text)
    if m:
        found["email"] = m.group(0)
        text = text.replace(m.group(0), " ", 1)
    m = _LINKEDIN_RE.search(text)
    if m:
        found["linkedin"] = m.group(0)
        text = text.replace(m.group(0), " ", 1)
    m = _GITHUB_RE.search(text)
    if m:
        found["github"] = m.group(0)
        text = text.replace(m.group(0), " ", 1)
    m = _TWITTER_RE.search(text)
    if m:
        found["twitter"] = m.group(0)
        text = _TWITTER_RE.sub(" ", text)
    m = _PHONE_RE.search(text)
    if m:
        found["phone"] = m.group(0).strip()
    m = _WEBSITE_RE.search(text)
    if m:
        found["website"] = m.group(0)
    return found


async def _lookup_identity_field(store: Any, field: str) -> str | None:
    """Retrieve a deterministic identity value from persona_embeddings."""
    try:
        from autofill.src.screener.rag import _embed_text

        emb = await _embed_text(f"candidate {field}")
    except Exception:
        emb = None
    if not emb:
        return None
    try:
        results = await store.search_similar_persona(emb, top_k=6)
    except Exception as e:
        logger.warning("Identity search failed", field=field, error=str(e))
        return None
    for r in results:
        if r.get("category") != "identity":
            continue
        if r["distance"] <= PERSONA_MATCH_THRESHOLD:
            answer = (r.get("answer") or "").strip()
            if answer:
                logger.info(
                    "Identity resolved from persona store",
                    field=field,
                    distance=r["distance"],
                )
                return answer
    return None


async def _lookup_resume_header(store: Any) -> str:
    """Collect raw resume header content deterministically."""
    if store is None:
        return ""
    try:
        from autofill.src.screener.rag import _embed_text

        emb = await _embed_text("candidate contact information resume header")
    except Exception:
        return ""
    if not emb:
        return ""
    try:
        rows = await store.search_similar_chunks(emb, top_k=8)
    except Exception as e:
        logger.warning("Resume header search failed", error=str(e))
        return ""
    parts: list[str] = []
    for r in rows:
        if r.get("section") == "header" and (r.get("content") or "").strip():
            parts.append(r["content"].strip())
    return " | ".join(parts)


async def build_profile(store: Any = None) -> Profile:
    """Build a Profile with deterministic-first identity resolution.

    Resolution order per field (the persona file is ground truth — it was
    curated by the user; semantic search over 81 tiny chunks is unreliable and
    previously returned the website for "location" and "Sahu" for "state"):
    1. persona.json ``identity`` block (exact values the user set)
    2. persona.json ``answers`` matched to the field (location, education, …)
    3. regex extraction from resume header chunks (contact fields only)
    4. persona_embeddings semantic search — STRICT fallback, only when the
       persona file has no value AND the match is well above threshold
    ``customAnswers`` is always populated from persona.json answers.
    """
    persona_data = _load_persona_json()
    identity_json = persona_data.get("identity", {})
    answers = persona_data.get("answers", []) or []

    profile = Profile()
    resolved: dict[str, str] = {}

    # 1+2: persona.json first — exact, curated, trustworthy.
    for field in _IDENTITY_FIELDS:
        value = identity_json.get(field)
        if value:
            resolved[field] = str(value).strip()

    # Name fallbacks from persona.json name field.
    if not resolved.get("firstName") or not resolved.get("lastName"):
        name = persona_data.get("name", "") or identity_json.get("name", "")
        parts = name.split(maxsplit=1)
        if parts:
            resolved.setdefault("firstName", parts[0])
            if len(parts) > 1:
                resolved.setdefault("lastName", parts[1])

    # 3: resume header regex for contact fields persona didn't set.
    header = await _lookup_resume_header(store) if store is not None else ""
    for field in ("email", "phone", "linkedin", "github", "website"):
        if not resolved.get(field):
            v = _regex_extract(header).get(field)
            if v:
                resolved[field] = v

    # 4: semantic search only for fields still missing (never overrides the
    # persona file). Restrict to identity-category chunks at a strict distance.
    if store is not None:
        for field in _IDENTITY_FIELDS:
            if resolved.get(field):
                continue
            val = await _lookup_identity_field(store, field)
            if val:
                resolved[field] = val

    for field, value in resolved.items():
        setattr(profile, field, value)

    # customAnswers from persona.json answers — the screener's knowledge base.
    for a in answers:
        q = (a.get("question") or "").strip()
        ans = (a.get("answer") or "").strip()
        if q and ans:
            profile.customAnswers[q] = ans

    # Preferred first name defaults to the first name when not otherwise known.
    if not profile.preferredName:
        profile.preferredName = profile.firstName

    # Location from the persona's current-location answer (question phrasing
    # varies across forms, so match on the persona's known question).
    if not profile.location:
        for _q, _a in profile.customAnswers.items():
            if (
                re.search(
                    r"current location|where (are|do) you (currently )?"
                    r"(based|live|located|stay|reside)|your location|location\b|"
                    r"where in the world",
                    _q,
                    re.I,
                )
                and _a.strip()
            ):
                profile.location = _a.strip()
                break

    # Education/school from the persona's education answer, if present.
    if not getattr(profile, "school", None) and not getattr(profile, "university", None):
        for _q, _a in profile.customAnswers.items():
            if re.search(r"school|university|college|education|institute", _q, re.I) and _a.strip():
                profile.school = _a.strip()
                break

    still_defaults = [f for f in _IDENTITY_FIELDS if f not in resolved]
    if still_defaults:
        logger.warning(
            "Profile fields unresolved; using placeholders",
            fields=still_defaults,
            source="persona.json / resume header",
        )

    return profile
