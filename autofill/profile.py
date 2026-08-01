"""Candidate profile used by the autofill pipeline.

Identity fields are resolved deterministically from the persona RAG store
(persona_embeddings identity chunks), with regex fallbacks over the resume
header and a direct persona.json read as the last real-data fallback.
No LLM calls happen here; the resolution is fully deterministic.
"""

import json
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from src.logging import get_logger

logger = get_logger("autofill.profile")

PERSONA_MATCH_THRESHOLD = 0.6

_IDENTITY_FIELDS = ("firstName", "lastName", "email", "phone", "linkedin", "github", "website")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?[\d][\d\s().-]{7,}\d")
_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w.-]+", re.I
)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+", re.I)
_WEBSITE_RE = re.compile(r"(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:/[\w./-]*)?", re.I)


class Profile(BaseModel):
    firstName: str = Field(default="John", alias="first_name")
    lastName: str = Field(default="Doe", alias="last_name")
    email: str = Field(default="john.doe@example.com")
    phone: str = Field(default="+1234567890")
    linkedin: Optional[str] = Field(default="https://linkedin.com/in/johndoe")
    github: Optional[str] = Field(default="https://github.com/johndoe")
    website: Optional[str] = Field(default="https://johndoe.dev")
    resumePath: Optional[str] = Field(default=None, alias="resume_path")
    customAnswers: Dict[str, str] = Field(default_factory=dict, alias="custom_answers")

    model_config = {
        "populate_by_name": True,
        "serialize_by_alias": False
    }


def _load_persona_json() -> dict[str, Any]:
    import os

    for path in ("persona.json", os.path.join(os.path.dirname(__file__), "..", "persona.json")):
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
        from autofill.rag import _embed_text
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
        from autofill.rag import _embed_text

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
    """Build a Profile with fully deterministic identity resolution.

    Resolution order per field:
    1. persona_embeddings identity chunk (semantic match, exact answer)
    2. regex extraction from resume header chunks (contact fields)
    3. persona.json ``identity`` block (when no store / nothing matched)
    4. Profile defaults (last resort, logged)
    ``customAnswers`` is populated from persona.json answers.
    """
    persona_data = _load_persona_json()
    identity_json = persona_data.get("identity", {})

    profile = Profile()
    resolved: dict[str, str] = {}

    header = await _lookup_resume_header(store) if store is not None else ""

    for field in _IDENTITY_FIELDS:
        value = None
        if store is not None:
            value = await _lookup_identity_field(store, field)
            if not value and field in ("email", "phone", "linkedin", "github", "website"):
                value = _regex_extract(header).get(field)
        if not value:
            value = identity_json.get(field)
        if value:
            resolved[field] = value

    for field, value in resolved.items():
        setattr(profile, field, value)

    if not resolved.get("firstName") and identity_json.get("name"):
        profile.firstName = identity_json["name"]
    if not resolved.get("firstName") or not resolved.get("lastName"):
        name = persona_data.get("name", "")
        parts = name.split(maxsplit=1)
        if parts:
            if not resolved.get("firstName"):
                profile.firstName = parts[0]
            if len(parts) > 1 and not resolved.get("lastName"):
                profile.lastName = parts[1]

    for a in persona_data.get("answers", []):
        q = (a.get("question") or "").strip()
        ans = (a.get("answer") or "").strip()
        if q and ans:
            profile.customAnswers[q] = ans

    still_defaults = [f for f in _IDENTITY_FIELDS if f not in resolved]
    if still_defaults:
        logger.warning(
            "Profile fields unresolved; using placeholders",
            fields=still_defaults,
            source="identity chunk / resume header / persona.json",
        )

    return profile
