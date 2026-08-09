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
    # Structured education facts (school, degree, program, major, enrollment,
    # start/grad years) — personal facts the LLM must never fabricate.
    education: dict[str, Any] = Field(default_factory=dict, alias="education")

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
    # Match the SCHOOL question specifically — "highest level of education
    # (degree + field)" is the DEGREE, not the school. Using a broad regex
    # ("education") matched that first and set school="BTech CSE in AI & ML".
    if not getattr(profile, "school", None) and not getattr(profile, "university", None):
        for _q, _a in profile.customAnswers.items():
            if (
                re.search(
                    r"university|college|institute|school (name|attended)|"
                    r"which (school|university|college)|where did you (study|attend)",
                    _q,
                    re.I,
                )
                and _a.strip()
            ):
                profile.school = _a.strip()
                profile.university = _a.strip()
                break

    # Structured education block (identity.education in persona.json) — the
    # source of truth for school/degree/major/enrollment/years. The LLM must
    # never fabricate these.
    edu = identity_json.get("education")
    if isinstance(edu, dict) and edu:
        profile.education = {str(k): v for k, v in edu.items() if v is not None}
    if not profile.education:
        # Fall back to the customAnswers-derived school for the school slot.
        for _q, _a in profile.customAnswers.items():
            if re.search(r"degree|major|field of study|specialization", _q, re.I) and _a.strip():
                profile.education.setdefault("degree_program", _a.strip())
                _parse_degree_into_education(_a.strip(), profile.education)
        if getattr(profile, "school", None):
            profile.education.setdefault("school", profile.school)
            profile.education.setdefault("university", profile.university or profile.school)

    # Start/graduation years from persona answers if present (e.g. a grill
    # answer like "Vellore Institute of Technology (2023 - 2027)").
    if not profile.education.get("start_year"):
        for _q, _a in profile.customAnswers.items():
            years = re.findall(r"\b(?:19|20)\d{2}\b", _a)
            if len(years) >= 1 and not profile.education.get("start_year"):
                profile.education.setdefault("start_year", years[0])
            if len(years) >= 2 and not profile.education.get("grad_year"):
                profile.education.setdefault("grad_year", years[-1])

    # Clean the school name: strip parenthetical years ("VIT (2023 - 2027)"
    # -> "VIT") so a school-name question gets the name, not the years.
    if profile.education.get("school"):
        school_clean = re.sub(r"\s*\([^)]*\)\s*$", "", profile.education["school"]).strip()
        if school_clean:
            profile.education["school"] = school_clean
    if profile.education.get("university"):
        uni_clean = re.sub(r"\s*\([^)]*\)\s*$", "", profile.education["university"]).strip()
        if uni_clean:
            profile.education["university"] = uni_clean
    # Keep profile.school/university in sync with the cleaned values.
    if profile.education.get("school"):
        profile.school = profile.education["school"]
    if profile.education.get("university"):
        profile.university = profile.education["university"]

    still_defaults = [f for f in _IDENTITY_FIELDS if f not in resolved]
    if still_defaults:
        logger.warning(
            "Profile fields unresolved; using placeholders",
            fields=still_defaults,
            source="persona.json / resume header",
        )

    return profile


def _parse_degree_into_education(degree_text: str, edu: dict) -> None:
    """Parse a persona degree answer like "BTech CSE in AI & ML" into the
    structured education fields: degree (BTech), discipline/major (Computer
    Science and Engineering), and specialization (AI & ML).

    CSE is the common shorthand for "Computer Science and Engineering" —
    a form asking for the discipline/major must get the expanded name, not
    the abbreviation.
    """
    text = re.sub(r"\s+", " ", degree_text.strip())
    if not text:
        return

    # Degree type: leading BTech/B.E./B.Sc./M.Tech/PhD/BS/MS etc.
    degree = ""
    m = re.match(
        r"^\s*((?:B\.?Tech|B\.?E|B\.?Sc|B\.?A|M\.?Tech|M\.?S|M\.?Sc|M\.?A"
        r"|PhD|B\.?Com|B\.?BA|D\.?Tech)[A-Za-z .]*?)(?=\s|$)",
        text,
        re.I,
    )
    if m:
        degree = m.group(1).strip().rstrip(".")
        edu.setdefault("degree", degree)

    # Discipline/major + specialization. "BTech CSE in AI & ML":
    #   discipline = "Computer Science and Engineering" (CSE expanded)
    #   specialization = "AI & ML"
    rest = text
    if degree:
        rest = text[len(m.group(0)) :].strip()
    # Split on " in " / " with specialization in " / " - ".
    parts = re.split(r"\s+(?:in|with speciali[sz]ation in|majoring in|-\s*)\s+", rest, maxsplit=1)
    discipline = parts[0].strip().strip(".,")
    spec = parts[1].strip() if len(parts) > 1 else ""
    if discipline:
        edu.setdefault("discipline", _expand_discipline(discipline))
    if spec:
        edu.setdefault("major", spec)

    # Common "CSE" shorthand -> full discipline name.
    if discipline and not spec and re.fullmatch(r"CSE|Comp\s*Sci|CS", discipline, re.I):
        edu.setdefault("major", "Computer Science and Engineering")


def _expand_discipline(discipline: str) -> str:
    """Expand discipline abbreviations a form would not accept as-is."""
    d = re.sub(r"\s+", " ", discipline.strip())
    if re.fullmatch(r"CSE", d, re.I):
        return "Computer Science and Engineering"
    if re.fullmatch(r"CS", d, re.I):
        return "Computer Science"
    if re.fullmatch(r"ECE", d, re.I):
        return "Electronics and Communication Engineering"
    if re.fullmatch(r"IT", d, re.I):
        return "Information Technology"
    return d
