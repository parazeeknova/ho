"""Non-LLM pre-submit consistency check.

Before an application is submitted, cross-check every filled form field
against the candidate's known ground truth (profile + persona.json) using:

- NLP: classify each field label (location / name / email / phone / education /
  compensation / work-authorization / ...) and compare the committed value to
  the expected one with normalized + fuzzy matching.
- RAG: for free-text fields (skills, experience, "why you", cover-letter-ish)
  use semantic similarity against the persona/resume embeddings; a committed
  value far from any known fact is flagged as suspicious.

No LLM call happens here. The result is a consistency report the worker uses
to block a submit (retryable) when a critical field contradicts the persona.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from src.logging import get_logger

logger = get_logger("autofill.consistency")

# persona.json lives at the repo root (same location rag.py uses). A question
# learned MID-RUN (via a Discord answer this session) is appended here, so the
# consistency gate reads it live instead of trusting a profile snapshot that
# was built once at worker start.
_PERSONA_JSON = Path(__file__).resolve().parents[3] / "data" / "persona.json"

# Field-label -> the profile field it maps to. The committed value is compared
# to that profile value.
_LABEL_TO_PROFILE: dict[str, str] = {
    "location": "location",
    "city": "location",
    "country": "location",
    "country of residence": "location",
    "current location": "location",
    "state": "location",
    "region": "location",
    "first name": "firstName",
    "last name": "lastName",
    "full name": "fullName",
    "name": "fullName",
    "email": "email",
    "phone": "phone",
    "phone number": "phone",
    "mobile": "phone",
    "telephone": "phone",
    "linkedin": "linkedin",
    "github": "github",
    "website": "website",
    "portfolio": "website",
    "twitter": "twitter",
    "current company": "company",
    "university": "school",
    "college": "school",
    "school": "school",
    "education": "school",
    "nationality": "nationality",
    "gender": "gender",
    # Additional common application-form fields mapped to persona answers.
    "availability": "availability",
    "start date": "availability",
    "start when": "availability",
    "notice period": "availability",
    "available to start": "availability",
    "current salary": "current_comp",
    "current compensation": "current_comp",
    "expected salary": "expected_comp",
    "salary expectation": "expected_comp",
    "desired compensation": "expected_comp",
    "work authorization": "work_authorization",
    "authorized to work": "work_authorization",
    "right to work": "work_authorization",
    "visa sponsorship": "visa_sponsorship",
    "sponsorship": "visa_sponsorship",
    "relocation": "relocation",
    "willing to relocate": "relocation",
    "remote": "work_model",
    "work model": "work_model",
    "preferred working style": "work_model",
    "working hours": "working_hours",
    "hours per week": "working_hours",
    "equity": "equity",
    "gender identity": "gender_identity",
    "ethnicity": "ethnicity",
    "disability": "disability",
    "veteran": "veteran_status",
    "pronouns": "pronouns",
    "linkedin profile": "linkedin",
    "github profile": "github",
    "personal website": "website",
}

# Critical fields: a mismatch here MUST block submit.
_CRITICAL = {
    "firstName",
    "lastName",
    "fullName",
    "email",
    "phone",
    "location",
    "linkedin",
    "github",
    "website",
}

# Multi-value fields (mismatch is softer — the form may be a variant).
_SOFT = {"school", "company", "nationality", "gender"}

# Registry field -> persona-question keyword(s) that ask about it. Lets a bare
# "Availability" or "Notice period" field map to the persona's availability
# question even when the wording shares no tokens.
_FIELD_TO_Q_KEYWORD: dict[str, tuple[str, ...]] = {
    "availability": ("start", "available", "notice", "soon"),
    "current_comp": ("compensation", "current"),
    "expected_comp": ("salary", "expectation", "compensation"),
    "working_hours": ("hours", "week", "available"),
    "visa_sponsorship": ("visa", "sponsor"),
    "work_authorization": ("authoriz", "work"),
    "relocation": ("relocat", "move"),
    "work_model": ("remote", "work"),
    "equity": ("equity", "esop", "rsu"),
}

# For persona-Q&A-matched NEW fields, a mismatch only blocks submit when the
# field label touches identity / legal status / contact — a wrong answer there
# is disqualifying. Free-form preferences ("why this role") never block.
_CRITICAL_FIELD_HINT_RE = re.compile(
    r"authoriz|visa|sponsor|work author|legal|citizen|national|relocat|"
    r"location|based|address|phone|email|name\b|linkedin|github|website|"
    r"salary|compensation|availability|start|notice|hours|disability|"
    r"veteran|gender|ethnic|pronoun",
    re.I,
)


def _norm(value: Any) -> str:
    return re.sub(r"[\s@_.\-/+()\[\]{}]+", "", str(value or "")).lower()


def _norm_loose(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip(" .,;:!?")


def _fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _normalise_q(text: str) -> str:
    """Tighten a question/field label into a comparable token string."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _kw_boost(question_norm: str, tokens: set[str]) -> float:
    """0.25 per high-signal shared keyword (salary, visa, start, hours...)."""
    if not question_norm or not tokens:
        return 0.0
    shared = tokens & set(question_norm.split())
    if not shared:
        return 0.0
    return 0.25 * len(shared)


def _live_persona_answers() -> dict[str, str]:
    """Read the freshest grilled/learned Q&A straight from persona.json.

    The worker builds ``profile.customAnswers`` once at start; a question the
    user answered mid-run via Discord/learn() is appended to persona.json and
    indexed live, but not in that snapshot. Loading the file here closes the
    gap so the pre-submit gate enforces the newest answers too. The on-disk
    entry always wins over the stale profile copy.
    """
    try:
        data = json.loads(_PERSONA_JSON.read_text())
    except OSError, json.JSONDecodeError:
        return {}
    answers: dict[str, str] = {}
    for a in data.get("answers", []) or []:
        q = (a.get("question") or "").strip()
        ans = (a.get("answer") or "").strip()
        if q and ans:
            answers[q] = ans
    return answers


def _match_persona_question(label: str, custom_answers: dict[str, str]) -> tuple[str, float] | None:
    """Best-matching persona question for an unknown field label.

    Compares the normalized label against every persona question key using
    token overlap + sequence ratio + substring containment. Returns
    (matched_question, score) when a confident match exists, else None. This
    is what lets NEW fields (not in the static _LABEL_TO_PROFILE registry) be
    verified against the persona's grilled answers — e.g. a form's "Expected
    salary" is matched to the persona's "What are your salary expectations?".
    """
    if not custom_answers:
        return None
    nl = _normalise_q(label)
    if not nl:
        return None
    l_tokens = set(nl.split())

    # If the label matches a registry field, prefer a persona question whose
    # text carries that field's keyword.
    for reg_key, profile_field in _LABEL_TO_PROFILE.items():
        if reg_key in nl or _normalise_q(reg_key) == nl:
            kw = _FIELD_TO_Q_KEYWORD.get(profile_field)
            if kw:
                for question in custom_answers:
                    nq = _normalise_q(question)
                    if nq and any(k in nq for k in kw):
                        return (question, 0.8)

    best_q: str | None = None
    best_score = 0.0
    for question in custom_answers:
        nq = _normalise_q(question)
        if not nq:
            continue
        q_tokens = set(nq.split())
        shared = l_tokens & q_tokens
        score = 0.0
        if shared:
            jaccard = len(shared) / max(1, len(l_tokens | q_tokens))
            score = max(jaccard, _fuzzy_ratio(nl, nq)) + 0.1 * len(shared)
        else:
            # No shared token: still catch "Expected salary" vs "...salary
            # expectations..." via substring + keyword boost.
            sub_score = 0.0
            for tok in l_tokens:
                if tok in nq:
                    sub_score += 0.25
            if sub_score:
                score = sub_score + _kw_boost(nq, l_tokens) + 0.1 * _fuzzy_ratio(nl, nq)
            else:
                score = _fuzzy_ratio(nl, nq)
        if score > best_score:
            best_score = score
            best_q = question
    if best_q and best_score >= 0.30:
        return (best_q, round(best_score, 3))
    return None


def _classify(label: str) -> tuple[str | None, bool]:
    """Map a field label to (profile_field, is_critical).

    Critical = identity/contact facts a mismatch must block; soft = education/
    demographics where a variant is acceptable.
    """
    low = _norm_loose(label)
    if not low:
        return None, False
    # Exact-ish match against known labels (with optional trailing junk).
    for key, profile_field in _LABEL_TO_PROFILE.items():
        if key in low or _norm_loose(key) == low:
            return profile_field, profile_field in _CRITICAL
    # "what is your X" style.
    m = re.search(
        r"(?:your|the)?\s*(first name|last name|full name|email|phone|"
        r"location|city|country of residence|university|college|school|"
        r"linkedin|github|website|nationality|gender)\s*",
        low,
    )
    if m:
        pf = _LABEL_TO_PROFILE.get(m.group(1))
        if pf:
            return pf, pf in _CRITICAL
    return None, False


def _values_match(expected: Any, committed: Any) -> float:
    """0..1 similarity between the expected persona value and the committed one."""
    e = str(expected or "")
    c = str(committed or "")
    if not e or not c:
        return 0.0
    en, cn = _norm(e), _norm(c)
    if not en or not cn:
        return 0.0
    if en == cn:
        return 1.0
    # One contains the other (e.g. "Bhopal, India" vs "Bhopal").
    if en in cn or cn in en:
        return 0.9
    # Email/URL: exact normalized match only (fuzzy on an email is useless).
    if "@" in e or "://" in e or e.startswith("+"):
        return 0.0
    return _fuzzy_ratio(_norm_loose(e), _norm_loose(c))


async def check_payload(
    filled_fields: dict[str, str],
    profile: Any,
    store: Any = None,
    rag: Any = None,
) -> dict[str, Any]:
    """Cross-check filled form values against the profile + persona.

    Returns a report:
      {
        "ok": bool,                 # False if any critical field mismatches
        "critical_mismatches": [ {label, filled, expected, score} ],
        "soft_warnings":   [ {label, filled, expected, score} ],
        "checked": int,             # fields that had an expected value
        "unchecked": int,           # fields we could not verify (no expected)
        "rag_flags": [ {label, filled, score} ],
      }
    """
    report: dict[str, Any] = {
        "ok": True,
        "critical_mismatches": [],
        "soft_warnings": [],
        "rag_flags": [],
        "checked": 0,
        "unchecked": 0,
    }
    if not filled_fields:
        return report

    profile_map: dict[str, str] = {}
    if profile is not None:
        for field in (
            "firstName",
            "lastName",
            "email",
            "phone",
            "location",
            "linkedin",
            "github",
            "website",
            "twitter",
            "school",
            "university",
        ):
            v = getattr(profile, field, None) or ""
            if v:
                profile_map[field] = str(v)
        full = f"{profile.firstName} {profile.lastName}".strip()
        if full:
            profile_map["fullName"] = full

    # Persona's grilled Q&A: question -> answer. Used to verify NEW/unknown
    # field labels by fuzzy-matching the label against the known questions.
    custom_answers: dict[str, str] = {}
    if profile is not None:
        ca = getattr(profile, "customAnswers", None) or {}
        if isinstance(ca, dict):
            custom_answers = {str(k): str(v) for k, v in ca.items() if v}
    # Freshest source wins: questions learned/grilled mid-run (after the worker
    # built the profile snapshot) are appended to persona.json and indexed, so
    # read them live and let them override the stale copy.
    live = _live_persona_answers()
    if live:
        custom_answers.update(live)

    for label, committed in (filled_fields or {}).items():
        if not committed or not str(committed).strip():
            continue
        profile_field, is_critical = _classify(label)
        expected = profile_map.get(profile_field or "", "") if profile_field else ""
        if expected:
            score = _values_match(expected, committed)
            report["checked"] += 1
            entry = {
                "label": label,
                "filled": str(committed)[:120],
                "expected": expected[:120],
                "score": round(score, 3),
            }
            if score < 0.45 and is_critical:
                report["critical_mismatches"].append(entry)
                report["ok"] = False
            elif score < 0.45:
                report["soft_warnings"].append(entry)
            continue

        # NEW / unknown field: try to match the label against the persona's
        # grilled Q&A and compare the committed value to that answer. This is
        # how fields outside the static registry are still verified — e.g. a
        # form's "When could you start?" is matched to the persona's
        # "How soon can you start if selected?" answer.
        matched = _match_persona_question(label, custom_answers)
        if matched:
            persona_q, match_score = matched
            persona_ans = custom_answers.get(persona_q, "")
            if persona_ans:
                score = _values_match(persona_ans, committed)
                report["checked"] += 1
                entry = {
                    "label": label,
                    "filled": str(committed)[:120],
                    "expected": persona_ans[:120],
                    "matched_question": persona_q[:120],
                    "score": round(score, 3),
                    "match": round(match_score, 3),
                }
                # A hard "No"/"N/A"/"Prefer not" persona answer that the form
                # filled as the opposite is a consistency failure, but only
                # block when the label itself is identity/authorization-ish.
                if score < 0.45:
                    low_label = label.lower()
                    if _CRITICAL_FIELD_HINT_RE.search(low_label):
                        report["critical_mismatches"].append(entry)
                        report["ok"] = False
                    else:
                        report["soft_warnings"].append(entry)
                continue

        report["unchecked"] += 1
        # RAG check: is the committed value close to any persona fact?
        if store is not None and str(committed).strip():
            try:
                from autofill.rag import _embed_text

                emb = await _embed_text(str(committed))
                if emb:
                    hits = await store.search_similar_persona(emb, top_k=3)
                    best = min((float(h.get("distance", 1.0)) for h in hits), default=1.0)
                    if best > 0.55:  # far from every persona fact
                        report["rag_flags"].append(
                            {
                                "label": label,
                                "filled": str(committed)[:120],
                                "score": round(best, 3),
                            }
                        )
            except Exception:
                pass

    if report["critical_mismatches"]:
        logger.warning(
            "Pre-submit consistency check FAILED",
            mismatches=report["critical_mismatches"],
        )
    else:
        logger.info(
            "Pre-submit consistency check passed",
            checked=report["checked"],
            unchecked=report["unchecked"],
            rag_flags=len(report["rag_flags"]),
        )
    return report
