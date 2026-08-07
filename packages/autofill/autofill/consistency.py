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
import re
from typing import Any

from src.logging import get_logger

logger = get_logger("autofill.consistency")

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


def _norm(value: Any) -> str:
    return re.sub(r"[\s@_.\-/+()\[\]{}]+", "", str(value or "")).lower()


def _norm_loose(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip(" .,;:!?")


def _fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


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
        else:
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
