"""Pre-LLM deterministic signal extraction from posting text.

Extracts salary, location, sponsorship/relocation/EOR, and
international-remote evidence before any LLM call.
"""

from __future__ import annotations

import re

from src.radar.core.models import NormalizedSalary
from src.radar.core.salary import normalize_salary

_SPONSOR_PATTERNS = [
    r"\bsponsor\b",
    r"\brelocation\b",
    r"\be-?verify\b",
    r"\bglobal\s+remote\b",
    r"\bwork\s+from\s+anywhere\b",
    r"\bvisa\s+(transfer|sponsorship)\b",
    r"\bh-?1b\b",
    r"\bemployer\s+of\s+record\b",
    r"\beor\b",
    r"\binternational\s+(ok|okay|friendly)\b",
    r"\bworldwide\b",
    r"\bremote\s+(ok|okay|friendly|first|friendly)\b",
    r"\bwork\s+from\s+(home|anywhere)\b",
]

_NO_SPONSOR_PATS = [
    r"\bno\s+sponsor\b",
    r"\bnot\s+sponsor\b",
    r"\bdoes\s+not\s+sponsor\b",
    r"\bunable\s+to\s+sponsor\b",
    r"\bsponsorship\s+not\s+available\b",
    r"\bmust\s+be\s+(?:a\s+)?(?:us|u\.s\.|united\s+states)\s+(?:citizen|person)\b",
    r"\b(?:us|u\.s\.)\s+citizens?\s+only\b",
    r"\bno\s+(?:visa|h1b|h-1b)\b",
]

# US location detection for location-eligibility filtering and card warnings.
_US_STATE_CODES = {
    "al",
    "ak",
    "az",
    "ar",
    "ca",
    "co",
    "ct",
    "de",
    "fl",
    "ga",
    "hi",
    "id",
    "il",
    "in",
    "ia",
    "ks",
    "ky",
    "la",
    "me",
    "md",
    "ma",
    "mi",
    "mn",
    "ms",
    "mo",
    "mt",
    "ne",
    "nv",
    "nh",
    "nj",
    "nm",
    "ny",
    "nc",
    "nd",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "vt",
    "va",
    "wa",
    "wv",
    "wi",
    "wy",
    "dc",
    "pr",
}
_US_CITIES = {
    "new york",
    "san francisco",
    "los angeles",
    "seattle",
    "austin",
    "boston",
    "chicago",
    "denver",
    "portland",
    "phoenix",
    "san diego",
    "dallas",
    "houston",
    "miami",
    "atlanta",
    "washington",
    "philadelphia",
    "detroit",
    "minneapolis",
    "raleigh",
    "nashville",
    "salt lake city",
    "pittsburgh",
    "boulder",
    "madison",
    "boise",
    "charlotte",
    "tampa",
    "orlando",
    "cincinnati",
    "cleveland",
    "kansas city",
    "st. louis",
    "milwaukee",
    "indianapolis",
    "columbus",
    "san jose",
    "santa clara",
    "mountain view",
    "palo alto",
    "sunnyvale",
    "redwood city",
    "bellevue",
    "irvine",
    "santa monica",
    "menlo park",
    "cambridge",
    "jersey city",
    "brooklyn",
}
_US_MARKERS = {
    "united states",
    "u.s.",
    "u.s.a",
    "usa",
    "us only",
    "us remote",
    "remote (us)",
    "remote us",
    "us based",
}


def is_us_location(location: str) -> bool:
    """Heuristic: does a normalized location string point at the US?

    Triggers on explicit markers ("united states", "usa", "remote (us)"),
    a standalone US state code (e.g. "San Francisco, CA"), or a major US
    city when the string is short (avoiding false positives on ambiguous
    long descriptions).
    """
    if not location:
        return False
    loc = location.lower().strip()
    if len(loc) > 120:
        loc = loc[:120]
    for marker in _US_MARKERS:
        if marker in loc:
            return True
    if re.search(r"\bus\b", loc):
        return True
    tokens = re.findall(r"\b([a-z]{2})\b", loc)
    for tok in tokens:
        if tok in _US_STATE_CODES:
            return True
    if len(loc) < 60:
        for city in _US_CITIES:
            if city in loc:
                return True
    return False


def extract_signals(markdown: str, title: str = "") -> dict:
    """Extract all deterministic signals from raw posting text."""
    text = (title + " " + markdown[:10000]).lower()

    salary = _extract_salary_from_text(markdown)
    sponsors = _extract_sponsorship(text)
    remote = _check_remote(text)

    return {
        "salary": salary,
        "salary_annual_usd": _salary_to_annual_usd(salary),
        "sponsors_visa": sponsors,
        "is_remote": remote,
    }


def _extract_salary_from_text(text: str) -> NormalizedSalary | None:
    """Find salary in first 3000 chars of posting text."""
    lines = text.split("\n")
    # Prioritize lines that look like salary lines
    salary_keywords = ("salary", "compensation", "pay range", "💰", "$", "₹", "€", "£")
    for line in lines[:60]:
        line_s = line.strip()
        if any(kw in line_s.lower() for kw in salary_keywords):
            s = normalize_salary(line_s)
            if s:
                return s
    # Fallback: scan full first section
    return normalize_salary(text[:3000])


def _extract_sponsorship(text: str) -> bool:
    """Check if posting explicitly offers visa sponsorship/relocation."""
    for pat in _NO_SPONSOR_PATS:
        if re.search(pat, text):
            return False
    return any(re.search(pat, text) for pat in _SPONSOR_PATTERNS)


def _check_remote(text: str) -> bool:
    remote_pats = [
        r"\bremote\b",
        r"\bwork\s+from\s+home\b",
        r"\bwork\s+from\s+anywhere\b",
        r"\bdistributed\b",
        r"\bvirtual\b",
        r"\btelecommut",
    ]
    return any(re.search(p, text) for p in remote_pats)


def _salary_to_annual_usd(salary: NormalizedSalary | None) -> float | None:
    if salary is None:
        return None
    try:
        rate = {"hour": 2000, "month": 12, "year": 1}.get(salary.period)
        if rate is None:
            return None
        fx = {"USD": 1.0, "INR": 1.0 / 86, "EUR": 1.1, "GBP": 1.25}.get(
            salary.currency,
            1.0,
        )
        return salary.amount * rate * fx
    except Exception:
        return None
