"""Deterministic pre-LLM gating engine.

Every job observation passes through these gates before consuming any LLM
budget. Gates are ordered from cheapest to most expensive.
"""

from __future__ import annotations

import re
import time
from typing import Any

from src.configuration import get_config
from src.radar.core.models import (
    EligibilityState,
    FreshnessLane,
    JobCandidate,
    JobObservation,
    RejectionReason,
    RoleFamily,
    make_canonical_id,
)

_INDEX_SOURCE_PREFIXES = ("github_index:", "searxng", "unknown")
_MONITORED_SOURCE_PREFIXES = (
    "greenhouse",
    "lever",
    "ashby",
    "workable",
    "smartrecruiters",
    "workday",
    "rippling",
    "ats:",
)

_GATE_ORDER: list[str] = [
    "url_quality",
    "url_duplicate",
    "title_seniority",
    "role_family",
    "salary",
    "explicit_experience",
    "explicit_ineligibility",
    "source_freshness",
]


_META_SOURCE_PREFIXES = (
    "ats:",
    "mass_poller",
    "github_index",
    "searxng",
    "unknown",
    "discovered",
    "blog",
    "news",
)


def _extract_company(observation: JobObservation) -> str:
    """Best-effort company name from the observation's ATS slug, source id,
    or URL hostname - never from the job title."""
    slug = observation.extra.get("company_slug")
    if slug:
        return str(slug).replace("-", " ").title()

    src = observation.source or ""
    if ":" in src:
        maybe = src.split(":", 1)[0].strip()
        if maybe and maybe not in _META_SOURCE_PREFIXES:
            return maybe.replace("-", " ").title()
    elif src and src not in _META_SOURCE_PREFIXES:
        return src.replace("-", " ").title()

    from urllib.parse import urlparse

    try:
        host = urlparse(observation.url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        parts = host.split(".")
        if len(parts) >= 2:
            return parts[-2].replace("-", " ").title()
    except Exception:
        pass
    return "Unknown"


async def run_gates(
    observation: JobObservation,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> tuple[JobCandidate | None, list[tuple[str, RejectionReason, str]]]:
    rejection_log: list[tuple[str, RejectionReason, str]] = []

    company = _extract_company(observation)
    candidate = JobCandidate(
        canonical_id=make_canonical_id(observation.title or "unknown", company, "Remote"),
        source=observation.source,
        direct_apply_url=observation.url,
        normalized_company=company,
        normalized_role="",
        normalized_location="Remote",
    )

    for gate_name in _GATE_ORDER:
        handler = _GATE_HANDLERS.get(gate_name)
        if handler is None:
            continue
        result = handler(observation, candidate, known_hashes, last_seen)
        if isinstance(result, RejectionReason):
            rejection_log.append((gate_name, result, _describe_rejection(result)))
            candidate.eligibility = EligibilityState.REJECTED
            candidate.rejection_reason = result
            return None, rejection_log

    return candidate, rejection_log


def gate_url_quality(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    url = obs.url
    if not url or not url.startswith("http"):
        return RejectionReason.URL_BAD

    url_lower = url.lower()
    _error_patterns = [
        "/404",
        "/error",
        "not-found",
        "page-not-found",
        "job-not-found",
        "position-filled",
        "no-longer-available",
    ]
    for pat in _error_patterns:
        if pat in url_lower:
            return RejectionReason.URL_ERROR_404

    _image_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico")
    if any(url_lower.endswith(ext) for ext in _image_exts):
        return RejectionReason.URL_BAD

    _directory_domains = (
        "internshala.com",
        "web3.career",
        "glassdoor.com",
        "indeed.com",
        "ziprecruiter.com",
        "simplyhired.com",
        "linkedin.com/jobs/collections",
        "remoteok.com",
    )
    for domain in _directory_domains:
        if domain in url_lower:
            return RejectionReason.URL_DIRECTORY

    _landing_paths = ("/jobs", "/careers", "/positions", "/")
    from urllib.parse import urlparse

    try:
        path = urlparse(url).path.lower().rstrip("/") or "/"
    except Exception:
        return RejectionReason.URL_BAD
    if path in _landing_paths:
        return RejectionReason.URL_LANDING_PAGE

    return None


def gate_url_duplicate(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    url_hash = obs.canonical_url_hash()
    if url_hash in known_hashes:
        return RejectionReason.URL_DUPLICATE
    return None


_SENIOR_TITLE_PATTERNS = [
    (r"\bsenior\b", "senior"),
    (r"\bsr\.?\b", "senior"),
    (r"\bstaff\b", "staff"),
    (r"\bdirector\b", "director"),
    (r"\bvp\b", "vp"),
    (r"\bvice\s+president\b", "vp"),
    (r"\bhead\s+of\b", "head_of"),
    (r"\bprincipal\b", "principal"),
    (r"\barchitect\b", "architect"),
    (r"\bmanager\b", "manager"),
]

_MANAGER_WHITELIST = re.compile(r"\b(product|project|program|community)\b", re.IGNORECASE)

_NON_TECH_TITLE_PATTERNS = [
    r"\bcontent\s+creator\b",
    r"\bhost\s+live\b",
    r"\bsales\s+(provider|executive|representative|manager|director)\b",
    r"\bproperty\s+development\b",
    r"\baccount\s+(?:executive|manager)\b",
    r"\bmarketing\b",
    r"\brecruiter\b",
    r"\bcustomer\s+(service|support)\b",
    r"\btelemarketing\b",
    r"\bsocial\s+media\b",
    r"\badministrative\s+assistant\b",
    r"\bstore\s+manager\b",
    r"\bcashier\b",
    r"\bdriver\b",
    r"\bchef\b",
    r"\bnurse\b",
    r"\breceptionist\b",
]


def gate_title_seniority(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    title = obs.title.lower()
    snippet = obs.snippet.lower()
    combined = f"{title} {snippet} {obs.raw_markdown[:1000]}".lower()

    for pat in _NON_TECH_TITLE_PATTERNS:
        if re.search(pat, combined):
            return RejectionReason.TITLE_NON_TECHNICAL

    # Semantic Override: Early-career / junior indicators override senior keyword drops
    early_career_override = re.search(
        r"\b(?:intern|internship|co-?op|coop|new\s*grad|entry.level|junior|early.career|graduate|0-1|0-2|0-3|1-2|1-3|associate)\b",
        combined,
    )
    if early_career_override:
        return None

    for pat, _label in _SENIOR_TITLE_PATTERNS:
        if re.search(pat, title):
            if _label == "manager" and _MANAGER_WHITELIST.search(title):
                continue
            return (
                RejectionReason.TITLE_SENIOR
                if _label != "manager"
                else RejectionReason.TITLE_MANAGER
            )

    return None


_TECH_ROLE_MAP: list[tuple[str, RoleFamily]] = [
    (
        r"\b(?:backend|back-end|back\s*end)\b",
        RoleFamily.BACKEND,
    ),
    (
        r"\b(?:systems?\s*engineer|distributed|sre|devops|infra|"
        r"infrastructure|platform|cloud|site\s*reliability)\b",
        RoleFamily.INFRA_PLATFORM,
    ),
    (
        r"\b(?:frontend|front-end|front\s*end|fullstack|"
        r"full-stack|full\s*stack|web\s*developer)\b",
        RoleFamily.FULLSTACK_FRONTEND,
    ),
    (
        r"\b(?:software\s*engineer|software\s*developer|sde|swe)\b",
        RoleFamily.GENERAL_SWE,
    ),
    (
        r"\b(?:data\s*engineer|etl|data\s*pipeline|data\s*infra)\b",
        RoleFamily.DATA_ENGINEERING,
    ),
    (
        r"\b(?:machine\s*learning|ml\s*engineer|ai\s*engineer|rag|llm|"
        r"nlp|computer\s*vision|applied\s*scientist)\b",
        RoleFamily.AI_ML,
    ),
    (
        r"\b(?:developer\s*tools?|dev\s*tools?|dx|tooling|"
        r"developer\s*experience|developer\s*relations|devrel)\b",
        RoleFamily.DEVELOPER_TOOLS,
    ),
    (
        r"\b(?:qa|quality|test\s*engineer|sd[ea]t|security\s*engineer|"
        r"seceng|technical\s*writer|technical\s*support)\b",
        RoleFamily.ADJACENT_TECHNICAL,
    ),
]


def gate_role_family(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    combined = f"{obs.title} {obs.snippet} {obs.raw_markdown[:2000]}".lower()

    for pat, family in _TECH_ROLE_MAP:
        if re.search(pat, combined):
            candidate.role_family = family
            candidate.normalized_role = obs.title or "Software Engineer"
            return None

    for pat in _NON_TECH_TITLE_PATTERNS:
        if re.search(pat, combined):
            return RejectionReason.ROLE_FAMILY_MISMATCH

    # Resilient Fallback: Any technical posting defaults to GENERAL_SWE instead of dropping
    candidate.role_family = RoleFamily.GENERAL_SWE
    candidate.normalized_role = obs.title or "Software Engineer"
    return None


_EXPLICIT_EXPERIENCE_PATTERNS = [
    (re.compile(r"\b7\+?\s*years?\b", re.IGNORECASE), "7_years"),
    (re.compile(r"\b10\+?\s*years?\b", re.IGNORECASE), "10_years"),
    (re.compile(r"\b\d{2,}\s*\+\s*years?\b", re.IGNORECASE), "many_years"),
    (re.compile(r"\bph\.?d\b", re.IGNORECASE), "phd"),
    (re.compile(r"\bdoctorate\b", re.IGNORECASE), "doctorate"),
    (re.compile(r"\bpostdoc\b", re.IGNORECASE), "postdoc"),
    (re.compile(r"\bclearance\b", re.IGNORECASE), "clearance"),
    (re.compile(r"\bcitizenship\s*required\b", re.IGNORECASE), "citizenship"),
]


def gate_explicit_experience(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    text = f"{obs.title} {obs.snippet} {obs.raw_markdown[:5000]}".lower()

    # Negation & Early-Career Override: "0-2 years", "don't need 5 years", "not required"
    negation_or_junior = re.search(
        r"\b(?:0-1|0-2|0-3|1-2|1-3|0\s+to\s+[23]|don'?t\s+need|no\s+experience|not\s+required|entry\s+level)\b",
        text,
    )
    if negation_or_junior:
        return None

    for pat, label in _EXPLICIT_EXPERIENCE_PATTERNS:
        if pat.search(text):
            if label in ("phd", "doctorate", "postdoc"):
                return RejectionReason.EXPERIENCE_PHD
            if label in ("clearance", "citizenship"):
                return RejectionReason.CLEARANCE_REQUIRED
            return RejectionReason.EXPERIENCE_HIGH

    return None


def gate_explicit_ineligibility(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    """Detect clear no-gos: no-sponsorship, citizenship-only, explicit ineligibility."""
    text = f"{obs.title} {obs.snippet} {obs.raw_markdown[:5000]}".lower()

    _no_sponsor_pats = [
        r"\b(?:no|not\s+able\s+to)\s+sponsor\b",
        r"\bdoes\s+not\s+sponsor\b",
        r"\bunable\s+to\s+sponsor\b",
        r"\bsponsorship\s*(?:is|)\s*not\s+(?:available|provided|offered)\b",
        r"\bmust\s+be\s+(?:a\s+)?(?:us|u\.s\.|united\s+states)\s+(?:citizen|person)\b",
        r"\b(?:us|u\.s\.)\s+citizens?\s+only\b",
        r"\bno\s+(?:visa|h1b|h-1b)\s+sponsorship\b",
    ]
    for pat in _no_sponsor_pats:
        if re.search(pat, text):
            return RejectionReason.NO_SPONSORSHIP

    return None


def gate_source_freshness(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    """Assign freshness lane from evidence and observation history.

    URGENT when:
      - Source provides a verified posting timestamp ≤24 hours old.
      - The URL is a delta from a previously persisted official-source snapshot.

    REVIEW for unknown posting date or first-ever source crawl.
    STALE for verified-old postings > stale_days.
    Index/discovery sources are never URGENT on first-seen alone.
    """
    cfg = get_config().radar

    url_hash = obs.canonical_url_hash()
    prev_seen = last_seen.get(url_hash, 0.0)
    now = time.time()
    window_secs = cfg.urgent_window_hours * 3600

    age_from_evidence = _parse_freshness_evidence(obs.source_freshness_evidence)

    is_official = obs.extra.get("official_source", False)
    is_snapshot_delta = obs.extra.get("is_snapshot_delta", False)
    is_baseline_crawl = prev_seen == 0

    if age_from_evidence is not None:
        candidate.posted_date = obs.source_freshness_evidence
        if age_from_evidence <= window_secs:
            candidate.freshness_lane = FreshnessLane.URGENT
            return None
        if age_from_evidence > cfg.stale_days * 86400:
            candidate.freshness_lane = FreshnessLane.STALE
            return RejectionReason.SOURCE_STALE

    if is_snapshot_delta and is_official:
        candidate.freshness_lane = FreshnessLane.URGENT
        return None

    if not is_baseline_crawl:
        age = now - prev_seen
        if age > cfg.stale_days * 86400:
            candidate.freshness_lane = FreshnessLane.STALE
            return RejectionReason.SOURCE_STALE

    return None


def _parse_freshness_evidence(evidence: str | None) -> float | None:
    """Parse freshness evidence into age in seconds, or None if unparseable."""
    if not evidence:
        return None
    import re as _re

    t = _re.search(
        r"posted\s+(\d+)\s*(hour|min|minute|second|day|week|month)s?\s+ago",
        evidence,
        _re.IGNORECASE,
    )
    if t:
        num = int(t.group(1))
        unit = t.group(2).lower()
        multipliers = {
            "second": 1,
            "min": 60,
            "minute": 60,
            "hour": 3600,
            "day": 86400,
            "week": 604800,
            "month": 2592000,
        }
        return num * multipliers.get(unit, 3600)

    iso = _re.search(r"\d{4}-\d{2}-\d{2}", evidence)
    if iso:
        try:
            from datetime import UTC
            from datetime import datetime as _dt

            posted = _dt.fromisoformat(iso.group(0)).replace(tzinfo=UTC)
            age = (_dt.now(UTC) - posted).total_seconds()
            return max(0, age)
        except Exception:
            return None

    rel = _re.search(r"(\d+)\s*(hour|min|minute|day|week|month)s?\s+ago", evidence, _re.IGNORECASE)
    if rel:
        num = int(rel.group(1))
        unit = rel.group(2).lower()
        multipliers = {
            "min": 60,
            "minute": 60,
            "hour": 3600,
            "day": 86400,
            "week": 604800,
            "month": 2592000,
        }
        return num * multipliers.get(unit, 3600)

    return None


def gate_salary_check(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    return None


_GATE_HANDLERS: dict[str, Any] = {
    "url_quality": gate_url_quality,
    "url_duplicate": gate_url_duplicate,
    "title_seniority": gate_title_seniority,
    "role_family": gate_role_family,
    "salary": gate_salary_check,
    "explicit_experience": gate_explicit_experience,
    "explicit_ineligibility": gate_explicit_ineligibility,
    "source_freshness": gate_source_freshness,
}


_REJECTION_DESCRIPTIONS: dict[RejectionReason, str] = {
    RejectionReason.URL_BAD: "URL is malformed or points to non-job content",
    RejectionReason.URL_DUPLICATE: "URL hash already seen in this sweep",
    RejectionReason.URL_DIRECTORY: "URL is a known job directory/aggregator",
    RejectionReason.URL_ERROR_404: "URL appears to be a 404 or error page",
    RejectionReason.URL_LANDING_PAGE: "URL is a generic landing page, not a direct posting",
    RejectionReason.TITLE_SENIOR: "Role title contains senior/staff/lead keyword",
    RejectionReason.TITLE_MANAGER: "Role title is a management position",
    RejectionReason.TITLE_NON_TECHNICAL: "Role is clearly non-technical",
    RejectionReason.EXPERIENCE_HIGH: "JD requires 5+ years of experience",
    RejectionReason.EXPERIENCE_PHD: "JD requires PhD/doctorate",
    RejectionReason.ROLE_FAMILY_MISMATCH: "Role does not match target families",
    RejectionReason.SALARY_BELOW_MIN: "Salary is below candidate minimum",
    RejectionReason.CLEARANCE_REQUIRED: "Requires security clearance or citizenship",
    RejectionReason.NO_SPONSORSHIP: "Company explicitly does not sponsor visas",
    RejectionReason.SOURCE_STALE: "Source has not produced fresh content",
    RejectionReason.SOURCE_LOW_CONFIDENCE: "Source quality score is too low",
    RejectionReason.MATCHER_NO_MATCH: "LLM matcher returned NO_MATCH verdict",
    RejectionReason.MATCHER_LOW_SCORE: "LLM match score below threshold",
    RejectionReason.UNKNOWN: "Unknown rejection reason",
}


def _describe_rejection(reason: RejectionReason) -> str:
    return _REJECTION_DESCRIPTIONS.get(reason, "Unknown")
