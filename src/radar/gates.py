"""Deterministic pre-LLM gating engine.

Every job observation passes through these gates before consuming any LLM
budget. Gates are ordered from cheapest to most expensive.
"""

from __future__ import annotations

import re
from typing import Any

from src.radar.models import (
    EligibilityState,
    FreshnessLane,
    JobCandidate,
    JobObservation,
    RejectionReason,
    RoleFamily,
    make_canonical_id,
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


async def run_gates(
    observation: JobObservation,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> tuple[JobCandidate | None, list[tuple[str, RejectionReason, str]]]:
    rejection_log: list[tuple[str, RejectionReason, str]] = []

    candidate = JobCandidate(
        canonical_id=make_canonical_id(observation.title or "unknown", "", "Remote"),
        source=observation.source,
        direct_apply_url=observation.url,
        normalized_company=observation.title or "unknown",
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
    r"\baccount\s+executive\b",
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
    combined = f"{title} {snippet}"

    for pat in _NON_TECH_TITLE_PATTERNS:
        if re.search(pat, combined):
            return RejectionReason.TITLE_NON_TECHNICAL

    for pat, _label in _SENIOR_TITLE_PATTERNS:
        if re.search(pat, title):
            if _label == "manager" and _MANAGER_WHITELIST.search(title):
                continue
            return (
                RejectionReason.TITLE_SENIOR
                if _label != "manager"
                else RejectionReason.TITLE_MANAGER
            )

    internship_kw = re.search(r"\b(?:intern|internship|co-?op|coop)\b", combined)
    newgrad_kw = re.search(r"\b(?:new\s*grad|entry.level|junior|early.career|graduate)\b", combined)
    if internship_kw or newgrad_kw:
        return None

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
            candidate.normalized_role = obs.title
            return None

    internship_newgrad = re.search(
        r"\b(?:intern|internship|co-?op|coop|new\s*grad|entry.level|junior|early.career|graduate)\b",
        combined,
    )
    if internship_newgrad:
        candidate.role_family = RoleFamily.GENERAL_SWE
        candidate.normalized_role = obs.title
        return None

    return RejectionReason.ROLE_FAMILY_MISMATCH


_EXPLICIT_EXPERIENCE_PATTERNS = [
    (re.compile(r"\b5\+?\s*years?\b", re.IGNORECASE), "5_years"),
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
    return None


def gate_source_freshness(
    obs: JobObservation,
    candidate: JobCandidate,
    known_hashes: set[str],
    last_seen: dict[str, float],
) -> RejectionReason | None:
    """Assign freshness lane from evidence and observation history.

    URGENT when:
      - Source provides a verified timestamp within 24 hours.
      - First-seen from an already-monitored official source.

    REVIEW for strong roles with unknown posting date.
    STALE for verified-old postings.
    """
    import time

    from src.configuration import get_config

    cfg = get_config().radar

    url_hash = obs.canonical_url_hash()
    prev_seen = last_seen.get(url_hash, 0.0)
    now = time.time()
    window_secs = cfg.urgent_window_hours * 3600

    has_timestamp_evidence = bool(obs.source_freshness_evidence)

    is_first_seen_from_monitored = prev_seen == 0 and obs.source not in (
        "github_index",
        "searxng",
        "unknown",
    )

    if has_timestamp_evidence or is_first_seen_from_monitored:
        candidate.freshness_lane = FreshnessLane.URGENT
        return None

    if prev_seen > 0:
        age = now - prev_seen
        if age > cfg.stale_days * 86400:
            candidate.freshness_lane = FreshnessLane.STALE
            return RejectionReason.SOURCE_STALE
        if age < window_secs:
            candidate.freshness_lane = FreshnessLane.URGENT

    candidate.freshness_lane = FreshnessLane.REVIEW
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
    RejectionReason.SOURCE_STALE: "Source has not produced fresh content",
    RejectionReason.SOURCE_LOW_CONFIDENCE: "Source quality score is too low",
    RejectionReason.MATCHER_NO_MATCH: "LLM matcher returned NO_MATCH verdict",
    RejectionReason.MATCHER_LOW_SCORE: "LLM match score below threshold",
    RejectionReason.UNKNOWN: "Unknown rejection reason",
}


def _describe_rejection(reason: RejectionReason) -> str:
    return _REJECTION_DESCRIPTIONS.get(reason, "Unknown")
