"""Canonical data models for the job radar v2 pipeline.

Every job enters the system as a JobObservation, passes through
deterministic gating into a JobCandidate, and is persisted with full
provenance.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any


class FreshnessLane(Enum):
    URGENT = auto()
    REVIEW = auto()
    STALE = auto()
    DROPPED = auto()


class EligibilityState(Enum):
    PENDING = auto()
    ACCEPTED = auto()
    REJECTED = auto()
    NEAR_MISS = auto()
    QUEUED = auto()
    ERROR = auto()


class RejectionReason(Enum):
    URL_BAD = "url_bad"
    URL_DUPLICATE = "url_duplicate"
    URL_DIRECTORY = "url_directory"
    URL_ERROR_404 = "url_error_404"
    URL_LANDING_PAGE = "url_landing_page"
    TITLE_SENIOR = "title_senior"
    TITLE_MANAGER = "title_manager"
    TITLE_NON_TECHNICAL = "title_non_technical"
    EXPERIENCE_HIGH = "experience_high"
    EXPERIENCE_PHD = "experience_phd"
    ROLE_FAMILY_MISMATCH = "role_family_mismatch"
    SALARY_BELOW_MIN = "salary_below_min"
    CLEARANCE_REQUIRED = "clearance_required"
    SOURCE_STALE = "source_stale"
    SOURCE_LOW_CONFIDENCE = "source_low_confidence"
    MATCHER_NO_MATCH = "matcher_no_match"
    MATCHER_LOW_SCORE = "matcher_low_score"
    UNKNOWN = "unknown"


class RoleFamily(Enum):
    BACKEND = "backend"
    INFRA_PLATFORM = "infra_platform"
    FULLSTACK_FRONTEND = "fullstack_frontend"
    GENERAL_SWE = "general_swe"
    DATA_ENGINEERING = "data_engineering"
    AI_ML = "ai_ml"
    DEVELOPER_TOOLS = "developer_tools"
    ADJACENT_TECHNICAL = "adjacent_technical"
    NON_TECHNICAL = "non_technical"
    UNKNOWN = "unknown"


@dataclass
class NormalizedSalary:
    amount: float
    currency: str
    period: str  # "hour", "month", "year"
    raw: str = ""

    @property
    def annual_usd_equivalent(self) -> float | None:
        monthly = self.to_monthly(self.currency)
        if monthly is None:
            return None
        return monthly * 12

    @staticmethod
    def to_monthly(currency: str) -> float | None:
        return _ANNUAL_TO_MONTHLY_FACTOR.get(currency.upper())


_ANNUAL_TO_MONTHLY_FACTOR: dict[str, float] = {}


@dataclass
class JobObservation:
    url: str
    source: str  # "greenhouse", "lever", "github_index", "searxng", etc.
    raw_markdown: str = ""
    title: str = ""
    snippet: str = ""
    observed_at: float = field(default_factory=time.time)
    source_freshness_evidence: str | None = None  # "posted <1h ago", etc.

    def canonical_url_hash(self) -> str:
        return _canonical_url_hash(self.url)


@dataclass
class JobCandidate:
    canonical_id: str
    source: str
    direct_apply_url: str
    normalized_company: str
    normalized_role: str
    normalized_location: str
    freshness_lane: FreshnessLane = FreshnessLane.REVIEW
    source_confidence: float = 0.5
    eligibility: EligibilityState = EligibilityState.PENDING
    rejection_reason: RejectionReason | None = None
    role_family: RoleFamily = RoleFamily.UNKNOWN
    salary: NormalizedSalary | None = None
    posted_date: str | None = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    matching_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    match_percent: int = 0
    shortlist_probability: int = 0
    verdict: str = "NO_MATCH"
    jd_summary: str = ""
    company_description: str = ""
    role_summary: str = ""
    is_remote: bool = False

    founders: list[dict[str, str]] = field(default_factory=list)
    funding_stage: str = ""
    funding_info: dict[str, Any] = field(default_factory=dict)
    founder_socials: list[str] = field(default_factory=list)
    company_news: str = ""
    osint_signals: list[str] = field(default_factory=list)

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        return self.canonical_id

    @property
    def is_urgent(self) -> bool:
        return self.freshness_lane == FreshnessLane.URGENT

    @property
    def is_rejected(self) -> bool:
        return self.eligibility == EligibilityState.REJECTED

    @property
    def is_accepted(self) -> bool:
        return self.eligibility == EligibilityState.ACCEPTED

    @property
    def is_near_miss(self) -> bool:
        return self.eligibility == EligibilityState.NEAR_MISS


@dataclass
class SourceCheckpoint:
    source_id: str
    source_type: str  # "ats_board", "company_careers", "github_index", "searxng_query"
    last_polled: float = 0.0
    last_snapshot_hash: str = ""
    last_snapshot_count: int = 0
    consecutive_failures: int = 0
    consecutive_empty: int = 0
    quality_score: float = 0.5
    active: bool = True
    backoff_until: float = 0.0
    total_jobs_produced: int = 0
    total_direct_url_rate: float = 0.0


@dataclass
class SourceState:
    checkpoint: SourceCheckpoint
    current_urls: set[str] = field(default_factory=set)
    new_urls: list[str] = field(default_factory=list)
    removed_urls: list[str] = field(default_factory=list)


@dataclass
class ColdOutreachCard:
    company: str
    why_now: str
    founder_profiles: list[dict[str, str]]
    official_contact_route: str
    role_relevance: str
    source_links: list[str]
    hiring_signal: str  # "funding", "launch", "open_role", "engineering_growth"
    confidence: float = 0.5


@dataclass
class LlmQueueItem:
    job_id: str
    priority: int  # higher = more important
    enqueued_at: float = field(default_factory=time.time)
    attempts: int = 0
    last_error: str = ""


def _canonical_url_hash(url: str) -> str:
    cleaned = _canonicalize_url(url)
    return hashlib.sha256(cleaned.encode()).hexdigest()[:16]


def _canonicalize_url(url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    try:
        parsed = urlparse(url)
    except Exception:
        return url.lower().strip()

    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]

    path = parsed.path.rstrip("/") or "/"

    query = ""
    if parsed.query:
        qp = [p for p in parsed.query.split("&") if p and not p.startswith("utm_")]
        if qp:
            query = "&".join(sorted(qp))

    return urlunparse(("https", host, path, "", query, "")).lower()


def make_canonical_id(company: str, role: str, location: str = "") -> str:
    c = "".join(ch for ch in company.lower() if ch.isalnum() or ch == ".")
    r = "".join(ch for ch in role.lower() if ch.isalnum() or ch == ".")
    loc = "".join(ch for ch in location.lower() if ch.isalnum() or ch == ".")
    return f"{c or 'unknown'}:{r or 'unknown'}:{loc or 'remote'}"


def make_observation_hash(observation: JobObservation) -> str:
    return observation.canonical_url_hash()


def now_utc_timestamp() -> float:
    return datetime.now(UTC).timestamp()
