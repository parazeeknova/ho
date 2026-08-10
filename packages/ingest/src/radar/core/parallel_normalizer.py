"""High-Performance Fast Normalizer & Canonicalizer Engine.

Transforms parsed job dictionaries into canonicalized JobCandidate data models,
classifying role family, seniority, location, salary, and canonical IDs.

Target Throughput: 1,500+ to 4,000+ Jobs canonicalized / minute.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.logging import get_logger
from src.radar.core.models import (
    EligibilityState,
    JobCandidate,
    NormalizedSalary,
    RoleFamily,
    make_canonical_id,
)

logger = get_logger("parallel_normalizer")

_SENIORITY_RE = re.compile(
    r"\b(intern|internship|new grad|entry level|junior|associate|senior|staff|principal|lead)\b",
    re.IGNORECASE,
)

_ROLE_FAMILY_RULES: list[tuple[RoleFamily, re.Pattern[str]]] = [
    (
        RoleFamily.AI_ML,
        re.compile(
            r"\b(ai|ml|machine learning|deep learning|llm|genai|research scientist)\b", re.I
        ),
    ),
    (
        RoleFamily.INFRA_PLATFORM,
        re.compile(
            r"\b(infra|infrastructure|platform|devops|sre|site reliability|cloud|kubernetes)\b",
            re.I,
        ),
    ),
    (
        RoleFamily.DATA_ENGINEERING,
        re.compile(r"\b(data engineer|data engineering|etl|analytics engineer|big data)\b", re.I),
    ),
    (
        RoleFamily.FULLSTACK_FRONTEND,
        re.compile(r"\b(fullstack|full-stack|frontend|front-end|react|web|ui)\b", re.I),
    ),
    (
        RoleFamily.BACKEND,
        re.compile(r"\b(backend|back-end|api|distributed systems|go|rust|python|java)\b", re.I),
    ),
    (
        RoleFamily.GENERAL_SWE,
        re.compile(r"\b(software engineer|swe|developer|software developer)\b", re.I),
    ),
]


def _extract_company_name(url: str, source: str) -> str:
    """Fast extraction of company name from URL or source."""
    from urllib.parse import urlparse

    try:
        parsed_url = urlparse(url)
        host = parsed_url.netloc.lower()
        if host.startswith("www."):
            host = host[4:]

        # ATS domain path slug extraction (e.g., boards.greenhouse.io/stripe/jobs/123 -> Stripe)
        if any(
            ats in host
            for ats in ("greenhouse", "lever", "ashbyhq", "workable", "smartrecruiters", "rippling")
        ):
            path_parts = [p for p in parsed_url.path.split("/") if p]
            if path_parts:
                slug = path_parts[0]
                if slug not in ("jobs", "j", "v0", "v1", "postings", "accounts"):
                    return slug.replace("-", " ").title()

        parts = host.split(".")
        if len(parts) >= 2:
            comp = parts[-2]
            return comp.replace("-", " ").title()
    except Exception:
        pass
    return source.capitalize() if source else "Tech Company"


def _normalize_single_job(parsed: dict[str, Any]) -> JobCandidate:
    """Canonicalize a single parsed job dictionary into a JobCandidate object."""
    raw_title = parsed.get("title", "Software Engineer").strip()
    url = parsed.get("url", "")
    source = parsed.get("source", "web")
    company = _extract_company_name(url, source)

    # Classify role family
    role_family = RoleFamily.GENERAL_SWE
    for fam, pat in _ROLE_FAMILY_RULES:
        if pat.search(raw_title):
            role_family = fam
            break

    # Determine seniority
    m_sen = _SENIORITY_RE.search(raw_title)
    seniority = m_sen.group(1).lower() if m_sen else "mid"

    # Location & remote
    is_remote = bool(parsed.get("is_remote", False))
    location = parsed.get("location", "Remote" if is_remote else "United States")

    # Salary normalization
    salary_obj: NormalizedSalary | None = None
    sal_annual_usd: float | None = None
    sal_raw = parsed.get("salary_raw", "")
    if sal_raw:
        nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", sal_raw)]
        if nums:
            avg_num = sum(nums) / len(nums)
            if avg_num > 30000:
                salary_obj = NormalizedSalary(
                    amount=avg_num, currency="USD", period="year", raw=sal_raw
                )
                sal_annual_usd = avg_num

    # Create canonical ID
    cid = make_canonical_id(raw_title, company, location)

    candidate = JobCandidate(
        canonical_id=cid,
        source=source,
        normalized_role=raw_title,
        normalized_company=company,
        normalized_location=location,
        direct_apply_url=url,
        role_family=role_family,
        eligibility=EligibilityState.PENDING,
        salary=salary_obj,
        salary_annual_usd=sal_annual_usd,
        is_remote=is_remote,
        first_seen=parsed.get("observed_at", time.time()),
        last_seen=parsed.get("observed_at", time.time()),
        extra={
            "raw_markdown": parsed.get("clean_text", ""),
            "seniority": seniority,
        },
    )
    return candidate


class ParallelNormalizerEngine:
    """High-performance parallel job normalization & canonicalization engine."""

    def __init__(self, max_workers: int = 16) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def normalize_parsed_batch(
        self,
        parsed_batch: Sequence[dict[str, Any]],
    ) -> list[JobCandidate]:
        """Canonicalize batch of parsed dictionaries into JobCandidates in parallel.

        Optimized for 1,500+ to 4,000+ jobs/minute.
        """
        if not parsed_batch:
            return []

        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(self.executor, _normalize_single_job, p) for p in parsed_batch
        ]

        results = await asyncio.gather(*futures, return_exceptions=True)
        valid_candidates: list[JobCandidate] = [c for c in results if isinstance(c, JobCandidate)]

        logger.info(
            "ParallelNormalizer: canonicalized candidates",
            canonicalized=len(valid_candidates),
            total=len(parsed_batch),
        )
        return valid_candidates

    def close(self) -> None:
        self.executor.shutdown(wait=False)
