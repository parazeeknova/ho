"""High-Performance Parallel Job Parser & Extractor.

Parses raw HTML, JSON, or Markdown from JobObservations into structured
parsed dictionaries using parallel execution pools.

Target Throughput: 1,500+ to 4,000+ Job postings parsed / minute.
"""

from __future__ import annotations

import asyncio
import html
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.logging import get_logger
from src.radar.core.models import JobObservation

logger = get_logger("parallel_parser")

_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_H1_TAG_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _parse_single_observation(obs: JobObservation) -> dict[str, Any]:
    """Parse raw observation content into structured job metadata dictionary."""
    content = obs.raw_markdown or ""
    url = obs.url or ""
    title = obs.title or ""

    # Fast HTML/Markdown title extraction if not provided
    if not title or title.lower() in ("unknown", "software engineer", ""):
        m_h1 = _H1_TAG_RE.search(content)
        if m_h1:
            title = _TAG_STRIP_RE.sub("", m_h1.group(1)).strip()
        else:
            m_title = _TITLE_TAG_RE.search(content)
            if m_title:
                raw_t = _TAG_STRIP_RE.sub("", m_title.group(1)).strip()
                title = raw_t.split("|")[0].split("-")[0].strip()

    if not title:
        title = "Software Engineer"

    # Clean text content
    clean_text = _TAG_STRIP_RE.sub(" ", content)
    clean_text = html.unescape(clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # Detect remote status
    is_remote = bool(
        re.search(
            r"\b(remote|work from home|anywhere|telecommute|wfh)\b", clean_text, re.IGNORECASE
        )
    )

    # Detect location
    loc_match = re.search(
        r"\b(San Francisco|New York|London|Berlin|Seattle|Austin|"
        r"Toronto|Bangalore|Remote|San Jose|Boston)\b",
        clean_text,
        re.IGNORECASE,
    )
    location = loc_match.group(1) if loc_match else ("Remote" if is_remote else "United States")

    # Fast salary regex extraction
    sal_match = re.search(
        r"\$([0-9]{2,3}(?:,[0-9]{3})+)\s*(?:-|to)?\s*\$?([0-9]{2,3}(?:,[0-9]{3})+)?",
        clean_text,
    )
    salary_raw = sal_match.group(0) if sal_match else ""

    return {
        "url": url,
        "source": obs.source,
        "title": title,
        "clean_text": clean_text[:4000],
        "is_remote": is_remote,
        "location": location,
        "salary_raw": salary_raw,
        "observed_at": obs.observed_at,
        "extra": obs.extra,
    }


class ParallelParserEngine:
    """High-throughput parallel job parsing engine."""

    def __init__(self, max_workers: int = 16) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def parse_observations_batch(
        self,
        observations: Sequence[JobObservation],
    ) -> list[dict[str, Any]]:
        """Parse batch of JobObservations in parallel.

        Optimized for 1,500+ to 4,000+ jobs/minute.
        """
        if not observations:
            return []

        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(self.executor, _parse_single_observation, obs)
            for obs in observations
        ]

        results = await asyncio.gather(*futures, return_exceptions=True)
        valid_parsed: list[dict[str, Any]] = [
            r for r in results if isinstance(r, dict) and r.get("title")
        ]

        logger.info(f"ParallelParser: parsed {len(valid_parsed)}/{len(observations)} postings")
        return valid_parsed

    def close(self) -> None:
        self.executor.shutdown(wait=False)
