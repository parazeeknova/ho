"""InstantPoller (Pillar 1): Sub-second instant job discovery via ATS sitemaps,
RSS/Atom feeds, and ETag/SHA256 hash delta tracking.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET

from src.http_client import get_client
from src.logging import get_logger
from src.radar.core.models import JobObservation

logger = get_logger("instant_poller")

_SITEMAP_SUFFIXES = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/feed",
    "/rss",
    "/atom.xml",
]

_XML_NS_RX = re.compile(r"\{[^}]+\}")


class InstantPoller:
    """Fast poller that uses HTTP ETags, SHA256 hashes, RSS feeds, and XML sitemaps
    to detect new job postings instantly.
    """

    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout
        self._etag_cache: dict[str, str] = {}
        self._hash_cache: dict[str, str] = {}
        self._seen_urls: set[str] = set()

    async def poll_source_instant(self, source_id: str, base_url: str) -> list[JobObservation]:
        """Polls an ATS source using sitemap/RSS feeds and hash deltas.
        Returns newly published job observations.
        """
        if not base_url or not base_url.startswith("http"):
            return []

        clean_url = base_url.rstrip("/")
        observations: list[JobObservation] = []

        # 1. Try sitemap / RSS feeds first
        for suffix in _SITEMAP_SUFFIXES:
            feed_url = clean_url + suffix
            try:
                obs = await self._fetch_feed_or_sitemap(source_id, feed_url)
                if obs:
                    observations.extend(obs)
                    break
            except Exception as e:
                logger.debug(f"Feed check failed for {feed_url}: {e}")

        # 2. Check HTML/JSON payload hash delta if no feed found
        if not observations:
            try:
                delta_obs = await self._check_hash_delta(source_id, base_url)
                observations.extend(delta_obs)
            except Exception as e:
                logger.debug(f"Hash delta check failed for {base_url}: {e}")

        return observations

    async def _fetch_feed_or_sitemap(self, source_id: str, url: str) -> list[JobObservation]:
        headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (compatible; HoRadar/1.0)",
        }
        etag = self._etag_cache.get(url)
        if etag:
            headers["If-None-Match"] = etag

        client = await get_client("instant_poller", timeout=self.timeout)
        resp = await client.get(url, headers=headers)
        if resp.status_code == 304:
            # Content unmodified since last check
            return []
        if resp.status_code != 200 or not resp.text.strip():
            return []

        new_etag = resp.headers.get("ETag") or resp.headers.get("Last-Modified")
        if new_etag:
            self._etag_cache[url] = new_etag

        body = resp.text
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if self._hash_cache.get(url) == content_hash:
            return []
        self._hash_cache[url] = content_hash

        return self._parse_xml_feed(source_id, body)

    def _parse_xml_feed(self, source_id: str, xml_content: str) -> list[JobObservation]:
        observations: list[JobObservation] = []
        try:
            root = ET.fromstring(xml_content)
        except Exception:
            return []

        # Parse <url><loc> in sitemap.xml
        for elem in root.iter():
            tag = _XML_NS_RX.sub("", elem.tag).lower()
            if tag == "loc" and elem.text:
                loc_url = elem.text.strip()
                if loc_url not in self._seen_urls and self._is_job_url(loc_url):
                    self._seen_urls.add(loc_url)
                    role_guess = loc_url.rstrip("/").split("/")[-1].replace("-", " ").title()
                    observations.append(
                        JobObservation(
                            url=loc_url,
                            source=source_id,
                            title=role_guess,
                        )
                    )
            elif tag in ("item", "entry"):
                # Parse RSS/Atom entry
                title = ""
                link = ""
                for child in elem:
                    ctag = _XML_NS_RX.sub("", child.tag).lower()
                    if ctag == "title" and child.text:
                        title = child.text.strip()
                    elif ctag == "link":
                        link = child.text.strip() if child.text else child.attrib.get("href", "")

                if link and link not in self._seen_urls:
                    self._seen_urls.add(link)
                    observations.append(
                        JobObservation(
                            url=link,
                            source=source_id,
                            title=title or "Software Role",
                        )
                    )

        return observations[:30]

    async def _check_hash_delta(self, source_id: str, base_url: str) -> list[JobObservation]:
        headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (compatible; HoRadar/1.0)",
        }
        etag = self._etag_cache.get(base_url)
        if etag:
            headers["If-None-Match"] = etag

        client = await get_client("instant_poller", timeout=self.timeout)
        resp = await client.get(base_url, headers=headers)
        if resp.status_code == 304 or resp.status_code != 200:
            return []

        new_etag = resp.headers.get("ETag")
        if new_etag:
            self._etag_cache[base_url] = new_etag

        body = resp.text
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if self._hash_cache.get(base_url) == content_hash:
            return []

        self._hash_cache[base_url] = content_hash

        # Extract job links using regex
        links = set(re.findall(r'href=["\'](https?://[^"\']+)["\']', body))
        job_links = [
            link_url
            for link_url in links
            if self._is_job_url(link_url) and link_url not in self._seen_urls
        ]

        observations: list[JobObservation] = []
        for link_url in job_links[:20]:
            self._seen_urls.add(link_url)
            role_guess = link_url.rstrip("/").split("/")[-1].replace("-", " ").title()
            observations.append(
                JobObservation(
                    url=link_url,
                    source=source_id,
                    title=role_guess,
                )
            )

        return observations

    @staticmethod
    def _is_job_url(url: str) -> bool:
        low = url.lower()
        if any(
            x in low
            for x in (
                "/jobs/",
                "/job/",
                "/positions/",
                "/careers/",
                "ashbyhq.com/",
                "boards.greenhouse.io/",
                "jobs.lever.co/",
            )
        ):
            return not any(
                skip in low for skip in (".png", ".jpg", ".css", ".js", "/privacy", "/terms")
            )
        return False
