"""Tests for Pillar 1 (InstantPoller) and Pillar 2 (DorkingEngine)."""

from __future__ import annotations

from src.radar.sources.dorking import DorkingEngine
from src.radar.sources.instant_poller import InstantPoller


class TestPillar1InstantPoller:
    def test_is_job_url(self) -> None:
        poller = InstantPoller()
        assert poller._is_job_url("https://ashbyhq.com/acme/jobs/12345")
        assert poller._is_job_url("https://boards.greenhouse.io/acme/jobs/9999")
        assert not poller._is_job_url("https://acme.com/privacy")
        assert not poller._is_job_url("https://acme.com/assets/logo.png")

    def test_parse_xml_feed(self) -> None:
        poller = InstantPoller()
        sample_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://boards.greenhouse.io/acme/jobs/123456</loc>
            </url>
        </urlset>
        """
        obs = poller._parse_xml_feed("ats-acme", sample_sitemap)
        assert len(obs) == 1
        assert obs[0].url == "https://boards.greenhouse.io/acme/jobs/123456"


class TestPillar2DorkingEngine:
    def test_extract_company_from_url(self) -> None:
        engine = DorkingEngine()
        url = "https://boards.greenhouse.io/stripe/jobs/1"
        assert engine._extract_company_from_url(url) == "stripe"
        assert engine._extract_company_from_url("https://ashbyhq.com/linear/jobs/2") == "linear"
        assert engine._extract_company_from_url("https://jobs.lever.co/vercel/3") == "vercel"
