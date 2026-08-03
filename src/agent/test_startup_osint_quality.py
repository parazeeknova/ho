"""Tests for OSINT company-name quality gates and signal pollution guards."""

from __future__ import annotations

from src.agent.startup_agent import (
    _SIGNAL_LOG,
    _company_domain,
    _is_plausible_company_name,
    _signal_is_pollution,
)


class TestPlausibleCompanyName:
    def test_legit_names_pass(self) -> None:
        for name in (
            "Lever",
            "Tripadvisor",
            "Theodo UK",
            "Tutor Intelligence",
            "Muun home",
            "Moonshot AI",
            "Anduril Industries",
        ):
            assert _is_plausible_company_name(name) is True, name

    def test_placeholders_rejected(self) -> None:
        for name in ("Unknown", "N/A", "Not specified", "Well-known tech firm", "Company"):
            assert _is_plausible_company_name(name) is False, name

    def test_headline_like_discovery_artifacts_rejected(self) -> None:
        for name in (
            "What Happens at YC - Y Combinator Though YC continu",
            "He left engineering midway and built a career with AI startu",
            "After Sonder's Collapse, Its Founder Is Trying to Engineer F",
            "VIVA Finance Triples Loan Origination Volume After Switching",
            "Tata Projects appoints Sukumar Hebbar as MD & CEO",
            "Strategic Wildlife Mitigation for Resilient Power Systems",
            "Platform Engineering Labs Advances Human-AI Infrastructure E",
            "Why every startup needs a co-founder from day one",
        ):
            assert _is_plausible_company_name(name) is False, name

    def test_domain_guard(self) -> None:
        assert _company_domain("Theodo UK") == "theodouk.com"
        assert _company_domain("Well-known tech firm") == ""
        assert _company_domain("What Happens at YC") == ""


class TestSignalPollution:
    def test_same_signal_dropped_across_companies(self) -> None:
        _SIGNAL_LOG.clear()
        first = "Open-sourced QM multiplayer agent harness for startups"
        second = "Open-sourced QM, a multiplayer agent harness for startups, on GitHub"
        assert _signal_is_pollution(first, "theodo") is False
        assert _signal_is_pollution(second, "oowlish") is True

    def test_distinct_signals_kept(self) -> None:
        _SIGNAL_LOG.clear()
        sig1 = "Launched marketplace with 55,000+ web scrapers"
        sig2 = "Raised Series A led by a16z (Mar 2025)"
        assert _signal_is_pollution(sig1, "apify") is False
        assert _signal_is_pollution(sig2, "apify") is False
