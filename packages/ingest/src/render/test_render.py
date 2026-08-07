"""Unit tests for the general-purpose render module (Firecrawl replacement)."""

import pytest

from src.render import (
    _absolutize,
    _is_job_url,
    _is_js_shell,
    markdownify,
)


def test_is_js_shell_detects_empty_and_empty_root():
    assert _is_js_shell("") is True
    assert _is_js_shell("<html><body><div id='root'></div></body></html>") is True
    assert _is_js_shell("<html><head><title>loading</title></head></html>") is True


def test_is_js_shell_false_for_rendered_content():
    html = (
        "<html><body><div id='root'>"
        "<h1>Software Engineer</h1><p>Apply now to join Replit.</p>"
        "</div><noscript>You need to enable JavaScript</noscript></body></html>"
    )
    assert _is_js_shell(html) is False


def test_is_js_shell_false_for_server_rendered():
    html = "<html><body><h1>Backend Engineer</h1><p>Stripe is hiring.</p></body></html>"
    assert _is_js_shell(html) is False


def test_is_job_url():
    assert _is_job_url("https://boards.greenhouse.io/stripe/jobs/123") is True
    assert _is_job_url("https://jobs.ashbyhq.com/replit/abc") is True
    assert _is_job_url("https://example.com/careers/software-engineer") is True
    assert _is_job_url("https://example.com/logo.png") is False
    assert _is_job_url("https://example.com/style.css") is False
    assert _is_job_url("") is False
    assert _is_job_url("https://example.com/about") is False


def test_absolutize_relative_links():
    html = '<a href="/jobs/123">Job</a><a href="stripe/engineer">x</a>'
    out = _absolutize(html, "https://boards.greenhouse.io/stripe/")
    # Root-relative stays root-relative to the origin.
    assert 'href="https://boards.greenhouse.io/jobs/123"' in out
    # Relative resolves against the base URL.
    assert 'href="https://boards.greenhouse.io/stripe/stripe/engineer"' in out


@pytest.mark.asyncio
async def test_markdownify_greenhouse_live():
    md = await markdownify("https://job-boards.greenhouse.io/gleanwork/jobs/4713977005")
    assert len(md) > 500
    assert "Glean" in md
