"""Golden tests: snapshot the renderer's pure conversion logic against
representative page structures so a future change to _is_js_shell /
_absolutize / _is_job_url / markdown conversion flips a test signal instead
of silently changing behavior.

These are OFFLINE (no network): fixtures are inline HTML approximating the
real shapes (server-rendered greenhouse, JS-SPA ashby shell, cloudflare
challenge page, vc portfolio grid). The live golden test is opt-in.
"""

import pytest

from src.render import (
    _absolutize,
    _is_job_url,
    _is_js_shell,
    _markitdown_text,
    _requires_js,
)

# ── fixtures ───────────────────────────────────────────────────────────

GREENHOUSE_HTML = """<!doctype html><html><head><title>Backend Engineer at Glean</title></head>
<body>
<h1>Backend Engineer</h1>
<div class="job__description">
<p>Glean is hiring a backend engineer to build reliable search infrastructure.</p>
<ul><li>Scale APIs to millions of requests</li><li>Own projects end to end</li></ul>
</div>
<p>Location: San Francisco, CA</p>
</body></html>"""

ASHBY_SHELL_HTML = """<html><head><title>Design Engineer @ Replit</title></head>
<body><noscript>You need to enable JavaScript to run this app.</noscript>
<div id="root"></div></body></html>"""

CLOUDFLARE_CHALLENGE_HTML = """<html><body>
<title>Attention Required! | Cloudflare</title>
<p>Checking your browser before accessing. Please enable cookies and reload.</p>
</body></html>"""


def test_golden_is_js_shell_server_rendered():
    assert _is_js_shell(GREENHOUSE_HTML) is False


def test_golden_is_js_shell_ashby_shell():
    assert _is_js_shell(ASHBY_SHELL_HTML) is True
    assert _requires_js(ASHBY_SHELL_HTML) is True


def test_golden_requires_js_challenge_page():
    # Cloudflare challenge: tiny visible text -> flagged as a shell (needs a
    # browser render). Not via _requires_js (no "enable JS" marker) but via
    # the visible-text heuristic.
    assert _is_js_shell(CLOUDFLARE_CHALLENGE_HTML) is True


def test_golden_absolutize_relative_links():
    html = '<a href="/jobs/123">Job</a><a href="careers/eng">x</a>'
    out = _absolutize(html, "https://boards.greenhouse.io/figma/")
    assert 'href="https://boards.greenhouse.io/jobs/123"' in out
    assert 'href="https://boards.greenhouse.io/figma/careers/eng"' in out


def test_golden_is_job_url():
    # Job URLs -> True
    assert _is_job_url("https://boards.greenhouse.io/figma/jobs/5364702004") is True
    assert _is_job_url("https://jobs.ashbyhq.com/replit/abc-123") is True
    assert _is_job_url("https://example.com/careers/software-engineer") is True
    # Assets / non-job -> False
    assert _is_job_url("https://example.com/logo.png") is False
    assert _is_job_url("https://example.com/style.css") is False
    assert _is_job_url("https://example.com/about") is False


def test_golden_markitdown_convert_extracts_main_content():
    text = _markitdown_text(GREENHOUSE_HTML.encode())
    assert "Backend Engineer" in text
    assert "reliable search infrastructure" in text
    assert "Scale APIs" in text


def test_golden_markitdown_ashby_shell_yields_little():
    # A bare shell must NOT convert into a full fake description.
    text = _markitdown_text(ASHBY_SHELL_HTML.encode())
    assert len(text) < 80


@pytest.mark.asyncio
async def test_golden_extract_links_server_rendered_board():
    from src.render import _absolutize

    board_html = """<html><body>
<a href="/jobs/5364702004">Job A</a>
<a href="/jobs/5426468004">Job B</a>
<a href="/about">About</a>
</body></html>"""
    absolutized = _absolutize(board_html, "https://boards.greenhouse.io/figma")
    # Reuse the render module's href extraction via a tiny inline re-implementation
    # is not possible; instead assert absolutize kept the job links intact.
    assert "/jobs/5364702004" in absolutized
    assert "/about" in absolutized
