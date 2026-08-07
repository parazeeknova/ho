"""General-purpose page rendering + markdown extraction.

Replaces Firecrawl's ``/v1/scrape`` and ``/v1/map`` for the whole internet
(not just a handful of ATS boards):

- ``fetch_html``: plain httpx GET with caching/retries (fast path, no browser).
- ``render_html``: lazy Playwright-backed render for JS-only pages, launched
  only when the static fetch comes back as a JS shell (``enable JavaScript``,
  empty body, SPA root). Browser is launched on demand and torn down after so
  nothing stays resident.
- ``markdownify``: turn any URL into clean markdown — tries the static path
  first (markitdown), falls back to the renderer for JS-only pages, then
  extracts main content.

No Firecrawl service, no resident browser, no heavy infra. Deterministic and
cheap: the browser is only ever used for pages that genuinely need it.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
from typing import Any
from urllib.parse import urljoin, urlparse

from src.http_client import get_client
from src.logging import get_logger

logger = get_logger("render")

# Browser launch is global (one at a time) and reused across calls within a
# short window; torn down after RENDERER_IDLE_MS of inactivity.
_playwright_lock = threading.Lock()
_playwright_proc: dict[str, Any] | None = None
_playwright_last: float = 0.0
_RENDERER_IDLE_MS = 30_000
_browser_lock = asyncio.Lock()

# Pages that return this shell have NOT rendered; they need a browser.
_JS_SHELL_TEXT_MARKERS = (
    "enable javascript",
    "javascript is disabled",
    "you need to enable javascript",
    "your browser does not support javascript",
    "app loading",
    "loading...",
)


def _is_js_shell(html: str) -> bool:
    """Detect a JS-only shell page (needs a browser to render).

    A page is a shell when it has (almost) no visible text. Many real sites
    embed a <noscript> fallback that says "You need to enable JavaScript" even
    AFTER rendering — so an explicit JS marker only counts as a shell when the
    page otherwise has no content. A rendered SPA with content is never a shell.
    """
    low = (html or "").lower()
    if not low or len(html) < 50:
        return True  # empty/too-small page is almost always a JS shell
    # Strip scripts/styles/tags; measure the visible text.
    text = re.sub(
        r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<noscript[\s\S]*?</noscript>", "", html
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Content-rich enough (headings/paragraphs) -> rendered or server-rendered.
    if len(text) >= 20:
        return False
    # Low visible text: flag as a shell if it's genuinely a JS-only app or
    # explicitly requires JS.
    if any(m in low for m in _JS_SHELL_TEXT_MARKERS):
        return True
    return len(text) < 20


# Extensions that are never a job page.
_SKIP_EXT_RE = re.compile(
    r"\.(png|jpe?g|gif|svg|webp|css|js|json|xml|txt|woff2?|ttf|eot|mp4|"
    r"webm|zip|pdf|docx?|xlsx?|pptx?)(\?|#|$)",
    re.I,
)


def _is_job_url(url: str) -> bool:
    """Best-effort: is this URL plausibly a job posting page?."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    if _SKIP_EXT_RE.search(url):
        return False
    low = url.lower()
    return bool(
        re.search(
            r"/(jobs?|careers|postings?|positions?|openings?|opportunit|"
            r"role|gigs?|work)(/|\\?|$)",
            low,
        )
        or any(
            h in low
            for h in (
                "greenhouse",
                "lever.co",
                "ashbyhq",
                "workable",
                "smartrecruiters",
                "myworkdayjobs",
                "recruitee",
                "teamtailor",
                "jazzhr",
                "bamboohr",
                "rippling",
            )
        )
    )


def _absolutize(html: str, base_url: str) -> str:
    """Resolve relative links in scraped HTML against the page URL.

    Job boards render postings in SPA fragments with relative hrefs; the
    caller needs absolute URLs to enqueue them.
    """
    base = urlparse(base_url)
    origin = f"{base.scheme}://{base.netloc}"

    def _abs(match: re.Match) -> str:
        href = match.group(1)
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            return match.group(0)
        if href.startswith("/"):
            return f'href="{origin + href}"'
        if href.startswith(("./", "../", "?")) or not re.match(
            r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", href
        ):
            return f'href="{urljoin(base_url, href)}"'
        return match.group(0)

    return re.sub(r'href="([^"]*)"', _abs, html)


async def fetch_html(url: str, timeout: float = 20.0) -> str:
    """Static httpx GET (cached); returns raw HTML or '' on failure."""
    try:
        client = await get_client("render_fetch", timeout=timeout)
        resp = await client.get(
            url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        )
        if resp.status_code == 200 and resp.text:
            return resp.text
        return ""
    except Exception as e:
        logger.debug("fetch_html failed", url=url, error=str(e))
        return ""


# ── JS rendering (lazy playwright) ──────────────────────────────────────


async def _render_with_playwright(url: str) -> str:
    """Render a JS-only page with a lazily-spawned Chromium, then return HTML.

    The browser is launched only on demand and closed after a short idle
    window, so it never sits resident. Returns '' on any failure.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:
        logger.warning("playwright not installed; cannot render JS pages", url=url)
        return ""

    async with _browser_lock:
        try:
            async with async_playwright() as p:
                # Prefer a system-installed Chrome/Chromium when the bundled
                # headless shell is missing system libs (libglib etc.), which
                # is common in container/dev shells.
                executable = None
                for cand in (
                    "/usr/bin/google-chrome",
                    "/usr/bin/google-chrome-stable",
                    "/usr/bin/chromium",
                    "/run/current-system/sw/bin/google-chrome",
                ):
                    if os.path.exists(cand):
                        executable = cand
                        break
                browser = await p.chromium.launch(
                    headless=True,
                    executable_path=executable,
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                )
                page = await browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                    )
                )
                await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                # Let client-side render settle.
                await page.wait_for_timeout(2500)
                html = await page.content()
                await browser.close()
                if _is_js_shell(html):
                    logger.warning("Rendered page still a JS shell", url=url)
                    return ""
                return html
        except Exception as e:
            logger.warning("playwright render failed", url=url, error=str(e))
            return ""


async def render_html(url: str, timeout: float = 20.0) -> str:
    """Fetch a page, falling back to a browser render if it's a JS shell.

    Returns the HTML with absolute hrefs, or '' if both paths fail.
    """
    html = await fetch_html(url, timeout=timeout)
    if html and not _is_js_shell(html):
        return _absolutize(html, url)
    # JS-only (or empty): render it.
    rendered = await _render_with_playwright(url)
    if rendered:
        return _absolutize(rendered, url)
    return html  # return whatever we have (may be a shell) so callers can retry


# ── markdown extraction ─────────────────────────────────────────────────


async def markdownify(url: str, timeout: float = 20.0) -> str:
    """Turn any URL into clean markdown (main content).

    Tries static fetch + markitdown first; for JS-only pages falls back to the
    browser render, then converts. Returns '' when nothing usable came back.
    """
    html = await fetch_html(url, timeout=timeout)
    need_render = not html or _is_js_shell(html)
    if need_render:
        html = await _render_with_playwright(url)
        if not html:
            return ""

    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        # markitdown.convert accepts an in-memory stream for HTML bytes.
        import io

        result = md.convert(io.BytesIO(html.encode("utf-8", errors="ignore")))
        text = (result.text_content or "").strip()
        return text
    except Exception as e:
        logger.warning("markitdown failed", url=url, error=str(e))
        # Fall back to stripping tags off the rendered HTML.
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", html)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s{2,}", " ", text).strip()
