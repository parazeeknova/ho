"""General-purpose page rendering + markdown extraction.

Replaces Firecrawl's ``/v1/scrape`` and ``/v1/map`` for the whole internet
(not just a handful of ATS boards):

- ``fetch_html``: plain httpx GET with caching/retries (fast path, no browser).
- ``render_html``: Playwright-backed render for JS-only pages, using a
  persistent browser pool — Chromium instances are launched once and reused
  across pages, then reaped after an idle window (no per-page launch cost,
  nothing resident when idle).
- ``markdownify``: turn any URL into clean markdown — tries the static path
  first (markitdown), falls back to the renderer for JS-only pages, then
  extracts main content.

No Firecrawl service, no resident browser between sweeps, no heavy infra.
Deterministic and cheap: the browser is only ever used for pages that
genuinely need it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random as _random
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

from src.http_client import get_client
from src.logging import get_logger

logger = get_logger("render")

# ── Politeness: per-host rate limiter + jitter ─────────────────────────
#
# Every request to a given host is spaced by _host_delay seconds (jittered),
# so bursts never hammer a single site into 429s. The limiter is global
# across static fetches AND JS renders so the two paths don't gang up on a
# host. Per-host delay is read from RenderConfig at first use.

_host_last: dict[str, float] = {}
_host_delay: float | None = None
_host_lock = asyncio.Lock()
_BLOCKED_STATUS = {403, 429}
# Hosts we already know block our plain datacenter IP (WWR, VC portfolios,
# Cloudflare-fronted boards) — always route those through the proxy.
_FORCED_PROXY_HOSTS = {"weworkremotely.com", "cloudflare.com", "vercel.app"}

# In-memory render cache: {url: (result, expiry_monotonic)}. A full browser
# render costs seconds; within a sweep the same board/job URL is often hit
# many times (poll + gate + re-verify), so cache both success AND negative
# (empty-shell) results for a short window. The negative cache is what stops
# an empty Ashby board from being re-rendered 2.5s every sweep.
_RENDER_CACHE_TTL = float(os.environ.get("RENDER_CACHE_TTL", "600"))
_render_cache: dict[str, tuple[str, float]] = {}
_render_cache_lock = asyncio.Lock()


async def _cache_get(url: str) -> str | None:
    async with _render_cache_lock:
        entry = _render_cache.get(url)
        if entry is None:
            return None
        result, expiry = entry
        if time.monotonic() > expiry:
            _render_cache.pop(url, None)
            return None
        return result


async def _cache_set(url: str, result: str) -> None:
    async with _render_cache_lock:
        # Bound memory: when the cache grows large, evict stale entries.
        if len(_render_cache) > 4000:
            now = time.monotonic()
            for u in list(_render_cache):
                if _render_cache[u][1] <= now:
                    _render_cache.pop(u, None)
        _render_cache[url] = (result, time.monotonic() + _RENDER_CACHE_TTL)


def _get_host(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


def _host_delay_seconds() -> float:
    global _host_delay
    if _host_delay is None:
        try:
            from src.configuration import get_config

            _host_delay = max(0.0, get_config().render.host_delay)
        except Exception:
            _host_delay = 0.5
    return _host_delay


async def _throttle(url: str) -> None:
    """Wait so requests to the same host are spaced by host_delay + jitter."""
    delay = _host_delay_seconds()
    if delay <= 0:
        return
    host = _get_host(url)
    async with _host_lock:
        now = time.monotonic()
        last = _host_last.get(host, 0.0)
        wait = (last + delay) - now
        if wait > 0:
            await asyncio.sleep(wait + _random.random() * delay * 0.5)
        _host_last[host] = time.monotonic()


def _is_blocked_status(url: str) -> bool:
    """Best-effort: does this URL live behind a known anti-bot / blocked host?"""
    host = _get_host(url)
    return any(h in host for h in _FORCED_PROXY_HOSTS)


def _proxy_url() -> str | None:
    """Return the SOCKS5 proxy URL when proxying is enabled, else None."""
    try:
        from src.configuration import get_config

        cfg = get_config().render
        return cfg.socks_proxy if cfg.use_proxy else None
    except Exception:
        return None


# Markers in a 403/429 body that indicate a real anti-bot challenge (Tor can
# help) vs a plain rate-limit (Tor does NOT help — proxying it would just burn
# time and get the same block). Only challenge/anti-bot responses get the
# proxy retry.
_CHALLENGE_MARKERS = (
    "cf-chl",
    "cf-challenge",
    "cloudflare",
    "just a moment",
    "attention required",
    "captcha",
    "verify you are human",
    "dDoS protection",
    "enable cookies and reload",
)


def _is_challenge(body: str) -> bool:
    low = (body or "").lower()
    return any(m in low for m in _CHALLENGE_MARKERS)


def _should_proxy(url: str, status: int | None, body: str = "") -> bool:
    """Route through SOCKS5 when the request hit a real anti-bot challenge
    (Cloudflare/captcha) or the host is a known anti-bot target. Plain
    rate-limit 403s/429s are NOT proxied — Tor would hit the same wall and
    only waste time."""
    if status in _BLOCKED_STATUS and _is_challenge(body):
        try:
            from src.configuration import get_config

            if get_config().render.proxy_on_block:
                return True
        except Exception:
            return True
    # Known anti-bot hosts (Cloudflare-fronted WWR, VC portfolios) always go
    # through the proxy when one is configured — that's the masking ask.
    return bool(_proxy_url() and _is_blocked_status(url))


# ── Persistent Chromium pool for JS rendering ──────────────────────────
#
# Browsers are launched once and REUSED across pages (a ~2s launch would
# otherwise be paid per page). The pool is bounded by _RENDER_CONCURRENCY
# (one browser per concurrent render slot); browsers sit idle in the pool
# between uses and are closed after _RENDERER_IDLE_MS of inactivity by a
# background reaper. Static fetches are NOT gated by this — they stay fully
# concurrent async I/O.
_RENDER_CONCURRENCY = max(1, int(os.environ.get("RENDER_CONCURRENCY", "4")))
_RENDERER_IDLE_MS = int(os.environ.get("RENDER_IDLE_MS", "30000"))
_render_sem = asyncio.Semaphore(_RENDER_CONCURRENCY)

# Total browsers launched (never exceeds the semaphore's concurrency).
_pw: Any = None  # shared async_playwright driver
_pw_lock = asyncio.Lock()
_pool: list[tuple[Any, float]] = []  # (browser, last_used_monotonic)
_pool_cond: asyncio.Condition | None = None
_reaper_task: Any = None

_PAGE_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def _browser_executable() -> str | None:
    # Prefer a system-installed Chrome/Chromium when the bundled headless
    # shell is missing system libs (libglib etc.), common in containers.
    for cand in (
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/run/current-system/sw/bin/google-chrome",
    ):
        if os.path.exists(cand):
            return cand
    return None


async def _get_playwright() -> Any:
    global _pw
    if _pw is None:
        async with _pw_lock:
            if _pw is None:
                from playwright.async_api import async_playwright  # type: ignore

                _pw = await async_playwright().start()
    return _pw


async def _launch_browser() -> Any:
    p = await _get_playwright()
    return await p.chromium.launch(
        headless=True,
        executable_path=_browser_executable(),
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )


async def _acquire_browser() -> Any:
    """Get a pooled browser (reusing an idle one, or launching a fresh one).

    The caller holds a ``_render_sem`` slot, so at most ``_RENDER_CONCURRENCY``
    browsers ever exist: every live render holds one, and released browsers sit
    in the pool for reuse. No waiting is needed — the semaphore bounds us.
    """
    global _pool_cond
    if _pool_cond is None:
        _pool_cond = asyncio.Condition()
    async with _pool_cond:
        if _pool:
            browser, _ = _pool.pop()
        else:
            browser = await _launch_browser()
        # Drop crashed browsers and relaunch.
        if not browser.is_connected():
            await _close_browser(browser)
            browser = await _launch_browser()
        return browser


async def _release_browser(browser: Any) -> None:
    global _pool_cond
    if _pool_cond is None:
        _pool_cond = asyncio.Condition()
    async with _pool_cond:
        if browser.is_connected() and len(_pool) < _RENDER_CONCURRENCY:
            _pool.append((browser, time.monotonic()))
        else:
            await _close_browser(browser)
        _pool_cond.notify()


async def _close_browser(browser: Any) -> None:
    with contextlib.suppress(Exception):
        await browser.close()


async def _reap_idle() -> None:
    """Background task: close browsers idle longer than the idle window."""
    global _pool
    while True:
        await asyncio.sleep(max(5.0, _RENDERER_IDLE_MS / 1000 / 2))
        if not _pool_cond:
            continue
        async with _pool_cond:
            now = time.monotonic()
            stale: list[Any] = []
            keep: list[tuple[Any, float]] = []
            for browser, last in _pool:
                if now - last > _RENDERER_IDLE_MS / 1000:
                    stale.append(browser)
                else:
                    keep.append((browser, last))
            _pool = keep
        for browser in stale:
            await _close_browser(browser)


def _ensure_reaper() -> None:
    global _reaper_task
    if _reaper_task is None or _reaper_task.done():
        _reaper_task = asyncio.create_task(_reap_idle())


async def _close_browser_pool() -> None:
    """Close all pooled browsers and the shared driver (on shutdown)."""
    global _pool, _pw, _reaper_task, _pool_cond
    if _reaper_task is not None:
        _reaper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await _reaper_task
        _reaper_task = None
    if _pool_cond is not None:
        async with _pool_cond:
            browsers = _pool
            _pool = []
        for browser, _ in browsers:
            await _close_browser(browser)
    if _pw is not None:
        with contextlib.suppress(Exception):
            await _pw.stop()
        _pw = None


# Pages that return this shell have NOT rendered; they need a browser.
_JS_SHELL_TEXT_MARKERS = (
    "enable javascript",
    "javascript is disabled",
    "you need to enable javascript",
    "your browser does not support javascript",
    "app loading",
    "loading...",
)


def _requires_js(html: str) -> bool:
    """Does the raw page explicitly say JavaScript is required?

    A JS-SPA served to a static fetch returns an HTML shell whose only real
    signal is a <noscript>/<meta> telling the visitor to enable JavaScript —
    even when a short <title> ("Design Engineer @ Replit") is present. Detect
    that marker in the RAW html (including noscript) so we force a browser
    render instead of extracting the stub.
    """
    low = (html or "").lower()
    return any(m in low for m in _JS_SHELL_TEXT_MARKERS)


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
    # Content-rich enough (a real job page always carries a description body
    # of hundreds+ chars) -> rendered or server-rendered.
    if len(text) >= 200:
        return False
    # Low visible text: flag as a shell if it's genuinely a JS-only app or
    # explicitly requires JS.
    if any(m in low for m in _JS_SHELL_TEXT_MARKERS):
        return True
    return len(text) < 200


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
    """Static httpx GET (cached); returns raw HTML or '' on failure.

    Applies per-host politeness spacing and falls back to the SOCKS5 proxy
    when the plain request is blocked (403/429) or the host is a known
    anti-bot target.
    """
    await _throttle(url)
    client = await get_client("render_fetch", timeout=timeout)
    resp = await _get_with_proxy(client, url, timeout)
    if resp.status_code == 200 and resp.text:
        return resp.text
    return ""


async def _get_with_proxy(client: Any, url: str, timeout: float) -> Any:
    """GET through the shared client; retry via SOCKS5 proxy on an anti-bot
    challenge. Plain rate-limit 403s/429s are NOT proxied (Tor can't help)."""
    headers = {"User-Agent": _PAGE_UA}
    try:
        resp = await client.get(url, headers=headers)
    except Exception as e:
        logger.debug("fetch_html failed", url=url, error=str(e))
        return type("R", (), {"status_code": 0, "text": ""})()
    if _should_proxy(url, resp.status_code, body=resp.text or ""):
        proxy_url = _proxy_url()
        if proxy_url:
            try:
                import httpx

                async def _proxied_get() -> Any:
                    async with httpx.AsyncClient(
                        proxy=proxy_url, timeout=max(5.0, timeout), follow_redirects=True
                    ) as pclient:
                        return await pclient.get(url, headers=headers)

                # Hard deadline: a slow/stalled Tor exit must never hang the
                # pipeline. Short cap so a proxy failure is cheap.
                presp = await asyncio.wait_for(_proxied_get(), timeout=max(5.0, timeout))
                if presp.status_code == 200 and presp.text:
                    return presp
            except Exception as e:
                logger.warning("proxy fetch failed", url=url, error=str(e))
    return resp


# ── JS rendering (lazy playwright) ──────────────────────────────────────


async def _render_with_playwright(url: str, use_proxy: bool = False) -> str:
    """Render a JS-only page with a pooled Chromium, then return HTML.

    Browsers are reused across pages (no per-page launch cost); idle ones are
    closed by a background reaper. When ``use_proxy`` is set (blocked/known
    anti-bot host), a separate SOCKS5-proxied browser is launched for this
    page so the datacenter IP is never the one that hits the target.
    Returns '' on any failure.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore  # noqa: F401

        _ = async_playwright
    except ImportError:
        logger.warning("playwright not installed; cannot render JS pages", url=url)
        return ""

    if use_proxy:
        return await _render_proxied(url)

    _ensure_reaper()
    async with _render_sem:
        browser = await _acquire_browser()
        try:
            page = await browser.new_page(user_agent=_PAGE_UA)
            await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            # Let client-side render settle.
            await page.wait_for_timeout(2500)
            html = await page.content()
            await page.close()
            if _is_js_shell(html):
                logger.warning("Rendered page still a JS shell", url=url)
                return ""
            return html
        except Exception as e:
            logger.warning("playwright render failed", url=url, error=str(e))
            return ""
        finally:
            await _release_browser(browser)


async def _render_proxied(url: str) -> str:
    """Render through a one-shot SOCKS5-proxied browser (not pooled)."""
    try:
        from playwright.async_api import async_playwright  # type: ignore  # noqa: F401
    except ImportError:
        return ""
    proxy_url = _proxy_url()
    if not proxy_url:
        return await _render_with_playwright(url, use_proxy=False)

    async def _run() -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path=_browser_executable(),
                proxy={"server": proxy_url},
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            page = await browser.new_page(user_agent=_PAGE_UA)
            await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
            await page.wait_for_timeout(3000)
            html = await page.content()
            await browser.close()
            if _is_js_shell(html):
                logger.warning("Proxied render still a JS shell", url=url)
                return ""
            return html

    try:
        # Hard deadline (40s): a stalled Tor exit must never hang the pipeline.
        return await asyncio.wait_for(_run(), timeout=40.0)
    except Exception as e:
        logger.warning("proxied playwright render failed", url=url, error=str(e))
        return ""


async def render_html(url: str, timeout: float = 20.0) -> str:
    """Fetch a page, falling back to a browser render if it's a JS shell.

    Returns the HTML with absolute hrefs, or '' if both paths fail.
    """
    html = await fetch_html(url, timeout=timeout)
    if html and not _is_js_shell(html) and not _requires_js(html):
        return _absolutize(html, url)
    # JS-only (or empty): render it. Use the proxy when the plain request was
    # blocked or the host is a known anti-bot target.
    use_proxy = _should_proxy(url, None) or bool(not html and _is_blocked_status(url))
    rendered = await _render_with_playwright(url, use_proxy=use_proxy)
    if rendered:
        return _absolutize(rendered, url)
    return html  # return whatever we have (may be a shell) so callers can retry


# ── link extraction (replaces Firecrawl /v1/map) ───────────────────────


async def extract_links(
    url: str,
    *,
    job_only: bool = True,
    limit: int = 300,
    timeout: float = 20.0,
) -> list[str]:
    """Extract job-posting links from a careers/board page.

    Replaces Firecrawl's ``/v1/map``: fetch the page (rendering JS-only sites
    when needed), pull every href, absolutize, and optionally keep only
    job-like URLs. Works across the whole internet, not just known ATS boards.
    """
    html = await render_html(url, timeout=timeout)
    if not html:
        return []
    html = _absolutize(html, url)
    hrefs: list[str] = []
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
        hrefs.append(m.group(1))
    # Also catch hrefs in JS chunks (many boards render links client-side).
    js_href_re = re.compile(
        r'["\']((?:https?:)?//[^"\']+/(?:jobs?|careers|postings?|positions?)[^"\']*)["\']'
    )
    for m in js_href_re.finditer(html):
        hrefs.append(m.group(1))
    out: list[str] = []
    seen: set[str] = set()
    for h in hrefs:
        if not h or h.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        full = h if re.match(r"^https?://", h) else urljoin(url, h)
        if job_only and not _is_job_url(full):
            continue
        if full in seen:
            continue
        seen.add(full)
        out.append(full)
        if len(out) >= limit:
            break
    return out


# ── markdown extraction ─────────────────────────────────────────────────


def _markitdown_text(html_bytes: bytes) -> str:
    """Convert raw HTML bytes to markdown in a worker thread.

    A fresh MarkItDown is created per call (it is not documented thread-safe),
    so concurrent conversions never share mutable state.
    """
    import io

    from markitdown import MarkItDown

    result = MarkItDown().convert(io.BytesIO(html_bytes))
    return (result.text_content or "").strip()


async def markdownify(url: str, timeout: float = 20.0) -> str:
    """Turn any URL into clean markdown (main content).

    Tries static fetch + markitdown first; for JS-only pages falls back to the
    browser render, then converts. Returns '' when nothing usable came back.
    Results (including empty/negative) are cached briefly so repeated hits in
    a sweep don't re-fetch or re-render.
    """
    cached = await _cache_get(url)
    if cached is not None:
        return cached

    html = await fetch_html(url, timeout=timeout)
    need_render = not html or _is_js_shell(html) or _requires_js(html)
    if need_render:
        use_proxy = _should_proxy(url, None) or bool(not html and _is_blocked_status(url))
        html = await _render_with_playwright(url, use_proxy=use_proxy)
        if not html:
            await _cache_set(url, "")
            return ""

    try:
        # markitdown.convert is CPU-bound (lxml parse + HTML→MD) and would
        # block the asyncio loop for every posting. Run it in a thread so
        # concurrent static fetches stay parallel while the conversion runs.
        html_bytes = html.encode("utf-8", errors="ignore")
        text = await asyncio.to_thread(_markitdown_text, html_bytes)
        await _cache_set(url, text)
        return text
    except Exception as e:
        logger.warning("markitdown failed", url=url, error=str(e))
        # Fall back to stripping tags off the rendered HTML.
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", html)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        await _cache_set(url, text)
        return text
