"""External salary estimation for job cards.

When a posting does not state a salary, we estimate it from public
sources. Tier 1 is levels.fyi (exact median compensation for the
company + role). Tier 2 is a SearXNG search of snippets (levels.fyi,
Glassdoor, job boards). Results are cached per (company, role) for a
week so the notify path never hammers the sources, and every estimate
carries a flag so the card can badge it as estimated.
"""

from __future__ import annotations

import hashlib
import re
import time
from statistics import median
from typing import Any

from src.configuration import get_config
from src.http_client import get_client
from src.logging import get_logger
from src.radar.core.models import NormalizedSalary

logger = get_logger("salary_lookup")

CACHE_TTL_SECONDS = 7 * 24 * 3600
_SEARXNG_RESULTS = 10
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36"
)

_FX_TO_USD = {
    "USD": 1.0,
    "INR": 0.012,
    "EUR": 1.08,
    "GBP": 1.27,
    "CAD": 0.73,
    "AUD": 0.66,
    "SGD": 0.74,
}

# Plausible annualized USD window: anything outside is noise (hourly
# rates, equity-only mentions, misparsed dates).
_MIN_ANNUAL_USD = 10_000
_MAX_ANNUAL_USD = 500_000

_CURRENCY_SYMBOLS = {"$": "USD", "₹": "INR", "£": "GBP", "€": "EUR"}

_RE_PATTERNS = [
    # $120,000 / $120K / $120k
    re.compile(r"\$\s*([\d][\d,]*)(?:\.\d+)?\s*(k|K)?\b"),
    # ₹12,00,000 / ₹12 LPA / ₹1.2L
    re.compile(r"₹\s*([\d][\d,.]*)\s*(LPA|lpa|L)?\b"),
    # 12 LPA (Indian convention, no symbol)
    re.compile(r"\b([\d]+(?:\.\d+)?)\s*(?:LPA|lpa)\b"),
    # € / £ figures
    re.compile(r"[€£]\s*([\d][\d,]*)(?:\.\d+)?\s*(k|K)?\b"),
]

_HAS_YEAR_CTX = re.compile(r"\b(?:per\s*year|per\s*annum|annual|yearly|/yr|p\.?a\.?|a\s*year)\b")
_HAS_MONTH_CTX = re.compile(r"\b(?:per\s*month|monthly|/mo|a\s*month|/month)\b")
_HAS_HOUR_CTX = re.compile(r"\b(?:per\s*hour|hourly|/hr|an\s*hour)\b")

# levels.fyi company slug cleanup: "Palantir Technologies" -> "palantir"
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:technologies?|tech|labs?|inc(?:orporated)?|ltd|llc|corp(?:oration)?|group|systems|software)\b$",
    re.IGNORECASE,
)

_ROLE_SLUGS = {
    "data scientist": "data-scientist",
    "data engineer": "data-engineer",
    "product manager": "product-manager",
    "product designer": "product-designer",
    "devops": "devops",
    "sre": "devops",
    "site reliability": "devops",
    "backend": "backend-engineer",
    "back end": "backend-engineer",
    "frontend": "frontend-engineer",
    "front end": "frontend-engineer",
    "ml engineer": "machine-learning-engineer",
    "machine learning": "machine-learning-engineer",
    "security": "security-engineer",
    "devrel": "developer-advocate",
    "developer advocate": "developer-advocate",
    "solutions engineer": "solutions-engineer",
    "engineering manager": "engineering-manager",
    "staff": "software-engineer",
    "senior": "software-engineer",
    "intern": "software-engineer",
    "software": "software-engineer",
}

_LEVELS_FYI_MEDIAN_RE = re.compile(
    r"median yearly compensation package in United States totals \$([\d,]+(?:\.\d+)?)\s*([kK]?)",
    re.IGNORECASE,
)
_LEVELS_FYI_RANGE_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)\s*([kK]?)\s*[–-]", re.IGNORECASE)


def _normalize_amount(value: str) -> float:
    """Strip commas and convert 1.2-style lakh notation."""
    if "." in value:
        return float(value.replace(",", ""))
    return float(value.replace(",", ""))


def _apply_suffix(amount: float, suffix: str) -> float:
    return amount * 1_000 if suffix in ("k", "K") else amount


def _annualize(amount_usd: float, text: str) -> float:
    if _HAS_MONTH_CTX.search(text):
        return amount_usd * 12
    return amount_usd


def _extract_annual_usd(texts: list[str]) -> list[float]:
    """Scan snippets for salary figures, annualize and convert to USD."""
    figures: list[float] = []
    for text in texts:
        if not text:
            continue
        if _HAS_HOUR_CTX.search(text):
            continue  # hourly rates are not comparable
        for pat in _RE_PATTERNS:
            for m in pat.finditer(text):
                try:
                    value = m.group(1)
                    if not value:
                        continue
                    if pat.pattern.startswith(r"\b[\d]+"):
                        # bare "12 LPA": amount is in lakhs of INR
                        annual_usd = _normalize_amount(value) * 100_000 * _FX_TO_USD["INR"]
                        if _MIN_ANNUAL_USD <= annual_usd <= _MAX_ANNUAL_USD:
                            figures.append(annual_usd)
                        continue
                    symbol = text[m.start() - 1] if m.start() > 0 else ""
                    currency = _CURRENCY_SYMBOLS.get(symbol, "USD")
                    if pat.pattern.startswith("₹"):
                        currency = "INR"
                    amount = _apply_suffix(_normalize_amount(value), m.group(2) or "")
                    annual_usd = _annualize(amount * _FX_TO_USD.get(currency, 1.0), text)
                    if _MIN_ANNUAL_USD <= annual_usd <= _MAX_ANNUAL_USD:
                        figures.append(annual_usd)
                except Exception:
                    continue
    return figures


def _best_estimate(company: str, role: str, texts: list[str]) -> NormalizedSalary | None:
    figures = _extract_annual_usd(texts)
    if not figures:
        return None
    est = median(figures)
    return NormalizedSalary(
        amount=round(est, 2),
        currency="USD",
        period="year",
        raw=f"~${est:,.0f}/yr (searched)",
    )


# Tier 1: levels.fyi direct


def _company_slug(company: str) -> str:
    slug = company.strip().lower()
    slug = _COMPANY_SUFFIX_RE.sub("", slug).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or ""


def _role_slug(role: str) -> str:
    role_l = role.lower()
    for key, slug in _ROLE_SLUGS.items():
        if key in role_l:
            return slug
    return "software-engineer"


def _parse_levels_fyi_page(html: str) -> NormalizedSalary | None:
    """Extract median (preferred) or range midpoint from a levels.fyi page."""
    m = _LEVELS_FYI_MEDIAN_RE.search(html)
    if m:
        amount = _apply_suffix(_normalize_amount(m.group(1)), m.group(2) or "")
        if _MIN_ANNUAL_USD <= amount <= _MAX_ANNUAL_USD:
            return NormalizedSalary(
                amount=round(amount, 2),
                currency="USD",
                period="year",
                raw=f"${amount:,.0f}/yr median",
            )
    lows: list[float] = []
    for m in _LEVELS_FYI_RANGE_RE.finditer(html):
        lows.append(_apply_suffix(_normalize_amount(m.group(1)), m.group(2) or ""))
    if lows:
        amount = median(lows)
        if _MIN_ANNUAL_USD <= amount <= _MAX_ANNUAL_USD:
            return NormalizedSalary(
                amount=round(amount, 2),
                currency="USD",
                period="year",
                raw=f"~${amount:,.0f}/yr",
            )
    return None


async def _levels_fyi_lookup(company: str, role: str) -> NormalizedSalary | None:
    company_slug = _company_slug(company)
    if not company_slug:
        return None
    url = f"https://www.levels.fyi/companies/{company_slug}/salaries/{_role_slug(role)}"
    try:
        client = await get_client("salary_lookup", timeout=15.0)
        resp = await client.get(url, headers={"User-Agent": _UA})
        if resp.status_code != 200:
            return None
        return _parse_levels_fyi_page(resp.text)
    except Exception as e:
        logger.warning("levels.fyi lookup failed", company=company, exception=str(e))
        return None


# Tier 2: SearXNG snippet search


async def _search_snippets(query: str) -> list[str]:
    """Run a SearXNG query and return title+content strings of results."""
    try:
        cfg = get_config().searxng
        client = await get_client("salary_lookup", timeout=cfg.timeout)
        resp = await client.get(
            cfg.url,
            params={
                "q": query,
                "format": "json",
                "language": "en",
                "safesearch": "0",
            },
            headers={"User-Agent": _UA},
        )
        if resp.status_code != 200:
            return []
        texts: list[str] = []
        for r in resp.json().get("results", [])[:_SEARXNG_RESULTS]:
            texts.append(f"{r.get('title', '')} {r.get('content', '')}")
        return texts
    except Exception as e:
        logger.warning("Salary search failed", query=query, exception=str(e))
        return []


async def _searxng_lookup(company: str, role: str) -> NormalizedSalary | None:
    queries = [
        f"{company} {role} salary",
        f'{company} "{role}" salary levels.fyi',
    ]
    texts: list[str] = []
    for q in queries:
        texts.extend(await _search_snippets(q))
        if _extract_annual_usd(texts):
            break
    return _best_estimate(company, role, texts)


# Cache


async def _load_cache(store: Any, lookup_key: str) -> tuple[NormalizedSalary | None, str]:
    try:
        async with store._pool.acquire() as conn:
            cached = await conn.fetchrow(
                "SELECT amount_usd, currency, period, raw, source, searched_at "
                "FROM salary_estimates WHERE lookup_key = $1",
                lookup_key,
            )
    except Exception:
        return None, ""
    if cached is None or not cached["amount_usd"]:
        return None, ""
    if time.time() - (cached["searched_at"] or 0) >= CACHE_TTL_SECONDS:
        return None, ""
    return (
        NormalizedSalary(
            amount=cached["amount_usd"],
            currency=cached["currency"] or "USD",
            period=cached["period"] or "year",
            raw=cached["raw"] or "",
        ),
        cached.get("source") or "",
    )


async def _store_cache(
    store: Any,
    lookup_key: str,
    company: str,
    role: str,
    est: NormalizedSalary | None,
    source: str,
) -> None:
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO salary_estimates "
                "(lookup_key, company, role, amount_usd, currency, period, raw, "
                "source, searched_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) "
                "ON CONFLICT (lookup_key) DO UPDATE SET "
                "amount_usd = EXCLUDED.amount_usd, currency = EXCLUDED.currency, "
                "period = EXCLUDED.period, raw = EXCLUDED.raw, "
                "source = EXCLUDED.source, searched_at = EXCLUDED.searched_at",
                lookup_key,
                company,
                role,
                est.amount if est else None,
                est.currency if est else "USD",
                est.period if est else "year",
                est.raw if est else "",
                source,
                time.time(),
            )
    except Exception:
        pass


async def estimate_salary(
    company: str,
    role: str,
    store: Any,
) -> tuple[NormalizedSalary | None, str]:
    """Estimate a salary for (company, role), using a 7-day DB cache.

    Returns (salary, source) where source is "levels.fyi", "searxng"
    or "". The caller badges the candidate as estimated when a salary
    comes back.
    """
    if not company or not role:
        return None, ""
    lookup_key = hashlib.sha1(f"{company}|{role.lower()}".encode()).hexdigest()[:16]

    cached, source = await _load_cache(store, lookup_key)
    if cached is not None:
        return cached, source

    est, source = await _levels_fyi_lookup(company, role), "levels.fyi"
    if est is None:
        est = await _searxng_lookup(company, role)
        source = "searxng"

    await _store_cache(store, lookup_key, company, role, est, source)
    return est, source
