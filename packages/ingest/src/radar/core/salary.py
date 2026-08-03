"""Normalize raw salary strings into structured amount/currency/period.

Never cast display strings like "$55-65/hour" directly in SQL.
"""

from __future__ import annotations

import re

from src.radar.core.models import NormalizedSalary

_SALARY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"\$(\d[\d,.]*)\s*[-–]\s*\$(\d[\d,.]*)\s*/?\s*(?:per\s+)?(hour|hr|month|mo|year|yr|annually)",
            re.IGNORECASE,
        ),
        "midpoint",
        "usd",
    ),
    (
        re.compile(
            r"\$(\d[\d,.]*)\s*(?:[-–]\s*(?:\$?(\d[\d,.]*))?\s*)?/?(?:per\s+)?(hour|hr|month|mo|year|yr|annually)",
            re.IGNORECASE,
        ),
        "single",
        "usd",
    ),
    (
        re.compile(
            r"₹\s*(\d[\d,.]*)\s*(?:\s*[-–]\s*₹?\s*(\d[\d,.]*))?\s*(?:per\s+)?(hour|month|monthly|lakh|lpa|lakhs)",
            re.IGNORECASE,
        ),
        "single",
        "inr",
    ),
    (
        re.compile(
            r"(?:INR|Rs\.?)\s*(\d[\d,.]*)\s*(?:\s*[-–]\s*(\d[\d,.]*))?\s*(?:per\s+)?(hour|month|monthly|lakh|lpa|lakhs)",
            re.IGNORECASE,
        ),
        "single",
        "inr",
    ),
    (
        re.compile(
            r"€\s*(\d[\d,.]*)\s*(?:\s*[-–]\s*€?\s*(\d[\d,.]*))?\s*(?:per\s+)?(hour|month|year|yr)",
            re.IGNORECASE,
        ),
        "single",
        "eur",
    ),
    (
        re.compile(
            r"£\s*(\d[\d,.]*)\s*(?:\s*[-–]\s*£?\s*(\d[\d,.]*))?\s*(?:per\s+)?(hour|month|year|yr)",
            re.IGNORECASE,
        ),
        "single",
        "gbp",
    ),
    (
        re.compile(
            r"(\d[\d,.]*)\s*(?:\s*[-–]\s*(\d[\d,.]*))?\s*(?:k|K)\s*(?:USD|usd)?\s*(?:per\s+)?(year|yr|annually)",
            re.IGNORECASE,
        ),
        "k_single",
        "usd",
    ),
    (
        re.compile(
            r"(\d[\d,.]*)\s*(?:\s*[-–]\s*(\d[\d,.]*))?\s*lakhs?\s*(?:per\s+)?(?:annum|year)?",
            re.IGNORECASE,
        ),
        "lakh_single",
        "inr",
    ),
    (
        re.compile(r"(\d[\d,.]*)\s*(?:\s*[-–]\s*(\d[\d,.]*))?\s*LPA", re.IGNORECASE),
        "lpa_single",
        "inr",
    ),
]

_PERIOD_NORMALIZE: dict[str, str] = {
    "hour": "hour",
    "hr": "hour",
    "month": "month",
    "mo": "month",
    "monthly": "month",
    "year": "year",
    "yr": "year",
    "annually": "year",
    "lakh": "year",
    "lpa": "year",
    "lakhs": "year",
}


def normalize_salary(raw: str | None) -> NormalizedSalary | None:
    if not raw or not raw.strip():
        return None
    raw = raw.strip()

    for pat, mode, currency in _SALARY_PATTERNS:
        m = pat.search(raw)
        if not m:
            continue
        groups = m.groups()
        if mode in ("single", "midpoint"):
            lo_str = groups[0]
            hi_str = groups[1] if len(groups) > 1 else None
            period_raw = groups[-1] if len(groups) > 2 else groups[1]
            if period_raw is None:
                continue
            period = _PERIOD_NORMALIZE.get(period_raw.lower(), "year")
            amount = _parse_amount(lo_str)
            if hi_str:
                hi_amt = _parse_amount(hi_str)
                if hi_amt > amount:
                    amount = (amount + hi_amt) / 2
            if amount > 0:
                return NormalizedSalary(
                    amount=amount,
                    currency=currency.upper(),
                    period=period,
                    raw=raw,
                )
        elif mode in ("k_single", "lakh_single", "lpa_single"):
            lo_str = groups[0]
            period = "year"
            amount = _parse_amount(lo_str)
            if mode == "k_single":
                amount *= 1000
            elif mode in ("lakh_single", "lpa_single"):
                amount *= 100000
            if amount > 0:
                return NormalizedSalary(
                    amount=amount,
                    currency=currency.upper(),
                    period=period,
                    raw=raw,
                )

    return None


def _parse_amount(s: str) -> float:
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def salary_meets_minimum(
    salary: NormalizedSalary | None,
    min_monthly_inr: float = 70000,
    min_annual_usd: float = 50000,
) -> bool:
    if salary is None:
        return True
    if salary.currency == "INR":
        if salary.period == "year":
            return salary.amount / 12 >= min_monthly_inr
        if salary.period == "month":
            return salary.amount >= min_monthly_inr
        return salary.amount * 160 >= min_monthly_inr
    if salary.currency == "USD":
        if salary.period == "year":
            return salary.amount >= min_annual_usd
        if salary.period == "month":
            return salary.amount * 12 >= min_annual_usd
        return salary.amount * 2080 >= min_annual_usd
    return True
