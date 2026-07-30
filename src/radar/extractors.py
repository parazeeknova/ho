"""Local markdown table parser for GitHub internship/new-grad indexes.

Parses the raw README.md files locally, extracting company/role/location/apply_link
from Markdown tables. Never sends the index itself or a fallback README link
to the LLM matcher.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from src.logging import get_logger
from src.radar.models import JobObservation

logger = get_logger("index_extractor")

GITHUB_INDEXES = [
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/LorenzoLaCorte/european-tech-internships-2026/main/README.md",
    "https://raw.githubusercontent.com/DereC4/internships-and-newgrad/main/README.md",
    "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/README.md",
    "https://raw.githubusercontent.com/vanshb03/Summer2027-Internships/main/README.md",
]

_TABLE_ROW_RE = re.compile(
    r"^\|(.+)\|$",
    re.MULTILINE,
)

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

_HTML_LINK_RE = re.compile(r"""<a\s[^>]*href\s*=\s*["']([^"']+)["'][^>]*>""", re.IGNORECASE)

_URL_LIKE_RE = re.compile(r"https?://[^\s)|<>'\"]+")


def extract_github_index_markdown(markdown: str, source_url: str) -> list[JobObservation]:
    """Parse a GitHub README.md markdown table into individual JobObservations.

    Only extracts the apply_link column; never feeds the raw index to the matcher.
    """
    lines = markdown.split("\n")
    observations: list[JobObservation] = []
    found_header = False
    columns: list[str] = []

    for line in lines:
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            if not found_header:
                continue
            break

        cells = _split_cells(line)

        if not found_header:
            if any(h in cell.lower() for h in _HEADER_NAMES for cell in cells):
                found_header = True
                columns = cells
            continue

        if _is_separator_row(cells):
            continue

        if len(cells) < 2:
            continue

        entry = _cells_to_entry(cells, columns, source_url)
        if entry:
            observations.append(entry)

    if found_header:
        logger.info("GitHub index parsed", source=source_url, observations=len(observations))

    return observations


def _split_cells(line: str) -> list[str]:
    stripped = line.strip("|")
    return [c.strip() for c in stripped.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r"[-: ]+", c) for c in cells)


_HEADER_NAMES = {"company", "role", "location", "notes", "name", "position", "title", "description"}

_COLUMN_ALIASES: dict[str, str] = {
    "company": "company",
    "name": "company",
    "org": "company",
    "organization": "company",
    "role": "role",
    "position": "role",
    "title": "role",
    "job": "role",
    "location": "location",
    "loc": "location",
    "place": "location",
    "apply": "apply_link",
    "link": "apply_link",
    "url": "apply_link",
    "application": "apply_link",
    "application/link": "apply_link",
    "application_link": "apply_link",
    "notes": "notes",
    "description": "notes",
    "details": "notes",
}


def _cells_to_entry(cells: list[str], columns: list[str], source_url: str) -> JobObservation | None:
    data: dict[str, str] = {}
    for i, col in enumerate(columns):
        key = _COLUMN_ALIASES.get(col.lower(), "")
        if key and i < len(cells):
            data[key] = cells[i]

    apply_link = _extract_link(data.get("apply_link", ""))
    if not apply_link:
        for cell in cells:
            url = _extract_link(cell)
            if url:
                apply_link = url
                break

    if not apply_link:
        return None

    company = _strip_markdown(data.get("company", "unknown"))
    role = _strip_markdown(data.get("role", "position"))
    location = _strip_markdown(data.get("location", "Remote"))

    if _is_non_job_url(apply_link):
        return None

    return JobObservation(
        url=apply_link,
        source=f"github_index:{source_url.rsplit('/', 1)[-1]}",
        title=f"{role} at {company}",
        snippet=f"{company} - {role} ({location})",
        raw_markdown="",
    )


def _extract_link(text: str) -> str:
    m = _LINK_RE.search(text)
    if m:
        return m.group(2)
    m = _HTML_LINK_RE.search(text)
    if m:
        return m.group(1)
    m = _URL_LIKE_RE.search(text)
    if m:
        return m.group(0)
    return text if text.startswith("http") else ""


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = _HTML_LINK_RE.sub(r"\1", text)
    return _LINK_RE.sub(r"\1", text).strip()


def _is_non_job_url(url: str) -> bool:
    url_lower = url.lower()
    _skip_hosts = (
        "github.com",
        "githubusercontent.com",
        "linkedin.com/company",
        "crunchbase.com",
        "twitter.com",
        "x.com",
    )
    for host in _skip_hosts:
        if host in url_lower:
            return True
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return True
    _skip_ext = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip")
    return bool(any(path.endswith(ext) for ext in _skip_ext))


def extract_apply_links_from_indexes(
    index_markdowns: list[tuple[str, str]],
) -> list[JobObservation]:
    """Process multiple GitHub index files and return all extracted observations."""
    all_obs: list[JobObservation] = []
    for url, md in index_markdowns:
        obs = extract_github_index_markdown(md, url)
        all_obs.extend(obs)
    return all_obs
