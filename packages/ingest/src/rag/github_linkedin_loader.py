"""GitHub, LinkedIn & Personal Portfolio link parser & loader.

Automatically extracts GitHub handles, LinkedIn profile links, and portfolio
website URLs from resume text to build a candidate Knowledge Graph in RAG
memory. All HTTP requests are async via httpx to avoid blocking the event loop.
"""  # noqa: E501

from __future__ import annotations

import os
import re
from typing import Any

from bs4 import BeautifulSoup

from src.http_client import get_client
from src.logging import get_logger

logger = get_logger("github_linkedin_loader")


def extract_links_from_text(text: str) -> dict[str, Any]:
    """Parse GitHub username, LinkedIn URL, and portfolio links from text using regex."""
    extracted: dict[str, Any] = {
        "github_username": None,
        "linkedin_url": None,
        "portfolio_urls": [],
    }

    gh_match = re.search(r"github\.com/([A-Za-z0-9_\-]+)", text, re.IGNORECASE)
    if gh_match:
        extracted["github_username"] = gh_match.group(1)

    li_match = re.search(
        r"(https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_\-]+)",
        text,
        re.IGNORECASE,
    )
    if li_match:
        extracted["linkedin_url"] = li_match.group(1)

    all_urls = re.findall(r"https?://[^\s<>\"']+", text)
    ignored_domains = [
        "github.com",
        "linkedin.com",
        "twitter.com",
        "x.com",
        "youtube.com",
        "google.com",
        "medium.com",
    ]
    for url in all_urls:
        url_clean = url.rstrip(".,);]")
        if not any(domain in url_clean.lower() for domain in ignored_domains):
            extracted["portfolio_urls"].append(url_clean)

    return extracted


async def fetch_github_profile(username: str | None = None) -> str:
    """Fetch public repositories and tech stack from GitHub API (async)."""
    user = username or os.environ.get("GITHUB_USERNAME") or "parazeeknova"
    url = f"https://api.github.com/users/{user}/repos?sort=updated&per_page=15"

    headers: dict[str, str] = {
        "User-Agent": "AntigravityJobSearch/1.0",
        "Accept": "application/vnd.github.v3+json",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        client = await get_client("github_linkedin_loader", timeout=10.0)
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            data: list[dict[str, Any]] = resp.json()
            repo_lines = [f"GitHub Profile: {user}"]
            languages: set[str] = set()

            for r in data:
                name = r.get("name", "")
                desc = r.get("description") or ""
                lang = r.get("language")
                stars = r.get("stargazers_count", 0)
                topics = r.get("topics", [])

                if lang:
                    languages.add(lang)

                topic_str = f" [Topics: {', '.join(topics)}]" if topics else ""
                repo_lines.append(
                    f"- Repo {name} ({lang or 'Tech'} | {stars} stars): {desc}{topic_str}"
                )

            if languages:
                repo_lines.insert(1, f"Primary Tech Stack: {', '.join(languages)}")

            return "\n".join(repo_lines)
    except Exception as e:
        logger.warning(
            "GitHub profile fetch failed",
            entity=user,
            exception=str(e),
        )

    return f"GitHub Username: {user}"


async def scrape_portfolio(url: str) -> str:
    """Scrape personal portfolio site to extract bio and projects (async).

    Some portfolios are SPAs whose raw HTML is mostly nav/shell (e.g.
    przknv.cc renders "raw/blogs/Toggle theme" first); the meaningful profile
    content sits further down. We keep up to 6000 chars and skip nothing up
    front so project bullets (stacks, revenue, descriptions) survive.
    """
    try:
        client = await get_client("github_linkedin_loader", timeout=10.0)
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        if resp.status_code == 200 and len(resp.text) > 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for el in soup(["script", "style", "meta", "noscript", "svg"]):
                el.decompose()
            clean_text = soup.get_text("\n", strip=True)
            if len(clean_text) > 100:
                return clean_text[:6000]
    except Exception as e:
        logger.warning(
            "Portfolio scrape failed",
            source=url,
            exception=str(e),
        )
    return ""


async def enrich_candidate_chunks(chunks: dict[str, str], resume_text: str = "") -> dict[str, str]:
    """Auto-parse resume links and enrich RAG chunks with GitHub, LinkedIn,
    and Portfolio nodes (all async)."""
    parsed_links = extract_links_from_text(resume_text)

    gh_user = parsed_links.get("github_username") or os.environ.get("GITHUB_USERNAME")
    github_text = await fetch_github_profile(gh_user)
    if github_text:
        user_display = gh_user or "parazeeknova"
        logger.info(
            "GitHub profile extracted",
            entity=user_display,
            extra={"chars": len(github_text)},
        )

    li_url = parsed_links.get("linkedin_url") or os.environ.get("LINKEDIN_PROFILE_URL")
    if li_url:
        chunks["linkedin_profile"] = f"LinkedIn Profile: {li_url}"
        logger.info("LinkedIn link extracted", entity=li_url)

    portfolio_urls = parsed_links.get("portfolio_urls", [])
    for p_url in portfolio_urls[:2]:
        p_text = await scrape_portfolio(p_url)
        if p_text:
            key = f"portfolio_{p_url.split('//')[-1][:15]}"
            chunks[key] = p_text
            logger.info(
                "Portfolio scraped",
                source=p_url,
                extra={"chars": len(p_text)},
            )

    return chunks
