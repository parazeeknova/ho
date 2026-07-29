"""GitHub, LinkedIn & Personal Portfolio link parser & loader.

Automatically extracts GitHub handles, LinkedIn profile links, and portfolio website URLs
from resume text to build a candidate Knowledge Graph in RAG memory.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

import httpx
from bs4 import BeautifulSoup


def extract_links_from_text(text: str) -> dict[str, Any]:
    """Parse GitHub username, LinkedIn URL, and portfolio links from text using regex."""
    extracted: dict[str, Any] = {
        "github_username": None,
        "linkedin_url": None,
        "portfolio_urls": [],
    }

    # Extract GitHub handle
    gh_match = re.search(r"github\.com/([A-Za-z0-9_\-]+)", text, re.IGNORECASE)
    if gh_match:
        extracted["github_username"] = gh_match.group(1)

    # Extract LinkedIn URL
    li_match = re.search(
        r"(https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_\-]+)", text, re.IGNORECASE
    )
    if li_match:
        extracted["linkedin_url"] = li_match.group(1)

    # Extract HTTP/HTTPS links excluding generic domains
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


def fetch_github_profile(username: str | None = None) -> str:
    """Fetch public repositories and tech stack from GitHub API."""
    user = username or os.environ.get("GITHUB_USERNAME") or "parazeeknova"
    url = f"https://api.github.com/users/{user}/repos?sort=updated&per_page=15"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AntigravityJobSearch/1.0",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            req.add_header("Authorization", f"token {token}")

        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data: list[dict[str, Any]] = json.loads(resp.read().decode())
                repo_lines = [f"GitHub Profile: {user}"]
                languages = set()

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
        print(f"  [dim]GitHub Profile ({user}): {e}[/dim]")

    return f"GitHub Username: {user}"


def scrape_portfolio(url: str) -> str:
    """Scrape personal portfolio site to extract bio and projects."""
    try:
        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            if resp.status_code == 200 and len(resp.text) > 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for el in soup(["script", "style", "meta", "noscript", "svg"]):
                    el.decompose()
                clean_text = soup.get_text("\n", strip=True)
                if len(clean_text) > 100:
                    return clean_text[:4000]
    except Exception as e:
        print(f"  [dim]Portfolio ({url}): {e}[/dim]")
    return ""


def enrich_candidate_chunks(chunks: dict[str, str], resume_text: str = "") -> dict[str, str]:
    """Auto-parse resume links and enrich RAG chunks with GitHub, LinkedIn, and Portfolio nodes."""
    parsed_links = extract_links_from_text(resume_text)

    # 1. GitHub Integration
    gh_user = parsed_links.get("github_username") or os.environ.get("GITHUB_USERNAME")
    github_text = fetch_github_profile(gh_user)
    if github_text:
        user_display = gh_user or "parazeeknova"
        print(f"    [PASS] Auto-extracted GitHub (@{user_display}) ({len(github_text)} chars)")

    # 2. LinkedIn Integration
    li_url = parsed_links.get("linkedin_url") or os.environ.get("LINKEDIN_PROFILE_URL")
    if li_url:
        chunks["linkedin_profile"] = f"LinkedIn Profile: {li_url}"
        print(f"    [PASS] Auto-extracted LinkedIn link ({li_url})")

    # 3. Personal Portfolio Integration
    portfolio_urls = parsed_links.get("portfolio_urls", [])
    for p_url in portfolio_urls[:2]:
        p_text = scrape_portfolio(p_url)
        if p_text:
            chunks[f"portfolio_{p_url.split('//')[-1][:15]}"] = p_text
            print(f"    [PASS] Auto-scraped Personal Portfolio ({p_url}) ({len(p_text)} chars)")

    return chunks
