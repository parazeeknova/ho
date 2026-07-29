"""GitHub & LinkedIn candidate profile loader.

Enriches resume chunks with public GitHub repository metadata, top languages,
and LinkedIn profile context for deep multi-modal RAG candidate matching.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


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


def fetch_linkedin_profile(profile_url: str | None = None) -> str:
    """Load LinkedIn context from env variable or profile link."""
    raw_info = os.environ.get("LINKEDIN_MARKDOWN", "").strip()
    if raw_info:
        return raw_info

    url = profile_url or os.environ.get("LINKEDIN_PROFILE_URL") or ""
    if url:
        return f"LinkedIn Profile: {url}"

    return ""


def enrich_candidate_chunks(chunks: dict[str, str]) -> dict[str, str]:
    """Enrich resume chunks with GitHub repos and LinkedIn profile text."""
    github_text = fetch_github_profile()
    if github_text:
        chunks["github_repos"] = github_text
        print(f"    [PASS] Enriched with GitHub context ({len(github_text)} chars)")

    linkedin_text = fetch_linkedin_profile()
    if linkedin_text:
        chunks["linkedin_profile"] = linkedin_text
        print(f"    [PASS] Enriched with LinkedIn context ({len(linkedin_text)} chars)")

    return chunks
