"""Resume loader: download, extract, verify, interactive review, embed-index."""

import hashlib
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import httpx

from src.configuration import get_config
from src.http_client import get_client


def _chunk_hash(content: str) -> str:
    """Stable content fingerprint for resume chunks (dedupes re-embedding)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def download_resume(url: str) -> tuple[Path, str]:
    print(f"  Downloading: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    content = resp.read()
    content_type = resp.headers.get("Content-Type", "")

    # If it's HTML pretending to be a PDF, save as HTML
    if content.startswith(b"<!") or b"<!doctype" in content[:100].lower():
        print("  Detected HTML response, saving as .html")
        suffix = ".html"
    else:
        suffix = Path(url).suffix or ".pdf"

    fd, path = tempfile.mkstemp(suffix=suffix)
    with open(fd, "wb") as f:
        f.write(content)
    return Path(path), content_type


_TABLE_SEP_RX = re.compile(r"^\|?[\s:\-|]+\|?\s*$")


def cleanup_markdown_tables(text: str) -> str:
    """Flatten markdown-table artifacts from PDF extraction into plain text.

    Resume PDFs are frequently table-heavy; markitdown converts every
    tabular layout into a markdown table, which then leaks pipe rows and
    ``--- | ---`` separators into resume chunks, the resume summary and the
    identity extraction. Table separator rows are dropped and true table
    rows (leading ``|``) are joined cell-by-cell.
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _TABLE_SEP_RX.match(stripped):
            continue
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            cells = [c for c in cells if c]
            if cells:
                out.append(" ".join(cells))
            continue
        out.append(stripped)
    return "\n".join(out)


def extract_text(path: Path) -> str:
    """Try markitdown first, then pymupdf, then pypdf as last resort."""

    # 1. markitdown (handles PDF, DOCX, HTML, images)
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(str(path))
        text = result.text_content
        if text and len(text) > 100:
            return cleanup_markdown_tables(text)
        print(f"    markitdown returned {len(text)} chars, trying next...")
    except Exception as e:
        print(f"    markitdown: {e}")

    # 2. pymupdf (best PDF extraction, needs libstdc++)
    try:
        import pymupdf

        parts = []
        doc = pymupdf.open(path)
        for page_num in range(len(doc)):
            t = doc[page_num].get_text()
            if t:
                parts.append(t)
        doc.close()
        text = "\n\n".join(parts)
        if text and len(text) > 100:
            return cleanup_markdown_tables(text)
        print(f"    pymupdf returned {len(text)} chars, trying next...")
    except Exception as e:
        print(f"    pymupdf: {e}")

    # 3. pypdf (pure Python, always works for PDFs)
    try:
        from pypdf import PdfReader

        parts = []
        reader = PdfReader(str(path))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        text = "\n\n".join(parts)
        if text and len(text) > 50:
            return cleanup_markdown_tables(text)
        print(f"    pypdf returned {len(text)} chars, trying next...")
    except Exception as e:
        print(f"    pypdf: {e}")

    # 4. raw text (last resort, for HTML or unknown formats)
    from bs4 import BeautifulSoup

    raw = path.read_text(errors="ignore")
    if raw.strip().startswith("<"):
        soup = BeautifulSoup(raw, "html.parser")
        for el in soup(["script", "style", "meta", "noscript", "svg"]):
            el.decompose()
        text = soup.get_text("\n", strip=True)
    else:
        text = raw
    print(f"    using bs4/html ({len(text)} chars)")
    return text


def verify_extraction(text: str) -> dict[str, bool | str]:
    checks: dict[str, bool | str] = {
        "has_email": "@" in text and "." in text.split("@")[-1],
        "has_phone": any(c.isdigit() for c in text) and len(text) > 200,
        "has_skills": any(
            kw in text.lower() for kw in ["python", "java", "react", "sql", "aws", "docker", "git"]
        ),
        "has_experience": any(kw in text.lower() for kw in ["experience", "intern", "work"]),
        "has_education": any(
            kw in text.lower()
            for kw in [
                "university",
                "college",
                "bachelor",
                "degree",
                "school",
                "b.tech",
                "b.e.",
                "b.s.",
                "m.s.",
                "m.tech",
                "institute",
                "iit",
                "nit",
                "academic",
                "education",
                "graduat",
            ]
        ),
        "length_ok": len(text) > 300,
    }
    checks["passes"] = all(checks.values())
    checks["length"] = str(len(text))
    return checks


def load_resume(default_url: str | None = None) -> tuple[str, dict[str, str]]:
    import os
    import sys

    import questionary

    url = default_url or os.environ.get("RESUME_URL")
    is_non_interactive = (
        bool(url)
        or os.environ.get("NON_INTERACTIVE", "false").lower() == "true"
        or not sys.stdin.isatty()
    )

    if not url:
        if is_non_interactive:
            url = "https://f.przknv.cc/raw/ayEBJQ.pdf"
        else:
            url = questionary.text("Resume URL (PDF/DOCX/HTML):").ask()

    if not url:
        raise ValueError("No URL provided")

    path, content_type = download_resume(url)

    text = extract_text(path)

    if not is_non_interactive:
        while True:
            action = questionary.select(
                f"Resume loaded ({len(text)} chars). What next?",
                choices=[
                    "Continue with pipeline",
                    "View full resume (scrollable)",
                    "Re-enter URL",
                ],
            ).ask()

            if action == "Continue with pipeline":
                break
            elif action == "View full resume (scrollable)":
                subprocess.run(["less", "-R"], input=text.encode(), check=False)
            elif action == "Re-enter URL":
                path.unlink(missing_ok=True)
                return load_resume()

    chunks = chunk_resume(text)

    path.unlink(missing_ok=True)
    return text, chunks


def chunk_resume(text: str) -> dict[str, str]:
    lines = text.split("\n")
    sections: dict[str, list[str]] = {}
    current_section = "header"

    section_headings = {
        "skills": ["skill", "tech stack", "technologies", "languages", "tools"],
        "experience": ["experience", "work", "employment", "internship"],
        "education": ["education", "university", "college", "academic"],
        "projects": ["project", "portfolio"],
        "achievements": ["achiev", "award", "certif", "publication"],
    }

    for line in lines:
        stripped = line.strip()
        stripped_lower = stripped.lower()
        matched = False

        if len(stripped) < 30:
            for section_name, keywords in section_headings.items():
                if any(kw in stripped_lower for kw in keywords):
                    current_section = section_name
                    matched = True
                    break

        if not matched and stripped:
            sections.setdefault(current_section, []).append(stripped)

    return {k: "\n".join(v) for k, v in sections.items() if v}


async def index_resume_in_pgvector(
    chunks: dict[str, str],
    store,
) -> None:
    cfg = get_config().embed
    embed_client = await get_client(
        "rag_embedder",
        timeout=httpx.Timeout(120.0, connect=10.0),
        extra_limits={"max_keepalive_connections": 2, "max_connections": 4},
    )
    try:
        records: list[dict[str, object]] = []
        current_hashes: set[str] = set()
        batches: list[tuple[str, list[str]]] = []
        for section, text in chunks.items():
            raw_lines = [ln.strip() for ln in text.split("\n")]
            lines = [ln for ln in raw_lines if ln and len(ln) > 10]
            for i in range(0, len(lines), 8):
                batches.append((section, lines[i : i + 8]))
            if len(text) > 20:
                batches.append((section, [text[:500]]))
        for _, batch in batches:
            current_hashes.update(_chunk_hash(c) for c in batch)

        # Skip chunks whose content hash already exists so unchanged resume
        # sections are never re-embedded or re-uploaded.
        if hasattr(store, "existing_resume_hashes"):
            existing = await store.existing_resume_hashes(list(current_hashes))
        else:
            existing = set()
        for section, batch in batches:
            missing = [c for c in batch if _chunk_hash(c) not in existing]
            if not missing:
                continue
            resp = await embed_client.post(
                f"{cfg.url}/embeddings",
                json={"model": cfg.model, "input": missing},
            )
            resp.raise_for_status()
            data = resp.json()
            for item, content in zip(data["data"], missing, strict=True):
                records.append(
                    {
                        "section": section,
                        "content": content,
                        "content_hash": _chunk_hash(content),
                        "embedding": item["embedding"],
                    }
                )
        if records:
            await store.index_resume_chunks(records, current_hashes=current_hashes)
    finally:
        await embed_client.aclose()
