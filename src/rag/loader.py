"""Resume loader: download from URL, extract with markitdown/pymupdf, verify."""

import tempfile
import urllib.request
from pathlib import Path


def download_resume(url: str) -> tuple[Path, str]:
    print(f"  Downloading: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    content = resp.read()
    content_type = resp.headers.get("Content-Type", "")

    print(f"  Content-Type: {content_type}")
    print(f"  Size: {len(content)} bytes")

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


def extract_text(path: Path) -> str:
    """Try markitdown first, then pymupdf, then pypdf as last resort."""

    # 1. markitdown (handles PDF, DOCX, HTML, images)
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(str(path))
        text = result.text_content
        if text and len(text) > 100:
            print("    using markitdown")
            return text
        print(f"    markitdown returned {len(text)} chars, trying next...")
    except Exception as e:
        print(f"    markitdown: {e}")

    # 2. pymupdf (best PDF extraction, needs libstdc++)
    try:
        import pymupdf

        parts = []
        doc = pymupdf.open(path)
        for page in doc:
            t = page.get_text()
            if t:
                parts.append(t)
        doc.close()
        text = "\n\n".join(parts)
        if text and len(text) > 100:
            print("    using pymupdf")
            return text
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
            print(f"    using pypdf ({len(text)} chars)")
            return text
        print(f"    pypdf returned {len(text)} chars, trying next...")
    except Exception as e:
        print(f"    pypdf: {e}")

    # 4. raw text (last resort, for HTML or unknown formats)
    from bs4 import BeautifulSoup

    raw = path.read_text(errors="ignore")
    if raw.strip().startswith("<"):
        soup = BeautifulSoup(raw, "html.parser")
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
            kw in text.lower() for kw in ["university", "college", "bachelor", "degree", "school"]
        ),
        "length_ok": len(text) > 300,
    }
    checks["passes"] = all(checks.values())
    checks["length"] = str(len(text))
    return checks


def load_resume() -> tuple[str, dict[str, str]]:
    url = input("Resume URL (PDF/DOCX/HTML): ").strip()
    if not url:
        raise ValueError("No URL provided")

    path, content_type = download_resume(url)

    print("  Extracting...")
    text = extract_text(path)

    # Show extracted text preview
    preview = text[:2000].replace("\n", "\n    ")
    print(f"\n  ── Extracted Text Preview (first 2000/{len(text)} chars) ──")
    print(f"    {preview}")
    print("  ─────────────────────────────────────────────\n")

    checks = verify_extraction(text)
    for key, val in checks.items():
        if key not in ("passes", "length"):
            status = "PASS" if val else "FAIL"
            print(f"    [{status}] {key}")
    print(f"    [INFO] extracted {checks['length']} chars")

    if not checks["passes"]:
        print("  WARNING: Some extraction checks failed, continuing anyway...")

    chunks = chunk_resume(text)
    print(f"  Sections: {list(chunks.keys())}")

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
