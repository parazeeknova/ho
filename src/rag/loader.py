"""Resume loader: download from URL, extract with markitdown/pymupdf, verify."""

import tempfile
import urllib.request
from pathlib import Path


def download_resume(url: str) -> Path:
    print(f"  Downloading: {url}")
    suffix = Path(url).suffix or ".pdf"
    fd, path = tempfile.mkstemp(suffix=suffix)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    with open(fd, "wb") as f, urllib.request.urlopen(req, timeout=30) as resp:
        f.write(resp.read())
    return Path(path)


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

    # 3. pypdf (pure Python, always works)
    from pypdf import PdfReader

    parts = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    text = "\n\n".join(parts)
    print(f"    using pypdf ({len(text)} chars)")
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

    path = download_resume(url)
    print(f"  Saved to: {path}")
    print(f"  Size: {path.stat().st_size} bytes")

    print("  Extracting...")
    text = extract_text(path)

    # Show extracted text preview
    preview = text[:800].replace("\n", "\n    ")
    print(f"\n  ── Extracted Text Preview (first 800/{len(text)} chars) ──")
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
        "skills": ["skill"],
        "experience": ["experience"],
        "education": ["education"],
        "projects": ["project"],
        "achievements": ["achiev", "award", "certif"],
    }

    for line in lines:
        stripped = line.strip().lower()
        matched = False

        if len(stripped) < 30:
            for section_name, keywords in section_headings.items():
                if any(kw in stripped for kw in keywords):
                    current_section = section_name
                    matched = True
                    break

        if not matched:
            sections.setdefault(current_section, []).append(line.strip())

    return {k: "\n".join(v) for k, v in sections.items() if v}
