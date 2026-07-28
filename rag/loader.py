"""Resume loader: download from URL, extract with markitdown/pymupdf, verify."""

import tempfile
import urllib.request
from pathlib import Path


def _get_pymupdf():
    import pymupdf

    return pymupdf


def download_resume(url: str) -> Path:
    print(f"  Downloading: {url}")
    suffix = Path(url).suffix or ".pdf"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with open(fd, "wb") as f, urllib.request.urlopen(url, timeout=30) as resp:
        f.write(resp.read())
    return Path(path)


def extract_with_pymupdf(path: Path) -> str:
    pymupdf = _get_pymupdf()
    text_parts = []
    doc = pymupdf.open(path)
    for page in doc:
        t = page.get_text()
        if t:
            text_parts.append(t)
    doc.close()
    return "\n\n".join(text_parts)


def extract_with_markitdown(path: Path) -> str:
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(path))
    return result.text_content


def verify_extraction(text: str) -> dict[str, bool | str]:
    checks = {
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

    # Try markitdown first (handles DOCX, HTML, PDF via pymupdf)
    try:
        print("  Extracting with markitdown...")
        text = extract_with_markitdown(path)
    except Exception:
        print("  Fallback to pymupdf...")
        text = extract_with_pymupdf(path)

    # Verify
    checks = verify_extraction(text)
    for key, val in checks.items():
        if key not in ("passes", "length"):
            status = "PASS" if val else "FAIL"
            print(f"    [{status}] {key}")
    print(f"    [INFO] extracted {checks['length']} chars")

    if not checks["passes"]:
        print("  WARNING: Some extraction checks failed, continuing anyway...")

    # Chunk
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
