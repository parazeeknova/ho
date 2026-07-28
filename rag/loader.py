"""Load resume from PDF, chunk into sections, extract structured info."""

from pathlib import Path

import pdfplumber

RESUME_DIR = Path("resume")


def extract_text(pdf_path: Path) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def chunk_resume(text: str) -> dict[str, str]:
    """Split resume into labeled sections using LLM-free heuristics."""
    lines = text.split("\n")
    sections: dict[str, list[str]] = {}
    current_section = "header"

    section_keywords = {
        "skills": [
            "skill",
            "technologies",
            "tech stack",
            "languages",
            "tools",
            "frameworks",
            "proficien",
        ],
        "experience": ["experience", "work", "employment", "internship", "job"],
        "education": [
            "education",
            "university",
            "college",
            "bachelor",
            "master",
            "degree",
            "school",
            "gpa",
            "cgpa",
        ],
        "projects": ["project", "portfolio", "hackathon"],
        "achievements": ["achiev", "award", "honor", "certif", "publication"],
    }

    for line in lines:
        stripped = line.strip().lower()
        matched = False
        for section_name, keywords in section_keywords.items():
            if any(kw in stripped for kw in keywords) and len(stripped) < 60:
                current_section = section_name
                matched = True
                break
        if not matched:
            sections.setdefault(current_section, []).append(line.strip())

    return {k: "\n".join(v) for k, v in sections.items() if v}


def load_resume() -> tuple[str, dict[str, str]]:
    pdfs = list(RESUME_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {RESUME_DIR}")
    pdf_path = pdfs[0]
    print(f"Loading resume: {pdf_path.name}")
    full_text = extract_text(pdf_path)
    chunks = chunk_resume(full_text)
    print(f"  Sections: {list(chunks.keys())}")
    return full_text, chunks
