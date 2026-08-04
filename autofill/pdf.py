import os
import re
import tempfile

from fpdf import FPDF

# Helvetica is a built-in core font and cannot render non-ASCII glyphs (smart
# quotes, em-dashes, en-dashes) — the LLM output is full of them, and a single
# unencodable character raises and kills the whole PDF. DejaVu Sans covers the
# full Latin + punctuation range and is embedded, so every generated letter
# renders correctly.
FONT_PATH = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FALLBACK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _font_path() -> str:
    if os.path.exists(FONT_PATH):
        return FONT_PATH
    for cand in FALLBACK_FONT_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return ""


def create_cover_letter_pdf(
    text: str,
    first_name: str = "",
    last_name: str = "",
    job_id: str = "unknown",
) -> str:
    """Generate a PDF from the given cover letter text and return its path.

    The file is named ``<First>_<Last>_Cover.pdf``, mirroring the resume
    convention (``<First>_<Last>_Resume.pdf``). The file lives in a per-job
    subdirectory so two concurrent jobs for the same person never overwrite
    each other's (job-specific) letter on disk. Falls back to
    ``cover_letter_<job_id>.pdf`` when no name is known.

    Uses the embedded DejaVu Sans TTF so unicode text (smart quotes, dashes)
    never aborts generation — a cover letter that fails to render is worse
    than no cover letter.
    """
    pdf = FPDF()
    pdf.add_page()
    font_path = _font_path()
    if font_path:
        pdf.add_font("CoverLetterFont", "", font_path)
        pdf.set_font("CoverLetterFont", size=11)
    else:
        # No unicode font on disk: sanitize to ASCII so the core Helvetica font
        # never raises on an unencodable character.
        text = _to_ascii(text)
        pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 5, text)

    tmp_dir = os.path.join(tempfile.gettempdir(), "ho_cover_letters")
    # Per-job subdir: concurrent jobs for the same person must not overwrite
    # each other's letter (the basename alone would collide). The uploaded
    # filename stays <First>_<Last>_Cover.pdf regardless of the directory.
    tmp_dir = os.path.join(tmp_dir, _safe_segment(job_id))
    os.makedirs(tmp_dir, exist_ok=True)

    name_slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{first_name}_{last_name}").strip("_")
    if name_slug:
        filename = f"{name_slug}_Cover.pdf"
    else:
        filename = f"cover_letter_{job_id}.pdf"

    pdf_path = os.path.join(tmp_dir, filename)
    pdf.output(pdf_path)

    return pdf_path


def _safe_segment(segment: str) -> str:
    """Sanitize a path segment (job id, name) for use as a directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(segment or "unknown")).strip("._")
    return cleaned or "unknown"


def _to_ascii(text: str) -> str:
    """Best-effort ASCII transliteration for the core-font fallback."""
    return (
        (text or "")
        .replace("\u2019", "'")  # ’
        .replace("\u2018", "'")  # ‘
        .replace("\u201c", '"')  # “
        .replace("\u201d", '"')  # ”
        .replace("\u2014", "-")  # —
        .replace("\u2013", "-")  # –
        .replace("\u2026", "...")  # …
        .encode("ascii", "replace")
        .decode("ascii")
    )
