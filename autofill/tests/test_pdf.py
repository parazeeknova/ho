"""Unit tests for cover-letter PDF generation (unicode-safe, naming)."""

import pathlib

import pytest

from autofill.pdf import create_cover_letter_pdf


@pytest.fixture(autouse=True)
def _isolate_tmpdir(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """Direct cover-letter output into the test tmp dir so tests never write
    to the real /tmp/ho_cover_letters directory."""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))


def test_cover_letter_named_first_last_cover(tmp_path: pytest.TempPathFactory) -> None:
    path = create_cover_letter_pdf(
        "Dear Hiring Team,", first_name="Aman", last_name="Aziz", job_id="job-abc12345"
    )
    p = pathlib.Path(path)
    assert p.exists()
    assert p.name == "Aman_Aziz_Cover.pdf"
    assert p.read_bytes().startswith(b"%PDF")


def test_cover_letter_basename_constant_across_jobs(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Two jobs for the same person share the uploaded basename but live in
    different per-job subdirectories — one job can never overwrite the other's
    letter on disk."""
    a = create_cover_letter_pdf(
        "Letter for A.", first_name="Aman", last_name="Aziz", job_id="job-aaa"
    )
    b = create_cover_letter_pdf(
        "Letter for B.", first_name="Aman", last_name="Aziz", job_id="job-bbb"
    )
    assert pathlib.Path(a).name == pathlib.Path(b).name == "Aman_Aziz_Cover.pdf"
    assert pathlib.Path(a).parent != pathlib.Path(b).parent
    assert pathlib.Path(a).read_bytes() != pathlib.Path(b).read_bytes()


def test_cover_letter_falls_back_to_job_id_without_name(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = create_cover_letter_pdf("Dear Hiring Team,", job_id="job-abc12345")
    assert pathlib.Path(path).name == "cover_letter_job-abc12345.pdf"
    assert pathlib.Path(path).exists()


def test_cover_letter_handles_unicode_and_ascii_fallback() -> None:
    """Unicode text renders with the embedded font; generation never raises
    (a letter that fails to render is worse than no letter)."""
    path = create_cover_letter_pdf(
        "I'm excited \u2014 it's a great fit \u201cindeed\u201d\u2026",
        first_name="Aman",
        last_name="Aziz",
        job_id="job-abc12345",
    )
    assert pathlib.Path(path).exists()
