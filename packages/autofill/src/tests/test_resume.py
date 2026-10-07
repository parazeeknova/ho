"""Unit tests for explicit local resume resolution (no URL fetching)."""

import os

import pytest

from autofill.src.filling.resume import resolve_resume_path


@pytest.mark.asyncio
async def test_resume_path_returns_existing_file(tmp_path):
    local = tmp_path / "local.pdf"
    local.write_bytes(b"local-pdf")
    os.environ["RESUME_PATH"] = str(local)

    try:
        result = await resolve_resume_path()
        assert result == str(local)
    finally:
        del os.environ["RESUME_PATH"]


@pytest.mark.asyncio
async def test_missing_resume_path_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_PATH", str(tmp_path / "absent.pdf"))

    assert await resolve_resume_path() is None


@pytest.mark.asyncio
async def test_no_resume_path_returns_none(monkeypatch):
    monkeypatch.delenv("RESUME_PATH", raising=False)

    assert await resolve_resume_path() is None
