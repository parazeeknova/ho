"""Unit tests for cached resume resolution."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import autofill.resume as resume_mod
from autofill.resume import resolve_resume_path


class _Resp:
    def __init__(self, content: bytes):
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._content


def _fake_urlopen(content: bytes = b"pdf-bytes"):
    return patch.object(resume_mod.urllib.request, "urlopen", return_value=_Resp(content))


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(resume_mod, "_ARTIFACTS_DIR", tmp_path)
    monkeypatch.setenv("RESUME_URL", "https://example.com/resume.pdf")
    monkeypatch.delenv("RESUME_PATH", raising=False)
    monkeypatch.delenv("RESUME_TTL_HOURS", raising=False)
    return tmp_path


def _write_cached(artifacts: Path, url: str, age_hours: float = 0.1):
    dest = artifacts / resume_mod._RESUME_FILENAME
    dest.write_bytes(b"cached-pdf")
    (artifacts / resume_mod._RESUME_URL_SIDECAR).write_text(url)
    if age_hours is not None:
        t = time.time()
        os.utime(dest, (t - age_hours * 3600, t - age_hours * 3600))
    return dest


@pytest.mark.asyncio
async def test_downloads_when_no_cache(artifacts):
    with _fake_urlopen() as urlopen:
        result = await resolve_resume_path()

    assert result == str(artifacts / resume_mod._RESUME_FILENAME)
    urlopen.assert_called_once()
    dest = artifacts / resume_mod._RESUME_FILENAME
    assert dest.read_bytes() == b"pdf-bytes"
    assert (
        artifacts / resume_mod._RESUME_URL_SIDECAR
    ).read_text() == "https://example.com/resume.pdf"


@pytest.mark.asyncio
async def test_reuses_fresh_cache(artifacts):
    _write_cached(artifacts, "https://example.com/resume.pdf")

    with _fake_urlopen() as urlopen:
        result = await resolve_resume_path()

    assert result == str(artifacts / resume_mod._RESUME_FILENAME)
    urlopen.assert_not_called()


@pytest.mark.asyncio
async def test_refreshes_when_url_changes(artifacts, monkeypatch):
    _write_cached(artifacts, "https://example.com/old.pdf")

    with _fake_urlopen() as urlopen:
        result = await resolve_resume_path()

    assert result == str(artifacts / resume_mod._RESUME_FILENAME)
    urlopen.assert_called_once()
    assert (
        artifacts / resume_mod._RESUME_URL_SIDECAR
    ).read_text() == "https://example.com/resume.pdf"


@pytest.mark.asyncio
async def test_refreshes_when_cache_expired(artifacts):
    _write_cached(artifacts, "https://example.com/resume.pdf", age_hours=7)

    with _fake_urlopen() as urlopen:
        result = await resolve_resume_path()

    assert result == str(artifacts / resume_mod._RESUME_FILENAME)
    urlopen.assert_called_once()


@pytest.mark.asyncio
async def test_refreshes_when_sidecar_missing(artifacts):
    dest = artifacts / resume_mod._RESUME_FILENAME
    dest.write_bytes(b"cached-pdf")

    with _fake_urlopen() as urlopen:
        result = await resolve_resume_path()

    assert result == str(dest)
    urlopen.assert_called_once()


@pytest.mark.asyncio
async def test_refreshes_when_ttl_is_zero(artifacts, monkeypatch):
    monkeypatch.setenv("RESUME_TTL_HOURS", "0")
    _write_cached(artifacts, "https://example.com/resume.pdf")

    with _fake_urlopen() as urlopen:
        result = await resolve_resume_path()

    assert result == str(artifacts / resume_mod._RESUME_FILENAME)
    urlopen.assert_called_once()


@pytest.mark.asyncio
async def test_resume_path_takes_priority(tmp_path):
    local = tmp_path / "local.pdf"
    local.write_bytes(b"local-pdf")
    os.environ["RESUME_PATH"] = str(local)

    try:
        with _fake_urlopen() as urlopen:
            result = await resolve_resume_path()

        assert result == str(local)
        urlopen.assert_not_called()
    finally:
        del os.environ["RESUME_PATH"]


@pytest.mark.asyncio
async def test_download_failure_returns_none(artifacts):
    with patch.object(resume_mod.urllib.request, "urlopen", side_effect=OSError("boom")):
        result = await resolve_resume_path()

    assert result is None


@pytest.mark.asyncio
async def test_no_url_no_resume_path_returns_none(artifacts, monkeypatch):
    monkeypatch.delenv("RESUME_URL", raising=False)

    with _fake_urlopen() as urlopen:
        result = await resolve_resume_path()

    assert result is None
    urlopen.assert_not_called()
