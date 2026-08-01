"""Resume file resolution for the autofill runner.

Downloads the candidate resume from RESUME_URL (or returns an existing local
RESUME_PATH) so the node adapter can upload it via setInputFiles().

Downloads are cached in node/artifacts/resume.pdf for RESUME_TTL_HOURS (default
6h); the artifact is refreshed when the source URL changes or the cache
expires.
"""

import os
import time
import urllib.request
from pathlib import Path

from src.logging import get_logger

logger = get_logger("autofill.resume")

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "node" / "artifacts"
_RESUME_FILENAME = "resume.pdf"
_RESUME_URL_SIDECAR = f"{_RESUME_FILENAME}.url"

_DEFAULT_TTL_HOURS = 6


def _ttl_hours() -> float:
    try:
        return max(0.0, float(os.environ.get("RESUME_TTL_HOURS", _DEFAULT_TTL_HOURS)))
    except (TypeError, ValueError):
        return float(_DEFAULT_TTL_HOURS)


def _cache_valid(dest: Path, url: str) -> bool:
    if not dest.exists():
        return False
    sidecar = dest.parent / _RESUME_URL_SIDECAR
    try:
        recorded = sidecar.read_text().strip()
    except OSError:
        return False
    if recorded != url:
        return False
    ttl = _ttl_hours()
    if ttl <= 0:
        return False
    age = time.time() - dest.stat().st_mtime
    return age < ttl * 3600


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, dest)
    (dest.parent / _RESUME_URL_SIDECAR).write_text(url)
    logger.info("Downloaded resume for upload", url=url, size=len(content), path=str(dest))


async def resolve_resume_path() -> str | None:
    """Return a local path to a resume file, downloading if necessary."""
    local = os.environ.get("RESUME_PATH")
    if local and os.path.exists(local):
        logger.info("Using local resume at RESUME_PATH", path=local)
        return local

    url = os.environ.get("RESUME_URL")
    if not url:
        logger.info("No RESUME_URL/RESUME_PATH set; skipping resume upload")
        return None

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _ARTIFACTS_DIR / _RESUME_FILENAME

    if _cache_valid(dest, url):
        logger.info("Reusing cached resume", path=str(dest))
        return str(dest)

    try:
        _download(url, dest)
        return str(dest)
    except Exception as e:
        logger.exception("Failed to download resume; skipping upload", error=str(e))
        return None
