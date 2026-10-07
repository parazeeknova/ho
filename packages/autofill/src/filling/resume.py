"""Resume file resolution for the autofill runner.

Returns an explicit local RESUME_PATH (per-user onboarding supplies the file
long-term) so the node adapter can upload it via setInputFiles().
"""

import os

from src.logging import get_logger

logger = get_logger("autofill.src.filling.resume")


async def resolve_resume_path() -> str | None:
    """Return an explicit local RESUME_PATH, or None to skip the upload."""
    local = os.environ.get("RESUME_PATH")
    if local and os.path.exists(local):
        logger.info("Using local resume at RESUME_PATH", path=local)
        return local
    logger.info("No RESUME_PATH set; skipping resume upload")
    return None
