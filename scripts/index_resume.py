#!/usr/bin/env python3
"""Standalone resume indexer: extract, chunk, embed, and store the resume.

Wraps src/rag/loader.py so a fresh user can build resume_embeddings with a
single command instead of running a radar pass.

Usage:
    uv run python scripts/index_resume.py --url https://example.com/resume.pdf
    uv run python scripts/index_resume.py --path ./resume.pdf
    uv run python scripts/index_resume.py --dry-run   # preview only, no DB writes
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import ux  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.logging import get_logger  # noqa: E402
from src.memory.pgvector_store import MemoryStore  # noqa: E402
from src.rag.loader import index_resume_in_pgvector, load_resume  # noqa: E402

logger = get_logger("index_resume")


async def embed_server_ready() -> bool:
    from src.configuration import get_config

    base = get_config().embed.url.rsplit("/v1", 1)[0]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            resp = await client.get(f"{base}/health")
            return resp.status_code == 200
    except Exception:
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(description="Index a resume into resume_embeddings.")
    parser.add_argument("--url", help="Resume download URL (overrides RESUME_URL)")
    parser.add_argument("--path", help="Local PDF/txt resume path (overrides RESUME_PATH)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract + chunk only; do not touch the DB",
    )
    args = parser.parse_args()

    ux.banner("RESUME INDEXER", "extract  ·  chunk  ·  embed  ·  store")

    with ux.console.status("Extracting resume text...", spinner="dots"):
        try:
            if args.path:
                from src.rag.loader import chunk_resume, extract_text

                path = Path(args.path)
                if not path.exists():
                    ux.chip("err", f"{path} does not exist")
                    sys.exit(1)
                full_text = extract_text(path)
                chunks = chunk_resume(full_text)
            else:
                full_text, chunks = await asyncio.to_thread(load_resume, args.url)
        except Exception as e:
            ux.chip("err", f"Failed to load resume: {e}")
            ux.bullet("Provide --url/--path or set RESUME_URL/RESUME_PATH in .env")
            sys.exit(1)

    ux.chip("ok", f"Extracted {len(full_text)} chars across {len(chunks)} sections")
    for section, text in chunks.items():
        ux.bullet(f"{section}: {len(text)} chars", style="white")

    if args.dry_run:
        ux.chip("info", "Dry run: not writing to the database.")
        return

    if not await embed_server_ready():
        ux.chip(
            "err",
            "Embedding server not reachable. "
            "Start it with `uv run python scripts/serve.py` (needs llama-server installed).",
        )
        sys.exit(1)

    ux.divider()
    store = await MemoryStore.create()
    try:
        with ux.console.status("Embedding and indexing into resume_embeddings...", spinner="dots"):
            await index_resume_in_pgvector(chunks, store)
            count = await store.chunk_count()
    finally:
        await store.close()
    ux.chip("ok", f"Done - {count} resume chunks indexed.")
    ux.bullet("Verify with: uv run python get_resume.py", style="dim")


if __name__ == "__main__":
    asyncio.run(main())
