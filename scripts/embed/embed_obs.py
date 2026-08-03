"""Batch-embed the job corpus into ``obs_embeddings`` via the local embed server.

Usage:
    uv run --with azure-storage-blob python3 scripts/embed/embed_obs.py [--all] [--limit N]

Checkpointed and resumable: rows that already have an embedding are skipped,
so re-running just continues from where it stopped. Software-role titles are
embedded first (the matcher budget lives there); pass ``--all`` to embed the
whole corpus rather than the software-first default.

The embed server (llama-server on :8900, Qwen3-Embedding-0.6B) is local and
free, so this costs no LLM tokens. It shares the server with the live
pipeline, so it throttles itself politely.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from src.agent.enrichment_agent import _get_embedding
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore

logger = get_logger("embed_obs")

BATCH = 32
THROTTLE = 0.05  # polite inter-batch sleep (s)


async def _embed_batch(texts: list[str]) -> list[list[float] | None]:
    return [await _get_embedding(t) for t in texts]


async def run(limit: int | None, all_corpus: bool) -> None:
    store = await MemoryStore.create()
    total = 0
    t0 = time.monotonic()

    while True:
        want = min(limit - total, 4000) if limit else 4000
        if want <= 0:
            break
        obs = await store.unembedded_obs(limit=want, software_first=not all_corpus)
        if not obs:
            break
        ok = 0
        for i in range(0, len(obs), BATCH):
            chunk = obs[i : i + BATCH]
            embs = await _embed_batch([o["text"] for o in chunk])
            for o, e in zip(chunk, embs):
                if e:
                    await store.upsert_obs_embedding(o["url_hash"], o["title"], o["company"], e)
                    ok += 1
            await asyncio.sleep(THROTTLE)
        total += ok
        rate = ok / max(time.monotonic() - t0, 0.001)
        logger.info(f"embedded {ok} obs this pass (running total {total}, ~{rate:.0f}/s)")
        if ok < len(obs):
            # Embed server rate-limited or text filtered out; don't spin.
            await asyncio.sleep(2)
    logger.info(f"embed backfill complete: {total} embeddings written")


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed job corpus into obs_embeddings")
    ap.add_argument(
        "--all", action="store_true", help="embed whole corpus (not just software-first)"
    )
    ap.add_argument("--limit", type=int, default=None, help="max observations to embed this run")
    args = ap.parse_args()
    asyncio.run(run(args.limit, args.all))


if __name__ == "__main__":
    main()
