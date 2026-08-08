"""Local llama.cpp model routing and configuration (backward compat re-exports)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Re-export from centralized config
from src.configuration import LLMConfig

EMBED_QUERY_INSTRUCTION = (
    "Instruct: Given a job description query, retrieve relevant candidate resume "
    "skills and experience\nQuery: "
)


@dataclass
class EmbedConfig:
    """Backward-compat wrapper matching the old api."""

    base_url: str = field(
        default_factory=lambda: os.getenv("EMBED_URL", "http://127.0.0.1:8899/v1")
    )
    model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    )
    context_length: int = 32768


class Config:
    """Backward-compat wrapper."""

    def __init__(self) -> None:
        from src.configuration import get_config as _get

        cfg = _get()
        self.llm = LLMConfig(
            api_key=cfg.llm.api_key,
            model=cfg.llm.model,
        )
        self.embed = EmbedConfig(
            base_url=cfg.embed.url,
            model=cfg.embed.model,
        )


def build_embed_query(text: str) -> str:
    """Prepend the query instruction for embedding *retrieval* queries.

    Do NOT call this when indexing resume chunks; only when generating
    a query embedding for semantic search against the store.
    """
    return EMBED_QUERY_INSTRUCTION + text
