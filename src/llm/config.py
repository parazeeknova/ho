"""Local llama.cpp model routing and configuration.

Primary LLM:  Qwen3.5-4B (Q4_K_M)  at :8899/v1  (8192 ctx, Flash Attention)
Embeddings:   Qwen3-Embedding-0.6B (Q8_0) at :8900/v1  (32768 ctx)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

EMBED_QUERY_INSTRUCTION = (
    "Instruct: Given a job description query, retrieve relevant candidate resume "
    "skills and experience\nQuery: "
)


@dataclass
class LLMConfig:
    api_key: str = field(
        default_factory=lambda: os.getenv(
            "GENERALCOMPUTE_API_KEY", "gc_beRSKulIP-ae0ojXrZ5tqaANK2oxFO5e"
        )
    )
    model: str = field(default_factory=lambda: os.getenv("GENERALCOMPUTE_MODEL", "gemma-4-31B-it"))
    context_length: int = 32768


@dataclass
class EmbedConfig:
    base_url: str = "http://127.0.0.1:8900/v1"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    context_length: int = 32768


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)


def build_embed_query(text: str) -> str:
    """Prepend the query instruction for embedding *retrieval* queries.

    Do NOT call this when indexing resume chunks; only when generating
    a query embedding for semantic search against the store.
    """
    return EMBED_QUERY_INSTRUCTION + text
