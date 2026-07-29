"""Tests for LLM configuration defaults and embedding query builder."""

from __future__ import annotations

from src.llm.config import (
    EMBED_QUERY_INSTRUCTION,
    EmbedConfig,
    LLMConfig,
    build_embed_query,
)


class TestBuildEmbedQuery:
    def test_prepends_instruction_prefix(self) -> None:
        result = build_embed_query("test query")
        assert result.startswith(EMBED_QUERY_INSTRUCTION)
        assert result.endswith("test query")

    def test_prefix_includes_instruct_keyword(self) -> None:
        result = build_embed_query("some job")
        assert "Instruct:" in result
        assert "Query:" in result


class TestLLMConfig:
    def test_default_values(self) -> None:
        cfg = LLMConfig()
        assert cfg.base_url == "http://127.0.0.1:8899/v1"
        assert cfg.model == "unsloth/Qwen3-4B-Instruct-2507-GGUF:UD-Q5_K_XL"
        assert cfg.context_length == 16384
        assert cfg.flash_attention is True


class TestEmbedConfig:
    def test_default_values(self) -> None:
        cfg = EmbedConfig()
        assert cfg.base_url == "http://127.0.0.1:8900/v1"
        assert cfg.model == "Qwen/Qwen3-Embedding-0.6B"
        assert cfg.context_length == 32768
