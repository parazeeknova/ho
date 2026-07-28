"""Tests for context manager: token tracking, flush, JSON schema enforcement."""

import pytest

from src.llm.context import ContextManager, _strip_markdown


class TestStripMarkdown:
    def test_no_wrapping(self) -> None:
        assert _strip_markdown('{"key": "value"}') == '{"key": "value"}'

    def test_with_backticks(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        assert _strip_markdown(raw) == '{"key": "value"}'

    def test_backticks_no_lang(self) -> None:
        raw = '```\n{"key": "value"}\n```'
        assert _strip_markdown(raw) == '{"key": "value"}'

    def test_single_line_backticks(self) -> None:
        raw = '```{"key": "value"}```'
        assert _strip_markdown(raw) == '{"key": "value"}'

    def test_whitespace_only(self) -> None:
        assert _strip_markdown("  \n  ") == ""


class TestContextManager:
    def test_init(self) -> None:
        ctx = ContextManager()
        assert ctx.cumulative_output_tokens == 0

    @pytest.mark.asyncio
    async def test_json_chat_dict(self, mocker) -> None:
        ctx = ContextManager()
        mock_chat = mocker.patch.object(ctx, "chat", return_value='{"a": 1}')
        result = await ctx.json_chat("prompt")
        assert result == {"a": 1}
        mock_chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_json_chat_list(self, mocker) -> None:
        ctx = ContextManager()
        mocker.patch.object(ctx, "chat", return_value="[1, 2, 3]")
        result = await ctx.json_chat("prompt")
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_json_chat_invalid_json(self, mocker) -> None:
        ctx = ContextManager()
        mocker.patch.object(ctx, "chat", return_value="not json")
        result = await ctx.json_chat("{")
        assert result == {}

    @pytest.mark.asyncio
    async def test_json_chat_with_content(self, mocker) -> None:
        ctx = ContextManager()
        mock_chat = mocker.patch.object(ctx, "chat", return_value='{"x": 1}')
        result = await ctx.json_chat("prompt", content="content", limit=100)
        assert result == {"x": 1}
        sent = mock_chat.call_args[0][0]
        assert "prompt" in sent
        assert "content" in sent

    @pytest.mark.asyncio
    async def test_json_chat_with_schema(self, mocker) -> None:
        ctx = ContextManager()
        mock_chat = mocker.patch.object(ctx, "chat", return_value='{"name": "test"}')
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await ctx.json_chat("prompt", schema=schema)
        assert result == {"name": "test"}
        mock_chat.assert_awaited_once_with("prompt", schema=schema)

    @pytest.mark.asyncio
    async def test_maybe_flush_below_threshold(self, mocker) -> None:
        ctx = ContextManager()
        mock_flush = mocker.patch.object(ctx, "flush")
        await ctx.maybe_flush()
        assert mock_flush.call_count == 0

    @pytest.mark.asyncio
    async def test_flush_called_above_threshold(self, mocker) -> None:
        ctx = ContextManager()
        ctx.cumulative_output_tokens = 10000
        mock_get = mocker.AsyncMock()
        mock_get.return_value = mocker.MagicMock(json=lambda: [])
        mocker.patch.object(ctx._client, "get", mock_get)
        await ctx.maybe_flush()
        assert ctx.cumulative_output_tokens == 0

    @pytest.mark.asyncio
    async def test_flush_uses_httpx(self, mocker) -> None:
        ctx = ContextManager()
        ctx.cumulative_output_tokens = 7000
        mock_get = mocker.AsyncMock()
        mock_get.return_value = mocker.MagicMock(json=lambda: [{"id": 0, "state": 1}])
        mocker.patch.object(ctx._client, "get", mock_get)
        await ctx.flush()
        assert ctx.cumulative_output_tokens == 0
        assert mock_get.call_count >= 1

    @pytest.mark.asyncio
    async def test_flush_resets_counter_even_on_error(self, mocker) -> None:
        ctx = ContextManager()
        ctx.cumulative_output_tokens = 7000
        mock_get = mocker.AsyncMock(side_effect=OSError("down"))
        mocker.patch.object(ctx._client, "get", mock_get)
        await ctx.flush()
        assert ctx.cumulative_output_tokens == 0

    def test_flush_sync_resets_counter_on_error(self, mocker) -> None:
        ctx = ContextManager()
        ctx.cumulative_output_tokens = 7000
        mocker.patch("urllib.request.urlopen", side_effect=OSError("down"))
        ctx._flush_sync()
        assert ctx.cumulative_output_tokens == 0

    def test_flush_sync_lock_protects(self, mocker) -> None:
        ctx = ContextManager()
        ctx.cumulative_output_tokens = 7000
        mock_read = mocker.MagicMock(return_value=b"[]")
        mocker.patch("urllib.request.urlopen", return_value=mocker.MagicMock(read=mock_read))
        ctx._flush_sync()
        assert ctx.cumulative_output_tokens == 0

    def test_flush_sync_skips_when_locked(self) -> None:
        ctx = ContextManager()
        ctx.cumulative_output_tokens = 7000
        ctx._lock.acquire()
        try:
            ctx._flush_sync()
            assert ctx.cumulative_output_tokens == 7000  # unchanged — lock was held
        finally:
            ctx._lock.release()
