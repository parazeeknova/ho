"""Tests for context manager: token tracking, flush, JSON parsing."""

from src.llm.context import ContextManager


class TestContextManager:
    def test_init(self) -> None:
        ctx = ContextManager()
        assert ctx.cumulative_output_tokens == 0

    def test_clean_json_no_wrapping(self) -> None:
        ctx = ContextManager()
        assert ctx.clean_json('{"key": "value"}') == '{"key": "value"}'

    def test_clean_json_with_backticks(self) -> None:
        ctx = ContextManager()
        raw = '```json\n{"key": "value"}\n```'
        assert ctx.clean_json(raw) == '{"key": "value"}'

    def test_clean_json_backticks_no_lang(self) -> None:
        ctx = ContextManager()
        raw = '```\n{"key": "value"}\n```'
        assert ctx.clean_json(raw) == '{"key": "value"}'

    def test_clean_json_single_line_backticks(self) -> None:
        ctx = ContextManager()
        raw = '```{"key": "value"}```'
        assert ctx.clean_json(raw) == '{"key": "value"}'

    def test_clean_json_whitespace_only(self) -> None:
        ctx = ContextManager()
        assert ctx.clean_json("  \n  ") == ""

    def test_json_chat_dict(self, mocker) -> None:
        ctx = ContextManager()
        mock_chat = mocker.patch.object(ctx, "chat", return_value='{"a": 1}')
        result = ctx.json_chat("prompt")
        assert result == {"a": 1}
        mock_chat.assert_called_once()

    def test_json_chat_list(self, mocker) -> None:
        ctx = ContextManager()
        mocker.patch.object(ctx, "chat", return_value="[1, 2, 3]")
        result = ctx.json_chat("prompt")
        assert result == [1, 2, 3]

    def test_json_chat_invalid_json(self, mocker) -> None:
        ctx = ContextManager()
        mocker.patch.object(ctx, "chat", return_value="not json")
        result = ctx.json_chat("{")
        assert result == {}  # fallback for dict-looking prompt

    def test_json_chat_with_content(self, mocker) -> None:
        ctx = ContextManager()
        mock_chat = mocker.patch.object(ctx, "chat", return_value='{"x": 1}')
        result = ctx.json_chat("prompt", "content", limit=100)
        assert result == {"x": 1}
        sent = mock_chat.call_args[0][0]
        assert "prompt" in sent
        assert "content" in sent

    def test_maybe_flush_below_threshold(self, mocker) -> None:
        ctx = ContextManager()
        mocker.patch.object(ctx, "chat", return_value="short")
        mock_flush = mocker.patch.object(ctx, "flush")
        ctx.json_chat("prompt")
        ctx.maybe_flush()
        assert mock_flush.call_count == 0  # below threshold

    def test_flush_called_above_threshold(self, mocker) -> None:
        ctx = ContextManager()
        ctx.cumulative_output_tokens = 10000  # above 6000 threshold
        mock_flush = mocker.patch.object(ctx, "flush")
        ctx.maybe_flush()
        assert mock_flush.call_count == 1
