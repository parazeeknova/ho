"""Tests for context manager: token tracking, flush, JSON schema enforcement."""

import pytest

from src.llm.context import ContextManager, _strip_markdown


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    """Prevent real GeneralCompute client creation in tests."""
    monkeypatch.setattr(
        "src.llm.context.GeneralCompute",
        lambda *a, **kw: None,
    )


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
        assert isinstance(ctx.model, str) and len(ctx.model) > 0

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
        mock_chat.assert_awaited_once_with(
            "prompt",
            schema=schema,
            max_tokens=None,
            interactive=False,
            system_prompt=None,
            skip_budget=False,
        )

    @pytest.mark.asyncio
    async def test_chat_sends_system_prompt(self, mocker) -> None:
        ctx = ContextManager()
        fake = mocker.MagicMock()
        fake.chat.completions.create.return_value = mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content="ok"))]
        )
        ctx._client = fake
        result = await ctx.chat("user text", system_prompt="system text")
        assert result == "ok"
        messages = fake.chat.completions.create.call_args.kwargs["messages"]
        assert messages == [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ]

    @pytest.mark.asyncio
    async def test_chat_without_system_prompt_sends_user_only(self, mocker) -> None:
        ctx = ContextManager()
        fake = mocker.MagicMock()
        fake.chat.completions.create.return_value = mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content="ok"))]
        )
        ctx._client = fake
        await ctx.chat("user text")
        messages = fake.chat.completions.create.call_args.kwargs["messages"]
        assert messages == [{"role": "user", "content": "user text"}]

    @pytest.mark.asyncio
    async def test_maybe_flush(self) -> None:
        ctx = ContextManager()
        await ctx.maybe_flush()
        assert True

    @pytest.mark.asyncio
    async def test_flush(self) -> None:
        ctx = ContextManager()
        await ctx.flush()
        assert True
