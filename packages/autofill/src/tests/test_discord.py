"""Unit tests for the autofill Discord question bridge."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from autofill.src.notify.discord import (
    _SKIP_SENTINEL,
    DiscordNotConfiguredError,
    DiscordQuestionBridge,
    DiscordSendError,
    edit_distance,
)


def _bridge(bot_token: str = "test-token", channel_id: str = "123") -> DiscordQuestionBridge:
    return DiscordQuestionBridge(bot_token=bot_token, channel_id=channel_id)


@pytest.mark.asyncio
async def test_not_configured_raises() -> None:
    bridge = DiscordQuestionBridge(bot_token="", channel_id="")
    with pytest.raises(DiscordNotConfiguredError):
        await bridge.ask("Hello?")


@pytest.mark.asyncio
async def test_ask_send_failure_raises() -> None:
    bridge = _bridge()
    bridge._send_payload = AsyncMock(return_value=None)  # type: ignore[method-assign]
    with pytest.raises(DiscordSendError):
        await bridge.ask("Hello?")


@pytest.mark.asyncio
async def test_ask_mailbox_flow() -> None:
    bridge = _bridge()
    db = MagicMock()
    db.open_mailbox_question = AsyncMock()
    db.poll_mailbox_question = AsyncMock(side_effect=[("pending", None), ("answered", "Yes")])
    db.close_mailbox_question = AsyncMock()
    bridge._send_payload = AsyncMock(return_value="msg-1")  # type: ignore[method-assign]
    bridge._db = AsyncMock(return_value=db)  # type: ignore[method-assign]

    result = await bridge.ask("Are you authorized?")

    assert result == "Yes"
    db.open_mailbox_question.assert_awaited_once()
    db.close_mailbox_question.assert_awaited_once()


@pytest.mark.asyncio
async def test_ask_mailbox_timeout_returns_none() -> None:
    bridge = _bridge()
    db = MagicMock()
    db.open_mailbox_question = AsyncMock()
    db.poll_mailbox_question = AsyncMock(return_value=("pending", None))
    db.close_mailbox_question = AsyncMock()
    bridge._send_payload = AsyncMock(return_value="msg-1")  # type: ignore[method-assign]
    bridge._db = AsyncMock(return_value=db)  # type: ignore[method-assign]

    result = await bridge.ask("Question?", timeout=0.05)

    assert result is None
    db.close_mailbox_question.assert_awaited_once()


def test_interpret_plain_answer() -> None:
    bridge = _bridge()
    assert bridge._interpret_answer(" Yes ", None, False) == " Yes "
    assert bridge._interpret_answer("skip", None, False) is _SKIP_SENTINEL
    assert bridge._interpret_answer(None, None, False) is None


def test_interpret_option_button() -> None:
    bridge = _bridge()
    opts = ["Bachelor's Degree", "Master's Degree"]
    assert bridge._interpret_answer("opt:1", opts, False) == "Master's Degree"
    assert bridge._interpret_answer("opt:0", opts, True) is None
    assert bridge._interpret_answer("opt:9", opts, False) is None


def test_interpret_option_text() -> None:
    bridge = _bridge()
    opts = ["Bachelor's Degree", "Master's Degree"]
    assert bridge._interpret_answer("bachelor's", opts, False) == "Bachelor's Degree"
    assert bridge._interpret_answer("2", opts, True) == "Master's Degree"
    assert bridge._interpret_answer("#1", opts, True) == "Bachelor's Degree"
    assert bridge._interpret_answer("0", opts, True) is None
    assert bridge._interpret_answer("nonsense", opts, False) is None


def test_interpret_skip_via_button() -> None:
    bridge = _bridge()
    assert bridge._interpret_answer("skip", ["Yes", "No"], False) is _SKIP_SENTINEL


def test_edit_distance() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


@pytest.mark.asyncio
async def test_send_uses_rest() -> None:
    bridge = _bridge()
    bridge._send_payload = AsyncMock(return_value="123")  # type: ignore[method-assign]
    assert await bridge.send("hello") is True
    bridge._send_payload.assert_awaited_once()


@pytest.mark.asyncio
async def test_option_buttons_for_long_list() -> None:
    bridge = _bridge()
    buttons = bridge._option_buttons([f"opt{i}" for i in range(8)], numbered=True)
    labels = [b.get("label") for b in buttons]
    assert labels == ["Skip"]
