"""Unit tests for TelegramQuestionBridge prompting (ask / ask_options)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from autofill.telegram import (
    TelegramNotConfiguredError,
    TelegramQuestionBridge,
    TelegramSendError,
    edit_distance,
)


def _bridge(bot_token: str = "test-token", chat_id: str = "123") -> TelegramQuestionBridge:
    bridge = TelegramQuestionBridge(bot_token=bot_token, chat_id=chat_id)
    # Mark the bridge warm so _fast_forward() short-circuits (no network).
    bridge._last_update_id = 1
    return bridge


def _reply_update(msg_id: int, text: str, chat_id: str = "123", update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": 100,
            "chat": {"id": chat_id},
            "from": {"is_bot": False, "id": 7},
            "text": text,
            "reply_to_message": {"message_id": msg_id},
        },
    }


def _skip_callback_update(msg_id: int, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "cb-1",
            "data": "skip",
            "message": {"message_id": msg_id, "chat": {"id": "123"}},
        },
    }


def test_is_configured() -> None:
    assert _bridge().is_configured
    assert not TelegramQuestionBridge(bot_token="", chat_id="").is_configured
    assert not TelegramQuestionBridge(bot_token="t", chat_id="").is_configured
    assert not TelegramQuestionBridge(bot_token="", chat_id="c").is_configured


@pytest.mark.asyncio
async def test_ask_returns_matching_reply() -> None:
    bridge = _bridge()
    bridge._send_question = AsyncMock(return_value=42)
    bridge._fetch_updates = AsyncMock(side_effect=[[_reply_update(42, "80K INR")], []])

    result = await bridge.ask("What is your expected salary?", timeout=5)

    assert result == "80K INR"
    assert bridge._send_question.called
    bridge._fetch_updates.assert_called_with(timeout=3.0)


@pytest.mark.asyncio
async def test_ask_skip_callback_returns_none() -> None:
    bridge = _bridge()
    bridge._send_question = AsyncMock(return_value=42)
    bridge._fetch_updates = AsyncMock(return_value=[_skip_callback_update(42)])

    result = await bridge.ask("Do you require visa sponsorship?", timeout=5)

    assert result is None


@pytest.mark.asyncio
async def test_ask_timeout_returns_none() -> None:
    bridge = _bridge()
    bridge._send_question = AsyncMock(return_value=42)
    bridge._fetch_updates = AsyncMock(return_value=[])

    result = await bridge.ask("When can you start?", timeout=0.2)

    assert result is None


@pytest.mark.asyncio
async def test_ask_ignores_non_matching_updates() -> None:
    bridge = _bridge()
    bridge._send_question = AsyncMock(return_value=42)
    bridge._fetch_updates = AsyncMock(
        return_value=[
            _reply_update(42, "hacker answer", chat_id="999"),
            _skip_callback_update(99),
        ]
    )

    result = await bridge.ask("Current location?", timeout=0.2)

    assert result is None


def test_extract_reply_matching_rules() -> None:
    bridge = _bridge()
    bridge._option_msg_ids = {42}
    assert bridge._extract_reply_any(_reply_update(42, "yes")) == "yes"
    assert bridge._extract_reply_any(_reply_update(42, "yes", chat_id="999")) is None
    assert bridge._extract_reply_any(_reply_update(43, "yes")) is None
    assert bridge._extract_reply_any({"update_id": 1, "callback_query": {"data": "skip"}}) is None

    plain = _reply_update(42, "male")
    plain["message"].pop("reply_to_message")
    assert bridge._extract_reply_any(plain) == "male"

    plain_other_chat = _reply_update(42, "male", chat_id="999")
    plain_other_chat["message"].pop("reply_to_message")
    assert bridge._extract_reply_any(plain_other_chat) is None

    bot_msg = _reply_update(42, "bot own message")
    bot_msg["message"]["from"] = {"is_bot": True}
    assert bridge._extract_reply_any(bot_msg) is None


def test_extract_reply_accepts_continuation_chunk_reply() -> None:
    """A reply to ANY sent chunk (not just the anchor) is accepted."""
    bridge = _bridge()
    bridge._option_msg_ids = {42, 43}
    assert bridge._extract_reply_any(_reply_update(43, "India")) == "India"


@pytest.mark.asyncio
async def test_ask_accepts_plain_message_without_reply() -> None:
    bridge = _bridge()
    bridge._send_question = AsyncMock(return_value=42)
    plain = _reply_update(42, "male")
    plain["message"].pop("reply_to_message")
    bridge._fetch_updates = AsyncMock(side_effect=[[plain], []])

    result = await bridge.ask("Gender?", timeout=5)

    assert result == "male"


def test_is_skip_callback_rules() -> None:
    bridge = _bridge()
    bridge._option_msg_ids = {42}
    assert bridge._is_skip_callback(_skip_callback_update(42))
    assert not bridge._is_skip_callback(_skip_callback_update(99))
    # Skip on a hint/continuation chunk id is also honored.
    bridge._option_msg_ids = {42, 100}
    assert bridge._is_skip_callback(_skip_callback_update(100))


@pytest.mark.asyncio
async def test_ask_send_failure_raises_send_error() -> None:
    """A question that cannot be delivered must NOT look like a user decline."""
    bridge = _bridge()
    bridge._send_question = AsyncMock(return_value=None)

    with pytest.raises(TelegramSendError):
        await bridge.ask("Some question?", timeout=1)


@pytest.mark.asyncio
async def test_ask_options_send_failure_raises_send_error() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=None)

    with pytest.raises(TelegramSendError):
        await bridge.ask_options("Pick?", ["A", "B"], timeout=1)


@pytest.mark.asyncio
async def test_ask_unconfigured_raises() -> None:
    bridge = TelegramQuestionBridge(bot_token="", chat_id="")
    with pytest.raises(TelegramNotConfiguredError):
        await bridge.ask("Some question?")


@pytest.mark.asyncio
async def test_send_question_http_success() -> None:
    bridge = _bridge()

    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 42}}

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        msg_id = await bridge._send_question("What is your notice period?")

    assert msg_id == 42
    payload = client.post.call_args.kwargs["json"]
    assert payload["chat_id"] == "123"
    assert payload["reply_markup"]["force_reply"] is True


@pytest.mark.asyncio
async def test_send_question_http_failure() -> None:
    bridge = _bridge()

    class FakeResp:
        status_code = 400

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        msg_id = await bridge._send_question("What is your notice period?")

    assert msg_id is None


@pytest.mark.asyncio
async def test_send_plain_message() -> None:
    bridge = _bridge()

    class FakeResp:
        status_code = 200

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        ok = await bridge.send("Morning digest")

    assert ok is True
    payload = client.post.call_args.kwargs["json"]
    assert payload["chat_id"] == "123"
    assert payload["text"] == "Morning digest"
    assert "reply_markup" not in payload


@pytest.mark.asyncio
async def test_send_plain_message_unconfigured() -> None:
    bridge = TelegramQuestionBridge(bot_token="", chat_id="")
    assert await bridge.send("hello") is False


# ── ask_options ────────────────────────────────────────────────────


def _option_callback_update(msg_id: int, index: int, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"cb-{index}",
            "data": f"opt:{index}",
            "message": {"message_id": msg_id, "chat": {"id": "123"}},
        },
    }


@pytest.mark.asyncio
async def test_ask_options_callback_pick_returns_option() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    bridge._fetch_updates = AsyncMock(return_value=[_option_callback_update(42, 1)])

    result = await bridge.ask_options("How do you work?", ["Onsite", "Hybrid"], timeout=5)

    assert result == "Hybrid"


@pytest.mark.asyncio
async def test_ask_options_skip_returns_none() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    bridge._fetch_updates = AsyncMock(return_value=[_skip_callback_update(42)])

    result = await bridge.ask_options("Gender?", ["Male", "Female"], timeout=5)

    assert result is None


@pytest.mark.asyncio
async def test_ask_options_numbered_reply_returns_option() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    bridge._option_msg_ids = {42}
    bridge._fetch_updates = AsyncMock(
        side_effect=[
            [_reply_update(42, "2", update_id=1)],
            [_reply_update(42, "#3", update_id=2)],
            [],
        ]
    )

    long_opts = [f"Option {i}" for i in range(1, 12)]
    assert await bridge.ask_options("Pick:", long_opts, timeout=5) == "Option 2"
    assert await bridge.ask_options("Pick:", long_opts, timeout=5) == "Option 3"


@pytest.mark.asyncio
async def test_ask_options_text_reply_matches_option() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    bridge._option_msg_ids = {42}
    bridge._fetch_updates = AsyncMock(return_value=[_reply_update(42, "hybrid")])

    result = await bridge.ask_options("How do you work?", ["Onsite", "Hybrid"], timeout=5)

    assert result == "Hybrid"


@pytest.mark.asyncio
async def test_ask_options_numbered_ignores_stale_opt_callbacks() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    bridge._option_msg_ids = {42}
    # Numbered mode has no opt: buttons; a stale opt: callback from an earlier
    # message must not be picked. Only the matching reply text works.
    bridge._fetch_updates = AsyncMock(
        side_effect=[
            [_option_callback_update(42, 0), _reply_update(42, "5", update_id=2)],
            [],
        ]
    )
    long_opts = [f"Option {i}" for i in range(1, 12)]

    result = await bridge.ask_options("Pick:", long_opts, timeout=5)

    assert result == "Option 5"


@pytest.mark.asyncio
async def test_send_options_builds_inline_keyboard() -> None:
    bridge = _bridge()

    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 42}}

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        msg_id = await bridge._send_options("Where?", ["Remote", "Onsite"], numbered=False)

    assert msg_id == 42
    payload = client.post.call_args.kwargs["json"]
    assert payload["chat_id"] == "123"
    buttons = [b["text"] for row in payload["reply_markup"]["inline_keyboard"] for b in row]
    assert buttons == ["Remote", "Onsite", "Skip"]


@pytest.mark.asyncio
async def test_send_options_numbered_lists_options() -> None:
    bridge = _bridge()

    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 42}}

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        long_opts = [f"Option {i}" for i in range(1, 12)]
        await bridge._send_options("Pick:", long_opts, numbered=True)

    payload = client.post.call_args.kwargs["json"]
    text = payload["text"]
    assert "1. Option 1" in text and "11. Option 11" in text
    buttons = [b["text"] for row in payload["reply_markup"]["inline_keyboard"] for b in row]
    assert buttons == ["Skip"]  # only the skip row in numbered mode


@pytest.mark.asyncio
async def test_send_options_numbered_truncates_huge_lists() -> None:
    bridge = _bridge()

    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 42}}

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        # ~200 countries: chunked so EVERY option is visible across messages.
        long_opts = [f"Country {i}" for i in range(1, 201)]
        await bridge._send_options("Pick:", long_opts, numbered=True)

    posts = client.post.await_args_list
    all_texts = "".join(p.kwargs["json"]["text"] for p in posts)
    assert "1. Country 1" in all_texts
    assert "200. Country 200" in all_texts
    assert "… and" not in all_texts  # never truncated
    for p in posts:
        assert len(p.kwargs["json"]["text"]) < 4096


@pytest.mark.asyncio
async def test_send_options_numbered_char_budget_shrinks_list() -> None:
    bridge = _bridge()

    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 42}}

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        # Long option texts: chunked so each message stays in budget and no
        # option is dropped.
        long_opts = ["x" * 500] * 40
        await bridge._send_options("Pick:", long_opts, numbered=True)

    posts = client.post.await_args_list
    for p in posts:
        assert len(p.kwargs["json"]["text"]) < 4096
    all_texts = "".join(p.kwargs["json"]["text"] for p in posts)
    assert "… and" not in all_texts
    # All 40 options (numbered 1..40) are present.
    for i in range(1, 41):
        assert f"{i}." in all_texts
    assert len(posts) > 1


@pytest.mark.asyncio
async def test_send_options_numbered_tracks_continuation_chunk_ids() -> None:
    bridge = _bridge()

    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 42}}

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        long_opts = [f"Option {i}" for i in range(1, 12)]
        await bridge._send_options("Pick:", long_opts, numbered=True)

    # The anchor AND every continuation chunk id are tracked so a reply to any
    # of them is accepted as the answer (never dropped as a stranger message).
    assert 42 in bridge._option_msg_ids
    assert len(bridge._option_msg_ids) == client.post.call_count


@pytest.mark.asyncio
async def test_send_hint_tracks_its_message_id() -> None:
    bridge = _bridge()
    bridge._option_msg_ids = {42}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 99}}

    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))

        await bridge._send_hint(["A", "B"], numbered=False)

    # The hint is a force-reply with its own Skip button; both the reply and
    # the Skip on it must be honored, so its id joins the tracked set.
    assert 99 in bridge._option_msg_ids


@pytest.mark.asyncio
async def test_ask_options_hint_skip_is_honored() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    bridge._option_msg_ids = {42}
    # Skip callback on the hint message (id 99, not the anchor) must still
    # count as a Skip.
    hint = _skip_callback_update(99)
    bridge._fetch_updates = AsyncMock(return_value=[hint])

    result = await bridge.ask_options("Pick?", ["A", "B"], timeout=5)

    assert result is None


# ── fast-forward (stale-update backlog) ────────────────────────────


@pytest.mark.asyncio
async def test_fast_forward_jumps_past_stale_updates() -> None:
    bridge = TelegramQuestionBridge(bot_token="t", chat_id="123")

    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": [{"update_id": 9000, "message": {}}]}

    client = MagicMock()
    client.get = AsyncMock(return_value=FakeResp())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))
        await bridge._fast_forward()

    # The bridge now treats anything up to update 9000 as read, so a stale
    # plain message can never be misattributed to a fresh question.
    assert bridge._last_update_id == 9000
    params = client.get.call_args.kwargs["params"]
    assert params["offset"] == -1


@pytest.mark.asyncio
async def test_fast_forward_skips_when_already_warm() -> None:
    bridge = _bridge()  # _last_update_id already > 0
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("autofill.telegram.httpx.AsyncClient", MagicMock(return_value=client))
        await bridge._fast_forward()
    client.get.assert_not_called()


@pytest.mark.asyncio
async def test_ask_fast_forwards_then_accepts_plain_message() -> None:
    bridge = TelegramQuestionBridge(bot_token="t", chat_id="123")  # cold: update_id 0
    bridge._fast_forward = AsyncMock()
    bridge._send_question = AsyncMock(return_value=42)
    plain = _reply_update(42, "male")
    plain["message"].pop("reply_to_message")
    bridge._fetch_updates = AsyncMock(side_effect=[[plain], []])

    result = await bridge.ask("Gender?", timeout=5)

    assert result == "male"
    bridge._fast_forward.assert_awaited_once_with()


# ── fuzzy option matching for normal typed answers ────────────────


def test_fuzzy_option_unambiguous_substring() -> None:
    bridge = _bridge()
    opts = ["Bachelor's Degree", "Master's Degree", "PhD"]
    assert bridge._fuzzy_option("bachelor's", opts) == "Bachelor's Degree"
    assert bridge._fuzzy_option("masters", opts) == "Master's Degree"
    assert bridge._fuzzy_option("", opts) is None


def test_fuzzy_option_ambiguous_returns_none() -> None:
    bridge = _bridge()
    # "degree" matches both bachelor's and master's — ambiguous, so no pick.
    assert bridge._fuzzy_option("degree", ["Bachelor's Degree", "Master's Degree"]) is None


def test_fuzzy_option_edit_distance_forgives_small_typo() -> None:
    bridge = _bridge()
    opts = ["Bachelor's Degree", "Master's Degree", "PhD"]
    assert bridge._fuzzy_option("bachlors", opts) == "Bachelor's Degree"
    # A typo that lands near two options must not pick one.
    assert bridge._fuzzy_option("degre", ["Bachelor's Degree", "Master's Degree"]) is None


def test_edit_distance() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


@pytest.mark.asyncio
async def test_ask_options_plain_typed_partial_answer_matches() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    opts = ["Bachelor's Degree", "Master's Degree", "PhD"]
    plain = _reply_update(42, "bachelor's")
    plain["message"].pop("reply_to_message")
    bridge._fetch_updates = AsyncMock(side_effect=[[plain], []])

    result = await bridge.ask_options("Highest degree?", opts, timeout=5)

    assert result == "Bachelor's Degree"


@pytest.mark.asyncio
async def test_ask_options_numbered_digit_takes_precedence_over_fuzzy() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    opts = ["Option 1", "Option 2", "Option 3"]
    plain = _reply_update(42, "2")
    plain["message"].pop("reply_to_message")
    bridge._fetch_updates = AsyncMock(side_effect=[[plain], []])

    result = await bridge.ask_options("Pick:", opts, timeout=5)

    # Numbered pick by digit wins over any fuzzy text match.
    assert result == "Option 2"


@pytest.mark.asyncio
async def test_ask_options_reprompts_once_on_unmatched_plain_reply() -> None:
    bridge = _bridge()
    bridge._send_options = AsyncMock(return_value=42)
    bridge._send_hint = AsyncMock(return_value=None)
    opts = ["Bachelor's Degree", "Master's Degree", "PhD"]
    gibberish = _reply_update(42, "zzzqqq")
    gibberish["message"].pop("reply_to_message")
    good = _reply_update(42, "bachelor's")
    good["message"].pop("reply_to_message")
    bridge._fetch_updates = AsyncMock(side_effect=[[gibberish], [good], []])

    result = await bridge.ask_options("Degree?", opts, timeout=5)

    assert result == "Bachelor's Degree"
    bridge._send_hint.assert_awaited_once()


@pytest.mark.asyncio
async def test_ask_options_uses_dropdown_ask_when_no_options_read() -> None:
    bridge = _bridge()
    bridge.ask_dropdown = AsyncMock(return_value="Bachelor's Degree")

    result = await bridge.ask_options("Degree?", [], timeout=5)

    assert result == "Bachelor's Degree"
    bridge.ask_dropdown.assert_awaited_once_with("Degree?", timeout=5)


@pytest.mark.asyncio
async def test_ask_dropdown_prompts_as_dropdown() -> None:
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value="Bachelor's Degree")

    result = await bridge.ask_dropdown("Degree?", timeout=5)

    assert result == "Bachelor's Degree"
    question_sent = bridge.ask.await_args.args[0]
    assert "dropdown" in question_sent
