"""Telegram prompting bridge for unknown job-application screener questions.

When the RAG pipeline cannot answer a question from the persona knowledge base
(``__ASK_USER__``), the bridge sends the question to the user on Telegram and
waits for their reply. Answers are then fed back into the form and persisted
into the persona knowledge base via ``ScreenerRAG.learn``.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from src.logging import get_logger

logger = get_logger("autofill.telegram")

TELEGRAM_BASE = "https://api.telegram.org/bot{token}"
TELEGRAM_SEND = f"{TELEGRAM_BASE}/sendMessage"
TELEGRAM_UPDATES = f"{TELEGRAM_BASE}/getUpdates"

DEFAULT_QUESTION_TIMEOUT = 300.0
_HTTP_TIMEOUT = 8.0
_MAX_POLL_TIMEOUT = 3.0
# Long option lists (country pickers etc.) are truncated in the message so it
# stays under Telegram's 4096-char limit; replies are still matched against
# the full list.
_NUMBERED_DISPLAY_LIMIT = 25
# Telegram's hard message-length cap is 4096 chars; keep headroom for the
# question text itself.
_MAX_MESSAGE_CHARS = 4000


class TelegramNotConfiguredError(RuntimeError):
    """Raised when a user prompt is required but Telegram is not configured."""


# Internal sentinel distinguishing "user hit Skip" from "no pick yet".
_SKIP_SENTINEL = object()


class TelegramQuestionBridge:
    """Send screener questions to the user on Telegram and collect replies.

    Only polls ``getUpdates`` while a question is pending, so interference with
    the scraper pipeline's command bot (same token) is limited to that window.
    """

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None) -> None:
        self.bot_token = bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")
        self._last_update_id = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    @property
    def _chat_ids(self) -> list[str]:
        raw = (self.chat_id or "").strip()
        if not raw:
            return []
        return [cid.strip() for cid in raw.split(",") if cid.strip()]

    @property
    def _primary_chat_id(self) -> str:
        ids = self._chat_ids
        return ids[0] if ids else ""

    async def ask(self, question: str, timeout: float = DEFAULT_QUESTION_TIMEOUT) -> str | None:
        """Send a question to Telegram and wait for the user's answer.

        Returns the reply text, or ``None`` when the user hits Skip or the
        timeout elapses.
        """
        if not self.is_configured:
            raise TelegramNotConfiguredError(
                "Telegram prompting unavailable: set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID to answer personal screener questions."
            )

        msg_id = await self._send_question(question)
        if msg_id is None:
            return None

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            updates = await self._fetch_updates(timeout=min(_MAX_POLL_TIMEOUT, remaining + 1))
            for upd in updates:
                self._last_update_id = max(self._last_update_id, upd.get("update_id", 0))
                if self._is_skip_callback(upd, msg_id):
                    return None
                reply = self._extract_reply(upd, msg_id)
                if reply is not None:
                    return reply
            await asyncio.sleep(min(1.0, max(0.05, remaining)))

    async def send(self, text: str) -> bool:
        """Send a plain message (deferral alerts, morning digest). No reply flow."""
        if not self.is_configured or not text:
            return False
        payload: dict[str, Any] = {
            "chat_id": self._primary_chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(TELEGRAM_SEND.format(token=self.bot_token), json=payload)
                return resp.status_code == 200
        except Exception as e:
            logger.warning("Telegram send message error", exception=str(e))
            return False

    async def _send_question(self, question: str) -> int | None:
        payload: dict[str, Any] = {
            "chat_id": self._primary_chat_id,
            "text": question,
            "disable_web_page_preview": True,
            "reply_markup": {
                "force_reply": True,
                "input_field_placeholder": "Reply with your answer...",
                "inline_keyboard": [[{"text": "Skip", "callback_data": "skip"}]],
            },
        }
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(TELEGRAM_SEND.format(token=self.bot_token), json=payload)
                if resp.status_code != 200:
                    logger.warning("Telegram send question failed", status=resp.status_code)
                    return None
                data = resp.json()
                if not data.get("ok"):
                    return None
                result = data.get("result") or {}
                return result.get("message_id")
        except Exception as e:
            logger.warning("Telegram send question error", exception=str(e))
            return None

    async def _fetch_updates(self, timeout: float = _MAX_POLL_TIMEOUT) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": timeout,
            "limit": 10,
            "allowed_updates": ["message", "callback_query"],
        }
        if self._last_update_id > 0:
            params["offset"] = self._last_update_id + 1
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    TELEGRAM_UPDATES.format(token=self.bot_token), params=params
                )
                if resp.status_code != 200:
                    return []
                data = resp.json()
                if not data.get("ok"):
                    return []
                return data.get("result", [])
        except Exception as e:
            logger.debug("Telegram getUpdates failed", source="autofill.telegram", exception=str(e))
            return []

    def _is_skip_callback(self, upd: dict[str, Any], msg_id: int) -> bool:
        cb = upd.get("callback_query") or {}
        cb_msg = cb.get("message") or {}
        return cb.get("data") == "skip" and cb_msg.get("message_id") == msg_id

    def _extract_reply(self, upd: dict[str, Any], msg_id: int) -> str | None:
        msg = upd.get("message") or {}
        if not msg:
            return None
        user = msg.get("from") or {}
        if user.get("is_bot"):
            return None
        chat = msg.get("chat") or {}
        if str(chat.get("id", "")) not in self._chat_ids:
            return None
        reply_to = (msg.get("reply_to_message") or {}).get("message_id")
        if reply_to is not None and reply_to != msg_id:
            return None
        text = (msg.get("text") or "").strip()
        return text or None

    async def ask_options(
        self,
        question: str,
        options: list[str],
        timeout: float = DEFAULT_QUESTION_TIMEOUT,
    ) -> str | None:
        """Ask a question with its dropdown options in the same message.

        Up to 7 options render as inline keyboard buttons; longer lists are
        numbered and answered by replying with the number or the option text.
        Returns the picked option text, or ``None`` on Skip/timeout.
        """
        if not self.is_configured:
            raise TelegramNotConfiguredError(
                "Telegram prompting unavailable: set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID to answer personal screener questions."
            )

        opts = [o.strip() for o in (options or []) if o and o.strip()]
        if not opts:
            return await self.ask(question, timeout=timeout)

        numbered = len(opts) > 7
        msg_id = await self._send_options(question, opts, numbered)
        if msg_id is None:
            return None

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            updates = await self._fetch_updates(timeout=min(_MAX_POLL_TIMEOUT, remaining + 1))
            for upd in updates:
                self._last_update_id = max(self._last_update_id, upd.get("update_id", 0))
                picked = self._extract_option_pick(upd, msg_id, opts, numbered)
                if picked is _SKIP_SENTINEL:
                    return None
                if picked is not None:
                    return picked
            await asyncio.sleep(min(1.0, max(0.05, remaining)))

    async def _send_options(self, question: str, opts: list[str], numbered: bool) -> int | None:
        text = question
        if numbered:
            shown = opts[:_NUMBERED_DISPLAY_LIMIT]
            text += "\n\nOptions:\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(shown, 1))
            hidden = len(opts) - len(shown)
            if hidden > 0:
                text += f"\n… and {hidden} more (reply with the exact option text)"
            text += "\n\nReply with a number or the exact option text."
            # Telegram caps messages at 4096 chars; keep the numbered list
            # within budget even when option texts are long.
            while len(text) > _MAX_MESSAGE_CHARS and shown:
                shown = shown[:-1]
                hidden = len(opts) - len(shown)
                text = question + "\n\nOptions:\n" + "\n".join(
                    f"{i}. {o}" for i, o in enumerate(shown, 1)
                )
                if hidden > 0:
                    text += f"\n… and {hidden} more (reply with the exact option text)"
                text += "\n\nReply with a number or the exact option text."
        keyboard = (
            []
            if numbered
            else [[{"text": o, "callback_data": f"opt:{i}"}] for i, o in enumerate(opts)]
        )
        keyboard.append([{"text": "Skip", "callback_data": "skip"}])
        payload: dict[str, Any] = {
            "chat_id": self._primary_chat_id,
            "text": text,
            "disable_web_page_preview": True,
            "reply_markup": {
                "force_reply": True,
                "input_field_placeholder": "Tap an option, or reply with your answer...",
                "inline_keyboard": keyboard,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(TELEGRAM_SEND.format(token=self.bot_token), json=payload)
                if resp.status_code != 200:
                    logger.warning("Telegram send options failed", status=resp.status_code)
                    return None
                data = resp.json()
                if not data.get("ok"):
                    return None
                return (data.get("result") or {}).get("message_id")
        except Exception as e:
            logger.warning("Telegram send options error", exception=str(e))
            return None

    def _extract_option_pick(
        self, upd: dict[str, Any], msg_id: int, opts: list[str], numbered: bool
    ) -> str | None | Any:
        """Return a picked option text, ``_SKIP_SENTINEL`` for Skip, or None."""
        cb = upd.get("callback_query") or {}
        cb_msg = cb.get("message") or {}
        if cb.get("message_id") == msg_id or cb_msg.get("message_id") == msg_id:
            data = cb.get("data") or ""
            if data == "skip":
                return _SKIP_SENTINEL
            # Numbered messages have no opt: buttons, so an opt: callback there
            # can only be stale/forged — ignore it.
            if data.startswith("opt:") and not numbered:
                try:
                    return opts[int(data[4:])]
                except (ValueError, IndexError):
                    return None
        reply = self._extract_reply(upd, msg_id)
        if reply is None:
            return None
        low = reply.lower()
        for o in opts:
            if o.lower() == low:
                return o
        if numbered:
            if reply.isdigit():
                try:
                    return opts[int(reply) - 1]
                except IndexError:
                    return None
            if low.startswith("#"):
                try:
                    return opts[int(low[1:]) - 1]
                except (ValueError, IndexError):
                    return None
        return None
