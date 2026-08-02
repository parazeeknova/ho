"""Telegram prompting bridge for unknown job-application screener questions.

When the RAG pipeline cannot answer a question from the persona knowledge base
(``__ASK_USER__``), the bridge sends the question to the user on Telegram and
waits for their reply. Answers are then fed back into the form and persisted
into the persona knowledge base via ``ScreenerRAG.learn``.
"""

from __future__ import annotations

import asyncio
import os
import re
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

# Edit-distance threshold for forgiving a small typo in a typed answer against
# an option label, relative to the reply length. Small discrepancies only —
# never enough to jump between distinct options.
_FUZZY_MAX_EDIT = 2


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings (case-sensitive; callers
    normalize first). Handles unicode correctly."""
    a, b = str(a or ""), str(b or "")
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


class TelegramNotConfiguredError(RuntimeError):
    """Raised when a user prompt is required but Telegram is not configured."""


class TelegramSendError(RuntimeError):
    """Raised when a question message could not be delivered to Telegram.

    A send failure is NOT a user decline: the prompt never reached the user, so
    ``resolve_question`` must never map it to Skip or to a decline option.
    """


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
        # Message ids of every chunk sent for the current question. A reply to
        # ANY of them (or a plain message) counts as the answer.
        self._option_msg_ids: set[int] = set()

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
        timeout elapses. The user may answer by replying to the question OR by
        typing the answer as a normal chat message.
        """
        if not self.is_configured:
            raise TelegramNotConfiguredError(
                "Telegram prompting unavailable: set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID to answer personal screener questions."
            )

        await self._fast_forward()
        msg_id = await self._send_question(question)
        if msg_id is None:
            raise TelegramSendError(f"Telegram question not sent: {question}")
        # Only the anchor message carries the Skip button and force-reply for a
        # plain question; replies to it (or plain messages) are the answer.
        self._option_msg_ids = {msg_id}

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            updates = await self._fetch_updates(timeout=min(_MAX_POLL_TIMEOUT, remaining + 1))
            for upd in updates:
                self._last_update_id = max(self._last_update_id, upd.get("update_id", 0))
                if self._is_skip_callback(upd):
                    return None
                reply = self._extract_reply_any(upd)
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

    async def _fast_forward(self) -> None:
        """Jump past stale, unconfirmed updates so this instance only ever
        polls for freshly-typed messages.

        A brand-new bridge starts at ``update_id`` 0. Without fast-forwarding,
        the first ``getUpdates`` call pages through the entire accumulated
        history (10 updates at a time) before reaching a new answer, which can
        stall the prompt or misattribute an old message (e.g. a plain answer
        the user typed during a previous run that was never confirmed).
        ``offset=-1`` returns the most recent update; everything older is then
        treated as read, and subsequent polls only see new messages.
        """
        if self._last_update_id > 0:
            return
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    TELEGRAM_UPDATES.format(token=self.bot_token),
                    params={"timeout": 0, "limit": 1, "offset": -1},
                )
                if resp.status_code != 200:
                    return
                data = resp.json()
                if not data.get("ok"):
                    return
                results = data.get("result") or []
                if results:
                    self._last_update_id = max(
                        self._last_update_id, int(results[-1].get("update_id") or 0)
                    )
        except Exception as e:
            logger.debug(
                "Telegram fast-forward failed", source="autofill.telegram", exception=str(e)
            )

    def _is_skip_callback(self, upd: dict[str, Any]) -> bool:
        cb = upd.get("callback_query") or {}
        cb_msg = cb.get("message") or {}
        # Accept a Skip on ANY message we sent for the current question (anchor,
        # continuation chunk, or hint) — each carries its own Skip button.
        return cb.get("data") == "skip" and cb_msg.get("message_id") in self._option_msg_ids

    def _extract_reply_any(self, upd: dict[str, Any]) -> str | None:
        """Like ``_extract_reply`` but accepts a reply to ANY of the chunk
        message ids sent for the current question (plus plain messages)."""
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
        if reply_to is not None and reply_to not in self._option_msg_ids:
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
        The user may reply to the question or type the answer normally; plain
        text answers are matched to options (exact, then unambiguous substring,
        then conservative edit-distance). A plain reply that matches nothing
        triggers a one-time hint re-prompt instead of propagating a bad answer.
        Returns the picked option text, or ``None`` on Skip/timeout.
        """
        if not self.is_configured:
            raise TelegramNotConfiguredError(
                "Telegram prompting unavailable: set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID to answer personal screener questions."
            )

        opts = [o.strip() for o in (options or []) if o and o.strip()]
        if not opts:
            # Options could not be read (e.g. async loader). Never ask a
            # dropdown as a plain text question — tell the user it is one so
            # they answer with an option label they saw on the form.
            return await self.ask_dropdown(question, timeout=timeout)

        await self._fast_forward()
        numbered = len(opts) > 7
        msg_id = await self._send_options(question, opts, numbered)
        if msg_id is None:
            raise TelegramSendError(f"Telegram options not sent: {question}")

        deadline = time.monotonic() + timeout
        hinted = False
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
                # A human message (reply to ANY chunk, or a plain message) we
                # couldn't match to any option: send a one-time hint and keep
                # waiting. The answer never silently becomes a bad fill.
                if self._extract_reply_any(upd) is not None and not hinted:
                    await self._send_hint(opts, numbered)
                    hinted = True
            await asyncio.sleep(min(1.0, max(0.05, remaining)))

    async def ask_dropdown(
        self, question: str, timeout: float = DEFAULT_QUESTION_TIMEOUT
    ) -> str | None:
        """Ask a dropdown question whose option list could not be read.

        The user is told it is a dropdown (so they reply with the option label
        they saw on the form) and shown the Skip affordance. Returns the typed
        answer, or ``None`` on Skip/timeout.
        """
        if not self.is_configured:
            raise TelegramNotConfiguredError(
                "Telegram prompting unavailable: set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID to answer personal screener questions."
            )
        await self._fast_forward()
        msg = (
            f"{question}\n\n(Note: this is a dropdown on the application form. "
            "Reply with the exact option you saw, or a close match.)"
        )
        return await self.ask(msg, timeout=timeout)

    async def _send_hint(self, opts: list[str], numbered: bool) -> None:
        """Send a one-time hint after a reply matched no option. The hint is a
        force-reply with its own Skip button, so replies to it and its Skip are
        both tracked as part of the current question."""
        if not opts:
            text = "That didn't match any option. Please reply again."
        elif numbered:
            text = (
                "That didn't match an option. Reply with a number or the exact "
                "option text from the list."
            )
        else:
            text = (
                "That didn't match an option. Reply with the exact option text "
                "shown above, or a close match."
            )
        payload: dict[str, Any] = {
            "chat_id": self._primary_chat_id,
            "text": text,
            "disable_web_page_preview": True,
            "reply_markup": {
                "force_reply": True,
                "input_field_placeholder": "Reply again...",
                "inline_keyboard": [[{"text": "Skip", "callback_data": "skip"}]],
            },
        }
        hint_id = await self._send_raw(payload)
        if hint_id is not None:
            self._option_msg_ids.add(hint_id)

    async def _send_options(self, question: str, opts: list[str], numbered: bool) -> int | None:
        self._option_msg_ids.clear()
        if numbered:
            # Long lists are chunked across multiple messages so EVERY option is
            # always visible — never truncated behind "... and N more". Each
            # chunk stays within Telegram's char budget. The first chunk is the
            # anchor (Skip button + force reply); continuation chunks are plain
            # messages. Numbering is global across the whole list.
            chunks = self._chunk_options(question, opts)
            if not chunks:
                return None
            first, first_text = chunks[0]
            keyboard = [[{"text": "Skip", "callback_data": "skip"}]]
            payload: dict[str, Any] = {
                "chat_id": self._primary_chat_id,
                "text": first_text,
                "disable_web_page_preview": True,
                "reply_markup": {
                    "force_reply": True,
                    "input_field_placeholder": "Reply with a number or option text...",
                    "inline_keyboard": keyboard,
                },
            }
            msg_id = await self._send_raw(payload)
            if msg_id is None:
                return None
            self._option_msg_ids.add(msg_id)
            for _, text in chunks[1:]:
                # Continuation chunks are plain messages, but a user may reply
                # to them — track their ids so the reply is accepted as the
                # answer (never dropped as a stranger message).
                chunk_id = await self._send_chunk(text)
                if chunk_id is not None:
                    self._option_msg_ids.add(chunk_id)
            return msg_id

        keyboard = [[{"text": o, "callback_data": f"opt:{i}"}] for i, o in enumerate(opts)]
        keyboard.append([{"text": "Skip", "callback_data": "skip"}])
        payload = {
            "chat_id": self._primary_chat_id,
            "text": question,
            "disable_web_page_preview": True,
            "reply_markup": {
                "force_reply": True,
                "input_field_placeholder": "Tap an option, or reply with your answer...",
                "inline_keyboard": keyboard,
            },
        }
        msg_id = await self._send_raw(payload)
        if msg_id is not None:
            self._option_msg_ids.add(msg_id)
        return msg_id

    async def _send_chunk(self, text: str) -> int | None:
        """Send a plain continuation message and return its message id."""
        payload: dict[str, Any] = {
            "chat_id": self._primary_chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        return await self._send_raw(payload)

    async def _send_raw(self, payload: dict[str, Any]) -> int | None:
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

    @staticmethod
    def _chunk_options(question: str, opts: list[str]) -> list[tuple[str, str]]:
        """Split a numbered option list into per-message chunks under the char
        budget. Returns ``[(anchor_id?, text), ...]`` — each entry is the
        display text of one message; the first is the anchor. ``anchor_id`` is
        unused placeholder (kept as text string for parity with callers).
        """
        chunks: list[tuple[str, str]] = []
        anchor_prefix = f"{question}\n\nOptions:\n"
        cont_prefix = "Options (continued):\n"
        trailer = "\n\nReply with a number or the exact option text."
        current: list[str] = []
        current_len = len(anchor_prefix) + len(trailer)
        first = True
        for i, o in enumerate(opts, 1):
            line = f"{i}. {o}"
            prefix = anchor_prefix if first else cont_prefix
            if current and current_len + len(line) + 1 > _MAX_MESSAGE_CHARS:
                chunks.append((prefix, prefix + "\n".join(current) + trailer))
                current = []
                current_len = len(cont_prefix) + len(trailer)
                first = False
            current.append(line)
            current_len += len(line) + 1
        if current:
            prefix = anchor_prefix if first else cont_prefix
            chunks.append((prefix, prefix + "\n".join(current) + trailer))
        return chunks

    def _extract_option_pick(
        self, upd: dict[str, Any], msg_id: int, opts: list[str], numbered: bool
    ) -> str | None | Any:
        """Return a picked option text, ``_SKIP_SENTINEL`` for Skip, or None."""
        cb = upd.get("callback_query") or {}
        cb_msg = cb.get("message") or {}
        # A Skip on ANY message we sent for this question (anchor, chunk, or
        # hint) is honored — each carries its own Skip button.
        if (
            cb.get("data") == "skip"
            and (
                cb.get("message_id") in self._option_msg_ids
                or cb_msg.get("message_id") in self._option_msg_ids
            )
        ):
            return _SKIP_SENTINEL
        if cb.get("message_id") == msg_id or cb_msg.get("message_id") == msg_id:
            data = cb.get("data") or ""
            # Numbered messages have no opt: buttons, so an opt: callback there
            # can only be stale/forged — ignore it.
            if data.startswith("opt:") and not numbered:
                try:
                    return opts[int(data[4:])]
                except (ValueError, IndexError):
                    return None
        # Accept a reply to ANY sent chunk (plus plain messages) so answering a
        # continuation chunk works; numbered picks use the global list index.
        reply = self._extract_reply_any(upd)
        if reply is None:
            return None
        low = reply.strip().lower()
        # 1) Exact option text (case-insensitive).
        for o in opts:
            if o.lower() == low:
                return o
        # 2) Numbered pick: a bare digit or "#n" (only meaningful for numbered
        #    lists). "0" is never a valid pick — int("0")-1 == -1 would wrap to
        #    the LAST option, so require a positive index.
        if numbered:
            if low.isdigit():
                try:
                    idx = int(low)
                    return opts[idx - 1] if idx >= 1 else None
                except IndexError:
                    return None
            if low.startswith("#"):
                try:
                    idx = int(low[1:])
                    return opts[idx - 1] if idx >= 1 else None
                except (ValueError, IndexError):
                    return None
        # 3) Fuzzy text match for answers typed as a normal message that are
        #    not the exact option label (e.g. "bachelor's" -> "Bachelor's Degree").
        fuzzy = self._fuzzy_option(reply, opts)
        if fuzzy is not None:
            return fuzzy
        return None

    @staticmethod
    def _fuzzy_option(reply: str, opts: list[str]) -> str | None:
        """Resolve a typed plain-text answer to an option via an unambiguous
        match. Order: unique substring ("bachelor's" -> "Bachelor's Degree"),
        then a conservative edit-distance match that forgives a small typo
        ("bachlors" -> "Bachelor's Degree") but only when exactly one option is
        within the tolerance — never enough to jump between distinct options.
        Apostrophes are normalized first."""
        def norm(t: str) -> str:
            return re.sub(r"[\u2019']", "", (t or "").strip().lower())

        low = norm(reply)
        if not low:
            return None
        normalized_opts = [(o, norm(o)) for o in opts]
        subs = [o for o, no in normalized_opts if low in no]
        if len(subs) == 1:
            return subs[0]
        for token in low.split():
            tok_subs = [o for o, no in normalized_opts if token in no]
            if len(tok_subs) == 1:
                return tok_subs[0]
        # Edit-distance fallback: a small typo against a single token of exactly
        # one option ("bachlors" -> "Bachelor's Degree", but "degre" matches
        # both degree tokens so it stays ambiguous).
        threshold = max(1, min(_FUZZY_MAX_EDIT, len(low) // 4))
        near: list[str] = []
        for o, no in normalized_opts:
            tokens = [t for t in no.split() if t]
            if any(t and edit_distance(low, t) <= threshold for t in tokens):
                near.append(o)
        if len(near) == 1:
            return near[0]
        return None
