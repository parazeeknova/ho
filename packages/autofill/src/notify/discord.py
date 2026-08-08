"""Discord question bridge for autofill.

The autofill.src.core.worker/CLI never opens a Discord gateway (that would race the
ingest DiscordAgent on the same bot token). Instead it sends questions over
the Discord REST API with interactive buttons, records the sent message ids in
the shared ``discord_question_mailbox`` table, and polls the DB for the user's
answer — which the single ingest DiscordAgent writes back (text replies via
``reply_to``, button presses via ``custom_id``).

Fallback: when no gateway agent is alive (heartbeat stale — standalone CLI),
the bridge polls the mailbox only; there is no direct Discord polling, so the
mailbox is the ONLY answer channel. If no agent is up the question times out
and the job is deferred (same as Telegram's ask-user fallback).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
from src.logging import get_logger
from src.retry import RateLimiter, retry_http

logger = get_logger("autofill.src.notify.discord")

DISCORD_API = "https://discord.com/api/v10"
DEFAULT_QUESTION_TIMEOUT = 300.0

# Discord's API rate limit for message sends is 5 requests / 5s per channel.
# A shared limiter serializes the autofill bridge's REST sends so a burst of
# questions (batch fills) never trips 429s — the same discipline the ingest
# side applies to its own outbound calls.
_DISCORD_RATE_LIMITER = RateLimiter(1.0)

_SKIP_SENTINEL = object()

_PERSONA_JSON = Path(__file__).resolve().parents[4] / "data" / "persona.json"


def _normalise_question(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _persona_exact_answers() -> dict[str, str]:
    """Normalized question -> answer from persona.json (never re-ask a known fact)."""
    try:
        data = json.loads(_PERSONA_JSON.read_text())
    except Exception:
        return {}
    out: dict[str, str] = {}
    for entry in data.get("answers", []):
        q = (entry.get("question") or "").strip()
        a = (entry.get("answer") or "").strip()
        if q and a:
            out[_normalise_question(q)] = a
    return out


class DiscordNotConfiguredError(RuntimeError):
    pass


class DiscordSendError(RuntimeError):
    pass


def _norm_option(text: str) -> str:
    return re.sub(r"[\u2019']", "", (text or "").strip().lower())


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class DiscordQuestionBridge:
    """Send autofill screener questions to Discord and collect the answer.

    Sends via REST (no gateway), then waits on the shared mailbox that the
    ingest DiscordAgent populates. Never opens its own gateway.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        channel_id: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.bot_token = bot_token if bot_token is not None else os.getenv("DISCORD_BOT_TOKEN", "")
        self.channel_id = (
            channel_id or chat_id
            if (channel_id or chat_id) is not None
            else os.getenv("DISCORD_CHANNEL_ID", "")
        )
        self._db_ref: Any | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.channel_id)

    async def _db(self) -> Any:
        if self._db_ref is None:
            from autofill.src.core.db import AutofillDB

            self._db_ref = await AutofillDB.create()
        return self._db_ref

    async def _poller_alive(self) -> bool:
        """True when the ingest DiscordAgent gateway is the live consumer."""
        try:
            db = await self._db()
            return await db.poller_alive()
        except Exception:
            return False

    # ── REST send ─────────────────────────────────────────────────────

    async def _send_payload(self, payload: dict[str, Any]) -> int | None:
        """POST a message to the channel (or active thread); returns the id or None.

        Rate-limited (1 req/s shared limiter) and retried with exponential
        backoff on transient network/5xx/429 failures via the shared retry
        module, so a flaky connection or Discord throttle doesn't silently
        drop a question the user never sees.

        When the ingest gateway has recorded an active sweep thread, messages
        are sent into that thread so deferred/captcha/queue notifications stay
        with the sweep instead of landing in the main channel.
        """
        await _DISCORD_RATE_LIMITER.acquire()

        target_id = self.channel_id
        try:
            db = await self._db()
            thread = await db.active_thread()
            # A valid Discord thread id is a 15-20 digit snowflake. Reject any
            # corrupt/stale value (e.g. "702") — posting to it 404s with
            # "Unknown Channel", which silently drops every notification.
            if thread and re.fullmatch(r"\d{15,20}", str(thread)):
                target_id = thread
        except Exception:
            pass

        async def _post() -> httpx.Response:
            async with httpx.AsyncClient(timeout=15.0) as client:
                return await client.post(
                    f"{DISCORD_API}/channels/{target_id}/messages",
                    headers={"Authorization": f"Bot {self.bot_token}"},
                    json=payload,
                )

        try:
            resp = await retry_http(_post, max_retries=2, base_delay=0.5, max_delay=4.0)
            # If a thread-targeted post 404s (thread archived/deleted), fall
            # back to the main channel so a notification is never lost.
            if resp.status_code == 404 and target_id != self.channel_id:
                with contextlib.suppress(Exception):
                    db = await self._db()
                    await db.clear_active_thread()
                fallback = await retry_http(
                    lambda: httpx.AsyncClient(timeout=15.0).post(
                        f"{DISCORD_API}/channels/{self.channel_id}/messages",
                        headers={"Authorization": f"Bot {self.bot_token}"},
                        json=payload,
                    ),
                    max_retries=1,
                    base_delay=0.5,
                    max_delay=2.0,
                )
                if fallback.status_code < 300:
                    return fallback.json().get("id")
                logger.warning(
                    "Discord send failed (channel fallback)",
                    status=fallback.status_code,
                    body=fallback.text[:200],
                )
                return None
            if resp.status_code >= 300:
                logger.warning(
                    "Discord send failed",
                    status=resp.status_code,
                    body=resp.text[:200],
                )
                return None
            data = resp.json()
            return data.get("id")
        except Exception as e:
            logger.warning("Discord send error", error=str(e))
            return None

    def _skip_button(self) -> dict[str, Any]:
        return {"type": 2, "style": 2, "label": "Skip", "custom_id": "skip"}

    def _option_buttons(self, opts: list[str], numbered: bool) -> list[dict[str, Any]]:
        buttons: list[dict[str, Any]] = []
        if numbered:
            # Long lists: keep the options in the message text; no per-option
            # buttons, but Skip + the reply flow.
            buttons.append(self._skip_button())
            return buttons
        for i, o in enumerate(opts[:5]):
            buttons.append({"type": 2, "style": 1, "label": o[:80], "custom_id": f"opt:{i}"})
        buttons.append(self._skip_button())
        return buttons

    async def send_question(self, question: str, options: list[str] | None = None) -> int | None:
        numbered = bool(options and len(options) > 5)
        buttons = self._option_buttons(options or [], numbered)
        payload: dict[str, Any] = {"content": question}
        if buttons:
            payload["components"] = [{"type": 1, "components": buttons}]
        return await self._send_payload(payload)

    async def send(self, text: str) -> bool:
        """Send a plain message (deferral alerts, digests). No reply flow."""
        if not self.is_configured or not text:
            return False
        return await self._send_payload({"content": text[:1900]}) is not None

    async def send_chunked(self, text: str, max_len: int = 1900) -> bool:
        if not text:
            return True
        if len(text) <= max_len:
            return await self.send(text)
        pieces: list[str] = []
        current = ""
        for para in text.split("\n\n"):
            if not para.strip():
                continue
            candidate = (current + "\n\n" + para).strip()
            if current and len(candidate) > max_len:
                pieces.append(current)
                current = para
            else:
                current = candidate
        if current.strip():
            pieces.append(current)
        ok = True
        for piece in pieces:
            if not await self.send(piece):
                ok = False
        return ok

    # ── ask flow ──────────────────────────────────────────────────────

    async def ask(self, question: str, timeout: float = DEFAULT_QUESTION_TIMEOUT) -> str | None:
        """Ask a question and wait for the answer. Returns text, None on timeout."""
        if not self.is_configured:
            raise DiscordNotConfiguredError(
                "Discord prompting unavailable: set DISCORD_BOT_TOKEN and "
                "DISCORD_CHANNEL_ID to answer personal screener questions."
            )
        known = _persona_exact_answers().get(_normalise_question(question))
        if known:
            return known
        msg_id = await self.send_question(question)
        if msg_id is None:
            raise DiscordSendError(f"Discord question not sent: {question}")
        return await self._collect_mailbox(question, None, False, msg_id, timeout)

    async def ask_options(
        self,
        question: str,
        options: list[str],
        timeout: float = DEFAULT_QUESTION_TIMEOUT,
    ) -> str | None:
        opts = [o.strip() for o in (options or []) if o and o.strip()]
        if not opts:
            return await self.ask_dropdown(question, timeout=timeout)
        known = _persona_exact_answers().get(_normalise_question(question))
        if known:
            return known
        msg_id = await self.send_question(question, opts)
        if msg_id is None:
            raise DiscordSendError(f"Discord options not sent: {question}")
        return await self._collect_mailbox(question, opts, len(opts) > 5, msg_id, timeout)

    async def ask_dropdown(
        self, question: str, timeout: float = DEFAULT_QUESTION_TIMEOUT
    ) -> str | None:
        msg = (
            f"{question}\n\n*(This is a dropdown on the application form — "
            "reply with the exact option you saw, or a close match.)*"
        )
        return await self.ask(msg, timeout=timeout)

    async def _collect_mailbox(
        self,
        question: str,
        opts: list[str] | None,
        numbered: bool,
        msg_id: int,
        timeout: float,
    ) -> str | None:
        qid = f"q-{uuid.uuid4().hex[:12]}"
        db = await self._db()
        # Discord's REST API returns snowflake ids as JSON STRINGS. The mailbox
        # column is BIGINT[], so coerce before insert — a str snowflake inside
        # the array is "invalid array element" for Postgres (the Deepgram job
        # died on exactly this: ['1535503494239359006']). Coerce only when the
        # value is a numeric snowflake; non-numeric test/fallback ids pass
        # through untouched.
        msg_id_int = None
        try:
            if msg_id is not None and str(msg_id).isdigit():
                msg_id_int = int(msg_id)
        except TypeError, ValueError:
            msg_id_int = None
        if msg_id_int is None and msg_id is not None:
            msg_id_int = msg_id  # non-numeric id (tests, legacy): keep as-is
        await db.open_mailbox_question(
            qid, self.channel_id, [msg_id_int] if msg_id_int is not None else [], question
        )
        deadline = asyncio.get_event_loop().time() + timeout
        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return None
                state_answer = await db.poll_mailbox_question(qid)
                if state_answer and state_answer[0] == "answered":
                    return self._interpret_answer(state_answer[1], opts, numbered)
                await asyncio.sleep(1.0)
        finally:
            await db.close_mailbox_question(qid, "timed_out")

    def _interpret_answer(
        self, answer: str | None, opts: list[str] | None, numbered: bool
    ) -> str | None | Any:
        if not answer:
            return None
        if answer == "skip":
            return _SKIP_SENTINEL
        if opts is None:
            return answer
        if answer.startswith("opt:") and not numbered:
            try:
                return opts[int(answer[4:])]
            except ValueError, IndexError:
                return None
        return self._match_reply_to_option(answer, opts, numbered)

    def _match_reply_to_option(self, reply: str, opts: list[str], numbered: bool) -> str | None:
        low = reply.strip().lower()
        for o in opts:
            if o.lower() == low:
                return o
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
                except ValueError, IndexError:
                    return None
        return self._fuzzy_option(reply, opts)

    @staticmethod
    def _fuzzy_option(reply: str, opts: list[str]) -> str | None:
        low = _norm_option(reply)
        if not low:
            return None
        normalized = [(o, _norm_option(o)) for o in opts]
        subs = [o for o, no in normalized if low in no]
        if len(subs) == 1:
            return subs[0]
        for token in low.split():
            tok_subs = [o for o, no in normalized if token in no]
            if len(tok_subs) == 1:
                return tok_subs[0]
        threshold = max(1, min(4, len(low) // 4))
        near: list[str] = []
        for o, no in normalized:
            tokens = [t for t in no.split() if t]
            if any(t and edit_distance(low, t) <= threshold for t in tokens):
                near.append(o)
        if len(near) == 1:
            return near[0]
        return None
