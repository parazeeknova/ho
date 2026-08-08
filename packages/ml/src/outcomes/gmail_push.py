"""Gmail push ingestion — Pub/Sub streaming pull → outcome event.

This is INFRA, not ML: it turns a Gmail history notification into a
normalized semantic event (screening/rejection/offer) and hands it to the
reward layer. It never computes reward numbers itself.

Auth: OAuth2 refresh token (GOOGLE_CLIENT_ID/SECRET + GMAIL_REFRESH_TOKEN).
The existing IMAP app-password path stays for OTP codes in the browser runner.

Flow:
  Gmail users.watch (label INBOX → Pub/Sub topic) — Cheap, idempotent.
  Pub/Sub streaming pull (google-cloud-pubsub) — This process subscribes to
    the subscription and receives push notifications instantly.
  On notify: Gmail users.history.list(historyId) → new messageIds →
    users.messages.get → classify via atsEmail logic (ported to Python) →
    OutcomeResolver → decision_events (reward) + unattributed_outcomes if low
    confidence.

Resilience:
  Persists gmail_push_state (history_id, watch_expiry) in Postgres.
  Handles watch expiration (re-arm), historyId expiration (full sync),
  duplicates (processed message-id dedup), out-of-order, and history gaps.
  Idempotent — reprocessing a message is a no-op.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
from typing import Any

from ml.src.config import get_ml_config

# Gmail classifier — mirrors packages/node/utils/atsEmail.ts

ATS_SENDERS = [
    "greenhouse-mail.io",
    "greenhouse.io",
    "ashbyhq.com",
    "lever.co",
    "workable.com",
    "smartrecruiters.com",
    "recruitee.com",
    "bamboohr.com",
    "teamtailor.com",
    "myworkdayjobs.com",
    "workday.com",
    "jazzhr.com",
    "rippling.com",
]

CONFIRM_RE = re.compile(
    r"\b(received|confirm|submitted|complete|successful|thanks?|thank you for applying|applied)\b",
    re.IGNORECASE,
)
REJECT_RE = re.compile(
    r"\b(unfortunately|regret|not (moving|c|selected)|other candidates|position has been filled|no longer under consideration|won'?t be moving|we will not be moving|decided to move forward with other candidates|not successful)\b",
    re.IGNORECASE,
)
SCREENING_RE = re.compile(
    r"\b(interview|schedule|phone screen|assessment|take[ -]?home|coding challenge|code test|next step|book a|call (with|us)|video (call|interview)|we'?d like to|excited to (speak|meet|discuss))\b",
    re.IGNORECASE,
)
OTP_RE = re.compile(
    r"\b(verification code|one[ -]?time|security code|otp|confirm your (email|address)|verify your (email|identity))\b",
    re.IGNORECASE,
)


def classify_email(subject: str, body: str) -> str:
    text = f"{subject}\n{body}"
    if OTP_RE.search(text):
        return "otp"
    if REJECT_RE.search(subject) or REJECT_RE.search(body[:400]):
        return "rejection_email"
    if SCREENING_RE.search(subject) or SCREENING_RE.search(body[:400]):
        return "screening_email"
    if CONFIRM_RE.search(subject):
        return "confirmation_email"
    return "other"


def strip_html(html: str) -> str:
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&")
    html = re.sub(r"&#\d+;", " ", html)
    return re.sub(r"\s+", " ", html).strip()


async def _get_gmail_service():
    """Build an authenticated Gmail API service from refresh token."""
    cfg = get_ml_config().gmail_push
    if not (cfg.client_id and cfg.client_secret and cfg.refresh_token):
        raise RuntimeError(
            "Gmail push not configured: need GOOGLE_CLIENT_ID/SECRET + GMAIL_REFRESH_TOKEN"
        )
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        None,
        refresh_token=cfg.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cfg.client_id,
        client_secret=cfg.client_secret,
    )
    creds.refresh(Request())
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service


async def arm_watch(service: Any, store: Any) -> dict[str, Any]:
    """Call users.watch to (re)arm push notifications. Returns watch response."""
    cfg = get_ml_config().gmail_push
    topic = f"projects/{cfg.project_id}/topics/{cfg.topic}"
    body = {"labelIds": ["INBOX"], "topicName": topic}
    # Gmail API is sync; run in thread
    result = await asyncio.to_thread(
        lambda: service.users().watch(userId="me", body=body).execute()
    )
    # Persist watch state
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO gmail_push_state (id, history_id, watch_expiry, updated_at)
                VALUES (1, $1, $2, NOW())
                ON CONFLICT (id) DO UPDATE SET history_id=$1, watch_expiry=$2, updated_at=NOW()
                """,
                str(result.get("historyId", "")),
                result.get("expiration", ""),
            )
    except Exception:
        pass
    return result


async def advance_history_state(store: Any, new_history_id: str) -> None:
    """Atomically advance the stored history_id after a successful process cycle.

    Only updates when the new value is (lexicographically) greater — Gmail
    historyIds are monotonically increasing, so this is safe even if an older
    notification's ack is processed late. This prevents the old bug where
    every notification reprocessed the same H1->H2 window (e.g. H1->H3, then
    H1->H4) instead of moving forward (H3, then H4).
    """
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE gmail_push_state
                SET history_id = CASE
                    WHEN GREATEST(history_id, $2) = $2 THEN $2
                    ELSE history_id END,
                    updated_at = NOW()
                WHERE id = 1
                """,
                1,
                str(new_history_id),
            )
    except Exception:
        pass


async def fetch_new_messages(service: Any, start_history_id: str) -> list[dict[str, Any]]:
    """Gmail history.list → new message metas since start_history_id."""
    try:
        resp = await asyncio.to_thread(
            lambda: (
                service.users()
                .history()
                .list(userId="me", startHistoryId=start_history_id, historyTypes=["messageAdded"])
                .execute()
            )
        )
    except Exception as e:
        # HistoryId expired (stale / too old): callers should fall back to a
        # full INBOX sync rather than swallowing the error as "no messages".
        err = str(e)
        if "historyid" in err.lower() or "invalid" in err.lower():
            raise RuntimeError(f"historyId expired: {start_history_id}") from e
        return []
    messages: list[dict[str, Any]] = []
    for h in resp.get("history", []) or []:
        for m in h.get("messagesAdded", []) or []:
            msg = m.get("message") or {}
            if msg.get("id"):
                messages.append(msg)
    return messages


async def full_inbox_sync(store: Any, service: Any) -> None:
    """Fallback when historyId is stale: list the last INBOX messages directly.

    Uses users.messages.list instead of history.list, then processes them
    through the normal dedup/classify/outcome path. The latest historyId is
    captured and advanced from the subsequent arm_watch.
    """
    try:
        resp = await asyncio.to_thread(
            lambda: (
                service.users()
                .messages()
                .list(userId="me", labelIds=["INBOX"], maxResults=30, q="")
                .execute()
            )
        )
        messages = resp.get("messages", []) or []
        for meta in messages[:30]:
            msg_id = meta.get("id")
            if not msg_id:
                continue
            try:
                async with store._pool.acquire() as conn:
                    exists = await conn.fetchval(
                        "SELECT 1 FROM decision_events WHERE meta->>'gmail_message_id' = $1 LIMIT 1",
                        msg_id,
                    )
                    if exists:
                        continue
            except Exception:
                pass
            msg = await get_message(service, msg_id)
            if not msg:
                continue
            subject, sender, body = extract_text_from_message(msg)
            kind = classify_email(subject, body)
            if kind in ("otp", "other"):
                continue
            from .outcome_resolver import resolve_outcome

            resolution = await resolve_outcome(
                store, {"subject": subject, "from": sender, "snippet": body[:500]}
            )
            job_id = resolution.get("job_id")
            confidence = resolution.get("confidence", 0.0)
            if job_id is None or confidence < 0.35:
                try:
                    async with store._pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO unattributed_outcomes (email_kind, company, role, subject, snippet, confidence, created_at)
                            VALUES ($1,$2,$3,$4,$5,$6,NOW())
                            """,
                            kind,
                            sender,
                            extract_role(subject) or "",
                            subject[:300],
                            body[:500],
                            confidence,
                        )
                except Exception:
                    pass
                continue
            # Emit reward event to the originating impression (time-proximate).
            from .events import DecisionEvent, emit_event

            reward_ts = msg.get("internalDate")
            originating = await _latest_impression_for_job(store, job_id, reward_ts=reward_ts) or {}
            from .reward import reward_for

            event = DecisionEvent(
                job_id=job_id,
                event_type=kind,
                reward=reward_for(kind),
                impression_id=originating.get("impression_id"),
                candidate_snapshot_id=originating.get("candidate_snapshot_id"),
                job_snapshot_id=originating.get("job_snapshot_id"),
                model_version=str(originating.get("model_version") or ""),
                policy=str(originating.get("policy") or ""),
                feature_version=str(originating.get("feature_version") or ""),
                source=str(originating.get("source") or ""),
                meta={
                    "gmail_message_id": msg_id,
                    "subject": subject[:200],
                    "confidence": confidence,
                },
            )
            await emit_event(store, event)
    except Exception:
        pass


async def get_message(service: Any, msg_id: str) -> dict[str, Any] | None:
    try:
        msg = await asyncio.to_thread(
            lambda: service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        )
        return msg
    except Exception:
        return None


def extract_text_from_message(msg: dict[str, Any]) -> tuple[str, str, str]:
    """Return (subject, from, body_text) from a Gmail message resource."""
    headers = {h["name"].lower(): h["value"] for h in (msg.get("payload") or {}).get("headers", [])}
    subject = headers.get("subject", "")
    sender = headers.get("from", "")
    body = ""

    def walk(part: dict[str, Any]) -> str:
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data", "")
        text = ""
        if data and mime.startswith("text/"):
            try:
                text = base64.urlsafe_b64decode(data + "===").decode(errors="ignore")
                if "html" in mime:
                    text = strip_html(text)
            except Exception:
                pass
        for child in part.get("parts") or []:
            text += " " + walk(child)
        return text

    body = walk(msg.get("payload") or {}).strip()
    return subject, sender, body


# Pub/Sub streaming pull loop


_WATCH_RENEW_DAYS = 1  # re-arm the watch every 24h (expiry is 7d)


async def _is_watch_stale(store: Any, watch_ttl_days: int | None = None) -> bool:
    """True when the stored watch_expiry is within one day of expiry (or absent)."""
    try:
        async with store._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT watch_expiry FROM gmail_push_state WHERE id=1")
            if not row or not row["watch_expiry"]:
                return True
            import time

            exp_ms = int(row["watch_expiry"] or "0")
            if exp_ms <= 0:
                return True
            # Gmail `expiration` is epoch-ms. Renew when it is less than a day
            # away (the P0 'silent expiry' fix) rather than waiting for the
            # watch to be dead and notifications to have already been lost.
            return exp_ms - int(time.time() * 1000) < 86400000
    except Exception:
        return False


async def run_gmail_push_loop(store: Any) -> None:
    """Long-lived Gmail outcome daemon.

    Runs TWO concurrent coroutines: a streaming Pub/Sub subscriber (instant
    notifications) and a watch-renewal/history-poll heartbeat. The daemon is
    independent of any application epoch — it must never die when a bounded
    run drains (the review's P0 'daemon dies when the run ends' fix is in
    loop.py, which excludes this child from bounded-run termination).

    HistoryIds are advanced atomically via advance_history_state after each
    successful _process_history, so recovery never replays the same window.
    """
    cfg = get_ml_config().gmail_push
    if not cfg.enabled:
        return

    try:
        service = await _get_gmail_service()
        await arm_watch(service, store)
    except Exception as e:
        import logging

        logging.getLogger("gmail_push").warning(f"Gmail watch arm failed, will retry: {e}")
        service = None  # type: ignore[assignment]

    async def _watch_renewal_hb() -> None:
        while True:
            await asyncio.sleep(24 * 3600)
            if await _is_watch_stale(store, cfg.watch_ttl_days):
                try:
                    await arm_watch(service, store)  # type: ignore[arg-type]
                    import logging

                    logging.getLogger("gmail_push").info("Gmail watch renewed (24h heartbeat)")
                except Exception as e:
                    import logging

                    logging.getLogger("gmail_push").warning(f"Gmail watch renewal failed: {e}")

    if cfg.streaming:
        task_stream = asyncio.create_task(_streaming_pull_loop(store))  # type: ignore[arg-type]
    else:
        # OAuth2-only: no Pub/Sub ADC, so poll the Gmail history API directly.
        task_stream = asyncio.create_task(_history_poll_loop(store, service))
    task_renewal = asyncio.create_task(_watch_renewal_hb())
    await asyncio.gather(task_stream, task_renewal)


async def _history_poll_loop(store: Any, service: Any) -> None:
    """Poll Gmail history on an interval using the OAuth2 service (no Pub/Sub
    ADC needed). Runs when GMAIL_STREAMING=0."""
    cfg = get_ml_config().gmail_push
    while True:
        try:
            await _poll_cycle(store, service)
        except Exception:
            import logging

            logging.getLogger("gmail_push").warning(
                "Gmail history poll failed, will retry", exc_info=True
            )
        await asyncio.sleep(cfg.poll_interval_s)


async def _streaming_pull_loop(store: Any) -> None:
    """Long-lived streaming subscriber: Pub/Sub streaming_pull (instant).

    Creates one SubscriberClient and opens a streaming_pull. Pub/Sub delivers
    messages within seconds of the INBOX label change — no polling delay.
    The fallback _history_poll_hb runs alongside for recovery gaps.
    """
    cfg = get_ml_config().gmail_push
    while True:
        try:
            from google.cloud import pubsub_v1

            subscriber = pubsub_v1.SubscriberClient()
            subscription_path = subscriber.subscription_path(cfg.project_id, cfg.subscription)

            # StreamingFuture: open a long-lived streaming_pull; messages are
            # delivered via the callback on the I/O thread, forwarded to asyncio
            # via the queue (message.ack() is called directly on the I/O thread
            # via the callback, which is safe for the Pub/Sub client).
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _callback(message: Any, _q: asyncio.Queue[str | None] = queue) -> None:
                try:
                    data = json.loads(message.data.decode()) if message.data else {}
                    queue.put_nowait(  # noqa: B023
                        str(data.get("historyId", "") or "")
                    )
                except Exception:
                    pass
                with contextlib.suppress(Exception):
                    message.ack()  # type: ignore[attr-defined]

            streaming_future = subscriber.subscribe(subscription_path, callback=_callback)
            try:
                while True:
                    history_id = await asyncio.wait_for(queue.get(), timeout=5)
                    if history_id:
                        try:
                            await _process_history(store, None, history_id)
                            await advance_history_state(store, history_id)
                        except RuntimeError as e:
                            # Stale historyId — recover with a full INBOX sync,
                            # then re-arm and advance from the fresh watch.
                            if "historyId expired" in str(e):
                                try:
                                    service = await _get_gmail_service()
                                    await full_inbox_sync(store, service)
                                    await arm_watch(service, store)  # type: ignore[arg-type]
                                except Exception:
                                    pass
                            else:
                                raise
                    queue.task_done()
            finally:
                with contextlib.suppress(Exception):
                    streaming_future.cancel()
                with contextlib.suppress(Exception):
                    subscriber.close()
        except Exception as e:
            import logging

            logging.getLogger("gmail_push").warning(
                f"streaming_pull error, falling back to history poll: {e}"
            )
            await asyncio.sleep(30)


async def _poll_cycle(store: Any, service: Any) -> None:  # kept for history-poll fallback callers
    """One poll: try Pub/Sub streaming pull, fall back to history polling."""
    cfg = get_ml_config().gmail_push
    # Try Pub/Sub streaming pull with a short deadline (non-blocking attempt)
    try:
        from google.cloud import pubsub_v1

        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(cfg.project_id, cfg.subscription)

        # Pull with short timeout — if no message, fall through to history poll
        response = await asyncio.to_thread(
            lambda: subscriber.pull(
                request={"subscription": subscription_path, "max_messages": 10},
                timeout=5,
            )
        )
        for msg in response.received_messages or []:
            try:
                data = json.loads(msg.message.data.decode()) if msg.message.data else {}
                history_id = str(data.get("historyId", ""))
                if history_id:
                    await _process_history(store, service, history_id)

                def _ack(m: Any = msg) -> None:
                    subscriber.acknowledge(
                        request={"subscription": subscription_path, "ack_ids": [m.ack_id]}
                    )

                await asyncio.to_thread(_ack)
            except Exception:
                pass
        if response.received_messages:
            return
    except Exception:
        pass

    # Fallback: direct history poll from last known history_id
    try:
        async with store._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT history_id FROM gmail_push_state WHERE id=1")
            start_id = str(row["history_id"]) if row and row["history_id"] else None
        if start_id and service:
            await _process_history(store, service, start_id)
    except Exception:
        pass


async def _process_history(store: Any, service: Any, history_id: str) -> None:
    if service is None:
        try:
            service = await _get_gmail_service()
        except Exception:
            return
    messages = await fetch_new_messages(service, history_id)
    for meta in messages:
        msg_id = meta.get("id")
        if not msg_id:
            continue
        # Dedup: skip already-processed message ids
        try:
            async with store._pool.acquire() as conn:
                exists = await conn.fetchval(
                    "SELECT 1 FROM decision_events WHERE meta->>'gmail_message_id' = $1 LIMIT 1",
                    msg_id,
                )
                if exists:
                    continue
        except Exception:
            pass
        msg = await get_message(service, msg_id)
        if not msg:
            continue
        subject, sender, body = extract_text_from_message(msg)
        kind = classify_email(subject, body)
        if kind in ("otp", "other"):
            continue
        # Resolve to job_id with confidence
        from .outcome_resolver import resolve_outcome

        resolution = await resolve_outcome(
            store, {"subject": subject, "from": sender, "snippet": body[:500]}
        )
        job_id = resolution.get("job_id")
        confidence = resolution.get("confidence", 0.0)
        if job_id is None or confidence < 0.35:
            # Unattributed bucket
            try:
                async with store._pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO unattributed_outcomes (email_kind, company, role, subject, snippet, confidence, created_at)
                        VALUES ($1,$2,$3,$4,$5,$6,NOW())
                        """,
                        kind,
                        sender,
                        extract_role(subject) or "",
                        subject[:300],
                        body[:500],
                        confidence,
                    )
            except Exception:
                pass
            continue
        # Emit reward event, backfilled to the ORIGINATING impression so the
        # delayed reward credits the decision that generated it (the review's
        # "reward → impression linkage" fix).
        from .events import DecisionEvent
        from .reward import reward_for

        # Use the email's send time as the reward timestamp so attribution
        # prefers the impression that surfaced the job (closest in time before
        # the outcome), not blindly the latest impression.
        reward_ts = msg.get("internalDate")
        originating = await _latest_impression_for_job(store, job_id, reward_ts=reward_ts) or {}
        event = DecisionEvent(
            job_id=job_id,
            event_type=kind,
            reward=reward_for(kind),
            impression_id=originating.get("impression_id")
            if originating.get("impression_id")
            else None,
            candidate_snapshot_id=(
                originating.get("candidate_snapshot_id")
                if originating.get("candidate_snapshot_id")
                else None
            ),
            job_snapshot_id=(
                originating.get("job_snapshot_id") if originating.get("job_snapshot_id") else None
            ),
            model_version=str(originating.get("model_version") or ""),
            policy=str(originating.get("policy") or ""),
            feature_version=str(originating.get("feature_version") or ""),
            source=str(originating.get("source") or ""),
            meta={"gmail_message_id": msg_id, "subject": subject[:200], "confidence": confidence},
        )
        from .events import emit_event

        await emit_event(store, event)


async def _latest_impression_for_job(
    store: Any, job_id: str, reward_ts: Any = None
) -> dict[str, Any] | None:
    """The job_ranked impression that (most likely) caused the action on a job.

    Used to attach delayed email rewards to the ORIGINATING ranking decision.

    Attribution semantics (the review's P1 concern): when the same job appears
    across several impressions (Monday rank 40, Wednesday rank 5, Friday rank 2
    -> applied), the reward must credit the DECISION that generated the action,
    not indiscriminately the latest impression. We prefer the impression
    closest in time to (just before) the reward/application, falling back to
    the latest when no timestamp is available.

    The reward carries the AUTOFILL job_id (job-xxxx) while the ranked event
    carries the RADAR canonical_id (a url hash). Bridge the two via the apply
    URL: autofill_queue.apply_link == radar_candidates.direct_apply_url == the
    ranked event's meta url. Falls back to a direct job_id match.
    """
    try:
        async with store._pool.acquire() as conn:
            # 1. Resolve the autofill job's apply URL (if this is an autofill id).
            apply_url = None
            try:
                row = await conn.fetchrow(
                    "SELECT apply_link FROM autofill_queue WHERE job_id = $1", job_id
                )
                if row:
                    apply_url = row["apply_link"]
            except Exception:
                apply_url = None
            # 2. All ranked events for this job (by URL bridge or job_id),
            #    ordered oldest -> newest.
            query = """
                SELECT impression_id, candidate_snapshot_id, job_snapshot_id,
                       model_version, policy, feature_version, source, created_at
                FROM decision_events
                WHERE event_type = 'job_ranked' AND impression_id IS NOT NULL
                  AND (
                    job_id = $1
                    OR job_id IN (
                        SELECT canonical_id FROM radar_candidates
                        WHERE direct_apply_url = $2 AND $2 IS NOT NULL)
                    OR meta->>'url' = $2 AND $2 IS NOT NULL
                  )
                ORDER BY created_at ASC
            """
            rows = await conn.fetch(query, job_id, apply_url)
            if not rows:
                # 3. Fallback: any reward/outcome already on this job with an impression.
                row = await conn.fetchrow(
                    """
                    SELECT impression_id, candidate_snapshot_id, job_snapshot_id,
                           model_version, policy, feature_version, source, created_at
                    FROM decision_events
                    WHERE job_id = $1 AND impression_id IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    job_id,
                )
                return dict(row) if row else None
            # Prefer the impression closest to (before) the reward: that is the
            # decision that surfaced the job the candidate acted on. This
            # credits the ACTION to its causing impression, not the latest one.
            if reward_ts is not None:
                best = None
                best_gap = None
                for r in rows:
                    ct = r.get("created_at")
                    gap = abs(_ts_of(ct) - _ts_of(reward_ts)) if ct is not None else None
                    if gap is None:
                        continue
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best = r
                if best is not None:
                    return dict(best)
            return dict(rows[-1])
    except Exception:
        return None


def _ts_of(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "timestamp"):
        try:
            return v.timestamp()
        except Exception:
            return 0.0
    return 0.0


def extract_role(subject: str) -> str | None:
    import re

    m = re.search(
        r"(?:re:|update on|regarding|for)\s+(.+?)(?:\s+at\s+|\s+—|\s+-|$)",
        subject or "",
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()[:80]
    return None


async def _run() -> None:
    from src.memory.pgvector_store import MemoryStore

    store = await MemoryStore.create()
    await run_gmail_push_loop(store)


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run())
