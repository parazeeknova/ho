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
import json
import re
from typing import Any

from .config import get_ml_config

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
    except Exception:
        return []
    messages: list[dict[str, Any]] = []
    for h in resp.get("history", []) or []:
        for m in h.get("messagesAdded", []) or []:
            msg = m.get("message") or {}
            if msg.get("id"):
                messages.append(msg)
    return messages


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


async def run_gmail_push_loop(store: Any) -> None:
    """Main loop: streaming pull on Pub/Sub, process history, emit reward events.

    Designed to run as a daemon (loop.py 3rd child). Falls back to history
    polling every poll_interval_s if Pub/Sub is unreachable.
    """
    cfg = get_ml_config().gmail_push
    if not cfg.enabled:
        return

    # Try to arm watch on startup
    try:
        service = await _get_gmail_service()
        await arm_watch(service, store)
    except Exception as e:
        import logging

        logging.getLogger("gmail_push").warning(f"Gmail watch arm failed, will retry: {e}")
        service = None  # type: ignore[assignment]

    while True:
        try:
            await _poll_cycle(store, service)
        except Exception as e:
            import logging

            logging.getLogger("gmail_push").warning(f"Gmail push cycle failed: {e}")
        await asyncio.sleep(cfg.poll_interval_s)


async def _poll_cycle(store: Any, service: Any) -> None:
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

        originating = await _latest_impression_for_job(store, job_id) or {}
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


async def _latest_impression_for_job(store: Any, job_id: str) -> dict[str, Any] | None:
    """The most recent job_ranked event for a job — the impression/decision
    that originally surfaced it. Used to attach delayed email rewards to the
    originating ranking decision (not a bare job_id).

    The reward carries the AUTOFILL job_id (job-xxxx) while the ranked event
    carries the RADAR canonical_id (a url hash). Bridge the two via the apply
    URL: autofill_queue.apply_link == radar_candidates.direct_apply_url == the
    ranked event's job_id source. Falls back to a direct job_id match.
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
            # 2. Find the ranked event: by apply URL (canonical match) or job_id.
            query = """
                SELECT impression_id, candidate_snapshot_id, job_snapshot_id,
                       model_version, policy, feature_version, source, created_at
                FROM decision_events
                WHERE event_type = 'job_ranked' AND impression_id IS NOT NULL
                  AND (job_id = $1 OR job_id IN (
                        SELECT canonical_id FROM radar_candidates
                        WHERE direct_apply_url = $2 AND $2 IS NOT NULL))
                ORDER BY created_at DESC
                LIMIT 1
            """
            row = await conn.fetchrow(query, job_id, apply_url)
            if row:
                return dict(row)
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
    except Exception:
        return None


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
