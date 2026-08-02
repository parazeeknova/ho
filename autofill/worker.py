"""Background worker queue processor for autofill service."""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
from typing import Any

from dotenv import load_dotenv

from autofill.db import AutofillDB
from autofill.profile import build_profile
from autofill.rag import ScreenerRAG
from autofill.resolve import (
    DEFER_MARKER,
    DeferredError,
    resolve_cover_letter,
    resolve_question,
)
from autofill.resume import resolve_resume_path
from autofill.telegram import (
    TelegramNotConfiguredError,
    TelegramQuestionBridge,
    TelegramSendError,
)
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore

logger = get_logger("autofill.worker")


def is_overnight() -> bool:
    """True when running in overnight mode (OVERNIGHT_LOOP=true).

    Overnight there is no human present: unknown screener questions defer the
    job for the morning digest instead of blocking on a Telegram prompt, and
    fully-fillable jobs are submitted automatically.
    """
    return os.getenv("OVERNIGHT_LOOP", "").strip().lower() == "true"


def _next_digest_time(summary_time: str) -> _dt.datetime:
    """Next local-time occurrence of the daily digest hour (e.g. "08:00")."""
    try:
        hh, mm = summary_time.strip().split(":")
        target = _dt.time(int(hh), int(mm))
    except Exception:
        logger.warning("Invalid AUTOFILL_DAILY_SUMMARY, using 08:00", value=summary_time)
        target = _dt.time(8, 0)
    now = _dt.datetime.now()
    candidate = _dt.datetime.combine(now.date(), target)
    if candidate <= now:
        candidate += _dt.timedelta(days=1)
    return candidate


class AutofillWorker:
    """Async background worker polling PostgreSQL queue and driving Stagehand processes."""

    def __init__(self, db: AutofillDB, max_concurrent: int = 2) -> None:
        self.db = db
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._running_tasks: set[asyncio.Task] = set()
        self._summary_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the worker polling loop and the daily digest scheduler."""
        self._running = True
        logger.info("AutofillWorker started polling loop...")
        self._summary_task = asyncio.create_task(self._daily_summary_loop())
        try:
            while self._running:
                # The slot is acquired BEFORE claiming and only released when
                # the runner exits, so max_concurrent bounds the number of
                # simultaneous Stagehand processes — not just claims.
                await self.semaphore.acquire()
                job = await self.db.claim_next_job(lease_seconds=3600)
                if not job:
                    self.semaphore.release()
                    await asyncio.sleep(2)
                    continue

                logger.info(
                    "Claimed job for processing", job_id=job["job_id"], link=job["apply_link"]
                )
                task = asyncio.create_task(self._process_job(job))
                self._running_tasks.add(task)
                task.add_done_callback(self._on_task_done)
        except asyncio.CancelledError:
            logger.info("AutofillWorker loop cancelled.")
        finally:
            self.stop()

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Release a processing slot once a job's runner has fully exited."""
        self._running_tasks.discard(task)
        self.semaphore.release()

    def stop(self) -> None:
        """Stop the worker loop and cancel active tasks."""
        self._running = False
        if self._summary_task:
            self._summary_task.cancel()
            self._summary_task = None
        for task in list(self._running_tasks):
            if not task.done():
                task.cancel()
        logger.info("AutofillWorker stopped and active tasks cancelled.")

    # ── morning digest ──────────────────────────────────────────────

    async def _daily_summary_loop(self) -> None:
        """Send the daily morning digest of deferred jobs at AUTOFILL_DAILY_SUMMARY."""
        summary_time = os.getenv("AUTOFILL_DAILY_SUMMARY", "08:00")
        bridge = TelegramQuestionBridge()
        while self._running:
            try:
                next_time = _next_digest_time(summary_time)
                delay = (next_time - _dt.datetime.now()).total_seconds()
                logger.info("Morning digest scheduled", at=str(next_time), in_seconds=round(delay))
                await asyncio.sleep(delay)
                if not self._running:
                    return
                await self._send_daily_digest(bridge)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Morning digest loop error", error=str(e))
                await asyncio.sleep(60)

    async def _send_daily_digest(self, bridge: TelegramQuestionBridge) -> None:
        """Query deferred jobs and send one summary message, if any exist."""
        rows = await self.db.get_pending_summary_jobs()
        if not rows:
            logger.info("Morning digest: no deferred jobs to report")
            return
        if not bridge.is_configured:
            logger.warning("Morning digest skipped: Telegram not configured")
            return
        text = self._format_digest(rows)
        ok = await bridge.send(text)
        if ok:
            await self.db.mark_summary_sent([r["job_id"] for r in rows])
            logger.info("Morning digest sent", count=len(rows))
        else:
            logger.warning("Morning digest send failed; jobs kept for next digest")

    @staticmethod
    def _format_digest(rows: list[dict[str, Any]]) -> str:
        lines = ["<b>⏰ Morning Digest — jobs deferred for your input</b>", ""]
        for i, r in enumerate(rows, 1):
            role = r.get("role") or "Position"
            company = r.get("company") or "Company"
            link = r.get("apply_link") or ""
            questions = r.get("pending_questions") or []
            lines.append(f"<b>{i}. {company}</b> — {role}")
            if link:
                lines.append(f'    <a href="{link}">Open posting →</a>')
            if questions:
                lines.append(f"    Needs input ({len(questions)}):")
                for entry in questions[:6]:
                    if isinstance(entry, str):
                        lines.append(f"    • {entry}")
                    else:
                        lines.append(f"    • {AutofillWorker._format_pending(entry)}")
            lines.append("")
        lines.append("Answer them with <code>python -m autofill resume &lt;job_id&gt;</code>")
        return "\n".join(lines)

    async def _process_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        apply_link = job["apply_link"]
        apply_mode = job.get("apply_mode", "review")

        deferred_pending: list[dict[str, Any]] = []
        rag: ScreenerRAG | None = None
        try:
            try:
                store = await MemoryStore.create()
            except Exception as e:
                logger.warning(
                    "Could not connect to memory store; persona retrieval disabled",
                    error=str(e),
                )
                store = None
            profile = await build_profile(store=store)
            profile.resumePath = await resolve_resume_path()
            rag = ScreenerRAG(profile=profile, store=store)
            bridge = TelegramQuestionBridge()
            question_timeout = float(os.getenv("AUTOFILL_QUESTION_TIMEOUT", "300"))
            overnight = is_overnight()
            run_mode = "auto" if overnight else apply_mode
            logger.info("Processing job", job_id=job_id, mode=run_mode, overnight=overnight)
            job_payload = {
                "jobId": job_id,
                "url": apply_link,
                "mode": run_mode,
                "profile": profile.model_dump(by_alias=False),
                # No-apply phase: the form is filled and verified but never submitted.
                "submitAllowed": False,
            }
            payload_str = json.dumps(job_payload)

            node_dir = os.path.join(os.path.dirname(__file__), "node")

            process_env = {**os.environ}
            if overnight:
                # No human inspects the browser in the overnight worker: hold the
                # review step only briefly so the queue never stalls. In day mode
                # a short review hold (AUTOFILL_REVIEW_HOLD_MS) applies so a
                # human has a moment to review the filled form without stalling
                # the queue on batch runs.
                process_env["AUTOFILL_REVIEW_HOLD_MS"] = "3000"
            else:
                process_env["AUTOFILL_REVIEW_HOLD_MS"] = "180000"

            process = await asyncio.create_subprocess_exec(
                "npx",
                "tsx",
                "runner.ts",
                cwd=node_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
            )

            if process.stdin:
                process.stdin.write(f"{payload_str}\n".encode())
                await process.stdin.drain()

            stderr_task = asyncio.create_task(self._read_stderr(process.stderr, job_id))

            # Job description context extracted by the Node adapter; arrives
            # via the job_context RPC before any question is resolved.
            job_context: dict[str, Any] = {}

            # Monitor stdout for status events and RPC requests
            while True:
                line = await process.stdout.readline() if process.stdout else b""
                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                if line_str.startswith("RPC_REQUEST:"):
                    raw_rpc = line_str[len("RPC_REQUEST:") :]
                    try:
                        rpc_req = json.loads(raw_rpc)
                        req_id = rpc_req.get("id")
                        method = rpc_req.get("method")
                        args = rpc_req.get("args", {})

                        logger.info(
                            "Received RPC request from Node runner",
                            job_id=job_id,
                            method=method,
                            req_id=req_id,
                        )

                        if method == "job_context":
                            # JD extracted by the Node adapter (title, company,
                            # location, description). Stored for the rest of the
                            # run; open-ended answers and scoped questions use it.
                            job_context = {str(k): v for k, v in (args or {}).items()}
                            logger.info(
                                "Received job context from runner",
                                job_id=job_id,
                                title=job_context.get("title"),
                                location=job_context.get("location"),
                            )
                            if process.stdin and not process.stdin.is_closing():
                                rpc_resp = json.dumps(
                                    {"type": "RPC_RESPONSE", "id": req_id, "result": {"ok": True}}
                                )
                                process.stdin.write(f"{rpc_resp}\n".encode())
                                await process.stdin.drain()
                            continue

                        if method == "cover_letter":
                            answer, source = await resolve_cover_letter(
                                rag, job_context=job_context
                            )
                            if process.stdin and not process.stdin.is_closing():
                                rpc_resp = json.dumps(
                                    {
                                        "type": "RPC_RESPONSE",
                                        "id": req_id,
                                        "result": {"answer": answer, "source": source},
                                    }
                                )
                                process.stdin.write(f"{rpc_resp}\n".encode())
                                await process.stdin.drain()
                            continue

                        if method == "answer_question":
                            question = str(args.get("question", "")).strip()
                            kind = str(args.get("kind", "text"))
                            options = [str(o) for o in (args.get("options") or [])]

                            # Never fabricate personal facts: unknown questions
                            # are answered by the user via Telegram (day) or
                            # defer the job for the morning digest (overnight).
                            try:
                                answer, source = await resolve_question(
                                    rag,
                                    bridge,
                                    question,
                                    kind=kind,
                                    options=options,
                                    overnight=overnight,
                                    timeout=question_timeout,
                                    job_context=job_context,
                                )
                            except TelegramNotConfiguredError as tg_err:
                                logger.error(
                                    "Cannot answer question without Telegram",
                                    job_id=job_id,
                                    error=str(tg_err),
                                )
                                if process.stdin and not process.stdin.is_closing():
                                    rpc_resp = json.dumps(
                                        {
                                            "type": "RPC_RESPONSE",
                                            "id": req_id,
                                            "error": str(tg_err),
                                        }
                                    )
                                    process.stdin.write(f"{rpc_resp}\n".encode())
                                    await process.stdin.drain()
                                continue
                            except TelegramSendError as send_err:
                                # A question that could not be delivered is NOT a
                                # user decline: abort loudly via the RPC error
                                # (the screener rethrows real RPC failures)
                                # rather than filling a decline for a prompt the
                                # user never saw.
                                logger.error(
                                    "Telegram question not sent",
                                    job_id=job_id,
                                    error=str(send_err),
                                )
                                if process.stdin and not process.stdin.is_closing():
                                    rpc_resp = json.dumps(
                                        {
                                            "type": "RPC_RESPONSE",
                                            "id": req_id,
                                            "error": str(send_err),
                                        }
                                    )
                                    process.stdin.write(f"{rpc_resp}\n".encode())
                                    await process.stdin.drain()
                                continue
                            except DeferredError as deferred:
                                pending = {
                                    "question": deferred.question,
                                    "kind": deferred.kind,
                                    "options": deferred.options,
                                }
                                # Accumulate: a form can raise several unknown
                                # questions. They are recorded once, together,
                                # after the runner exits (see below), so the
                                # digest lists every question — never only the
                                # last one.
                                deferred_pending.append(pending)
                                if process.stdin and not process.stdin.is_closing():
                                    rpc_resp = json.dumps(
                                        {
                                            "type": "RPC_RESPONSE",
                                            "id": req_id,
                                            "error": DEFER_MARKER,
                                        }
                                    )
                                    process.stdin.write(f"{rpc_resp}\n".encode())
                                    await process.stdin.drain()
                                continue

                            if process.stdin and not process.stdin.is_closing():
                                rpc_resp = json.dumps(
                                    {
                                        "type": "RPC_RESPONSE",
                                        "id": req_id,
                                        "result": {"answer": answer, "source": source},
                                    }
                                )
                                process.stdin.write(f"{rpc_resp}\n".encode())
                                await process.stdin.drain()

                    except Exception as rpc_err:
                        logger.error("Error handling RPC request", error=str(rpc_err))

                elif line_str.startswith("STATUS_EVENT:"):
                    event_raw = line_str[len("STATUS_EVENT:") :]
                    try:
                        event = json.loads(event_raw)
                        status = event.get("status")
                        screenshot_path = event.get("screenshotPath")
                        filled_fields = event.get("filledFields")

                        logger.info("Received IPC status event", job_id=job_id, status=status)

                        if status == "awaiting_review":
                            if deferred_pending:
                                # The runner keeps filling after a deferral and
                                # still reaches the review step. Never clobber
                                # the deferred status: it is terminal for the
                                # claim loop and drives the morning digest.
                                logger.info(
                                    "Job has deferred questions; keeping status deferred",
                                    job_id=job_id,
                                )
                            else:
                                await self.db.update_status(
                                    job_id,
                                    status="awaiting_review",
                                    filled_payload=filled_fields,
                                    screenshot_path=screenshot_path,
                                )
                                if job_payload.get("submitAllowed", True):
                                    # Submission path: wait for a human decision.
                                    decision = await self._wait_for_human_decision(job_id)
                                    if process.stdin and not process.stdin.is_closing():
                                        action_payload = json.dumps({"action": decision})
                                        process.stdin.write(f"{action_payload}\n".encode())
                                        await process.stdin.drain()

                        elif status == "submitted":
                            await self.db.update_status(
                                job_id,
                                status="submitted",
                                filled_payload=filled_fields,
                                screenshot_path=screenshot_path,
                            )
                        elif status == "skipped":
                            if deferred_pending:
                                logger.info(
                                    "Job skipped after deferral; keeping status deferred",
                                    job_id=job_id,
                                )
                            else:
                                await self.db.update_status(
                                    job_id,
                                    status="skipped",
                                    filled_payload=filled_fields,
                                    screenshot_path=screenshot_path,
                                )
                        elif status == "failed":
                            error_msg = event.get("error", "Runner failed")
                            if DEFER_MARKER in error_msg or deferred_pending:
                                # Abort of a deferred job: the row is already
                                # marked deferred; never overwrite with failed.
                                logger.info(
                                    "Runner aborted for deferred job",
                                    job_id=job_id,
                                )
                            else:
                                await self.db.update_status(
                                    job_id, status="failed", error=error_msg
                                )
                                if "CAPTCHA_DETECTED" in error_msg:
                                    # A bot-detection challenge blocked the form:
                                    # the fill could not proceed. Alert the user
                                    # so they know the application needs manual
                                    # attention instead of silently failing.
                                    await self._notify_captcha(
                                        bridge,
                                        job_id,
                                        apply_link,
                                        error_msg,
                                        role=job.get("role"),
                                        company=job.get("company"),
                                    )

                    except json.JSONDecodeError:
                        logger.error("Failed to parse status event JSON", raw=event_raw)

            await process.wait()
            stderr_task.cancel()

            if deferred_pending:
                # Record every unknown question once, together, so the morning
                # digest and resume flow see the full list.
                await self._defer_job(
                    job_id,
                    apply_link,
                    deferred_pending,
                    bridge,
                    role=job.get("role"),
                    company=job.get("company"),
                )
            elif process.returncode != 0:
                # Runner exited abnormally without a status event (startup
                # failure, schema error, browser crash). Fail the job loudly
                # instead of leaving it stuck in 'filling' until the lease.
                await self.db.update_status(
                    job_id,
                    status="failed",
                    error=f"Runner exited with code {process.returncode}",
                )

        except Exception as e:
            logger.exception("Error processing job", job_id=job_id, error=str(e))
            if deferred_pending:
                # Questions were collected before the crash — record them so the
                # digest/resume still sees the full list.
                await self._defer_job(
                    job_id,
                    apply_link,
                    deferred_pending,
                    bridge,
                    role=job.get("role"),
                    company=job.get("company"),
                )
            else:
                await self.db.update_status(job_id, status="failed", error=str(e))
        finally:
            if rag is not None:
                await rag.close()

    async def _defer_job(
        self,
        job_id: str,
        apply_link: str,
        questions: list[dict[str, Any]],
        bridge: TelegramQuestionBridge,
        role: Any = None,
        company: Any = None,
    ) -> None:
        """Mark a job deferred (needs user input) and alert via Telegram."""
        await self.db.mark_deferred(job_id, questions=questions, reason="needs user input")
        if bridge.is_configured:
            role_str = str(role or "Position")
            company_str = str(company or "Company")
            text = (
                f"⛔ <b>Deferred</b>: {company_str} — {role_str}\n"
                f'<a href="{apply_link}">Open posting →</a>\n'
                f"Needs your input ({len(questions)}):\n"
                + "\n".join(self._format_pending(q) for q in questions[:6])
            )
            await bridge.send(text)

    async def _notify_captcha(
        self,
        bridge: TelegramQuestionBridge,
        job_id: str,
        apply_link: str,
        error_msg: str,
        role: Any = None,
        company: Any = None,
    ) -> None:
        """Alert the user that a captcha/challenge blocked the application form.

        The fill was aborted with status failed (error CAPTCHA_DETECTED); unlike
        a deferred question this is terminal — the form cannot be completed by
        automation, so the user needs to know it failed and why.
        """
        if not bridge.is_configured:
            logger.warning(
                "Captcha blocked fill but Telegram not configured",
                job_id=job_id,
                error=error_msg,
            )
            return
        role_str = str(role or "Position")
        company_str = str(company or "Company")
        text = (
            f"🛡️ <b>Captcha blocked</b>: {company_str} — {role_str}\n"
            f'<a href="{apply_link}">Open posting →</a>\n'
            f"The application form is behind a bot-detection challenge "
            f"({error_msg.split(':', 1)[-1].strip()}). Automation could not "
            f"fill it — the job was marked <b>failed</b>.\n"
            f"Submit it manually or retry later."
        )
        ok = await bridge.send(text)
        if ok:
            logger.info("Captcha-blocked notification sent", job_id=job_id)
        else:
            logger.warning("Captcha-blocked notification send failed", job_id=job_id)

    @staticmethod
    def _format_pending(entry: dict[str, Any]) -> str:
        q = entry.get("question") or "?"
        options = entry.get("options") or []
        hint = f"  [{', '.join(str(o) for o in options[:6])}]" if options else ""
        return f"• {q}{hint}"

    async def _wait_for_human_decision(self, job_id: str, poll_interval: float = 2.0) -> str:
        """Poll the database until job status moves to 'approved' or 'skipped'."""
        logger.info("Waiting for human approval/skip decision...", job_id=job_id)
        while self._running:
            job = await self.db.get_job(job_id)
            if not job:
                return "skip"

            status = job.get("status")
            if status == "approved":
                logger.info("Human APPROVED job submission", job_id=job_id)
                return "submit"
            elif status == "skipped":
                logger.info("Human SKIPPED job submission", job_id=job_id)
                return "skip"

            await asyncio.sleep(poll_interval)
        return "skip"

    async def _read_stderr(self, stderr_stream: asyncio.StreamReader | None, job_id: str) -> None:
        if not stderr_stream:
            return
        while True:
            line = await stderr_stream.readline()
            if not line:
                break
            logger.debug(f"[NodeRunner Stderr-{job_id}] {line.decode('utf-8').strip()}")


async def run_worker() -> None:
    """CLI Entrypoint to run the background worker."""
    load_dotenv()
    db = await AutofillDB.create()
    worker = AutofillWorker(db)
    try:
        await worker.start()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
