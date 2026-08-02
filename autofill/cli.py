import argparse
import asyncio
import contextlib
import json
import os
import sys
import uuid
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
from autofill.worker import is_overnight, run_worker
from src.memory.pgvector_store import MemoryStore

load_dotenv()


async def _stream_runner(
    payload: dict[str, Any],
    rag: ScreenerRAG,
    bridge: TelegramQuestionBridge,
    question_timeout: float,
    overnight: bool,
    job_id: str,
) -> dict[str, Any] | None:
    """Spawn the Node runner, stream status events, handle RPC answers and
    review decisions. Returns the final status event (submitted/failed/skipped).
    """
    node_dir = os.path.join(os.path.dirname(__file__), "node")
    print(
        f"[Python CLI] Spawning Stagehand Node runner for {payload['url']} "
        f"(ID: {job_id}, Mode: {payload['mode']})..."
    )

    try:
        process = await asyncio.create_subprocess_exec(
            "npx",
            "tsx",
            "runner.ts",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=node_dir,
        )
    except FileNotFoundError:
        print("[Python CLI] Error: Node.js or npx not found. Ensure Node.js 18+ is installed.")
        return None
    except Exception as e:
        print(f"[Python CLI] Error starting Node runner process: {e}")
        return None

    async def read_stderr():
        while True:
            line = await process.stderr.readline() if process.stderr else b""
            if not line:
                break
            print(f"[Node Log] {line.decode().rstrip()}")

    stderr_task = asyncio.create_task(read_stderr())
    final_event: dict[str, Any] | None = None

    # Job description context extracted by the Node adapter; arrives via the
    # job_context RPC before any question is resolved.
    job_context: dict[str, Any] = {}

    try:
        if process.stdin:
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()

        while True:
            line = await process.stdout.readline() if process.stdout else b""
            if not line:
                break
            decoded_line = line.decode().rstrip()
            if decoded_line.startswith("STATUS_EVENT:"):
                raw_event = decoded_line.replace("STATUS_EVENT:", "")
                status_event = json.loads(raw_event)
                print(f"\n[Python CLI] Received Status Event: {json.dumps(status_event, indent=2)}")

                if status_event.get("status") == "awaiting_review":
                    screenshot = status_event.get("screenshotPath")
                    print("\n========================================================")
                    print("APPLICATION FILLED! Screenshot saved at:")
                    print(f"  {screenshot}")
                    print("========================================================")

                    if not payload.get("submitAllowed", True):
                        # No-apply phase: the runner closes without prompting.
                        print(
                            "[Python CLI] Submission is disabled in this phase — "
                            "the form was filled and verified, nothing was applied."
                        )
                        print(
                            "Review the filled answers in the browser, "
                            "then press Enter to close it."
                        )
                        with contextlib.suppress(EOFError):
                            input("[Python CLI] Press Enter to close: ")
                        if process.stdin:
                            action_payload = json.dumps({"action": "skip"}) + "\n"
                            process.stdin.write(action_payload.encode())
                            await process.stdin.drain()
                        continue

                    choice = (
                        input(
                            "\nReview Form Action -> (S)ubmit application / (K)ip / Cancel [s/k]: "
                        )
                        .strip()
                        .lower()
                    )

                    if choice == "s" and process.stdin:
                        action_payload = json.dumps({"action": "submit"}) + "\n"
                        process.stdin.write(action_payload.encode())
                        await process.stdin.drain()
                    elif process.stdin:
                        action_payload = json.dumps({"action": "skip"}) + "\n"
                        process.stdin.write(action_payload.encode())
                        await process.stdin.drain()
                elif status_event.get("status") in ["submitted", "failed", "skipped"]:
                    final_event = status_event
                    break
            elif decoded_line.startswith("RPC_REQUEST:"):
                raw_rpc = decoded_line[len("RPC_REQUEST:") :]
                try:
                    rpc_req = json.loads(raw_rpc)
                    req_id = rpc_req.get("id")
                    method = rpc_req.get("method")
                    args = rpc_req.get("args", {})

                    if method == "job_context":
                        job_context = {str(k): v for k, v in (args or {}).items()}
                        print(
                            f"\n[Python CLI] Job context: "
                            f"{job_context.get('title')} @ {job_context.get('company')} "
                            f"({job_context.get('location')})"
                        )
                        if process.stdin:
                            rpc_resp = json.dumps(
                                {"type": "RPC_RESPONSE", "id": req_id, "result": {"ok": True}}
                            )
                            process.stdin.write(f"{rpc_resp}\n".encode())
                            await process.stdin.drain()
                        continue

                    if method == "cover_letter":
                        answer, source = await resolve_cover_letter(rag, job_context=job_context)
                        if process.stdin:
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
                            print(f"\n[Python CLI] ERROR: {tg_err}")
                            if process.stdin:
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
                            print(f"\n[Python CLI] ERROR: {send_err}")
                            if process.stdin:
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
                            # Overnight: no human present — record for the digest.
                            pending = {
                                "question": deferred.question,
                                "kind": deferred.kind,
                                "options": deferred.options,
                            }
                            print(
                                f"\n[Python CLI] DEFERRED ({len(pending['options'] or [])} "
                                f"options): {deferred.question}"
                            )
                            try:
                                db = await AutofillDB.create()
                                try:
                                    deferred_id = await db.enqueue_job(
                                        apply_link=payload["url"], apply_mode="review"
                                    )
                                    await db.mark_deferred(
                                        deferred_id,
                                        questions=[pending],
                                        reason="needs user input",
                                    )
                                    print(
                                        f"[Python CLI] Recorded as {deferred_id} — "
                                        "listed in the morning digest; resume with "
                                        f"`python -m autofill resume {deferred_id}`"
                                    )
                                    if bridge.is_configured:
                                        option_hint = ""
                                        if pending["options"]:
                                            option_hint = "\n" + " | ".join(pending["options"][:6])
                                        text = (
                                            f"⛔ <b>Deferred</b>: "
                                            f"{payload.get('company', 'Company')} — "
                                            f"{payload.get('role', 'Position')}\n"
                                            f'<a href="{payload["url"]}">Open posting →</a>\n'
                                            f"Needs your input:\n"
                                            f"• {deferred.question}{option_hint}"
                                        )
                                        await bridge.send(text)
                                finally:
                                    await db.close()
                            except Exception as db_err:
                                print(f"[Python CLI] WARNING: could not record deferral: {db_err}")

                            if process.stdin:
                                rpc_resp = json.dumps(
                                    {"type": "RPC_RESPONSE", "id": req_id, "error": DEFER_MARKER}
                                )
                                process.stdin.write(f"{rpc_resp}\n".encode())
                                await process.stdin.drain()
                            continue

                        if process.stdin:
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
                    print(f"[Python CLI] Error handling RPC request: {rpc_err}")
            else:
                print(f"[Node Stdout] {decoded_line}")

        await process.wait()
        print(f"\n[Python CLI] Process finished with exit code {process.returncode}")

    finally:
        if process.stdin:
            try:
                process.stdin.close()
                await process.stdin.wait_closed()
            except Exception:
                pass

        stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stderr_task

    return final_event


async def run_apply(url: str, mode: str = "review"):
    """Phase 1 direct execution mode."""
    try:
        store = await MemoryStore.create()
    except Exception as e:
        print(
            f"[Python CLI] WARNING: Could not connect to memory store; "
            f"persona retrieval disabled: {e}"
        )
        store = None
    profile = await build_profile(store=store)
    profile.resumePath = await resolve_resume_path()
    if not profile.resumePath:
        print(
            "[Python CLI] WARNING: No resume available to upload "
            "(RESUME_URL unreachable / RESUME_PATH not set). "
            "The form will be filled without a resume attachment."
        )
    rag = ScreenerRAG(profile=profile, store=store)
    bridge = TelegramQuestionBridge()
    question_timeout = float(os.getenv("AUTOFILL_QUESTION_TIMEOUT", "300"))
    overnight = is_overnight()
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    if overnight:
        mode = "auto"
        print("[Python CLI] OVERNIGHT_LOOP=true: running in auto-submit mode")

    payload = {
        "jobId": job_id,
        "url": url,
        "mode": mode,
        "profile": profile.model_dump(by_alias=False),
        # Overnight (no human): fully-fillable jobs are submitted automatically.
        # In day mode the form is filled and verified but never submitted.
        "submitAllowed": overnight,
    }

    try:
        await _stream_runner(payload, rag, bridge, question_timeout, overnight, job_id)
    finally:
        await rag.close()


async def run_resume(job_id: str, review: bool = False):
    """Answer a deferred job's pending questions (Telegram, options included),
    learn the answers, and re-run the fill. Never submits in this phase."""
    db = await AutofillDB.create()
    try:
        job = await db.get_job(job_id)
        if not job:
            print(f"Job {job_id} not found.")
            return
        if job.get("status") != "deferred":
            print(f"Job {job_id} is not deferred (status={job.get('status')}); nothing to resume.")
            return

        pending = job.get("pending_questions") or []
        if not pending:
            print(f"Job {job_id} has no recorded pending questions; re-running fill...")

        store = await MemoryStore.create()
        profile = await build_profile(store=store)
        profile.resumePath = await resolve_resume_path()
        rag = ScreenerRAG(profile=profile, store=store)
        bridge = TelegramQuestionBridge()
        question_timeout = float(os.getenv("AUTOFILL_QUESTION_TIMEOUT", "300"))

        try:
            answered = 0
            for entry in pending:
                if isinstance(entry, str):
                    q = entry
                    kind = "text"
                    options: list[str] = []
                else:
                    q = str(entry.get("question") or "")
                    kind = str(entry.get("kind") or "text")
                    options = [str(o) for o in (entry.get("options") or [])]
                if not q:
                    continue
                print(f"\n[Python CLI] Question: {q}")
                try:
                    if kind in ("select", "multi") and options:
                        ans = await bridge.ask_options(q, options, timeout=question_timeout)
                    else:
                        ans = await bridge.ask(q, timeout=question_timeout)
                except TelegramNotConfiguredError as tg_err:
                    print(f"[Python CLI] ERROR: {tg_err}")
                    return
                if ans and ans.strip():
                    await rag.learn(q, ans.strip())
                    answered += 1

            if answered < len(pending):
                print(
                    f"\n[Python CLI] WARNING: {len(pending) - answered} question(s) left "
                    "unanswered; those fields will be blank."
                )
                review = True

            mode = "review" if review else "auto"
            payload = {
                "jobId": job_id,
                "url": job["apply_link"],
                "mode": mode,
                "profile": profile.model_dump(by_alias=False),
                "submitAllowed": False,
            }

            final_event = await _stream_runner(
                payload, rag, bridge, question_timeout, False, job_id
            )

            if final_event is None:
                print("[Python CLI] Runner did not report a final status.")
            elif final_event.get("status") == "skipped":
                await db.update_status(job_id, status="skipped", override_terminal=True)
                await db.clear_pending_questions(job_id)
                print("[Python CLI] Fill completed (not submitted). Pending questions cleared.")
            elif final_event.get("status") == "failed":
                error_msg = final_event.get("error", "Runner failed")
                if DEFER_MARKER in error_msg:
                    await db.mark_deferred(job_id, questions=pending)
                    print(f"[Python CLI] Job {job_id} still deferred (questions unanswered).")
                else:
                    await db.update_status(
                        job_id, status="failed", error=error_msg, override_terminal=True
                    )
                    print(f"[Python CLI] Job {job_id} failed: {error_msg}")
        finally:
            await rag.close()
    finally:
        await db.close()


async def enqueue_command(url: str, mode: str = "review", role: str = None, company: str = None):
    db = await AutofillDB.create()
    try:
        job_id = await db.enqueue_job(apply_link=url, role=role, company=company, apply_mode=mode)
        print(f"Successfully enqueued job {job_id} in mode '{mode}'.")
    finally:
        await db.close()


async def approve_command(job_id: str):
    db = await AutofillDB.create()
    try:
        updated = await db.update_status(job_id, status="approved")
        if updated:
            print(f"Job {job_id} marked as APPROVED for submission.")
        else:
            print(f"Failed to update job {job_id}. Verify the job_id exists.")
    finally:
        await db.close()


async def skip_command(job_id: str):
    db = await AutofillDB.create()
    try:
        updated = await db.update_status(job_id, status="skipped")
        if updated:
            print(f"Job {job_id} marked as SKIPPED.")
        else:
            print(f"Failed to update job {job_id}. Verify the job_id exists.")
    finally:
        await db.close()


async def deferred_command():
    db = await AutofillDB.create()
    try:
        rows = await db.get_deferred_jobs()
        if not rows:
            print("No deferred jobs.")
            return
        print(f"{len(rows)} deferred job(s):\n")
        for r in rows:
            company = r.get("company") or "?"
            role = r.get("role") or "?"
            print(f"{r['job_id']}  {company} — {role}")
            print(f"  {r['apply_link']}")
            for entry in r.get("pending_questions") or []:
                if isinstance(entry, str):
                    print(f"    ? {entry}")
                    continue
                q = entry.get("question") or "?"
                options = entry.get("options") or []
                hint = f"  [{', '.join(options[:6])}]" if options else ""
                print(f"    ? {q}{hint}")
            print()
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Job Application Autofill CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Direct apply (Phase 1)
    apply_parser = subparsers.add_parser("apply", help="Direct synchronous apply")
    apply_parser.add_argument("url", help="Job posting URL")
    apply_parser.add_argument("--mode", choices=["auto", "review"], default="review")

    # Queue Enqueue
    enqueue_parser = subparsers.add_parser(
        "enqueue", help="Enqueue job into PostgreSQL worker queue"
    )
    enqueue_parser.add_argument("url", help="Job posting URL")
    enqueue_parser.add_argument("--mode", choices=["auto", "review"], default="review")
    enqueue_parser.add_argument("--role", help="Job role title")
    enqueue_parser.add_argument("--company", help="Company name")

    # Approve
    approve_parser = subparsers.add_parser(
        "approve", help="Approve an enqueued job awaiting review"
    )
    approve_parser.add_argument("job_id", help="Job ID to approve")

    # Skip
    skip_parser = subparsers.add_parser("skip", help="Skip an enqueued job awaiting review")
    skip_parser.add_argument("job_id", help="Job ID to skip")

    # Resume a deferred job
    resume_parser = subparsers.add_parser(
        "resume", help="Answer a deferred job's pending questions and re-run the fill"
    )
    resume_parser.add_argument("job_id", help="Job ID to resume")
    resume_parser.add_argument(
        "--review", action="store_true", help="Pause at awaiting_review instead of auto-submitting"
    )

    # List deferred jobs
    subparsers.add_parser("deferred", help="List jobs deferred for user input")

    # Worker
    subparsers.add_parser("worker", help="Start the background worker process")

    args = parser.parse_args()

    if args.command == "apply" or (
        not args.command and len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    ):
        # Default fallback for backwards compatibility if positional url passed without subcommand
        url = args.url if hasattr(args, "url") else sys.argv[1]
        mode = getattr(args, "mode", "review")
        asyncio.run(run_apply(url, mode=mode))
    elif args.command == "enqueue":
        asyncio.run(enqueue_command(args.url, mode=args.mode, role=args.role, company=args.company))
    elif args.command == "approve":
        asyncio.run(approve_command(args.job_id))
    elif args.command == "skip":
        asyncio.run(skip_command(args.job_id))
    elif args.command == "resume":
        asyncio.run(run_resume(args.job_id, review=args.review))
    elif args.command == "deferred":
        asyncio.run(deferred_command())
    elif args.command == "worker":
        asyncio.run(run_worker())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
