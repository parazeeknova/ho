"""Background worker queue processor for autofill service."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional, Set

from src.logging import get_logger
from autofill.db import AutofillDB
from autofill.profile import build_profile
from autofill.rag import ScreenerRAG, ASK_USER
from autofill.resume import resolve_resume_path
from src.memory.pgvector_store import MemoryStore

logger = get_logger("autofill.worker")


class AutofillWorker:
    """Async background worker polling PostgreSQL queue and driving Stagehand processes."""

    def __init__(self, db: AutofillDB, max_concurrent: int = 2) -> None:
        self.db = db
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._running_tasks: Set[asyncio.Task] = set()

    async def start(self) -> None:
        """Start the worker polling loop."""
        self._running = True
        logger.info("AutofillWorker started polling loop...")
        try:
            while self._running:
                async with self.semaphore:
                    job = await self.db.claim_next_job(lease_seconds=600)
                    if not job:
                        await asyncio.sleep(2)
                        continue

                    logger.info("Claimed job for processing", job_id=job["job_id"], link=job["apply_link"])
                    task = asyncio.create_task(self._process_job(job))
                    self._running_tasks.add(task)
                    task.add_done_callback(self._running_tasks.discard)
        except asyncio.CancelledError:
            logger.info("AutofillWorker loop cancelled.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the worker loop and cancel active tasks."""
        self._running = False
        for task in list(self._running_tasks):
            if not task.done():
                task.cancel()
        logger.info("AutofillWorker stopped and active tasks cancelled.")

    async def _process_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        apply_link = job["apply_link"]
        apply_mode = job.get("apply_mode", "review")

        try:
            store = await MemoryStore.create()
        except Exception as e:
            logger.warning("Could not connect to memory store; persona retrieval disabled", error=str(e))
            store = None
        profile = await build_profile(store=store)
        profile.resumePath = await resolve_resume_path()
        rag = ScreenerRAG(profile=profile, store=store)
        job_payload = {
            "jobId": job_id,
            "url": apply_link,
            "mode": apply_mode,
            "profile": profile.model_dump(by_alias=False),
        }
        payload_str = json.dumps(job_payload)

        node_dir = os.path.join(os.path.dirname(__file__), "node")

        try:
            process = await asyncio.create_subprocess_exec(
                "npx",
                "tsx",
                "runner.ts",
                cwd=node_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            if process.stdin:
                process.stdin.write(f"{payload_str}\n".encode("utf-8"))
                await process.stdin.drain()

            stderr_task = asyncio.create_task(self._read_stderr(process.stderr, job_id))

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

                        logger.info("Received RPC request from Node runner", job_id=job_id, method=method, req_id=req_id)

                        if method == "answer_questions":
                            questions = args.get("questions", [])
                            answers = await rag.answer_questions(questions)

                            # Never fabricate personal facts in the background worker:
                            # drop them so the field is left blank for human review.
                            unanswered = [q for q, a in answers.items() if a == ASK_USER]
                            if unanswered:
                                logger.warning(
                                    "Leaving personal questions blank for human review",
                                    job_id=job_id,
                                    questions=unanswered,
                                )
                            answers = {q: a for q, a in answers.items() if a != ASK_USER}

                            if process.stdin and not process.stdin.is_closing():
                                rpc_resp = json.dumps({"type": "RPC_RESPONSE", "id": req_id, "result": answers})
                                process.stdin.write(f"{rpc_resp}\n".encode("utf-8"))
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
                            await self.db.update_status(
                                job_id,
                                status="awaiting_review",
                                filled_payload=filled_fields,
                                screenshot_path=screenshot_path,
                            )
                            decision = await self._wait_for_human_decision(job_id)

                            if process.stdin and not process.stdin.is_closing():
                                action_payload = json.dumps({"action": decision})
                                process.stdin.write(f"{action_payload}\n".encode("utf-8"))
                                await process.stdin.drain()

                        elif status == "submitted":
                            await self.db.update_status(
                                job_id,
                                status="submitted",
                                filled_payload=filled_fields,
                                screenshot_path=screenshot_path,
                            )
                        elif status == "failed":
                            error_msg = event.get("error", "Runner failed")
                            await self.db.update_status(job_id, status="failed", error=error_msg)

                    except json.JSONDecodeError:
                        logger.error("Failed to parse status event JSON", raw=event_raw)

            await process.wait()
            stderr_task.cancel()

        except Exception as e:
            logger.exception("Error processing job", job_id=job_id, error=str(e))
            await self.db.update_status(job_id, status="failed", error=str(e))
        finally:
            await rag.close()

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

    async def _read_stderr(self, stderr_stream: Optional[asyncio.StreamReader], job_id: str) -> None:
        if not stderr_stream:
            return
        while True:
            line = await stderr_stream.readline()
            if not line:
                break
            logger.debug(f"[NodeRunner Stderr-{job_id}] {line.decode('utf-8').strip()}")


async def run_worker() -> None:
    """CLI Entrypoint to run the background worker."""
    db = await AutofillDB.create()
    worker = AutofillWorker(db)
    try:
        await worker.start()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
