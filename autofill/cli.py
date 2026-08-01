import argparse
import asyncio
import json
import os
import sys
import uuid
from dotenv import load_dotenv

from autofill.profile import build_profile
from autofill.db import AutofillDB
from autofill.worker import run_worker
from autofill.rag import ScreenerRAG, ASK_USER
from autofill.resume import resolve_resume_path
from src.memory.pgvector_store import MemoryStore

load_dotenv()


def self_prompt(question: str) -> str:
    """Blocking prompt for an unknown personal-fact question (runs in a thread)."""
    print(f"\n[Python CLI] No known answer for: {question}")
    val = input("  Enter answer (or press Enter to leave blank): ").strip()
    return val


async def run_apply(url: str, mode: str = "review"):
    """Phase 1 direct execution mode."""
    try:
        store = await MemoryStore.create()
    except Exception as e:
        print(f"[Python CLI] WARNING: Could not connect to memory store; persona retrieval disabled: {e}")
        store = None
    profile = await build_profile(store=store)
    profile.resumePath = await resolve_resume_path()
    if not profile.resumePath:
        print(
            "[Python CLI] WARNING: No resume available to upload (RESUME_URL unreachable / RESUME_PATH not set). "
            "The form will be filled without a resume attachment."
        )
    rag = ScreenerRAG(profile=profile, store=store)
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    payload = {
        "jobId": job_id,
        "url": url,
        "mode": mode,
        "profile": profile.model_dump(by_alias=False),
    }

    payload_json = json.dumps(payload)
    node_dir = os.path.join(os.path.dirname(__file__), "node")

    print(f"[Python CLI] Spawning Stagehand Node runner for {url} (ID: {job_id}, Mode: {mode})...")

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
        sys.exit(1)
    except Exception as e:
        print(f"[Python CLI] Error starting Node runner process: {e}")
        sys.exit(1)

    async def read_stderr():
        while True:
            line = await process.stderr.readline() if process.stderr else b""
            if not line:
                break
            print(f"[Node Log] {line.decode().rstrip()}")

    stderr_task = asyncio.create_task(read_stderr())

    try:
        if process.stdin:
            process.stdin.write((payload_json + "\n").encode("utf-8"))
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
                    print(f"APPLICATION FILLED! Screenshot saved at:")
                    print(f"  {screenshot}")
                    print("========================================================")

                    choice = input("\nReview Form Action -> (S)ubmit application / (K)ip / Cancel [s/k]: ").strip().lower()

                    if choice == "s" and process.stdin:
                        action_payload = json.dumps({"action": "submit"}) + "\n"
                        process.stdin.write(action_payload.encode("utf-8"))
                        await process.stdin.drain()
                    elif process.stdin:
                        action_payload = json.dumps({"action": "skip"}) + "\n"
                        process.stdin.write(action_payload.encode("utf-8"))
                        await process.stdin.drain()
                elif status_event.get("status") in ["submitted", "failed", "skipped"]:
                    break
            elif decoded_line.startswith("RPC_REQUEST:"):
                raw_rpc = decoded_line[len("RPC_REQUEST:") :]
                try:
                    rpc_req = json.loads(raw_rpc)
                    req_id = rpc_req.get("id")
                    method = rpc_req.get("method")
                    args = rpc_req.get("args", {})

                    if method == "answer_questions":
                        questions = args.get("questions", [])
                        answers = await rag.answer_questions(questions)

                        unknown = [q for q, a in answers.items() if a == ASK_USER]
                        for q in unknown:
                            ans = await asyncio.to_thread(self_prompt, q)
                            if ans:
                                await rag.learn(q, ans)
                            answers[q] = ans

                        if process.stdin:
                            rpc_resp = json.dumps({"type": "RPC_RESPONSE", "id": req_id, "result": answers})
                            process.stdin.write(f"{rpc_resp}\n".encode("utf-8"))
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
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass

        await rag.close()


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


def main():
    parser = argparse.ArgumentParser(description="Job Application Autofill CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Direct apply (Phase 1)
    apply_parser = subparsers.add_parser("apply", help="Direct synchronous apply")
    apply_parser.add_argument("url", help="Job posting URL")
    apply_parser.add_argument("--mode", choices=["auto", "review"], default="review")

    # Queue Enqueue
    enqueue_parser = subparsers.add_parser("enqueue", help="Enqueue job into PostgreSQL worker queue")
    enqueue_parser.add_argument("url", help="Job posting URL")
    enqueue_parser.add_argument("--mode", choices=["auto", "review"], default="review")
    enqueue_parser.add_argument("--role", help="Job role title")
    enqueue_parser.add_argument("--company", help="Company name")

    # Approve
    approve_parser = subparsers.add_parser("approve", help="Approve an enqueued job awaiting review")
    approve_parser.add_argument("job_id", help="Job ID to approve")

    # Skip
    skip_parser = subparsers.add_parser("skip", help="Skip an enqueued job awaiting review")
    skip_parser.add_argument("job_id", help="Job ID to skip")

    # Worker
    subparsers.add_parser("worker", help="Start the background worker process")

    args = parser.parse_args()

    if args.command == "apply" or (not args.command and len(sys.argv) > 1 and not sys.argv[1].startswith("-")):
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
    elif args.command == "worker":
        asyncio.run(run_worker())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
