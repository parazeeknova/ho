import argparse
import asyncio
import json
import os
import sys
import uuid
from dotenv import load_dotenv
from autofill.profile import Profile

load_dotenv()

async def run_apply(url: str, mode: str = "review"):
    profile = Profile()
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    payload = {
        "jobId": job_id,
        "url": url,
        "profile": profile.model_dump(),
        "mode": mode
    }

    payload_json = json.dumps(payload)
    node_dir = os.path.join(os.path.dirname(__file__), "node")

    print(f"[Python CLI] Spawning Stagehand Node runner for {url} (ID: {job_id}, Mode: {mode})...")

    try:
        process = await asyncio.create_subprocess_exec(
            "npx", "tsx", "runner.ts",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=node_dir
        )
    except FileNotFoundError:
        print("[Python CLI] Error: Node.js or npx not found. Ensure Node.js 18+ is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"[Python CLI] Error starting Node runner process: {e}")
        sys.exit(1)

    # Read stderr asynchronously
    async def read_stderr():
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            print(f"[Node Log] {line.decode().rstrip()}")

    stderr_task = asyncio.create_task(read_stderr())

    try:
        # Write payload to stdin and flush
        process.stdin.write((payload_json + "\n").encode('utf-8'))
        await process.stdin.drain()

        while True:
            line = await process.stdout.readline()
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
                    
                    # Interactive prompt for review
                    choice = input("\nReview Form Action -> (S)ubmit application / (K)ip / Cancel [s/k]: ").strip().lower()
                    
                    if choice == "s":
                        action_payload = json.dumps({"action": "submit"}) + "\n"
                        process.stdin.write(action_payload.encode('utf-8'))
                        await process.stdin.drain()
                    else:
                        action_payload = json.dumps({"action": "skip"}) + "\n"
                        process.stdin.write(action_payload.encode('utf-8'))
                        await process.stdin.drain()
                elif status_event.get("status") in ["submitted", "failed", "skipped"]:
                    break
            else:
                print(f"[Node Stdout] {decoded_line}")

        await process.wait()
        print(f"\n[Python CLI] Process finished with exit code {process.returncode}")

    finally:
        # Clean up process stdin and background stderr reader task
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

def main():
    parser = argparse.ArgumentParser(description="Job Application Autofill Phase 1 CLI")
    parser.add_argument("url", help="Job posting URL to fill")
    parser.add_argument("--mode", choices=["auto", "review"], default="review", help="Application mode")
    args = parser.parse_args()

    asyncio.run(run_apply(args.url, mode=args.mode))

if __name__ == "__main__":
    main()
