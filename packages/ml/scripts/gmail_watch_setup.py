"""One-time Gmail Pub/Sub watch setup — gcloud-automated + one Console step.

Usage:
  A. GCP infra (this can ALL be done via gcloud CLI — you're already authed):

     uv run python packages/ml/scripts/gmail_watch_setup.py --gcloud
     # enables Gmail+PubSub APIs, creates topic + subscription, grants Gmail
     # publish rights, writes GCP_PUBSUB_* to .env

  B. OAuth client for Gmail (ONE Console step — gcloud can't create the
     Gmail-scoped "Desktop" client, only openid/email/profile scopes):

     Google Cloud Console → APIs & Services → Credentials →
       Create Credentials → OAuth client ID → Desktop app → download JSON

     Put client_id / client_secret into .env, then:

     uv run python packages/ml/scripts/gmail_watch_setup.py --authorize
     # opens browser for YOUR Gmail account → paste code → GMAIL_REFRESH_TOKEN

  C. Arm the watch:
     uv run python packages/ml/scripts/gmail_watch_setup.py --arm
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Ensure ml package is importable when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


def _sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(f"  ✗ {cmd[0]} failed: {r.stderr.strip()[-300:]}", file=sys.stderr)
    return (r.stdout or "").strip()


def _env_upsert(key: str, value: str) -> None:
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"  ✓ wrote {key} to .env")


def gcloud_setup() -> None:
    """Everything GCP that can be done from the CLI."""
    print("[gcloud] Setting up Gmail push infrastructure...")

    # 0. Project + topic name
    project = _sh(["gcloud", "config", "get-value", "project"])
    if not project:
        print("  ✗ no active gcloud project. Run: gcloud config set project <id>")
        sys.exit(1)
    print(f"  ✓ project: {project}")
    _env_upsert("GCP_PUBSUB_PROJECT", project)

    topic = os.getenv("GCP_PUBSUB_TOPIC", "ho-gmail-events")
    subscription = os.getenv("GCP_PUBSUB_SUBSCRIPTION", "ho-gmail-events-sub")
    _env_upsert("GCP_PUBSUB_TOPIC", topic)
    _env_upsert("GCP_PUBSUB_SUBSCRIPTION", subscription)

    # 1. Enable APIs
    print("[gcloud] enabling Gmail + Pub/Sub APIs...")
    _sh(["gcloud", "services", "enable", "gmail.googleapis.com", "pubsub.googleapis.com"])

    # 2. Topic (idempotent)
    print(f"[gcloud] ensuring topic {topic}...")
    exists = _sh(["gcloud", "pubsub", "topics", "list", "--format=value(name)"])
    if not any(f"topics/{topic}" in t for t in exists.splitlines()):
        _sh(["gcloud", "pubsub", "topics", "create", topic])
    else:
        print("  ✓ topic already exists")

    # 3. Grant Gmail publisher rights on the topic
    print("[gcloud] granting gmail-api-push publisher on topic...")
    _sh(
        [
            "gcloud",
            "projects",
            "add-iam-policy-binding",
            project,
            "--member=serviceAccount:gmail-api-push@system.gserviceaccount.com",
            "--role=roles/pubsub.publisher",
            "--condition=None",
        ]
    )

    # 4. Pull subscription (idempotent)
    print(f"[gcloud] ensuring subscription {subscription}...")
    subs = _sh(["gcloud", "pubsub", "subscriptions", "list", "--format=value(name)"])
    if not any(f"subscriptions/{subscription}" in s for s in subs.splitlines()):
        _sh(
            [
                "gcloud",
                "pubsub",
                "subscriptions",
                "create",
                subscription,
                "--topic=" + topic,
                "--ack-deadline=60",
            ]
        )
    else:
        print("  ✓ subscription already exists")

    print("\n" + "=" * 60)
    print("GCP infra done via gcloud.")
    print("Remaining ONE step (Console — gcloud can't make a Gmail-scoped OAuth client):")
    print("  console.cloud.google.com → APIs & Services → Credentials →")
    print("    Create Credentials → OAuth client ID → Desktop app → download JSON")
    print("  Put client_id + client_secret in .env (GOOGLE_CLIENT_ID/SECRET), then run:")
    print("    uv run python packages/ml/scripts/gmail_watch_setup.py --authorize")
    print("=" * 60)


async def authorize() -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    if not (client_id and client_secret):
        print("Need GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")
        print("(Console → Credentials → OAuth client ID → Desktop app → download JSON)")
        sys.exit(1)
    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )
    creds = flow.run_local_server(port=0)
    print(f"\nGMAIL_REFRESH_TOKEN={creds.refresh_token}\n")
    print("Add this to your .env and set GMAIL_PUSH=1")
    _env_upsert("GMAIL_REFRESH_TOKEN", creds.refresh_token)
    _env_upsert("GMAIL_PUSH", "1")
    print("\nNow arm the watch: uv run python packages/ml/scripts/gmail_watch_setup.py --arm")


async def arm() -> None:
    from ml.src.config import get_ml_config

    cfg = get_ml_config().gmail_push
    if not cfg.enabled:
        print("Set GMAIL_PUSH=1 and GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN")
        sys.exit(1)
    from ml.src.outcomes.gmail_push import _get_gmail_service, arm_watch

    try:
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
    except Exception as e:
        print(f"Could not connect to pgvector store: {e} — watch armed, state not persisted")
        store = None  # type: ignore[assignment]
    service = await _get_gmail_service()
    result = await arm_watch(service, store)
    print(f"Watch armed: historyId={result.get('historyId')} expiry={result.get('expiration')}")
    if store:
        await store.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Gmail Pub/Sub watch setup")
    ap.add_argument(
        "--gcloud", action="store_true", help="GCP infra via gcloud CLI (apis/topic/sub/iam)"
    )
    ap.add_argument("--authorize", action="store_true", help="OAuth flow to mint refresh token")
    ap.add_argument("--arm", action="store_true", help="Arm users.watch push")
    args = ap.parse_args()
    if args.gcloud:
        gcloud_setup()
    elif args.authorize:
        asyncio.run(authorize())
    elif args.arm:
        asyncio.run(arm())
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
