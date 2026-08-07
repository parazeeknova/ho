"""One-time Gmail Pub/Sub watch setup — mints OAuth token and arms push.

Usage:
  1. Set GOOGLE_CLIENT_ID/SECRET in .env, then:
     uv run python -m ml.scripts.gmail_watch_setup --authorize
     # opens browser, copy/paste code → writes GMAIL_REFRESH_TOKEN

  2. Arm the watch:
     uv run python -m ml.scripts.gmail_watch_setup --arm

Requires GCP project with Gmail API + Pub/Sub API enabled, and a topic
`projects/$GCP_PUBSUB_PROJECT/topics/$GCP_PUBSUB_TOPIC` that Gmail can publish to
(service account gmail-api-push@system.gserviceaccount.com needs publish rights).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure ml package is importable when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()


async def authorize() -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    if not (client_id and client_secret):
        print("Need GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")
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


async def arm() -> None:
    from ml.config import get_ml_config

    cfg = get_ml_config().gmail_push
    if not cfg.enabled:
        print("Set GMAIL_PUSH=1 and GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN")
        sys.exit(1)
    # Import here to avoid hard dep when not using Gmail push
    from ml.gmail_push import _get_gmail_service, arm_watch

    # Minimal store shim for arm_watch — uses real pgvector store if available
    try:
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
    except Exception as e:
        print(
            f"Could not connect to pgvector store: {e} — watch will be armed but state not persisted"
        )
        store = None  # type: ignore[assignment]
    service = await _get_gmail_service()
    result = await arm_watch(service, store)
    print(f"Watch armed: historyId={result.get('historyId')} expiry={result.get('expiration')}")
    if store:
        await store.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Gmail Pub/Sub watch setup")
    ap.add_argument("--authorize", action="store_true", help="Run OAuth flow to mint refresh token")
    ap.add_argument("--arm", action="store_true", help="Arm users.watch push")
    args = ap.parse_args()
    if args.authorize:
        asyncio.run(authorize())
    elif args.arm:
        asyncio.run(arm())
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
