"""Retry the stalled R2 backup upload until the network to Cloudflare recovers.

The 1.2GB pg dump upload stalled under degraded network; the small files
(metadata.json, neo4j tarball) already landed in R2. This script runs
detached, attempts `rclone copyto` of the full staging dir every RETRY_EVERY
seconds, and exits once the pg dump is confirmed present in R2.

    setsid nohup env PYTHONPATH=$PWD uv run python3 scripts/retry_r2_upload.py \
        > logs/retry_r2_upload.log 2>&1 &

Staging: backups/staging/20260802-092016/ (mirrors the /tmp staging used by
backup.py but on persistent disk so a reboot cannot lose the dump).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
STAGING = PROJECT / "backups" / "staging" / "20260802-092016"
RETRY_EVERY = 120  # seconds between attempts
CHECK_MARKER = "pg_agent_memory.dump.gz"

R2_ENV = PROJECT / "scripts" / ".r2.env"


def _env_loaded() -> dict[str, str]:
    env = dict(os.environ)
    if R2_ENV.exists():
        for line in R2_ENV.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for key in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        env.setdefault(f"RCLONE_CONFIG_R2_{key[3:] if key.startswith('R2_') else key}", "")
    # rclone reads config from env: build explicit mapping
    env["RCLONE_CONFIG_R2_TYPE"] = "s3"
    env["RCLONE_CONFIG_R2_PROVIDER"] = "Cloudflare"
    env["RCLONE_CONFIG_R2_ENDPOINT"] = env.get("R2_ENDPOINT", "")
    env["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] = env.get("R2_ACCESS_KEY_ID", "")
    env["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"] = env.get("R2_SECRET_ACCESS_KEY", "")
    env["RCLONE_CONFIG_R2_NO_CHECK_BUCKET"] = "true"
    return env


def rclone(*args: str) -> tuple[int, str]:
    env = _env_loaded()
    cmd = ["rclone", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def r2_dest() -> str:
    env = _env_loaded()
    bucket = env.get("R2_BUCKET", "ho-backups")
    return f"r2:{bucket}/daily/20260802-092016"


def upload_complete() -> bool:
    """True if the pg dump is already present in R2 at full size."""
    if not STAGING.exists():
        return True
    src = STAGING / CHECK_MARKER
    if not src.exists():
        return True
    size = src.stat().st_size
    code, out = rclone("lsl", r2_dest())
    if code != 0:
        return False
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == CHECK_MARKER:
            try:
                return int(parts[0]) == size
            except ValueError:
                return False
    return False


def main() -> None:
    if upload_complete():
        print("pg dump already in R2; nothing to do", flush=True)
        return
    attempt = 0
    while True:
        attempt += 1
        ts = time.strftime("%H:%M:%S")
        code, out = rclone(
            "copyto",
            str(STAGING),
            r2_dest(),
            "--s3-no-check-bucket",
            "--transfers",
            "4",
        )
        if code == 0 and upload_complete():
            print(f"[{ts}] upload complete after {attempt} attempt(s)", flush=True)
            return
        print(
            f"[{ts}] attempt {attempt} rc={code} not complete; retrying in {RETRY_EVERY}s "
            f"({out[-120:]})",
            flush=True,
        )
        time.sleep(RETRY_EVERY)


if __name__ == "__main__":
    sys.exit(main())
