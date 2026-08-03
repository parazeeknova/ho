"""Local crawl worker + ingest, routed through Tor for anti-ban.

Runs the Azure crawl worker in LOCAL mode:
  - scrapes ATS boards, workatastartup, HN, remoteok, arbeitnow, etc.
  - writes JSONL blobs to ./crawler_out/ (no cloud needed)
  - routes all HTTP through the torproxy SOCKS5 (:9050) so the public
    IP isn't exposed to job boards
Then runs the local ingest to pull crawler_out/ into Postgres.

Usage:
    uv run python scripts/cli/local_crawler.py             # crawler + ingest
    uv run python scripts/cli/local_crawler.py --crawl     # crawler only
    uv run python scripts/cli/local_crawler.py --ingest    # ingest only
    uv run python scripts/cli/local_crawler.py --no-tor    # crawl without Tor
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]

_EXTRA_DEPS = ["azure-storage-blob", "httpx[socks]"]


def _uv_run(args: list[str], env: dict[str, str]) -> None:
    cmd = [
        "uv",
        "run",
        "--with",
        "azure-storage-blob",
        "--with",
        "httpx[socks]",
        "python",
        *args,
    ]
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT, env=env, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local crawler + ingest via Tor")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--crawl", action="store_true", help="crawler only (no ingest)")
    group.add_argument("--ingest", action="store_true", help="ingest only (no crawl)")
    group.add_argument("--no-tor", action="store_true", help="crawl without the Tor proxy")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(PROJECT))
    env["CRAWL_OUT"] = env.get("CRAWL_OUT", str(PROJECT / "crawler_out"))
    env["CRAWL_PROXY"] = env.get("CRAWL_PROXY", "socks5://127.0.0.1:9050")
    if args.no_tor:
        env["CRAWL_PROXY"] = ""

    if args.ingest:
        print(f">>> Local ingest only (from {env['CRAWL_OUT']})")
        _uv_run(["scripts/azure/ingest.py"], env)
        return

    if args.crawl:
        print(
            f">>> Local crawler only -> {env['CRAWL_OUT']} (proxy: {env['CRAWL_PROXY'] or 'none'})"
        )
        _uv_run(["scripts/azure/crawl_worker.py"], env)
        return

    print(f">>> Local crawler (proxy: {env['CRAWL_PROXY'] or 'none'}) + ingest")
    crawler = subprocess.Popen(
        [
            "uv",
            "run",
            "--with",
            "azure-storage-blob",
            "--with",
            "httpx[socks]",
            "python",
            "scripts/azure/crawl_worker.py",
        ],
        cwd=PROJECT,
        env=env,
    )

    def _stop(_sig: int, _frame: object) -> None:
        crawler.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        _uv_run(["scripts/azure/ingest.py"], env)
        crawler.wait()
    finally:
        if crawler.poll() is None:
            crawler.terminate()
            try:
                crawler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                crawler.kill()


if __name__ == "__main__":
    main()
