#!/usr/bin/env bash
# Local crawl worker + ingest, routed through Tor for anti-ban.
#
# Runs the Azure crawl worker in LOCAL mode:
#   - scrapes ATS boards, workatastartup, HN, remoteok, arbeitnow, etc.
#   - writes JSONL blobs to ./crawler_out/ (no cloud needed)
#   - routes all HTTP through the torproxy SOCKS5 (:9050) so the public
#     IP isn't exposed to job boards
# Then runs the local ingest to pull crawler_out/ into Postgres.
#
# Usage:
#   ./scripts/local_crawler.sh            # crawler + ingest, foreground
#   ./scripts/local_crawler.sh --crawl    # crawler only
#   ./scripts/local_crawler.sh --ingest   # ingest only
#   ./scripts/local_crawler.sh --no-tor   # crawl without the Tor proxy
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="${PYTHONPATH:-$PROJECT}"

export CRAWL_OUT="${CRAWL_OUT:-$PWD/crawler_out}"
export CRAWL_PROXY="${CRAWL_PROXY:-socks5://127.0.0.1:9050}"

if [[ "${1:-}" == "--no-tor" ]]; then
  export CRAWL_PROXY=""
fi

if [[ "${1:-}" == "--ingest" ]]; then
  echo ">>> Local ingest only (from $CRAWL_OUT)"
  exec env PYTHONPATH="$PROJECT" uv run --with azure-storage-blob --with "httpx[socks]" python scripts/azure/ingest.py
fi

if [[ "${1:-}" == "--crawl" ]]; then
  echo ">>> Local crawler only -> $CRAWL_OUT (proxy: ${CRAWL_PROXY:-none})"
  exec env PYTHONPATH="$PROJECT" uv run --with azure-storage-blob --with "httpx[socks]" python scripts/azure/crawl_worker.py
fi

echo ">>> Local crawler (proxy: ${CRAWL_PROXY:-none}) + ingest"
env PYTHONPATH="$PROJECT" uv run --with azure-storage-blob --with "httpx[socks]" python scripts/azure/crawl_worker.py &
CRAWLPID=$!
trap "kill $CRAWLPID 2>/dev/null || true" EXIT
sleep 3
env PYTHONPATH="$PROJECT" uv run --with azure-storage-blob --with "httpx[socks]" python scripts/azure/ingest.py
wait $CRAWLPID
