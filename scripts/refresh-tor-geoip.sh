#!/usr/bin/env bash
# Refresh the Tor geoip databases mounted into the torproxy container.
# The stock dperson/torproxy image ships a 2020 geoip whose country data
# misclassifies modern exits, so we mount current ones from Tor's GitLab.
#
# Usage:
#   ./scripts/refresh-tor-geoip.sh          # download + restart torproxy
#   ./scripts/refresh-tor-geoip.sh --quiet  # skip the restart
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"

mkdir -p .tor-geoip
echo ">>> Downloading current Tor geoip databases..."
curl -fsSL -o .tor-geoip/geoip \
  "https://gitlab.torproject.org/tpo/core/tor/-/raw/main/src/config/geoip"
curl -fsSL -o .tor-geoip/geoip6 \
  "https://gitlab.torproject.org/tpo/core/tor/-/raw/main/src/config/geoip6"
echo ">>> geoip: $(stat -c%s .tor-geoip/geoip) bytes, geoip6: $(stat -c%s .tor-geoip/geoip6) bytes"

if [[ "${1:-}" != "--quiet" ]]; then
  echo ">>> Restarting torproxy..."
  docker compose -f docker-compose.yaml up -d torproxy
fi
echo "Done."
