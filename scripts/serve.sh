#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/Models}"

exec llama-server \
  --models-dir "$MODELS_DIR" \
  --models-max 1 \
  --sleep-idle-seconds 300 \
  -ngl 999 \
  -fa on \
  --port "${PORT:-8899}" \
  -c "${CTX_SIZE:-32768}"
