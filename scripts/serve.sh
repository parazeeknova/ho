#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/Models}"

exec llama-server \
  --models-dir "$MODELS_DIR" \
  --models-max 1 \
  --sleep-idle-seconds 300 \
  -ngl 999 \
  -fa on \
  --port "${LLAMA_PORT:-8899}" \
  -c "${CTX_SIZE:-16384}" \
  --parallel "${LLAMA_PARALLEL:-2}"
