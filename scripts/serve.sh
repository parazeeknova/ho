#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/Models}"

cleanup() {
    echo ""
    echo "Shutting down llama-server processes..."
    kill "$LLM_PID" 2>/dev/null || true
    kill "$EMBED_PID" 2>/dev/null || true
    wait "$LLM_PID" 2>/dev/null || true
    wait "$EMBED_PID" 2>/dev/null || true
    echo "All servers stopped."
}
trap cleanup EXIT INT TERM

# ── Process 1: LLM (Qwen3.5-4B, chat completions) ──────────────────────

echo "Starting LLM server on :8899..."
llama-server \
    --models-dir "$MODELS_DIR" \
    --model "Qwen/Qwen3.5-4B-Instruct-GGUF" \
    --port 8899 \
    --ctx-size 8192 \
    --flash-attn on \
    --n-gpu-layers 999 \
    --parallel 2 \
    --sleep-idle-seconds 300 &
LLM_PID=$!

# ── Process 2: Embeddings (Qwen3-Embedding-0.6B) ───────────────────────

echo "Starting Embedding server on :8900..."
llama-server \
    --models-dir "$MODELS_DIR" \
    --model "Qwen/Qwen3-Embedding-0.6B-GGUF" \
    --port 8900 \
    --ctx-size 32768 \
    --embedding \
    --flash-attn on \
    --n-gpu-layers 999 \
    --parallel 2 \
    --sleep-idle-seconds 300 &
EMBED_PID=$!

echo ""
echo "LLM server PID:   $LLM_PID  (:8899)"
echo "Embed server PID: $EMBED_PID  (:8900)"
echo ""

wait "$LLM_PID" "$EMBED_PID"
