#!/usr/bin/env python3
"""Spawn two llama-server processes: LLM (:8899) and Embeddings (:8900).

The LLM (3.3GB) starts first to secure GPU memory. The embedding model
(639MB) follows and uses limited GPU layers to avoid OOM.
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", os.path.expanduser("~/Models")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LLM_HF = "unsloth/Qwen3-4B-Instruct-2507-GGUF:UD-Q5_K_XL"
EMBED_HF = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"

procs: list[subprocess.Popen] = []


def cleanup() -> None:
    print("\nShutting down llama-server processes...")
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("All servers stopped.")


def _signal_handler(signum: int, frame: object) -> None:
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ── Process 1: LLM (Qwen3.5-4B Q5_K_M, chat completions) ─────────────

print(f"Starting LLM server on :8899 ({LLM_HF})...")
p_llm = subprocess.Popen(
    [
        "llama-server",
        "-hf", LLM_HF,
        "--port", "8899",
        "--ctx-size", "8192",
        "--flash-attn", "on",
        "--parallel", "2",
        "--sleep-idle-seconds", "300",
    ],
)
procs.append(p_llm)

# Wait for LLM to load before starting embeddings (avoids GPU OOM)
print("  waiting for LLM to load...")
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:8899/health", timeout=2)
        print("  LLM ready.")
        break
    except Exception:
        time.sleep(1)

# ── Process 2: Embeddings (Qwen3-Embedding-0.6B Q8_0) ──────────────────

print(f"Starting Embedding server on :8900 ({EMBED_HF})...")
p_embed = subprocess.Popen(
    [
        "llama-server",
        "-hf", EMBED_HF,
        "--port", "8900",
        "--ctx-size", "32768",
        "--embedding",
        "--flash-attn", "on",
        "--parallel", "2",
        "--sleep-idle-seconds", "300",
    ],
)
procs.append(p_embed)

print(f"\nLLM server PID:   {p_llm.pid}  (:8899)")
print(f"Embed server PID: {p_embed.pid}  (:8900)\n")

for p in procs:
    p.wait()
