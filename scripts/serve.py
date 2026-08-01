#!/usr/bin/env python3
"""Spawn local embedding llama-server process on :8900.

Primary LLM completions are offloaded to GeneralCompute Cloud (gemma-4-31B-it).
The embedding server runs locally on GPU/CPU for pgvector RAG semantic search.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", os.path.expanduser("~/Models")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_HF = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"

procs: list[subprocess.Popen] = []


def cleanup() -> None:
    print("\nShutting down llama-server embedding process...")
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("Embedding server stopped.")


def _signal_handler(signum: int, frame: object) -> None:
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

print(f"Starting Embedding server on :8900 ({EMBED_HF})...")
p_embed = subprocess.Popen(
    [
        "llama-server",
        "-hf",
        EMBED_HF,
        "--port",
        "8900",
        "--ctx-size",
        "8192",
        "--embedding",
        "--flash-attn",
        "on",
        "--parallel",
        "2",
        "--sleep-idle-seconds",
        "300",
    ],
)
procs.append(p_embed)

print(f"Embed server PID: {p_embed.pid}  (:8900)\n")

for p in procs:
    p.wait()
