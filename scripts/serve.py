#!/usr/bin/env python3
"""Spawn two llama-server processes: LLM (:8899) and Embeddings (:8900).

Models are auto-downloaded from HuggingFace via llama-server's -hf flag.
"""

import os
import signal
import subprocess
import sys

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/Models"))
os.makedirs(MODELS_DIR, exist_ok=True)

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

# ── Process 1: LLM (Qwen3.5-4B Q4_K_M, chat completions) ──────────────

print("Starting LLM server on :8899 (auto-downloads Q4_K_M if needed)...")
p_llm = subprocess.Popen(
    [
        "llama-server",
        "--models-dir", MODELS_DIR,
        "-hf", "Qwen/Qwen3.5-4B-Instruct-GGUF:Q4_K_M",
        "--port", "8899",
        "--ctx-size", "8192",
        "--flash-attn", "on",
        "--n-gpu-layers", "999",
        "--parallel", "2",
        "--sleep-idle-seconds", "300",
    ],
)
procs.append(p_llm)

# ── Process 2: Embeddings (Qwen3-Embedding-0.6B Q8_0) ──────────────────

print("Starting Embedding server on :8900 (auto-downloads Q8_0 if needed)...")
p_embed = subprocess.Popen(
    [
        "llama-server",
        "--models-dir", MODELS_DIR,
        "-hf", "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
        "--port", "8900",
        "--ctx-size", "32768",
        "--embedding",
        "--flash-attn", "on",
        "--n-gpu-layers", "999",
        "--parallel", "2",
        "--sleep-idle-seconds", "300",
    ],
)
procs.append(p_embed)

print(f"\nLLM server PID:   {p_llm.pid}  (:8899)")
print(f"Embed server PID: {p_embed.pid}  (:8900)\n")

for p in procs:
    p.wait()
