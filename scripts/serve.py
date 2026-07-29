#!/usr/bin/env python3
"""Spawn two llama-server processes: LLM (:8899) and Embeddings (:8900).

Uses existing GGUF files in MODELS_DIR. Falls back to -hf auto-download
for the embedding model (not gated). The LLM model must be downloaded
manually if the Qwen3.5-4B Instruct repo is gated on HuggingFace.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", os.path.expanduser("~/Models")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LLM_FILE = "Qwen3.5-4B-Instruct-Q4_K_M.gguf"
LLM_FALLBACK = "Qwen3.5-4B.Q5_K_M.gguf"
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


def resolve_llm_model() -> str:
    """Find an available Qwen3.5-4B GGUF file in MODELS_DIR."""
    for candidate in (LLM_FILE, LLM_FALLBACK):
        if (MODELS_DIR / candidate).exists():
            return candidate
    # fallback: try -hf download (may need HF_TOKEN for gated repos)
    print(f"  No local GGUF found in {MODELS_DIR}, trying -hf download...")
    return "Qwen/Qwen3.5-4B-Instruct-GGUF:Q4_K_M"


# ── Process 1: LLM (Qwen3.5-4B, chat completions) ─────────────────────

llm_model = resolve_llm_model()
is_hf = llm_model.startswith("Qwen/")
if is_hf:
    print("Starting LLM server on :8899 (auto-downloading from HuggingFace)...")
    p_llm = subprocess.Popen(
        [
            "llama-server",
            "--models-dir",
            str(MODELS_DIR),
            "-hf",
            llm_model,
            "--port",
            "8899",
            "--ctx-size",
            "8192",
            "--flash-attn",
            "on",
            "--n-gpu-layers",
            "999",
            "--parallel",
            "2",
            "--sleep-idle-seconds",
            "300",
        ],
    )
else:
    print(f"Starting LLM server on :8899 ({llm_model})...")
    p_llm = subprocess.Popen(
        [
            "llama-server",
            "--models-dir",
            str(MODELS_DIR),
            "--model",
            llm_model,
            "--port",
            "8899",
            "--ctx-size",
            "8192",
            "--flash-attn",
            "on",
            "--n-gpu-layers",
            "999",
            "--parallel",
            "2",
            "--sleep-idle-seconds",
            "300",
        ],
    )
procs.append(p_llm)

# ── Process 2: Embeddings (Qwen3-Embedding-0.6B Q8_0) ──────────────────

print("Starting Embedding server on :8900 (auto-downloads Q8_0 if needed)...")
p_embed = subprocess.Popen(
    [
        "llama-server",
        "--models-dir",
        str(MODELS_DIR),
        "-hf",
        EMBED_HF,
        "--port",
        "8900",
        "--ctx-size",
        "32768",
        "--embedding",
        "--flash-attn",
        "on",
        "--n-gpu-layers",
        "999",
        "--parallel",
        "2",
        "--sleep-idle-seconds",
        "300",
    ],
)
procs.append(p_embed)

print(f"\nLLM server PID:   {p_llm.pid}  (:8899)")
print(f"Embed server PID: {p_embed.pid}  (:8900)\n")

for p in procs:
    p.wait()
