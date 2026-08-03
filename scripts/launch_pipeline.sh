#!/usr/bin/env bash
# Robust detached launcher for the ho pipeline - survives shell exit.
set -euo pipefail
cd /home/parazeeknova/Projects/ho
export LLM_QUEUE_RPM=240 LLM_QUEUE_MAX_IN_FLIGHT=30 LLM_QUEUE_TPM=400000
export LLM_BUDGET_RADAR_RPM=240 LLM_BUDGET_RADAR_TPM=400000
exec nohup uv run python scripts/run.py > logs/run_pipeline.log 2>&1 < /dev/null &
echo $! > logs/pipeline.pid
