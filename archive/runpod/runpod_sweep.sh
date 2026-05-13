#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/runpod_setup.sh"

STOP_SCRIPT="$(dirname "$0")/runpod_stop.sh"
trap '"$STOP_SCRIPT"' EXIT

SWEEP_CONFIG="${SWEEP_CONFIG:-configs/pipeline_sweep.yaml}"
echo "Sweep config: $SWEEP_CONFIG"

# run_prelim.py launches train_grpo.py internally via accelerate, so we run it directly
python run_prelim.py --config "$SWEEP_CONFIG" "$@"
