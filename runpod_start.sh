#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/runpod_setup.sh"

STOP_SCRIPT="$(dirname "$0")/runpod_stop.sh"
trap '"$STOP_SCRIPT"' EXIT

TRAIN_CONFIG="${GRPO_CONFIG:-configs/grpo_runpod.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_runpod.yaml}"

if [ -f "$ACCELERATE_CONFIG" ]; then
  echo "Launching with accelerate ($ACCELERATE_CONFIG)"
  accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    train_grpo.py \
    --config "$TRAIN_CONFIG" \
    --auto-resume \
    "$@"
else
  echo "Launching without accelerate"
  python train_grpo.py \
    --config "$TRAIN_CONFIG" \
    --auto-resume \
    "$@"
fi
