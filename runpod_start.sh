#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/runpod_setup.sh"

TRAIN_CONFIG="${GRPO_CONFIG:-configs/grpo_runpod.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_runpod.yaml}"

EXTRA_ARGS=""
if [ "$GPU_COUNT" -gt 1 ]; then
  EXTRA_ARGS="--num_processes=$GPU_COUNT"
fi

# Use accelerate if config exists, otherwise torchrun fallback
if [ -f "$ACCELERATE_CONFIG" ]; then
  echo "Launching with accelerate ($ACCELERATE_CONFIG), $GPU_COUNT process(es)"
  accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    $EXTRA_ARGS \
    train_grpo.py \
    --config "$TRAIN_CONFIG" \
    --auto-resume \
    "$@"
elif [ "$GPU_COUNT" -gt 1 ]; then
  echo "No accelerate config found, falling back to torchrun"
  torchrun --nproc_per_node="$GPU_COUNT" \
    train_grpo.py \
    --config "$TRAIN_CONFIG" \
    --auto-resume \
    "$@"
else
  echo "Launching single-GPU without accelerate"
  python train_grpo.py \
    --config "$TRAIN_CONFIG" \
    --auto-resume \
    "$@"
fi
