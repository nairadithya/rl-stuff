#!/usr/bin/env bash
set -euo pipefail

accelerate launch train_grpo.py --config configs/grpo_smoke.yaml "$@"
