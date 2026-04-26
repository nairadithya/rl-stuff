#!/usr/bin/env bash
set -euo pipefail

uv run python run_prelim.py --config configs/pipeline_prelim.yaml "$@"
