#!/usr/bin/env bash
# Optimized runner for M4 MacBook Air with 16GB RAM
set -euo pipefail

echo "🍎 Running GRPO training optimized for M4 MacBook Air (16GB RAM)"
echo ""
echo "Optimizations enabled:"
echo "  • Metal Performance Shaders (MPS) backend"
echo "  • BF16 precision (hardware accelerated on M4)"
echo "  • Gradient checkpointing for memory efficiency"
echo "  • LoRA fine-tuning (rank=8) to reduce memory usage"
echo "  • Conservative batch sizes for 16GB constraint"
echo "  • Using all 8 CPU cores (4P+4E)"
echo ""

# Run with M4-optimized config
# Use the custom accelerate config for M4 if it exists
if [ -f "configs/accelerate_m4.yaml" ]; then
    echo "Using M4-optimized Accelerate configuration"
    uv run accelerate launch --config_file configs/accelerate_m4.yaml train_grpo.py --config configs/grpo_m4_macbook.yaml "$@"
else
    echo "Using default Accelerate configuration"
    uv run accelerate launch train_grpo.py --config configs/grpo_m4_macbook.yaml "$@"
fi
