#!/usr/bin/env bash
set -euo pipefail

PERSISTENT_DIR="${PERSISTENT_DIR:-/workspace}"
export HF_HOME="${HF_HOME:-$PERSISTENT_DIR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PERSISTENT_DIR/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-$PERSISTENT_DIR/torch_cache}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$PERSISTENT_DIR/outputs"

TORCH_CUDA="${TORCH_CUDA:-cu124}"

if python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA OK')" 2>/dev/null; then
    echo "PyTorch already installed with CUDA, skipping torch in requirements"
    grep -v '^torch' requirements.txt > /tmp/requirements_notorch.txt || true
    pip install --upgrade pip
    pip install -r /tmp/requirements_notorch.txt
else
    pip install --upgrade pip
    pip install -r requirements.txt \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi
pip install -e .

# NCCL tuning for multi-GPU cloud instances
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TRAIN_CONFIG="${GRPO_CONFIG:-configs/grpo_runpod.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_runpod.yaml}"

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
echo "Detected $GPU_COUNT GPU(s)"

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
