#!/usr/bin/env bash
set -euo pipefail

PERSISTENT_DIR="${PERSISTENT_DIR:-/workspace}"
export HF_HOME="${HF_HOME:-$PERSISTENT_DIR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PERSISTENT_DIR/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-$PERSISTENT_DIR/torch_cache}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$PERSISTENT_DIR/outputs"

TORCH_CUDA="${TORCH_CUDA:-cu124}"

PIP_OPTS="--ignore-installed --root-user-action=ignore"

if python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA OK')" 2>/dev/null; then
    echo "PyTorch already installed with CUDA, skipping torch in requirements"
    grep -v '^torch' requirements.txt > /tmp/requirements_notorch.txt || true
    pip install --upgrade pip $PIP_OPTS
    pip install -r /tmp/requirements_notorch.txt $PIP_OPTS
else
    pip install --upgrade pip $PIP_OPTS
    pip install -r requirements.txt $PIP_OPTS \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi
pip install -e . $PIP_OPTS

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SWEEP_CONFIG="${SWEEP_CONFIG:-configs/pipeline_sweep.yaml}"
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
echo "Detected $GPU_COUNT GPU(s)"
echo "Sweep config: $SWEEP_CONFIG"

# run_prelim.py launches train_grpo.py internally via accelerate, so we run it directly
python run_prelim.py --config "$SWEEP_CONFIG" "$@"
