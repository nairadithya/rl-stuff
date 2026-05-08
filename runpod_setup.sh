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

TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

if python -c "import torch; torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA OK')" 2>/dev/null; then
    echo "PyTorch already installed with CUDA, skipping torch in requirements"
    grep -v '^torch' requirements.txt > /tmp/requirements_notorch.txt || true
    pip install --upgrade pip $PIP_OPTS
    pip install -r /tmp/requirements_notorch.txt $PIP_OPTS
    PIP_EXTRAS=""
elif python -c "import torch; print('torch', torch.__version__, 'installed but CUDA unavailable')" 2>/dev/null; then
    echo "PyTorch installed but CUDA unavailable (driver mismatch). Replacing with cu124 torch..."
    pip install --upgrade pip $PIP_OPTS
    grep -v '^torch' requirements.txt > /tmp/requirements_notorch.txt || true
    pip install -r /tmp/requirements_notorch.txt $PIP_OPTS
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch torchvision --index-url "$TORCH_EXTRA_INDEX" --extra-index-url "https://pypi.org/simple"
    PIP_EXTRAS=""
else
    echo "PyTorch not found. Installing with CUDA support..."
    pip install --upgrade pip $PIP_OPTS
    pip install -r requirements.txt $PIP_OPTS --extra-index-url "$TORCH_EXTRA_INDEX"
    PIP_EXTRAS="--extra-index-url $TORCH_EXTRA_INDEX"
fi
# Install the package itself — no --ignore-installed here to avoid needlessly
# reinstalling already-satisfied deps (especially torch).
pip install -e . $PIP_EXTRAS

# NCCL tuning for multi-GPU cloud instances
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
echo "Detected $GPU_COUNT GPU(s)"
