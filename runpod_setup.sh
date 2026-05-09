#!/usr/bin/env bash
set -euo pipefail

PERSISTENT_DIR="${PERSISTENT_DIR:-/workspace}"
export HF_HOME="${HF_HOME:-$PERSISTENT_DIR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PERSISTENT_DIR/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-$PERSISTENT_DIR/torch_cache}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$PERSISTENT_DIR/outputs"

TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_VERSION_SPEC="${TORCH_VERSION_SPEC:->=2.0.0}"

PIP_OPTS="--root-user-action=ignore"

# Pre-install blinker with --ignore-installed to work around RunPod's
# system distutils blinker 1.4 that pip can't uninstall (required by streamlit).
pip install --ignore-installed blinker>=1.9.0 $PIP_OPTS 2>/dev/null || true

TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

install_non_torch_deps() {
    pip install --upgrade pip $PIP_OPTS
    pip install -e ".[dev]" --no-deps $PIP_OPTS
    pip install accelerate datasets peft transformers trl pyyaml math-verify tqdm streamlit $PIP_OPTS
}

if python -c "import torch; torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA OK')" 2>/dev/null; then
    echo "PyTorch already installed with CUDA, skipping torch reinstall"
    install_non_torch_deps
elif python -c "import torch; print('torch', torch.__version__, 'installed but CUDA unavailable')" 2>/dev/null; then
    echo "PyTorch installed but CUDA unavailable (driver mismatch). Replacing with ${TORCH_CUDA} torch..."
    pip install --upgrade pip $PIP_OPTS
    pip install -e ".[dev]" --no-deps $PIP_OPTS
    pip uninstall -y torch torchaudio 2>/dev/null || true
    pip install "torch${TORCH_VERSION_SPEC}" \
      --index-url "$TORCH_EXTRA_INDEX" \
      --extra-index-url "https://pypi.org/simple" \
      $PIP_OPTS
else
    echo "PyTorch not found. Installing with CUDA support..."
    pip install --upgrade pip $PIP_OPTS
    pip install -e ".[dev]" --no-deps $PIP_OPTS
    pip install "torch${TORCH_VERSION_SPEC}" \
      --index-url "$TORCH_EXTRA_INDEX" \
      --extra-index-url "https://pypi.org/simple" \
      $PIP_OPTS
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo ""
echo "=== Environment sanity check ==="
python -c "
import torch, transformers, trl
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'transformers {transformers.__version__}, trl {trl.__version__}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  GPU: none (CUDA not available)')
"
