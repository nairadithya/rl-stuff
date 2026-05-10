# %% [markdown]
# # Kaggle Sweep for GRPO
# This notebook sets up the environment and runs the ablative sweep.
# It is designed to be executed as a Kaggle script or imported as a notebook using Jupytext.
# Ensure you run this from the root of the cloned repository.

# %% [markdown]
# ## Environment Setup
# First, we configure the necessary cache directories to utilize Kaggle's persistent `/kaggle/working` directory
# so that model weights and datasets aren't re-downloaded unnecessarily across sessions.

# %%
import os
import sys

# Set persistent directories for Kaggle
os.environ["PERSISTENT_DIR"] = "/kaggle/working"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_cache/hub"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/working/hf_datasets"
os.environ["TORCH_HOME"] = "/kaggle/working/torch_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

# Create directories
for d in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"]:
    os.makedirs(os.environ[d], exist_ok=True)
os.makedirs("/kaggle/working/outputs", exist_ok=True)

# %% [markdown]
# ## Dependency Installation
# Install all required packages. We also force PyTorch to install with CUDA 12.8 support.

# %%
!pip install --upgrade pip --root-user-action=ignore
!pip install --ignore-installed "blinker>=1.9.0" --root-user-action=ignore || true
!pip install -e ".[dev]" --no-deps --root-user-action=ignore
!pip install accelerate datasets peft transformers trl pyyaml math-verify tqdm streamlit --root-user-action=ignore
# Ensure Torch has the correct CUDA backend installed
!pip install torch>=2.8.0 --index-url "https://download.pytorch.org/whl/cu128" --extra-index-url "https://pypi.org/simple" --upgrade --root-user-action=ignore

# %% [markdown]
# ## Environment Verification
# Let's verify that Torch can see the GPUs available in the Kaggle environment.

# %%
import torch
import transformers
import trl

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"transformers {transformers.__version__}, trl {trl.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  GPU: none (CUDA not available)")

# %% [markdown]
# ## Configure the Ablative Sweep
# Here we define our hyperparameters for the sweep. You can easily modify these values directly
# in the notebook before running the cell to tune the experiment. We write this out to a YAML file
# for `run_prelim.py` to pick up.

# %%
import yaml

sweep_config = {
    # Models to evaluate and train
    "models": [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "google/gemma-3-1b-it",
    ],
    
    # Dataset configurations
    "dataset_name": "trl-lib/DeepMath-103K",
    "train_split": "train",
    "test_split": "test",
    "train_max_samples": 2000,
    "test_max_samples": 500,
    
    # Training hyperparameters
    "max_steps": 400,
    "learning_rate": 1.0e-6,
    "reward_type": "accuracy_format",
    "max_completion_length": 384,
    "eval_max_new_tokens": 640,
    "temperature": 0.7,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "mask_truncated_completions": True,
    "num_generations": 8,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "beta": 0.01,
    "num_train_epochs": 1.0,
    
    # Sweep configuration (ablation over loss types)
    "loss_types": ["grpo", "dapo"],
    
    # System and I/O
    "output_root": "/kaggle/working/outputs/sweep",
    "train_config": "configs/grpo_runpod.yaml",
    "eval_config": "configs/eval_prelim.yaml",
    "eval_backend": "transformers",
    "run_name_prefix": "sweep",
    "seed": 42,
    "use_accelerate": True,
    
    # Set to True if Kaggle environment has enough resources to run multiple models at once
    "parallel": False, 
}

config_path = "/kaggle/working/kaggle_sweep_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(sweep_config, f)
    
print(f"Sweep config written to {config_path}")

# %% [markdown]
# ## Run the Sweep
# Execute the preliminary run script with the generated configuration. This handles baseline evaluations,
# GRPO training over the different loss types, and tuned evaluations.

# %%
!python run_prelim.py --config /kaggle/working/kaggle_sweep_config.yaml
