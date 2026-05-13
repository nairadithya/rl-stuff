# GRPO Small LM Environment - Agent Guide

This repository implements Group Relative Policy Optimization (GRPO) for small language models, designed to scale from local debugging to massively parallel hyperparameter sweeps on Modal GPUs.

## Environment & Toolchain
- **Dependency Management:** Use `uv` (e.g., `uv run python script.py`, `uv run streamlit run ui/app.py`).
- **Linting & Formatting:** Use `ruff` (`uv run ruff check .` and `uv run ruff format .`).
- **Healthchecks:** Run `uv run python scripts/healthcheck.py` to verify PyTorch, CUDA, and crucial library bindings (`trl`, `peft`, `transformers`) before complex RL tasks.

## Code Architecture
- **Core RL Logic:** `train_grpo.py` (TRL hook and model initialization), `reward_fns.py` (reinforcement learning reward calculation), `eval_grpo.py` (standardized metric generation), and `grpo_config.py`.
- **Pipeline Orchestration:** Found in `pipeline/` (abstracts matrix configurations, child process management, and markdown/csv reporting). The root scripts `run_prelim.py` and `compare_runs.py` are thin executable wrappers around this module.
- **Visual Analytics:** Streamlit UI in `ui/`. `ui/app.py` is the frontend, and `ui/core.py` handles model inference and loading results.
- **Config Storage:** YAML configs map to pipeline runs. Primary single-run configs sit in `configs/`, while matrix sweep definitions go in `configs/sweeps/`.
- **Archive:** Ignore files in `archive/` (such as legacy Runpod scripts). The active target environment is Modal.

## Key Developer Commands
- **Modal Training (Primary Execution):** `modal run modal_train.py`
  - Folds pipeline configurations (defaults to `configs/sweeps/pipeline_sweep_presweep.yaml`) and dispatches to Modal ephemeral GPUs.
- **Local Smoke Test:** `./scripts/run_smoke.sh`
  - Uses `accelerate launch` to rapidly test training flow locally against `configs/grpo_smoke.yaml`.
- **Local Pipeline Pipeline:** `./scripts/run_prelim.sh`
  - Runs a local sequential pipeline against `configs/pipeline_prelim.yaml`.
- **UI Launch:** `./scripts/run_ui.sh`
  - Starts the interactive Streamlit analytics dashboard.

## Working Constraints & Gotchas
- **Modal Persistent Output:** Modal instances mount a persistent volume at `/workspace/outputs`. Always write evaluation artifacts (like `manifest.json`, `metrics.json`, and model checkpoints) inside `outputs/` or the specific relative path indicated by pipeline configurations to ensure persistence.
- **Avoid Absolute Host Paths:** Code must use relative paths to the repository root (e.g., using `Path(__file__).resolve().parent`) because execution shifts dynamically between local CPU environments and Modal's `/workspace` container directories.
