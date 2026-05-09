# RunPod Deployment Guide

## Hardware

Sweeping two ~1B models (Qwen2.5-0.5B + Gemma-3-1B) with LoRA, bf16, 8 generations of 384 tokens.

| GPU | Spot $/hr | Fits? |
|-----|-----------|-------|
| RTX 4090 (24 GB) | ~$0.34 | Yes, both models comfortable |
| RTX 3090 (24 GB) | ~$0.22 | Yes, slightly slower |
| A4000 (16 GB) | ~$0.20 | Borderline for 1B model — reduce `num_generations` or `max_completion_length` |

A **single** RTX 4090 spot is the sweet spot. Multi-GPU not worth it at this model scale. A full sweep (2 models × 2 loss types × 400 steps) runs ~2-4 hours, about $1-1.50 total.

## Base Template

Use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. Do NOT build a custom Docker image — the base template already has PyTorch with CUDA 12.4, triton, and all the CUDA kernels. Building your own Docker is unnecessary and risks version mismatches.

Settings on the RunPod pod:
- **Template**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Network volume**: Attach to `/workspace` (checkpoints, HF cache, and dataset cache survive spot restarts)
- **Startup command**: First run `./runpod_setup.sh` once, then `./runpod_sweep.sh` (or whatever script you need)

## Spot Instance Resilience

You will be using spot (interruptible) instances. The code handles this:

### SIGTERM handler (`train_grpo.py`)
When RunPod sends SIGTERM (~30s grace period before kill):
1. A global `_interrupted` flag is set
2. `SpotInterruptCallback` (a HuggingFace `TrainerCallback`) checks the flag on each `on_step_end`
3. Sets `control.should_save = True` and `control.should_training_stop = True`
4. Trainer saves a checkpoint at the end of the current step, then exits cleanly
5. On re-spawn, `runpod_sweep.sh` passes `--auto-resume` which scans `output_dir` for `checkpoint-*` dirs and resumes from the latest

### Frequent checkpoints
The `configs/grpo_runpod.yaml` sets `save_steps: 50` and `save_total_limit: 3`, so at most 50 steps of progress are lost on spot interruption.

### Persistent cache
All HuggingFace caches (`HF_HOME`, `HF_DATASETS_CACHE`, `TORCH_HOME`) point to `/workspace` (the network volume) to avoid re-downloading models and datasets on every restart.

## Dependency Install Pitfalls

The base template already has `torch 2.4.1+cu124`. The scripts must NOT try to upgrade it.

### What can go wrong

1. **PyPI has CPU-only torch**. If pip tries to upgrade torch from PyPI's default index, it downloads a CPU-only build. CUDA breaks silently and training runs on CPU (10x slower).

2. **System distutils packages**. The template has `blinker 1.4` installed as a system distutils package. Pip can't uninstall it when upgrading to `blinker>=1.9.0` (required by streamlit), causing a hard failure.

### How the scripts handle it

Dependency installation is extracted into `runpod_setup.sh`. Both `runpod_start.sh` and `runpod_sweep.sh` source it, so you run setup once and the training/sweep scripts skip it.

**Run setup once:**
```bash
./runpod_setup.sh
```

Then run training or sweep without re-installing:
```bash
./runpod_start.sh       # single training run
# or
./runpod_sweep.sh        # full sweep pipeline
```

**What `runpod_setup.sh` does:**

**Step 1 — Detect torch:**
```bash
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null
```
If torch + CUDA works, skip torch reinstall and install other deps from `pyproject.toml`. If not, install from scratch with `--index-url` pointing to the CUDA wheelhouse.

**Step 2 — Install deps:**
```bash
pip install -e ".[dev]" --no-deps
pip install accelerate datasets peft transformers trl ...
```
Dependencies are installed directly from `pyproject.toml` — no `requirements.txt` needed. The `--no-deps` flag on the editable install prevents pip from re-resolving torch against PyPI.

## Config Files

### `configs/grpo_runpod.yaml`
Training config optimized for RunPod:
- `bf16: true` — required for NVIDIA GPU performance
- `save_steps: 50`, `save_total_limit: 3` — checkpoint frequently for spot resilience
- `beta: 0.01` — small KL penalty to prevent reward collapse
- `num_generations: 8` — proper GRPO group size
- `max_completion_length: 384` — sufficient for math reasoning
- `output_dir: /workspace/outputs/...` — network volume for spot survival

### `configs/accelerate_runpod.yaml`
Single-GPU accelerate config (bf16 mixed precision, `distributed_type: NO`).

## Sweep Pipeline

### What it does
The pipeline (`run_prelim.py`) now supports a `loss_types` field for model × loss_type grid sweeps. For each model:

```
baseline_eval  →  for each loss_type: train → tuned_eval → compare
```

Directory structure:
```
outputs/sweep/<experiment_id>/
  manifest.json
  leaderboard.json / leaderboard.csv
  summary.md
  <model_slug>/
    baseline_eval/  (shared across loss types)
    train/groups/
    tuned_eval/groups/
    comparison/groups/
```

### Smoke sweep
Quick validation (5-10 min):
```bash
SWEEP_CONFIG=configs/pipeline_sweep_smoke.yaml ./runpod_sweep.sh
```
- 10 steps, 1 generation, 128-token completions, `length` reward
- 2 models × 2 loss types = 4 tiny training runs
- Validates: torch/CUDA works, HF model download, dataset loads, training completes, eval runs, comparison outputs are written

### Full sweep
The real experiment:
```bash
SWEEP_CONFIG=configs/pipeline_sweep.yaml ./runpod_sweep.sh
```
- 400 steps, 8 generations, 384-token completions, `accuracy_format` reward
- 2 models × 2 loss types = 4 proper training runs

### Override at runtime
```bash
# Override number of steps for a quick test
SWEEP_CONFIG=configs/pipeline_sweep_smoke.yaml ./runpod_sweep.sh --max-steps 5

# Override loss types
./runpod_sweep.sh --loss-types grpo dapo dr_grpo
```

### What gets compared
The leaderboard shows `model | loss_type | baseline_accuracy | tuned_accuracy | delta` across all combinations. This lets you directly compare which loss type works best for which model.

## Tmux

RunPod web terminal supports tmux:

```bash
tmux new -s sweep          # start session
./runpod_setup.sh          # install deps once
./runpod_sweep.sh          # run the command
Ctrl+B, then D             # detach (keeps running)

tmux attach -t sweep       # reattach later
tmux ls                    # list sessions
```

Inside tmux:
- `Ctrl+B` `D` — detach
- `Ctrl+B` `C` — new window
- `Ctrl+B` `N` / `P` — next/previous window

## File Reference

| File | Purpose |
|------|---------|
| `runpod_setup.sh` | One-time dependency install + env setup (sourced by the scripts below) |
| `runpod_start.sh` | Single-training entrypoint for RunPod (runs `train_grpo.py`) |
| `runpod_sweep.sh` | Sweep entrypoint for RunPod (runs `run_prelim.py`) |
| `configs/grpo_runpod.yaml` | Training config for RunPod (bf16, save_steps=50) |
| `configs/accelerate_runpod.yaml` | Single-GPU accelerate config |
| `configs/pipeline_sweep.yaml` | Full sweep pipeline config (400 steps, 2 models, 2 loss types) |
| `configs/pipeline_sweep_smoke.yaml` | Smoke sweep pipeline config (10 steps, quick validation) |
| `train_grpo.py` | Training script with SIGTERM handler, auto-resume, SpotInterruptCallback |
| `grpo_config.py` | TrainingConfig dataclass with `auto_resume` field |

## Quick Checklist

Before your first RunPod run:

- [ ] Pod selected: `runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel`, 1× RTX 4090 spot
- [ ] Network volume attached at `/workspace`
- [ ] Repo cloned into `/workspace` (or uploaded)
- [ ] Run setup once: `./runpod_setup.sh`
- [ ] Run smoke sweep first: `SWEEP_CONFIG=configs/pipeline_sweep_smoke.yaml ./runpod_sweep.sh`
- [ ] Smoke passes → full sweep: `SWEEP_CONFIG=configs/pipeline_sweep.yaml ./runpod_sweep.sh`
- [ ] Started in tmux: `tmux new -s sweep`
