# GRPO Small-LM Training Environment

Minimal scaffold to train small language models with **Group Relative Policy Optimization (GRPO)** using Hugging Face TRL.

## What this includes

- `train_grpo.py`: CLI entrypoint for GRPO training
- `eval_grpo.py`: deterministic post-training evaluation on held-out data
- `run_prelim.py`: end-to-end baseline -> GRPO -> tuned evaluation pipeline
- `compare_runs.py`: result comparison helper producing JSON/Markdown/CSV
- `grpo_config.py`: typed defaults for training parameters
- `reward_fns.py`: custom reward functions (`format_reward`, `length_reward`)
- `configs/grpo_small.yaml`: practical baseline config
- `configs/grpo_smoke.yaml`: tiny smoke-test config
- `configs/eval_prelim.yaml`: evaluation defaults for preliminary results
- `configs/pipeline_prelim.yaml`: full post-training pipeline defaults
- `scripts/run_smoke.sh`: helper to run smoke training
- `scripts/run_prelim.sh`: helper to run the full preliminary pipeline
- `ui/model_testing_ui.py`: Streamlit UI to inspect results and prompt models
- `scripts/run_ui.sh`: helper to launch the Streamlit UI

## Requirements

- Python `>=3.10`
- CUDA-capable GPU recommended
- `accelerate` configured (`accelerate config`)

Install dependencies:

```bash
pip install -e .
```

## Quick start

### One-command preliminary results (recommended)

Run baseline eval on `test`, GRPO train on `train`, and tuned eval + comparison artifacts:

```bash
./scripts/run_prelim.sh
```

Equivalent command:

```bash
uv run python run_prelim.py --config configs/pipeline_prelim.yaml
```

This writes a timestamped directory under `outputs/prelim/` with:

- `manifest.json`: full run metadata and artifact paths
- `leaderboard.json` + `leaderboard.csv`: base vs tuned metrics per model
- `summary.md`: quick comparison table
- Per-model subdirectories containing:
  - `baseline_eval/metrics.json` and `predictions.jsonl`
  - `train/` (GRPO adapter/checkpoints)
  - `tuned_eval/metrics.json` and `predictions.jsonl`
  - `comparison/comparison.json`, `comparison/summary.md`, `comparison/leaderboard.csv`

To run multiple base models in one sweep:

```bash
uv run python run_prelim.py \
  --config configs/pipeline_prelim.yaml \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
```

To compare already-trained adapters without re-training:

```bash
uv run python run_prelim.py \
  --config configs/pipeline_prelim.yaml \
  --skip-training \
  --models Qwen/Qwen2.5-0.5B-Instruct \
  --tuned-model-paths outputs/grpo-qwen2.5-0.5b
```

If answers look cut off, increase eval decode length independently of training:

```bash
uv run python run_prelim.py \
  --config configs/pipeline_prelim.yaml \
  --eval-max-new-tokens 640
```

`max_completion_length` controls training rollouts, while `eval_max_new_tokens` controls held-out decoding.

### Standard GPU training

Smoke test (recommended first run):

```bash
accelerate launch train_grpo.py --config configs/grpo_smoke.yaml
```

Baseline run:

```bash
accelerate launch train_grpo.py --config configs/grpo_small.yaml
```

Or use the helper script:

```bash
./scripts/run_smoke.sh
```

## CLI usage

See all options:

```bash
accelerate launch train_grpo.py --help
uv run python eval_grpo.py --help
uv run python run_prelim.py --help
uv run python compare_runs.py --help
```

### CPU-friendly evaluation (larger sample)

If you run a fast end-to-end pipeline with tiny eval sizes, use this to get a more stable
accuracy estimate on CPU:

```bash
uv run python eval_grpo.py \
  --config configs/eval_cpu.yaml \
  --model-name-or-path outputs/prelim/<experiment_id>/<model_slug>/train
```

## Model testing UI

Launch a basic Streamlit UI to inspect run outputs and quickly prompt a selected model:

```bash
./scripts/run_ui.sh
```

The UI includes:

- Results browser for `outputs/prelim/<experiment_id>` runs
- Per-run baseline/tuned metrics and prediction tables
- Experiment leaderboard preview
- Prompt playground with a small model dropdown and generation controls

Model options in the prompt panel include:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- discovered local checkpoints under `outputs/`

If no experiment appears, run the preliminary pipeline first:

```bash
./scripts/run_prelim.sh
```

Override config values from CLI:

```bash
accelerate launch train_grpo.py \
  --config configs/grpo_small.yaml \
  --max-steps 50 \
  --learning-rate 5e-7 \
  --reward-type format
```

Run standalone evaluation on held-out split:

```bash
uv run python eval_grpo.py \
  --config configs/eval_prelim.yaml \
  --model-name-or-path outputs/grpo-qwen2.5-0.5b \
  --tokenizer-name-or-path Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/eval-grpo-qwen2.5-0.5b
```

Manually compare two runs:

```bash
uv run python compare_runs.py \
  --output-dir outputs/compare-example \
  --model base:Qwen/Qwen2.5-0.5B-Instruct:outputs/base_eval/metrics.json \
  --model tuned:outputs/grpo-qwen2.5-0.5b:outputs/tuned_eval/metrics.json
```

## Reward options

- `accuracy`: robust accuracy reward (math parser + string fallback)
- `accuracy_format`: weighted reward `0.8 * accuracy + 0.2 * format`
- `format`: gives reward if completion includes `\\boxed{...}` or `Answer:`
- `length`: token-length reward (useful for smoke/throughput checks)

## Notes

- Defaults are tuned for a small model setup with LoRA enabled.
- `num_generations` should divide the effective generation batch size.
- Outputs are written under `outputs/`.
- `eval_grpo.py` expects datasets with `prompt` and `solution` columns.
- `run_prelim.py` is configured for DeepMath `train`/`test` by default.
- Training in `run_prelim.py` uses `accelerate` by default; disable with `--no-use-accelerate` if needed.
- Current prelim defaults favor reduced truncation (`max_completion_length=192`) and denser rewards (`accuracy_format`).
- Training now exposes rollout quality knobs (`temperature`, `top_p`, `repetition_penalty`) and `mask_truncated_completions`.
- Preliminary pipeline defaults to `eval_backend: transformers`.

## RunPod deployment

See **[RUNPOD.md](RUNPOD.md)** for the complete guide covering:

- Hardware recommendations (spot GPU selection)
- Base template setup (no custom Docker needed)
- Spot instance resilience (SIGTERM handling, auto-resume, checkpointing)
- Dependency install pitfalls (blinker + torch version mismatch fixes)
- Smoke vs full sweep pipeline
- Tmux workflow
- File reference and checklist