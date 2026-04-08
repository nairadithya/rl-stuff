# GRPO Small-LM Training Environment

Minimal scaffold to train small language models with **Group Relative Policy Optimization (GRPO)** using Hugging Face TRL.

## What this includes

- `train_grpo.py`: CLI entrypoint for GRPO training
- `grpo_config.py`: typed defaults for training parameters
- `reward_fns.py`: custom reward functions (`format_reward`, `length_reward`)
- `configs/grpo_small.yaml`: practical baseline config
- `configs/grpo_smoke.yaml`: tiny smoke-test config
- `configs/grpo_m4_macbook.yaml`: optimized config for M4 MacBook Air (16GB RAM)
- `scripts/run_smoke.sh`: helper to run smoke training
- `scripts/run_m4_macbook.sh`: helper to run on M4 MacBook Air

## Requirements

- Python `>=3.10`
- CUDA-capable GPU recommended (or Apple Silicon M1/M2/M3/M4 with MPS support)
- `accelerate` configured (`accelerate config`)

Install dependencies:

```bash
pip install -e .
```

## Quick start

### For M4 MacBook Air (16GB RAM) - Optimized! 🍎

Run with Apple Silicon optimizations:

```bash
./scripts/run_m4_macbook.sh
```

Or manually:

```bash
accelerate launch train_grpo.py --config configs/grpo_m4_macbook.yaml
```

**M4-specific optimizations:**
- Metal Performance Shaders (MPS) backend for Apple Silicon
- BF16 precision (hardware accelerated on M4)
- Memory-efficient batch sizes (batch_size=1, grad_accum=4)
- Reduced LoRA rank (r=8) for lower memory usage
- Gradient checkpointing enabled
- Optimized for 16GB RAM constraint
- Uses all 8 CPU cores (4 performance + 4 efficiency)

📖 **See [M4_OPTIMIZATIONS.md](M4_OPTIMIZATIONS.md) for detailed explanation of all optimizations.**

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
```

Override config values from CLI:

```bash
accelerate launch train_grpo.py \
  --config configs/grpo_small.yaml \
  --max-steps 50 \
  --learning-rate 5e-7 \
  --reward-type format
```

## Reward options

- `accuracy`: TRL built-in `accuracy_reward`
- `format`: gives reward if completion includes `\\boxed{...}` or `Answer:`
- `length`: token-length reward (useful for smoke/throughput checks)

## Notes

- Defaults are tuned for a small model setup with LoRA enabled.
- `num_generations` should divide the effective generation batch size.
- Outputs are written under `outputs/`.
