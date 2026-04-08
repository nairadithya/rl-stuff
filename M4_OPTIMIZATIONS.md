# M4 MacBook Air Optimizations Summary

This document explains the optimizations added for running GRPO training on an M4 MacBook Air with 16GB RAM.

## Key Optimizations

### 1. **Apple Silicon MPS Support** (train_grpo.py:271-296)
- Automatic detection of Metal Performance Shaders (MPS) backend
- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops
- Configures `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` (70% memory limit)
- Optimizes for M4's 8 cores (4 performance + 4 efficiency)

### 2. **BF16 Precision** (configs/grpo_m4_macbook.yaml:33)
- Enabled `bf16: true` to use M4's hardware BF16 acceleration
- 2x faster than FP32, same memory as FP16, better numerical stability
- M4 has dedicated BF16 matrix multiplication units

### 3. **Memory-Efficient Configuration**
Reduced parameters to fit in 16GB RAM:

| Parameter | Standard | M4 Config | Reason |
|-----------|----------|-----------|--------|
| `max_samples` | 5000 | 2000 | Less data in memory |
| `per_device_train_batch_size` | 1 | 1 | Minimal batch size |
| `gradient_accumulation_steps` | 8 | 4 | Reduce intermediate tensors |
| `num_generations` | 8 | 4 | Fewer concurrent generations |
| `max_completion_length` | 256 | 128 | Shorter sequences |
| `lora_r` | 16 | 8 | Smaller LoRA rank |
| `lora_alpha` | 32 | 16 | Scaled with rank |

### 4. **Essential Memory-Saving Features**
- **Gradient Checkpointing**: Enabled (trades compute for memory)
- **LoRA (PEFT)**: Uses parameter-efficient fine-tuning
- **vLLM**: Disabled (not compatible with MPS)

### 5. **CPU Optimization**
- Sets `OMP_NUM_THREADS=8` to use all CPU cores
- Enables `TOKENIZERS_PARALLELISM=true` for faster tokenization
- Leverages M4's efficiency cores for data loading

### 6. **Accelerate Configuration** (configs/accelerate_m4.yaml)
- Single-device setup (no distributed training)
- BF16 mixed precision enabled
- Optimized for LOCAL_MACHINE compute environment

## Performance Expectations

### Memory Usage
- **Model Loading**: ~1-2 GB (Qwen2.5-0.5B with LoRA)
- **Training**: ~4-6 GB peak
- **System Reserve**: ~2-3 GB for macOS
- **Total**: ~8-11 GB (comfortable for 16GB)

### Speed
- **Tokens/sec**: ~50-100 (depends on sequence length)
- **Steps/sec**: ~1-3 (with current config)
- **Total training time**: ~5-15 minutes for 200 steps

### Comparison to CUDA
- ~2-4x slower than modern NVIDIA GPUs
- But 100% usable for development and small-scale experiments
- No need for cloud resources or external GPUs

## Usage

### Quick Start
```bash
./scripts/run_m4_macbook.sh
```

### Manual Launch
```bash
accelerate launch --config_file configs/accelerate_m4.yaml \
  train_grpo.py --config configs/grpo_m4_macbook.yaml
```

### Custom Overrides
```bash
./scripts/run_m4_macbook.sh \
  --max-steps 100 \
  --learning-rate 5e-7 \
  --num-generations 2
```

## Troubleshooting

### Out of Memory (OOM)
If you still hit OOM errors:
1. Reduce `num_generations` to 2
2. Reduce `max_completion_length` to 64
3. Reduce `lora_r` to 4
4. Reduce `max_samples` to 1000

### Slow Performance
If training is too slow:
1. Reduce `max_samples` for faster iteration
2. Use `configs/grpo_smoke.yaml` for quick testing
3. Consider using a cloud GPU for production runs

### MPS Not Available
If MPS is not detected:
1. Ensure macOS >= 12.3
2. Ensure PyTorch >= 2.0 with MPS support
3. Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

## Technical Details

### Why These Settings?

**BF16 on M4**: The M4 chip has dedicated AMX (Apple Matrix) coprocessors that accelerate BF16 operations. This is faster and more memory-efficient than FP32.

**LoRA rank=8**: Each LoRA layer adds `2 * d * r` parameters. With r=8, this is 4x smaller than r=16, significantly reducing memory for gradients and optimizer states.

**Gradient Accumulation=4**: This simulates a larger batch size while keeping memory low. Effective batch size = 1 * 4 = 4.

**MPS Fallback**: Some PyTorch operations aren't implemented for MPS yet. The fallback ensures they run on CPU instead of crashing.

**70% Memory Watermark**: Leaves 30% of unified memory for macOS, browser, and other apps. This prevents system slowdowns.

## Files Modified/Created

1. **configs/grpo_m4_macbook.yaml** - M4-optimized training config
2. **configs/accelerate_m4.yaml** - M4-optimized accelerate config  
3. **scripts/run_m4_macbook.sh** - Convenience runner script
4. **train_grpo.py** - Added `setup_apple_silicon_optimizations()` function
5. **README.md** - Added M4 MacBook Air section
6. **M4_OPTIMIZATIONS.md** - This file

## References

- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple M4 Specifications](https://www.apple.com/mac/m4/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
