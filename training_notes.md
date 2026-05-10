# Training Notes Log

## 2026-05-10: Sweep Analysis (Run: 20260510-033624)

### What Went Wrong

1. **Only Gemma 3 Trained (Parallel Execution Conflict)**
   In `configs/pipeline_sweep.yaml`, `parallel: true` was set with multiple models (`Qwen2.5-0.5B-Instruct` and `gemma-3-1b-it`). The `run_prelim.py` script launched them concurrently as child processes without GPU isolation (e.g., `CUDA_VISIBLE_DEVICES`). This caused an Out-Of-Memory (OOM) error or Accelerate port conflict for Qwen, crashing it immediately, while Gemma happened to survive and trained.

2. **Performance Regression (Severe Truncation)**
   The GRPO logs (`trainer_state.json`) indicated:
   - `"completions/max_length": 384.0`
   - `"completions/mean_length": 384.0`
   
   Every single generated sequence hit the ceiling of the `max_completion_length` (`384`). DeepMath requires step-by-step Chain-of-Thought (CoT) reasoning. Because the model was cut off, it never successfully reached the point where it could output the `\boxed{answer}` format. As a result, the reward signal stayed flat (~0.05 instead of 1.0). The model learned almost nothing, leading to a performance drop from 0.080 (baseline) to 0.078 (tuned).

### Parameter Recommendations & Next Steps

1. **`max_completion_length` (Critical)**: Increase to at least `1024` or `2048` in `pipeline_sweep.yaml`. The model needs enough tokens to finish its reasoning path and emit the `\boxed{}` answer to receive a reward.
2. **`parallel`**: Change to `parallel: false` so that the pipeline evaluates and trains sequentially. If parallel runs are strictly desired, modify `run_prelim.py` to assign a distinct `CUDA_VISIBLE_DEVICES` environment variable per child process.
3. **`eval_max_new_tokens`**: Since you'll be extending the completion length for training, make sure this is also pushed to at least `1024` or `2048` to properly evaluate the baseline and tuned models.
4. **`num_generations`**: `8` is a decent baseline, but once you fix the sequence length, if you find the reward variance is still low, try bumping this to `16` (adjusting `per_device_train_batch_size` or `gradient_accumulation_steps` if VRAM becomes an issue).
5. **`learning_rate` & `beta`**: `1.0e-6` and `beta: 0.01` are safe RL parameter choices. Leave them as-is until the sequence length truncation issue is resolved, after which you might experiment with `5e-6` for the LR.
