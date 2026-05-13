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

## 2026-05-12: Modal Pre-Sweep Analysis (100 steps)

### Overview
Ran a 100-step pre-sweep on Modal (4 A10G concurrent containers) using `configs/pipeline_sweep_presweep.yaml` to identify the best hyperparameter combinations before running a full sweep.

### Results
- **Qwen 0.5B + GRPO**: 5.5% → 4.0% (-1.5%)
- **Qwen 0.5B + DAPO**: 5.5% → 5.5% (0.0%)
- **Gemma 1B + GRPO**: 10.0% → 10.0% (0.0%)
- **Gemma 1B + DAPO**: 10.0% → 10.0% (0.0%)

### Key Insights

1. **The Gemma Truncation Wall**
   Gemma-3-1B has a mathematically stronger baseline (10% vs Qwen's 5.5%), but its generation behavior is completely incompatible with a `max_completion_length` of 512 tokens.
   - **Gemma Truncation Rate:** 99.5%
   - **Gemma Format Rate:** 9.5%
   - **Avg tokens:** ~511.6
   Gemma almost never emitted a `\boxed{}` answer before getting cut off, meaning it received zero meaningful reward signal and learned nothing in 100 steps.

2. **Qwen Behaves Better Under Constraints**
   Qwen 0.5B fits much better within the 512-token constraint:
   - **Qwen Truncation Rate:** ~65.5%
   - **Qwen Format Rate:** ~35.5%
   While 65% truncation is still high, it means ~1/3rd of the time it successfully formatted an answer and received a reward signal. Between the two losses, **DAPO** prevented the model from degrading, whereas standard GRPO led to a slight regression (-1.5%).

### Next Steps & Recommendations
- **Increase `max_completion_length`**: For the full sweep, bump `max_completion_length` to 768 or 1024 in `configs/pipeline_sweep_full.yaml`. Both models are severely struggling to finish their reasoning paths within 512 tokens. Giving them more room is necessary to actually hit the format reward and start improving.
- **Winner of the Pre-Sweep**: If optimizing for speed and cost under tight constraints, **Qwen + DAPO** is the winner. To unlock Gemma's higher baseline performance, increasing the completion length is strictly required.

## 2026-05-12: Full Sweep Analysis (250 steps, Qwen 0.5B, 768 context)

### Overview
Ran a focused 250-step full sweep on Qwen 0.5B to compare GRPO vs DAPO, using a budget-optimized `max_completion_length` of 768 (increased from 512 based on pre-sweep findings). 

### Results (500 eval samples)
- **Baseline Accuracy**: 9.4% (up from 5.5% in the pre-sweep simply because the model has room to finish generating the answer)
- **Qwen 0.5B + GRPO**: 9.4% → 8.8% (-0.6%)
- **Qwen 0.5B + DAPO**: 9.4% → 9.4% (0.0%)

### Metrics Under the Hood
Increasing context length to 768 worked exactly as intended:
- **Baseline Format Rate**: ~61.8% (up from 35.5% at 512 context)
- **Baseline Truncation Rate**: ~37.2% (down from 65.5% at 512 context)
- **Average Tokens**: ~605-612 tokens per generation

### Key Insights
1. **Context Expansion was a Success**: Expanding context length to 768 effectively doubled the format success rate and nearly doubled baseline accuracy (since responses weren't truncated right before the `\boxed{}` output).
2. **GRPO vs DAPO**: 
   - **GRPO** slightly *decreased* accuracy (-0.6%), but slightly *increased* formatting rate (from 61.8% to 63.2%). This suggests standard GRPO might be hacking the format reward without actually improving reasoning (reward hacking).
   - **DAPO** prevented accuracy degradation and maintained the exact baseline performance, acting as a stronger regularizer against reward hacking.
3. **Training Duration**: 250 steps is still very early in the RLHF process. The models are learning the format (as seen by GRPO's format rate going up) but haven't trained long enough to exhibit strong accuracy gains on complex reasoning paths. 

### Conclusion
DAPO is more stable than GRPO under these constraints, preventing the slight regression seen with standard GRPO. The context length of 768 is a great sweet spot for Qwen 0.5B on DeepMath. Further gains will require scaling `max_steps` higher (e.g. 1000+) or increasing `num_generations` (batch size) for a cleaner advantage signal.
