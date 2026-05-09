from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "trl-lib/DeepMath-103K"
    dataset_split: str = "train"
    max_samples: int | None = None
    output_dir: str = "outputs/grpo-qwen2.5-0.5b"
    run_name: str = "grpo-small-lm"
    seed: int = 42
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 1e-6
    warmup_steps: int = 0
    warmup_ratio: float | None = None
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 8
    max_completion_length: int = 384
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    beta: float = 0.0
    scale_rewards: str = "group"
    loss_type: str = "dapo"
    mask_truncated_completions: bool = True
    reward_type: str = "accuracy_format"
    use_vllm: bool = False
    gradient_checkpointing: bool = True
    bf16: bool = False
    fp16: bool = False
    report_to: str | list[str] = "none"
    log_completions: bool = True
    resume_from_checkpoint: str | None = None
    auto_resume: bool = False
    hf_token: str | None = None
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        )
    )
