from __future__ import annotations

import argparse
import ast
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import torch
import yaml

from grpo_config import TrainingConfig
from reward_fns import (
    accuracy_format_reward,
    format_reward,
    length_reward,
    robust_accuracy_reward,
)

_interrupted = False


def _signal_handler(signum: int, frame) -> None:
    global _interrupted
    _interrupted = True
    name = signal.Signals(signum).name
    sys.stderr.write(
        f"\n[{name}] Interrupt received. Will save checkpoint at end of current step.\n"
    )
    sys.stderr.flush()


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small LM with GRPO.")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config file path (values overridden by explicit CLI flags).",
    )
    parser.add_argument("--model-name", default=None, help="Base model name or path.")
    parser.add_argument("--dataset-name", default=None, help="HF dataset name.")
    parser.add_argument("--dataset-split", default=None, help="Dataset split.")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Optional dataset size cap."
    )
    parser.add_argument("--output-dir", default=None, help="Training output directory.")
    parser.add_argument("--run-name", default=None, help="Experiment run name.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--num-train-epochs", type=float, default=None, help="Number of epochs."
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max optimization steps."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate."
    )
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps.")
    parser.add_argument(
        "--warmup-ratio", type=float, default=None, help="Warmup ratio."
    )
    parser.add_argument(
        "--logging-steps", type=int, default=None, help="Logging frequency."
    )
    parser.add_argument(
        "--save-steps", type=int, default=None, help="Checkpoint frequency."
    )
    parser.add_argument(
        "--save-total-limit", type=int, default=None, help="Max kept checkpoints."
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-generations", type=int, default=None, help="Generations per prompt."
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=None, help="Completion length cap."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature used for rollouts.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter used for rollouts.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty during rollout generation.",
    )
    parser.add_argument("--beta", type=float, default=None, help="KL beta coefficient.")
    parser.add_argument(
        "--scale-rewards",
        default=None,
        choices=["group", "batch", "none"],
        help="Reward scaling mode.",
    )
    parser.add_argument(
        "--loss-type",
        default=None,
        choices=["grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo", "luspo", "vespo"],
        help="GRPO loss variant.",
    )
    parser.add_argument(
        "--mask-truncated-completions",
        action="store_true",
        help="Mask truncated completions in policy loss.",
    )
    parser.add_argument(
        "--no-mask-truncated-completions",
        action="store_true",
        help="Disable masking for truncated completions.",
    )
    parser.add_argument(
        "--reward-type",
        default=None,
        choices=["accuracy", "accuracy_format", "format", "length"],
        help="Built-in reward to use.",
    )
    parser.add_argument(
        "--use-vllm", action="store_true", help="Use vLLM generation backend."
    )
    parser.add_argument(
        "--no-use-vllm", action="store_true", help="Disable vLLM backend."
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing.",
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument(
        "--report-to",
        default=None,
        help="Tracking backend: none, wandb, tensorboard, or comma-separated list.",
    )
    parser.add_argument(
        "--log-completions", action="store_true", help="Log sampled completions."
    )
    parser.add_argument(
        "--no-log-completions", action="store_true", help="Disable completion logging."
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Auto-resume from latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable auto-resume from checkpoint.",
    )
    parser.add_argument(
        "--use-peft", action="store_true", help="Enable LoRA fine-tuning."
    )
    parser.add_argument(
        "--no-use-peft", action="store_true", help="Disable LoRA fine-tuning."
    )
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha.")
    parser.add_argument(
        "--lora-dropout", type=float, default=None, help="LoRA dropout."
    )
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help='Python list/tuple of module names, e.g. \'["q_proj","v_proj"]\'.',
    )
    return parser.parse_args()


def load_config_file(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return data


def merge_config(defaults: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    data = asdict(defaults)

    if args.config is not None:
        file_overrides = load_config_file(args.config)
        unknown = [key for key in file_overrides if key not in data]
        if unknown:
            unknown_display = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown keys in config file: {unknown_display}")
        data.update(file_overrides)

    overrides = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "max_samples": args.max_samples,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "beta": args.beta,
        "scale_rewards": args.scale_rewards,
        "loss_type": args.loss_type,
        "reward_type": args.reward_type,
        "report_to": args.report_to,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }

    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if args.use_vllm:
        data["use_vllm"] = True
    if args.no_use_vllm:
        data["use_vllm"] = False

    if args.gradient_checkpointing:
        data["gradient_checkpointing"] = True
    if args.no_gradient_checkpointing:
        data["gradient_checkpointing"] = False

    if args.bf16:
        data["bf16"] = True
        data["fp16"] = False
    if args.fp16:
        data["fp16"] = True
        data["bf16"] = False

    if args.log_completions:
        data["log_completions"] = True
    if args.no_log_completions:
        data["log_completions"] = False

    if args.mask_truncated_completions:
        data["mask_truncated_completions"] = True
    if args.no_mask_truncated_completions:
        data["mask_truncated_completions"] = False

    if args.use_peft:
        data["use_peft"] = True
    if args.no_use_peft:
        data["use_peft"] = False

    if args.auto_resume:
        data["auto_resume"] = True
    if args.no_auto_resume:
        data["auto_resume"] = False

    if args.lora_target_modules:
        parsed = ast.literal_eval(args.lora_target_modules)
        data["lora_target_modules"] = tuple(parsed)

    if isinstance(data.get("lora_target_modules"), list):
        data["lora_target_modules"] = tuple(data["lora_target_modules"])

    if data.get("warmup_steps") is not None and data["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be >= 0")

    if data.get("warmup_steps") in (None, 0) and data.get("warmup_ratio") not in (
        None,
        0,
    ):
        max_steps = data.get("max_steps", -1)
        if max_steps and max_steps > 0:
            data["warmup_steps"] = int(max_steps * float(data["warmup_ratio"]))
        else:
            data["warmup_steps"] = 0

    if data["report_to"] == "none":
        data["report_to"] = []
    elif isinstance(data["report_to"], str) and "," in data["report_to"]:
        data["report_to"] = [
            item.strip() for item in data["report_to"].split(",") if item.strip()
        ]

    return TrainingConfig(**data)


def get_reward_function(reward_type: str):
    if reward_type == "accuracy":
        return robust_accuracy_reward

    if reward_type == "format":
        return format_reward

    if reward_type == "accuracy_format":
        return accuracy_format_reward

    if reward_type == "length":
        return length_reward

    raise ValueError(f"Unsupported reward type: {reward_type}")


def build_peft_config(config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        task_type="CAUSAL_LM",
    )


def sanitize_tokenizer_special_tokens(trainer: GRPOTrainer) -> None:
    tokenizer = getattr(trainer, "processing_class", None)
    model_config = getattr(getattr(trainer, "model", None), "config", None)
    generation_config = getattr(getattr(trainer, "model", None), "generation_config", None)

    if tokenizer is None:
        return

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_config is not None:
        if tokenizer.pad_token_id is not None:
            model_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            model_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            model_config.bos_token_id = tokenizer.bos_token_id

    if generation_config is not None:
        if tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            generation_config.bos_token_id = tokenizer.bos_token_id


class SpotInterruptCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        global _interrupted
        if _interrupted:
            control.should_save = True
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        global _interrupted
        if _interrupted:
            control.should_save = True
            control.should_training_stop = True


def main() -> None:
    args = parse_args()
    config = merge_config(TrainingConfig(), args)

    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    if config.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))

    reward_func = get_reward_function(config.reward_type)

    grpo_args = GRPOConfig(
        output_dir=config.output_dir,
        run_name=config.run_name,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        beta=config.beta,
        scale_rewards=config.scale_rewards,
        loss_type=config.loss_type,
        mask_truncated_completions=config.mask_truncated_completions,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to=config.report_to,
        log_completions=config.log_completions,
        use_vllm=config.use_vllm,
        seed=config.seed,
        dataloader_pin_memory=not torch.cuda.is_available(),
    )

    trainer_kwargs = {
        "model": config.model_name,
        "args": grpo_args,
        "reward_funcs": reward_func,
        "train_dataset": dataset,
    }

    if config.use_peft:
        trainer_kwargs["peft_config"] = build_peft_config(config)

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.add_callback(SpotInterruptCallback())
    sanitize_tokenizer_special_tokens(trainer)

    resume = config.resume_from_checkpoint
    if resume is None and config.auto_resume:
        output = Path(config.output_dir)
        checkpoints = sorted(output.glob("checkpoint-*"))
        if checkpoints:
            resume = str(checkpoints[-1])
            print(
                f"Auto-resuming from latest checkpoint: {resume}",
                flush=True,
            )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
