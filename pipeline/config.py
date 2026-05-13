from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

@dataclass
class PrelimConfig:
    models: list[str] | str = "Qwen/Qwen2.5-0.5B-Instruct"
    tuned_model_paths: list[str] | None = None
    dataset_name: str = "trl-lib/DeepMath-103K"
    train_split: str = "train"
    test_split: str = "test"
    train_max_samples: int | None = 2000
    test_max_samples: int | None = 500
    max_steps: int = 100
    learning_rate: float = 1.0e-6
    reward_type: str = "accuracy"
    max_completion_length: int = 128
    eval_max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    mask_truncated_completions: bool = True
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    loss_types: list[str] | None = None
    beta: float = 0.0
    num_train_epochs: float = 1.0
    output_root: str = "outputs/prelim"
    train_config: str | None = None
    eval_config: str | None = None
    run_name_prefix: str = "prelim"
    seed: int = 42
    skip_training: bool = False
    python_bin: str | None = None
    use_accelerate: bool = True
    accelerate_config: str | None = None
    eval_backend: str = "transformers"
    parallel: bool = False
    max_concurrent: int | None = None
    notes: str | None = None
    tags: dict[str, str] | None = None
    yes: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline -> GRPO train -> tuned eval for preliminary results."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config file path (values overridden by explicit CLI flags).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more base model names/paths to run.",
    )
    parser.add_argument(
        "--tuned-model-paths",
        nargs="+",
        default=None,
        help=(
            "Optional tuned model paths matching --models order. "
            "Required when --skip-training is set."
        ),
    )
    parser.add_argument("--dataset-name", default=None, help="HF dataset name.")
    parser.add_argument("--train-split", default=None, help="Training split name.")
    parser.add_argument("--test-split", default=None, help="Evaluation split name.")
    parser.add_argument(
        "--train-max-samples", type=int, default=None, help="Train sample cap."
    )
    parser.add_argument(
        "--test-max-samples", type=int, default=None, help="Eval sample cap."
    )
    parser.add_argument("--max-steps", type=int, default=None, help="GRPO max steps.")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="GRPO learning rate."
    )
    parser.add_argument(
        "--reward-type",
        default=None,
        choices=["accuracy", "accuracy_format", "format", "length"],
        help="Reward function for GRPO training.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=None,
        help="Training max completion length.",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=None,
        help="Evaluation max new tokens (decoupled from training completion length).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Training rollout temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Training rollout top-p.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Training rollout repetition penalty.",
    )
    parser.add_argument(
        "--mask-truncated-completions",
        action="store_true",
        help="Mask truncated rollouts in policy loss.",
    )
    parser.add_argument(
        "--no-mask-truncated-completions",
        action="store_true",
        help="Disable masking of truncated rollouts.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Number of generations per prompt during training.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Training gradient accumulation steps.",
    )
    parser.add_argument(
        "--loss-types",
        nargs="+",
        default=None,
        help="Loss types to sweep over (grpo, dapo, dr_grpo, etc.).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="KL penalty coefficient.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root directory for all artifacts.",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="Base training config YAML passed to train_grpo.py.",
    )
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Base eval config YAML passed to eval_grpo.py.",
    )
    parser.add_argument(
        "--eval-backend",
        default=None,
        choices=["transformers"],
        help="Evaluation backend.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default=None,
        help="Prefix for per-model run names.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip GRPO training and only run evaluations/comparison.",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python executable used for subprocesses.",
    )
    parser.add_argument(
        "--accelerate-config",
        default=None,
        help="Optional accelerate config file used for training launch.",
    )
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="Launch training via accelerate.",
    )
    parser.add_argument(
        "--no-use-accelerate",
        action="store_true",
        help="Launch training directly with python.",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Train multiple models simultaneously."
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Train models sequentially (default).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max models to train simultaneously (default: all if --parallel).",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Free-text notes describing this experiment.",
    )
    parser.add_argument(
        "--tags",
        default=None,
        nargs="+",
        help='Key-value tags, e.g. --tags dataset=foo lr=1e-6.',
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip interactive prompts.",
    )
    return parser.parse_args()


def load_config_file(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return data


def load_config_file(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return data


def merge_config(defaults: PrelimConfig, args: argparse.Namespace) -> PrelimConfig:
    data = asdict(defaults)

    if args.config is not None:
        file_overrides = load_config_file(args.config)
        unknown = [key for key in file_overrides if key not in data]
        if unknown:
            unknown_display = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown keys in config file: {unknown_display}")
        data.update(file_overrides)

    overrides = {
        "dataset_name": args.dataset_name,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_max_samples": args.train_max_samples,
        "test_max_samples": args.test_max_samples,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "reward_type": args.reward_type,
        "max_completion_length": args.max_completion_length,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_generations": args.num_generations,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "beta": args.beta,
        "num_train_epochs": args.num_train_epochs,
        "output_root": args.output_root,
        "train_config": args.train_config,
        "eval_config": args.eval_config,
        "run_name_prefix": args.run_name_prefix,
        "seed": args.seed,
        "python_bin": args.python_bin,
        "accelerate_config": args.accelerate_config,
        "eval_backend": args.eval_backend,
        "max_concurrent": args.max_concurrent,
        "notes": args.notes,
    }

    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if args.models is not None:
        data["models"] = args.models

    if args.tuned_model_paths is not None:
        data["tuned_model_paths"] = args.tuned_model_paths

    if args.loss_types is not None:
        data["loss_types"] = args.loss_types

    if args.skip_training:
        data["skip_training"] = True

    if args.use_accelerate:
        data["use_accelerate"] = True
    if args.no_use_accelerate:
        data["use_accelerate"] = False

    if args.mask_truncated_completions:
        data["mask_truncated_completions"] = True
    if args.no_mask_truncated_completions:
        data["mask_truncated_completions"] = False

    if args.parallel:
        data["parallel"] = True
    if args.no_parallel:
        data["parallel"] = False

    if args.tags is not None:
        tags_dict: dict[str, str] = {}
        for tag in args.tags:
            if "=" not in tag:
                raise ValueError(
                    f"Invalid tag format: '{tag}'. Expected key=value."
                )
            key, _, value = tag.partition("=")
            if not key:
                raise ValueError(f"Tag key cannot be empty: '{tag}'.")
            tags_dict[key] = value
        data["tags"] = tags_dict

    if hasattr(args, "yes") and args.yes:
        data["yes"] = True

    return PrelimConfig(**data)

