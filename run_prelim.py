from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

    models = data["models"]
    if isinstance(models, str):
        models = [models]
    if not models:
        raise ValueError("At least one model must be specified.")
    data["models"] = models

    tuned_model_paths = data.get("tuned_model_paths")
    if tuned_model_paths is not None:
        if len(tuned_model_paths) != len(models):
            raise ValueError(
                "--tuned-model-paths must have the same number of entries as --models"
            )

    if data["skip_training"] and tuned_model_paths is None:
        raise ValueError(
            "When --skip-training is set, provide --tuned-model-paths in model order."
        )

    if (not data["skip_training"]) and tuned_model_paths is not None:
        raise ValueError(
            "--tuned-model-paths can only be used with --skip-training."
        )

    if data["max_steps"] <= 0:
        raise ValueError("max_steps must be > 0")
    if data["learning_rate"] <= 0:
        raise ValueError("learning_rate must be > 0")
    if data["max_completion_length"] <= 0:
        raise ValueError("max_completion_length must be > 0")
    if data["eval_max_new_tokens"] <= 0:
        raise ValueError("eval_max_new_tokens must be > 0")
    if data["temperature"] < 0:
        raise ValueError("temperature must be >= 0")
    if not (0 < data["top_p"] <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if data["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty must be > 0")
    if data["num_generations"] <= 0:
        raise ValueError("num_generations must be > 0")
    if data["per_device_train_batch_size"] <= 0:
        raise ValueError("per_device_train_batch_size must be > 0")
    if data["gradient_accumulation_steps"] <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")

    if data.get("python_bin") in (None, ""):
        data["python_bin"] = sys.executable

    return PrelimConfig(**data)


def _sanitize_name(name: str) -> str:
    safe = name.replace("/", "--").replace(" ", "-")
    keep = []
    for ch in safe:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("-")
    sanitized = "".join(keep).strip("-")
    return sanitized or "model"


def _run(command: list[str], *, cwd: Path) -> None:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _eval_model(
    *,
    python_bin: str,
    repo_root: Path,
    model_name_or_path: str,
    dataset_name: str,
    split: str,
    max_samples: int | None,
    output_dir: Path,
    seed: int,
    eval_config: str | None,
    max_new_tokens: int,
    repetition_penalty: float,
) -> Path:
    eval_script = "eval_grpo.py"

    command = [python_bin, eval_script]
    if eval_config is not None:
        command.extend(["--config", eval_config])
    command.extend(
        [
            "--model-name-or-path",
            model_name_or_path,
            "--dataset-name",
            dataset_name,
            "--dataset-split",
            split,
            "--max-new-tokens",
            str(max_new_tokens),
            "--batch-size",
            "1",
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
        ]
    )
    command.extend(["--repetition-penalty", str(repetition_penalty)])
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])

    _run(command, cwd=repo_root)
    return output_dir / "metrics.json"


def _train_model(
    *,
    python_bin: str,
    repo_root: Path,
    base_model: str,
    dataset_name: str,
    train_split: str,
    train_max_samples: int | None,
    output_dir: Path,
    run_name: str,
    seed: int,
    max_steps: int,
    learning_rate: float,
    reward_type: str,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    mask_truncated_completions: bool,
    num_generations: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    train_config: str | None,
    use_accelerate: bool,
    accelerate_config: str | None,
    notes: str | None = None,
    tags: dict[str, str] | None = None,
    loss_type: str | None = None,
    beta: float | None = None,
    num_train_epochs: float | None = None,
) -> None:
    if use_accelerate:
        command = [python_bin, "-m", "accelerate.commands.launch"]
        if accelerate_config is not None:
            command.extend(["--config_file", accelerate_config])
        command.append("train_grpo.py")
    else:
        command = [python_bin, "train_grpo.py"]

    if train_config is not None:
        command.extend(["--config", train_config])

    command.extend(
        [
            "--model-name",
            base_model,
            "--dataset-name",
            dataset_name,
            "--dataset-split",
            train_split,
            "--output-dir",
            str(output_dir),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--max-steps",
            str(max_steps),
            "--learning-rate",
            str(learning_rate),
            "--reward-type",
            reward_type,
            "--max-completion-length",
            str(max_completion_length),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
            "--repetition-penalty",
            str(repetition_penalty),
            "--num-generations",
            str(num_generations),
            "--per-device-train-batch-size",
            str(per_device_train_batch_size),
            "--gradient-accumulation-steps",
            str(gradient_accumulation_steps),
        ]
    )
    if loss_type is not None:
        command.extend(["--loss-type", loss_type])
    if beta is not None:
        command.extend(["--beta", str(beta)])
    if num_train_epochs is not None:
        command.extend(["--num-train-epochs", str(num_train_epochs)])
    if train_max_samples is not None:
        command.extend(["--max-samples", str(train_max_samples)])
    if mask_truncated_completions:
        command.append("--mask-truncated-completions")
    else:
        command.append("--no-mask-truncated-completions")

    if notes is not None:
        command.extend(["--notes", notes])
    if tags is not None:
        for key, value in tags.items():
            command.extend(["--tags", f"{key}={value}"])

    _run(command, cwd=repo_root)


def _run_compare(config, repo_root, model_dir, model, model_slug,
                 baseline_metrics_path, tuned_model_path, tuned_metrics_path,
                 lt_slug="default"):
    compare_dir = model_dir / "comparison" / lt_slug
    label = f"{model_slug}-{lt_slug}"
    _run(
        [
            config.python_bin,
            "compare_runs.py",
            "--output-dir",
            str(compare_dir),
            "--model",
            f"base:{model}:{baseline_metrics_path}",
            "--model",
            f"{label}:{tuned_model_path}:{tuned_metrics_path}",
        ],
        cwd=repo_root,
    )


def _add_leaderboard_row(rows, model, model_slug, loss_type,
                         baseline_metrics, tuned_metrics, tuned_model_path,
                         baseline_metrics_path, tuned_metrics_path):
    baseline_accuracy = baseline_metrics.get("accuracy_mean")
    tuned_accuracy = tuned_metrics.get("accuracy_mean")
    accuracy_delta = (
        float(tuned_accuracy) - float(baseline_accuracy)
        if baseline_accuracy is not None and tuned_accuracy is not None
        else None
    )
    row = {
        "model": model,
        "baseline_accuracy": baseline_accuracy,
        "tuned_accuracy": tuned_accuracy,
        "accuracy_delta": accuracy_delta,
        "baseline_metrics_path": str(baseline_metrics_path),
        "tuned_metrics_path": str(tuned_metrics_path),
        "tuned_model_path": tuned_model_path,
    }
    if loss_type is not None:
        row["loss_type"] = loss_type
    rows.append(row)


def _record_run(manifest_path, model, model_slug, loss_type,
                baseline_metrics_path, tuned_metrics_path,
                tuned_model_path, model_dir, notes=None, tags=None):
    lt_slug = loss_type or "default"
    run_record = {
        "model": model,
        "model_slug": model_slug,
        "loss_type": loss_type,
        "baseline_metrics": str(baseline_metrics_path),
        "tuned_metrics": str(tuned_metrics_path),
        "train_dir": str(model_dir / "train" / lt_slug),
        "tuned_model_path": tuned_model_path,
        "comparison_dir": str(model_dir / "comparison" / lt_slug),
    }
    if notes is not None:
        run_record["notes"] = notes
    if tags is not None:
        run_record["tags"] = tags
    manifest = _read_json(manifest_path)
    manifest["runs"].append(run_record)
    _write_json(manifest_path, manifest)


def _prompt_annotations(config: PrelimConfig) -> tuple[str | None, dict[str, str] | None]:
    notes = config.notes
    tags = config.tags

    if notes is None:
        print("\n--- Experiment Annotations ---")
        try:
            raw = input("Notes (free-text description of this run, or blank to skip): ").strip()
            notes = raw if raw else None
        except (EOFError, KeyboardInterrupt):
            notes = None

    if tags is None:
        try:
            raw = input("Tags (key=value pairs separated by spaces, or blank to skip): ").strip()
            if raw:
                parsed: dict[str, str] = {}
                for tag in raw.split():
                    if "=" not in tag:
                        print(f"  Skipping invalid tag: '{tag}' (expected key=value)")
                        continue
                    key, _, value = tag.partition("=")
                    if key:
                        parsed[key] = value
                tags = parsed if parsed else None
        except (EOFError, KeyboardInterrupt):
            tags = None

    return notes, tags


def run_pipeline(config: PrelimConfig, config_path: str | None = None) -> None:
    repo_root = Path(__file__).resolve().parent
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    notes, tags = _prompt_annotations(config)

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "config": asdict(config),
        "notes": notes,
        "tags": tags,
        "runs": [],
    }
    manifest_path = experiment_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    metadata = {
        "notes": notes,
        "tags": tags,
        "experiment_id": experiment_id,
    }
    metadata_path = experiment_dir / "run_metadata.json"
    _write_json(metadata_path, metadata)

    leaderboard_rows: list[dict[str, Any]] = []

    loss_types = config.loss_types if config.loss_types else [None]

    if config.parallel and len(config.models) > 1:
        max_workers = config.max_concurrent if config.max_concurrent is not None else len(config.models)
        model_queue = list(config.models)
        active: list[tuple[str, subprocess.Popen]] = []
        failures: list[tuple[str, int]] = []

        print(f"Parallel mode: {len(model_queue)} models, max {max_workers} concurrent")

        while model_queue or active:
            while model_queue and len(active) < max_workers:
                model = model_queue.pop(0)
                child_cmd = [sys.executable, __file__]
                if config_path is not None:
                    child_cmd += ["--config", config_path]
                child_cmd += ["--models", model, "--no-parallel"]
                if config.skip_training:
                    child_cmd.append("--skip-training")
                if notes is not None:
                    child_cmd += ["--notes", notes]
                if tags is not None:
                    for k, v in tags.items():
                        child_cmd += ["--tags", f"{k}={v}"]
                print(f"  Starting: {model}")
                proc = subprocess.Popen(child_cmd)
                active.append((model, proc))

            still_active: list[tuple[str, subprocess.Popen]] = []
            for model, proc in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((model, proc))
                elif ret != 0:
                    failures.append((model, ret))
                    print(f"  Finished: {model} (exit {ret})")
                else:
                    print(f"  Finished: {model}")
            active = still_active

            if active:
                time.sleep(5)

        if failures:
            msg = "; ".join(f"{m} exit {r}" for m, r in failures)
            raise RuntimeError(f"Parallel pipeline failures: {msg}")
        print("\nAll parallel pipelines completed.", flush=True)
        return

    for idx, model in enumerate(config.models):
        model_slug = _sanitize_name(model)
        model_dir = experiment_dir / model_slug
        baseline_eval_dir = model_dir / "baseline_eval"
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Pipeline for {model} ===")

        baseline_metrics_path = _eval_model(
            python_bin=config.python_bin,
            repo_root=repo_root,
            model_name_or_path=model,
            dataset_name=config.dataset_name,
            split=config.test_split,
            max_samples=config.test_max_samples,
            output_dir=baseline_eval_dir,
            seed=config.seed,
            eval_config=config.eval_config,
            max_new_tokens=config.eval_max_new_tokens,
            repetition_penalty=config.repetition_penalty,
        )
        baseline_metrics = _read_json(baseline_metrics_path)

        if config.skip_training:
            tuned_model_path = str(config.tuned_model_paths[idx])
            tuned_eval_dir = model_dir / "tuned_eval"
            tuned_metrics_path = _eval_model(
                python_bin=config.python_bin,
                repo_root=repo_root,
                model_name_or_path=tuned_model_path,
                dataset_name=config.dataset_name,
                split=config.test_split,
                max_samples=config.test_max_samples,
                output_dir=tuned_eval_dir,
                seed=config.seed,
                eval_config=config.eval_config,
                max_new_tokens=config.eval_max_new_tokens,
                repetition_penalty=config.repetition_penalty,
            )
            tuned_metrics = _read_json(tuned_metrics_path)
            _run_compare(
                config, repo_root, model_dir, model, model_slug,
                baseline_metrics_path, tuned_model_path, tuned_metrics_path,
            )
            _add_leaderboard_row(
                leaderboard_rows, model, model_slug, None,
                baseline_metrics, tuned_metrics, tuned_model_path,
                baseline_metrics_path, tuned_metrics_path,
            )
            _record_run(manifest_path, model, model_slug, None,
                        baseline_metrics_path, tuned_metrics_path,
                        tuned_model_path, model_dir, notes=notes, tags=tags)
            continue

        for lt in loss_types:
            lt_label = lt if lt else "default"
            lt_slug = lt or "default"
            print(f"\n--- {model_slug}  |  loss_type={lt_label} ---")

            train_dir = model_dir / "train" / lt_slug
            tuned_eval_dir = model_dir / "tuned_eval" / lt_slug
            train_dir.mkdir(parents=True, exist_ok=True)
            tuned_eval_dir.mkdir(parents=True, exist_ok=True)

            run_name = f"{config.run_name_prefix}-{model_slug}"
            if lt:
                run_name += f"-{lt}"

            _train_model(
                python_bin=config.python_bin,
                repo_root=repo_root,
                base_model=model,
                dataset_name=config.dataset_name,
                train_split=config.train_split,
                train_max_samples=config.train_max_samples,
                output_dir=train_dir,
                run_name=run_name,
                seed=config.seed,
                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                reward_type=config.reward_type,
                max_completion_length=config.max_completion_length,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                mask_truncated_completions=config.mask_truncated_completions,
                num_generations=config.num_generations,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                train_config=config.train_config,
                use_accelerate=config.use_accelerate,
                accelerate_config=config.accelerate_config,
                notes=notes,
                tags=tags,
                loss_type=lt,
                beta=config.beta,
                num_train_epochs=config.num_train_epochs,
            )
            tuned_model_path = str(train_dir)

            tuned_metrics_path = _eval_model(
                python_bin=config.python_bin,
                repo_root=repo_root,
                model_name_or_path=tuned_model_path,
                dataset_name=config.dataset_name,
                split=config.test_split,
                max_samples=config.test_max_samples,
                output_dir=tuned_eval_dir,
                seed=config.seed,
                eval_config=config.eval_config,
                max_new_tokens=config.eval_max_new_tokens,
                repetition_penalty=config.repetition_penalty,
            )
            tuned_metrics = _read_json(tuned_metrics_path)

            per_model_compare_dir = model_dir / "comparison" / lt_slug
            _run_compare(
                config, repo_root, model_dir, model, model_slug,
                baseline_metrics_path, tuned_model_path, tuned_metrics_path,
                lt_slug,
            )

            _add_leaderboard_row(
                leaderboard_rows, model, model_slug, lt,
                baseline_metrics, tuned_metrics, tuned_model_path,
                baseline_metrics_path, tuned_metrics_path,
            )
            _record_run(manifest_path, model, model_slug, lt,
                        baseline_metrics_path, tuned_metrics_path,
                        tuned_model_path, model_dir, notes=notes, tags=tags)

    fieldnames = ["model", "baseline_accuracy", "tuned_accuracy", "accuracy_delta"]
    if len(loss_types) > 1 or (len(loss_types) == 1 and loss_types[0] is not None):
        fieldnames.insert(1, "loss_type")
    fieldnames += ["baseline_metrics_path", "tuned_metrics_path", "tuned_model_path"]

    leaderboard_path = experiment_dir / "leaderboard.json"
    _write_json(
        leaderboard_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "rows": leaderboard_rows,
        },
    )

    leaderboard_csv_path = experiment_dir / "leaderboard.csv"
    with leaderboard_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leaderboard_rows)

    has_loss_col = "loss_type" in fieldnames
    loss_col = "| Loss type " if has_loss_col else ""
    loss_align = "|---" if has_loss_col else ""
    summary_lines = [
        "# Preliminary GRPO Pipeline Output",
        "",
        f"Experiment: `{experiment_id}`",
        "",
        "## Models",
        "",
        f"| Model {loss_col}| Baseline acc | Tuned acc | Delta |",
        f"|----{loss_align}|---|---:|---:|",
    ]
    for row in leaderboard_rows:
        baseline = row["baseline_accuracy"]
        tuned = row["tuned_accuracy"]
        delta = row["accuracy_delta"]
        baseline_text = "n/a" if baseline is None else f"{float(baseline):.6f}"
        tuned_text = "n/a" if tuned is None else f"{float(tuned):.6f}"
        delta_text = "n/a" if delta is None else f"{float(delta):+.6f}"
        if has_loss_col:
            lt = row.get("loss_type", "n/a")
            summary_lines.append(
                f"| {row['model']} | {lt} | {baseline_text} | {tuned_text} | {delta_text} |"
            )
        else:
            summary_lines.append(
                f"| {row['model']} | {baseline_text} | {tuned_text} | {delta_text} |"
            )

    summary_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Leaderboard: `{leaderboard_path}`",
            f"- Leaderboard CSV: `{leaderboard_csv_path}`",
            f"- Per-model outputs: `{experiment_dir}`",
            "- Each model directory includes `baseline_eval/`, `train/<loss_type>/`, `tuned_eval/<loss_type>/`, and `comparison/<loss_type>/`",
        ]
    )

    summary_path = experiment_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))
        handle.write("\n")

    print("\nPipeline complete.")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    config = merge_config(PrelimConfig(), args)

    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent)

    run_pipeline(config, config_path=args.config)


if __name__ == "__main__":
    main()
