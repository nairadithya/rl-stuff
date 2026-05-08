from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from tqdm.auto import tqdm

from reward_fns import format_reward, robust_accuracy_reward


@dataclass
class MlxEvalConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_name_or_path: str | None = None
    dataset_name: str = "trl-lib/DeepMath-103K"
    dataset_split: str = "test"
    max_samples: int | None = 500
    max_new_tokens: int = 192
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    batch_size: int = 1
    seed: int = 42
    output_dir: str = "outputs/eval-prelim-mlx"
    log_every: int = 25
    show_progress_bar: bool = True
    fallback_to_transformers: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-training evaluation using MLX when available."
    )
    parser.add_argument("--config", default=None, help="YAML config file path.")
    parser.add_argument(
        "--model-name-or-path", default=None, help="Model or checkpoint path."
    )
    parser.add_argument(
        "--tokenizer-name-or-path", default=None, help="Optional tokenizer path."
    )
    parser.add_argument("--dataset-name", default=None, help="HF dataset name.")
    parser.add_argument("--dataset-split", default=None, help="Dataset split.")
    parser.add_argument("--max-samples", type=int, default=None, help="Sample cap.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Max generated tokens."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature."
    )
    parser.add_argument(
        "--top-p", type=float, default=None, help="Top-p nucleus sampling."
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=None, help="Repetition penalty."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Unused; accepted for CLI compatibility.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument(
        "--log-every", type=int, default=None, help="Periodic log interval."
    )
    parser.add_argument(
        "--no-progress-bar", action="store_true", help="Disable progress bar."
    )
    parser.add_argument(
        "--no-fallback-to-transformers",
        action="store_true",
        help="Fail if MLX backend is unavailable.",
    )
    return parser.parse_args()


def _load_yaml(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return data


def merge_config(defaults: MlxEvalConfig, args: argparse.Namespace) -> MlxEvalConfig:
    data = asdict(defaults)
    if args.config is not None:
        file_overrides = _load_yaml(args.config)
        unknown = [key for key in file_overrides if key not in data]
        if unknown:
            raise ValueError(
                f"Unknown keys in config file: {', '.join(sorted(unknown))}"
            )
        data.update(file_overrides)

    overrides = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "log_every": args.log_every,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if args.no_progress_bar:
        data["show_progress_bar"] = False
    if args.no_fallback_to_transformers:
        data["fallback_to_transformers"] = False

    if data["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if data["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")
    if data["temperature"] < 0:
        raise ValueError("temperature must be >= 0")
    if not (0 < data["top_p"] <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if data["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty must be > 0")
    if data["log_every"] <= 0:
        raise ValueError("log_every must be > 0")

    return MlxEvalConfig(**data)


def _format_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                else:
                    parts.append(f"{role}: {content}")
        if not parts:
            return ""
        return "\n".join(parts) + "\nAssistant:"
    return str(prompt)


def _evaluate_with_transformers_fallback(
    config: MlxEvalConfig, reason: str
) -> dict[str, Any]:
    if not config.fallback_to_transformers:
        raise RuntimeError(
            f"MLX backend unavailable and fallback disabled. Reason: {reason}"
        )

    print(f"MLX backend unavailable; falling back to transformers. Reason: {reason}")
    from eval_grpo import EvalConfig, evaluate

    tf_config = EvalConfig(
        model_name_or_path=config.model_name_or_path,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        dataset_name=config.dataset_name,
        dataset_split=config.dataset_split,
        max_samples=config.max_samples,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        batch_size=1,
        seed=config.seed,
        output_dir=config.output_dir,
        log_every=config.log_every,
        show_progress_bar=config.show_progress_bar,
    )
    metrics = evaluate(tf_config)
    metrics_path = Path(config.output_dir) / "metrics.json"
    if metrics_path.exists():
        on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
        on_disk["backend"] = "transformers_fallback"
        on_disk["backend_note"] = reason
        metrics_path.write_text(
            json.dumps(on_disk, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metrics = on_disk
    return metrics


def evaluate(config: MlxEvalConfig) -> dict[str, Any]:
    mlx_spec = importlib.util.find_spec("mlx_lm")
    if mlx_spec is None:
        return _evaluate_with_transformers_fallback(
            config, "mlx_lm package not installed"
        )

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    def mlx_generate_text(model: Any, tokenizer: Any, prompt_text: str) -> str:
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=config.repetition_penalty,
        )

        attempts = [
            lambda: generate(
                model,
                tokenizer,
                prompt=prompt_text,
                max_tokens=config.max_new_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False,
            ),
            lambda: generate(
                model,
                tokenizer,
                prompt_text,
                max_tokens=config.max_new_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False,
            ),
            lambda: generate(
                model,
                tokenizer,
                prompt_text,
                max_tokens=config.max_new_tokens,
                sampler=sampler,
                verbose=False,
            ),
        ]

        last_exc: Exception | None = None
        for attempt in attempts:
            try:
                return str(attempt())
            except TypeError as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to generate with MLX.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    if config.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))

    required_columns = {"prompt", "solution"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}. "
            f"Found columns: {dataset.column_names}"
        )

    examples = list(dataset)
    total_examples = len(examples)

    print(
        "Starting MLX evaluation "
        f"(model={config.model_name_or_path}, dataset={config.dataset_name}:{config.dataset_split}, "
        f"examples={total_examples})",
        flush=True,
    )

    try:
        model, tokenizer = load(config.model_name_or_path)
    except Exception as exc:
        return _evaluate_with_transformers_fallback(config, f"mlx load failed: {exc}")

    predictions: list[dict[str, Any]] = []
    all_accuracy: list[float] = []
    all_format: list[float] = []
    all_lengths: list[int] = []
    all_truncated: list[bool] = []

    start_time = time.time()
    next_log_at = min(config.log_every, total_examples) if total_examples else 0

    iterator = enumerate(examples)
    if config.show_progress_bar:
        iterator = tqdm(
            iterator, total=total_examples, desc="Evaluating (MLX)", unit="example"
        )

    for idx, example in iterator:
        prompt_text = _format_prompt(example["prompt"])
        try:
            completion_text = mlx_generate_text(model, tokenizer, prompt_text)
        except Exception as exc:
            if idx == 0 and config.fallback_to_transformers:
                return _evaluate_with_transformers_fallback(
                    config,
                    f"mlx generate failed: {exc}",
                )
            raise
        completion_text = completion_text.strip()

        completion_ids = tokenizer.encode(completion_text)
        completion_messages = [[{"role": "assistant", "content": completion_text}]]
        solutions = [str(example["solution"])]

        accuracy_value = float(
            robust_accuracy_reward(completions=completion_messages, solution=solutions)[
                0
            ]
        )
        format_value = float(format_reward(completions=completion_messages)[0])

        token_len = len(completion_ids)
        truncated = token_len >= config.max_new_tokens

        all_accuracy.append(accuracy_value)
        all_format.append(format_value)
        all_lengths.append(token_len)
        all_truncated.append(truncated)

        predictions.append(
            {
                "index": idx,
                "prompt": example["prompt"],
                "solution": example["solution"],
                "completion": completion_text,
                "accuracy_reward": accuracy_value,
                "format_reward": format_value,
                "completion_tokens": token_len,
                "truncated": truncated,
            }
        )

        if next_log_at and (idx + 1) >= next_log_at:
            elapsed = max(time.time() - start_time, 1e-8)
            mean_acc = sum(all_accuracy) / len(all_accuracy)
            mean_format = sum(all_format) / len(all_format)
            print(
                "Progress: "
                f"{idx + 1}/{total_examples} examples, "
                f"acc={mean_acc:.4f}, format_rate={mean_format:.4f}, "
                f"{(idx + 1) / elapsed:.2f} ex/s",
                flush=True,
            )
            while next_log_at and next_log_at <= (idx + 1):
                next_log_at += config.log_every

    elapsed_seconds = time.time() - start_time
    metrics = {
        "backend": "mlx",
        "model_name_or_path": config.model_name_or_path,
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "num_examples": len(predictions),
        "accuracy_mean": sum(all_accuracy) / len(all_accuracy)
        if all_accuracy
        else None,
        "accuracy_valid_examples": len(all_accuracy),
        "accuracy_valid_fraction": 1.0 if predictions else 0.0,
        "format_rate": sum(all_format) / len(all_format) if all_format else 0.0,
        "avg_completion_tokens": sum(all_lengths) / len(all_lengths)
        if all_lengths
        else 0.0,
        "median_completion_tokens": statistics.median(all_lengths)
        if all_lengths
        else 0.0,
        "truncation_rate": (
            sum(1.0 for flag in all_truncated if flag) / len(all_truncated)
            if all_truncated
            else 0.0
        ),
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": (
            len(predictions) / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "seed": config.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(
        "Evaluation complete "
        f"(backend=mlx, accuracy={metrics['accuracy_mean']}, "
        f"format_rate={metrics['format_rate']:.4f}, "
        f"truncation_rate={metrics['truncation_rate']:.4f}, "
        f"n={metrics['num_examples']})."
    )

    return metrics


def main() -> None:
    args = parse_args()
    config = merge_config(MlxEvalConfig(), args)
    evaluate(config)


if __name__ == "__main__":
    main()
