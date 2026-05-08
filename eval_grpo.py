from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, PeftConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from reward_fns import format_reward, robust_accuracy_reward


@dataclass
class EvalConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_name_or_path: str | None = None
    dataset_name: str = "trl-lib/DeepMath-103K"
    dataset_split: str = "test"
    max_samples: int | None = 500
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    batch_size: int = 1
    log_every: int = 25
    show_progress_bar: bool = True
    seed: int = 42
    output_dir: str = "outputs/eval-prelim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-training GRPO evaluation.")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config file path (values overridden by explicit CLI flags).",
    )
    parser.add_argument(
        "--model-name-or-path", default=None, help="Model or checkpoint path."
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        default=None,
        help="Optional tokenizer source. If omitted, inferred from model/checkpoint.",
    )
    parser.add_argument("--dataset-name", default=None, help="HF dataset name.")
    parser.add_argument("--dataset-split", default=None, help="Dataset split.")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Optional dataset size cap."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Max generated tokens."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Set 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p", type=float, default=None, help="Nucleus sampling parameter."
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty during generation.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Evaluation batch size."
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Print periodic progress every N evaluated examples.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir", default=None, help="Directory for metrics and predictions."
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


def merge_config(defaults: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    data = asdict(defaults)

    if args.config is not None:
        file_overrides = load_config_file(args.config)
        unknown = [key for key in file_overrides if key not in data]
        if unknown:
            unknown_display = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown keys in config file: {unknown_display}")
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
        "log_every": args.log_every,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }

    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if data["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")
    if data["log_every"] <= 0:
        raise ValueError("log_every must be > 0")
    if data["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if data["temperature"] < 0:
        raise ValueError("temperature must be >= 0")
    if not (0 < data["top_p"] <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if data["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty must be > 0")

    if args.no_progress_bar:
        data["show_progress_bar"] = False

    return EvalConfig(**data)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _as_python_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _batch_iter(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _format_prompt(prompt: Any, tokenizer: AutoTokenizer) -> str:
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        try:
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            parts = []
            for message in prompt:
                if isinstance(message, dict):
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    parts.append(f"{role}: {content}")
            return "\n".join(parts) + "\nassistant:"

    return str(prompt)


def _load_model(model_name_or_path: str, device: str, dtype: torch.dtype):
    load_kwargs: dict[str, Any] = {"dtype": dtype}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

    if device in {"cpu", "mps"}:
        model = model.to(device)

    model.eval()
    return model


def _resolve_tokenizer_source(model_name_or_path: str, explicit: str | None) -> str:
    if explicit is not None:
        return explicit

    try:
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        base_model_name_or_path = peft_config.base_model_name_or_path
        if not base_model_name_or_path:
            raise ValueError(
                "Could not resolve base model from adapter checkpoint. "
                "Pass --tokenizer-name-or-path explicitly."
            )
        return str(base_model_name_or_path)
    except Exception:
        pass

    return model_name_or_path


def evaluate(config: EvalConfig) -> dict[str, Any]:
    set_seed(config.seed)

    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    if config.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))

    required_columns = {"prompt", "solution"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing_display}. "
            f"Found columns: {dataset.column_names}"
        )

    examples = list(dataset)
    total_examples = len(examples)

    device = resolve_device()
    dtype = resolve_dtype(device)

    tokenizer_source = _resolve_tokenizer_source(
        model_name_or_path=config.model_name_or_path,
        explicit=config.tokenizer_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id or pad_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(config.model_name_or_path, device=device, dtype=dtype)

    print(
        "Starting evaluation "
        f"(model={config.model_name_or_path}, "
        f"dataset={config.dataset_name}:{config.dataset_split}, "
        f"examples={total_examples}, "
        f"device={device}, dtype={str(dtype).replace('torch.', '')})",
        flush=True,
    )

    predictions: list[dict[str, Any]] = []
    all_accuracy: list[float | None] = []
    all_format: list[float] = []
    all_lengths: list[int] = []
    all_truncated: list[bool] = []

    do_sample = config.temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "top_k": None,
        "repetition_penalty": config.repetition_penalty,
    }
    if do_sample:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    start_time = time.time()
    next_log_at = min(config.log_every, total_examples) if total_examples else 0
    processed_examples = 0

    batch_iterator = _batch_iter(examples, config.batch_size)
    total_batches = (
        math.ceil(total_examples / config.batch_size) if total_examples else 0
    )
    if config.show_progress_bar:
        batch_iterator = tqdm(
            batch_iterator,
            total=total_batches,
            desc="Evaluating",
            unit="batch",
        )

    for start_index, batch in batch_iterator:
        batch_prompts = [
            _format_prompt(example["prompt"], tokenizer) for example in batch
        ]
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        with torch.no_grad():
            generated = model.generate(**tokenized, **generation_kwargs)

        completion_ids_batch: list[list[int]] = []
        completion_text_batch: list[str] = []
        for row_index in range(generated.shape[0]):
            input_length = int(tokenized["attention_mask"][row_index].sum().item())
            completion_ids = generated[row_index, input_length:]
            completion_ids_list = completion_ids.tolist()
            completion_ids_batch.append(completion_ids_list)
            completion_text_batch.append(
                tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            )

        completion_messages = [
            [{"role": "assistant", "content": text}] for text in completion_text_batch
        ]
        solutions = [str(example["solution"]) for example in batch]

        accuracy_scores = robust_accuracy_reward(
            completions=completion_messages,
            solution=solutions,
        )
        format_scores = format_reward(completions=completion_messages)

        for offset, example in enumerate(batch):
            completion_ids = completion_ids_batch[offset]
            completion_length = len(completion_ids)
            ended_with_eos = (
                completion_length > 0
                and tokenizer.eos_token_id is not None
                and completion_ids[-1] == tokenizer.eos_token_id
            )
            truncated = (
                completion_length >= config.max_new_tokens and not ended_with_eos
            )

            accuracy_value = _as_python_float(accuracy_scores[offset])
            format_value = float(format_scores[offset])

            all_accuracy.append(accuracy_value)
            all_format.append(format_value)
            all_lengths.append(completion_length)
            all_truncated.append(truncated)

            predictions.append(
                {
                    "index": start_index + offset,
                    "prompt": example["prompt"],
                    "solution": example["solution"],
                    "completion": completion_text_batch[offset],
                    "accuracy_reward": accuracy_value,
                    "format_reward": format_value,
                    "completion_tokens": completion_length,
                    "truncated": truncated,
                }
            )

        processed_examples += len(batch)
        if config.show_progress_bar and hasattr(batch_iterator, "set_postfix"):
            current_accuracy = [v for v in all_accuracy if v is not None]
            mean_acc = (
                sum(current_accuracy) / len(current_accuracy)
                if current_accuracy
                else None
            )
            postfix = {
                "examples": processed_examples,
                "acc": f"{mean_acc:.4f}" if mean_acc is not None else "n/a",
            }
            batch_iterator.set_postfix(postfix, refresh=False)

        if next_log_at and processed_examples >= next_log_at:
            elapsed = max(time.time() - start_time, 1e-8)
            current_accuracy = [v for v in all_accuracy if v is not None]
            mean_acc = (
                sum(current_accuracy) / len(current_accuracy)
                if current_accuracy
                else None
            )
            mean_format = sum(all_format) / len(all_format) if all_format else 0.0
            print(
                "Progress: "
                f"{processed_examples}/{total_examples} examples, "
                f"acc={mean_acc if mean_acc is not None else 'n/a'}, "
                f"format_rate={mean_format:.4f}, "
                f"{processed_examples / elapsed:.2f} ex/s",
                flush=True,
            )
            while next_log_at and next_log_at <= processed_examples:
                next_log_at += config.log_every

    elapsed_seconds = time.time() - start_time

    valid_accuracy = [value for value in all_accuracy if value is not None]
    metrics = {
        "backend": "transformers",
        "model_name_or_path": config.model_name_or_path,
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "num_examples": len(predictions),
        "accuracy_mean": (
            sum(valid_accuracy) / len(valid_accuracy) if valid_accuracy else None
        ),
        "accuracy_valid_examples": len(valid_accuracy),
        "accuracy_valid_fraction": (
            len(valid_accuracy) / len(predictions) if predictions else 0.0
        ),
        "format_rate": sum(all_format) / len(all_format) if all_format else 0.0,
        "avg_completion_tokens": (
            sum(all_lengths) / len(all_lengths) if all_lengths else 0.0
        ),
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
        "batch_size": config.batch_size,
        "seed": config.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(
        "Evaluation complete "
        f"(accuracy={metrics['accuracy_mean']}, "
        f"format_rate={metrics['format_rate']:.4f}, "
        f"truncation_rate={metrics['truncation_rate']:.4f}, "
        f"n={metrics['num_examples']})."
    )

    return metrics


def main() -> None:
    args = parse_args()
    config = merge_config(EvalConfig(), args)
    evaluate(config)


if __name__ == "__main__":
    main()
