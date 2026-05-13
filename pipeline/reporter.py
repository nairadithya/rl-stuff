from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ModelEval:
    label: str
    model_name_or_path: str
    metrics_path: Path
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and GRPO-tuned evaluation results."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model spec in format label:model_name_or_path:metrics_json_path. "
            "Pass multiple times."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for comparison artifacts.",
    )
    return parser.parse_args()


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Metrics file must contain an object: {path}")
    return data


def _parse_model_spec(spec: str) -> ModelEval:
    parts = spec.split(":", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(
            "Invalid --model spec. Expected label:model_name_or_path:metrics_json_path"
        )

    label, model_name_or_path, metrics_path_raw = parts
    if not label:
        raise ValueError("Model label cannot be empty.")
    if not model_name_or_path:
        raise ValueError("Model name/path cannot be empty.")

    metrics_path = Path(metrics_path_raw)
    metrics = _load_metrics(metrics_path)

    return ModelEval(
        label=label,
        model_name_or_path=model_name_or_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _find_reference(models: list[ModelEval]) -> ModelEval:
    for model in models:
        if model.label.lower() == "base":
            return model
    return models[0]


def _delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def _format_value(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_signed(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def compare(models: list[ModelEval], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = _find_reference(models)
    ref_accuracy = _as_float(reference.metrics.get("accuracy_mean"))
    ref_format = _as_float(reference.metrics.get("format_rate"))
    ref_len = _as_float(reference.metrics.get("avg_completion_tokens"))

    rows: list[dict[str, Any]] = []
    for model in models:
        accuracy = _as_float(model.metrics.get("accuracy_mean"))
        format_rate = _as_float(model.metrics.get("format_rate"))
        avg_tokens = _as_float(model.metrics.get("avg_completion_tokens"))

        rows.append(
            {
                "label": model.label,
                "model_name_or_path": model.model_name_or_path,
                "metrics_path": str(model.metrics_path),
                "num_examples": int(model.metrics.get("num_examples", 0)),
                "accuracy_mean": accuracy,
                "format_rate": format_rate,
                "avg_completion_tokens": avg_tokens,
                "accuracy_delta_vs_reference": _delta(accuracy, ref_accuracy),
                "format_delta_vs_reference": _delta(format_rate, ref_format),
                "tokens_delta_vs_reference": _delta(avg_tokens, ref_len),
            }
        )

    comparison = {
        "reference_label": reference.label,
        "reference_model_name_or_path": reference.model_name_or_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": rows,
    }

    json_path = output_dir / "comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")

    md_path = output_dir / "summary.md"
    md_lines = [
        "# GRPO Preliminary Results",
        "",
        f"Reference model: `{reference.label}` ({reference.model_name_or_path})",
        "",
        "| Model | Accuracy | Delta vs ref | Format rate | Delta vs ref | Avg completion tokens | Delta vs ref | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            f"{row['label']} | "
            f"{_format_value(row['accuracy_mean'])} | "
            f"{_format_signed(row['accuracy_delta_vs_reference'])} | "
            f"{_format_value(row['format_rate'])} | "
            f"{_format_signed(row['format_delta_vs_reference'])} | "
            f"{_format_value(row['avg_completion_tokens'])} | "
            f"{_format_signed(row['tokens_delta_vs_reference'])} | "
            f"{row['num_examples']} |"
        )
    md_lines.append("")
    md_lines.append("## Inputs")
    md_lines.append("")
    for model in models:
        md_lines.append(
            f"- `{model.label}`: model=`{model.model_name_or_path}`, metrics=`{model.metrics_path}`"
        )

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines))
        handle.write("\n")

    csv_path = output_dir / "leaderboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "model_name_or_path",
                "num_examples",
                "accuracy_mean",
                "accuracy_delta_vs_reference",
                "format_rate",
                "format_delta_vs_reference",
                "avg_completion_tokens",
                "tokens_delta_vs_reference",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "model_name_or_path": row["model_name_or_path"],
                    "num_examples": row["num_examples"],
                    "accuracy_mean": _format_value(row["accuracy_mean"]),
                    "accuracy_delta_vs_reference": _format_signed(
                        row["accuracy_delta_vs_reference"]
                    ),
                    "format_rate": _format_value(row["format_rate"]),
                    "format_delta_vs_reference": _format_signed(
                        row["format_delta_vs_reference"]
                    ),
                    "avg_completion_tokens": _format_value(
                        row["avg_completion_tokens"]
                    ),
                    "tokens_delta_vs_reference": _format_signed(
                        row["tokens_delta_vs_reference"]
                    ),
                }
            )

    print(f"Wrote comparison JSON: {json_path}")
    print(f"Wrote summary markdown: {md_path}")
    print(f"Wrote leaderboard CSV: {csv_path}")

    return comparison


def main() -> None:
    args = parse_args()
    models = [_parse_model_spec(spec) for spec in args.model]
    if len(models) < 2:
        raise ValueError("Provide at least two --model entries to compare runs.")
    compare(models=models, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
