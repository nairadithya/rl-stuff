from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunArtifacts:
    model: str
    model_slug: str
    baseline_metrics_path: Path | None
    tuned_metrics_path: Path | None
    baseline_predictions_path: Path | None
    tuned_predictions_path: Path | None
    tuned_model_path: str | None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_rows(path: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_path(repo_root: Path, path_like: str | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def list_experiments(output_root: Path) -> list[Path]:
    if not output_root.exists() or not output_root.is_dir():
        return []
    return sorted(
        [path for path in output_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )


def load_experiment_manifest(experiment_dir: Path) -> dict[str, Any] | None:
    return _read_json(experiment_dir / "manifest.json")


def load_experiment_leaderboard(experiment_dir: Path) -> list[dict[str, Any]]:
    leaderboard_json = _read_json(experiment_dir / "leaderboard.json")
    if leaderboard_json and isinstance(leaderboard_json.get("rows"), list):
        rows = leaderboard_json["rows"]
        return [row for row in rows if isinstance(row, dict)]

    return _read_csv_rows(experiment_dir / "leaderboard.csv")


def load_runs(repo_root: Path, experiment_dir: Path) -> list[RunArtifacts]:
    manifest = load_experiment_manifest(experiment_dir)
    if not manifest:
        return []

    runs_raw = manifest.get("runs", [])
    if not isinstance(runs_raw, list):
        return []

    runs: list[RunArtifacts] = []
    for run in runs_raw:
        if not isinstance(run, dict):
            continue

        model = str(run.get("model", "unknown-model"))
        model_slug = str(run.get("model_slug", model))
        baseline_metrics_path = _resolve_path(repo_root, run.get("baseline_metrics"))
        tuned_metrics_path = _resolve_path(repo_root, run.get("tuned_metrics"))

        baseline_predictions_path = (
            baseline_metrics_path.parent / "predictions.jsonl"
            if baseline_metrics_path is not None
            else None
        )
        tuned_predictions_path = (
            tuned_metrics_path.parent / "predictions.jsonl"
            if tuned_metrics_path is not None
            else None
        )

        runs.append(
            RunArtifacts(
                model=model,
                model_slug=model_slug,
                baseline_metrics_path=baseline_metrics_path,
                tuned_metrics_path=tuned_metrics_path,
                baseline_predictions_path=baseline_predictions_path,
                tuned_predictions_path=tuned_predictions_path,
                tuned_model_path=run.get("tuned_model_path"),
            )
        )

    return runs


def load_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _read_json(path)


def load_predictions(path: Path | None, limit: int = 100) -> list[dict[str, Any]]:
    if path is None:
        return []
    return _read_jsonl_rows(path=path, limit=limit)


def discover_model_options(repo_root: Path) -> list[str]:
    options = {
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    }

    outputs_dir = repo_root / "outputs"
    if outputs_dir.exists() and outputs_dir.is_dir():
        for path in outputs_dir.glob("grpo-*"):
            if path.is_dir():
                options.add(str(path))
        for path in outputs_dir.glob("prelim/*/*/train"):
            if path.is_dir():
                options.add(str(path))

    return sorted(options)
