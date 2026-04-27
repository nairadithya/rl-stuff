from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ui.inference import GenerationConfig, generate_text, load_model_bundle
from ui.results_loader import (
    RunArtifacts,
    discover_model_options,
    list_experiments,
    load_experiment_leaderboard,
    load_metrics,
    load_predictions,
    load_runs,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "outputs" / "prelim"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _render_metrics(title: str, metrics: dict[str, Any] | None) -> None:
    st.markdown(f"#### {title}")
    if metrics is None:
        st.info("No metrics found for this run.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", _fmt(metrics.get("accuracy_mean")))
    c2.metric("Format rate", _fmt(metrics.get("format_rate")))
    c3.metric("Truncation rate", _fmt(metrics.get("truncation_rate")))
    c4.metric("Avg tokens", _fmt(metrics.get("avg_completion_tokens"), digits=1))

    c5, c6, c7 = st.columns(3)
    c5.metric("Examples", str(metrics.get("num_examples", "n/a")))
    c6.metric("Ex/s", _fmt(metrics.get("examples_per_second"), digits=2))
    c7.metric("Backend", str(metrics.get("backend", "n/a")))


def _prediction_view_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for row in rows:
        formatted.append(
            {
                "index": row.get("index"),
                "prompt": row.get("prompt"),
                "completion": row.get("completion"),
                "accuracy_reward": row.get("accuracy_reward"),
                "format_reward": row.get("format_reward"),
                "completion_tokens": row.get("completion_tokens"),
                "truncated": row.get("truncated"),
            }
        )
    return formatted


def _show_run_results(run: RunArtifacts) -> None:
    baseline_metrics = load_metrics(run.baseline_metrics_path)
    tuned_metrics = load_metrics(run.tuned_metrics_path)

    left, right = st.columns(2)
    with left:
        _render_metrics("Baseline", baseline_metrics)
    with right:
        _render_metrics("Tuned", tuned_metrics)

    max_rows = st.slider(
        "Prediction rows", min_value=10, max_value=200, value=40, step=10
    )
    baseline_predictions = load_predictions(
        run.baseline_predictions_path, limit=max_rows
    )
    tuned_predictions = load_predictions(run.tuned_predictions_path, limit=max_rows)

    tab1, tab2 = st.tabs(["Baseline predictions", "Tuned predictions"])
    with tab1:
        if baseline_predictions:
            st.dataframe(
                _prediction_view_rows(baseline_predictions), use_container_width=True
            )
        else:
            st.info("No baseline predictions found.")
    with tab2:
        if tuned_predictions:
            st.dataframe(
                _prediction_view_rows(tuned_predictions), use_container_width=True
            )
        else:
            st.info("No tuned predictions found.")


@st.cache_resource(show_spinner=False)
def _cached_model_bundle(model_name_or_path: str) -> dict[str, Any]:
    return load_model_bundle(model_name_or_path)


def _prompt_panel() -> None:
    st.markdown("### Prompt Playground")

    model_options = discover_model_options(REPO_ROOT)
    selected_model = st.selectbox("Model", model_options)
    custom_model = st.text_input(
        "Custom model path (optional)",
        value="",
        placeholder="outputs/grpo-qwen2.5-0.5b",
    ).strip()
    model_name_or_path = custom_model or selected_model

    c1, c2, c3, c4 = st.columns(4)
    max_new_tokens = c1.number_input(
        "Max new tokens", min_value=1, max_value=2048, value=256
    )
    temperature = c2.number_input(
        "Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.05
    )
    top_p = c3.number_input(
        "Top-p", min_value=0.01, max_value=1.0, value=0.95, step=0.01
    )
    repetition_penalty = c4.number_input(
        "Repetition penalty", min_value=0.5, max_value=2.0, value=1.05, step=0.01
    )

    prompt = st.text_area(
        "Prompt",
        value="Solve: If x + 4 = 9, what is x? Return final answer in \\boxed{}.",
        height=150,
    )

    if st.button("Generate", type="primary"):
        if not prompt.strip():
            st.error("Prompt cannot be empty.")
            return

        config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
        )

        try:
            with st.spinner(f"Loading {model_name_or_path}..."):
                model_bundle = _cached_model_bundle(model_name_or_path)
            with st.spinner("Generating..."):
                completion = generate_text(
                    model_bundle=model_bundle, prompt=prompt, config=config
                )
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            return

        st.success("Done")
        st.caption(
            f"Backend: {model_bundle.get('backend', 'unknown')} | Device: {model_bundle.get('device', 'unknown')}"
        )
        st.markdown("#### Completion")
        st.code(completion)


def main() -> None:
    st.set_page_config(page_title="Model Testing UI", layout="wide")
    st.title("GRPO Model Testing UI")
    st.caption("Inspect eval artifacts and prompt candidate models from one page.")

    experiments = list_experiments(OUTPUT_ROOT)
    experiment_map = {exp.name: exp for exp in experiments}

    with st.sidebar:
        st.markdown("### Results Source")
        st.text(f"{OUTPUT_ROOT}")
        if not experiments:
            st.warning("No experiments found under outputs/prelim.")
            selected_experiment = None
        else:
            selected_name = st.selectbox("Experiment", list(experiment_map.keys()))
            selected_experiment = experiment_map[selected_name]

    top_left, top_right = st.columns([3, 2])

    with top_left:
        st.markdown("### Run Results")
        if selected_experiment is None:
            st.info("Run `./scripts/run_prelim.sh` first to generate results.")
        else:
            runs = load_runs(repo_root=REPO_ROOT, experiment_dir=selected_experiment)
            if not runs:
                st.info("No run entries found in manifest for this experiment.")
            else:
                run_labels = {f"{run.model} [{run.model_slug}]": run for run in runs}
                selected_run_label = st.selectbox("Model run", list(run_labels.keys()))
                _show_run_results(run_labels[selected_run_label])

            leaderboard_rows = load_experiment_leaderboard(selected_experiment)
            st.markdown("#### Leaderboard")
            if leaderboard_rows:
                st.dataframe(leaderboard_rows, use_container_width=True)
            else:
                st.info("No leaderboard found for this experiment.")

    with top_right:
        _prompt_panel()


if __name__ == "__main__":
    main()
