#set page(paper: "a4", margin: (top: 2.5cm, bottom: 2.5cm, left: 3cm, right: 3cm))
#set text(font: ("IBM Plex Serif", "Linux Libertine", "Serif"), size: 11pt, fill: rgb("#1a1a1a"))
#set par(justify: true, leading: 0.75em)
#show heading.where(level: 1): it => block(above: 2em, below: 1em)[#set text(size: 1.6em, weight: 700, fill: rgb("#111111")); #it.body]
#show heading.where(level: 2): it => block(above: 1.5em, below: 0.75em)[#set text(size: 1.2em, weight: 600, fill: rgb("#222222")); #it.body]
#show heading.where(level: 3): it => block(above: 1.2em, below: 0.5em)[#set text(size: 1em, weight: 600, fill: rgb("#444444")); #it.body]
#show link: it => text(fill: rgb("#2563a8"))[#underline(it)]
#show raw: it => text(font: ("IBM Plex Mono", "Courier", "Monospace"), size: 0.9em, fill: rgb("#1a1a1a"))[#it]
#show raw.where(block: true): it => block(fill: rgb("#f4f4f4"), inset: 1em, radius: 4pt, width: 100%)[#it]
#show quote.where(block: true): it => pad(left: 1.5em)[#block(stroke: (left: 3pt + rgb("#cccccc")), inset: (left: 1em))[#text(style: "italic", fill: rgb("#555555"))[#it.body]]]
#show figure.where(kind: raw): it => align(left, it.body)

= Codebase Architecture Guide

This document outlines the architecture and organization of the GRPO reinforcement learning pipeline, optimized for Modal GPU environments. 

== Core Execution

The repository is built to seamlessly transition from local debugging to distributed parallel execution on Modal GPUs. 

- `modal_train.py`: The primary entry point for Modal execution. This script orchestrates container builds, sets up volume mounts, and coordinates the pipeline module over a swept configuration matrix.
- `train_grpo.py`: Core TRL training script using Group Relative Policy Optimization. Contains the Hugging Face `SFTTrainer`/`GRPOTrainer` hooks.
- `eval_grpo.py`: Standardized evaluation pipeline for post-training inference benchmarks.
- `reward_fns.py`: Implements the distinct reinforcement learning reward algorithms (e.g., format matching, mathematical accuracy).

== Pipeline Orchestration (`pipeline/`)

The `pipeline` module abstracts over the complex matrix executions previously bound inside large monolithic scripts.

- `pipeline/runner.py`: Handles child process coordination, passing appropriate hardware arguments (like Accelerate configurations) down to `train_grpo.py` and `eval_grpo.py`. It is invoked natively by `modal_train.py` or through the wrapper `run_prelim.py`.
- `pipeline/config.py`: Exposes a robust dataclass and YAML merger for configuring models, datasets, bounds, and hardware overrides.
- `pipeline/reporter.py`: Consolidates leaderboards, metrics, and comparisons between baseline models and tuned variants. It generates Markdown and CSV artifacts for immediate observation. 

== Visual Analytics (`ui/`)

The Streamlit interface allows users to review the sweeping output and interactively test trained models.

- `ui/app.py`: The frontend UI, abstracting Streamlit layouts.
- `ui/core.py`: The backend logic module covering HF model loading, inference logic, and historical `run_metadata.json` / leaderboard loading.

== Supporting Subsystems

- `configs/`: Houses YAML definition files dictating training hyper-parameters and pipeline models. The `sweeps/` sub-directory isolates hyper-parameter permutations.
- `scripts/`: Utilities for the platform, notably `healthcheck.py` to ensure PyTorch, CUDA, and environment compatibility before launching deep training loops. 
- `archive/`: Historical environment deployments (like Runpod scripts and guides) kept for lineage.

