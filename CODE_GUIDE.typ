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

= Codebase Guide

This document provides a comprehensive, technical walkthrough of the GRPO Small LM reinforcement learning repository. This repository is built to scale from local debugging on CPUs/single GPUs to massively parallel hyperparameter sweeps using Modal's serverless infrastructure.

== 1. Directory Structure Overview

```text
.
├── archive/              # Historic deployment scripts (e.g., Runpod) kept for reference
├── configs/              # YAML configuration files for pipeline and training
│   └── sweeps/           # Matrix execution configurations for hyperparameter search
├── pipeline/             # Automated orchestration for sweeps and metrics extraction
├── scripts/              # Utility bash and Python scripts (UI launching, healthchecks)
├── ui/                   # Streamlit analytics and inference playground
├── compare_runs.py       # Pipeline wrapper: execution entry point for comparisons
├── eval_grpo.py          # Core evaluation script (generates metrics.json)
├── grpo_config.py        # Configuration schemas specific to GRPO/TRL
├── modal_train.py        # Modal serverless deployment and distributed runner
├── reward_fns.py         # Reinforcement learning reward functions
├── run_prelim.py         # Pipeline wrapper: execution entry point for full train+eval loop
└── train_grpo.py         # Core Hugging Face TRL GRPOTrainer script
```

== 2. Core Reinforcement Learning (RL) Pipeline

The primary mechanics of the GRPO (Group Relative Policy Optimization) algorithm are contained in a few flat scripts at the root level. These scripts can be run entirely standalone without the orchestration pipeline.

- *`train_grpo.py`*: The heart of the training loop. It parses arguments, sets up Hugging Face Accelerate/PEFT (LoRA by default for small models), initializes the datasets, and passes `reward_fns` to the `GRPOTrainer` from the `trl` library.
- *`reward_fns.py`*: Contains the reward calculations that guide the policy model. For example, it might contain functions that check if a generated mathematical answer exactly matches a gold standard (`accuracy_reward`) or if the formatting follows a specific `<think>` -> `<answer>` pattern (`format_reward`).
- *`eval_grpo.py`*: Loads a pre-trained or newly-tuned model, runs batched inference against a holdout test split, and computes average metrics (accuracy, formatting adherence, average generation length). Outputs a `metrics.json` file.
- *`grpo_config.py`*: Helper schema and parameter parsing strictly for `train_grpo.py` args (like max steps, beta KL-penalty, generation limits).

== 3. Pipeline Automation (`pipeline/`)

While `train_grpo.py` runs a *single* model, the `pipeline` module orchestrates *experiments*. An experiment might involve taking multiple baseline models, training them with various loss configurations, and comparing the evaluations against the baselines.

- *`pipeline/config.py`*: Implements `PrelimConfig`, which merges CLI arguments with YAML configuration files.
- *`pipeline/runner.py`*: Contains the logic to execute a full loop:
  1. Evaluate Baseline Model.
  2. Run `train_grpo.py` (via subprocess, potentially using `accelerate`).
  3. Evaluate Tuned Model.
  4. Repeat for all combinations of models and loss types requested.
- *`pipeline/reporter.py`*: Computes deltas between the baseline and the newly tuned models, generating CSV leaderboards and Markdown summaries inside the `outputs/` directory.

#quote(block: true)[*Note:* The root files `run_prelim.py` and `compare_runs.py` are thin wrappers that invoke `pipeline/runner.py` and `pipeline/reporter.py`, ensuring backwards compatibility for users accustomed to the previous flat structure.]

== 4. Modal Distributed Execution (`modal_train.py`)

This repository is optimized for #link("https://modal.com")[Modal]. 

`modal_train.py` defines:
1. *Container Image*: A Debian-based image pre-loaded with `torch`, `transformers`, `trl`, `vllm`, and `accelerate`. It mounts the local repository to `/workspace`.
2. *Persistent Volume*: Automatically mounts a volume to `/workspace/outputs` so that all checkpoints, metrics, and leaderboards persist after the ephemeral GPUs spin down.
3. *Execution Logic*: `run_sweep_job` takes a specific configuration permutation (e.g., `model="Qwen/Qwen2.5-0.5B"`, `loss_type="grpo"`) and executes `run_prelim.py` inside the container.
4. *Starmap Dispatch*: The `main` function locally parses a sweep configuration (like `configs/sweeps/pipeline_sweep_presweep.yaml`) and maps over the Cartesian product of parameters, fanning out massively parallel jobs to Modal.

*Usage:*
```bash
modal run modal_train.py
```

== 5. UI and Analytics (`ui/`)

To visualize the massive amount of data generated by sweeps, a Streamlit app is provided.

- *`ui/app.py`*: The Streamlit frontend layout. It allows users to browse experiments (directories within `outputs/`), view leaderboard tables side-by-side, and inspect specific prompt/completion rows from the evaluations.
- *`ui/core.py`*: The backend logic. It abstracts loading `manifest.json` and `metrics.json` files, and provides an interactive "Playground" that loads a given model bundle (using vLLM or Hugging Face Transformers) to perform live inference directly in the browser.

*Usage:*
```bash
./scripts/run_ui.sh
```

== 6. Configurations (`configs/`)

Configurations dictate what the pipeline does. They are YAML representations of `PrelimConfig`.

- *`configs/base.yaml` / `configs/grpo_small.yaml`*: Standard single-run configurations for standard debugging.
- *`configs/sweeps/*.yaml`*: Configuration matrices defining multiple models, multiple learning rates, or different loss variants. `modal_train.py` defaults to reading these.

== 7. Diagnostics (`scripts/healthcheck.py`)

Before doing major refactors or submitting heavy jobs, `python scripts/healthcheck.py` evaluates your PyTorch CUDA bindings, GPU availability, and critical library versions (`trl`, `peft`, `transformers`).
