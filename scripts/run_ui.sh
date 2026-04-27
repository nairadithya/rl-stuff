#!/usr/bin/env bash
set -euo pipefail

uv run streamlit run ui/model_testing_ui.py "$@"
