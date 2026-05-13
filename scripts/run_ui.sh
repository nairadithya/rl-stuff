#!/usr/bin/env bash
set -euo pipefail

uv run streamlit run ui/app.py "$@"
