import re
from pathlib import Path

# 1. Read files
run_prelim_text = Path("run_prelim.py").read_text()
compare_runs_text = Path("compare_runs.py").read_text()

# 2. Extract config parts
config_code = """from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

"""

# Extract classes and functions for config
for match in re.finditer(r'(@dataclass\nclass PrelimConfig:.*?)(?=\ndef |$)', run_prelim_text, re.DOTALL):
    config_code += match.group(1)
    
for func in ['parse_args', 'load_config_file', 'merge_config']:
    match = re.search(rf'(def {func}\(.*?\):.*?)(?=\ndef |$)', run_prelim_text, re.DOTALL)
    if match:
        config_code += "\n" + match.group(1)

Path("pipeline").mkdir(exist_ok=True)
Path("pipeline/__init__.py").touch()
Path("pipeline/config.py").write_text(config_code)

# We will just keep the logic in run_prelim.py for now but move the core functions to a generic runner if we want,
# but a full AST split is risky in one shot. 
