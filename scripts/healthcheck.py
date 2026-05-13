#!/usr/bin/env python3
"""Environment diagnostic script for the GRPO training pipeline."""

import sys
import subprocess

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def check(name, condition, details=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"       {details}")

print("=== Environment Diagnostic ===\n")

# Python version
print("Python:")
py_ver = sys.version.split()[0]
check("version", True, py_ver)

# PyTorch
print("\nPyTorch:")
output, _, rc = run("python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'")
if rc == 0:
    lines = output.split("\n")
    check("installed", True, lines[0])
    check("CUDA available", lines[1] == "True", lines[1])
    if lines[1] == "True":
        gpu_count, _, _ = run("python -c 'import torch; print(torch.cuda.device_count())'")
        check("GPU count", True, gpu_count)
        if int(gpu_count) > 0:
            gpu_name, _, _ = run("python -c 'import torch; print(torch.cuda.get_device_name(0))'")
            check("GPU 0", True, gpu_name)
else:
    check("installed", False, output)

# Transformers
print("\nTransformers:")
output, _, rc = run("python -c 'import transformers; print(transformers.__version__)'")
if rc == 0:
    check("installed", True, output)
else:
    check("installed", False, output)

# peft
print("\npeft:")
output, _, rc = run("python -c 'import peft; print(peft.__version__)'")
if rc == 0:
    check("installed", True, output)
else:
    check("installed", False, output)

# trl
print("\ntrl:")
output, _, rc = run("python -c 'import trl; print(trl.__version__)'")
if rc == 0:
    check("installed", True, output)
else:
    check("installed", False, output)

# Critical import test
print("\nCritical import (peft):")
output, err, rc = run("python -c 'from peft import AutoPeftModelForCausalLM' 2>&1")
if rc == 0:
    check("peft import", True)
else:
    check("peft import", False, err[:500])

# HuggingFace hub
print("\nHuggingFace hub:")
output, _, rc = run("python -c 'import huggingface_hub; print(huggingface_hub.__version__)'")
if rc == 0:
    check("installed", True, output)
else:
    check("installed", False, output)

print("\n=== Summary ===")
print("Run this to see full traceback:")
print("  python -c 'from peft import AutoPeftModelForCausalLM'")
