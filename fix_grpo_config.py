#!/usr/bin/env python3
"""Patch train_grpo.py to only pass GRPOConfig-accepted args. Run on RunPod."""

import inspect
import re
import sys

try:
    from trl import GRPOConfig
except ImportError as e:
    print(f"ERROR: trl not installed - {e}")
    sys.exit(1)

sig = inspect.signature(GRPOConfig.__init__)
valid_params = set(sig.parameters.keys())
print(f"Valid GRPOConfig params ({len(valid_params)}):")
for p in sorted(valid_params):
    print(f"  {p}")

script_path = sys.argv[1] if len(sys.argv) > 1 else "train_grpo.py"
with open(script_path, "r") as f:
    content = f.read()

start = content.index("grpo_args = GRPOConfig(")
depth = 1
pos = start + len("grpo_args = GRPOConfig(")
while depth > 0:
    if content[pos] == "(":
        depth += 1
    elif content[pos] == ")":
        depth -= 1
    pos += 1
end = pos

old_block = content[start:end]
lines = []
removed = []
for line in old_block.split("\n"):
    m = re.match(r"^\s*(\w+)\s*=", line)
    if m:
        param = m.group(1)
        if param in valid_params:
            lines.append(line)
        else:
            removed.append(param)
    else:
        lines.append(line)

print(f"\nRemoved params: {removed}")
new_block = "\n".join(lines)
new_content = content[:start] + new_block + content[end:]

with open(script_path, "w") as f:
    f.write(new_content)

print(f"Patched {script_path}")
