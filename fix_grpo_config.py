#!/usr/bin/env python3
"""Patch train_grpo.py and reward_fns.py for installed trl version compatibility."""

import re
import inspect
import sys

# --- Part 1: Fix GRPOConfig params in train_grpo.py ---
try:
    from trl import GRPOConfig
except ImportError as e:
    print(f"ERROR: trl not installed - {e}")
    sys.exit(1)

sig = inspect.signature(GRPOConfig.__init__)
valid_params = set(sig.parameters.keys())
print(f"[GRPOConfig] Valid params ({len(valid_params)}):")
for p in sorted(valid_params):
    print(f"  {p}")

script_path = "train_grpo.py"
with open(script_path, "r") as f:
    content = f.read()

start_marker = "grpo_args = GRPOConfig("
start = content.index(start_marker)
call_start = start + len(start_marker)
depth = 1
pos = call_start
while depth > 0:
    if content[pos] == "(":
        depth += 1
    elif content[pos] == ")":
        depth -= 1
    pos += 1
end = pos

args_block = content[call_start:end-1]
arg_lines = args_block.split("\n")

new_lines = []
removed = []
param_pattern = re.compile(r"^\s*(\w+)\s*=")

for line in arg_lines:
    m = param_pattern.match(line)
    if m:
        param = m.group(1)
        if param in valid_params:
            new_lines.append(line)
        else:
            removed.append(param)
    else:
        new_lines.append(line)

print(f"\n[GRPOConfig] Removed params: {removed}")

new_args = "\n".join(line for line in new_lines if line.strip())
new_block = f"{start_marker}\n{new_args}\n    )"
new_content = content[:start] + new_block + content[end:]

with open(script_path, "w") as f:
    f.write(new_content)

print(f"[GRPOConfig] Patched {script_path}")

# --- Part 2: Fix trl.rewards import in reward_fns.py ---
print("\n[reward_fns] Checking trl.rewards import...")
reward_fns_path = "reward_fns.py"
with open(reward_fns_path, "r") as f:
    rf_content = f.read()

if "from trl.rewards import accuracy_reward" in rf_content:
    print("[reward_fns] Found trl.rewards import, inlining accuracy_reward...")

    # Build new accuracy_reward function
    new_func = '''
def accuracy_reward(completions, solution, **kwargs):
    rewards = []
    for completion, gold in zip(completions, solution):
        gold_str = str(gold).strip().lower()
        reward = 0.0
        completion_text = _extract_text(completion)
        completion_lower = completion_text.strip().lower()
        if completion_lower == gold_str:
            reward = 1.0
        elif completion_lower.endswith(gold_str) or gold_str in completion_lower:
            reward = 1.0
        rewards.append(reward)
    return rewards
'''

    # Replace the import with the inline function
    rf_content = rf_content.replace(
        "from trl.rewards import accuracy_reward",
        new_func
    )

    with open(reward_fns_path, "w") as f:
        f.write(rf_content)

    print(f"[reward_fns] Patched {reward_fns_path}")
else:
    print("[reward_fns] No trl.rewards import found, nothing to patch")
