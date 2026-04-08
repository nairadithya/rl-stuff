from __future__ import annotations

from typing import Any


def _extract_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion)


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        has_answer = "\\boxed{" in text or "Answer:" in text
        rewards.append(1.0 if has_answer else 0.0)
    return rewards


def length_reward(completion_ids, **kwargs):
    return [float(len(ids)) for ids in completion_ids]
