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


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    depth = 1
    i = start
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


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


def robust_accuracy_reward(completions, solution, **kwargs):
    scores = accuracy_reward(completions=completions, solution=solution, **kwargs)
    rewards = []
    for completion, gold, score in zip(completions, solution, scores):
        if score > 0.0:
            rewards.append(float(score))
            continue

        completion_text = _extract_text(completion)
        boxed = _extract_boxed(completion_text)
        pred_norm = _normalize_text(boxed if boxed is not None else completion_text)
        gold_norm = _normalize_text(str(gold))

        if not pred_norm or not gold_norm:
            rewards.append(0.0)
            continue

        if pred_norm == gold_norm:
            rewards.append(1.0)
            continue

        if pred_norm.endswith(gold_norm) or gold_norm in pred_norm:
            rewards.append(1.0)
            continue

        rewards.append(0.0)

    return rewards


def accuracy_format_reward(completions, solution, **kwargs):
    accuracy_scores = robust_accuracy_reward(
        completions=completions,
        solution=solution,
        **kwargs,
    )
    format_scores = format_reward(completions=completions)

    rewards = []
    for accuracy_score, format_score in zip(accuracy_scores, format_scores):
        rewards.append(0.8 * float(accuracy_score) + 0.2 * float(format_score))
    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        has_answer = "\\boxed{" in text or "Answer:" in text
        rewards.append(1.0 if has_answer else 0.0)
    return rewards


def length_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        if isinstance(completion, str):
            rewards.append(float(len(completion)))
        else:
            rewards.append(float(len(completion)))
    return rewards
