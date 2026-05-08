from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    repetition_penalty: float = 1.05


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _load_transformers_model(
    model_name_or_path: str, tokenizer_name_or_path: str | None
):
    device = _resolve_device()
    dtype = _resolve_dtype(device)

    load_kwargs: dict[str, Any] = {"dtype": dtype}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

    if device in {"cpu", "mps"}:
        model = model.to(device)
    model.eval()

    tokenizer_source = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return {
        "backend": "transformers",
        "device": device,
        "model": model,
        "tokenizer": tokenizer,
    }


def _load_mlx_model(model_name_or_path: str):
    from mlx_lm import load

    model, tokenizer = load(model_name_or_path)
    return {"backend": "mlx", "device": "mlx", "model": model, "tokenizer": tokenizer}


def load_model_bundle(
    model_name_or_path: str, tokenizer_name_or_path: str | None = None
) -> dict[str, Any]:
    mlx_spec = importlib.util.find_spec("mlx_lm")
    if mlx_spec is not None:
        try:
            return _load_mlx_model(model_name_or_path)
        except Exception:
            pass
    return _load_transformers_model(model_name_or_path, tokenizer_name_or_path)


def _generate_with_transformers(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    config: GenerationConfig,
) -> str:
    tokenized = tokenizer(prompt, return_tensors="pt")
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    do_sample = config.temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": config.repetition_penalty,
    }
    if do_sample:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    with torch.no_grad():
        generated = model.generate(**tokenized, **generation_kwargs)

    input_length = int(tokenized["input_ids"].shape[-1])
    completion_ids = generated[0, input_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _generate_with_mlx(
    *, model: Any, tokenizer: Any, prompt: str, config: GenerationConfig
) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    sampler = make_sampler(temp=config.temperature, top_p=config.top_p)
    logits_processors = make_logits_processors(
        repetition_penalty=config.repetition_penalty,
    )

    completion = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=config.max_new_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )
    return str(completion).strip()


def generate_text(
    model_bundle: dict[str, Any], prompt: str, config: GenerationConfig
) -> str:
    backend = str(model_bundle.get("backend", "transformers"))
    if backend == "mlx":
        return _generate_with_mlx(
            model=model_bundle["model"],
            tokenizer=model_bundle["tokenizer"],
            prompt=prompt,
            config=config,
        )

    return _generate_with_transformers(
        model=model_bundle["model"],
        tokenizer=model_bundle["tokenizer"],
        device=model_bundle["device"],
        prompt=prompt,
        config=config,
    )
