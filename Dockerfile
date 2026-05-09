FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG PRECACHE_MODELS=false

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    git curl \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > /tmp/requirements_notorch.txt \
  && pip install --no-cache-dir -r /tmp/requirements_notorch.txt

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

ENV HF_HOME=/workspace/hf_cache
ENV HF_HUB_CACHE=/workspace/hf_cache/hub
ENV HF_DATASETS_CACHE=/workspace/hf_datasets

RUN if [ "$PRECACHE_MODELS" = "true" ]; then \
      python3 -c "\
from datasets import load_dataset; \
load_dataset('trl-lib/DeepMath-103K', split='train'); \
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True); \
"; \
    fi

CMD ["bash", "/workspace/runpod_start.sh"]
