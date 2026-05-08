FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG PRECACHE_MODELS=false

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl \
  && rm -rf /var/lib/apt/lists/*

ENV PIP_ROOT_USER_ACTION=ignore

RUN python3 -m pip install --upgrade pip

COPY requirements.txt ./requirements.txt
RUN pip install --ignore-installed --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r requirements.txt

COPY . .
RUN pip install -e . --ignore-installed

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
