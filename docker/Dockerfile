# Host GPU driver is CUDA 13.x (verified: NVIDIA-SMI 580, CUDA Version: 13.0).
# Use the CUDA 13 devel image so nvcc matches the runtime libraries vLLM/FlashInfer need.
#
# Build:
#   docker build -f docker/Dockerfile -t xplainverse:latest .
#
# Install order: evaluator deps first (pins transformers @ git SHA), then ms-swift,
# then vLLM (brings in torch cu13 + matching CUDA libs — that is correct for this hardware).

FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ca-certificates \
    build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Challenge evaluator (pins transformers @ git SHA; torch version will be overridden by vLLM below)
WORKDIR /tmp
COPY evaluation/env/xplainverse_eval_env.txt ./xplainverse_eval_env.txt
RUN pip install -r xplainverse_eval_env.txt

# --- Baselines (see baselines/README.md)
ARG MS_SWIFT_REF=main
WORKDIR /opt/ms-swift
RUN git clone https://github.com/modelscope/ms-swift.git /opt/ms-swift \
    && git -C /opt/ms-swift checkout "${MS_SWIFT_REF}" \
    && pip install -e /opt/ms-swift

# vLLM brings in the correct torch+CUDA build for the hardware (cu13 on CUDA 13 hosts).
# It explicitly supports transformers !=5.0-5.5, >=4.56 which matches the eval pin (5.6.0.dev0).
RUN pip install vllm

# Video / Qwen-VL helpers (ms-swift prints a warning and some paths need these for Qwen3-VL).
RUN pip install "qwen_vl_utils>=0.0.14" "decord>=0.6.0"

# Eval-time NLG metrics (rouge/bleu) for ms-swift sft with --predict_with_generate true.
# `swift sft` defaults --eval_metric to nlg in that mode and imports jieba/nltk/rouge.
RUN pip install jieba nltk rouge

# Experiment tracking — used by scripts/train_vlm_*.sh (REPORT_TO=wandb is default).
RUN pip install -U wandb

# Flash Attention 2 — pre-built wheel (no compilation). Pin exact wheel that matches
# Python 3.10 + CUDA 13.0 + PyTorch 2.11 shipped by vLLM. Source: mjun0812/flash-attention-prebuild-wheels.
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp310-cp310-linux_x86_64.whl

# --- Repository tree (weights excluded via .dockerignore)
WORKDIR /workspace/XPlainVerse-ACMChallenge
COPY . .

CMD ["bash"]
