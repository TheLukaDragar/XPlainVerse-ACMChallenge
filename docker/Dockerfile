# NVIDIA Slurm + Pyxis: CUDA 12.4 devel matches PyTorch cu124 wheels in
# evaluation/env/xplainverse_eval_env.txt. Baselines: ms-swift (editable) + vLLM.
#
# Use the cuda *devel* image (includes nvcc). vLLM v1 may JIT-compile FlashInfer
# kernels; cuda *runtime*-only images have no compiler and fail with
# "/usr/local/cuda/bin/nvcc: not found" during engine startup.
#
# Build:
#   docker build -f docker/Dockerfile -t xplainverse:latest .
#
# Install order: pinned evaluator deps first, then ms-swift (may add/upgrade
# packages), then vLLM. If you need strict adherence to every pin after that,
# reconcile conflicts explicitly — re-running the eval requirements can break
# ms-swift or vLLM.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

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

# --- Challenge evaluator (pinned; includes PyTorch cu124 + transformers @ git SHA)
WORKDIR /tmp
COPY evaluation/env/xplainverse_eval_env.txt ./xplainverse_eval_env.txt
COPY docker/torch-cu124-constraints.txt ./torch-cu124-constraints.txt
RUN pip install -r xplainverse_eval_env.txt

# --- Baselines (see baselines/README.md)
ARG MS_SWIFT_REF=main
WORKDIR /opt/ms-swift
RUN git clone https://github.com/modelscope/ms-swift.git /opt/ms-swift \
    && git -C /opt/ms-swift checkout "${MS_SWIFT_REF}" \
    && pip install -e /opt/ms-swift

# ms-swift deps may bump PyTorch to a newer CUDA wheel (e.g. cu13 NVRTC); realign with evaluator pins.
RUN pip install --no-cache-dir \
    "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install vllm \
    -c /tmp/torch-cu124-constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Video / Qwen-VL helpers (ms-swift prints a warning and some paths need these for Qwen3-VL).
RUN pip install "qwen_vl_utils>=0.0.14" "decord>=0.6.0"

# --- Repository tree (weights excluded via .dockerignore)
WORKDIR /workspace/XPlainVerse-ACMChallenge
COPY . .

CMD ["bash"]
