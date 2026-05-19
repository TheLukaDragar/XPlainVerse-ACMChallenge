# NVIDIA Slurm + Pyxis: CUDA 12.4 runtime matches PyTorch cu124 wheels in
# evaluation/env/xplainverse_eval_env.txt. Baselines: ms-swift (editable) + vLLM.
#
# Build:
#   docker build -f docker/Dockerfile -t xplainverse:latest .
#
# Install order: pinned evaluator deps first, then ms-swift (may add/upgrade
# packages), then vLLM. If you need strict adherence to every pin after that,
# reconcile conflicts explicitly — re-running the eval requirements can break
# ms-swift or vLLM.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

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
RUN pip install -r xplainverse_eval_env.txt

# --- Baselines (see baselines/README.md)
ARG MS_SWIFT_REF=main
WORKDIR /opt/ms-swift
RUN git clone https://github.com/modelscope/ms-swift.git /opt/ms-swift \
    && git -C /opt/ms-swift checkout "${MS_SWIFT_REF}" \
    && pip install -e /opt/ms-swift

RUN pip install vllm

# --- Repository tree (weights excluded via .dockerignore)
WORKDIR /workspace/XPlainVerse-ACMChallenge
COPY . .

CMD ["bash"]
