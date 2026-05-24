#!/usr/bin/env bash
# Smoke-test a merged baseline checkpoint with ms-swift + vLLM on a few val samples.
#
# Prerequisites:
#   - Val images as flat files {sample_id}.<ext>, or Hugging Face layout under
#     VAL_IMAGES_DIR with fake/ and real/ subfolders (same filenames).
#   - ms-swift + vLLM installed and `swift` on PATH; NVIDIA GPU.
#
# Usage:
#   ./run_baseline_infer_smoke.sh
#   NUM_SAMPLES=10 VAL_IMAGES_DIR=/path/to/images ./run_baseline_infer_smoke.sh
#   MODEL_TYPE=... ./run_baseline_infer_smoke.sh   # default: qwen3_vl for Qwen3-VL-8B baseline
#   INFER_BACKEND=transformers ./run_baseline_infer_smoke.sh  # no vLLM; avoids needing nvcc (slower)
#   TORCH_COMPILE_DISABLE=0 ./run_baseline_infer_smoke.sh       # allow torch.compile (needs matching NVRTC/CUDA)

set -euo pipefail

# --- Layout on FRIDA / shared workspace (override with env vars if needed) ---
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/data/XPlainVerse}"

# After you extract archives: point this at the folder that contains image files.
VAL_IMAGES_DIR="${VAL_IMAGES_DIR:-${DATA_ROOT}/val/images}"

MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/baseline_models/Qwen3-VL-8B-XPlainVerse}"
# ms-swift may not infer this from checkpoints that match multiple template types (qwen3_vl vs emb vs reranker).
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
GROUND_TRUTH="${GROUND_TRUTH:-${CODE_ROOT}/evaluation/data/val_ground_truth.jsonl}"

NUM_SAMPLES="${NUM_SAMPLES:-5}"
RUN_DIR="${RUN_DIR:-${WORKSPACE_ROOT}/runs}"
DATASET_JSONL="${DATASET_JSONL:-${RUN_DIR}/baseline_smoke_infer_dataset.jsonl}"
RESULT_JSONL="${RESULT_JSONL:-${RUN_DIR}/baseline_smoke_infer_output.jsonl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"

# torch.compile uses NVRTC; a PyTorch wheel built for CUDA 13 expects
# libnvrtc-builtins.so.13.0. If that user-space lib is missing or the stack was
# accidentally upgraded past the evaluator's cu124 pins, smoke fails. Default: no compile.
TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCH_COMPILE_DISABLE

mkdir -p "${RUN_DIR}"

if ! command -v swift >/dev/null 2>&1; then
  echo "error: 'swift' not found on PATH (install ms-swift / activate your env)." >&2
  exit 1
fi

# Warn when PyTorch is not on the evaluator's cu124 line (common after `pip install vllm` / ms-swift).
_tver="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unavailable')"
if [[ "${_tver}" != unavailable && "${_tver}" != *+cu124* ]]; then
  echo "warning: PyTorch is ${_tver} (evaluator pins torch==2.6.0+cu124). See baselines/README.md, \"PyTorch CUDA line\"." >&2
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "error: model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${GROUND_TRUTH}" ]]; then
  echo "error: ground-truth JSONL not found: ${GROUND_TRUTH}" >&2
  exit 1
fi

if [[ ! -d "${VAL_IMAGES_DIR}" ]]; then
  echo "error: VAL_IMAGES_DIR does not exist yet: ${VAL_IMAGES_DIR}" >&2
  echo "       Extract val images there (flat files: {sample_id}.png or .jpg), then re-run." >&2
  exit 1
fi

echo "Building ms-swift dataset from first ${NUM_SAMPLES} reference rows with matching images..."
python3 - "${GROUND_TRUTH}" "${VAL_IMAGES_DIR}" "${NUM_SAMPLES}" "${DATASET_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

gt_path, images_dir, num_samples, out_path = sys.argv[1:]
images_dir = Path(images_dir)
num_samples = int(num_samples)
out_path = Path(out_path)

extensions = (".png", ".jpg", ".jpeg", ".webp")

user_content = (
    "<image>\n"
    "Detect whether the image is real or fake and provide reasoning for it.\n\n"
    "Respond in the following format:\n"
    "<reasoning>your reasoning here</reasoning>\n"
    "<answer>real or fake</answer>"
)

records = []
with open(gt_path, encoding="utf-8") as handle:
    for line in handle:
        if len(records) >= num_samples:
            break
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if not sid:
            continue

        image_path = None
        search_roots = (
            images_dir,
            images_dir / "fake",
            images_dir / "real",
        )
        for root in search_roots:
            for ext in extensions:
                candidate = root / f"{sid}{ext}"
                if candidate.is_file():
                    image_path = candidate.resolve()
                    break
            if image_path is not None:
                break

        if image_path is None:
            continue

        records.append(
            {
                "id": f"smoke__{sid}",
                "messages": [{"role": "user", "content": user_content}],
                "images": [str(image_path)],
            }
        )

if not records:
    print(
        "No matching images found. Expected files like:",
        f"{images_dir}/<sample_id>.png or {images_dir}/{{fake,real}}/<sample_id>.png",
        file=sys.stderr,
    )
    sys.exit(1)

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as out:
    for record in records:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} lines to {out_path}")
PY

echo "Running swift infer (backend=${INFER_BACKEND})..."

infer_extra=()
if [[ "${INFER_BACKEND}" == "vllm" ]]; then
  infer_extra+=(--vllm_gpu_memory_utilization 0.9)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" swift infer \
  --model "${MODEL_DIR}" \
  --model_type "${MODEL_TYPE}" \
  --val_dataset "${DATASET_JSONL}" \
  --infer_backend "${INFER_BACKEND}" \
  --max_new_tokens 2048 \
  --temperature 0.0 \
  --torch_dtype bfloat16 \
  --stream false \
  --use_hf true \
  "${infer_extra[@]}" \
  --max_model_len 4096 \
  --max_batch_size 1 \
  --result_path "${RESULT_JSONL}"

echo "Done. Outputs:"
echo "  dataset: ${DATASET_JSONL}"
echo "  results: ${RESULT_JSONL}"
