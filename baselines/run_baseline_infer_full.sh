#!/usr/bin/env bash
# Full-val inference for both baseline models (Qwen3-VL-8B and InternVL3.5-14B).
# Produces a JSONL ready for the challenge evaluator.
#
# Usage:
#   ./run_baseline_infer_full.sh                          # Qwen3-VL-8B, vllm backend
#   MODEL=internvl ./run_baseline_infer_full.sh            # InternVL3.5-14B
#   INFER_BACKEND=transformers ./run_baseline_infer_full.sh
#   NUM_SAMPLES=100 ./run_baseline_infer_full.sh           # subset for testing
#
# Outputs (per model):
#   $RUN_DIR/<model_tag>_output.jsonl        raw swift infer output
#   $RUN_DIR/<model_tag>_submission.jsonl    challenge-format submission

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/data/XPlainVerse}"
VAL_IMAGES_DIR="${VAL_IMAGES_DIR:-${DATA_ROOT}/val/images}"
GROUND_TRUTH="${GROUND_TRUTH:-${CODE_ROOT}/evaluation/data/val_ground_truth.jsonl}"
RUN_DIR="${RUN_DIR:-${WORKSPACE_ROOT}/runs}"

# --- Model selection ---
# MODEL=qwen  → Qwen3-VL-8B-XPlainVerse
# MODEL=internvl → InternVL3_5-14B-XPlainVerse
MODEL="${MODEL:-qwen}"

if [[ "${MODEL}" == "internvl" ]]; then
    MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/baseline_models/InternVL3_5-14B-XPlainVerse}"
    MODEL_TYPE="${MODEL_TYPE:-internvl3_5}"
    MODEL_TAG="internvl3_5_14b"
else
    MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/baseline_models/Qwen3-VL-8B-XPlainVerse}"
    MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
    MODEL_TAG="qwen3_vl_8b"
fi

INFER_BACKEND="${INFER_BACKEND:-vllm}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# -1 = all samples; any positive integer = subset
NUM_SAMPLES="${NUM_SAMPLES:--1}"

DATASET_JSONL="${RUN_DIR}/${MODEL_TAG}_val_dataset.jsonl"
RAW_OUTPUT="${RUN_DIR}/${MODEL_TAG}_output.jsonl"
SUBMISSION="${RUN_DIR}/${MODEL_TAG}_submission.jsonl"

# nvrtc fix: vLLM cu13 wheels install libnvrtc-builtins.so.13.0 under the pip
# nvidia/cu13 package but that dir is not in the default ld search path.
_CU13_LIB="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib"
if [[ -d "${_CU13_LIB}" ]]; then
    export LD_LIBRARY_PATH="${_CU13_LIB}:${LD_LIBRARY_PATH:-}"
fi

TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCH_COMPILE_DISABLE

mkdir -p "${RUN_DIR}"

if ! command -v swift >/dev/null 2>&1; then
    echo "error: 'swift' not found — install ms-swift or activate your env." >&2; exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "error: model not found: ${MODEL_DIR}" >&2; exit 1
fi
if [[ ! -f "${GROUND_TRUTH}" ]]; then
    echo "error: ground truth not found: ${GROUND_TRUTH}" >&2; exit 1
fi
if [[ ! -d "${VAL_IMAGES_DIR}" ]]; then
    echo "error: VAL_IMAGES_DIR not found: ${VAL_IMAGES_DIR}" >&2; exit 1
fi

echo "Model:   ${MODEL_TAG} (${INFER_BACKEND})"
echo "Samples: $([ "${NUM_SAMPLES}" -lt 0 ] && echo 'all' || echo "${NUM_SAMPLES}")"
echo "Output:  ${RAW_OUTPUT}"

# --- Build val dataset JSONL from ground truth + images ---
python3 - "${GROUND_TRUTH}" "${VAL_IMAGES_DIR}" "${NUM_SAMPLES}" "${DATASET_JSONL}" <<'PY'
import json, sys
from pathlib import Path

gt_path, images_dir, num_samples, out_path = sys.argv[1:]
images_dir = Path(images_dir)
num_samples = int(num_samples)
out_path = Path(out_path)
extensions = (".png", ".jpg", ".jpeg", ".webp")
search_roots = (images_dir, images_dir / "fake", images_dir / "real")

user_content = (
    "<image>\n"
    "Detect whether the image is real or fake and provide reasoning for it.\n\n"
    "Respond in the following format:\n"
    "<reasoning>your reasoning here</reasoning>\n"
    "<answer>real or fake</answer>"
)

records = []
with open(gt_path, encoding="utf-8") as f:
    for line in f:
        if num_samples >= 0 and len(records) >= num_samples:
            break
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if not sid:
            continue
        image_path = None
        for root in search_roots:
            for ext in extensions:
                c = root / f"{sid}{ext}"
                if c.is_file():
                    image_path = c.resolve()
                    break
            if image_path:
                break
        if not image_path:
            continue
        records.append({
            "id": sid,
            "messages": [{"role": "user", "content": user_content}],
            "images": [str(image_path)],
        })

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Dataset: {len(records)} samples → {out_path}")
PY

# --- Run inference ---
# Always write fresh (remove old output so swift doesn't append)
rm -f "${RAW_OUTPUT}"

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
    --max_batch_size 16 \
    --result_path "${RAW_OUTPUT}"

# --- Convert to challenge submission format ---
python3 - "${RAW_OUTPUT}" "${SUBMISSION}" <<'PY'
import json, re, sys
from pathlib import Path

raw_path, sub_path = sys.argv[1:]
seen = {}
with open(raw_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sid = Path(row["images"][0]["path"]).stem
        if sid in seen:
            continue
        m_r = re.search(r"<reasoning>(.*?)</reasoning>", row["response"], re.S | re.I)
        m_a = re.search(r"<answer>(.*?)</answer>", row["response"], re.I)
        reasoning = m_r.group(1).strip() if m_r else row["response"].strip()
        label = m_a.group(1).strip().lower() if m_a else "unknown"
        seen[sid] = {
            "sample_id": sid,
            "label": label,
            # model produces one reasoning; map to both fields for the evaluator
            "complex_explanation": reasoning,
            "simple_explanation": reasoning,
        }

with open(sub_path, "w", encoding="utf-8") as f:
    for row in seen.values():
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Submission: {len(seen)} samples → {sub_path}")
PY

echo ""
echo "Done."
echo "  raw output:  ${RAW_OUTPUT}"
echo "  submission:  ${SUBMISSION}"
echo ""
echo "Run evaluator:"
echo "  cd ${CODE_ROOT}"
echo "  python3 evaluation/evaluate_simple_explanations.py \\"
echo "    --submission ${SUBMISSION} \\"
echo "    --output ${RUN_DIR}/${MODEL_TAG}_simple_eval.json"
