#!/usr/bin/env bash
# Full VLM LoRA SFT on XPlainVerse train_vlm.jsonl — Ljubljana (elixir-lj-gpu-01).
#
# Run inside the Apptainer container (4× A100 80GB, Torch 2.2.2+cu121).
# Do NOT run on the host conda shell; use ~/xplainverse_exec.sh.
#
# === Quick start (from host on elixir-lj-gpu-01) ===
#
#   # 1) Interactive shell (optional)
#   ~/xplainverse_exec.sh
#
#   # 2) Build JSONL once (if dataset/*.jsonl missing) — inside container:
#   python3 dataset/build_swift_jsonl.py \
#     --data-root /primoz/luka/XPlainVerse/data/XPlainVerse \
#     --output-dir dataset
#   # For baseline-style 260k balanced train (130k/class):
#   #   ... --train-max-per-class 130000
#
#   # 3) Full 4-GPU training (defaults below)
#   ~/xplainverse_exec.sh bash scripts/train_vlm_full_lj.sh
#
# === From Slurm login node (this worker) ===
#
#   ./scripts/lj_gpu_exec.sh python3 dataset/build_swift_jsonl.py \
#     --data-root /primoz/luka/XPlainVerse/data/XPlainVerse --output-dir dataset
#   ./scripts/lj_gpu_exec.sh bash scripts/train_vlm_full_lj.sh
#
# Smoke (tiny steps; see scripts/LJ_TRAINING.md):
#   LJ_GPU_TIME=01:00:00 ./scripts/lj_gpu_exec.sh bash -lc \
#     'REPORT_TO=tensorboard MAX_STEPS=4 TRAIN_SLICE=32 VAL_SLICE=4 \
#      SAVE_STEPS=999999 EVAL_STEPS=999999 PREDICT_WITH_GENERATE=false \
#      OUTPUT_DIR=/home/jakob/luka/runs/vlm_smoke_lj bash scripts/train_vlm_full_lj.sh'
#
# === Paths (host vs container) ===
#
#   Host code:        /home/jakob/luka/code/XPlainVerse-ACMChallenge
#   Container code:   /workspace/XPlainVerse-ACMChallenge  (bind-mounted)
#   Dataset (images): /primoz/luka/XPlainVerse/data/XPlainVerse  (GPU node NVMe; bind /primoz in Apptainer)
#   Fallback:         /home/jakob/luka/data/XPlainVerse
#   Checkpoints:      /home/jakob/luka/runs/vlm_full        (default OUTPUT_DIR)
#
# Override any default:
#   CODE_ROOT=/workspace/XPlainVerse-ACMChallenge OUTPUT_DIR=/home/jakob/luka/runs/my_run \
#     ~/xplainverse_exec.sh bash scripts/train_vlm_full_lj.sh
#
#   REPORT_TO=tensorboard ./scripts/train_vlm_full_lj.sh
#   PACKING=false PADDING_FREE=false ATTN_IMPL=sdpa ./scripts/train_vlm_full_lj.sh
#   PREDICT_WITH_GENERATE=false VAL_SLICE=2000 ./scripts/train_vlm_full_lj.sh
#
# Hyperparams match scripts/train_vlm_full.sh (ms-swift Qwen3-VL recipe).
# Defaults: 4 GPUs, eff batch 32 (2 × 4 × 4 with packing).

set -euo pipefail

# --- Lj path defaults (auto-detect container bind) ---
if [[ -d /workspace/XPlainVerse-ACMChallenge ]]; then
  _CODE_DEFAULT="/workspace/XPlainVerse-ACMChallenge"
else
  _CODE_DEFAULT="/home/jakob/luka/code/XPlainVerse-ACMChallenge"
fi

CODE_ROOT="${CODE_ROOT:-${_CODE_DEFAULT}}"
if [[ -n "${XPLAINVERSE_DATA_ROOT:-}" ]]; then
  LJ_DATA_ROOT="${XPLAINVERSE_DATA_ROOT}"
elif [[ -d /primoz/luka/XPlainVerse/data/XPlainVerse/train ]]; then
  LJ_DATA_ROOT="/primoz/luka/XPlainVerse/data/XPlainVerse"
else
  LJ_DATA_ROOT="/home/jakob/luka/data/XPlainVerse"
fi
LJ_RUNS_ROOT="${LJ_RUNS_ROOT:-/home/jakob/luka/runs}"

# --- Container Python (HOME bind-mount must not shadow torch/ms-swift) ---
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# --- CUDA / PyTorch env (lj SIF: torch +cu121; prefer cu121/cu12 before cu13) ---
for _cuda_lib in cu121 cu12 cu13; do
  _nv_lib="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  if [[ -d "${_nv_lib}" ]]; then
    export LD_LIBRARY_PATH="${_nv_lib}:${LD_LIBRARY_PATH:-}"
    break
  fi
done
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# VLLM_USE_FLASHINFER_SAMPLER is for vLLM inference only; not required for swift sft.
# The lj SIF does not set LD_LIBRARY_PATH at login — the loop above handles NVRTC if needed.

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

# --- Multi-GPU (lj default: 4× A100) ---
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
  GRAD_ACCUM="${GRAD_ACCUM:-4}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
  GRAD_ACCUM="${GRAD_ACCUM:-16}"
fi

# --- Model ---
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"

# --- Data ---
TRAIN_JSONL="${TRAIN_JSONL:-${CODE_ROOT}/dataset/train_vlm.jsonl}"
VAL_JSONL="${VAL_JSONL:-${CODE_ROOT}/dataset/val_vlm.jsonl}"
# Optional ms-swift row cap: TRAIN_SLICE=500 → --dataset path.jsonl#500 (smoke / debug).
TRAIN_SLICE="${TRAIN_SLICE:-}"
VAL_SLICE="${VAL_SLICE:-2000}"
# When set, passes --max_steps to swift (omit full-epoch training; good for smoke tests).
MAX_STEPS="${MAX_STEPS:-}"

# --- Hyperparams (same as train_vlm_full.sh) ---
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/vlm_full}"
SEED="${SEED:-42}"

SAVE_STEPS="${SAVE_STEPS:-400}"
EVAL_STEPS="${EVAL_STEPS:-400}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

PACKING="${PACKING:-true}"
PADDING_FREE="${PADDING_FREE:-true}"
DEEPSPEED="${DEEPSPEED:-zero2}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"

PREDICT_WITH_GENERATE="${PREDICT_WITH_GENERATE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export WANDB_SAMPLE_N="${WANDB_SAMPLE_N:-16}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-vlm_full_lj_${NUM_EPOCHS}ep}"

if [[ "${REPORT_TO}" == *wandb* ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
  if wandb status 2>/dev/null | grep -q '"api_key": null'; then
    echo "warning: wandb not logged in. Run: wandb login  (or set WANDB_API_KEY, or REPORT_TO=tensorboard)" >&2
  fi
fi

EXTERNAL_PLUGIN="${CODE_ROOT}/external_plugins/wandb_predictions.py"
USE_PRED_CALLBACK=false
if [[ "${PREDICT_WITH_GENERATE}" == "true" && "${REPORT_TO}" == *wandb* ]]; then
  if [[ -f "${EXTERNAL_PLUGIN}" ]]; then
    USE_PRED_CALLBACK=true
  else
    echo "warning: plugin not found at ${EXTERNAL_PLUGIN}; W&B preds disabled." >&2
  fi
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "error: 'swift' (ms-swift) not on PATH. Install inside the lj SIF or run via ~/xplainverse_exec.sh." >&2
  exit 1
fi

_TRAIN_JSONL_PATH="${TRAIN_JSONL%%#*}"
if [[ ! -f "${_TRAIN_JSONL_PATH}" ]]; then
  echo "error: ${_TRAIN_JSONL_PATH} missing." >&2
  echo "  Build inside container (from CODE_ROOT=${CODE_ROOT}):" >&2
  echo "    python3 dataset/build_swift_jsonl.py \\" >&2
  echo "      --data-root ${LJ_DATA_ROOT} \\" >&2
  echo "      --output-dir ${CODE_ROOT}/dataset" >&2
  echo "  Optional baseline cap: --train-max-per-class 130000" >&2
  exit 1
fi

if [[ -n "${TRAIN_SLICE}" ]]; then
  TRAIN_DATASET="${TRAIN_JSONL}#${TRAIN_SLICE}"
else
  TRAIN_DATASET="${TRAIN_JSONL}"
fi

TRAIN_SCHEDULE_ARGS=(--num_train_epochs "${NUM_EPOCHS}")
if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_SCHEDULE_ARGS=(--max_steps "${MAX_STEPS}")
fi

if [[ "${ATTN_IMPL}" == "flash_attn" ]] && ! python3 -c "import flash_attn" 2>/dev/null; then
  echo "warning: flash_attn not importable; falling back to ATTN_IMPL=sdpa PACKING=false PADDING_FREE=false" >&2
  ATTN_IMPL=sdpa
  PACKING=false
  PADDING_FREE=false
fi

# ms-swift calls transformers.require_version('deepspeed') when --deepspeed is set.
if [[ -n "${DEEPSPEED:-}" ]] && ! python3 -c "import importlib.metadata as m; m.version('deepspeed')" 2>/dev/null; then
  echo "warning: deepspeed distribution not found; multi-GPU will use DDP without ZeRO (install deepspeed in SIF to enable)." >&2
  DEEPSPEED=""
fi

mkdir -p "${OUTPUT_DIR}"

EFF_BATCH=$(( PER_DEVICE_BS * NPROC_PER_NODE * GRAD_ACCUM ))
echo "=== VLM full SFT (lj / Apptainer) ==="
echo "code_root:           ${CODE_ROOT}"
echo "data_root (build):   ${LJ_DATA_ROOT}"
echo "model:               ${MODEL}"
echo "train:               ${TRAIN_DATASET}"
echo "val (eval):          ${VAL_JSONL}#${VAL_SLICE}  (wandb table: ${WANDB_SAMPLE_N} samples)"
echo "gpus:                NPROC=${NPROC_PER_NODE}  CUDA=${CUDA_VISIBLE_DEVICES}"
echo "per_device_bs:       ${PER_DEVICE_BS}  grad_accum: ${GRAD_ACCUM}  → eff batch ${EFF_BATCH}"
if [[ -n "${MAX_STEPS}" ]]; then
  echo "train_schedule:      max_steps=${MAX_STEPS}  (NUM_EPOCHS=${NUM_EPOCHS} ignored)"
else
  echo "train_schedule:      num_train_epochs=${NUM_EPOCHS}"
fi
echo "max_length:          ${MAX_LENGTH}  lr: ${LEARNING_RATE}  rank: ${LORA_RANK}"
echo "attn_impl:           ${ATTN_IMPL}"
echo "deepspeed:           ${DEEPSPEED:-<off>}"
echo "packing:             ${PACKING}  padding_free: ${PADDING_FREE}"
echo "predict_with_gen:    ${PREDICT_WITH_GENERATE}  (max_new_tokens=${MAX_NEW_TOKENS})"
echo "wandb_pred_callback: ${USE_PRED_CALLBACK}  (sample_n=${WANDB_SAMPLE_N})"
echo "output:              ${OUTPUT_DIR}"
echo "report_to:           ${REPORT_TO}"
if [[ "${REPORT_TO}" == *wandb* ]]; then
  echo "wandb:               ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_RUN_NAME})"
fi
echo

DEEPSPEED_FLAG=()
if [[ "${NPROC_PER_NODE}" -gt 1 && -n "${DEEPSPEED}" ]]; then
  DEEPSPEED_FLAG=(--deepspeed "${DEEPSPEED}")
fi

PLUGIN_FLAG=()
CALLBACK_FLAG=()
if [[ "${USE_PRED_CALLBACK}" == "true" ]]; then
  PLUGIN_FLAG=(--external_plugins "${EXTERNAL_PLUGIN}")
  CALLBACK_FLAG=(--callbacks wandb_predictions)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
swift sft \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  --dataset "${TRAIN_DATASET}" \
  --val_dataset "${VAL_JSONL}#${VAL_SLICE}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --attn_impl "${ATTN_IMPL}" \
  --packing "${PACKING}" \
  --padding_free "${PADDING_FREE}" \
  --max_length "${MAX_LENGTH}" \
  "${TRAIN_SCHEDULE_ARGS[@]}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_llm false \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 5 \
  --logging_steps "${LOGGING_STEPS}" \
  --predict_with_generate "${PREDICT_WITH_GENERATE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --report_to ${REPORT_TO} \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --load_from_cache_file true \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  "${DEEPSPEED_FLAG[@]}" \
  "${PLUGIN_FLAG[@]}" \
  "${CALLBACK_FLAG[@]}"

echo
echo "Done. Merge LoRA for faster infer:"
echo "  swift export --adapters ${OUTPUT_DIR}/vx-*/checkpoint-XXXX --merge_lora true"
