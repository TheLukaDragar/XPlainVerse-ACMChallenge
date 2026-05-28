#!/usr/bin/env bash
# Run official CodaBench xdd-scorer via Apptainer on elixir-lj-gpu-01.
#
# Usage:
#   ./scripts/run_codabench_scorer_lj.sh /path/to/input /path/to/output
#   ./scripts/run_codabench_scorer_lj.sh /path/to/input /path/to/output --mock-bert --mock-sle
#
# Input layout (CodaBench):
#   input/ref/config.json
#   input/res/submission.zip   (detection.jsonl + complex.jsonl + simple.jsonl)
#
# One-time pull:
#   apptainer pull ~/containers/xdd-scorer_2026-v5.sif docker://abhijeet1317/xdd-scorer:2026-v5

set -euo pipefail

INPUT_DIR="${1:?usage: $0 <input_dir> <output_dir> [scoring.py args...]}"
OUTPUT_DIR="${2:?usage: $0 <input_dir> <output_dir> [scoring.py args...]}"
shift 2

SIF="${XDD_SCORER_SIF:-${HOME}/containers/xdd-scorer_2026-v5.sif}"
PARTITION="${LJ_PARTITION:-elixir-interno}"
GPU_NODE="${LJ_GPU_NODE:-elixir-lj-gpu-01.elixir.ul.si}"

run_scorer() {
  if [[ ! -f "${SIF}" ]]; then
    echo "error: scorer SIF not found at ${SIF}" >&2
    echo "  apptainer pull ${SIF} docker://abhijeet1317/xdd-scorer:2026-v5" >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_DIR}"
  # --no-home: host ~/.local numpy/sklearn breaks container imports on Lj.
  apptainer exec --no-home --cleanenv --nv \
    -B "${INPUT_DIR}:${INPUT_DIR}" \
    -B "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    "${SIF}" \
    python3 /app/program/scoring.py "${INPUT_DIR}" "${OUTPUT_DIR}" "$@"
}

if hostname 2>/dev/null | grep -q 'elixir-lj-gpu'; then
  run_scorer "$@"
fi

srun --partition="${PARTITION}" -w "${GPU_NODE}" --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=02:00:00 \
  bash -lc "$(declare -f run_scorer); SIF='${SIF}'; INPUT_DIR='${INPUT_DIR}'; OUTPUT_DIR='${OUTPUT_DIR}'; run_scorer $(printf '%q ' "$@")"
