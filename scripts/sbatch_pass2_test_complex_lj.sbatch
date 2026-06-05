#!/bin/bash
# Pass-2 complex explanations on full 200k test — 4 GPU shards in parallel.
#
# Submit:
#   sbatch scripts/sbatch_pass2_test_complex_lj.sbatch
#
# Resume merge after all shards finish:
#   OUT_DIR=/home/jakob/luka/runs/pass2_test_complex/YYYYMMDD-HHMMSS \
#     MERGE_ONLY=1 SHARD_COUNT=4 ./scripts/run_pass2_test_complex_lj.sh
#
#SBATCH --job-name=pass2-test
#SBATCH --partition=elixir-interno
#SBATCH --nodelist=elixir-lj-gpu-01.elixir.ul.si
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=/home/jakob/luka/runs/pass2_test_complex/slurm-%j.out
#SBATCH --error=/home/jakob/luka/runs/pass2_test_complex/slurm-%j.err

set -euo pipefail

PROJECT_DIR="${HOME}/luka/code/XPlainVerse-ACMChallenge"
DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
LJ_APPTAINER_IMAGE="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${DEFAULT_REPO_LC}-lj:latest}"
APPTAINER_BIND="${LJ_APPTAINER_BIND:-${HOME}:${HOME},/primoz:/primoz},${PROJECT_DIR}:/workspace/XPlainVerse-ACMChallenge"

PASS1_PRED="${PASS1_PRED:-/home/jakob/luka/runs/pass1_test_tta/20260605-170149/predictions.parquet}"
ADAPTERS="${ADAPTERS:-/home/jakob/luka/runs/vlm_v2_grpo/job_48855/checkpoint-26400}"
_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass2_test_complex/${_TS}}"

mkdir -p "${OUT_DIR}" "${HOME}/luka/runs/pass2_test_complex"
cd "${PROJECT_DIR}"

export PYTHONNOUSERSITE=1
export PASS1_PRED ADAPTERS OUT_DIR SHARD_COUNT=4
export _PASS2_TEST_IN_CONTAINER=1

echo "=== sbatch_pass2_test_complex_lj ==="
echo "job:        ${SLURM_JOB_ID}"
echo "adapters:   ${ADAPTERS}"
echo "pass1_pred: ${PASS1_PRED}"
echo "out_dir:    ${OUT_DIR}"
echo "started:    $(date -Is)"

_INNER='export _PASS2_TEST_IN_CONTAINER=1 PYTHONNOUSERSITE=1 PASS1_PRED='"$(printf '%q' "${PASS1_PRED}")"' ADAPTERS='"$(printf '%q' "${ADAPTERS}")"' OUT_DIR='"$(printf '%q' "${OUT_DIR}")"' SHARD_COUNT=4; cd /workspace/XPlainVerse-ACMChallenge; chmod +x scripts/run_pass2_test_complex_lj.sh'

for SHARD in 0 1 2 3; do
  apptainer exec --nv \
    -B "${APPTAINER_BIND}" \
    --env PYTHONNOUSERSITE=1 \
    --env _PASS2_TEST_IN_CONTAINER=1 \
    --env CUDA_VISIBLE_DEVICES="${SHARD}" \
    --env SHARD_ID="${SHARD}" \
    --env SHARD_COUNT=4 \
    --env PASS1_PRED="${PASS1_PRED}" \
    --env ADAPTERS="${ADAPTERS}" \
    --env OUT_DIR="${OUT_DIR}" \
    --pwd /workspace/XPlainVerse-ACMChallenge \
    "${LJ_APPTAINER_IMAGE}" \
    bash scripts/run_pass2_test_complex_lj.sh &
done

wait

apptainer exec --nv \
  -B "${APPTAINER_BIND}" \
  --env PYTHONNOUSERSITE=1 \
  --env _PASS2_TEST_IN_CONTAINER=1 \
  --env MERGE_ONLY=1 \
  --env SHARD_ID=0 \
  --env SHARD_COUNT=4 \
  --env PASS1_PRED="${PASS1_PRED}" \
  --env ADAPTERS="${ADAPTERS}" \
  --env OUT_DIR="${OUT_DIR}" \
  --pwd /workspace/XPlainVerse-ACMChallenge \
  "${LJ_APPTAINER_IMAGE}" \
  bash scripts/run_pass2_test_complex_lj.sh

echo "=== pass2 test job ${SLURM_JOB_ID} done $(date -Is) ==="
