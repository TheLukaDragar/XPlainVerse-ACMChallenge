#!/usr/bin/env bash
# Download Hugging Face vision backbones for Pass-1 (Stage 1) binary classifier training.
#
# Core (default): DINOv3-Large + SigLIP2-SO400M — see research/04_strategy.md
# Optional extras: DINOv3-Huge+ and SigLIP2-giant (larger ensemble partners)
#
# Usage:
#   ./scripts/download_pass1_models.sh              # core models only
#   EXTRAS=1 ./scripts/download_pass1_models.sh     # also download larger backbones
#   SELECT=dinov3-large ./scripts/download_pass1_models.sh
#
# Env:
#   OUT_DIR          — default: <repo>/baseline_models/pass1
#   HF_TOKEN           — HuggingFace token (required for gated Meta DINOv3 repos)
#   HUGGING_FACE_HUB_TOKEN — alias for HF_TOKEN
#   HF_HUB_ENABLE_HF_TRANSFER=1 — faster downloads if hf_transfer is installed
#
# Gated models (Meta license — accept terms at the HF repo page, then login):
#   facebook/dinov3-vitl16-pretrain-lvd1689m
#   facebook/dinov3-vith16plus-pretrain-lvd1689m
#
#   huggingface-cli login
#   HF_TOKEN=hf_... ./scripts/download_pass1_models.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/baseline_models/pass1}"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli)
elif command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf)
else
  echo "error: install huggingface_hub (huggingface-cli or hf not found)" >&2
  exit 1
fi

declare -A MODELS=(
  [dinov3-large]="facebook/dinov3-vitl16-pretrain-lvd1689m"
  [siglip2-so400m]="google/siglip2-so400m-patch14-384"
  [dinov3-hugeplus]="facebook/dinov3-vith16plus-pretrain-lvd1689m"
  [siglip2-giant]="google/siglip2-giant-opt-patch16-384"
)
declare -A GATED_MODELS=(
  [dinov3-large]=1
  [dinov3-hugeplus]=1
)

DEFAULT_MODELS=(dinov3-large siglip2-so400m)
EXTRA_MODELS=(dinov3-hugeplus siglip2-giant)

resolve_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "${HF_TOKEN}"
    return
  fi
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "${HUGGING_FACE_HUB_TOKEN}"
    return
  fi
  if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    cat "${HOME}/.cache/huggingface/token"
    return
  fi
  echo ""
}

download_one() {
  local key="$1"
  local repo="${MODELS[$key]}"
  local dest="${OUT_DIR}/${key}"
  local token
  token="$(resolve_hf_token)"

  if [[ -f "${dest}/config.json" ]]; then
    echo "== skip ${key} (${repo}) — already at ${dest}"
    return 0
  fi

  if [[ -n "${GATED_MODELS[$key]+x}" && -z "${token}" ]]; then
    echo "error: ${key} (${repo}) is gated — set HF_TOKEN or run 'huggingface-cli login'" >&2
    echo "       accept Meta terms at https://huggingface.co/${repo}" >&2
    return 1
  fi

  echo "== download ${key}"
  echo "   repo : ${repo}"
  echo "   dest : ${dest}"
  mkdir -p "${dest}"

  local -a cmd=("${HF_CLI[@]}" download "${repo}" --local-dir "${dest}")
  if [[ -n "${token}" ]]; then
    cmd+=(--token "${token}")
  fi

  if ! "${cmd[@]}"; then
    if [[ -n "${GATED_MODELS[$key]+x}" ]]; then
      echo "error: download failed for gated model ${repo}" >&2
      echo "       1) log in: huggingface-cli login" >&2
      echo "       2) accept terms: https://huggingface.co/${repo}" >&2
      echo "       3) retry with HF_TOKEN=hf_... $0" >&2
    fi
    return 1
  fi
  echo
}

select_models() {
  local failed=0
  local -a to_download=()

  if [[ -n "${SELECT:-}" ]]; then
    IFS=',' read -r -a to_download <<< "${SELECT}"
    for i in "${!to_download[@]}"; do
      to_download[$i]="${to_download[$i]// /}"
    done
  else
    to_download=("${DEFAULT_MODELS[@]}")
    if [[ "${EXTRAS:-0}" == "1" ]]; then
      to_download+=("${EXTRA_MODELS[@]}")
    fi
  fi

  for key in "${to_download[@]}"; do
    if [[ -z "${MODELS[$key]+x}" ]]; then
      echo "error: unknown model key '${key}'. Valid: ${!MODELS[*]}" >&2
      exit 1
    fi
    download_one "${key}" || failed=1
  done

  return "${failed}"
}

echo "Pass-1 model download"
echo "  output dir : ${OUT_DIR}"
echo "  hf cli     : ${HF_CLI[*]}"
echo "  extras     : ${EXTRAS:-0}"
echo

if ! select_models; then
  echo "Some downloads failed (see above)." >&2
  exit 1
fi

echo "Done. Local paths:"
for key in "${!MODELS[@]}"; do
  dest="${OUT_DIR}/${key}"
  if [[ -f "${dest}/config.json" ]]; then
    echo "  ${key} -> ${dest}  (${MODELS[$key]})"
  fi
done

echo
echo "Train with local weights, e.g.:"
echo "  BACKBONE=${OUT_DIR}/dinov3-large research/experiments/02_pass1_classifier/run.sh"
