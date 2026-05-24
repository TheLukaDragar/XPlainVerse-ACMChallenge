#!/usr/bin/env bash
# Unpack XPlainVerse tar segments: explanations (JSON) + image shards (val + train).
#
# Expected layout under data root: each listed subdir contains *.tar.part-* files.
#
# Usage:
#   ./scripts/extract_xplainverse_explanations.sh
#   XPLAINVERSE_DATA_ROOT=/path/to/data/XPlainVerse ./scripts/extract_xplainverse_explanations.sh
#
# Train images are large; to unpack only val (smoke tests):
#   UNPACK_TRAIN_IMAGES=0 ./scripts/extract_xplainverse_explanations.sh
#
# Re-running tar on an already extracted tree is extremely slow. By default we skip if
#   <subdir>/.xplainverse_unpack_done exists. One-time migration marks val/complex complete
#   when ~110k JSON files already exist. Override with FORCE_EXTRACT=1.

set -euo pipefail

# Default matches shared workspace layout next to this repo checkout.
BASE="${XPLAINVERSE_DATA_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/data/XPlainVerse}"
UNPACK_TRAIN_IMAGES="${UNPACK_TRAIN_IMAGES:-1}"

migrate_markers() {
  # If data was extracted before this script wrote markers, create markers once.
  [[ "${MIGRATE_MARKERS:-1}" == "0" ]] && return 0
  local d n
  d="${BASE}/val/complex_explanations"
  if [[ -d "$d" && ! -f "$d/.xplainverse_unpack_done" ]]; then
    n=$(find "$d" -type f -name '*.json' 2>/dev/null | wc -l)
    if [[ "$n" -ge 109500 ]]; then
      touch "$d/.xplainverse_unpack_done"
      echo "migration: marked ${d##*/} complete (${n} json files)" >&2
    fi
  fi
}

unpack_dir() {
  local subdir=$1
  local d="${BASE}/${subdir}"
  if [[ ! -d "$d" ]]; then
    echo "skip (missing dir): ${subdir}" >&2
    return 0
  fi

  if [[ "${FORCE_EXTRACT:-0}" != "1" ]] && [[ -f "$d/.xplainverse_unpack_done" ]]; then
    echo "skip (already unpacked): ${subdir}" >&2
    return 0
  fi

  (
    cd "$d"
    shopt -s nullglob
    local parts=( *.tar.part-* )
    if ((${#parts[@]} == 0)); then
      echo "skip (no *.tar.part-* in): ${subdir}" >&2
      return 0
    fi

    echo "==> ${subdir} (${#parts[@]} segment(s))" >&2
    echo "[$(date -Is)] start tar ${subdir}" >&2
    if ((${#parts[@]} == 1)); then
      tar -xf "${parts[0]}"
    else
      mapfile -t sorted < <(printf '%s\n' "${parts[@]}" | sort -V)
      cat "${sorted[@]}" | tar -xf -
    fi
    touch .xplainverse_unpack_done
    echo "[$(date -Is)] done  tar ${subdir}" >&2
  )
}

migrate_markers

echo "Data root: ${BASE}" >&2
unpack_dir "val/complex_explanations"
unpack_dir "val/simple_explanations"
unpack_dir "train/complex_explanations"
unpack_dir "train/simple_explanations"
unpack_dir "val/images"
if [[ "${UNPACK_TRAIN_IMAGES}" == "1" ]]; then
  unpack_dir "train/images"
else
  echo "skip train/images (UNPACK_TRAIN_IMAGES=0)" >&2
fi

echo "Done. Extracted under: ${BASE} (see dataset README for layout)." >&2
