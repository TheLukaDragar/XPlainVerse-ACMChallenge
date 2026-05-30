#!/usr/bin/env bash
# Download XPlainVerse test image shards from HuggingFace and extract to test/images/.
#
# HF layout (released May 2026):
#   test/test_images.tar.part-*
# Join + extract (per dataset README):
#   cd XPlainVerse
#   cat test/test_images.tar.part-* > test/test_images.tar
#   tar -xf test/test_images.tar -C test
#
# Usage:
#   ./scripts/download_extract_xplainverse_test.sh
#   XPLAINVERSE_DATA_ROOT=/path/to/data/XPlainVerse ./scripts/download_extract_xplainverse_test.sh

set -euo pipefail

BASE="${XPLAINVERSE_DATA_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/data/XPlainVerse}"
REPO_ID="${XPLAINVERSE_REPO_ID:-Abhijeet8901/XPlainVerse}"
KEEP_TAR="${KEEP_TAR:-0}"
MARKER="${BASE}/test/.xplainverse_unpack_done"

mkdir -p "${BASE}/test"

echo "=== XPlainVerse test images ==="
echo "data root:  ${BASE}"
echo "hf repo:    ${REPO_ID}"
echo "started:    $(date -Is)"
echo

echo "[$(date -Is)] downloading test/test_images.tar.part-* from HuggingFace..." >&2
python3 - <<PY
from huggingface_hub import HfApi, hf_hub_download
import os

repo_id = "${REPO_ID}"
local_dir = "${BASE}"
api = HfApi()

files = sorted(
    f for f in api.list_repo_files(repo_id, repo_type="dataset")
    if f.startswith("test/test_images.tar.part-")
)
if not files:
    raise SystemExit("error: no test/test_images.tar.part-* in repo")

print(f"found {len(files)} part file(s) in repo", flush=True)
for i, relpath in enumerate(files, 1):
    dest = os.path.join(local_dir, relpath)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"[{i}/{len(files)}] skip existing {relpath} ({os.path.getsize(dest)} bytes)", flush=True)
        continue
    print(f"[{i}/{len(files)}] downloading {relpath} ...", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=relpath,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[{i}/{len(files)}] ok {relpath}", flush=True)

print("download: ok")
PY

shopt -s nullglob
parts=( "${BASE}/test"/test_images.tar.part-* )
if ((${#parts[@]} == 0)); then
  echo "error: no test_images.tar.part-* under ${BASE}/test" >&2
  exit 1
fi
echo "found ${#parts[@]} tar part(s)" >&2

if [[ -f "${MARKER}" && "${FORCE_EXTRACT:-0}" != "1" ]]; then
  echo "skip extract (already done): ${MARKER}" >&2
  exit 0
fi

combined="${BASE}/test/test_images.tar"
echo "[$(date -Is)] joining parts -> ${combined}" >&2
mapfile -t sorted < <(printf '%s\n' "${parts[@]}" | sort -V)
cat "${sorted[@]}" > "${combined}"

echo "[$(date -Is)] extracting ${combined} -> ${BASE}/test/" >&2
tar -xf "${combined}" -C "${BASE}/test"

touch "${MARKER}"
echo "[$(date -Is)] extract done; marker ${MARKER}" >&2

if [[ "${KEEP_TAR}" != "1" ]]; then
  echo "[$(date -Is)] removing combined tar ${combined}" >&2
  rm -f "${combined}"
fi

if [[ -d "${BASE}/test/images" ]]; then
  n=$(find "${BASE}/test/images" -type f 2>/dev/null | wc -l)
  echo "test/images file count: ${n}" >&2
fi

echo "Done at $(date -Is)" >&2
