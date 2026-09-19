#!/usr/bin/env bash
# Stage a finished run's per-prompt results into this project tree.
#
# export_geneval.py writes geneval_inputs/<method>/<id>.png as symlinks into this project's
# outputs tree and the evaluator asserts CUDA, which is why scoring normally happens inside the
# generating session. When that session is over, its output is attached as a kernel source
# (same owner) and this script copies the measured tree into the project:
#
#   STAGE_ITEMS       which directories/files to copy
#   STAGE_RUN_EXPORT  1 = also rebuild geneval_inputs via export_geneval.py
#   STAGE_MIN_FILES   reject a source tree with fewer files than this
#
# A mounted kernel output contains every experiment directory of that checkout, including
# small development runs, so the source is chosen by file count and validated instead of
# taking the first directory that happens to be named outputs/metadata.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${STAGE_EXP_DIR:-$ROOT/exps/single_stage_validation_553}"
INPUTS="${GENEAL_INPUTS_ROOT:-/kaggle/input}"
ITEMS="${STAGE_ITEMS:-outputs metadata metrics protocol_manifest.json prompts_geneval_all_553.jsonl FROZEN_SCHEDULE.json}"
RUN_EXPORT="${STAGE_RUN_EXPORT:-1}"
MIN_FILES="${STAGE_MIN_FILES:-100}"

echo "[stage] searching $INPUTS for a generated experiment tree"
SRC_EXP=""
BEST_N=0
while IFS= read -r candidate; do
  exp_dir="$(dirname "$candidate")"
  [ -d "$exp_dir" ] || continue
  n=$(find "$exp_dir" -maxdepth 2 \( -name '*.png' -o -name '*.json' \) 2>/dev/null | wc -l)
  echo "[stage] candidate $exp_dir -> $n files"
  if [ "$n" -gt "$BEST_N" ]; then BEST_N="$n"; SRC_EXP="$exp_dir"; fi
done < <(find "$INPUTS" -maxdepth 7 -type d \( -name outputs -o -name metadata \) 2>/dev/null | sort)
[ -n "$SRC_EXP" ] || { echo "[stage] FATAL: no outputs/metadata directory found under $INPUTS"; ls -la "$INPUTS"; exit 1; }
[ "$BEST_N" -ge "$MIN_FILES" ] || { echo "[stage] FATAL: best candidate $SRC_EXP has only $BEST_N files (< $MIN_FILES)"; exit 1; }
echo "[stage] source experiment directory: $SRC_EXP ($BEST_N files)"

mkdir -p "$EXP"
copied=0
for name in $ITEMS; do
  if [ -e "$SRC_EXP/$name" ]; then
    rm -rf "$EXP/$name"
    cp -r "$SRC_EXP/$name" "$EXP/$name"
    copied=$((copied + 1))
    echo "[stage] copied $name ($(find "$EXP/$name" -type f 2>/dev/null | wc -l) files)"
  else
    echo "[stage] absent at source: $name"
  fi
done
[ "$copied" -gt 0 ] || { echo "[stage] FATAL: none of the requested items exist at $SRC_EXP"; exit 1; }
PNGS=$(find "$EXP/outputs" -name '*.png' 2>/dev/null | wc -l)
MD=$(find "$EXP/metadata" -type f 2>/dev/null | wc -l)
MT=$(find "$EXP/metrics" -type f 2>/dev/null | wc -l)
GV=$(find "$EXP/geneval_results" -type f 2>/dev/null | wc -l)
echo "[stage] pngs=$PNGS metadata=$MD metrics=$MT geneval=$GV"

if [ "$RUN_EXPORT" = "1" ]; then
  [ "$PNGS" -gt 0 ] || { echo "[stage] FATAL: no PNGs to score in $EXP/outputs; wrong source tree?"; exit 1; }
  python "$EXP/export_geneval.py"
  echo "[stage] geneval inputs: $(find "$EXP/geneval_inputs" 2>/dev/null | wc -l)"
else
  echo "[stage] skipping export_geneval.py (STAGE_RUN_EXPORT=$RUN_EXPORT)"
fi
