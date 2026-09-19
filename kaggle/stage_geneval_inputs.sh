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
#
# Two shapes are used: an evaluation-only session copies outputs too and rebuilds the symlinks;
# a report-only session copies just the per-prompt results and skips export_geneval.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${STAGE_EXP_DIR:-$ROOT/exps/single_stage_validation_553}"
INPUTS="${GENEAL_INPUTS_ROOT:-/kaggle/input}"
ITEMS="${STAGE_ITEMS:-outputs metadata metrics protocol_manifest.json prompts_geneval_all_553.jsonl FROZEN_SCHEDULE.json}"
RUN_EXPORT="${STAGE_RUN_EXPORT:-1}"

echo "[stage] searching $INPUTS for a generated experiment tree"
SRC_EXP=""
while IFS= read -r candidate; do
  if [ -n "$(find "$candidate" -maxdepth 3 -type f -print -quit 2>/dev/null)" ]; then
    SRC_EXP="$(dirname "$candidate")"
    break
  fi
done < <(find "$INPUTS" -maxdepth 7 -type d \( -name outputs -o -name metadata \) 2>/dev/null | sort)
[ -n "$SRC_EXP" ] || { echo "[stage] FATAL: no outputs/metadata directory found under $INPUTS"; ls -la "$INPUTS"; exit 1; }
echo "[stage] source experiment directory: $SRC_EXP"

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
echo "[stage] pngs=$(find "$EXP/outputs" -name '*.png' 2>/dev/null | wc -l) metadata=$(find "$EXP/metadata" -type f 2>/dev/null | wc -l) metrics=$(find "$EXP/metrics" -type f 2>/dev/null | wc -l) geneval=$(find "$EXP/geneval_results" -type f 2>/dev/null | wc -l)"

if [ "$RUN_EXPORT" = "1" ]; then
  python "$EXP/export_geneval.py"
  echo "[stage] geneval inputs: $(find "$EXP/geneval_inputs" 2>/dev/null | wc -l)"
else
  echo "[stage] skipping export_geneval.py (STAGE_RUN_EXPORT=$RUN_EXPORT)"
fi
