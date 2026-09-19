#!/usr/bin/env bash
# Stage GenEval inputs for an evaluation-only session.
#
# export_geneval.py writes geneval_inputs/<method>/<id>.png as symlinks into this project's
# outputs tree and the evaluator asserts CUDA, which is why scoring normally happens inside the
# generating session. When that session is already over, its output is attached as a kernel
# source instead (same owner) and this script copies the measured tree into the project, so
# export_geneval.py can rebuild the symlinks here without regenerating a single image.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${STAGE_EXP_DIR:-$ROOT/exps/single_stage_validation_553}"
INPUTS="${GENEAL_INPUTS_ROOT:-/kaggle/input}"

echo "[stage] searching $INPUTS for a generated outputs tree"
SRC=""
while IFS= read -r candidate; do
  if [ -n "$(find "$candidate" -name '*.png' -print -quit 2>/dev/null)" ]; then SRC="$candidate"; break; fi
done < <(find "$INPUTS" -maxdepth 7 -type d -name outputs 2>/dev/null | sort)
[ -n "$SRC" ] || { echo "[stage] FATAL: no outputs directory containing PNGs under $INPUTS"; ls -la "$INPUTS"; exit 1; }
SRC_EXP="$(dirname "$SRC")"
echo "[stage] source experiment directory: $SRC_EXP"

mkdir -p "$EXP"
for name in outputs metadata metrics protocol_manifest.json prompts_geneval_all_553.jsonl FROZEN_SCHEDULE.json; do
  if [ -e "$SRC_EXP/$name" ]; then
    rm -rf "$EXP/$name"
    cp -r "$SRC_EXP/$name" "$EXP/$name"
    echo "[stage] copied $name ($(find "$EXP/$name" -type f 2>/dev/null | wc -l) files)"
  fi
done
echo "[stage] pngs=$(find "$EXP/outputs" -name '*.png' 2>/dev/null | wc -l) metadata=$(find "$EXP/metadata" -type f 2>/dev/null | wc -l) metrics=$(find "$EXP/metrics" -type f 2>/dev/null | wc -l)"

python "$EXP/export_geneval.py"
echo "[stage] geneval inputs: $(find "$EXP/geneval_inputs" 2>/dev/null | wc -l)"
