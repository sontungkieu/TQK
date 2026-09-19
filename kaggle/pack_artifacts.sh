#!/usr/bin/env bash
# One tarball instead of a thousand files.
#
# Kaggle's kernel-output listing endpoint (ListKernelSessionOutput) is the one that
# answers HTTP 429 after repeated large downloads, and a 553-prompt run exposes 1106 PNGs
# plus metadata. Packing the run into a single archive keeps every later fetch to one
# listing page and a handful of signed URLs, which is what the local client should cost.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${PACK_EXP_DIR:-$ROOT/exps/single_stage_validation_553}"
OUT="${PACK_OUT:-/kaggle/working/artifacts.tar.gz}"

[ -d "$EXP" ] || { echo "[pack] FATAL: experiment directory not found: $EXP"; exit 1; }
cd "$EXP"
items=""
# PACK_ITEMS lets an evaluation-only session archive just the scores it produced; the images it
# scored already live in the generating kernel's output.
#
# An `if` rather than `[ -e "$name" ] && items=...`: under `set -e` a test that fails as the last
# command of the loop body aborts the script, so one missing optional item - a FINAL_REPORT.md
# that the aggregate step has not written yet - would have destroyed the archive protecting the
# hours of generation that ran before it.
for name in ${PACK_ITEMS:-outputs metadata metrics geneval_inputs geneval_results protocol_manifest.json prompts_geneval_all_553.jsonl FROZEN_SCHEDULE.json FINAL_REPORT.md}; do
  if [ -e "$name" ]; then
    items="$items $name"
  fi
done
[ -n "$items" ] || { echo "[pack] nothing to pack in $EXP"; exit 1; }

echo "[pack] packing:$items"
echo "[pack] disk before:"; df -h "$(dirname "$OUT")" | tail -1
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$OUT" $items
ls -lh "$OUT"
echo "[pack] done"
