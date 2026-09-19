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

cd "$EXP"
items=""
for name in outputs metadata metrics geneval_inputs protocol_manifest.json prompts_geneval_all_553.jsonl FROZEN_SCHEDULE.json; do
  [ -e "$name" ] && items="$items $name"
done
[ -n "$items" ] || { echo "[pack] nothing to pack in $EXP"; exit 1; }

echo "[pack] packing:$items"
echo "[pack] disk before:"; df -h "$(dirname "$OUT")" | tail -1
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$OUT" $items
ls -lh "$OUT"
echo "[pack] done"
