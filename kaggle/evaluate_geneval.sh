#!/usr/bin/env bash
# Run the official GenEval evaluator over the images in this session, then delete the /tmp
# environment so nothing bulky survives into the run output.
#
# Three dependencies are deliberate:
#   * geneval/evaluation/evaluate_images.py asserts CUDA is available, so this only runs in a
#     GPU session (the install itself is happy on CPU);
#   * exps/single_stage_validation_553/export_geneval.py links geneval_inputs/<method>/<id>.png
#     to outputs/<method>/<id>.png, so evaluation needs that outputs tree - either generated
#     here or staged from the generating session by kaggle/stage_geneval_inputs.sh;
#   * the environment came from kaggle/install_geneval_env.sh and lives under /tmp.
#
# Two knobs exist because one method took longer than the 2 h step timeout on a T4 in
# run tqk-eval-553-geneval-260919-0908:
#   GENEAL_METHODS   which methods to score (default: psp ours)
#   GENEAL_PARALLEL  1 = one method per GPU (default), 0 = sequentially
# Each method writes its own log plus a five-minute progress line, so a timeout can be sized
# from evidence instead of guessed at.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${GENEAL_EXP_DIR:-$ROOT/exps/single_stage_validation_553}"
ENV_DIR="${GENEAL_ENV_DIR:-/tmp/geneval_env}"
WEIGHTS_DIR="${GENEAL_WEIGHTS_DIR:-/tmp/geneval_weights}"
MANIFEST="/tmp/geneval_env_manifest.json"
PY="$ENV_DIR/bin/python"
METHODS="${GENEAL_METHODS:-psp ours}"
PARALLEL="${GENEAL_PARALLEL:-1}"

[ -x "$PY" ] || { echo "[geneval] no environment at $ENV_DIR; run kaggle/install_geneval_env.sh first"; exit 1; }
"$PY" -c "import torch; assert torch.cuda.is_available(), 'GenEval evaluator requires CUDA'"
mkdir -p "$EXP/geneval_results" "$EXP/metrics" "$EXP/logs"
CONFIG=$("$PY" -c "import json;print(json.load(open('$MANIFEST'))['mmdet_config'])")
echo "[geneval] model config $CONFIG | weights $WEIGHTS_DIR | methods $METHODS | parallel=$PARALLEL"

NGPU=$("$PY" -c "import torch;print(torch.cuda.device_count())")
if [ "$PARALLEL" = "1" ] && [ "$NGPU" -lt 2 ]; then
  echo "[geneval] only $NGPU GPU(s) visible; falling back to sequential evaluation"
  PARALLEL=0
fi

# Progress watcher: the evaluator appends to its JSONL silently, so without this a step that
# times out leaves no evidence of how far it got.
watch_progress() {
  local method="$1" out="$2" start=$SECONDS n
  while true; do
    sleep 300 || return 0
    n=0
    [ -f "$out" ] && n=$(wc -l < "$out" 2>/dev/null || echo 0)
    echo "[geneval] progress $method lines=$n elapsed=$((SECONDS - start))s"
  done
}

run_method() {
  local method="$1" out="$EXP/geneval_results/$1.jsonl" log="$EXP/logs/geneval_$1.log"
  local start rc n
  start=$(date +%s)
  echo "[geneval] start $method -> $out (log $log, cuda=${CUDA_VISIBLE_DEVICES:-all})"
  set +e
  "$PY" "$ROOT/geneval/evaluation/evaluate_images.py" "$EXP/geneval_inputs/$method" \
    --outfile "$out" --model-config "$CONFIG" --model-path "$WEIGHTS_DIR" >> "$log" 2>&1
  rc=$?
  set -e
  n=0
  [ -f "$out" ] && n=$(wc -l < "$out" 2>/dev/null || echo 0)
  echo "[geneval] $method finished rc=$rc lines=$n elapsed=$(( $(date +%s) - start ))s"
  if [ "$rc" -ne 0 ]; then
    echo "[geneval] $method log tail:"
    tail -n 30 "$log" || true
  fi
  return "$rc"
}

status=0
watchers=''
if [ "$PARALLEL" = "1" ]; then
  echo "[geneval] parallel mode on $NGPU GPUs"
  position=0
  pids=""
  for method in $METHODS; do
    watch_progress "$method" "$EXP/geneval_results/$method.jsonl" >> "$EXP/logs/geneval_$method.log" 2>&1 &
    watchers="$watchers $!"
    CUDA_VISIBLE_DEVICES="$position" run_method "$method" &
    pids="$pids $!"
    position=$((position + 1))
  done
  for pid in $pids; do
    if ! wait "$pid"; then status=1; fi
  done
else
  for method in $METHODS; do
    watch_progress "$method" "$EXP/geneval_results/$method.jsonl" >> "$EXP/logs/geneval_$method.log" 2>&1 &
    watchers="$watchers $!"
    if ! run_method "$method"; then status=1; fi
  done
fi
for pid in $watchers; do kill "$pid" 2>/dev/null || true; done

for method in $METHODS; do
  out="$EXP/geneval_results/$method.jsonl"
  if [ -s "$out" ]; then
    echo "[geneval] summary $method ($(wc -l < "$out") scored images)"
    "$PY" "$ROOT/geneval/evaluation/summary_scores.py" "$out" | tee "$EXP/metrics/geneval_${method}_summary.txt" || status=1
  else
    echo "[geneval] $method produced no results"
    status=1
  fi
done

rm -rf "$ENV_DIR" "$WEIGHTS_DIR" "$MANIFEST"
echo "[geneval] removed the /tmp environment; step status=$status"
df -h /tmp /kaggle/working || true
exit "$status"
