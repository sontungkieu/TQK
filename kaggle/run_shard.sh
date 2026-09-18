#!/usr/bin/env bash
# Shard-aware runner for the TQK experiments on Kaggle (2x T4 or any 2-GPU node).
#
# The published run_all.sh asserts two RTX 4090s and owns the whole protocol on one
# machine. This runner only owns one shard: it starts the experiment worker for each
# requested worker index on one CUDA device and waits for all of them.
#
#   kaggle/run_shard.sh --phase bank --num-workers 2 --worker-indices 0,1
#   kaggle/run_shard.sh --phase bank --num-workers 8 --worker-indices 4,5 --limit-prompts 2
#   kaggle/run_shard.sh --phase eval --num-workers 2 --worker-indices 0,1
set -euo pipefail

PHASE=""
NUM_WORKERS=""
WORKER_INDICES=""
LIMIT="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --worker-indices) WORKER_INDICES="$2"; shift 2 ;;
    --limit-prompts) LIMIT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$PHASE" && -n "$NUM_WORKERS" && -n "$WORKER_INDICES" ]] || {
  echo "usage: run_shard.sh --phase bank|eval --num-workers N --worker-indices a,b [--limit-prompts K] [--dry-run]" >&2
  exit 2
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$PHASE" in
  bank)
    EXP="$ROOT/exps/single_stage_calibration"
    WORKER="$EXP/generate_bank_worker.py"
    ;;
  eval)
    EXP="$ROOT/exps/single_stage_validation_553"
    WORKER="$EXP/run_worker.py"
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

mkdir -p "$EXP/logs"

# T4 (16 GiB) memory profile, chosen from kaggle/bench_decode.py measurements on this
# GPU: decode of the 25-candidate pool takes 4.64 s at chunk 1 and 6.69 s at chunk 2,
# while chunks 4/6/8/12/16/25 all die with torch.cuda.OutOfMemoryError even for decode
# alone. So diffusers' VAE slicing (one image per call) is both the only fitting option
# and the fastest one; PSP_VAE_CHUNK stays 0, i.e. the published single reward call.
# On a 24 GiB card set PSP_VAE_SLICING=0 PSP_VAE_CHUNK=0 to reproduce the paper exactly.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PSP_VAE_SLICING="${PSP_VAE_SLICING:-1}"
export PSP_VAE_CHUNK="${PSP_VAE_CHUNK:-0}"
export PSP_ATTENTION_SLICING="${PSP_ATTENTION_SLICING:-0}"

echo "[shard] phase=$PHASE num_workers=$NUM_WORKERS worker_indices=$WORKER_INDICES limit=$LIMIT"
echo "[shard] memory: vae_chunk=$PSP_VAE_CHUNK vae_slicing=$PSP_VAE_SLICING attention_slicing=$PSP_ATTENTION_SLICING alloc=$PYTORCH_CUDA_ALLOC_CONF"
echo "[shard] python=$(command -v python) version=$(python -V 2>&1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/[shard] gpu /'

extra=""
if [[ "$LIMIT" != "0" ]]; then
  extra="--limit-prompts $LIMIT"
fi

pids=""
position=0
for worker_index in $(printf '%s' "$WORKER_INDICES" | tr ',' ' '); do
  log="$EXP/logs/shard_"$PHASE"_worker"$worker_index".log"
  echo "[shard] worker=$worker_index cuda_visible_devices=$position log=$log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[shard]   CUDA_VISIBLE_DEVICES=$position python $WORKER --worker-index $worker_index --num-workers $NUM_WORKERS --resume $extra"
  else
    CUDA_VISIBLE_DEVICES="$position" python "$WORKER" \
      --worker-index "$worker_index" --num-workers "$NUM_WORKERS" --resume $extra \
      >> "$log" 2>&1 &
    pids="$pids $!"
  fi
  position=$((position + 1))
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

status=0
for pid in $pids; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" != "0" ]]; then
  echo "[shard] at least one worker failed; last log lines:" >&2
  for log in "$EXP"/logs/shard_"$PHASE"_worker*.log; do
    [[ -f "$log" ]] || continue
    echo "--- $log" >&2
    tail -n 20 "$log" >&2
  done
fi
exit "$status"
