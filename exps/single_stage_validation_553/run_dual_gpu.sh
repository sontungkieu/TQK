#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-0}"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${EXP_DIR}"/{logs,outputs/psp,outputs/ours,metadata/gpu0,metadata/gpu1}
extra=()
if [[ "${LIMIT}" != "0" ]]; then
  extra=(--limit-prompts "${LIMIT}")
fi

nvidia-smi dmon -s pucm -d 5 -o DT > "${EXP_DIR}/logs/nvidia_dmon.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=0 python "${EXP_DIR}/run_worker.py" \
  --worker-index 0 --num-workers 2 --resume "${extra[@]}" \
  >> "${EXP_DIR}/logs/gpu0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 python "${EXP_DIR}/run_worker.py" \
  --worker-index 1 --num-workers 2 --resume "${extra[@]}" \
  >> "${EXP_DIR}/logs/gpu1.log" 2>&1 &
pid1=$!

status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
exit "${status}"

