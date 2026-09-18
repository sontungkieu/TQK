#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAL="${ROOT}/exps/single_stage_calibration"
VAL="${ROOT}/exps/single_stage_validation_553"
GEN_VENV="${GEN_VENV:-/workspace/psp_sd15_env}"
GENEVAL_VENV="${GENEVAL_VENV:-/workspace/psp_geneval_env}"
GENEVAL_MMDET="${GENEVAL_MMDET:-/workspace/mmdetection-v2.28.2}"
GENEVAL_WEIGHTS="${GENEVAL_WEIGHTS:-/workspace/geneval_weights}"

mkdir -p "${CAL}"/{logs,bank_raw/gpu0,bank_raw/gpu1,replay,plots,metrics} \
  "${VAL}"/{logs,outputs/psp,outputs/ours,metadata/gpu0,metadata/gpu1,metrics,geneval_results}
export HF_HOME="/workspace/.hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/workspace/.torch"
export XDG_CACHE_HOME="/workspace/.cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 2 ]]
[[ "${gpu_names[0]}" == "NVIDIA GeForce RTX 4090" ]]
[[ "${gpu_names[1]}" == "NVIDIA GeForce RTX 4090" ]]
nvidia-smi > "${CAL}/logs/nvidia_smi_preflight.txt"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' && {
  echo "GPU already has a compute process; refusing to mix workloads" >&2
  exit 1
} || true
[[ -x "${GEN_VENV}/bin/python" ]] || {
  echo "Missing established generation environment ${GEN_VENV}" >&2
  exit 1
}
export PATH="${GEN_VENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GEN_VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

"${GEN_VENV}/bin/python" - <<'PY'
import importlib.util, subprocess, sys
missing = [name for name in ("pyarrow",) if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
print("analysis dependencies ready")
PY

echo "[protocol] verify fixed independent prompts and schedule grid"
"${GEN_VENV}/bin/python" "${CAL}/select_prompts.py" --verify-existing
"${GEN_VENV}/bin/python" "${CAL}/budget_check.py"

"${GEN_VENV}/bin/python" "${CAL}/record_run_info.py" --phase phase1

echo "[phase1 preflight] one calibration prompt per GPU where available"
"${CAL}/run_dual_gpu.sh" 2
"${GEN_VENV}/bin/python" "${CAL}/validate_bank.py" --expected-prompts 2 \
  | tee "${CAL}/logs/preflight_validation.log"

echo "[phase1 full] 120 independent prompts x 25 complete trajectories"
phase1_started="$(date +%s)"
"${CAL}/run_dual_gpu.sh" 0
phase1_ended="$(date +%s)"
"${GEN_VENV}/bin/python" "${CAL}/validate_bank.py" --expected-prompts 120 \
  | tee "${CAL}/logs/bank_validation.log"
"${GEN_VENV}/bin/python" - "${CAL}/metrics/run_info.json" "${phase1_started}" "${phase1_ended}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]); row = json.loads(path.read_text())
row["phase1_started_unix"] = int(sys.argv[2]); row["phase1_ended_unix"] = int(sys.argv[3])
row["phase1_wall_s"] = int(sys.argv[3]) - int(sys.argv[2])
path.write_text(json.dumps(row, indent=2) + "\n")
PY

echo "[phase1 replay] search 80, validate top-3 on untouched 40, freeze once"
"${GEN_VENV}/bin/python" "${CAL}/analyze_bank.py" | tee "${CAL}/logs/analysis.log"
"${GEN_VENV}/bin/python" "${CAL}/plot_results.py" | tee "${CAL}/logs/plots.log"
echo "FROZEN SCHEDULE"
cat "${CAL}/FROZEN_SCHEDULE.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "${CAL}/COMPLETE"

echo "[phase2 protocol] checksum-lock frozen schedule and all 553 GenEval prompts"
"${GEN_VENV}/bin/python" "${VAL}/prepare_protocol.py" | tee "${VAL}/logs/protocol.log"
"${GEN_VENV}/bin/python" "${VAL}/budget_check.py" | tee "${VAL}/logs/budget.log"

echo "[phase2 preflight] four prompts across two GPUs"
"${VAL}/run_dual_gpu.sh" 4
"${GEN_VENV}/bin/python" "${VAL}/validate_generation.py" --expected-prompts 4 \
  | tee "${VAL}/logs/preflight_validation.log"

echo "[phase2 full] fixed PSP versus frozen schedule on all 553 prompts"
phase2_started="$(date +%s)"
"${VAL}/run_dual_gpu.sh" 0
phase2_ended="$(date +%s)"
"${GEN_VENV}/bin/python" "${VAL}/validate_generation.py" --expected-prompts 553 \
  | tee "${VAL}/logs/generation_validation.log"
"${GEN_VENV}/bin/python" "${CAL}/record_run_info.py" --phase phase2 \
  --started "${phase2_started}" --ended "${phase2_ended}"

echo "[phase2 HPS] final winners only"
CUDA_VISIBLE_DEVICES=0 "${GEN_VENV}/bin/python" "${VAL}/evaluate_hps.py" \
  | tee "${VAL}/logs/hps.log"

echo "[phase2 GenEval] official final-winner evaluator"
"${GEN_VENV}/bin/python" "${VAL}/export_geneval.py"
"${ROOT}/exps/direct_sd15_psp_vs_10to2_553/setup_geneval_env.sh" \
  | tee "${VAL}/logs/geneval_setup.log"
for method in psp ours; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${GENEVAL_MMDET}:${PYTHONPATH:-}" \
    "${GENEVAL_VENV}/bin/python" "${ROOT}/geneval/evaluation/evaluate_images.py" \
    "${VAL}/geneval_inputs/${method}" \
    --outfile "${VAL}/geneval_results/${method}.jsonl" \
    --model-config "${GENEVAL_MMDET}/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
    --model-path "${GENEVAL_WEIGHTS}" \
    2> "${VAL}/logs/geneval_${method}.log"
  [[ "$(wc -l < "${VAL}/geneval_results/${method}.jsonl")" -eq 553 ]]
done

echo "[phase2 analysis] 10,000 paired bootstrap resamples and final report"
"${GEN_VENV}/bin/python" "${VAL}/aggregate.py" | tee "${VAL}/logs/aggregate.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${VAL}/COMPLETE"

tar -C "${CAL}" -czf "${CAL}/results_bundle.tar.gz" \
  README.md PROTOCOL.md PHASE1_REPORT.md FROZEN_SCHEDULE.json COMPLETE \
  prompts replay plots logs metrics bank_raw
tar -C "${VAL}" -czf "${VAL}/results_bundle.tar.gz" \
  README.md PROTOCOL.md FINAL_REPORT.md FROZEN_SCHEDULE.json protocol_manifest.json COMPLETE \
  metrics metadata logs geneval_results outputs

echo "=== FINAL EVIDENCE ==="
cat "${VAL}/FINAL_REPORT.md"
echo "COMPLETE: ${CAL}/PHASE1_REPORT.md"
echo "COMPLETE: ${VAL}/FINAL_REPORT.md"
