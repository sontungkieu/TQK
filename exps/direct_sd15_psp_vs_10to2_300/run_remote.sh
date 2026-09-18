#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/exps/direct_sd15_psp_vs_10to2_300"
GEN_VENV="${GEN_VENV:-/workspace/psp_sd15_env}"
GENEVAL_VENV="${GENEVAL_VENV:-/workspace/psp_geneval_env}"
GENEVAL_MMDET="${GENEVAL_MMDET:-/workspace/mmdetection-v2.28.2}"
GENEVAL_WEIGHTS="${GENEVAL_WEIGHTS:-/workspace/geneval_weights}"

mkdir -p "${EXP}"/{logs,outputs/psp,outputs/early10to2,metadata/gpu0,metadata/gpu1,metrics,plots,geneval_results}
RECOVERY="/workspace/psp_direct_sd15_resume_cache"
RECOVERED_GENERATION=0
if [[ -d "${RECOVERY}" ]] && \
   [[ "$(find "${RECOVERY}/outputs/psp" -name '*.png' 2>/dev/null | wc -l)" -eq 300 ]] && \
   [[ "$(find "${RECOVERY}/outputs/early10to2" -name '*.png' 2>/dev/null | wc -l)" -eq 300 ]]; then
  echo "[recovery] importing validated 300+300 winner outputs from prior tracked run"
  cp -a "${RECOVERY}/outputs/." "${EXP}/outputs/"
  cp -a "${RECOVERY}/metadata/." "${EXP}/metadata/"
  cp -a "${RECOVERY}/logs/." "${EXP}/logs/"
  cp -a "${RECOVERY}/metrics/." "${EXP}/metrics/"
  cp -a "${RECOVERY}/geneval_results/." "${EXP}/geneval_results/" 2>/dev/null || true
  RECOVERED_GENERATION=1
fi
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
nvidia-smi > "${EXP}/logs/nvidia_smi_preflight.txt"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' && {
  echo "GPU already has a compute process; refusing to mix workloads" >&2
  exit 1
} || true

if [[ ! -x "${GEN_VENV}/bin/python" ]]; then
  uv python install 3.10
  uv venv --seed --python 3.10 "${GEN_VENV}"
  "${GEN_VENV}/bin/python" -m pip install --upgrade pip wheel "setuptools==69.5.1"
  "${GEN_VENV}/bin/python" -m pip install --no-build-isolation -r "${ROOT}/Fk-Diffusion-Steering/requirements.txt"
fi

"${GEN_VENV}/bin/python" - <<'PY'
import importlib.util, pathlib, urllib.request
spec = importlib.util.find_spec("hpsv2")
if spec is None or spec.origin is None:
    raise SystemExit("hpsv2 missing")
target = pathlib.Path(spec.origin).resolve().parent / "src/open_clip/bpe_simple_vocab_16e6.txt.gz"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz", target)
print(target)
PY

"${GEN_VENV}/bin/python" "${EXP}/budget_check.py"
"${GEN_VENV}/bin/python" "${EXP}/select_prompts.py" --verify-existing

export PATH="${GEN_VENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GEN_VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

PSP_COMMIT="590f59f58384169c719e431dc01d14006fb0fc0c"
if git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_STATUS="$(git -C "${ROOT}" status --short | tr '\n' ';' | sed 's/;/; /g')"
  GIT_DIFF_STAT="$(git -C "${ROOT}" diff --stat | tr '\n' ';' | sed 's/;/; /g')"
else
  GIT_STATUS="ORX source snapshot (no .git directory)"
  GIT_DIFF_STAT="See recorded source commit and experiment patch"
fi
if [[ "${RECOVERED_GENERATION}" -eq 0 ]]; then
cat > "${EXP}/metrics/run_info.json" <<JSON
{
  "psp_commit": "${PSP_COMMIT}",
  "git_status": "${GIT_STATUS}",
  "git_diff_stat": "${GIT_DIFF_STAT}",
  "host_started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
fi

if [[ "${RECOVERED_GENERATION}" -eq 0 ]]; then
  echo "[trace] one prompt on GPU0"
  CUDA_VISIBLE_DEVICES=0 "${GEN_VENV}/bin/python" "${EXP}/run_worker.py" \
    --worker-index 0 --num-workers 1 --gpu-id 0 --resume --limit-subset-positions 1 \
    | tee "${EXP}/logs/one_prompt_trace.log"

  echo "[smoke] eight prompts on two GPUs"
  "${EXP}/run_dual_gpu.sh" 8
fi
"${GEN_VENV}/bin/python" "${EXP}/validate_smoke.py" | tee "${EXP}/logs/smoke_validation.log"

echo "[geneval preflight] exact environment and eight generated images"
"${GEN_VENV}/bin/python" "${EXP}/export_geneval.py"
"${EXP}/setup_geneval_env.sh" | tee "${EXP}/logs/geneval_setup.log"
if [[ ! -s "${EXP}/geneval_results/smoke_psp.jsonl" ]] || \
   [[ "$(wc -l < "${EXP}/geneval_results/smoke_psp.jsonl")" -ne 8 ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${GENEVAL_MMDET}:${PYTHONPATH:-}" \
    "${GENEVAL_VENV}/bin/python" "${ROOT}/geneval/evaluation/evaluate_images.py" \
    "${EXP}/geneval_inputs/psp" \
    --outfile "${EXP}/geneval_results/smoke_psp.jsonl" \
    --model-config "${GENEVAL_MMDET}/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
    --model-path "${GENEVAL_WEIGHTS}" \
    2> "${EXP}/logs/geneval_smoke.log"
fi
[[ "$(wc -l < "${EXP}/geneval_results/smoke_psp.jsonl")" -eq 8 ]]

echo "[full] 300 prompts on two GPUs"
if [[ "${RECOVERED_GENERATION}" -eq 0 ]]; then
generation_started="$(date +%s)"
"${EXP}/run_dual_gpu.sh" 0
generation_ended="$(date +%s)"
"${GEN_VENV}/bin/python" - "${EXP}/metrics/run_info.json" "${generation_started}" "${generation_ended}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
row = json.loads(path.read_text())
row["generation_started_unix"] = int(sys.argv[2])
row["generation_ended_unix"] = int(sys.argv[3])
row["generation_wall_s"] = int(sys.argv[3]) - int(sys.argv[2])
path.write_text(json.dumps(row, indent=2) + "\n")
PY
else
  echo "[full] recovered complete generation; diffusion will not be rerun"
fi

echo "[hps] final winners only"
CUDA_VISIBLE_DEVICES=0 "${GEN_VENV}/bin/python" "${EXP}/evaluate_hps.py" | tee "${EXP}/logs/hps.log"

echo "[geneval] official evaluator"
"${GEN_VENV}/bin/python" "${EXP}/export_geneval.py"
for method in psp early10to2; do
  if [[ -s "${EXP}/geneval_results/${method}.jsonl" ]] && \
     [[ "$(wc -l < "${EXP}/geneval_results/${method}.jsonl")" -eq 300 ]]; then
    echo "[geneval] reusing complete ${method} result"
    continue
  fi
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${GENEVAL_MMDET}:${PYTHONPATH:-}" \
    "${GENEVAL_VENV}/bin/python" "${ROOT}/geneval/evaluation/evaluate_images.py" \
    "${EXP}/geneval_inputs/${method}" \
    --outfile "${EXP}/geneval_results/${method}.jsonl" \
    --model-config "${GENEVAL_MMDET}/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
    --model-path "${GENEVAL_WEIGHTS}" \
    2> "${EXP}/logs/geneval_${method}.log"
done

echo "[report] aggregation and paired bootstrap"
"${GEN_VENV}/bin/python" "${EXP}/aggregate.py" | tee "${EXP}/logs/aggregate.log"
tar -C "${EXP}" -czf "${EXP}/results_bundle.tar.gz" \
  README.md PROTOCOL.md REPORT.md prompt_ids_300.json prompts_geneval_balanced_300.jsonl \
  metrics metadata logs geneval_results outputs
date -u +%Y-%m-%dT%H:%M:%SZ > "${EXP}/COMPLETE"
echo "COMPLETE: ${EXP}/REPORT.md"
