#!/usr/bin/env bash
# Run the official GenEval evaluator over the images this session generated, then delete the
# /tmp environment so nothing bulky survives into the run output.
#
# Three dependencies are deliberate:
#   * geneval/evaluation/evaluate_images.py asserts CUDA is available, so this only runs in a
#     GPU session (the install itself is happy on CPU);
#   * exps/single_stage_validation_553/export_geneval.py links samples/00000.png to
#     outputs/<method>/<id>.png, so the evaluation must run in the session that owns them;
#   * the environment came from kaggle/install_geneval_env.sh and lives under /tmp.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${GENEAL_EXP_DIR:-$ROOT/exps/single_stage_validation_553}"
ENV_DIR="${GENEAL_ENV_DIR:-/tmp/geneval_env}"
WEIGHTS_DIR="${GENEAL_WEIGHTS_DIR:-/tmp/geneval_weights}"
MANIFEST="/tmp/geneval_env_manifest.json"
PY="$ENV_DIR/bin/python"

[ -x "$PY" ] || { echo "[geneval] no environment at $ENV_DIR; run kaggle/install_geneval_env.sh first"; exit 1; }
"$PY" -c "import torch; assert torch.cuda.is_available(), 'GenEval evaluator requires CUDA'"
mkdir -p "$EXP/geneval_results" "$EXP/metrics"
CONFIG=$("$PY" -c "import json;print(json.load(open('$MANIFEST'))['mmdet_config'])")
echo "[geneval] model config $CONFIG | weights $WEIGHTS_DIR"

for method in psp ours; do
  echo "[geneval] evaluating $method"
  "$PY" "$ROOT/geneval/evaluation/evaluate_images.py" "$EXP/geneval_inputs/$method" \
    --outfile "$EXP/geneval_results/$method.jsonl" \
    --model-config "$CONFIG" --model-path "$WEIGHTS_DIR" 2>&1 | tail -20
  "$PY" "$ROOT/geneval/evaluation/summary_scores.py" "$EXP/geneval_results/$method.jsonl" \
    | tee "$EXP/metrics/geneval_$method"'_summary.txt'
done

rm -rf "$ENV_DIR" "$WEIGHTS_DIR" "$MANIFEST"
echo "[geneval] evaluated both methods and removed the /tmp environment"
df -h /tmp /kaggle/working || true
