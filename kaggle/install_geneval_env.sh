#!/usr/bin/env bash
# Install the official GenEval evaluator for use INSIDE a session, not as a deliverable.
#
# Mirrors setup/setup_geneval.sh (CUDA 12.1 path): torch 2.1.2 with cu121 has a prebuilt
# mmcv-full 1.7.2 wheel, so nothing is compiled. Two design rules matter:
#
# 1. Everything bulky lives under /tmp. /kaggle/working is the kernel output, and a
#    multi-GB environment or a thousand files there makes every later listing and fetch
#    slow and eventually HTTP 429 (see SKILL.md, Notebook And Dependency Contract).
# 2. The only thing written back to /kaggle/working is a tiny stage log, so a failed run
#    still tells us where it died without pulling a large output tree.
set -euo pipefail

WORK="${GENEAL_STATE_DIR:-/kaggle/working}"
ENV_DIR="${GENEAL_ENV_DIR:-/tmp/geneval_env}"
WEIGHTS_DIR="${GENEAL_WEIGHTS_DIR:-/tmp/geneval_weights}"
MANIFEST="/tmp/geneval_env_manifest.json"
STAGE_LOG="/tmp/geneval_stages.txt"
STAGE_REPORT="$WORK/geneval_stages.txt"
MMCV_WHEEL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv_full-1.7.2-cp310-cp310-manylinux1_x86_64.whl"
MMDET_PYTHON="3.10"

: > "$STAGE_LOG"

publish_stages() {
  cp -f "$STAGE_LOG" "$STAGE_REPORT" 2>/dev/null || true
}
mark() {
  echo "$(date -u +%H:%M:%S) OK $1" >> "$STAGE_LOG"
  echo "[geneval] stage ok: $1"
  publish_stages
}
on_error() {
  local code=$?
  echo "$(date -u +%H:%M:%S) FAILED rc=$code at line $LINENO: $BASH_COMMAND" >> "$STAGE_LOG"
  echo "[geneval] FAILED rc=$code at line $LINENO: $BASH_COMMAND"
  df -h /tmp "$WORK" || true
  publish_stages
  exit $code
}
trap on_error ERR

find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
  find "/kaggle/working/.kjo" -maxdepth 6 -type f -name uv -perm -u+x 2>/dev/null | head -1
}

UV="$(find_uv || true)"
if [ -n "$UV" ]; then echo "[geneval] uv: $("$UV" --version 2>&1)"; else echo "[geneval] uv not found; using venv+pip"; fi
echo "[geneval] state dir $WORK | env dir $ENV_DIR | weights $WEIGHTS_DIR"

rm -rf "$ENV_DIR" "$WEIGHTS_DIR"
mkdir -p "$ENV_DIR" "$WEIGHTS_DIR"

if [ -n "$UV" ]; then
  "$UV" venv --python "$MMDET_PYTHON" "$ENV_DIR"
else
  python -m venv "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
pipi() {
  if [ -n "$UV" ]; then "$UV" pip install --python "$PY" "$@"; else "$ENV_DIR/bin/pip" install "$@"; fi
}
mark "venv created ($("$PY" -V 2>&1))"

# Wheels only: a dependency without a wheel fails in seconds instead of compiling for
# ten minutes and dying with an opaque error.
pipi -q --only-binary=:all: --upgrade pip wheel setuptools==69.5.1
mark "base tooling"

pipi -q --only-binary=:all: torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
mark "torch 2.1.2+cu121"

pipi -q --only-binary=:all: networkx==2.8.8 open-clip-torch==2.26.1 einops lightning \
  diffusers transformers tomli platformdirs timm pycocotools terminaltables
mark "evaluator deps"
pipi -q --only-binary=:all: mmengine==0.10.4
mark "mmengine"

pipi -q --only-binary=:all: "$MMCV_WHEEL"
mark "mmcv-full 1.7.2 (prebuilt)"

if [ ! -d "$ENV_DIR/mmdetection/.git" ]; then
  git clone --quiet --depth 1 --branch 2.x https://github.com/open-mmlab/mmdetection.git "$ENV_DIR/mmdetection"
fi
# mmdet 2.x is pure Python (the CUDA ops live in mmcv), so a .pth entry is enough and
# avoids an editable install whose setup.py imports torch at build time.
echo "$ENV_DIR/mmdetection" > "$(dirname "$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')")/mmdet_source.pth"
mark "mmdetection source on PYTHONPATH"

"$PY" - <<'PYCHK'
import pathlib
import mmcv, mmdet, mmengine, torch
print("[geneval] torch", torch.__version__, "| mmcv", mmcv.__version__, "| mmdet", mmdet.__version__, "| mmengine", mmengine.__version__)
print("[geneval] mmdet config dir:", pathlib.Path(mmdet.__file__).resolve().parent.parent / "configs")
PYCHK
mark "import check"

WEIGHT_NAME=mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
wget -q -O "$WEIGHTS_DIR/$WEIGHT_NAME" \
  "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
mark "Mask2Former weights"

CONFIG=$(find "$ENV_DIR/mmdetection/configs/mask2former" -name "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" | head -1)
[ -n "$CONFIG" ] || { echo "[geneval] FATAL: mask2former config not found"; false; }
"$PY" - "$ENV_DIR" "$WEIGHTS_DIR" "$CONFIG" "$MANIFEST" "$WEIGHT_NAME" <<'PYMAN'
import hashlib, json, sys
from pathlib import Path

env_dir, weights_dir, config, out, weight_name = sys.argv[1:6]
weight = Path(weights_dir) / weight_name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


Path(out).write_text(json.dumps({
    "venv": str(env_dir),
    "weights": str(weight),
    "weights_sha256": sha256(weight) if weight.exists() else "",
    "mmdet_config": str(config),
}, indent=2) + chr(10))
print("[geneval] manifest:", out)
PYMAN
mark "manifest"

if [ "${PACK_ARTIFACT:-0}" = "1" ]; then
  # Deliberate export: pack, then remove the bulk so the run output stays small.
  tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.git' -czf "$WORK/geneval_env_artifact.tar.gz" \
    -C /tmp "$(basename "$ENV_DIR")" "$(basename "$WEIGHTS_DIR")" "$(basename "$MANIFEST")"
  rm -rf "$ENV_DIR" "$WEIGHTS_DIR"
  mark "packed artifact and removed the environment from /tmp"
else
  mark "done (environment kept in /tmp for this session only)"
fi
publish_stages
df -h /tmp "$WORK" || true
echo "[geneval] install complete"
