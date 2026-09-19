#!/usr/bin/env bash
# Install the official GenEval evaluator for use INSIDE a session, not as a deliverable.
#
# The evaluator (geneval/evaluation/evaluate_images.py) needs exactly this chain:
#   numpy/pandas/PIL/torch, mmdet.apis (Mask2Former), open_clip, and
#   clip_benchmark.metrics.zeroshot_classification (zero_shot_classifier/run_classification).
#
# Four rules are load-bearing; each one cost a failed build:
# 1. numpy must stay on 1.x. This env installs prebuilt wheels (mmcv-full 1.7.2, torch 2.1.2);
#    an unpinned resolve picks numpy 2.2.6 + opencv-python 5.x, and the mmcv C-extension then
#    dies at import ("numpy.core.multiarray failed to import") after a 5 minute download.
# 2. Every mmdet 2.x runtime requirement must be present (matplotlib, scipy, six, terminaltables,
#    pycocotools, numpy). Installing the source tree through a .pth entry skips mmdet's own
#    install_requires, so they are pinned explicitly here.
# 3. clip-benchmark 1.4.0 (the pin in geneval/environment.yml, a torch 1.12 env) declares
#    torch<2 and cannot be resolved against torch 2.1.2. 1.5.0+ dropped the ceiling and its
#    zeroshot_classification keeps the same five-positional-argument API the evaluator calls.
# 4. Everything bulky lives under /tmp, because /kaggle/working is the kernel output: a
#    multi-GB environment there makes every later listing and fetch slow and eventually
#    HTTP 429. Only a tiny stage log is written back to /kaggle/working, and on failure it
#    carries the tail of the full install log so the cause is readable from a 2 KB fetch.
set -euo pipefail

WORK="${GENEAL_STATE_DIR:-/kaggle/working}"
ENV_DIR="${GENEAL_ENV_DIR:-/tmp/geneval_env}"
WEIGHTS_DIR="${GENEAL_WEIGHTS_DIR:-/tmp/geneval_weights}"
MANIFEST="/tmp/geneval_env_manifest.json"
STAGE_LOG="/tmp/geneval_stages.txt"
STAGE_REPORT="$WORK/geneval_stages.txt"
INSTALL_LOG="/tmp/geneval_install.log"
MMCV_WHEEL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv_full-1.7.2-cp310-cp310-manylinux1_x86_64.whl"
MMDET_TAG="v2.28.2"
MMDET_PYTHON="3.10"

# fd 3 stays on the step's stdout: short progress lines keep the Kaggle log readable,
# while every command's full output goes to INSTALL_LOG for the failure tail.
exec 3>&1
: > "$STAGE_LOG"
: > "$INSTALL_LOG"
exec >>"$INSTALL_LOG" 2>&1

publish_stages() {
  cp -f "$STAGE_LOG" "$STAGE_REPORT" 2>/dev/null || true
}
# Every mark records the free space and the environment size, so the stage report alone shows
# whether a run died from a full disk (Kaggle CPU sessions have a hard disk quota) or from a
# real install error.
mark() {
  printf '%s OK %-46s free_tmp=%s env=%s\n' "$(date -u +%H:%M:%S)" "$1" \
    "$(df -Pk /tmp 2>/dev/null | awk 'NR==2{printf "%.1fG", $4/1048576}')" \
    "$(du -sh "$ENV_DIR" 2>/dev/null | cut -f1)" >> "$STAGE_LOG"
  printf '[geneval] stage ok: %s\n' "$1" >&3
  publish_stages
}
on_error() {
  local code=$?
  {
    printf '%s FAILED rc=%s at line %s: %s\n' "$(date -u +%H:%M:%S)" "$code" "$LINENO" "$BASH_COMMAND"
    echo "--- last 40 lines of $INSTALL_LOG ---"
    tail -n 40 "$INSTALL_LOG" 2>/dev/null || true
    echo "--- disk ---"
    df -h /tmp "$WORK" 2>/dev/null || true
  } >> "$STAGE_LOG"
  printf '[geneval] FAILED rc=%s at line %s: %s\n' "$code" "$LINENO" "$BASH_COMMAND" >&3
  sync || true
  publish_stages
  exit "$code"
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
# --no-cache/--no-cache-dir keep the download cache out of the picture: the cached wheels would
# roughly double the peak disk footprint of an environment that is already several GB.
pipi() {
  if [ -n "$UV" ]; then "$UV" pip install --no-cache --python "$PY" "$@"; else "$ENV_DIR/bin/pip" install --no-cache-dir "$@"; fi
}
mark "venv created ($("$PY" -V 2>&1))"

# Wheels only: a dependency without a wheel fails in seconds instead of compiling for
# ten minutes and dying with an opaque error.
pipi -q --only-binary=:all: --upgrade pip wheel setuptools==69.5.1
mark "base tooling"

# Rule 1: claim numpy before anything else, so no later resolution can drag in numpy 2.x.
pipi -q --only-binary=:all: numpy==1.26.4
mark "numpy 1.x pinned ($("$PY" -c 'import numpy; print(numpy.__version__)'))"

pipi -q --only-binary=:all: torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
mark "torch 2.1.2+cu121 / torchvision 0.16.2"

# Rule 2 and 3 in one resolution: the evaluator imports, mmdet 2.x runtime requirements, and a
# clip-benchmark that accepts torch 2.x.
pipi -q --only-binary=:all: \
  pandas==2.0.3 pillow==10.4.0 tqdm==4.66.5 pyyaml==6.0.2 opencv-python==4.11.0.86 \
  huggingface_hub==0.23.5 open-clip-torch==2.26.1 timm==0.9.2 einops==0.8.0 ftfy==6.2.3 \
  sentencepiece==0.2.0 protobuf==4.25.5 clip-benchmark==1.6.2 \
  scipy==1.10.1 matplotlib==3.7.5 six==1.16.0 terminaltables==3.1.10 pycocotools==2.0.7
mark "evaluator deps (clip-benchmark 1.6.2, mmdet runtime requirements)"

pipi -q --only-binary=:all: mmengine==0.10.4
mark "mmengine 0.10.4"

pipi -q --only-binary=:all: "$MMCV_WHEEL"
mark "mmcv-full 1.7.2 (prebuilt)"

if [ ! -d "$ENV_DIR/mmdetection/.git" ]; then
  git clone --quiet --depth 1 --branch "$MMDET_TAG" https://github.com/open-mmlab/mmdetection.git "$ENV_DIR/mmdetection"
fi
# mmdet 2.x is pure Python (the CUDA ops live in mmcv), so a .pth entry is enough and
# avoids an editable install whose setup.py imports torch at build time.
echo "$ENV_DIR/mmdetection" > "$(dirname "$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')")/mmdet_source.pth"
mark "mmdetection $MMDET_TAG source on PYTHONPATH"

# Smoke test the whole evaluator import chain, so a broken environment fails here (CPU, cheap)
# instead of after burning a GPU session. evaluate_images.py derives the Mask2Former config
# path from mmdet.__file__, so that path is checked too.
"$PY" - <<'PYCHK'
import inspect
from pathlib import Path

import clip_benchmark.metrics.zeroshot_classification as zsc
import mmcv
import mmdet
import mmengine
import numpy as np
import open_clip
import pandas as pd
import torch
import torchvision

assert np.__version__.startswith("1."), f"numpy {np.__version__} would break the mmcv C extension"
assert callable(zsc.zero_shot_classifier) and callable(zsc.run_classification)
assert zsc.tqdm is not None, "evaluate_images.py rebinds clip_benchmark's tqdm"
sig = inspect.signature(zsc.zero_shot_classifier)
print("[geneval] zero_shot_classifier", sig)
assert len(sig.parameters) >= 5, "evaluator calls it with five positional arguments"

config = Path(mmdet.__file__).resolve().parent.parent / "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
assert config.exists(), f"Mask2Former config missing at {config}"

open_clip.get_tokenizer("ViT-L-14")
print(
    "[geneval] numpy", np.__version__,
    "| torch", torch.__version__,
    "| torchvision", torchvision.__version__,
    "| pandas", pd.__version__,
    "| mmcv", mmcv.__version__,
    "| mmdet", mmdet.__version__,
    "| mmengine", mmengine.__version__,
    "| open_clip", getattr(open_clip, "__version__", "?"),
)
print("[geneval] mmdet config:", config)
PYCHK
mark "import check (evaluator chain)"

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
