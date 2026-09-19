#!/usr/bin/env bash
# Build the official GenEval evaluator environment once, inside a CPU Kaggle session,
# and tar it for publication as a private dataset artifact.
#
# This mirrors setup/setup_geneval.sh (the CUDA 12.1 path, not the Hopper one):
# torch 2.1.2 + cu121 has a PREBUILT mmcv-full 1.7.2 wheel, so nothing is compiled.
# GenEval needs its own venv because the project environment pins torch 2.4.0.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${GENEAL_WORK_ROOT:-/kaggle/working}"
ENV_DIR="$WORK/geneval_env"
WEIGHTS_DIR="$WORK/geneval_weights"
ARTIFACT="$WORK/geneval_env_artifact.tar.gz"
MMCV_WHEEL_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"

echo "[geneval] python: $(python -V 2>&1)"
rm -rf "$ENV_DIR" "$WEIGHTS_DIR"
mkdir -p "$ENV_DIR" "$WEIGHTS_DIR"

echo "[geneval] creating venv at $ENV_DIR"
python -m venv "$ENV_DIR"
PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"
"$PIP" install -q --upgrade pip wheel
"$PIP" install -q "setuptools==69.5.1"

echo "[geneval] torch 2.1.2 + cu121 (matches the prebuilt mmcv wheel)"
"$PIP" install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[geneval] evaluator dependencies"
"$PIP" install -q networkx==2.8.8 open-clip-torch==2.26.1 clip-benchmark einops lightning \
  diffusers transformers tomli platformdirs timm openmim pycocotools terminaltables

echo "[geneval] mmengine + mmcv-full 1.7.2 (prebuilt wheel)"
"$PIP" install -q "mmengine==0.10.4"
if ! "$PIP" install -q "mmcv-full==1.7.2" -f "$MMCV_WHEEL_INDEX"; then
  echo "[geneval] wheel index path failed, falling back to mim"
  "$ENV_DIR/bin/mim" install "mmcv-full==1.7.2" -y
fi

echo "[geneval] mmdetection 2.x from source (configs ship with the clone)"
if [ ! -d "$ENV_DIR/mmdetection/.git" ]; then
  git clone --quiet --depth 1 --branch 2.x https://github.com/open-mmlab/mmdetection.git "$ENV_DIR/mmdetection"
fi
"$PIP" install -q -e "$ENV_DIR/mmdetection"

echo "[geneval] Mask2Former detector weights"
if [ ! -f "$WEIGHTS_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
  wget -q -O "$WEIGHTS_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" \
    https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
fi

echo "[geneval] import check"
"$PY" - <<'PY'
import pathlib
import mmcv, mmdet, mmengine, torch
print("torch", torch.__version__, "| mmcv", mmcv.__version__, "| mmdet", mmdet.__version__, "| mmengine", mmengine.__version__)
print("mmdet config dir:", pathlib.Path(mmdet.__file__).resolve().parent.parent / "configs")
PY

CONFIG=$(find "$ENV_DIR/mmdetection/configs/mask2former" -name "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" | head -1)
echo "[geneval] model config: $CONFIG"
[ -n "$CONFIG" ] || { echo "[geneval] FATAL: expected mask2former config not found"; exit 1; }

echo "[geneval] writing manifest"
"$PY" - "$ENV_DIR" "$WEIGHTS_DIR" "$CONFIG" "$WORK/geneval_env_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

env_dir, weights_dir, config, out = (Path(p) for p in sys.argv[1:5])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


weights = sorted(weights_dir.glob("*.pth"))
manifest = {
    "python": "3.10",
    "venv_relative": "geneval_env",
    "mmdet_config_relative": str(config.relative_to(env_dir.parent)),
    "weights_relative": "geneval_weights/" + weights[0].name if weights else "",
    "weights_sha256": sha256(weights[0]) if weights else "",
    "weights_bytes": weights[0].stat().st_size if weights else 0,
}
Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

echo "[geneval] tarring $ARTIFACT"
du -sh "$ENV_DIR" "$WEIGHTS_DIR" || true
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.a' -czf "$ARTIFACT" \
  -C "$WORK" geneval_env geneval_weights geneval_env_manifest.json
ls -lh "$ARTIFACT"
echo "[geneval] build complete"
