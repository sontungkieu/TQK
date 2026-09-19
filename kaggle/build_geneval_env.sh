#!/usr/bin/env bash
# Build the official GenEval evaluator environment once, inside a CPU Kaggle session,
# and tar it for publication as a private dataset artifact.
#
# Mirrors setup/setup_geneval.sh (the CUDA 12.1 path, not the Hopper one): torch 2.1.2 with
# cu121 has a PREBUILT mmcv-full 1.7.2 wheel, so nothing is compiled. GenEval needs its own
# venv because the project environment pins torch 2.4.0 and KJO manages exactly one uv
# environment per project_dir.
#
# KJO installs uv into the session and prepends its directory to PATH for every step
# (kjo_runtime.prepare_job_environment -> uv_executable, then the step PATH prefix), so the
# build uses uv when it is there and falls back to venv+pip otherwise. Either way the
# installed wheels are identical; uv is simply faster and lets us record a frozen lock.
set -euo pipefail
trap 'echo "[geneval] FAILED at line $LINENO: $BASH_COMMAND"; exit 1' ERR

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${GENEAL_WORK_ROOT:-/kaggle/working}"
ENV_DIR="$WORK/geneval_env"
WEIGHTS_DIR="$WORK/geneval_weights"
LOCK_FILE="$WORK/geneval_env_requirements.lock.txt"
ARTIFACT="$WORK/geneval_env_artifact.tar.gz"
MMCV_WHEEL_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"
MMCV_WHEEL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv_full-1.7.2-cp310-cp310-manylinux1_x86_64.whl"

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi
  find /kaggle/working/.kjo -maxdepth 6 -type f -name uv -perm -u+x 2>/dev/null | head -1
}

UV="$(find_uv || true)"
if [ -n "$UV" ]; then
  echo "[geneval] using uv at $UV ($("$UV" --version 2>&1))"
else
  echo "[geneval] uv not found; falling back to venv + pip"
fi

echo "[geneval] python: $(python -V 2>&1)"
rm -rf "$ENV_DIR" "$WEIGHTS_DIR" "$LOCK_FILE"
mkdir -p "$ENV_DIR" "$WEIGHTS_DIR"

if [ -n "$UV" ]; then
  "$UV" venv --python 3.10 "$ENV_DIR"
else
  python -m venv "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"

pipi() {
  if [ -n "$UV" ]; then
    "$UV" pip install --python "$PY" "$@"
  else
    "$ENV_DIR/bin/pip" install "$@"
  fi
}

echo "[geneval] base tooling"
pipi -q --upgrade pip wheel
pipi -q "setuptools==69.5.1"

echo "[geneval] torch 2.1.2 + cu121 (matches the prebuilt mmcv wheel)"
pipi -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[geneval] evaluator dependencies"
pipi -q networkx==2.8.8 open-clip-torch==2.26.1 clip-benchmark einops lightning \
  diffusers transformers tomli platformdirs timm openmim pycocotools terminaltables

echo "[geneval] mmengine + mmcv-full 1.7.2 (prebuilt wheel, no compilation)"
pipi -q "mmengine==0.10.4"
if ! pipi -q "$MMCV_WHEEL"; then
  echo "[geneval] wheel index path failed, falling back to mim"
  "$ENV_DIR/bin/mim" install "mmcv-full==1.7.2" -y
fi

echo "[geneval] mmdetection 2.x from source (configs ship with the clone)"
if [ ! -d "$ENV_DIR/mmdetection/.git" ]; then
  git clone --quiet --depth 1 --branch 2.x https://github.com/open-mmlab/mmdetection.git "$ENV_DIR/mmdetection"
fi
pipi -q --no-build-isolation -e "$ENV_DIR/mmdetection"

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

if [ -n "$UV" ]; then
  echo "[geneval] freezing the resolved environment"
  "$UV" pip freeze --python "$PY" > "$LOCK_FILE"
  wc -l "$LOCK_FILE"
fi

echo "[geneval] writing manifest"
"$PY" - "$ENV_DIR" "$WEIGHTS_DIR" "$CONFIG" "$WORK/geneval_env_manifest.json" "$LOCK_FILE" "$UV" <<'PY'
import hashlib, json, sys
from pathlib import Path

env_dir, weights_dir, config, out, lock_file, uv = sys.argv[1:7]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


weights = sorted(weights_dir.glob("*.pth"))
lock = Path(lock_file)
manifest = {
    "python": "3.10",
    "venv_relative": "geneval_env",
    "installer": "uv" if uv else "pip",
    "uv_executable": uv,
    "mmdet_config_relative": str(config.relative_to(env_dir.parent)),
    "weights_relative": "geneval_weights/" + weights[0].name if weights else "",
    "weights_sha256": sha256(weights[0]) if weights else "",
    "weights_bytes": weights[0].stat().st_size if weights else 0,
    "lock_relative": lock.name if lock.exists() else "",
}
Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

echo "[geneval] disk before tar:"; df -h "$WORK" | tail -1
echo "[geneval] installed: $(find "$ENV_DIR" -name '*.dist-info' -maxdepth 5 | wc -l) dist-info dirs"
echo "[geneval] tarring $ARTIFACT"
du -sh "$ENV_DIR" "$WEIGHTS_DIR" || true
TAR_EXTRA=""
[ -f "$LOCK_FILE" ] && TAR_EXTRA="geneval_env_requirements.lock.txt"
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.a' -czf "$ARTIFACT" \
  -C "$WORK" geneval_env geneval_weights geneval_env_manifest.json $TAR_EXTRA
ls -lh "$ARTIFACT"
echo "[geneval] build complete"
