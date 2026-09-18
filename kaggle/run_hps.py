#!/usr/bin/env python3
"""Run the published HPS evaluator after mirroring setup/setup.sh in the uv environment.

setup/setup.sh downloads the CLIP BPE asset that the hpsv2 wheel does not ship:

    https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz

The uv environment never runs that script, so the asset is missing and hpsv2 fails while
building its tokenizer (a few seconds in, with no captured stderr under KJO). This wrapper
restores the asset first, then executes exps/single_stage_validation_553/evaluate_hps.py
with runpy and prints any traceback on stdout, because KJO echoes step stdout but not step
stderr.
"""
from __future__ import annotations

import importlib.util
import json
import runpy
import sys
import traceback
import urllib.request
from pathlib import Path

ASSET_URL = "https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz"
EVALUATOR = Path(__file__).resolve().parents[1] / "exps" / "single_stage_validation_553" / "evaluate_hps.py"


def ensure_bpe_asset() -> dict:
    spec = importlib.util.find_spec("hpsv2")
    if spec is None or spec.origin is None:
        raise RuntimeError("hpsv2 is not importable in this environment")
    target = Path(spec.origin).resolve().parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    state = {"target": str(target), "existed": target.exists()}
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ASSET_URL, str(target))
        state["downloaded"] = True
        state["bytes"] = target.stat().st_size
    return state


def main() -> int:
    report: dict = {"evaluator": str(EVALUATOR)}
    try:
        report["bpe_asset"] = ensure_bpe_asset()
        import hpsv2

        report["hpsv2"] = getattr(hpsv2, "__version__", "unknown")
        print("KJO_HPS_SETUP " + json.dumps(report), flush=True)
    except Exception:
        report["setup_error"] = traceback.format_exc()
        print("KJO_HPS_SETUP " + json.dumps(report), flush=True)
        return 1

    sys.argv = [str(EVALUATOR)]
    try:
        runpy.run_path(str(EVALUATOR), run_name="__main__")
    except SystemExit as exc:
        code = int(exc.code or 0)
        if code:
            print("KJO_HPS_ERROR " + json.dumps({"exit": code, "traceback": traceback.format_exc()}), flush=True)
        return code
    except Exception:
        print("KJO_HPS_ERROR " + json.dumps({"traceback": traceback.format_exc()}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
