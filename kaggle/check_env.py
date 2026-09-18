#!/usr/bin/env python3
"""Compatibility gate for the TQK reproduction environment.

This runs inside the KJO Job Spec (schema v2, uv environment) and proves that the
uv-managed interpreter matches the published reproduction instead of merely
importing something that happens to exist.

Checks:
  1. Python is 3.10.x.
  2. Every pin in Fk-Diffusion-Steering/requirements.txt is installed at exactly
     that version (flat pip freeze -> uv lock).
  3. The two editable git requirements resolve to the pinned upstream commits via
     PEP 610 direct_url.json, not to a PyPI release.
  4. torch sees the expected CUDA devices.
  5. The project modules that the workers import are importable.
  6. The SD1.5 weights are reachable from the model id the workers hardcode.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "Fk-Diffusion-Steering" / "requirements.txt"
GIT_RE = re.compile(r"^-e\s+git\+(?P<url>[^@#\s]+?)(?:@(?P<rev>[0-9a-fA-F]{7,40}))?#egg=(?P<egg>[A-Za-z0-9_.-]+)$")

PUBLISHED_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def parse_pins() -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    pins: dict[str, str] = {}
    git_pins: list[tuple[str, str, str]] = []
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = GIT_RE.match(line)
        if match:
            git_pins.append((match.group("egg"), match.group("url"), match.group("rev")))
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins, git_pins


def dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit_of(name: str) -> str | None:
    try:
        raw = metadata.distribution(name).read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return ((json.loads(raw).get("vcs_info") or {}).get("commit_id")) or None
    except Exception:
        return None


def check_pins() -> tuple[list[str], list[dict]]:
    pins, git_pins = parse_pins()
    problems: list[str] = []
    rows: list[dict] = []
    for name, expected in sorted(pins.items()):
        actual = dist_version(name)
        ok = actual == expected
        rows.append({"name": name, "expected": expected, "actual": actual, "ok": ok})
        if not ok:
            problems.append(f"pin mismatch: {name} expected {expected} got {actual}")
    for egg, url, rev in git_pins:
        commit = git_commit_of(egg) or git_commit_of(egg.replace("_", "-"))
        ok = bool(commit) and commit.lower().startswith(rev.lower())
        rows.append({"name": egg, "expected": f"git:{rev}", "actual": commit, "ok": ok})
        if not ok:
            problems.append(f"git pin mismatch: {egg} expected {rev} got {commit}")
    return problems, rows


def check_python() -> list[str]:
    if sys.version_info[:2] != (3, 10):
        return [f"python {sys.version.split()[0]} is not 3.10.x"]
    return []


def check_cuda(expect_gpus: int, require_cuda: bool) -> tuple[list[str], dict]:
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    problems: list[str] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        info["devices"].append({
            "index": index,
            "name": props.name,
            "vram_gib": round(props.total_memory / 2**30, 2),
            "capability": f"sm{props.major}{props.minor}",
        })
    if require_cuda:
        if not info["cuda_available"]:
            problems.append("CUDA is not available")
        elif expect_gpus and info["device_count"] != expect_gpus:
            problems.append(f"expected {expect_gpus} CUDA devices, found {info['device_count']}")
    return problems, info


def check_imports() -> tuple[list[str], dict]:
    sys.path.insert(0, str(ROOT / "Fk-Diffusion-Steering" / "text_to_image"))
    sys.path.insert(0, str(ROOT / "exps" / "single_stage_calibration"))
    targets = [
        ("fkd_diffusers.fkd_pipeline_sd", "FKDStableDiffusion"),
        ("fkd_diffusers.rewards", "do_image_reward"),
        ("schedules", "CHECKPOINTS"),
    ]
    problems: list[str] = []
    seen: dict[str, str] = {}
    for module_name, attribute in targets:
        try:
            module = __import__(module_name, fromlist=[attribute])
            getattr(module, attribute)
            seen[module_name] = "ok"
        except Exception as exc:  # noqa: BLE001 - report the exact failure
            seen[module_name] = f"{type(exc).__name__}: {exc}"
            problems.append(f"import failed: {module_name}.{attribute} -> {seen[module_name]}")
    return problems, seen


def check_model() -> tuple[list[str], dict]:
    import urllib.error
    import urllib.request

    url = f"https://huggingface.co/{PUBLISHED_MODEL_ID}/resolve/main/model_index.json"
    request = urllib.request.Request(url, method="HEAD")
    token = __import__("os").environ.get("HF_TOKEN") or __import__("os").environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return [], {"model_id": PUBLISHED_MODEL_ID, "http_status": response.status, "reachable": True}
    except urllib.error.HTTPError as exc:
        return [f"{PUBLISHED_MODEL_ID} unreachable: HTTP {exc.code}"], {"model_id": PUBLISHED_MODEL_ID, "http_status": exc.code, "reachable": False}
    except Exception as exc:  # noqa: BLE001
        return [f"{PUBLISHED_MODEL_ID} unreachable: {type(exc).__name__}: {exc}"], {"model_id": PUBLISHED_MODEL_ID, "reachable": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="env_check.json")
    parser.add_argument("--expect-gpus", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()

    problems: list[str] = []
    report: dict = {"python": sys.version.split()[0]}

    problems += check_python()
    pin_problems, pin_rows = check_pins()
    problems += pin_problems
    report["pins"] = {"checked": len(pin_rows), "failed": sum(1 for row in pin_rows if not row["ok"]), "rows": pin_rows}

    cuda_problems, cuda_info = check_cuda(args.expect_gpus, args.require_cuda)
    problems += cuda_problems
    report["cuda"] = cuda_info

    import_problems, imports = check_imports()
    problems += import_problems
    report["imports"] = imports

    if not args.skip_model_check:
        model_problems, model_info = check_model()
        problems += model_problems
        report["model"] = model_info

    report["ok"] = not problems
    report["problems"] = problems
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("KJO_ENV_CHECK " + json.dumps({key: report[key] for key in ("ok", "python", "problems")}))
    if not report["ok"]:
        print(json.dumps(report, indent=2)[:4000])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
