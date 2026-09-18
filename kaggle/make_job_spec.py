#!/usr/bin/env python3
"""Generate a KJO Job Spec (schema v2) for one TQK shard.

The Job Spec is generator input for KJO, not a file Kaggle must attach: render-job-notebook
and stage-job-package embed the normalized spec in the notebook. Keep it non-secret.

Examples:
  kaggle/make_job_spec.py --phase bank --worker-indices 0,1 --out /tmp/bank_01.json
  kaggle/make_job_spec.py --phase bank --num-workers 8 --worker-indices 4,5 --limit-prompts 2 \
      --out /tmp/bank_45.json
  kaggle/make_job_spec.py --phase smoke-cpu --out /tmp/smoke.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def current_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def build(args: argparse.Namespace) -> dict:
    commit = args.commit or current_commit()
    if len(commit) not in (40, 64):
        raise SystemExit(f"--commit must be a full 40- or 64-hex commit, got {commit!r}")

    environment = {
        "manager": "uv",
        "mode": "locked",
        "uv_version": args.uv_version,
        "python": args.python,
        "project_dir": "{project_root}",
        "lockfile": "uv.lock",
        "extras": [],
        "groups": [],
    }
    code_source = {"mode": "git", "url": args.repo_url, "commit": commit}

    if args.phase == "smoke-cpu":
        runtime = {"accelerator": "cpu"}
        steps = [
            {
                "id": "env-check",
                "kind": "python-script",
                "path": "kaggle/check_env.py",
                "args": ["--out", "{working_root}/env_check.json"],
            },
            {
                "id": "shard-dry-run",
                "kind": "shell-script",
                "path": "kaggle/run_shard.sh",
                "args": ["--phase", "bank", "--num-workers", "4", "--worker-indices", "0,1", "--dry-run"],
            },
        ]
        outputs = [
            {"id": "env-check", "kind": "json", "path": "{working_root}/env_check.json", "required": True, "min_bytes": 2}
        ]
    elif args.phase == "bench-decode":
        runtime = {"accelerator": "gpu", "submit_accelerator": args.submit_accelerator}
        steps = [
            {
                "id": "env-check",
                "kind": "python-script",
                "path": "kaggle/check_env.py",
                "args": ["--out", "{working_root}/env_check.json", "--expect-gpus", str(args.expect_gpus), "--require-cuda"],
            },
            {
                "id": "bench-decode",
                "kind": "python-script",
                "path": "kaggle/bench_decode.py",
                "args": ["--out", "{working_root}/decode_bench.json", "--with-reward"],
            },
        ]
        outputs = [
            {"id": "env-check", "kind": "json", "path": "{working_root}/env_check.json", "required": True, "min_bytes": 2},
            {"id": "bench-decode", "kind": "json", "path": "{working_root}/decode_bench.json", "required": True, "min_bytes": 2},
        ]
    elif args.phase == "smoke-gpu":
        runtime = {"accelerator": "gpu", "submit_accelerator": args.submit_accelerator}
        steps = [
            {
                "id": "env-check",
                "kind": "python-script",
                "path": "kaggle/check_env.py",
                "args": ["--out", "{working_root}/env_check.json", "--expect-gpus", str(args.expect_gpus), "--require-cuda"],
            },
            {
                "id": "shard",
                "kind": "shell-script",
                "path": "kaggle/run_shard.sh",
                "args": ["--phase", "bank", "--num-workers", "2", "--worker-indices", "0,1", "--limit-prompts", "2"],
            },
            {
                "id": "validate",
                "kind": "python-script",
                "path": "exps/single_stage_calibration/validate_bank.py",
                "args": ["--expected-prompts", "2"],
            },
        ]
        outputs = [
            {"id": "env-check", "kind": "json", "path": "{working_root}/env_check.json", "required": True, "min_bytes": 2},
            {
                "id": "shard-artifacts",
                "kind": "directory",
                "path": "{project_root}/exps/single_stage_calibration/bank_raw",
                "required": True,
            },
        ]
    else:
        runtime = {"accelerator": "gpu", "submit_accelerator": args.submit_accelerator}
        phase_args = [
            "--phase", args.phase,
            "--num-workers", str(args.num_workers),
            "--worker-indices", args.worker_indices,
        ]
        if args.limit_prompts:
            phase_args += ["--limit-prompts", str(args.limit_prompts)]
        exp_dir = "single_stage_calibration" if args.phase == "bank" else "single_stage_validation_553"
        artifact_dir = "bank_raw" if args.phase == "bank" else "outputs"
        steps = [
            {
                "id": "env-check",
                "kind": "python-script",
                "path": "kaggle/check_env.py",
                "args": ["--out", "{working_root}/env_check.json", "--expect-gpus", str(args.expect_gpus), "--require-cuda"],
            },
            {"id": "shard", "kind": "shell-script", "path": "kaggle/run_shard.sh", "args": phase_args},
        ]
        outputs = [
            {"id": "env-check", "kind": "json", "path": "{working_root}/env_check.json", "required": True, "min_bytes": 2},
            {
                "id": "shard-artifacts",
                "kind": "directory",
                "path": "{project_root}/exps/" + exp_dir + "/" + artifact_dir,
                "required": False,
            },
        ]

    return {
        "schema_version": 2,
        "run_id": args.run_id,
        "title": args.title,
        "runtime": runtime,
        "code_source": code_source,
        "environment": environment,
        "steps": steps,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["bank", "eval", "smoke-cpu", "smoke-gpu", "bench-decode"], required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--worker-indices", default="0,1")
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--expect-gpus", type=int, default=2)
    parser.add_argument("--commit", default="")
    parser.add_argument("--repo-url", default="https://github.com/sontungkieu/TQK")
    parser.add_argument("--python", default="3.10.17")
    parser.add_argument("--uv-version", default="0.10.2")
    parser.add_argument("--submit-accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not args.run_id:
        suffix = args.worker_indices.replace(",", "")
        if args.phase.startswith("smoke"):
            args.run_id = "tqk_" + args.phase.replace("-", "_")
        else:
            args.run_id = f"tqk_{args.phase}_w{args.num_workers}_s{suffix}"
    if not args.title:
        args.title = f"TQK {args.phase} shard {args.worker_indices}"

    spec = build(args)
    Path(args.out).write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"out": args.out, "run_id": spec["run_id"], "steps": [step["id"] for step in spec["steps"]]}))


if __name__ == "__main__":
    main()
