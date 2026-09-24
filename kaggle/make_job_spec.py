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
    elif args.phase == "geneval-build":
        # CPU: building the GenEval evaluator environment needs no GPU, and CPU sessions
        # carry no weekly quota in the local policy. The tar is published as a private
        # dataset and attached to the evaluation job later.
        runtime = {"accelerator": "cpu"}
        steps = [
            {
                "id": "env-check",
                "kind": "python-script",
                "path": "kaggle/check_env.py",
                "args": ["--out", "{working_root}/env_check.json"],
            },
            {
                "id": "install-geneval",
                "kind": "shell-script",
                "path": "kaggle/install_geneval_env.sh",
                "timeout_s": 5400,
            },
        ]
        outputs = [
            {"id": "env-check", "kind": "json", "path": "{working_root}/env_check.json", "required": True, "min_bytes": 2},
            # The installer keeps the environment under /tmp and writes back only this
            # tiny stage file, so the kernel output stays cheap to list and fetch.
            {"id": "geneval-stages", "kind": "file", "path": "{working_root}/geneval_stages.txt", "required": False},
        ]
    elif args.phase == "report":
        # Report-only session: the generation and evaluation are already done, so attach the
        # finished kernel as a kernel source (--kernel-source owner/slug) and compute the phase-2
        # report on CPU. No GPU quota, no regeneration, and nothing depends on the Kaggle output
        # endpoint, which is the one that rate-limits.
        validation = "exps/single_stage_validation_553"
        runtime = {"accelerator": "cpu"}
        report_items = (
            "metadata metrics geneval_results protocol_manifest.json "
            "prompts_geneval_all_553.jsonl FROZEN_SCHEDULE.json"
        )
        steps = [
            {
                "id": "stage-run-output",
                "kind": "shell-script",
                "path": "kaggle/stage_geneval_inputs.sh",
                "timeout_s": 1800,
                "env": {"STAGE_RUN_EXPORT": "0", "STAGE_ITEMS": report_items},
            },
            {"id": "aggregate", "kind": "shell-script", "path": "kaggle/aggregate_geneval.sh", "timeout_s": 1800},
            {
                "id": "pack-report",
                "kind": "shell-script",
                "path": "kaggle/pack_artifacts.sh",
                "env": {
                    "PACK_ITEMS": report_items + " FINAL_REPORT.md",
                    "PACK_OUT": "{working_root}/t4_report_553.tar.gz",
                },
            },
        ]
        outputs = [
            {"id": "final-report", "kind": "file", "path": "{project_root}/" + validation + "/FINAL_REPORT.md", "required": False},
            {"id": "metrics", "kind": "directory", "path": "{project_root}/" + validation + "/metrics", "required": True},
            {"id": "report-artifacts", "kind": "file", "path": "{working_root}/t4_report_553.tar.gz", "required": False},
        ]
    elif args.phase == "bench-speedup":
        # Measures UNet-level acceleration options (attention backend, torch.compile, CUDA graphs,
        # timestep-embedding cache skipping) on the real workload shape: speed, VRAM, UNet calls
        # and the latent drift against the untouched baseline.
        runtime = {"accelerator": "gpu", "submit_accelerator": args.submit_accelerator}
        steps = [
            {
                "id": "env-check",
                "kind": "python-script",
                "path": "kaggle/check_env.py",
                "args": ["--out", "{working_root}/env_check.json", "--expect-gpus", str(args.expect_gpus), "--require-cuda"],
            },
            {"id": "bench-speedup", "kind": "python-script", "path": "kaggle/bench_speedup.py", "timeout_s": 5400},
        ]
        outputs = [
            {"id": "env-check", "kind": "json", "path": "{working_root}/env_check.json", "required": True, "min_bytes": 2},
            {"id": "speedup-bench", "kind": "json", "path": "{working_root}/speedup_bench.json", "required": True, "min_bytes": 2},
        ]
    elif args.phase == "bench-batch":
        # Largest VAE-decode and UNet batch that still fits on one T4, with and without
        # torch.compile, so "does compile raise the batch ceiling?" is answered by data.
        runtime = {"accelerator": "gpu", "submit_accelerator": args.submit_accelerator}
        steps = [
            {
                "id": "env-check",
                "kind": "python-script",
                "path": "kaggle/check_env.py",
                "args": ["--out", "{working_root}/env_check.json", "--expect-gpus", str(args.expect_gpus), "--require-cuda"],
            },
            {"id": "bench-batch", "kind": "python-script", "path": "kaggle/bench_batch_memory.py", "timeout_s": 3600},
        ]
        outputs = [
            {"id": "env-check", "kind": "json", "path": "{working_root}/env_check.json", "required": True, "min_bytes": 2},
            {"id": "batch-memory", "kind": "json", "path": "{working_root}/batch_memory.json", "required": True, "min_bytes": 2},
        ]
    elif args.phase == "geneval-eval":
        # Evaluation-only session. export_geneval.py needs the outputs tree, and regenerating
        # 553 prompts x 2 methods for a scoring bug is wasteful, so the finished generation
        # kernel is attached as a kernel source (--kernel-source owner/slug, same owner) and
        # kaggle/stage_geneval_inputs.sh copies the measured tree into this project.
        validation = "exps/single_stage_validation_553"
        runtime = {"accelerator": "gpu", "submit_accelerator": args.submit_accelerator}
        steps = [
            {"id": "stage-geneval-inputs", "kind": "shell-script", "path": "kaggle/stage_geneval_inputs.sh", "timeout_s": 1800},
            {"id": "install-geneval", "kind": "shell-script", "path": "kaggle/install_geneval_env.sh", "timeout_s": 3600},
            {"id": "evaluate-geneval", "kind": "shell-script", "path": "kaggle/evaluate_geneval.sh", "timeout_s": 18000},
            {"id": "aggregate", "kind": "shell-script", "path": "kaggle/aggregate_geneval.sh", "timeout_s": 900},
            {"id": "pack-geneval", "kind": "shell-script", "path": "kaggle/pack_artifacts.sh",
             "env": {"PACK_ITEMS": "geneval_results metrics FINAL_REPORT.md",
                     "PACK_OUT": "{working_root}/geneval_artifacts.tar.gz"}},
        ]
        outputs = [
            {
                "id": "geneval-results",
                "kind": "directory",
                "path": "{project_root}/" + validation + "/geneval_results",
                "required": True,
            },
            {
                "id": "metrics",
                "kind": "directory",
                "path": "{project_root}/" + validation + "/metrics",
                "required": True,
            },
            {"id": "geneval-artifacts", "kind": "file", "path": "{working_root}/geneval_artifacts.tar.gz", "required": False},
            {"id": "final-report", "kind": "file", "path": "{project_root}/" + validation + "/FINAL_REPORT.md", "required": False},
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
    elif args.phase == "eval":
        runtime = {"accelerator": "gpu", "submit_accelerator": args.submit_accelerator}
        phase_args = [
            "--phase", "eval",
            "--num-workers", str(args.num_workers),
            "--worker-indices", args.worker_indices,
        ]
        if args.limit_prompts:
            phase_args += ["--limit-prompts", str(args.limit_prompts)]
        expected = args.limit_prompts or 553
        validation = "exps/single_stage_validation_553"
        steps = [
            {
                "id": "env-check",
                "kind": "python-script",
                "path": "kaggle/check_env.py",
                "args": ["--out", "{working_root}/env_check.json", "--expect-gpus", str(args.expect_gpus), "--require-cuda"],
            },
            # Locks the 553 prompts and the frozen schedule copied from the calibration run;
            # it refuses to overwrite a mismatched FROZEN_SCHEDULE.json, so phase 1 must
            # already be committed into the checkout this Job Spec pins.
            {"id": "prepare-protocol", "kind": "python-script", "path": validation + "/prepare_protocol.py"},
            {"id": "budget-check", "kind": "python-script", "path": validation + "/budget_check.py"},
            {"id": "shard", "kind": "shell-script", "path": "kaggle/run_shard.sh", "args": phase_args},
            {
                "id": "validate",
                "kind": "python-script",
                "path": validation + "/validate_generation.py",
                "args": ["--expected-prompts", str(expected)],
            },
            {"id": "export-geneval", "kind": "python-script", "path": validation + "/export_geneval.py"},
            # Archive the hours of generation before touching GenEval: a scoring step that
            # dies (timeout, OOM, evaluator bug) must not also cost the measurements. One
            # archive keeps later downloads to a single output listing, the Kaggle endpoint
            # that rate-limits (HTTP 429) on repeated large fetches.
            {"id": "pack-artifacts", "kind": "shell-script", "path": "kaggle/pack_artifacts.sh"},
            # GenEval installs into /tmp inside this session (never into /kaggle/working) and
            # scores here, because the evaluator asserts CUDA and geneval_inputs are symlinks
            # into this session's outputs tree. One method per GPU: the measured single-method
            # time on a T4 overran the old 2 h timeout, hence the wider budget.
            {"id": "install-geneval", "kind": "shell-script", "path": "kaggle/install_geneval_env.sh", "timeout_s": 3600},
            {"id": "evaluate-geneval", "kind": "shell-script", "path": "kaggle/evaluate_geneval.sh", "timeout_s": 18000},
            # Reports are computed here, in the session that owns the data and the project
            # environment, instead of on a laptop afterwards. The step is best effort.
            {"id": "aggregate", "kind": "shell-script", "path": "kaggle/aggregate_geneval.sh", "timeout_s": 900},
            # Scores and report only: the images are already in the essential archive.
            {"id": "pack-geneval", "kind": "shell-script", "path": "kaggle/pack_artifacts.sh",
             "env": {"PACK_ITEMS": "geneval_results metrics FINAL_REPORT.md",
                     "PACK_OUT": "{working_root}/geneval_artifacts.tar.gz"}},
        ]
        if args.with_hps:
            # kaggle/run_hps.py restores the CLIP BPE asset that setup/setup.sh downloads
            # and surfaces tracebacks on stdout, then runs the published evaluator.
            steps.insert(5, {
                "id": "hps",
                "kind": "python-script",
                "path": "kaggle/run_hps.py",
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
            })
        outputs = [
            {"id": "env-check", "kind": "json", "path": "{working_root}/env_check.json", "required": True, "min_bytes": 2},
            {
                "id": "generation",
                "kind": "directory",
                "path": "{project_root}/exps/single_stage_validation_553/outputs",
                "required": False,
            },
            {
                "id": "geneval-inputs",
                "kind": "directory",
                "path": "{project_root}/exps/single_stage_validation_553/geneval_inputs",
                "required": False,
            },
            {"id": "artifacts", "kind": "file", "path": "{working_root}/artifacts.tar.gz", "required": False},
            {
                "id": "geneval-results",
                "kind": "directory",
                "path": "{project_root}/exps/single_stage_validation_553/geneval_results",
                "required": False,
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
    parser.add_argument(
        "--phase",
        choices=[
            "bank",
            "eval",
            "smoke-cpu",
            "smoke-gpu",
            "bench-decode",
            "geneval-build",
            "geneval-eval",
            "bench-speedup",
            "bench-batch",
            "report",
        ],
        required=True,
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--worker-indices", default="0,1")
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--expect-gpus", type=int, default=2)
    parser.add_argument("--with-hps", action="store_true", help="append evaluate_hps.py after the generation step")
    parser.add_argument("--commit", default="")
    parser.add_argument("--repo-url", default="https://github.com/sontungkieu/TQK")
    parser.add_argument("--python", default="3.10.17")
    parser.add_argument("--uv-version", default="0.10.2")
    parser.add_argument("--submit-accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.with_hps and args.limit_prompts:
        raise SystemExit(
            "--with-hps requires the full 553-prompt run: evaluate_hps.py iterates the whole\n"
            "prompt list and fails on the first missing outputs/<method>/<prompt_id>.png.\n"
            "run_all.sh likewise runs generation+validation only in its preflight."
        )

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
