#!/usr/bin/env python3
"""Merge TQK bank shards produced by separate Kaggle sessions.

Each session writes bank_raw/gpu<worker-index>/<prompt_id:05d>.json. Because the
workers partition deterministic prompt ids by (worker_index, num_workers), the
union of all shard directories is exactly the complete bank. This tool copies the
per-prompt JSON files into one bank_raw directory, refuses silent conflicts, and
can then run the upstream validator.

  kaggle/merge_shards.py --input run_a/output --input run_b/output \
      --dest exps/single_stage_calibration/bank_raw --expect-prompts 120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root: Path, bank_dir: str = "bank_raw") -> list[Path]:
    return sorted(
        path
        for path in root.rglob(f"{bank_dir}/gpu*/*.json")
        if path.name != "hardware.json" and path.stem.isdigit()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="downloaded output root; repeatable")
    parser.add_argument("--dest", required=True, help="target bank_raw directory")
    parser.add_argument("--expect-prompts", type=int, default=0)
    parser.add_argument(
        "--bank-dir",
        default="bank_raw",
        help="shard directory name to collect (default bank_raw; the 200-prompt bank uses bank200)",
    )
    parser.add_argument("--copy-hardware", action="store_true", help="also copy hardware.json files into dest/hardware")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    per_input: dict[str, int] = {}
    prompt_files: dict[int, tuple[Path, str]] = {}
    duplicates = 0
    conflicts: list[str] = []
    copied = 0

    for raw_input in args.input:
        root = Path(raw_input)
        files = collect(root, args.bank_dir)
        per_input[str(root)] = len(files)
        for path in files:
            prompt_id = int(path.stem)
            digest = sha256(path)
            if prompt_id in prompt_files:
                duplicates += 1
                previous_path, previous_digest = prompt_files[prompt_id]
                if previous_digest != digest:
                    conflicts.append(f"prompt {prompt_id}: {previous_path} != {path}")
                continue
            prompt_files[prompt_id] = (path, digest)

    for prompt_id, (path, _digest) in sorted(prompt_files.items()):
        target = dest / f"gpu0" / f"{prompt_id:05d}.json"
        if target.exists() and sha256(target) == sha256(path):
            continue
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        copied += 1

    if args.copy_hardware and not args.dry_run:
        hardware_dir = dest / "hardware"
        hardware_dir.mkdir(parents=True, exist_ok=True)
        for raw_input in args.input:
            for path in Path(raw_input).rglob(f"{args.bank_dir}/gpu*/hardware.json"):
                target = hardware_dir / f"{path.parent.name}_{path.parent.parent.parent.parent.name}_hardware.json"
                shutil.copy2(path, target)

    report = {
        "inputs": per_input,
        "unique_prompts": len(prompt_files),
        "copied": copied,
        "duplicates_skipped": duplicates,
        "conflicts": conflicts,
        "dest": str(dest),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(report, indent=2))

    if conflicts:
        raise SystemExit("conflicting bank payloads for the same prompt id; refusing to merge")
    if args.expect_prompts and len(prompt_files) != args.expect_prompts:
        raise SystemExit(f"expected {args.expect_prompts} prompts, collected {len(prompt_files)}")
    missing = []
    if args.expect_prompts:
        missing = [pid for pid in range(args.expect_prompts) if pid not in prompt_files]
        if missing:
            raise SystemExit(f"missing prompt ids: {missing[:20]}{' ...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    main()
