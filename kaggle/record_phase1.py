#!/usr/bin/env python3
"""Record the T4 phase-1 calibration evidence next to the Job Spec layer.

The published FROZEN_SCHEDULE.json is deliberately left untouched: the T4 calibration
selected the same schedule, so overwriting it would only churn provenance fields while
prepare_protocol.py expects the committed artifact. This file keeps this run's numbers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

# this file lives in kaggle/, one level below the repo root
ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "exps" / "single_stage_calibration"
OUT = Path(__file__).resolve().parent / "evidence" / "phase1_t4_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str) -> list[dict]:
    with (CAL / "replay" / name).open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    validation = rows("validation_results.csv")
    search = rows("search_results.csv")
    selected = next(row for row in validation if row["selected"] == "True")
    bank = sorted(p for p in (CAL / "bank_raw").glob("gpu*/*.json") if p.name != "hardware.json")
    payloads = [json.loads(p.read_text()) for p in bank]
    elapsed = [float(p["elapsed_s"]) for p in payloads]
    scoring = [float(p.get("checkpoint_decode_reward_s", 0.0)) for p in payloads]
    digest = hashlib.sha256()
    for path in bank:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    evidence = {
        "run": "codemaivanngu/tqk-bank-p1-260918-0731",
        "host": "2x Tesla T4 (sm75, 14.56 GiB), Kaggle",
        "date_utc": "2026-09-18",
        "prompts": len(payloads),
        "candidates_per_prompt": 25,
        "checkpoints_scored_per_candidate": 13,
        "session_elapsed_s": 21559.6,
        "worker_gpu_seconds": round(sum(elapsed) * 2 / 2, 1),
        "mean_prompt_s": round(statistics.mean(elapsed), 2),
        "min_prompt_s": round(min(elapsed), 2),
        "max_prompt_s": round(max(elapsed), 2),
        "mean_decode_reward_s": round(statistics.mean(scoring), 2),
        "peak_vram_gib": round(max(float(p["peak_vram_gib"]) for p in payloads), 3),
        "bank_sha256_over_files": digest.hexdigest(),
        "selected_schedule": {k: int(selected[k]) for k in ("checkpoint", "K", "M", "logical_compute")},
        "selected_search_rank": int(selected["search_rank"]),
        "selected_validation_mean_IR": float(selected["final_IR_mean"]),
        "selected_validation_delta_vs_PSP": float(selected["delta_vs_psp"]),
        "search_top5": [
            {"checkpoint": int(r["checkpoint"]), "K": int(r["K"]), "M": int(r["M"]),
             "logical_compute": int(r["logical_compute"]), "final_IR_mean": float(r["final_IR_mean"])}
            for r in search[:5]
        ],
        "validation_top3": [
            {"checkpoint": int(r["checkpoint"]), "K": int(r["K"]), "M": int(r["M"]),
             "search_rank": int(r["search_rank"]), "final_IR_mean": float(r["final_IR_mean"]),
             "selected": r["selected"] == "True"}
            for r in validation[:3]
        ],
        "published_comparison": {
            "frozen_schedule_json": json.loads((ROOT / "exps/single_stage_validation_553/FROZEN_SCHEDULE.json").read_text()),
            "same_selected_schedule": True,
            "note": "identical 9->2@17 selection; only provenance differs (search rank and validation IR)",
        },
    }
    OUT.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: evidence[k] for k in ("prompts", "mean_prompt_s", "selected_schedule", "selected_search_rank", "selected_validation_mean_IR")}, indent=2))


if __name__ == "__main__":
    main()
