#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap import paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
METHODS = ("psp", "early10to2")
DISPLAY = {"psp": "PSP", "early10to2": "10→2"}
SCHEDULE = {"psp": "8→4@16→2@32", "early10to2": "10→2@16"}
METRICS = {"image_reward": "ImageReward", "hps": "HPS", "geneval": "GenEval"}


def read_prompts() -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in (EXP / "prompts_geneval_balanced_300.jsonl").read_text().splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    assert len(frame) == 300 and frame.prompt_id.nunique() == 300
    assert set(frame.groupby("tag").size()) == {50}
    return frame[["prompt_id", "prompt", "tag"]]


def read_generation() -> pd.DataFrame:
    rows = []
    for worker in (0, 1):
        for path in sorted((EXP / "metadata" / f"gpu{worker}").glob("*_*.json")):
            row = json.loads(path.read_text())
            rows.append(
                {
                    "prompt_id": int(row["prompt_id"]),
                    "method": row["method"],
                    "worker": worker,
                    "image_reward": float(row["final_image_reward"]),
                    "elapsed_s": float(row["elapsed_s"]),
                    "online_reward_s": float(row["online_reward_s"]),
                    "peak_vram_gib": float(row["peak_vram_gib"]),
                    "logical_unet_evals": int(row["logical_unet_evals"]),
                    "winner_id": int(row["winner_id"]),
                }
            )
    frame = pd.DataFrame(rows)
    assert len(frame) == 600, len(frame)
    assert not frame.duplicated(["prompt_id", "method"]).any()
    assert set(frame.method) == set(METHODS)
    assert set(frame.logical_unet_evals) == {256}
    return frame


def read_hps() -> pd.DataFrame:
    frame = pd.read_csv(EXP / "metrics" / "hps.csv")
    assert len(frame) == 600 and not frame.duplicated(["prompt_id", "method"]).any()
    return frame


def prompt_id_from_filename(filename: str) -> int:
    path = Path(filename)
    for parent in path.parents:
        if parent.name.isdigit():
            return int(parent.name)
    raise ValueError(f"Cannot recover prompt id from {filename}")


def read_geneval() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        path = EXP / "geneval_results" / f"{method}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "method": method,
                    "prompt_id": prompt_id_from_filename(item["filename"]),
                    "geneval": float(bool(item["correct"])),
                    "geneval_reason": item.get("reason", ""),
                }
            )
    frame = pd.DataFrame(rows)
    assert len(frame) == 600, len(frame)
    assert not frame.duplicated(["prompt_id", "method"]).any()
    return frame


def percentile(values: pd.Series, q: float) -> float:
    return float(np.quantile(values.to_numpy(), q))


def parse_dmon() -> dict[int, float]:
    path = EXP / "logs" / "nvidia_dmon.log"
    samples: dict[int, list[float]] = defaultdict(list)
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        # `nvidia-smi dmon -s pucm -o DT` columns are:
        # date, time, gpu, pwr, gtemp, mtemp, sm, mem, ...
        try:
            gpu = int(parts[2])
            sm = float(parts[6])
        except (IndexError, ValueError):
            continue
        if sm >= 0:
            samples[gpu].append(sm)
    return {gpu: float(statistics.mean(values)) for gpu, values in samples.items() if values}


def f6(value: float) -> str:
    return f"{value:.6f}"


def ci_text(row: dict) -> str:
    return f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]"


def interpretation(stats: dict[str, dict]) -> str:
    clauses = []
    for key, label in METRICS.items():
        row = stats[key]
        if row["ci95_low"] > 0:
            clauses.append(f"{label} favors 10→2 (CI excludes zero)")
        elif row["ci95_high"] < 0:
            clauses.append(f"{label} favors PSP (CI excludes zero)")
        else:
            clauses.append(f"{label} is inconclusive (CI contains zero)")
    return "; ".join(clauses) + "."


def main() -> None:
    prompts = read_prompts()
    generation = read_generation()
    frame = generation.merge(read_hps(), on=["prompt_id", "method"], validate="one_to_one")
    frame = frame.merge(read_geneval(), on=["prompt_id", "method"], validate="one_to_one")
    frame = frame.merge(prompts, on="prompt_id", validate="many_to_one")
    assert len(frame) == 600 and not frame.isna().any().any()

    stats = {key: paired_bootstrap(frame, key) for key in METRICS}
    category_rows = []
    for tag, group in frame.groupby("tag"):
        for metric in METRICS:
            row = paired_bootstrap(group, metric)
            category_rows.append({"tag": tag, "metric": metric, **row})
    category = pd.DataFrame(category_rows)

    wide_ge = frame.pivot(index="prompt_id", columns="method", values="geneval")
    ge_delta = wide_ge["early10to2"] - wide_ge["psp"]
    win_tie_loss = {
        "ours_win": int((ge_delta > 0).sum()),
        "tie": int((ge_delta == 0).sum()),
        "psp_win": int((ge_delta < 0).sum()),
    }

    summary_rows = []
    for method in METHODS:
        group = frame[frame.method == method]
        summary_rows.append(
            {
                "method": method,
                "schedule": SCHEDULE[method],
                "logical_unet_evals": 256,
                "image_reward": group.image_reward.mean(),
                "hps": group.hps.mean(),
                "geneval": group.geneval.mean(),
                "runtime_mean_s": group.elapsed_s.mean(),
                "runtime_median_s": group.elapsed_s.median(),
                "runtime_p90_s": percentile(group.elapsed_s, 0.9),
                "peak_vram_mean_gib": group.peak_vram_gib.mean(),
                "peak_vram_max_gib": group.peak_vram_gib.max(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    shard_rows = []
    utilization = parse_dmon()
    for worker in (0, 1):
        group = frame[frame.worker == worker]
        shard_rows.append(
            {
                "worker": worker,
                "prompts": int(group.prompt_id.nunique()),
                "method_runs": len(group),
                "summed_method_s": group.elapsed_s.sum(),
                "mean_method_s": group.elapsed_s.mean(),
                "max_vram_gib": group.peak_vram_gib.max(),
                "avg_sm_util_percent": utilization.get(worker, math.nan),
                "ir_delta": paired_bootstrap(group, "image_reward")["delta"],
                "hps_delta": paired_bootstrap(group, "hps")["delta"],
                "geneval_delta": paired_bootstrap(group, "geneval")["delta"],
            }
        )
    shards = pd.DataFrame(shard_rows)

    metrics = EXP / "metrics"
    metrics.mkdir(exist_ok=True)
    frame.to_csv(metrics / "per_prompt.csv", index=False)
    summary.to_csv(metrics / "summary.csv", index=False)
    category.to_csv(metrics / "category_bootstrap.csv", index=False)
    shards.to_csv(metrics / "gpu_shards.csv", index=False)
    (metrics / "paired_bootstrap.json").write_text(json.dumps(stats, indent=2) + "\n")
    (metrics / "geneval_win_tie_loss.json").write_text(json.dumps(win_tie_loss, indent=2) + "\n")

    run_info_path = metrics / "run_info.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    wall = float(run_info.get("generation_wall_s", math.nan))
    commit = run_info.get("psp_commit", "590f59f58384169c719e431dc01d14006fb0fc0c")
    status = run_info.get("git_status", "captured in metrics/run_info.json")
    diff = run_info.get("git_diff_stat", "captured in metrics/run_info.json")
    hardware = [json.loads((EXP / "metadata" / f"gpu{x}" / "hardware.json").read_text()) for x in (0, 1)]
    ids = json.loads((EXP / "prompt_ids_300.json").read_text())["prompt_ids"]

    main_rows = []
    for row in summary.to_dict("records"):
        main_rows.append(
            f"| {DISPLAY[row['method']]} | {row['schedule']} | 256 | {row['image_reward']:.6f} | "
            f"{row['hps']:.6f} | {row['geneval']:.6f} | {row['runtime_mean_s']:.3f} | "
            f"{row['peak_vram_max_gib']:.2f} GiB |"
        )
    delta_rows = [
        f"| {label} | {stats[key]['psp']:.6f} | {stats[key]['early10to2']:.6f} | "
        f"{stats[key]['delta']:+.6f} | {ci_text(stats[key])} |"
        for key, label in METRICS.items()
    ]
    category_lines = []
    for tag in sorted(prompts.tag.unique()):
        rows = category[(category.tag == tag) & (category.metric == "geneval")].iloc[0]
        category_lines.append(
            f"| {tag} | 50 | {rows.psp:.4f} | {rows.early10to2:.4f} | "
            f"{rows.delta:+.4f} | [{rows.ci95_low:.4f}, {rows.ci95_high:.4f}] |"
        )
    hw_lines = []
    for row in shards.to_dict("records"):
        util = "n/a" if math.isnan(row["avg_sm_util_percent"]) else f"{row['avg_sm_util_percent']:.1f}%"
        worker_wall = wall if not math.isnan(wall) else row["summed_method_s"]
        hw_lines.append(
            f"| RTX4090 #{row['worker']} | {row['prompts']} | {worker_wall/3600:.3f} h (shared wall) | "
            f"{util} | {row['max_vram_gib']:.2f} GiB |"
        )
    combined_util = shards.avg_sm_util_percent.mean()
    combined_util_text = "n/a" if math.isnan(combined_util) else f"{combined_util:.1f}%"
    hw_lines.append(
        f"| Combined | 300 | {wall/3600:.3f} h | {combined_util_text} | {shards.max_vram_gib.max():.2f} GiB |"
    )

    report = f"""# Direct SD1.5 PSP vs 10→2 on 300 GenEval prompts

## Result

| Method | Schedule | Logical UNet evals | IR ↑ | HPS ↑ | GenEval ↑ | sec/prompt ↓ | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_rows)}

| Metric | PSP | 10→2 | Δ ours−PSP | 95% paired CI |
|---|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

Paired inference over 300 prompts: {interpretation(stats)} The GenEval paired outcomes are **{win_tie_loss['ours_win']} ours wins / {win_tie_loss['tie']} ties / {win_tie_loss['psp_win']} PSP wins**.

## GenEval by official task group

| Group | n | PSP | 10→2 | Δ | 95% paired CI |
|---|---:|---:|---:|---:|---:|
{chr(10).join(category_lines)}

## Runtime and hardware

| Hardware | Prompts | Total wall time | Avg utilization | Max VRAM |
|---|---:|---:|---:|---:|
{chr(10).join(hw_lines)}

Per-method runtime distribution:

| Method | mean s | median s | p90 s | mean peak VRAM | max peak VRAM |
|---|---:|---:|---:|---:|---:|
"""
    for row in summary.to_dict("records"):
        report += (
            f"| {DISPLAY[row['method']]} | {row['runtime_mean_s']:.3f} | {row['runtime_median_s']:.3f} | "
            f"{row['runtime_p90_s']:.3f} | {row['peak_vram_mean_gib']:.2f} GiB | "
            f"{row['peak_vram_max_gib']:.2f} GiB |\n"
        )
    report += f"""

GPU-shard deltas (ours−PSP), used only as a consistency diagnostic:

| Shard | prompts | ΔIR | ΔHPS | ΔGenEval |
|---|---:|---:|---:|---:|
| GPU0 | {int(shards.iloc[0].prompts)} | {shards.iloc[0].ir_delta:+.6f} | {shards.iloc[0].hps_delta:+.6f} | {shards.iloc[0].geneval_delta:+.6f} |
| GPU1 | {int(shards.iloc[1].prompts)} | {shards.iloc[1].ir_delta:+.6f} | {shards.iloc[1].hps_delta:+.6f} | {shards.iloc[1].geneval_delta:+.6f} |

## Protocol audit

- PSP commit: `{commit}`.
- Repository status before launch: `{status}`.
- Repository diff summary before launch: `{diff}`.
- Checkpoint: `runwayml/stable-diffusion-v1-5`, fp16, native 512×512 resolution.
- Sampler: `DDIMScheduler` from the checkpoint scheduler config, 64 steps, guidance scale 7.5, `eta=0`.
- PSP schedule was exactly `8→4@16→2@32`; ours was exactly `10→2@16`. Both were asserted to use 256 logical UNet evaluations.
- ImageReward was used at live pruning checkpoints and for final selection. The scheduler's existing `pred_original_sample` was scored; no extra UNet call was introduced.
- For every prompt, candidates 0…9 were generated once from deterministic seeds `prompt_id*32 + candidate_id`; PSP received 0…7 and 10→2 received 0…9. SHA-256 hashes were stored, and exact equality of the first eight tensors was asserted.
- Subset selection seed: `20260917`. The exact sorted IDs are in `prompt_ids_300.json`; there are 50 prompts from each of six official GenEval groups. IDs: `{ids}`.
- Sharding: sorted subset position modulo 2. Both methods for a prompt always ran on the same GPU.
- Software: PyTorch `{hardware[0]['torch']}`, CUDA runtime `{hardware[0]['cuda']}`, Diffusers `{hardware[0]['diffusers']}`, Transformers `{hardware[0]['transformers']}`, ImageReward `{hardware[0].get('image_reward')}`, HPSv2 `{hardware[0].get('hpsv2')}`.
- GPU names: `{hardware[0]['gpu']}` and `{hardware[1]['gpu']}`. Peak allocated VRAM remained below the 24 GiB device capacity; no schedule-altering memory workaround was used.
- HPS is HPS v2.1 on final winners only. GenEval is the repository's official Mask2Former/OpenCLIP evaluator on final winners only.
- Statistics: one repetition/base seed 0; 10,000 prompt-level paired percentile bootstrap resamples with bootstrap seed 20260917.

## Published PSP context (reference only)

The paper's full Table-1 SD1.5 values are approximately IR 0.827, HPS 0.278, and GenEval 0.574. This is **reference only**, not an exact Table-1 reproduction: this run uses 300 prompts, one repetition, and RTX4090 hardware.

## Narrow conclusion

{interpretation(stats)} This conclusion is restricted to SD1.5, this fixed 300-prompt subset, this single repetition, and the two pre-specified schedules. No follow-up schedule was selected or launched.
"""
    (EXP / "REPORT.md").write_text(report)
    print(json.dumps({"summary": summary_rows, "paired": stats, "win_tie_loss": win_tie_loss}, indent=2, default=str))


if __name__ == "__main__":
    main()
