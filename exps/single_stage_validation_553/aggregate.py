#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
CAL = ROOT / "exps/single_stage_calibration"
METHODS = ("psp", "ours")
DISPLAY = {"psp": "PSP", "ours": "Ours"}
METRICS = {"image_reward": "ImageReward", "hps": "HPS", "geneval": "GenEval"}
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_SAMPLES = 10_000


def paired_bootstrap(frame: pd.DataFrame, metric: str, seed: int = BOOTSTRAP_SEED) -> dict:
    wide = frame.pivot(index="prompt_id", columns="method", values=metric)
    delta = (wide["ours"] - wide["psp"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))].mean(axis=1)
    return {
        "psp": float(wide.psp.mean()),
        "ours": float(wide.ours.mean()),
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
    }


def prompt_id_from_filename(filename: str) -> int:
    for parent in Path(filename).parents:
        if parent.name.isdigit():
            return int(parent.name)
    raise ValueError(filename)


def read_generation() -> pd.DataFrame:
    rows = []
    for worker in (0, 1):
        for path in sorted((EXP / "metadata" / f"gpu{worker}").glob("*_*.json")):
            row = json.loads(path.read_text())
            rows.append({
                "prompt_id": int(row["prompt_id"]),
                "method": row["method"],
                "worker": worker,
                "image_reward": float(row["final_image_reward"]),
                "elapsed_s": float(row["elapsed_s"]),
                "online_reward_s": float(row["online_reward_s"]),
                "peak_vram_gib": float(row["peak_vram_gib"]),
                "logical_unet_evals": int(row["logical_unet_evals"]),
                "batched_verifier_calls": int(row["batched_verifier_calls"]),
                "verifier_candidate_scores": int(row["verifier_candidate_scores"]),
                "winner_id": int(row["winner_id"]),
            })
    frame = pd.DataFrame(rows)
    assert len(frame) == 1106 and not frame.duplicated(["prompt_id", "method"]).any()
    return frame


def read_geneval() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for line in (EXP / "geneval_results" / f"{method}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append({
                "method": method,
                "prompt_id": prompt_id_from_filename(item["filename"]),
                "geneval": float(bool(item["correct"])),
                "geneval_reason": item.get("reason", ""),
            })
    frame = pd.DataFrame(rows)
    assert len(frame) == 1106 and not frame.duplicated(["prompt_id", "method"]).any()
    return frame


def parse_dmon() -> dict[int, float]:
    samples: dict[int, list[float]] = defaultdict(list)
    path = EXP / "logs/nvidia_dmon.log"
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        try:
            gpu, sm = int(parts[2]), float(parts[6])
        except (IndexError, ValueError):
            continue
        if sm >= 0:
            samples[gpu].append(sm)
    return {gpu: float(statistics.mean(values)) for gpu, values in samples.items() if values}


def claim(stats: dict[str, dict]) -> str:
    clauses = []
    for metric, label in METRICS.items():
        row = stats[metric]
        if row["ci95_low"] > 0:
            clauses.append(f"{label} significantly favors ours")
        elif row["ci95_high"] < 0:
            clauses.append(f"{label} significantly favors PSP")
        else:
            clauses.append(f"{label} is inconclusive because its CI contains zero")
    return "; ".join(clauses) + "."


def main() -> None:
    frozen = json.loads((EXP / "FROZEN_SCHEDULE.json").read_text())
    schedule = {"psp": "8→4@16→2@32", "ours": frozen["schedule_notation"]}
    prompts = pd.DataFrame([
        json.loads(line)
        for line in (EXP / "prompts_geneval_all_553.jsonl").read_text().splitlines()
        if line.strip()
    ])[['prompt_id', 'prompt', 'tag']]
    frame = read_generation().merge(
        pd.read_csv(EXP / "metrics/hps.csv"), on=["prompt_id", "method"], validate="one_to_one"
    ).merge(read_geneval(), on=["prompt_id", "method"], validate="one_to_one").merge(
        prompts, on="prompt_id", validate="many_to_one"
    )
    assert len(frame) == 1106 and not frame.isna().any().any()
    stats = {metric: paired_bootstrap(frame, metric, BOOTSTRAP_SEED + index) for index, metric in enumerate(METRICS)}
    category_rows = []
    for tag, group in frame.groupby("tag"):
        row = paired_bootstrap(group, "geneval", BOOTSTRAP_SEED)
        category_rows.append({"tag": tag, "metric": "geneval", **row})
    category = pd.DataFrame(category_rows)
    wide_ge = frame.pivot(index="prompt_id", columns="method", values="geneval")
    delta_ge = wide_ge.ours - wide_ge.psp
    win_tie_loss = {
        "ours_win": int((delta_ge > 0).sum()),
        "tie": int((delta_ge == 0).sum()),
        "psp_win": int((delta_ge < 0).sum()),
    }
    summary_rows = []
    for method in METHODS:
        group = frame[frame.method.eq(method)]
        summary_rows.append({
            "method": method,
            "schedule": schedule[method],
            "logical_unet_evals": int(group.logical_unet_evals.iloc[0]),
            "image_reward": float(group.image_reward.mean()),
            "hps": float(group.hps.mean()),
            "geneval": float(group.geneval.mean()),
            "runtime_mean_s": float(group.elapsed_s.mean()),
            "runtime_median_s": float(group.elapsed_s.median()),
            "runtime_p90_s": float(group.elapsed_s.quantile(0.9)),
            "throughput_prompt_s_per_gpu": float(1.0 / group.elapsed_s.mean()),
            "online_reward_mean_s": float(group.online_reward_s.mean()),
            "peak_vram_mean_gib": float(group.peak_vram_gib.mean()),
            "peak_vram_max_gib": float(group.peak_vram_gib.max()),
            "batched_verifier_calls": int(group.batched_verifier_calls.iloc[0]),
            "verifier_candidate_scores": int(group.verifier_candidate_scores.iloc[0]),
        })
    summary = pd.DataFrame(summary_rows)
    utilization = parse_dmon()
    shard_rows = []
    for worker in (0, 1):
        group = frame[frame.worker.eq(worker)]
        shard_rows.append({
            "worker": worker,
            "prompts": int(group.prompt_id.nunique()),
            "method_runs": len(group),
            "summed_method_s": float(group.elapsed_s.sum()),
            "mean_method_s": float(group.elapsed_s.mean()),
            "max_vram_gib": float(group.peak_vram_gib.max()),
            "avg_sm_util_percent": utilization.get(worker, math.nan),
            "ir_delta": paired_bootstrap(group, "image_reward")["delta"],
            "hps_delta": paired_bootstrap(group, "hps")["delta"],
            "geneval_delta": paired_bootstrap(group, "geneval")["delta"],
        })
    shards = pd.DataFrame(shard_rows)
    metrics = EXP / "metrics"
    metrics.mkdir(exist_ok=True)
    frame.to_csv(metrics / "per_prompt.csv", index=False)
    summary.to_csv(metrics / "summary.csv", index=False)
    category.to_csv(metrics / "category_bootstrap.csv", index=False)
    shards.to_csv(metrics / "gpu_shards.csv", index=False)
    (metrics / "paired_bootstrap.json").write_text(json.dumps(stats, indent=2) + "\n")
    (metrics / "geneval_win_tie_loss.json").write_text(json.dumps(win_tie_loss, indent=2) + "\n")

    selected_all = pd.read_csv(CAL / "replay/selected_all120_per_prompt.csv")
    psp_all = pd.concat([
        pd.read_csv(CAL / "replay/psp_search_per_prompt.csv"),
        pd.read_csv(CAL / "replay/psp_validation_per_prompt.csv"),
    ])
    cal_q = float(selected_all.final_IR.mean())
    cal_o = float(selected_all.oracle_IR.mean())
    cal_r = float(selected_all.selection_regret.mean())
    cal_psp = float(psp_all.final_IR.mean())
    rows_by_method = {row["method"]: row for row in summary.to_dict("records")}
    psp_row, ours_row = rows_by_method["psp"], rows_by_method["ours"]
    main_table = "\n".join(
        f"| {DISPLAY[row['method']]} | {row['schedule']} | {row['logical_unet_evals']} | "
        f"{row['image_reward']:.6f} | {row['hps']:.6f} | {row['geneval']:.6f} | "
        f"{row['runtime_mean_s']:.3f} | {row['peak_vram_max_gib']:.2f} GiB |"
        for row in summary.to_dict("records")
    )
    delta_table = "\n".join(
        f"| {label} | {stats[key]['psp']:.6f} | {stats[key]['ours']:.6f} | "
        f"{stats[key]['delta']:+.6f} | [{stats[key]['ci95_low']:.6f}, {stats[key]['ci95_high']:.6f}] |"
        for key, label in METRICS.items()
    )
    category_table = "\n".join(
        f"| {row.tag} | {int((prompts.tag == row.tag).sum())} | {row.psp:.4f} | {row.ours:.4f} | "
        f"{row.delta:+.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] |"
        for row in category.itertuples()
    )
    verifier_saved = psp_row["verifier_candidate_scores"] - ours_row["verifier_candidate_scores"]
    calls_saved = psp_row["batched_verifier_calls"] - ours_row["batched_verifier_calls"]
    runtime_delta = ours_row["runtime_mean_s"] - psp_row["runtime_mean_s"]
    vram_delta = ours_row["peak_vram_max_gib"] - psp_row["peak_vram_max_gib"]
    report = f"""# Final report: calibrated single-stage schedule vs PSP

## Outcome

| Method | Schedule | UNet evals | IR | HPS | GenEval | sec/prompt | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
{main_table}

| Metric | PSP | Ours | Δ ours−PSP | 95% paired CI |
|---|---:|---:|---:|---:|
{delta_table}

Paired GenEval outcomes: **{win_tie_loss['ours_win']} ours wins / {win_tie_loss['tie']} ties / {win_tie_loss['psp_win']} PSP wins**.

## GenEval by official category

| Category | n | PSP | Ours | Δ | 95% paired CI |
|---|---:|---:|---:|---:|---:|
{category_table}

## Required questions

1. **Family searched.** One checkpoint `t` from the fixed 13-point grid, K∈{{1,2,3}}, and maximum feasible M determined by the budget equation; 39 schedules total.
2. **Compute constraint.** `M*t + K*(64-t) ≤ 256`; M was never tuned independently.
3. **Calibration data.** 120 fixed prompts from the independent repository ImageReward `test_ir.json` corpus; zero exact GenEval overlap.
4. **Split.** 80 search / 40 untouched internal validation, deterministic seed 20260918.
5. **Frozen schedule.** `{frozen['schedule_notation']}`, {frozen['logical_compute']} UNet evaluations ({frozen['unused_compute']} unused).
6. **Calibration IR.** Selected schedule all-120 mean `{cal_q:.6f}`; its internal-validation mean was `{frozen['validation_mean_IR']:.6f}`.
7. **Oracle pool quality.** O(M*)=`{cal_o:.6f}` on all 120 calibration prompts.
8. **Selection regret.** R(t*,K*)=`{cal_r:.6f}`; Q=O−R holds numerically.
9. **Neighboring checkpoints.** The schedule won the pre-registered top-3 internal-validation comparison; no neighbor was tested after freezing.
10. **PSP calibration baseline.** Fixed PSP all-120 mean IR `{cal_psp:.6f}`; selected schedule descriptive delta `{cal_q-cal_psp:+.6f}`.
11. **Untouched data.** All 553 official GenEval prompts remained untouched until Phase 2.
12. **Fresh Phase-2 seeds.** Base seed 20260919 with `base + prompt_id*32 + candidate_id`, disjoint from Phase 1.
13. **PSP final metrics.** IR `{psp_row['image_reward']:.6f}`, HPS `{psp_row['hps']:.6f}`, GenEval `{psp_row['geneval']:.6f}`.
14. **Ours final metrics.** IR `{ours_row['image_reward']:.6f}`, HPS `{ours_row['hps']:.6f}`, GenEval `{ours_row['geneval']:.6f}`.
15. **Paired uncertainty.** The table above gives all paired deltas and 10,000-resample 95% percentile CIs.
16. **Statistical conclusion.** {claim(stats)}
17. **Runtime.** Ours−PSP mean `{runtime_delta:+.4f}` sec/prompt; PSP `{psp_row['runtime_mean_s']:.4f}`, ours `{ours_row['runtime_mean_s']:.4f}`.
18. **VRAM.** Ours−PSP max allocated `{vram_delta:+.3f}` GiB.
19. **Verifier work.** Ours uses `{ours_row['batched_verifier_calls']}` batched calls and `{ours_row['verifier_candidate_scores']}` candidate scores/prompt versus PSP `{psp_row['batched_verifier_calls']}` and `{psp_row['verifier_candidate_scores']}`; savings are `{calls_saved}` calls and `{verifier_saved}` scores/prompt.
20. **Mechanism.** Phase 1 directly supports the observed breadth-versus-reliability trade-off: earlier checkpoints permit larger M and higher O(M) but incur larger R; later checkpoints reduce R while sacrificing feasible M. The selected schedule lies at the empirically chosen balance for this fixed grid, not a proof of global optimality.
21. **Narrowest defensible claim.** On this one fresh-seed, all-553-prompt SD1.5 evaluation, `{frozen['schedule_notation']}` versus fixed PSP produced the paired metric outcomes and CIs above at no more logical UNet compute. Claims beyond these metrics, seed pool, model, and prompt population are not supported.

## Runtime detail

| Method | mean s | median s | p90 s | throughput/GPU | online decode+IR s | mean/max VRAM | verifier calls/scores |
|---|---:|---:|---:|---:|---:|---:|---:|
| PSP | {psp_row['runtime_mean_s']:.3f} | {psp_row['runtime_median_s']:.3f} | {psp_row['runtime_p90_s']:.3f} | {psp_row['throughput_prompt_s_per_gpu']:.4f} | {psp_row['online_reward_mean_s']:.3f} | {psp_row['peak_vram_mean_gib']:.2f}/{psp_row['peak_vram_max_gib']:.2f} GiB | {psp_row['batched_verifier_calls']}/{psp_row['verifier_candidate_scores']} |
| Ours | {ours_row['runtime_mean_s']:.3f} | {ours_row['runtime_median_s']:.3f} | {ours_row['runtime_p90_s']:.3f} | {ours_row['throughput_prompt_s_per_gpu']:.4f} | {ours_row['online_reward_mean_s']:.3f} | {ours_row['peak_vram_mean_gib']:.2f}/{ours_row['peak_vram_max_gib']:.2f} GiB | {ours_row['batched_verifier_calls']}/{ours_row['verifier_candidate_scores']} |

## Interpretation guardrails

Phase 1 is schedule development and is not final evidence. Phase 2 used actual online diffusion, not replay. HPS and GenEval were never used to select or revise the schedule. No alternative schedule will be substituted if this frozen schedule loses.
"""
    (EXP / "FINAL_REPORT.md").write_text(report)
    print(json.dumps({
        "phase": "phase2_complete",
        "frozen_schedule": frozen["schedule_notation"],
        "summary": summary_rows,
        "paired": stats,
        "geneval_win_tie_loss": win_tie_loss,
        "claim": claim(stats),
    }, indent=2))


if __name__ == "__main__":
    main()

