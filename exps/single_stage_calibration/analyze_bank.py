#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from schedules import CHECKPOINTS, valid_schedules

EXP = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 20260918
BOOTSTRAP_RESAMPLES = 10_000
TIE_MARGIN = 0.002


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def load_bank() -> pd.DataFrame:
    prompt_rows = []
    paths = sorted(
        path for path in (EXP / "bank_raw").glob("gpu*/*.json") if path.name != "hardware.json"
    )
    assert len(paths) == 120
    for path in paths:
        payload = json.loads(path.read_text())
        for candidate in payload["candidates"]:
            prompt_rows.append({
                "prompt_id": int(payload["prompt_id"]),
                "prompt": payload["prompt"],
                "source_id": payload["source_id"],
                "source_index": int(payload["source_index"]),
                "split": payload["split"],
                **candidate,
            })
    bank = pd.DataFrame(prompt_rows).sort_values(["prompt_id", "candidate_id"]).reset_index(drop=True)
    assert len(bank) == 120 * 25
    assert bank.groupby("prompt_id").size().eq(25).all()
    score_columns = [f"IR_t{step}" for step in CHECKPOINTS] + ["IR_final"]
    assert np.isfinite(bank[score_columns].to_numpy()).all()
    (EXP / "prompts").mkdir(exist_ok=True)
    bank.to_parquet(EXP / "prompts/calibration_bank.parquet", index=False)
    bank.to_csv(EXP / "prompts/calibration_bank.csv", index=False)
    return bank


def top_indices(values: np.ndarray, candidate_ids: np.ndarray, keep: int) -> np.ndarray:
    return np.lexsort((candidate_ids, -values))[:keep]


def replay_single(bank: pd.DataFrame, schedule: dict[str, int], prompt_ids: list[int]) -> pd.DataFrame:
    t, k, m = schedule["checkpoint"], schedule["K"], schedule["M"]
    rows = []
    for prompt_id in prompt_ids:
        group = bank[bank.prompt_id.eq(prompt_id) & bank.candidate_id.lt(m)].sort_values("candidate_id")
        assert len(group) == m
        candidate_ids = group.candidate_id.to_numpy(dtype=int)
        final = group.IR_final.to_numpy(dtype=float)
        intermediate = group[f"IR_t{t}"].to_numpy(dtype=float)
        survivor_positions = top_indices(intermediate, candidate_ids, k)
        survivor_ids = candidate_ids[survivor_positions]
        oracle_position = top_indices(final, candidate_ids, 1)[0]
        oracle_id = int(candidate_ids[oracle_position])
        oracle_quality = float(final[oracle_position])
        quality = float(final[survivor_positions].max())
        rows.append({
            "prompt_id": prompt_id,
            "checkpoint": t,
            "K": k,
            "M": m,
            "logical_compute": schedule["logical_compute"],
            "unused_compute": schedule["unused_compute"],
            "oracle_IR": oracle_quality,
            "final_IR": quality,
            "selection_regret": oracle_quality - quality,
            "oracle_candidate_id": oracle_id,
            "oracle_survived": int(oracle_id in set(int(value) for value in survivor_ids)),
            "survivor_candidate_ids": ",".join(str(int(value)) for value in survivor_ids),
        })
    return pd.DataFrame(rows)


def replay_psp(bank: pd.DataFrame, prompt_ids: list[int]) -> pd.DataFrame:
    rows = []
    for prompt_id in prompt_ids:
        group = bank[bank.prompt_id.eq(prompt_id) & bank.candidate_id.lt(8)].sort_values("candidate_id")
        ids = group.candidate_id.to_numpy(dtype=int)
        step16 = group.IR_t16.to_numpy(dtype=float)
        first = top_indices(step16, ids, 4)
        ids4 = ids[first]
        step32 = group.IR_t32.to_numpy(dtype=float)[first]
        second_local = top_indices(step32, ids4, 2)
        survivors = first[second_local]
        final = group.IR_final.to_numpy(dtype=float)
        winner_local = top_indices(final[survivors], ids[survivors], 1)[0]
        winner_position = survivors[winner_local]
        rows.append({
            "prompt_id": prompt_id,
            "final_IR": float(final[winner_position]),
            "winner_candidate_id": int(ids[winner_position]),
        })
    return pd.DataFrame(rows)


def summarize(replayed: pd.DataFrame, psp: pd.DataFrame, schedule: dict[str, int], seed: int) -> dict:
    merged = replayed.merge(psp, on="prompt_id", suffixes=("", "_psp"), validate="one_to_one")
    q = merged.final_IR.to_numpy(dtype=float)
    oracle = merged.oracle_IR.to_numpy(dtype=float)
    regret = merged.selection_regret.to_numpy(dtype=float)
    delta = q - merged.final_IR_psp.to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_mean(q, seed)
    delta_low, delta_high = bootstrap_mean(delta, seed + 100_000)
    decomposition_error = abs(float(q.mean()) - (float(oracle.mean()) - float(regret.mean())))
    assert decomposition_error < 1e-10
    return {
        **schedule,
        "n_prompts": len(merged),
        "oracle_IR": float(oracle.mean()),
        "selection_regret_mean": float(regret.mean()),
        "selection_regret_median": float(np.median(regret)),
        "selection_regret_p90": float(np.quantile(regret, 0.9)),
        "oracle_survival_probability": float(merged.oracle_survived.mean()),
        "final_IR_mean": float(q.mean()),
        "final_IR_median": float(np.median(q)),
        "final_IR_se": float(q.std(ddof=1) / math.sqrt(len(q))),
        "final_IR_ci_low": ci_low,
        "final_IR_ci_high": ci_high,
        "psp_IR_mean": float(merged.final_IR_psp.mean()),
        "delta_vs_psp": float(delta.mean()),
        "delta_vs_psp_ci_low": delta_low,
        "delta_vs_psp_ci_high": delta_high,
        "decomposition_abs_error": decomposition_error,
    }


def reliability(bank: pd.DataFrame, prompt_ids: list[int]) -> pd.DataFrame:
    rows = []
    subset = bank[bank.prompt_id.isin(prompt_ids)]
    for step in CHECKPOINTS:
        spearman = []
        kendall = []
        for _, group in subset.groupby("prompt_id"):
            x = group[f"IR_t{step}"].to_numpy(dtype=float)
            y = group.IR_final.to_numpy(dtype=float)
            spearman.append(float(spearmanr(x, y).statistic))
            kendall.append(float(kendalltau(x, y).statistic))
        rows.append({
            "checkpoint": step,
            "n_prompts": len(spearman),
            "mean_spearman": float(np.nanmean(spearman)),
            "median_spearman": float(np.nanmedian(spearman)),
            "mean_kendall": float(np.nanmean(kendall)),
            "median_kendall": float(np.nanmedian(kendall)),
        })
    return pd.DataFrame(rows)


def oracle_curve(bank: pd.DataFrame, prompt_ids: list[int]) -> pd.DataFrame:
    rows = []
    for m in range(2, 26):
        values = []
        for prompt_id in prompt_ids:
            group = bank[bank.prompt_id.eq(prompt_id) & bank.candidate_id.lt(m)]
            values.append(float(group.IR_final.max()))
        rows.append({"M": m, "n_prompts": len(values), "oracle_IR": float(np.mean(values))})
    return pd.DataFrame(rows)


def write_phase1_report(
    search: pd.DataFrame,
    validation: pd.DataFrame,
    selected: dict,
    psp_search: pd.DataFrame,
    psp_validation: pd.DataFrame,
    selected_all: pd.DataFrame,
) -> None:
    top3 = validation.sort_values("search_rank").checkpoint_schedule.tolist()
    best_search = search.sort_values("search_rank").iloc[0]
    selected_name = f'{selected["M"]}→{selected["K"]}@{selected["checkpoint"]}'
    all_q = float(selected_all.final_IR.mean())
    all_oracle = float(selected_all.oracle_IR.mean())
    all_regret = float(selected_all.selection_regret.mean())
    validation_row = validation[validation.selected.eq(True)].iloc[0]
    psp_all_mean = float(pd.concat([psp_search, psp_validation]).final_IR.mean())
    non_monotonic = any(
        not group.sort_values("checkpoint").final_IR_mean.is_monotonic_increasing
        for _, group in search.groupby("K")
    )
    report = f"""# Phase-1 calibration report

## Outcome

The frozen single-stage schedule is **{selected_name}** with
{selected['logical_compute']} logical UNet evaluations ({selected['unused_compute']} unused).
It was selected only after the top three search schedules were evaluated on the
untouched 40-prompt internal-validation split.

## Required protocol answers

1. **Calibration corpus.** 120 fixed prompts from the repository's ImageReward
   `test_ir.json` corpus, selected with seed 20260918.
2. **GenEval prompts used?** No. Exact normalized-string overlap with all 553
   official GenEval prompts is zero.
3. **Scale.** 120 prompts × 25 independent deterministic seeds = 3,000 complete
   64-step trajectories; 13 checkpoint scores plus one final score per trajectory.
4. **Valid schedules.** All 39 pre-registered combinations of 13 checkpoints and
   K∈{{1,2,3}} satisfy M>K, M≤25, and compute≤256.
5. **Oracle quality O(M).** See `replay/oracle_curve_search.csv` and plot B; it is
   non-decreasing in M by construction.
6. **Selection regret.** See `replay/search_results.csv` and plot C; later
   checkpoints generally improve ranking reliability while reducing feasible breadth.
7. **Is Q non-monotonic in t?** {non_monotonic}.
8. **Interior optimum?** The best search schedule was
   {int(best_search.M)}→{int(best_search.K)}@{int(best_search.checkpoint)}; whether
   this is interior is visible in plot A and the full search table.
9. **Top three from search.** {', '.join(top3)}.
10. **Internal-validation selection.** {selected_name}, validation mean IR
    {validation_row.final_IR_mean:.6f}.
11. **Frozen schedule.** Exactly `{selected_name}`; see `FROZEN_SCHEDULE.json`.
12. **Why selected?** It won under the pre-registered validation rule, including
    the 0.002 tie margin and fixed verifier/M/compute tie-breakers.
13. **Versus PSP on calibration.** Selected schedule all-120 mean IR {all_q:.6f};
    fixed PSP all-120 mean IR {psp_all_mean:.6f}; descriptive delta
    {all_q-psp_all_mean:+.6f}. This is development evidence, not Phase-2 confirmation.
14. **Breadth contribution.** Its oracle-pool quality O(M) is {all_oracle:.6f}.
15. **Selection loss.** Its mean selection regret is {all_regret:.6f}, verifying
    Q=O−R ({all_q:.6f}={all_oracle:.6f}−{all_regret:.6f}) up to floating point.

## Search and validation separation

- `search_results.csv` contains all 39 schedules on only the 80 search prompts.
- `validation_results.csv` contains only the three promoted schedules on only
  the 40 untouched internal-validation prompts.
- Reliability, heatmap, frontier, and broad mechanism sweeps use the search split.
- After freezing, only the selected schedule and fixed PSP are summarized over
  all 120 calibration prompts.
"""
    (EXP / "PHASE1_REPORT.md").write_text(report)


def main() -> None:
    replay_dir = EXP / "replay"
    replay_dir.mkdir(exist_ok=True)
    bank = load_bank()
    search_ids = sorted(bank.loc[bank.split.eq("search"), "prompt_id"].unique().tolist())
    validation_ids = sorted(bank.loc[bank.split.eq("validation"), "prompt_id"].unique().tolist())
    assert len(search_ids) == 80 and len(validation_ids) == 40
    schedules = valid_schedules()
    pd.DataFrame(schedules).to_csv(replay_dir / "all_schedules.csv", index=False)
    psp_search = replay_psp(bank, search_ids)
    psp_validation = replay_psp(bank, validation_ids)
    psp_search.assign(split="search").to_csv(replay_dir / "psp_search_per_prompt.csv", index=False)
    psp_validation.assign(split="validation").to_csv(replay_dir / "psp_validation_per_prompt.csv", index=False)

    search_summaries = []
    search_per_prompt = []
    for index, schedule in enumerate(schedules):
        replayed = replay_single(bank, schedule, search_ids)
        search_per_prompt.append(replayed)
        search_summaries.append(summarize(replayed, psp_search, schedule, BOOTSTRAP_SEED + index))
    search = pd.DataFrame(search_summaries).sort_values(
        ["final_IR_mean", "verifier_candidate_scores", "M", "logical_compute"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    search.insert(0, "search_rank", np.arange(1, len(search) + 1))
    search["checkpoint_schedule"] = (
        search.M.astype(str) + "→" + search.K.astype(str) + "@" + search.checkpoint.astype(str)
    )
    search.to_csv(replay_dir / "search_results.csv", index=False)
    pd.concat(search_per_prompt, ignore_index=True).to_csv(replay_dir / "search_per_prompt.csv", index=False)

    promoted = search.head(3).copy()
    validation_summaries = []
    validation_per_prompt = []
    for index, promoted_row in promoted.iterrows():
        schedule = {key: int(promoted_row[key]) for key in schedules[0]}
        replayed = replay_single(bank, schedule, validation_ids)
        validation_per_prompt.append(replayed)
        summary = summarize(replayed, psp_validation, schedule, BOOTSTRAP_SEED + 10_000 + index)
        summary["search_rank"] = int(promoted_row.search_rank)
        summary["checkpoint_schedule"] = promoted_row.checkpoint_schedule
        validation_summaries.append(summary)
    validation = pd.DataFrame(validation_summaries)
    best_mean = float(validation.final_IR_mean.max())
    eligible = validation[best_mean - validation.final_IR_mean < TIE_MARGIN].copy()
    winner = eligible.sort_values(
        ["verifier_candidate_scores", "M", "logical_compute", "final_IR_mean"],
        ascending=[True, True, False, False],
    ).iloc[0]
    validation["within_tie_margin"] = best_mean - validation.final_IR_mean < TIE_MARGIN
    validation["selected"] = (
        validation.checkpoint.eq(int(winner.checkpoint))
        & validation.K.eq(int(winner.K))
        & validation.M.eq(int(winner.M))
    )
    validation = validation.sort_values("search_rank").reset_index(drop=True)
    validation.to_csv(replay_dir / "validation_results.csv", index=False)
    pd.concat(validation_per_prompt, ignore_index=True).to_csv(
        replay_dir / "validation_top3_per_prompt.csv", index=False
    )

    selected_schedule = {
        key: int(winner[key])
        for key in ("checkpoint", "K", "M", "logical_compute", "unused_compute", "verifier_calls", "verifier_candidate_scores")
    }
    selected_all = replay_single(bank, selected_schedule, search_ids + validation_ids)
    selected_all.to_csv(replay_dir / "selected_all120_per_prompt.csv", index=False)
    reliability(bank, search_ids).to_csv(replay_dir / "ranking_reliability_search.csv", index=False)
    oracle_curve(bank, search_ids).to_csv(replay_dir / "oracle_curve_search.csv", index=False)

    bank_path = EXP / "prompts/calibration_bank.parquet"
    frozen = {
        **selected_schedule,
        "schedule_notation": f'{selected_schedule["M"]}→{selected_schedule["K"]}@{selected_schedule["checkpoint"]}',
        "model": "runwayml/stable-diffusion-v1-5",
        "steps": 64,
        "eta": 0.0,
        "guidance_scale": 7.5,
        "calibration_prompt_corpus": "ImageReward test_ir.json",
        "calibration_prompt_count": 120,
        "search_prompt_count": 80,
        "validation_prompt_count": 40,
        "selection_seed": 20260918,
        "split_seed": 20260918,
        "selection_rule": "best validation mean among search top-3; within 0.002 prefer fewer verifier scores, smaller M, higher used compute",
        "search_rank": int(winner.search_rank),
        "validation_mean_IR": float(winner.final_IR_mean),
        "validation_delta_vs_PSP": float(winner.delta_vs_psp),
        "calibration_bank_sha256": sha256(bank_path),
    }
    frozen_path = EXP / "FROZEN_SCHEDULE.json"
    rendered = json.dumps(frozen, indent=2) + "\n"
    if frozen_path.exists() and frozen_path.read_text() != rendered:
        raise RuntimeError("FROZEN_SCHEDULE.json already exists with different content")
    frozen_path.write_text(rendered)
    write_phase1_report(search, validation, frozen, psp_search, psp_validation, selected_all)
    print(json.dumps({
        "phase": "phase1_complete",
        "frozen_schedule": frozen,
        "top3_search": promoted.checkpoint_schedule.tolist(),
        "psp_search_mean_IR": float(psp_search.final_IR.mean()),
        "psp_validation_mean_IR": float(psp_validation.final_IR.mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
