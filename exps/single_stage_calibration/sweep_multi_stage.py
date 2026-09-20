#!/usr/bin/env python3
"""Two-round seed-pruning sweep over the phase-1 calibration bank (offline replay).

The published search space prunes exactly once before the final selection
(`M -> K @ t`), and PSP prunes twice (`8 -> 4 @ 16 -> 2 @ 32 -> 1 @ 64`). This sweep asks
the question in between: with the same 256 logical UNet evaluations per prompt, which
two-round schedule - initial pool N, first prune to k1 at t1, second prune to k2 at t2,
final winner at step 64 - maximises the final ImageReward of the chosen image?

Why replay is exact: pruning only drops candidates. The survivors keep their latent and
their per-candidate generator (the repository's second correctness constraint), so a
candidate's step-64 result is the same whether or not its neighbours were pruned. The bank
already contains every candidate's ImageReward at each checkpoint and at the final step, so
the whole space can be scored without a GPU.

Discipline: the 80 search prompts choose the schedule; the 40 internal-validation prompts
are only read afterwards, exactly as in the published calibration.

Budget follows the repository: `unet = N*t1 + k1*(t2-t1) + k2*(64-t2) <= 256`, with the
initial pool derived as the largest feasible value. Verifier work is reported separately
(`N + k1 + k2` scores in 3 calls) and also converted to NFE equivalents with the factor
measured on 2x T4 (288 ms/score vs 151 ms/UNet-eval, so 1 score ~ 1.9 UNet evaluations).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "exps/single_stage_calibration"
BANK = CAL / "bank_raw"
OUT = CAL / "replay/multi_stage_sweep_2round.csv"

STEPS = 64
BUDGET = 256
MAX_POOL = 25
MIN_POOL = 2
VERIFIER_NFE_EQUIV = 1.9
CHECKPOINTS = (8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 28, 32)
PSP = {"n": 8, "t1": 16, "k1": 4, "t2": 32, "k2": 2}
FROZEN = {"n": 9, "t1": 17, "k1": 2}


def load_bank() -> list[dict]:
    """Every prompt with its candidates' ImageReward at each checkpoint and at step 64."""
    prompts = []
    skipped = []
    for path in sorted(BANK.rglob("*.json")):
        record = json.loads(path.read_text())
        # bank_raw also holds summary/companion records; only per-prompt trajectory files
        # carry the candidate list.
        if not isinstance(record, dict) or "candidates" not in record:
            skipped.append(path.name)
            continue
        candidates = record["candidates"]
        prompts.append({
            "prompt_id": int(record["prompt_id"]),
            "split": str(record["split"]),
            "ir": {
                step: [float(candidate[f"IR_t{step}"]) for candidate in candidates]
                for step in CHECKPOINTS
            },
            "final": [float(candidate["IR_final"]) for candidate in candidates],
        })
    if skipped:
        print(f"skipped non-trajectory records: {skipped}")
    return prompts


def two_round_selection(prompt: dict, n: int, t1: int, k1: int, t2: int, k2: int) -> int:
    """Index of the winner after pruning to k1 at t1 and k2 at t2, finalised at step 64."""
    first = sorted(range(n), key=lambda i: -prompt["ir"][t1][i])[:k1]
    second = sorted(first, key=lambda i: -prompt["ir"][t2][i])[:k2]
    return max(second, key=lambda i: prompt["final"][i])


def one_round_selection(prompt: dict, n: int, t: int, k: int) -> int:
    keep = sorted(range(n), key=lambda i: -prompt["ir"][t][i])[:k]
    return max(keep, key=lambda i: prompt["final"][i])


def mean_final(prompts: list[dict], split: str, pick) -> float:
    values = [prompt["final"][pick(prompt)] for prompt in prompts if prompt["split"] == split]
    return statistics.fmean(values)


def two_round_configs() -> list[dict]:
    rows = []
    for t1 in CHECKPOINTS:
        for t2 in CHECKPOINTS:
            if t2 <= t1:
                continue
            for k1 in range(2, MAX_POOL + 1):
                for k2 in range(1, k1):
                    tail = k1 * (t2 - t1) + k2 * (STEPS - t2)
                    n = min((BUDGET - tail) // t1, MAX_POOL)
                    if n < k1 or n < MIN_POOL:
                        continue
                    unet = n * t1 + tail
                    scores = n + k1 + k2
                    rows.append({
                        "rounds": 2, "n": n, "t1": t1, "k1": k1, "t2": t2, "k2": k2,
                        "unet": unet, "unused": BUDGET - unet,
                        "verifier_calls": 3, "scores": scores,
                        "equiv": unet + VERIFIER_NFE_EQUIV * scores,
                        "prune_percent": f"{round(100 * t1 / STEPS)}/{round(100 * t2 / STEPS)}",
                    })
    return rows


def one_round_configs() -> list[dict]:
    rows = []
    for t in CHECKPOINTS:
        for k in (1, 2, 3):
            n = min((BUDGET - k * (STEPS - t)) // t, MAX_POOL)
            if n < k or n < MIN_POOL:
                continue
            unet = n * t + k * (STEPS - t)
            scores = n + k
            rows.append({
                "rounds": 1, "n": n, "t1": t, "k1": k, "t2": STEPS, "k2": 1,
                "unet": unet, "unused": BUDGET - unet,
                "verifier_calls": 2, "scores": scores,
                "equiv": unet + VERIFIER_NFE_EQUIV * scores,
                "prune_percent": f"{round(100 * t / STEPS)}",
            })
    return rows


def evaluate(prompts: list[dict], rows: list[dict]) -> list[dict]:
    for row in rows:
        if row["rounds"] == 2:
            pick = (lambda p, r=row: two_round_selection(p, r["n"], r["t1"], r["k1"], r["t2"], r["k2"]))
        else:
            pick = (lambda p, r=row: one_round_selection(p, r["n"], r["t1"], r["k1"]))
        row["search_IR"] = mean_final(prompts, "search", pick)
        row["validation_IR"] = mean_final(prompts, "validation", pick)
        row["all120_IR"] = mean_final(prompts, "search", pick) * 0.0 + statistics.fmean(
            [p["final"][pick(p)] for p in prompts]
        )
        row["oracle_n"] = statistics.fmean(
            max(prompt["final"][: row["n"]]) for prompt in prompts
        )
        row["regret"] = row["oracle_n"] - row["all120_IR"]
    return rows


def main() -> None:
    prompts = load_bank()
    splits = {split: sum(1 for p in prompts if p["split"] == split) for split in ("search", "validation")}
    print(f"bank: {len(prompts)} prompts {splits} | checkpoints {CHECKPOINTS}")
    print(f"oracle(all 25) = {statistics.fmean(max(p['final']) for p in prompts):.6f}")

    two = evaluate(prompts, two_round_configs())
    one = evaluate(prompts, one_round_configs())
    feasible_equiv = [row for row in two if row["equiv"] <= BUDGET]

    def line(row: dict) -> str:
        return (
            f"{row['n']:2d} -> {row['k1']:2d}@{row['t1']:<2d} -> {row['k2']:2d}@{row['t2']:<2d} -> 1@64 "
            f"| prune {row['prune_percent']:>5}% | unet {row['unet']:3d} scores {row['scores']:2d} "
            f"equiv {row['equiv']:6.1f} | search {row['search_IR']:.6f} | valid {row['validation_IR']:.6f} "
            f"| regret {row['regret']:.4f}"
        )

    print("\n=== two-round schedules ranked by the 80 search prompts ===")
    for row in sorted(two, key=lambda r: -r["search_IR"])[:15]:
        print("  " + line(row))

    print("\n=== two-round schedules that also fit 256 NFE-equivalents ===")
    for row in sorted(feasible_equiv, key=lambda r: -r["search_IR"])[:10]:
        print("  " + line(row))

    print("\n=== one-round reference (repository grid: t x K in {1,2,3}) ===")
    for row in sorted(one, key=lambda r: -r["search_IR"])[:5]:
        print("  " + line(row))

    best2 = max(two, key=lambda r: r["search_IR"])
    best1 = max(one, key=lambda r: r["search_IR"])
    psp_pick = lambda p: two_round_selection(p, PSP["n"], PSP["t1"], PSP["k1"], PSP["t2"], PSP["k2"])
    frozen_pick = lambda p: one_round_selection(p, FROZEN["n"], FROZEN["t1"], FROZEN["k1"])
    print("\n=== baselines on the same data ===")
    print(f"  PSP 8->4@16->2@32->1@64 : search {mean_final(prompts, 'search', psp_pick):.6f} "
          f"| valid {mean_final(prompts, 'validation', psp_pick):.6f}")
    print(f"  frozen 9->2@17->1@64    : search {mean_final(prompts, 'search', frozen_pick):.6f} "
          f"| valid {mean_final(prompts, 'validation', frozen_pick):.6f}")
    print(f"\nbest two-round by search ({line(best2).split('|')[0].strip()})")
    print(f"   -> validation {best2['validation_IR']:.6f} | best one-round validation {best1['validation_IR']:.6f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        keys = ["rounds", "n", "t1", "k1", "t2", "k2", "unet", "unused", "verifier_calls", "scores",
                "equiv", "prune_percent", "search_IR", "validation_IR", "all120_IR", "oracle_n", "regret"]
        handle.write(",".join(keys) + "\n")
        for row in sorted(two + one, key=lambda r: -r["search_IR"]):
            handle.write(",".join(str(row[key]) for key in keys) + "\n")
    print(f"\nwrote {OUT} ({len(two) + len(one)} schedules)")


if __name__ == "__main__":
    main()
