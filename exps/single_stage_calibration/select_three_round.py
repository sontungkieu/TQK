#!/usr/bin/env python3
"""Three-discard extension of the 200-prompt calibration (PROTOCOL_200_EXT.md).

Reuses the frozen machinery of select_fold_schedule.py: same draws, same seed, same surface
shapes, same gates. The only new code is the three-discard space, its offline replay (pruning
drops candidates, so a survivor's step-64 sample is unchanged), and an exact aggregation of the
IRLS so the fit stays cheap with ~16.8k shapes.

Aggregation is exact: the covariates of a shape are prompt-independent, so the n_train binomial
cells of one shape collapse into a single Binomial(n_train*500, p) row for p_miss, and the Gamma
score equations collapse into sums of the miss count and of the regret. The ridge is added once
per fit in both formulations, so coefficients match the un-aggregated fit.

Outputs: replay200/three_round.json, replay200/schedule_ranking_3round.csv.
"""
from __future__ import annotations

import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from select_fold_schedule import (
    BOOTSTRAP, BOOTSTRAP_SEED, DRAWS, GATE_MIN_POSITIVE_FOLDS, GATE_MIN_SPEARMAN,
    LCB_QUANTILE, RIDGE, SUBSET_SEED, bank_digest, fit_logistic, load_bank,
    oracle_curve, policy_space, subset_draws,
)
from sweep_multi_stage import BUDGET, CHECKPOINTS, MAX_POOL, MIN_POOL, PSP, STEPS

EXP = Path(__file__).resolve().parent
OUT = EXP / "replay200"
FOLDS = 5
VERIFIER_NFE = 1.9


def three_discard_configs() -> list[dict]:
    rows = []
    for t1, t2, t3 in itertools.combinations(CHECKPOINTS, 3):
        for k1 in range(2, MAX_POOL + 1):
            for k2 in range(1, k1):
                for k3 in range(1, k2):
                    tail = k1 * (t2 - t1) + k2 * (t3 - t2) + k3 * (STEPS - t3)
                    n = min((BUDGET - tail) // t1, MAX_POOL)
                    if n < k1 or n < MIN_POOL:
                        continue
                    unet = n * t1 + tail
                    scores = n + k1 + k2 + k3
                    rows.append({
                        "rounds": 3, "n": int(n), "t1": t1, "k1": k1, "t2": t2, "k2": k2,
                        "t3": t3, "k3": k3, "unet": int(unet), "unused": int(BUDGET - unet),
                        "scores": int(scores), "equiv": unet + VERIFIER_NFE * scores,
                        "prune_percent": "%d/%d/%d" % (round(100 * t1 / STEPS), round(100 * t2 / STEPS), round(100 * t3 / STEPS)),
                    })
    return rows


def features(policy: dict) -> tuple[list[float], tuple[int, ...]]:
    n = policy["n"]
    if policy["rounds"] == 3:
        return ([policy["t1"] / 64.0, policy["t2"] / 64.0, policy["t3"] / 64.0, math.log2(n),
                 policy["k1"] / n, policy["k2"] / policy["k1"], policy["k3"] / policy["k2"]], (0, 1, 2))
    stage2 = policy["k2"] / policy["k1"] if policy["rounds"] == 2 else 0.0
    stage3 = 0.0
    return ([policy["t1"] / 64.0, (policy["t2"] if policy["rounds"] == 2 else policy["t1"]) / 64.0, 0.0,
             math.log2(n), policy["k1"] / n, stage2, stage3], (0, 1, 2))


def measure_three(bank: dict, policies: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per (prompt, shape) miss count and total regret, with the shared subset draws."""
    prompt_ids = sorted(bank)
    n_prompts, n_policies = len(prompt_ids), len(policies)
    miss = np.zeros((n_prompts, n_policies), dtype=np.int32)
    regret = np.zeros((n_prompts, n_policies))
    prefixes: dict[tuple[int, int, int, int, int], list[tuple[int, int, int]]] = {}
    for policy in policies:
        key = (policy["n"], policy["t1"], policy["k1"], policy["t2"], policy["k2"])
        prefixes.setdefault(key, []).append((policy["t3"], policy["k3"], policy["index"]))
    by_m: dict[int, list] = {}
    for key in prefixes:
        by_m.setdefault(key[0], []).append(key)
    rows = np.arange(DRAWS)
    for pi, pid in enumerate(prompt_ids):
        final, x = bank[pid]["final"], bank[pid]["x"]
        for m, keys in by_m.items():
            draw = subset_draws(pid, m)
            values = final[draw]
            best = values.max(axis=1)
            for (_, t1, k1, t2, k2) in keys:
                first = np.argpartition(-x[t1][draw], k1 - 1, axis=1)[:, :k1]
                inner = np.take_along_axis(draw, first, axis=1)
                second = np.argpartition(-x[t2][inner], k2 - 1, axis=1)[:, :k2]
                kept2 = np.take_along_axis(first, second, axis=1)
                for t3, k3, index in prefixes[(m, t1, k1, t2, k2)]:
                    third = np.argpartition(-x[t3][kept2], k3 - 1, axis=1)[:, :k3]
                    kept3 = np.take_along_axis(kept2, third, axis=1)
                    vals3 = values[rows[:, None], kept3]
                    winner = kept3[rows, vals3.argmax(axis=1)]
                    loss = np.clip(best - values[rows, winner], 0.0, None)
                    regret[pi, index] = loss.sum()
                    miss[pi, index] = int((loss > 1e-9).sum())
    return miss, regret


def fit_gamma_aggregated(design, miss_rows, regret_rows, ridge=RIDGE, iters=60):
    """Gamma log-link score equations collapsed over prompts (exact, see module docstring)."""
    m = miss_rows.astype(float)
    r = regret_rows.astype(float)
    total_miss = m.sum(axis=0)
    total_regret = r.sum(axis=0)
    mask = m > 0
    weighted_log = np.where(mask, m * np.log(np.clip(np.divide(r, m, out=np.ones_like(r), where=mask), 1e-12, None)), 0.0).sum(axis=0)
    penalty = np.full(design.shape[1], ridge)
    penalty[0] = 0.0
    beta = np.zeros(design.shape[1])
    beta[0] = math.log(max(float(total_regret.sum() / max(total_miss.sum(), 1e-9)), 1e-6))

    def deviance(b):
        mu = np.exp(np.clip(design @ b, -20, 20))
        return 2.0 * float((-weighted_log + total_miss * np.log(mu) + (total_regret - total_miss * mu) / mu).sum())

    current = deviance(beta)
    for _ in range(iters):
        eta = np.clip(design @ beta, -20, 20)
        mu = np.exp(eta)
        working = eta * total_miss + (total_regret - total_miss * mu) / mu
        matrix = design.T @ (total_miss[:, None] * design) + np.diag(penalty)
        step = np.linalg.solve(matrix + np.eye(matrix.shape[0]) * 1e-10, design.T @ working) - beta
        for shrink in (1.0, 0.5, 0.25, 0.125):
            trial = beta + shrink * step
            value = deviance(trial)
            if value <= current + 1e-12:
                beta, current = trial, value
                break
        else:
            break
    return beta


FIELDS = ("rounds", "n", "t1", "k1", "t2", "k2", "t3", "k3", "unet", "scores", "equiv", "prune_percent")


def fields(policy: dict) -> dict:
    """Shape description that is safe for one/two-discard policies (no t3/k3/prune_percent)."""
    return {k: policy.get(k) for k in FIELDS}


def main() -> None:
    bank = load_bank()
    prompt_ids = sorted(bank)
    folds = np.array([bank[p]["fold"] for p in prompt_ids])
    oracle = np.full((len(bank), 26), np.nan)
    for pi, pid in enumerate(prompt_ids):
        for m, value in oracle_curve(bank[pid]["final"]).items():
            oracle[pi, m] = value

    base = policy_space()
    print(f"one/two-discard shapes: {len(base)}")
    from select_fold_schedule import measure as measure12
    data12 = measure12(bank, base)
    three = three_discard_configs()
    for i, policy in enumerate(three):
        policy["index"] = i
    print(f"three-discard shapes: {len(three)} | measuring ...")
    miss3, regret3 = measure_three(bank, three)

    psp_loss = data12["psp_regret"] / DRAWS
    psp_q = oracle[:, PSP["n"]] - psp_loss
    loss12 = data12["regret"] / DRAWS
    q12 = oracle[:, [p["n"] for p in base]] - loss12
    delta12 = q12 - psp_q[:, None]
    loss3 = regret3 / DRAWS
    q3 = oracle[:, [p["n"] for p in three]] - loss3
    delta3 = q3 - psp_q[:, None]
    n3 = len(three)
    print(f"invariant 3-discard: {np.max(np.abs(regret3.sum(0) / (DRAWS * len(bank)) - (miss3.sum(0) / (DRAWS * len(bank))) * np.divide(regret3.sum(0), miss3.sum(0), out=np.zeros(n3), where=miss3.sum(0) > 0))):.2e}")

    design3 = np.column_stack([np.ones(n3), np.array([features(p)[0] for p in three])])
    constrained = features(three[0])[1]
    mean_q3 = delta3.mean(axis=0)
    order3 = np.argsort(-mean_q3)

    # out-of-fold gate for the three-discard family
    fold_reports = []
    for fold in range(FOLDS):
        train, test = folds != fold, folds == fold
        n_train = int(train.sum())
        beta, _ = fit_logistic(design3, np.full(n3, n_train * DRAWS), miss3[train].sum(axis=0), constrained)
        gamma = fit_gamma_aggregated(design3, miss3[train], regret3[train])
        o_train = np.full(26, np.nan)
        for m in range(2, 26):
            o_train[m] = float(np.nanmean(oracle[train, m]))
        psp_term = o_train[PSP["n"]] - float(np.mean(data12["psp_miss"][train] / DRAWS * np.divide(data12["psp_regret"][train], data12["psp_miss"][train], out=np.zeros(n_train), where=data12["psp_miss"][train] > 0)))
        predicted = o_train[np.array([p["n"] for p in three])] - (1.0 / (1.0 + np.exp(-np.clip(design3 @ beta, -30, 30)))) * np.clip(np.exp(np.clip(design3 @ gamma, -20, 20)), 1e-6, None) - psp_term
        actual = delta3[test].mean(axis=0)
        chosen = int(np.argmax(predicted))
        rank = int(np.where(np.argsort(-actual) == chosen)[0][0]) + 1
        rho = float(spearmanr(predicted, actual).statistic)
        fold_reports.append({
            "fold": fold, "selected_policy": {k: three[chosen][k] for k in ("n", "t1", "k1", "t2", "k2", "t3", "k3", "unet", "scores", "equiv", "prune_percent")},
            "held_out_delta_ir": float(actual[chosen]), "held_out_rank": rank,
            "spearman_predicted_vs_actual": rho,
        })
        print(f"fold {fold}: held-out {fold_reports[-1]['held_out_delta_ir']:+.6f} rank {rank:>5} rho {rho:+.3f} | {fold_reports[-1]['selected_policy']}")
    mean_held = float(np.mean([f["held_out_delta_ir"] for f in fold_reports]))
    positive = int(sum(1 for f in fold_reports if f["held_out_delta_ir"] > 0))
    mean_rho = float(np.nanmean([f["spearman_predicted_vs_actual"] for f in fold_reports]))
    gate = {"mean_held_out_delta_ir": mean_held, "positive_folds": positive, "mean_spearman": mean_rho,
            "pass": bool(mean_held > 0 and positive >= GATE_MIN_POSITIVE_FOLDS and mean_rho >= GATE_MIN_SPEARMAN)}

    # measured (post-hoc, clearly labelled) ranking over the union of all families
    measured = sorted([(float(delta3[:, j].mean()), three[j]) for j in range(n3)], key=lambda x: -x[0])
    all_measured = sorted(
        [(float(delta12[:, j].mean()), base[j]) for j in range(len(base))] +
        [(float(delta3[:, j].mean()), three[j]) for j in range(n3)], key=lambda x: -x[0])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, len(bank), size=(BOOTSTRAP, len(bank)))
    # Resample multiplicities instead of fancy-indexing the (prompts, shapes) matrix: the direct
    # form would materialise (BOOTSTRAP, prompts, shapes) ~ 27 GB for the three-discard family.
    weights = np.zeros((BOOTSTRAP, len(bank)))
    np.add.at(weights, (np.arange(BOOTSTRAP)[:, None], picks), 1.0)
    weights /= len(bank)
    lcb3 = np.percentile(weights @ delta3, LCB_QUANTILE * 100, axis=0)
    lcb12 = np.percentile(weights @ delta12, LCB_QUANTILE * 100, axis=0)
    best3 = int(np.argmax(lcb3))
    union_lcb = sorted(
        [(float(lcb12[j]), "one/two", base[j]) for j in range(len(base))] +
        [(float(lcb3[j]), "three", three[j]) for j in range(n3)], key=lambda x: -x[0])

    feasible = [j for j in range(n3) if three[j]["equiv"] <= BUDGET]
    best_equiv = max(feasible, key=lambda j: delta3[:, j].mean()) if feasible else None

    summary = {
        "bank_sha256": bank_digest(), "prompts": len(bank),
        "shapes_one_two": len(base), "shapes_three": n3,
        "draws_per_prompt_per_shape": DRAWS, "subset_seed": SUBSET_SEED,
        "bootstrap": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "lcb_quantile": LCB_QUANTILE,
        "ridge": RIDGE, "numpy": np.__version__, "python": platform.python_version(),
        "three_discard_feasible_unet256": n3,
        "three_discard_feasible_equiv256": len(feasible),
        "baseline_psp": {"policy": PSP, "mean_q_ir": float(psp_q.mean())},
        "out_of_fold_gate_three": {**gate, "required_positive_folds": GATE_MIN_POSITIVE_FOLDS, "required_spearman": GATE_MIN_SPEARMAN, "folds": fold_reports},
        "measured_union_top10": [
            {"delta_ir": value, **fields(p)}
            for value, p in all_measured[:10]
        ],
        "measured_three_top10": [
            {"delta_ir": value, **fields(p)}
            for value, p in measured[:10]
        ],
        "freeze_lcb80_union_top10": [
            {"lcb80_delta_ir": value, "family": fam, **fields(p)}
            for value, fam, p in union_lcb[:10]
        ],
        "freeze_lcb80_three_best": {"lcb80_delta_ir": float(lcb3[best3]), "mean_delta_ir": float(delta3[:, best3].mean()),
                                    **{k: three[best3][k] for k in ("n", "t1", "k1", "t2", "k2", "t3", "k3", "unet", "scores", "equiv")}},
        "equiv256_best_three": None if best_equiv is None else {
            "mean_delta_ir": float(delta3[:, best_equiv].mean()),
            **{k: three[best_equiv][k] for k in ("n", "t1", "k1", "t2", "k2", "t3", "k3", "unet", "scores", "equiv")}},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "three_round.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "schedule_ranking_3round.csv").open("w") as handle:
        handle.write("index,rounds,n,t1,k1,t2,k2,t3,k3,unet,scores,equiv,mean_delta_ir,lcb80_delta_ir\n")
        for j in order3:
            p = three[j]
            handle.write("%d,3,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.6f,%.6f\n" % (
                j, p["n"], p["t1"], p["k1"], p["t2"], p["k2"], p["t3"], p["k3"], p["unet"], p["scores"], p["equiv"],
                delta3[:, j].mean(), lcb3[j]))
    print(json.dumps({"out_of_fold_gate_three": gate,
                      "measured_union_top3": summary["measured_union_top10"][:3],
                      "freeze_lcb80_union_top3": union_lcb[:3],
                      "equiv256_best_three": summary["equiv256_best_three"]}, indent=2, default=str))
    print(f"wrote {OUT}/three_round.json")


if __name__ == "__main__":
    main()
