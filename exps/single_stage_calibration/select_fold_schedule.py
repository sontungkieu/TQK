#!/usr/bin/env python3
"""Five-fold out-of-fold selection and the freeze gate for the 200-prompt bank.

Everything in this file is pinned *before* any bank200 ImageReward value was inspected
(PROTOCOL_200.md sections 5-9). A re-run must reproduce every number here, so all draws are
seeded and every estimator is deterministic.

Pinned constants (must not be tuned after looking at a reward):

  DRAWS = 500 subsets per prompt and pool size, SUBSET_SEED = 20260927
  BOOTSTRAP = 1000 prompt resamples, BOOTSTRAP_SEED = 20260928, LCB at the 20th percentile
  RIDGE = 1.0 on standardized features (intercept unpenalized)
  p_miss: binomial logistic IRLS on the 500 aggregated draws; the coefficient of q is
          constrained <= 0, because the miss rate must not grow with denoising depth
  ell:    Gamma GLM with log link, same features, frequency weights = number of misses
  Q(M)  = O(M) - p_miss * ell, with O(M) estimated on the training folds only
  gate:   mean held-out Delta IR > 0 AND >= 4/5 folds positive AND Spearman >= 0.8
  freeze: argmax_policy LCB_80(Delta IR) over the 1000 bootstrap resamples, must be > 0

The surfaces are functions of *policy* attributes only, which is what the protocol declares
(p_miss and ell in q, log M and K/M). Every prompt contributes one observation per policy, so a
prompt resample or fold split changes the fitted coefficients but never the covariate design.
The two families share the estimator shape but not the coefficients:

  one round: x = [q1, log2(M), K/M]                  q1 = t1/64
  two rounds: x = [q1, q2, log2(N), k1/N, k2/k1]     q2 = t2/64

Outputs (replay200/): fold_selection.json, schedule_ranking.csv.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sweep_multi_stage import PSP, CHECKPOINTS, one_round_configs, two_round_configs

EXP = Path(__file__).resolve().parent
OUT = EXP / "replay200"

DRAWS = 500
SUBSET_SEED = 20260927
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260928
LCB_QUANTILE = 0.20
RIDGE = 1.0
FOLDS = 5
GATE_MIN_POSITIVE_FOLDS = 4
GATE_MIN_SPEARMAN = 0.8


# --------------------------------------------------------------------------- data


def load_bank() -> dict[int, dict]:
    bank = {}
    for path in sorted((EXP / "bank200/gpu0").glob("*.json")):
        if not path.stem.isdigit():
            continue
        payload = json.loads(path.read_text())
        bank[int(payload["prompt_id"])] = {
            "fold": int(payload["fold"]),
            "final": np.array([float(c["IR_final"]) for c in payload["candidates"]]),
            "x": {t: np.array([float(c[f"IR_t{t}"]) for c in payload["candidates"]]) for t in CHECKPOINTS},
        }
    assert len(bank) == 200, len(bank)
    return bank


def bank_digest() -> str:
    h = hashlib.sha256()
    for path in sorted((EXP / "bank200/gpu0").glob("*.json")):
        if not path.stem.isdigit():
            continue
        h.update(path.name.encode())
        h.update(b":")
        h.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        h.update(b"\n")
    return h.hexdigest()


def oracle_curve(final: np.ndarray) -> dict[int, float]:
    """Exact E[max of a uniformly random M-subset] = sum_{j>=M} Y_(j) C(j-1,M-1)/C(25,M)."""
    ordered = np.sort(final)
    out = {}
    for m in range(2, 26):
        denom = math.comb(25, m)
        out[m] = float(sum(ordered[j - 1] * math.comb(j - 1, m - 1) / denom for j in range(m, 26)))
    return out


# --------------------------------------------------------------------------- policies


def policy_space() -> list[dict]:
    rows = [dict(row) for row in one_round_configs() + two_round_configs()]
    for i, row in enumerate(rows):
        row["index"] = i
    return rows


def family_features(policy: dict) -> tuple[str, list[float], tuple[int, ...]]:
    m = policy["n"]
    if policy["rounds"] == 1:
        return "one", [policy["t1"] / 64.0, math.log2(m), policy["k1"] / m], (0,)
    return (
        "two",
        [policy["t1"] / 64.0, policy["t2"] / 64.0, math.log2(m), policy["k1"] / m, policy["k2"] / policy["k1"]],
        (0, 1),
    )


# --------------------------------------------------------------------------- measurement


def subset_draws(prompt_id: int, m: int) -> np.ndarray:
    """500 uniform M-subsets of the 25 candidates, identical for every policy with this M."""
    rng = np.random.default_rng([SUBSET_SEED, prompt_id, m])
    return np.argsort(rng.random((DRAWS, 25)), axis=1)[:, :m]


def survivor_positions(policy: dict, draw: np.ndarray, x: dict[int, np.ndarray]) -> np.ndarray:
    k1 = policy["k1"]
    if policy["rounds"] == 1:
        return np.argpartition(-x[policy["t1"]][draw], k1 - 1, axis=1)[:, :k1]
    first = np.argpartition(-x[policy["t1"]][draw], k1 - 1, axis=1)[:, :k1]
    inner = np.take_along_axis(draw, first, axis=1)
    second = np.argpartition(-x[policy["t2"]][inner], policy["k2"] - 1, axis=1)[:, : policy["k2"]]
    return np.take_along_axis(first, second, axis=1)


def measure(bank: dict[int, dict], policies: list[dict]) -> dict:
    """Per (prompt, policy) miss count and total regret over the shared subset draws."""
    prompt_ids = sorted(bank)
    n_prompts, n_policies = len(prompt_ids), len(policies)
    psp = dict(PSP)
    psp.update({"rounds": 2, "index": -1})
    by_m: dict[int, list[dict]] = {}
    for policy in policies:
        by_m.setdefault(policy["n"], []).append(policy)
    by_m.setdefault(psp["n"], []).append(psp)

    miss = np.zeros((n_prompts, n_policies), dtype=np.int32)
    regret = np.zeros((n_prompts, n_policies))
    psp_regret = np.zeros(n_prompts)
    psp_miss = np.zeros(n_prompts, dtype=np.int32)
    oracle = np.full((n_prompts, 26), np.nan)
    rows = np.arange(DRAWS)

    for pi, pid in enumerate(prompt_ids):
        final, x = bank[pid]["final"], bank[pid]["x"]
        curve = oracle_curve(final)
        for m, value in curve.items():
            oracle[pi, m] = value
        for m, group in by_m.items():
            draw = subset_draws(pid, m)
            values = final[draw]
            best = values.max(axis=1)
            for policy in group:
                pos = survivor_positions(policy, draw, x)
                kept = values[rows[:, None], pos]
                winner = pos[rows, kept.argmax(axis=1)]
                loss = np.clip(best - values[rows, winner], 0.0, None)
                if policy["index"] < 0:
                    psp_regret[pi] = loss.sum()
                    psp_miss[pi] = int((loss > 1e-9).sum())
                else:
                    j = policy["index"]
                    regret[pi, j] = loss.sum()
                    miss[pi, j] = int((loss > 1e-9).sum())
    return {
        "prompt_ids": prompt_ids,
        "oracle": oracle,
        "miss": miss,
        "regret": regret,
        "psp_regret": psp_regret,
        "psp_miss": psp_miss,
    }


# --------------------------------------------------------------------------- estimators


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(a + np.eye(a.shape[0]) * 1e-10, b)


def fit_logistic(design: np.ndarray, trials: np.ndarray, successes: np.ndarray,
                 constrained: tuple[int, ...], ridge: float = RIDGE, iters: int = 60):
    """Binomial logistic IRLS with ridge (intercept unpenalized). A constrained feature that
    wants a positive coefficient is dropped and the fit restarted, which enforces p_miss
    non-increasing in the constrained coordinates."""
    d = design.shape[1] - 1
    active = list(range(1, d + 1))
    coeff = np.zeros(design.shape[1])
    for _ in range(d + 1):
        columns = [0] + active
        sub = design[:, columns]
        beta = np.zeros(sub.shape[1])
        penalty = np.full(sub.shape[1], ridge)
        penalty[0] = 0.0
        for _ in range(iters):
            eta = np.clip(sub @ beta, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            w = np.clip(trials * p * (1.0 - p), 1e-9, None)
            z = eta + (successes - trials * p) / w
            beta = _solve(sub.T @ (w[:, None] * sub) + np.diag(penalty), sub.T @ (w * z))
        full = np.zeros(design.shape[1])
        full[columns] = beta
        coeff = full
        violated = [i for i in constrained if (i + 1) in active and coeff[i + 1] > 0]
        if not violated:
            break
        active = [c for c in active if c not in [i + 1 for i in violated]]
    return coeff, active


def fit_gamma_log(design: np.ndarray, y: np.ndarray, weights: np.ndarray,
                  ridge: float = RIDGE, iters: int = 60) -> np.ndarray:
    """Gamma GLM, log link, frequency weights; IRLS with step halving on the deviance."""
    penalty = np.full(design.shape[1], ridge)
    penalty[0] = 0.0
    beta = np.zeros(design.shape[1])
    beta[0] = math.log(max(float(np.average(y, weights=weights)), 1e-6))

    def deviance(b):
        mu = np.exp(np.clip(design @ b, -20, 20))
        return 2.0 * float(np.sum(weights * (-np.log(np.clip(y / mu, 1e-12, None)) + (y - mu) / mu)))

    current = deviance(beta)
    for _ in range(iters):
        eta = np.clip(design @ beta, -20, 20)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        step = _solve(design.T @ (weights[:, None] * design) + np.diag(penalty),
                      design.T @ (weights * z)) - beta
        for shrink in (1.0, 0.5, 0.25, 0.125):
            trial = beta + shrink * step
            value = deviance(trial)
            if value <= current + 1e-12:
                beta, current = trial, value
                break
        else:
            break
    return beta


def fit_surface(policy_x: np.ndarray, constrained: tuple[int, ...], miss_rows: np.ndarray,
                regret_rows: np.ndarray) -> dict:
    """Fit p_miss and ell for one family on a set of prompts.

    miss_rows/regret_rows are (n_prompts, n_policies) integers over the shared draws.
    """
    mu, sd = policy_x.mean(axis=0), policy_x.std(axis=0)
    sd[sd == 0] = 1.0
    z = (policy_x - mu) / sd
    design = np.column_stack([np.ones(len(z)), z])
    n_prompts, n_policies = miss_rows.shape
    tiled = np.tile(design, (n_prompts, 1))
    flat_miss = miss_rows.ravel()
    flat_regret = regret_rows.ravel()
    beta, active = fit_logistic(tiled, np.full(tiled.shape[0], DRAWS), flat_miss, constrained)
    mask = flat_miss > 0
    ell_y = np.divide(flat_regret, flat_miss, out=np.zeros_like(flat_regret, dtype=float), where=mask)
    gamma = fit_gamma_log(tiled[mask], ell_y[mask], flat_miss[mask].astype(float))
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
    e_hat = np.clip(np.exp(np.clip(design @ gamma, -20, 20)), 1e-6, None)
    return {"regret_hat": p_hat * e_hat, "p_hat": p_hat, "ell_hat": e_hat, "active": active}


# --------------------------------------------------------------------------- main


def main() -> None:
    bank = load_bank()
    policies = policy_space()
    data = measure(bank, policies)
    oracle, loss = data["oracle"], data["regret"] / DRAWS
    psp_loss = data["psp_regret"] / DRAWS
    folds = np.array([bank[pid]["fold"] for pid in data["prompt_ids"]])
    draws_total = DRAWS * len(data["prompt_ids"])
    print(f"bank {len(bank)} prompts | policies {len(policies)} | {draws_total} draws per policy")

    # invariant (PROTOCOL_200 sections 3 and 7): direct mean regret == p_miss * ell
    total_miss = data["miss"].sum(axis=0)
    pooled_miss = total_miss / draws_total
    pooled_ell = np.divide(data["regret"].sum(axis=0), total_miss, out=np.zeros(len(policies)), where=total_miss > 0)
    direct = data["regret"].sum(axis=0) / draws_total
    invariant = float(np.max(np.abs(direct - pooled_miss * pooled_ell)))
    print(f"max |direct regret - p_miss*ell| = {invariant:.3e}")

    family: dict[str, dict] = {}
    for name in ("one", "two"):
        idx = np.array([p["index"] for p in policies if family_features(p)[0] == name])
        family[name] = {
            "idx": idx,
            "x": np.array([family_features(policies[j])[1] for j in idx]),
            "constrained": family_features(policies[idx[0]])[2],
        }
        print(f"family {name}: {len(idx)} policies, {family[name]['x'].shape[1]} features")

    psp_pred_prompt = data["psp_miss"] / DRAWS * np.divide(data["psp_regret"], data["psp_miss"],
                                                           out=np.zeros(len(bank)), where=data["psp_miss"] > 0)
    # Q(policy) = O_p(M) - R_p(policy); Delta IR is the paired difference against PSP at M=8.
    pool_sizes = np.array([p["n"] for p in policies])
    q_value = oracle[:, pool_sizes] - loss
    psp_q = oracle[:, PSP["n"]] - psp_loss

    def predict_surfaces(train_idx: np.ndarray) -> np.ndarray:
        """Predicted mean Delta IR for every policy, fitted on the given prompts."""
        o_train = np.full(26, np.nan)
        for m in range(2, 26):
            o_train[m] = float(np.nanmean(oracle[train_idx, m]))
        psp_term = o_train[PSP["n"]] - float(np.mean(psp_pred_prompt[train_idx]))
        out = np.full(len(policies), np.nan)
        for name in ("one", "two"):
            idx = family[name]["idx"]
            fit = fit_surface(family[name]["x"], family[name]["constrained"],
                              data["miss"][train_idx][:, idx],
                              data["regret"][train_idx][:, idx])
            pools = np.array([policies[j]["n"] for j in idx])
            out[idx] = o_train[pools] - fit["regret_hat"] - psp_term
        return out

    fold_reports = []
    for fold in range(FOLDS):
        train = np.where(folds != fold)[0]
        test = np.where(folds == fold)[0]
        predicted = predict_surfaces(train)
        actual = (q_value[test] - psp_q[test][:, None]).mean(axis=0)
        chosen = int(np.nanargmax(predicted))
        rank = int(np.where(np.argsort(-actual) == chosen)[0][0]) + 1
        rho = float(spearmanr(predicted, actual).statistic)
        report = {
            "fold": fold,
            "train_prompts": int(len(train)),
            "test_prompts": int(len(test)),
            "selected_policy": {k: policies[chosen][k] for k in ("rounds", "n", "t1", "k1", "t2", "k2", "unet", "scores", "equiv")},
            "held_out_delta_ir": float(actual[chosen]),
            "held_out_rank": rank,
            "predicted_winner_in_actual_top3": bool(rank <= 3),
            "predicted_winner_in_actual_top5": bool(rank <= 5),
            "spearman_predicted_vs_actual": rho,
            "best_actual_policy": {k: policies[int(np.argmax(actual))][k] for k in ("rounds", "n", "t1", "k1", "t2", "k2")},
        }
        fold_reports.append(report)
        print(f"fold {fold}: held-out delta {report['held_out_delta_ir']:+.6f} | rank {rank:>4} | "
              f"rho {rho:+.3f} | {report['selected_policy']}")

    mean_held_out = float(np.mean([d["held_out_delta_ir"] for d in fold_reports]))
    positive = int(sum(1 for d in fold_reports if d["held_out_delta_ir"] > 0))
    mean_rho = float(np.nanmean([d["spearman_predicted_vs_actual"] for d in fold_reports]))
    gate_pass = bool(mean_held_out > 0 and positive >= GATE_MIN_POSITIVE_FOLDS and mean_rho >= GATE_MIN_SPEARMAN)

    predicted_all = predict_surfaces(np.arange(len(bank)))
    model_choice = int(np.nanargmax(predicted_all))
    actual_delta = q_value - psp_q[:, None]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, len(bank), size=(BOOTSTRAP, len(bank)))
    boot = actual_delta[picks].mean(axis=1)
    lcb = np.percentile(boot, LCB_QUANTILE * 100, axis=0)
    freeze_choice = int(np.argmax(lcb))

    ranking = [{
        "index": j, "rounds": p["rounds"], "n": p["n"], "t1": p["t1"], "k1": p["k1"],
        "t2": p["t2"], "k2": p["k2"], "unet": p["unet"], "scores": p["scores"], "equiv": p["equiv"],
        "prune_percent": p["prune_percent"],
        "p_miss": float(pooled_miss[j]), "ell": float(pooled_ell[j]),
        "mean_delta_ir": float(actual_delta[:, j].mean()),
        "lcb80_delta_ir": float(lcb[j]),
        "predicted_delta_ir_all200": float(predicted_all[j]),
    } for j, p in enumerate(policies)]
    ranking.sort(key=lambda r: -r["lcb80_delta_ir"])

    summary = {
        "bank_sha256": bank_digest(),
        "prompts": len(bank), "policies": len(policies),
        "draws_per_prompt_per_policy": DRAWS, "subset_seed": SUBSET_SEED,
        "bootstrap": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "lcb_quantile": LCB_QUANTILE,
        "ridge": RIDGE, "numpy": np.__version__, "python": platform.python_version(),
        "policies_one_round": int(len(family["one"]["idx"])), "policies_two_round": int(len(family["two"]["idx"])),
        "invariant_max_abs_error": invariant,
        "psp_reference": {"policy": PSP, "mean_q_ir": float(psp_q.mean())},
        "folds": fold_reports,
        "out_of_fold_gate": {
            "mean_held_out_delta_ir": mean_held_out, "positive_folds": positive,
            "mean_spearman": mean_rho, "required_positive_folds": GATE_MIN_POSITIVE_FOLDS,
            "required_spearman": GATE_MIN_SPEARMAN, "pass": gate_pass,
        },
        "model_choice_all200": {
            "policy": {k: policies[model_choice][k] for k in ("rounds", "n", "t1", "k1", "t2", "k2", "unet", "scores", "equiv")},
            "mean_delta_ir": float(actual_delta[:, model_choice].mean()),
            "lcb80_delta_ir": float(lcb[model_choice]),
        },
        "freeze_gate": {
            "policy": {k: policies[freeze_choice][k] for k in ("rounds", "n", "t1", "k1", "t2", "k2", "unet", "scores", "equiv")},
            "lcb80_delta_ir": float(lcb[freeze_choice]),
            "mean_delta_ir": float(actual_delta[:, freeze_choice].mean()),
            "pass": bool(lcb[freeze_choice] > 0),
        },
        "top_by_lcb80": ranking[:10],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fold_selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    header = list(ranking[0])
    with (OUT / "schedule_ranking.csv").open("w") as handle:
        handle.write(",".join(header) + "\n")
        for row in ranking:
            handle.write(",".join(str(row[k]) for k in header) + "\n")

    print(json.dumps({k: summary[k] for k in ("out_of_fold_gate", "model_choice_all200", "freeze_gate")}, indent=2))
    print(f"wrote {OUT}/fold_selection.json, {OUT}/schedule_ranking.csv")


if __name__ == "__main__":
    main()
