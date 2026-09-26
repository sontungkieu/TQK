# Addendum: three-discard shapes, declared before their rewards are computed

PROTOCOL_200 §5 fixed the search space as one- and two-discard shapes. The original research
question also asks for **three** discards, so this addendum extends the space and freezes the
extension *before* any three-discard reward is computed from `bank200/`. Nothing in
PROTOCOL_200.md changes; the one- and two-discard results already recorded stand.

## Space

Budget, checkpoints, feasibility rule and the offline-replay argument are unchanged. A
three-discard shape is `N -> k1@t1 -> k2@t2 -> k3@t3 -> 1@64` with

```text
N*t1 + k1*(t2-t1) + k2*(t3-t2) + k3*(64-t3) <= 256
t1 < t2 < t3 on CHECKPOINTS, 2 <= k1 <= 25, 1 <= k2 < k1, 1 <= k3 < k2
N = min(floor((256 - tail)/t1), 25), N >= max(k1, 2)
```

that is 657,800 enumerated shapes, **16,817 feasible at 256 logical UNet**. Counted before any
reward: this addendum records feasibility only.

## Procedure

Identical to PROTOCOL_200 §3, §6, §7 and §8, with no re-tuning:

- the same 500 subset draws per prompt and pool size, seed 20260927, shared by every shape;
- the same shared-surface families: p_miss binomial logistic (q coefficients constrained <= 0)
  and ell Gamma log-link, fitted per family with ridge 1.0;
- the same five-fold out-of-fold gate (mean held-out Delta IR > 0, >= 4/5 positive folds,
  Spearman >= 0.8) and the same freeze gate (argmax LCB_80 over 1000 prompt resamples,
  seed 20260928, must be > 0);
- Delta IR is Q(policy) - Q(PSP) with Q = O_p(M) - R_p and PSP fixed at `8->4@16->2@32->1@64`.

## Multiple comparisons, stated in advance

16,817 shapes with a paired standard error near 0.003 ImageReward means the in-sample argmax of
this family is expected to look positive even if no shape is truly better. A positive-looking
argmax is therefore **not** evidence; only the out-of-fold and freeze gates are. If they fail,
the reported conclusion is the pre-declared stop, exactly as for the two-discard space.

## Cost readings

All results are reported under both accountings, because they disagree:

1. **256 logical UNet evaluations** — the repository's definition and PROTOCOL_200's primary
   budget;
2. **256 NFE-equivalents**, charging the ImageReward scorer at the measured T4 rate
   (1 score = 1.9 UNet evaluations). Under this reading fixed PSP (282.6) is infeasible and only
   six of the measured one/two-discard shapes fit at all, so the accounting choice is itself a
   result worth reporting.
