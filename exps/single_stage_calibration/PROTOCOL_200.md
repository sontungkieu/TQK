# Pre-registered 200-prompt calibration bank (branch `calib-200`)

This note freezes the second, denser calibration before any of its rewards exist. Once the
200-prompt bank has been generated and its ImageReward values inspected, the selection corpus,
the grouping, the search space, the gates, the endpoints, and the thresholds below must not
change. Phase 1 (`PROTOCOL.md`, 120 prompts, frozen `9→2@17`) is untouched on `main`; this
branch adds a new corpus and reuses the phase-1 worker and validator with explicit arguments.

## 1. Why a second bank

The offline replay of the phase-1 bank (`kaggle/evidence/multi_stage_sweep_2round.md`) scored
1403 two-round schedules and found two problems: the 80 search / 40 internal-validation split
disagreed enough that picking one schedule on 120 prompts would be selection noise, and the
verifier is not free. At the repository's 256-**UNet** budget the sweet point keeps a larger
initial pool and prunes early; under 256 **NFE-equivalents** (1 scorer score ≈ 1.9 UNet
evaluations on T4: 288 ms versus 151 ms) the scorer cost forces a smaller pool and late prunes,
and ImageReward collapses. The second bank enlarges the corpus and replaces the single split
with deterministic out-of-fold selection.

## 2. Prompt selection (fixed)

- Source corpus: `Fk-Diffusion-Steering/text_to_image/prompt_files/test_ir.json`,
  `source_sha256 = 3381897271e4ec863a459f233de7cc7147c18f2dac092bd612b3649a92569b51`.
- `selection seed = 20260926`; exactly 200 unique prompts, ids `0..199` in selection order.
- Zero exact string overlap with the 553 official GenEval prompts is asserted against
  `geneval_metadata.jsonl`,
  `geneval_sha256 = 5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13`.
- Produced by `select_prompts.py --count 200 --seed 20260926 --folds 5 --out-dir prompts200`
  and committed under `prompts200/`. The 120-prompt artefacts stay byte-identical:
  `select_prompts.py --verify-existing` is the regression check and the default arguments still
  reproduce the phase-1 files exactly.

## 3. Five-fold grouping (fixed)

- `split seed = 20260918`; a deterministic five-fold assignment over prompt ids, recorded per
  prompt in `prompts200/calibration_prompts.jsonl` as `fold` and summarised in
  `prompt_manifest.json` (`folds = 5`, `fold_counts = {"0".."4": 40}`).
- Each fold serves exactly once as the held-out set; selection for a fold reads the other 160
  prompts only.

## 4. Bank generation (fixed)

Identical to phase 1 apart from corpus size, grouping key, and destination:

- SD1.5 fp16, DDIM, 64 steps, `eta = 0`, guidance 7.5, batch of 25 candidates per prompt with
  `candidate seed = prompt_id*32 + candidate_id`; ImageReward is recorded at checkpoints
  8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 28, 32 as `X[p,i,t]` and at step 64 as `Y[p,i]`.
- Sharding is `prompt_id % num_workers == worker_index`; this run uses
  `--num-workers 60`, one Kaggle session per pair of worker indices, and
  `--out-dir exps/single_stage_calibration/bank200` so the phase-1 `bank_raw` is never
  rewritten. Merge with `kaggle/merge_shards.py --expect-prompts 200` and check with
  `validate_bank.py --expected-prompts 200 --bank-dir bank200`.
- The per-prompt record carries `fold` instead of `split`; no reward, checkpoint, or seed
  changes.

## 5. Search space (fixed, extended to two pruning rounds)

Feasibility is the repository budget in **logical UNet evaluations**, `CHECKPOINTS =
(8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 28, 32)` from `schedules.py`:

```text
one round:  N*t + K*(64-t) <= 256,                K in {1,2,3}
two rounds: N*t1 + k1*(t2-t1) + k2*(64-t2) <= 256,  t1 < t2, 2 <= k1 <= 25, 1 <= k2 < k1
N = min(floor((256 - tail) / t1), 25), N >= max(k1, 2)
```

That is the 1403-schedule space already replayed by `sweep_multi_stage.py` on phase 1. The
single-stage family from the README pre-registration (`t ∈ {10..28}`, `K ∈ {1,2,3}`, 57
policies) remains the primary hypothesis family, because the claim under test is single-stage
versus PSP; the two-round shapes are evaluated on the same corpus and reported alongside.
Cost is reported for every candidate policy as logical UNet evaluations, verifier calls,
verifier candidate scores, and NFE-equivalents with `1 score = 1.9 UNet evaluations`. A
"fits 256 NFE-equivalents" restriction is a secondary reporting view, not a second primary gate.

## 6. Out-of-fold selection (fixed)

For each fold `f`, in order:

1. fit the shared surfaces of §7 on the 160 prompts outside `f`;
2. select exactly one policy using those 160 prompts only;
3. replay the selected policy and fixed PSP `8→4@16→2@32→1@64` on the 40 held-out prompts;
4. record the paired held-out `Delta IR = Q(policy) - Q(PSP)`;
5. report Spearman rank correlation between the predicted and the actual ordering of all 1403
   schedules on the held-out fold, and whether the predicted winner is in the actual top 3 and
   top 5. Neither number may be used to revise the selector.

In-fold selection criterion: highest predicted `Delta IR` versus PSP, ties broken by fewer
verifier candidate scores, then smaller `N`, then more used UNet evaluations.

Pre-declared out-of-fold gate, all three, no tuning: mean held-out `Delta IR > 0`; at least
four of the five folds positive; rank correlation `>= 0.8`.

## 7. Shared reliability model (families fixed here, constants with the analysis commit)

No cell-wise means. The surfaces are shared and low-complexity, in the notation of the README
pre-registration:

- `O_p(M) = sum_{j=M..25} Y_(j) * C(j-1, M-1) / C(25, M)` with `Y_(j)` the ascending final
  rewards, `O(M) = mean_p O_p(M)`; reported unsmoothed, with a monotone-concave fit as a
  sensitivity check only.
- `p_miss`: logistic regression on `[q, log(M), K/M]` with `q = t/64`, constrained so the miss
  probability does not increase with denoising depth, fit with a deterministic solver.
- `ell`: positive regression with log link on the same three features.
- 500 subset draws per prompt and per required pool size, one fixed `subset seed = 20260927`,
  reused identically by every `(t,K)` comparison; PSP is replayed on the same draws.
- Invariant that must hold to numerical precision: direct mean regret equals `p_miss * ell`.
- `R = p_miss * ell`, `Q = O(M) - R`, `Delta = Q(policy) - Q(PSP)`.

The exact formula, regularization constants, and software versions are pinned in the analysis
script in a commit that must land before any ImageReward value of the new bank is read. That
commit may only add those constants; it may not touch the seeds, splits, gates, endpoints, or
thresholds above.

## 8. Freeze gate (fixed)

Only if the out-of-fold gate passes: refit the surfaces on all 200 prompts, bootstrap prompts
1000 times with `bootstrap seed = 20260928`, and compute the 80% lower confidence bound of
`Delta IR` for every candidate policy. Freeze `argmax_policy LCB_80(Delta IR)` and proceed only
if that bound is strictly positive. If no policy passes, stop without running GenEval-553.
The threshold is not lowered afterwards and no substitute schedule is chosen.

## 9. Confirmatory validation (fixed)

Only on a passing freeze gate, compare the frozen schedule with unchanged PSP on all 553
official GenEval prompts, using three fresh candidate-pool base seeds
`20300000, 20320000, 20340000` with `base + prompt_id*32 + candidate_id`. Each base spans at
most `552*32 + 24 = 17688` seeds, so the three ranges are disjoint from each other, from the
phase-1 bank (base 0), and from the phase-2 validation base `20260919`.

For prompt `p` and repetition `r` compute the paired difference `d[p,r]`; average the three
repetitions within prompt first, then bootstrap the 553 prompt-level averages 10000 times with
`bootstrap seed = 20260928`. The 1659 images are never treated as independent samples.

Primary endpoint: ImageReward, strong success requires positive mean `Delta IR` with a 95%
paired-confidence lower bound above zero. HPS and GenEval are secondary and must not show clear
degradation; if ImageReward wins while an independent metric declines, the claim is restricted
to better ImageReward optimisation. Report logical UNet evaluations, runtime mean/median/p90,
peak allocated VRAM, verifier calls and candidate scores per prompt, all prompt ids and seeds,
and all paired confidence intervals.

## 10. What this branch does *not* change

`main`'s phase-1 artefacts, the frozen `9→2@17` schedule, `PROTOCOL.md`, the 120-prompt
manifest, and the phase-2 validation report. All new behaviour is reachable only through
explicit arguments (`--prompts/--expected-prompts/--out-dir`, `--bank-dir`), whose defaults
still reproduce the phase-1 invocation exactly.
