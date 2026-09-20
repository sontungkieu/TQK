# Final report: calibrated single-stage schedule vs PSP

## Outcome

| Method | Schedule | UNet evals | IR | HPS | GenEval | sec/prompt | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| PSP | 8→4@16→2@32 | 256 | 0.844619 | 0.278091 | 0.544304 | 42.699 | 5.25 GiB |
| Ours | 9→2@17 | 247 | 0.828312 | 0.278399 | 0.547920 | 40.734 | 5.37 GiB |

| Metric | PSP | Ours | Δ ours−PSP | 95% paired CI |
|---|---:|---:|---:|---:|
| ImageReward | 0.844619 | 0.828312 | -0.016307 | [-0.042217, 0.010073] |
| HPS | 0.278091 | 0.278399 | +0.000308 | [-0.001007, 0.001614] |
| GenEval | 0.544304 | 0.547920 | +0.003617 | [-0.019892, 0.027125] |

Paired GenEval outcomes: **22 ours wins / 511 ties / 20 PSP wins**.

## GenEval by official category

| Category | n | PSP | Ours | Δ | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| color_attr | 100 | 0.1600 | 0.1800 | +0.0200 | [-0.0400, 0.0800] |
| colors | 94 | 0.8723 | 0.8404 | -0.0319 | [-0.0851, 0.0106] |
| counting | 80 | 0.5250 | 0.5250 | +0.0000 | [-0.0875, 0.0875] |
| position | 100 | 0.1300 | 0.1500 | +0.0200 | [-0.0200, 0.0600] |
| single_object | 80 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] |
| two_object | 99 | 0.6869 | 0.6970 | +0.0101 | [-0.0606, 0.0707] |

## Required questions

1. **Family searched.** One checkpoint `t` from the fixed 13-point grid, K∈{1,2,3}, and maximum feasible M determined by the budget equation; 39 schedules total.
2. **Compute constraint.** `M*t + K*(64-t) ≤ 256`; M was never tuned independently.
3. **Calibration data.** 120 fixed prompts from the independent repository ImageReward `test_ir.json` corpus; zero exact GenEval overlap.
4. **Split.** 80 search / 40 untouched internal validation, deterministic seed 20260918.
5. **Frozen schedule.** `9→2@17`, 247 UNet evaluations (9 unused).
6. **Calibration IR.** Selected schedule all-120 mean `0.818419`; its internal-validation mean was `0.597439`.
7. **Oracle pool quality.** O(M*)=`0.938442` on all 120 calibration prompts.
8. **Selection regret.** R(t*,K*)=`0.120023`; Q=O−R holds numerically.
9. **Neighboring checkpoints.** The schedule won the pre-registered top-3 internal-validation comparison; no neighbor was tested after freezing.
10. **PSP calibration baseline.** Fixed PSP all-120 mean IR `0.817079`; selected schedule descriptive delta `+0.001340`.
11. **Untouched data.** All 553 official GenEval prompts remained untouched until Phase 2.
12. **Fresh Phase-2 seeds.** Base seed 20260919 with `base + prompt_id*32 + candidate_id`, disjoint from Phase 1.
13. **PSP final metrics.** IR `0.844619`, HPS `0.278091`, GenEval `0.544304`.
14. **Ours final metrics.** IR `0.828312`, HPS `0.278399`, GenEval `0.547920`.
15. **Paired uncertainty.** The table above gives all paired deltas and 10,000-resample 95% percentile CIs.
16. **Statistical conclusion.** ImageReward is inconclusive because its CI contains zero; HPS is inconclusive because its CI contains zero; GenEval is inconclusive because its CI contains zero.
17. **Runtime.** Ours−PSP mean `-1.9653` sec/prompt; PSP `42.6989`, ours `40.7335`.
18. **VRAM.** Ours−PSP max allocated `+0.128` GiB.
19. **Verifier work.** Ours uses `2` batched calls and `11` candidate scores/prompt versus PSP `3` and `14`; savings are `1` calls and `3` scores/prompt.
20. **Mechanism.** Phase 1 directly supports the observed breadth-versus-reliability trade-off: earlier checkpoints permit larger M and higher O(M) but incur larger R; later checkpoints reduce R while sacrificing feasible M. The selected schedule lies at the empirically chosen balance for this fixed grid, not a proof of global optimality.
21. **Narrowest defensible claim.** On this one fresh-seed, all-553-prompt SD1.5 evaluation, `9→2@17` versus fixed PSP produced the paired metric outcomes and CIs above at no more logical UNet compute. Claims beyond these metrics, seed pool, model, and prompt population are not supported.

## Runtime detail

| Method | mean s | median s | p90 s | throughput/GPU | online decode+IR s | mean/max VRAM | verifier calls/scores |
|---|---:|---:|---:|---:|---:|---:|---:|
| PSP | 42.699 | 41.921 | 43.857 | 0.0234 | 4.055 | 5.25/5.25 GiB | 3/14 |
| Ours | 40.734 | 39.952 | 41.854 | 0.0245 | 3.167 | 5.37/5.37 GiB | 2/11 |

## Interpretation guardrails

Phase 1 is schedule development and is not final evidence. Phase 2 used actual online diffusion, not replay. HPS and GenEval were never used to select or revise the schedule. No alternative schedule will be substituted if this frozen schedule loses.
