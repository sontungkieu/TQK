# Final report: calibrated single-stage schedule vs PSP

## Outcome

| Method | Schedule | UNet evals | IR | HPS | GenEval | sec/prompt | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| PSP | 8→4@16→2@32 | 256 | 0.843532 | 0.277765 | 0.533454 | 5.009 | 8.27 GiB |
| Ours | 9→2@17 | 247 | 0.830516 | 0.278149 | 0.547920 | 4.816 | 8.77 GiB |

| Metric | PSP | Ours | Δ ours−PSP | 95% paired CI |
|---|---:|---:|---:|---:|
| ImageReward | 0.843532 | 0.830516 | -0.013016 | [-0.038019, 0.012314] |
| HPS | 0.277765 | 0.278149 | +0.000384 | [-0.000908, 0.001658] |
| GenEval | 0.533454 | 0.547920 | +0.014467 | [-0.007233, 0.036166] |

Paired GenEval outcomes: **23 ours wins / 515 ties / 15 PSP wins**.

## GenEval by official category

| Category | n | PSP | Ours | Δ | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| color_attr | 100 | 0.1600 | 0.1800 | +0.0200 | [-0.0300, 0.0800] |
| colors | 94 | 0.8511 | 0.8617 | +0.0106 | [-0.0319, 0.0532] |
| counting | 80 | 0.5000 | 0.5250 | +0.0250 | [-0.0500, 0.1000] |
| position | 100 | 0.1200 | 0.1400 | +0.0200 | [-0.0300, 0.0700] |
| single_object | 80 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] |
| two_object | 99 | 0.6768 | 0.6869 | +0.0101 | [-0.0505, 0.0707] |

## Required questions

1. **Family searched.** One checkpoint `t` from the fixed 13-point grid, K∈{1,2,3}, and maximum feasible M determined by the budget equation; 39 schedules total.
2. **Compute constraint.** `M*t + K*(64-t) ≤ 256`; M was never tuned independently.
3. **Calibration data.** 120 fixed prompts from the independent repository ImageReward `test_ir.json` corpus; zero exact GenEval overlap.
4. **Split.** 80 search / 40 untouched internal validation, deterministic seed 20260918.
5. **Frozen schedule.** `9→2@17`, 247 UNet evaluations (9 unused).
6. **Calibration IR.** Selected schedule all-120 mean `0.805482`; its internal-validation mean was `0.597439`.
7. **Oracle pool quality.** O(M*)=`0.936557` on all 120 calibration prompts.
8. **Selection regret.** R(t*,K*)=`0.131075`; Q=O−R holds numerically.
9. **Neighboring checkpoints.** The schedule won the pre-registered top-3 internal-validation comparison; no neighbor was tested after freezing.
10. **PSP calibration baseline.** Fixed PSP all-120 mean IR `0.809195`; selected schedule descriptive delta `-0.003712`.
11. **Untouched data.** All 553 official GenEval prompts remained untouched until Phase 2.
12. **Fresh Phase-2 seeds.** Base seed 20260919 with `base + prompt_id*32 + candidate_id`, disjoint from Phase 1.
13. **PSP final metrics.** IR `0.843532`, HPS `0.277765`, GenEval `0.533454`.
14. **Ours final metrics.** IR `0.830516`, HPS `0.278149`, GenEval `0.547920`.
15. **Paired uncertainty.** The table above gives all paired deltas and 10,000-resample 95% percentile CIs.
16. **Statistical conclusion.** ImageReward is inconclusive because its CI contains zero; HPS is inconclusive because its CI contains zero; GenEval is inconclusive because its CI contains zero.
17. **Runtime.** Ours−PSP mean `-0.1925` sec/prompt; PSP `5.0090`, ours `4.8165`.
18. **VRAM.** Ours−PSP max allocated `+0.501` GiB.
19. **Verifier work.** Ours uses `2` batched calls and `11` candidate scores/prompt versus PSP `3` and `14`; savings are `1` calls and `3` scores/prompt.
20. **Mechanism.** Phase 1 directly supports the observed breadth-versus-reliability trade-off: earlier checkpoints permit larger M and higher O(M) but incur larger R; later checkpoints reduce R while sacrificing feasible M. The selected schedule lies at the empirically chosen balance for this fixed grid, not a proof of global optimality.
21. **Narrowest defensible claim.** On this one fresh-seed, all-553-prompt SD1.5 evaluation, `9→2@17` versus fixed PSP produced the paired metric outcomes and CIs above at no more logical UNet compute. Claims beyond these metrics, seed pool, model, and prompt population are not supported.

## Runtime detail

| Method | mean s | median s | p90 s | throughput/GPU | online decode+IR s | mean/max VRAM | verifier calls/scores |
|---|---:|---:|---:|---:|---:|---:|---:|
| PSP | 5.009 | 5.009 | 5.071 | 0.1996 | 0.608 | 8.27/8.27 GiB | 3/14 |
| Ours | 4.816 | 4.818 | 4.889 | 0.2076 | 0.471 | 8.77/8.77 GiB | 2/11 |

## Interpretation guardrails

Phase 1 is schedule development and is not final evidence. Phase 2 used actual online diffusion, not replay. HPS and GenEval were never used to select or revise the schedule. No alternative schedule will be substituted if this frozen schedule loses.
