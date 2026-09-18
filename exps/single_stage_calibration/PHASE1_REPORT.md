# Phase-1 calibration report

## Outcome

The frozen single-stage schedule is **9→2@17** with
247 logical UNet evaluations (9 unused).
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
   K∈{1,2,3} satisfy M>K, M≤25, and compute≤256.
5. **Oracle quality O(M).** See `replay/oracle_curve_search.csv` and plot B; it is
   non-decreasing in M by construction.
6. **Selection regret.** See `replay/search_results.csv` and plot C; later
   checkpoints generally improve ranking reliability while reducing feasible breadth.
7. **Is Q non-monotonic in t?** True.
8. **Interior optimum?** The best search schedule was
   9→2@18; whether
   this is interior is visible in plot A and the full search table.
9. **Top three from search.** 9→2@18, 10→2@16, 9→2@17.
10. **Internal-validation selection.** 9→2@17, validation mean IR
    0.597439.
11. **Frozen schedule.** Exactly `9→2@17`; see `FROZEN_SCHEDULE.json`.
12. **Why selected?** It won under the pre-registered validation rule, including
    the 0.002 tie margin and fixed verifier/M/compute tie-breakers.
13. **Versus PSP on calibration.** Selected schedule all-120 mean IR 0.805482;
    fixed PSP all-120 mean IR 0.809195; descriptive delta
    -0.003712. This is development evidence, not Phase-2 confirmation.
14. **Breadth contribution.** Its oracle-pool quality O(M) is 0.936557.
15. **Selection loss.** Its mean selection regret is 0.131075, verifying
    Q=O−R (0.805482=0.936557−0.131075) up to floating point.

## Search and validation separation

- `search_results.csv` contains all 39 schedules on only the 80 search prompts.
- `validation_results.csv` contains only the three promoted schedules on only
  the 40 untouched internal-validation prompts.
- Reliability, heatmap, frontier, and broad mechanism sweeps use the search split.
- After freezing, only the selected schedule and fixed PSP are summarized over
  all 120 calibration prompts.
