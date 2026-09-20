# T4 replication of the published Phase-2 comparison

Provenance for the numbers this deployment produced on Kaggle 2x Tesla T4. This is not a
substitute for the published 2x RTX 4090 result; it exists so the T4 run can be audited and
compared against it.

## Runs

| Piece | Kernel | Session |
| --- | --- | --- |
| 553-prompt generation, HPS v2.1, GenEval export | `codemaivanngu/tqk-eval-553-260918-1759` | 24 394.7 s (6.78 h): `shard` 24 094.9 s, `hps` 211.1 s, 553 prompts validated |
| GenEval scoring (evaluation-only session, one method per GPU) | `codemaivanngu/tqk-geneval-eval-260919-1846` | ~20 min, COMPLETE |
| GenEval environment CPU preflight | `codemaivanngu/tqk-geneval-env-build-260919-0903` | COMPLETE (env built in ~72 s) |
| Second generation session on another account | `kieutung/tqk-eval-553-geneval-260919-0908` | generation and GenEval complete; the job hit the 2 h `evaluate-geneval` step timeout before its own report, so the numbers above come from the first kernel |

Code: TQK `b6946ba`; `sontungkieu/my-gpt-skill` `b7dcf69` (PR #18) carries the KJO fixes that made
the in-session GenEval environment reproducible.

## Result

One run, 553 prompts, paired differences per prompt, 10 000-resample percentile bootstrap
(seeds 20260919/20260920/20260921).

| Metric | PSP | Ours | Delta (ours - PSP) | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| ImageReward | 0.844619 | 0.828312 | -0.016307 | [-0.042217, +0.010073] |
| HPS | 0.278091 | 0.278399 | +0.000308 | [-0.001007, +0.001614] |
| GenEval | 0.544304 | 0.547920 | +0.003617 | [-0.019892, +0.027125] |

Every interval contains zero. The published 4090 run reported ImageReward -0.013016
[-0.038019, +0.012314] and HPS +0.000384, so the T4 replication agrees with the published
conclusion: an efficiency trade-off, not a statistically supported quality improvement.

Cost side, from `metrics/summary.csv`: 247 versus 256 logical UNet evaluations (-3.5%), 40.734
versus 42.699 s/prompt (-4.6%), peak VRAM 5.37 versus 5.25 GiB.

Official GenEval task-average (the evaluator's own headline) is 0.56237 (PSP) versus 0.56540
(Ours); per-prompt win/tie/loss is 22 / 511 / 20. Category deltas: `color_attr` +0.020,
`position` +0.020, `two_object` +0.010, `colors` -0.032, `counting` 0, `single_object` 0.

## Contents

- `FINAL_REPORT.md` - the report `exps/single_stage_validation_553/aggregate.py` wrote inside the
  run, including the paired tables and the 21 required protocol questions.
- `metrics/` - `summary.csv`, `paired_bootstrap.json`, `category_bootstrap.csv`,
  `geneval_win_tie_loss.json`, `gpu_shards.csv`, `per_prompt.csv`, `hps.csv` and the two
  official `summary_scores.py` outputs.
- `geneval_results/` - raw per-image GenEval decisions (553 x 2 JSONL) behind the GenEval numbers.
- `geneval_stages.txt` - the in-session install report, showing which stage produced what.

## Notes

* The pipeline is deterministic under the pinned seeds, so the second session on `kieutung`
  reproduced the same GenEval and HPS values; it is a reproducibility check, not an independent
  sample.
* Generation used the frozen schedule `9->2@17`; `FROZEN_SCHEDULE.json` was never modified, and
  phase-1 provenance for this host is in `kaggle/evidence/phase1_t4_freeze.json`.
* GenEval must be scored in the session that owns the images: `export_geneval.py` writes
  `geneval_inputs/<method>/<id>.png` as symlinks into that session's `outputs/` tree. The
  evaluation-only job therefore mounts the finished kernel as a kernel source instead of
  regenerating anything.
