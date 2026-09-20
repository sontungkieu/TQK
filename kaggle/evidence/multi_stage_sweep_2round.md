# Two-round seed-pruning sweep on the 120-prompt calibration bank

Offline replay of the phase-1 bank (`exps/single_stage_calibration/bank_raw`, 120 prompts x 25
candidates, ImageReward recorded at checkpoints 8..32 and at step 64). Tool:
`exps/single_stage_calibration/sweep_multi_stage.py`. 1403 schedules: every two-round shape
`N -> k1@t1 -> k2@t2 -> 1@64` that satisfies the repository budget
`N*t1 + k1*(t2-t1) + k2*(64-t2) <= 256`, plus the one-round grid `t x K in {1,2,3}` as a
reference. Selection uses the 80 search prompts only; the 40 internal-validation prompts are read
afterwards.

## Machinery check

The replay reproduces the published phase-1 numbers exactly, so the sweep is measuring the same
thing the calibration did:

| Schedule | Search IR (80) | Validation IR (40) |
| --- | ---: | ---: |
| `9->2@17` (frozen, one prune) | 0.933569 | 0.588119 |
| published phase-1 values | 0.9335689 | 0.5881194 |

## What the sweep says

Best two-round shapes by the search split (all prune early and keep two at the end):

| Schedule | Prune | UNet | Scores | NFE-equiv | Search IR | Validation IR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `10 -> 6@14 -> 2@18 -> 1@64` | 22% / 28% | 256 | 18 | 290.2 | 0.964735 | 0.614923 |
| `10 -> 5@14 -> 2@18 -> 1@64` | 22% / 28% | 252 | 17 | 284.3 | 0.962726 | 0.615075 |
| `11 -> 7@10 -> 2@17 -> 1@64` | 16% / 27% | 253 | 20 | 291.0 | 0.956838 | 0.635002 |
| `9 -> 2@17 -> 1@64` (frozen) | 27% | 247 | 11 | 267.9 | 0.933569 | 0.588119 |
| `8 -> 4@16 -> 2@32 -> 1@64` (PSP) | 25% / 50% | 256 | 14 | 282.6 | 0.891587 | 0.668063 |

## Two caveats that decide what to do next

1. **The two splits disagree, so 120 prompts is too small for a 1403-schedule search.** PSP is
   worst on the search split (0.8916) and best on the validation split (0.6681); the
   search-best two-round shape is best on search and second on validation. This is the same
   disagreement the published calibration already reported (`validation - search` delta versus PSP
   was -0.0799), now amplified by searching many more shapes. Selecting one schedule from this
   table would be selection noise, not a sweet point.
2. **The verifier is not free.** The winning shapes spend 17-20 scorer scores, i.e. 284-294 NFE
   equivalents once the measured T4 factor (1 score ~ 1.9 UNet evaluations: 288 ms versus
   151 ms) is applied. Only **one** two-round schedule fits 256 NFE-equivalents, and it is much
   worse on both splits:

| Schedule | Prune | UNet | Scores | NFE-equiv | Search IR | Validation IR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `7 -> 2@28 -> 1@32 -> 1@64` | 44% / 50% | 236 | 10 | 255.0 | 0.779357 | 0.501877 |

So the trade-off is explicit: at the repository's 256-**UNet** budget the sweet point wants a
larger initial pool and early prunes; at 256 **NFE-equivalents** the scorer cost forces a small
pool and late prunes, and quality collapses.

## Next step

Enlarge the bank before spending GPU hours on a validation run: 340 s/prompt measured on T4 means
400 prompts cost ~37.8 GPU-h, i.e. about 4.7 h wall on 4 accounts at 2 GPUs each, or 2.4 h on 8.
With 400 prompts the same sweep can be run with a 267/133 split, which is what makes a
selection over hundreds of shapes defensible.
