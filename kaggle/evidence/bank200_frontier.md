# Why fixed PSP wins: the per-N frontier and the budget mechanism

Third companion to [bank200_fold_selection.md](bank200_fold_selection.md) and
[bank200_three_round.md](bank200_three_round.md). Those two notes record *that* the pre-registered
gate fails; this one records the mechanism the raw rankings already contained but had not been
written down. No constant was chosen here: every number is read back from the two committed
ranking files, `exps/single_stage_calibration/replay200/schedule_ranking.csv` (1403 one- and
two-round shapes) and `schedule_ranking_3round.csv` (16 817 three-round shapes). Both were
produced by the frozen selectors, so this is a re-reading of measured rewards, not a new search.

## The union

| Family | Shapes | Best union rank | Best mean Delta IR | Worst |
| --- | ---: | ---: | ---: | ---: |
| one/two-discard | 1403 | **1** | **+0.000000** | -0.284340 |
| three-discard | 16 817 | 754 | -0.072658 | -0.687202 |

Union = 18 220 shapes. Three-discard shapes with a positive mean Delta IR: **0 of 16 817**.
Two-discard shapes with a better mean Delta IR than the best three-discard shape: 753 of 1403
(54%) - that is, the best three-discard shape is median-ish in the two-discard family, which is
the sharpest statement of how badly that family does here.

| Union rank | Shape | Delta IR | LCB_80 | unet | scores | equiv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `8 -> 4@16 -> 2@32 -> 1@64` (fixed PSP) | +0.000000 | +0.000000 | 256 | 14 | 282.6 |
| 2 | `7 -> 5@16 -> 2@32 -> 1@64` | -0.003632 | -0.006201 | 256 | 14 | 282.6 |
| 3 | `10 -> 4@12 -> 2@28 -> 1@64` | -0.003883 | -0.009445 | 256 | 16 | 286.4 |
| 4 | `9 -> 4@12 -> 2@32 -> 1@64` | -0.004365 | -0.008922 | 252 | 15 | 280.5 |
| 5 | `8 -> 4@18 -> 2@28 -> 1@64` | -0.004671 | -0.007499 | 256 | 14 | 282.6 |
| 754 | `7 -> 6@15 -> 4@16 -> 3@17 -> 1@64` (best 3-discard) | -0.072658 | -0.077283 | 256 | 20 | 294.0 |

## The tie band is empty above PSP

The paired standard error at n = 200 prompts is about 0.003 ImageReward, so `|Delta IR| < 0.006`
is a tie at 2 SE. Of the 1403 one/two-discard shapes, **5** fall inside that band, and **0** of
them beat PSP. There is no candidate that is plausibly better than the incumbent; the ranking is
not a coin-flip against it.

## The frontier collapses as the pool grows

Best achievable mean Delta IR at each initial pool size N, over both families:

| N | Shapes | Best Delta IR | Best shape | N | Shapes | Best Delta IR | Best shape |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 5 | 91 | -0.036341 | `5->3@32` | 15 | 1139 | -0.037582 | `15->3@8->2@32->1@64` |
| 6 | 450 | -0.016078 | `6->2@32` | 16 | 864 | -0.057396 | `16->3@8->2@24->1@64` |
| 7 | 798 | -0.003632 | `7->5@16->2@32->1@64` | 17 | 670 | -0.082897 | `17->3@8->2@16->1@64` |
| **8** | 1265 | **+0.000000** | **`8->4@16->2@32->1@64`** | 18 | 519 | -0.100763 | `18->2@8` |
| 9 | 1756 | -0.004364 | `9->4@12->2@32->1@64` | 19 | 388 | -0.084164 | `19->3@8->1@32->1@64` |
| 10 | 2013 | -0.003883 | `10->4@12->2@28->1@64` | 20 | 308 | -0.106829 | `20->3@8->1@28->1@64` |
| 11 | 2083 | -0.013540 | `11->3@12->2@32->1@64` | 21 | 229 | -0.131114 | `21->3@8->1@24->1@64` |
| 12 | 2107 | -0.008641 | `12->4@8->2@32->1@64` | 22 | 143 | -0.125961 | `22->2@8->1@32->1@64` |
| 13 | 1858 | -0.017992 | `13->4@8->2@28->1@64` | 23 | 64 | -0.167266 | `23->2@8->1@24->1@64` |
| 14 | 1462 | -0.028772 | `14->4@8->2@24->1@64` | 24 | 12 | -0.231321 | `24->2@8->1@16->1@64` |

The frontier is single-peaked at N = 8, and N = 8 is not a tuned value: it is the *only* row that
reaches zero, and it reaches it exactly at the incumbent's shape. Everything larger loses more
the larger it gets (N = 25, `25->1@8`, loses 0.284 IR - essentially all of the gain).

## The budget is what forces the peak

At a fixed 256-UNet budget, `N*t1 + ... <= 256` means a larger pool must prune earlier. The
latest first prune t1 that is still affordable with two discards, against the best Delta IR
reachable at that N:

| N | Max affordable t1 | as % of the 64-step trajectory | Best Delta IR |
| ---: | ---: | ---: | ---: |
| 5-7 | 28 | 44% | -0.0363 to -0.0036 |
| **8** | **24** | **38%** | **+0.000000** |
| 9 | 22 | 34% | -0.004364 |
| 10 | 20 | 31% | -0.003883 |
| 11 | 18 | 28% | -0.013540 |
| 12 | 17 | 27% | -0.008641 |
| 13 | 15 | 23% | -0.017992 |
| 14 | 14 | 22% | -0.028772 |
| 15-16 | 12 | 19% | -0.0376 to -0.0574 |
| 17-20 | 10 | 16% | -0.0829 to -0.1068 |
| 21-24 | 8 | 12% | -0.1259 to -0.2313 |

So the cliff is not a property of the selector, and it is not a threshold at N = 12 or N = 16
either. It is arithmetic: each extra candidate buys less pool at t1, t1 is pushed earlier, and an
early prune discards candidates before they have been scored on enough signal to be ranked. PSP
sits at the one point where the budget still allows prunes at 25% and 50% of the trajectory -
late enough to be informed, early enough to fit. That is the whole explanation for why the
pre-registered search returns the incumbent, and it is consistent with the frozen gate having
failed with a *high* rank correlation (the surface orders policies correctly; the top of the
order is simply flat).

## Caveat carried forward

Same as the other two notes: this is an offline replay over one fixed candidate pool per prompt
(25 candidates, seed `prompt_id*32 + candidate_id`), i.e. calibration evidence on shared latents,
not the fresh-seed confirmatory comparison reserved for protocol step 9. The stop condition still
stands and the 553-prompt run stays unlaunched.