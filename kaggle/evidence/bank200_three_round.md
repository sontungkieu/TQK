# Three-discard shapes on the 200-prompt bank, and the two budget readings

Companion to [bank200_fold_selection.md](bank200_fold_selection.md). The space, seeds, surfaces
and gates were frozen in `PROTOCOL_200_EXT.md` before any three-discard reward was computed;
the run is `exps/single_stage_calibration/select_three_round.py` (11 m 47 s, deterministic,
invariant `|direct regret - p_miss*ell|` = 1.1e-16).

## Space

```text
N -> k1@t1 -> k2@t2 -> k3@t3 -> 1@64,  N*t1 + k1*(t2-t1) + k2*(t3-t2) + k3*(64-t3) <= 256
enumerated 657 800 | feasible at 256 logical UNet 16 817 | feasible at 256 NFE-equivalent 0
```

## Outcome

| Gate | Requirement | Three-discard family | Result |
| --- | --- | --- | --- |
| mean held-out Delta IR | > 0 | -0.081219 | fail |
| positive folds | >= 4 of 5 | 0 | fail |
| rank correlation | >= 0.8 | +0.9638 | pass |
| **out-of-fold gate** | all three | | **FAIL** |

Every fold selects the same shape, `10 -> 5@8 -> 4@10 -> 3@12` (prune at 12% / 16% / 19% of the
denoising trajectory, `unet 254`, `scores 22`, `equiv 295.8`), and every fold loses to PSP:
held-out Delta IR -0.089504, -0.061646, -0.070731, -0.096549, -0.087665.

Adding the family to the measured union (1403 + 16 817 = 18 220 shapes) changes nothing at the
top, because the top is not a three-discard shape at all:

| Rank (measured mean Delta IR) | Shape | Delta IR | equiv |
| ---: | --- | ---: | ---: |
| 1 | `8 -> 4@16 -> 2@32 -> 1@64` (fixed PSP) | +0.000000 | 282.6 |
| 2 | `7 -> 5@16 -> 2@32 -> 1@64` | -0.003632 | 282.6 |
| 3 | `10 -> 4@12 -> 2@28 -> 1@64` | -0.003883 | 286.4 |

Freeze gate over the union (LCB_80, 1000 prompt resamples): the three highest lower bounds are
also one/two-discard shapes `8->4@16->2@32` (+0.000000), `7->5@16->2->32` (-0.006201) and
`8->4@18->2->28` (-0.007499). No policy in any family has a positive lower bound, so the
pre-declared stop condition stands.

## The two budget readings disagree

| Accounting | Best shape | Delta IR | Comment |
| --- | --- | --- | --- |
| 256 logical UNet (repository definition) | PSP `8->4@16->2@32` | +0.000000 | 1403 + 16 817 shapes compete; PSP wins |
| 256 NFE-equivalent, scorer at 1.9x UNet-eval | `6->2@28` (one-discard) | -0.035620 | only 6 of 1403 one/two-discard shapes fit; **no** three-discard shape fits (0 of 16 817) |

At 256 NFE-equivalents fixed PSP (282.6) is not even feasible, and nothing fits at all below
240. The accounting choice therefore decides the answer: charging the scorer removes PSP from
the comparison and forces small pools with late prunes, which cost about 0.036 ImageReward.

## Reading

The three-discard family is the one the protocol expects to be hardest to pin down, and it is:
its in-sample optimum prunes at 12% of the trajectory, and that choice loses about 0.081 IR
out of fold - more than three times the two-discard family's -0.026. The high rank correlation
(0.96) says the shared surface predicts the *order* of 16 817 shapes well; it does not make the
predicted winner good. Under the repository budget the answer to "how many discards, and when"
is therefore: two, at 25% and 50%, which is exactly fixed PSP.
