# Pre-registered 200-prompt bank: out-of-fold selection and freeze gate

Result of `exps/single_stage_calibration/select_fold_schedule.py` on the bank in
`bank200/`. The protocol, the search space, the seeds, the surfaces and both gates were
frozen in PROTOCOL_200.md before the bank was generated; nothing below was tuned after
seeing a reward. The run is deterministic (1 m 14 s on one CPU core).

| Item | Value |
| --- | --- |
| bank prompts | 200 |
| bank sha256 | 54a1fa41a5525718a285c5ef8035472571a30a6678419dc413b787cbb4160997 |
| policies | 1403 (one round 39, two rounds 1364) |
| subset draws | 500 per prompt and pool size, seed 20260927 |
| bootstrap | 1000 prompt resamples, seed 20260928, LCB at the 20th percentile |
| ridge | 1.0 |
| numpy | 1.26.4 |
| invariant max abs(direct regret - p_miss*ell) | 1.1e-16 |
| PSP reference mean Q(ImageReward) | 0.836698 |

## Out-of-fold gate (five folds, 160 fit / 40 held out)

| Fold | Held-out Delta IR | Rank of predicted winner | Spearman predicted vs actual |
| ---: | ---: | ---: | ---: |
| 0 | -0.033718 | 228 / 1403 | +0.845 |
| 1 | -0.011263 | 17 / 1403 | +0.880 |
| 2 | -0.014419 | 95 / 1403 | +0.746 |
| 3 | -0.047721 | 496 / 1403 | +0.891 |
| 4 | -0.025014 | 79 / 1403 | +0.868 |

| Gate | Requirement | Observed | Pass |
| --- | --- | --- | --- |
| mean held-out Delta IR | > 0 | -0.026427 | False |
| positive folds | >= 4 of 5 | 0 | False |
| rank correlation | >= 0.8 | +0.846 | True |
| **out-of-fold gate** | all three | | **FAIL** |

Every fold selects the same policy, `8->4@8->3@32->1@64` (`unet 256`, `scores 15`, `equiv 284.5`),
and every fold loses to PSP on the held-out prompts. The selector ranks policies well
(mean Spearman +0.846) but its argmax is not the argmax of the truth.

## Freeze gate (refit on all 200, 1000 bootstrap resamples)

| Field | Value |
| --- | --- |
| model choice on all 200 | 8->4@8->3@32->1@64 |
| its mean Delta IR | -0.026427 |
| its LCB_80 | -0.033392 |
| argmax LCB_80 | 8->4@16->2@32->1@64 |
| argmax LCB_80 value | +0.000000 |
| freeze gate (LCB_80 > 0) | **FAIL** |

## What the bank says about the search space

| Policy | Actual rank of 1403 | Mean Delta IR | LCB_80 |
| --- | ---: | ---: | ---: |
| PSP 8->4@16->2@32->1@64 | 1 | +0.000000 | +0.000000 |
| phase-1 frozen 9->2@17 | 529 | -0.050987 | -0.055952 |
| model choice 8->4@8->3@32->1@64 | 115 | -0.026427 | -0.033392 |
| 25->1@8 (largest pool, earliest single prune) | 1402 | -0.283531 | -0.313330 |

- Policies with a positive mean Delta IR on the 200-prompt bank: **0 of 1403**.
- Fixed PSP is the best schedule in the bank. The closest non-PSP policy is
  `7->5@16->2@32->1@64`, still 0.003633 ImageReward behind, and its LCB_80 is negative.

## Conclusion

The pre-declared stop condition fires: the freeze gate found no policy whose 80% lower
confidence bound on Delta IR is positive, so per PROTOCOL_200 section 8 the experiment stops
here and the 553-prompt confirmatory run is **not** launched. The honest reading is that on
an independent 200-prompt bank, at the repository 256-UNet budget, the fixed progressive
schedule is not beaten by any of the 1403 one- and two-round shapes that the pre-registered
space contains. The high rank correlation says a shared surface can *order* policies; it
does not make the top of that order better than PSP. Reporting this is the point of fixing
the gate beforehand.

Caveat: this is an offline replay of one fixed candidate pool per prompt (25 candidates,
seed `prompt_id*32 + candidate_id`), so it compares schedules on shared latents rather than
on fresh candidate seeds. It is therefore calibration evidence, not the confirmatory
fresh-seed comparison the protocol reserves for step 9.
