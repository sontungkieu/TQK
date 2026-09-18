# Fixed-budget single-stage calibration

This experiment calibrates exactly one pruning checkpoint for deterministic
SD1.5/DDIM inference under a logical budget of at most 256 UNet evaluations.
The calibration corpus is a fixed 120-prompt subset of the repository's
independent ImageReward prompt set (`test_ir.json`), with no exact prompt overlap
with the 553 official GenEval prompts.

The run is deliberately two-stage:

1. Generate 25 complete trajectories per calibration prompt and score only the
   13 pre-registered checkpoints plus the final sample with ImageReward.
2. Replay every valid `(checkpoint, survivors)` schedule offline, rank on the
   80-prompt search split, select among the top three on the untouched 40-prompt
   internal-validation split, and write `FROZEN_SCHEDULE.json` once.

The selected schedule is consumed unchanged by
`../single_stage_validation_553/`.

