# Phase-1 protocol (fixed before calibration inference)

- Model: `runwayml/stable-diffusion-v1-5`, fp16.
- Scheduler: DDIM, 64 steps, `eta=0`, guidance scale 7.5.
- Intermediate score: ImageReward on the scheduler-provided predicted-clean
  sample (`x0_preds`); no extra UNet evaluation.
- Candidate bank: 25 deterministic initial latents per prompt, candidate IDs
  0–24, seed `prompt_id*32 + candidate_id`.
- Prompt corpus: 120 prompts sampled once from the PSP repository's
  `test_ir.json` using selection seed `20260918`; exact string overlap with
  official GenEval is asserted to be zero.
- Nested calibration split: 80 search prompts and 40 internal-validation
  prompts, fixed with split seed `20260918`.
- Checkpoint grid: 8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 28, 32.
- Survivor counts: 1, 2, 3.
- For each `(t,K)`, `M=floor((256-K*(64-t))/t)`; schedules with `M<=K` or
  `M>25` are invalid. No independent tuning of `M` is permitted.
- Search objective: mean final ImageReward `Q(t,K)` on the 80 search prompts.
- Advance exactly the top three schedules to internal validation.
- Final selection: highest mean ImageReward on the untouched 40 prompts. If
  schedules are within 0.002 of the best, prefer fewer verifier candidate
  scores (`M+K`), then smaller `M`, then more used UNet evaluations.
- PSP replay is fixed at `8→4@16→2@32→1@64` and is never tuned.
- HPS, GenEval, CLIP, adaptive criteria, DFS, MPPI, learned uncertainty, and
  multiple pruning stages for the proposed method are excluded from Phase 1.
- Once written, `FROZEN_SCHEDULE.json` is immutable for Phase 2.

