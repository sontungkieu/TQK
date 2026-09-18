# Phase-2 protocol (fixed before GenEval inference)

- Evaluation population: all 553 official GenEval prompts. None were used to
  select the single-stage schedule.
- Fresh seed pool: `20260919 + prompt_id*32 + candidate_id`; no candidate seed
  overlaps the Phase-1 pool.
- PSP: `8→4@16→2@32→1@64`, exactly 256 logical UNet evaluations.
- Proposed method: read only from the checksum-locked
  `FROZEN_SCHEDULE.json`; one intermediate pruning checkpoint and final
  ImageReward winner selection, at most 256 logical UNet evaluations.
- Shared candidates: both methods use the same explicitly constructed latents
  for overlapping candidate IDs. Pool size is `max(8,M*)`; neither method's
  prescribed pool size is changed.
- Model: `runwayml/stable-diffusion-v1-5`, fp16.
- Sampler: DDIM, 64 steps, eta 0, guidance scale 7.5.
- Hardware: two RTX 4090 prompt workers. Both methods for a prompt execute on
  the same GPU. No DDP, cross-GPU trajectory, TensorRT, TPU, or JAX.
- Primary paired metrics: final ImageReward, HPS v2.1, official GenEval.
- Statistics: 10,000 prompt-level paired percentile-bootstrap resamples with
  seed 20260919.
- Runtime includes diffusion, checkpoint/final predicted-clean decode, and
  online ImageReward; it excludes model loading, HPS, GenEval, and persistence
  decode after the winner is already selected.
- No PSP tuning, schedule change, neighboring-schedule test, or post-validation
  search is permitted.

