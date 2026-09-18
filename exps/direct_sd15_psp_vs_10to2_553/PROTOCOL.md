# Full 553-prompt extension protocol

- All 553 official GenEval prompts, in original order; no subset selection.
- One repetition, base seed 0.
- Stable Diffusion v1.5 fp16; DDIM 64 steps, eta 0, guidance 7.5.
- PSP `8→4@16→2@32` versus `10→2@16`; both exactly 256 logical UNet evaluations.
- Live ImageReward on the scheduler's existing predicted-clean sample and final ImageReward selection.
- Explicit candidates use seed `prompt_id*32 + candidate_id`; PSP receives candidates 0…7 and 10→2 receives 0…9, with exact first-eight tensor pairing.
- Two independent RTX4090 prompt workers; both methods for a prompt remain on the same physical GPU.
- The completed predecessor's 300 prompts are reused only after validating model, scheduler, schedule traces, compute, latent hashes, and paired GPU ownership. The remaining 253 prompts are newly generated.
- HPS v2.1 and official GenEval evaluate only the 553 final winners per method.
- Prompt-level paired inference uses 10,000 percentile bootstrap resamples with seed 20260917.
- No DDP, NCCL, cross-GPU trajectory, offline replay, adaptive criterion, or schedule tuning.
