# Pre-specified protocol

## Fixed comparison

- Source repository: official PSP, commit `590f59f58384169c719e431dc01d14006fb0fc0c`.
- Backbone: `runwayml/stable-diffusion-v1-5`, fp16, native 512×512.
- Sampler: DDIM from the checkpoint scheduler configuration, 64 steps,
  guidance scale 7.5, `eta=0`.
- PSP: `8→4@16→2@32`, followed by final ImageReward selection.
- Proposed method: `10→2@16`, followed by final ImageReward selection.
- Logical compute: `8*16 + 4*16 + 2*32 = 256` and
  `10*16 + 2*48 = 256`.
- Repetitions: one; base experiment seed 0.
- Hardware: exactly two RTX4090 24GB GPUs, independent prompt workers.

## Locked prompts and pairing

The source is the repository's official `geneval_metadata.jsonl`. Exactly 50
prompts are sampled from each of its six task groups with Python's deterministic
`random.Random(20260917)`, then sorted by original prompt ID. The resulting 300
IDs are committed before generation.

For prompt ID `p`, candidate `c` uses seed `p*32+c`. Candidates 0…9 are created
in one explicit latent-pool operation. PSP receives 0…7 and 10→2 receives 0…9.
The first eight tensors are asserted exactly equal and their SHA-256 hashes are
stored. Methods then execute independently.

## Live pruning and evaluation

At each checkpoint, the callback scores the DDIM scheduler's already-computed
`pred_original_sample`; it does not run another UNet evaluation. The survivor
latents and their corresponding prompt embeddings are pruned together. Since
`eta=0`, DDIM has no post-initialization sampling noise.

Only final winners are evaluated by HPS v2.1 and the repository's official
GenEval evaluator (Mask2Former Swin-S plus OpenCLIP ViT-L-14). ImageReward in
the result table is the final winner's ImageReward, not a checkpoint score.

## Statistics

All comparisons are paired over the 300 prompts. ImageReward, HPS, and GenEval
report the mean difference `10→2 - PSP` and a 95% percentile interval from
10,000 prompt-level bootstrap resamples with seed 20260917. GenEval additionally
reports paired win/tie/loss and all six category scores. No schedule or analysis
will be changed after results are observed.
