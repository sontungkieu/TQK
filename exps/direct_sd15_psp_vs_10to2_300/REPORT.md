# Direct SD1.5 PSP vs 10→2 on 300 GenEval prompts

## Result

| Method | Schedule | Logical UNet evals | IR ↑ | HPS ↑ | GenEval ↑ | sec/prompt ↓ | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| PSP | 8→4@16→2@32 | 256 | 0.819063 | 0.278239 | 0.546667 | 4.958 | 8.27 GiB |
| 10→2 | 10→2@16 | 256 | 0.817583 | 0.277111 | 0.556667 | 4.995 | 9.27 GiB |

| Metric | PSP | 10→2 | Δ ours−PSP | 95% paired CI |
|---|---:|---:|---:|---:|
| ImageReward | 0.819063 | 0.817583 | -0.001480 | [-0.042951, 0.041256] |
| HPS | 0.278239 | 0.277111 | -0.001129 | [-0.003264, 0.001003] |
| GenEval | 0.546667 | 0.556667 | +0.010000 | [-0.023333, 0.043333] |

Paired inference over 300 prompts: ImageReward is inconclusive (CI contains zero); HPS is inconclusive (CI contains zero); GenEval is inconclusive (CI contains zero). The GenEval paired outcomes are **15 ours wins / 273 ties / 12 PSP wins**.

## GenEval by official task group

| Group | n | PSP | 10→2 | Δ | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| color_attr | 50 | 0.1000 | 0.1200 | +0.0200 | [0.0000, 0.0600] |
| colors | 50 | 0.8200 | 0.8800 | +0.0600 | [0.0000, 0.1400] |
| counting | 50 | 0.5800 | 0.5600 | -0.0200 | [-0.1600, 0.1200] |
| position | 50 | 0.1000 | 0.1200 | +0.0200 | [-0.0400, 0.0800] |
| single_object | 50 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] |
| two_object | 50 | 0.6800 | 0.6600 | -0.0200 | [-0.1200, 0.0800] |

## Runtime and hardware

| Hardware | Prompts | Total wall time | Avg utilization | Max VRAM |
|---|---:|---:|---:|---:|
| RTX4090 #0 | 150 | 0.424 h (shared wall) | 93.3% | 9.27 GiB |
| RTX4090 #1 | 150 | 0.424 h (shared wall) | 94.2% | 9.27 GiB |
| Combined | 300 | 0.424 h | 93.7% | 9.27 GiB |

Per-method runtime distribution:

| Method | mean s | median s | p90 s | mean peak VRAM | max peak VRAM |
|---|---:|---:|---:|---:|---:|
| PSP | 4.958 | 4.967 | 5.003 | 8.27 GiB | 8.27 GiB |
| 10→2 | 4.995 | 5.000 | 5.049 | 9.27 GiB | 9.27 GiB |


GPU-shard deltas (ours−PSP), used only as a consistency diagnostic:

| Shard | prompts | ΔIR | ΔHPS | ΔGenEval |
|---|---:|---:|---:|---:|
| GPU0 | 150 | +0.006052 | -0.002019 | +0.006667 |
| GPU1 | 150 | -0.009011 | -0.000239 | +0.013333 |

## Protocol audit

- PSP commit: `590f59f58384169c719e431dc01d14006fb0fc0c`.
- Repository status before launch: `ORX source snapshot (no .git directory)`.
- Repository diff summary before launch: `See recorded source commit and experiment patch`.
- Checkpoint: `runwayml/stable-diffusion-v1-5`, fp16, native 512×512 resolution.
- Sampler: `DDIMScheduler` from the checkpoint scheduler config, 64 steps, guidance scale 7.5, `eta=0`.
- PSP schedule was exactly `8→4@16→2@32`; ours was exactly `10→2@16`. Both were asserted to use 256 logical UNet evaluations.
- ImageReward was used at live pruning checkpoints and for final selection. The scheduler's existing `pred_original_sample` was scored; no extra UNet call was introduced.
- For every prompt, candidates 0…9 were generated once from deterministic seeds `prompt_id*32 + candidate_id`; PSP received 0…7 and 10→2 received 0…9. SHA-256 hashes were stored, and exact equality of the first eight tensors was asserted.
- Subset selection seed: `20260917`. The exact sorted IDs are in `prompt_ids_300.json`; there are 50 prompts from each of six official GenEval groups. IDs: `[1, 2, 3, 4, 6, 7, 11, 13, 14, 15, 16, 18, 21, 24, 25, 26, 30, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 46, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 63, 65, 66, 67, 69, 71, 74, 75, 76, 77, 78, 79, 82, 83, 84, 85, 89, 90, 91, 92, 96, 97, 98, 100, 104, 105, 106, 109, 111, 113, 114, 118, 120, 125, 126, 127, 129, 131, 133, 135, 137, 139, 140, 143, 144, 145, 149, 151, 154, 156, 159, 160, 161, 162, 164, 170, 172, 173, 175, 176, 177, 178, 180, 182, 183, 185, 188, 189, 190, 191, 192, 194, 195, 196, 197, 199, 203, 205, 206, 207, 210, 211, 213, 214, 215, 216, 218, 219, 220, 222, 224, 225, 227, 230, 231, 233, 234, 238, 240, 243, 244, 246, 247, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 261, 262, 263, 264, 266, 267, 268, 269, 271, 272, 273, 274, 275, 276, 279, 281, 283, 284, 285, 286, 287, 290, 292, 293, 294, 295, 296, 297, 307, 308, 315, 316, 318, 319, 320, 321, 322, 323, 325, 329, 330, 331, 336, 338, 342, 344, 345, 346, 352, 353, 354, 355, 357, 359, 360, 361, 365, 366, 367, 368, 370, 371, 374, 375, 381, 382, 384, 385, 388, 389, 391, 392, 397, 398, 401, 405, 406, 408, 410, 411, 415, 417, 420, 421, 424, 425, 427, 428, 432, 435, 439, 440, 441, 445, 446, 447, 448, 450, 452, 453, 455, 456, 458, 465, 466, 468, 469, 471, 472, 473, 477, 478, 479, 480, 481, 482, 483, 485, 486, 487, 490, 492, 495, 497, 499, 500, 503, 507, 514, 516, 518, 519, 520, 521, 522, 523, 524, 526, 527, 529, 531, 532, 536, 537, 542, 544, 546, 549, 550]`.
- Sharding: sorted subset position modulo 2. Both methods for a prompt always ran on the same GPU.
- Software: PyTorch `2.4.0+cu121`, CUDA runtime `12.1`, Diffusers `0.31.0.dev0`, Transformers `4.38.2`, ImageReward `1.5`, HPSv2 `1.2.0`.
- GPU names: `NVIDIA GeForce RTX 4090` and `NVIDIA GeForce RTX 4090`. Peak allocated VRAM remained below the 24 GiB device capacity; no schedule-altering memory workaround was used.
- HPS is HPS v2.1 on final winners only. GenEval is the repository's official Mask2Former/OpenCLIP evaluator on final winners only.
- Statistics: one repetition/base seed 0; 10,000 prompt-level paired percentile bootstrap resamples with bootstrap seed 20260917.

## Published PSP context (reference only)

The paper's full Table-1 SD1.5 values are approximately IR 0.827, HPS 0.278, and GenEval 0.574. This is **reference only**, not an exact Table-1 reproduction: this run uses 300 prompts, one repetition, and RTX4090 hardware.

## Narrow conclusion

ImageReward is inconclusive (CI contains zero); HPS is inconclusive (CI contains zero); GenEval is inconclusive (CI contains zero). This conclusion is restricted to SD1.5, this fixed 300-prompt subset, this single repetition, and the two pre-specified schedules. No follow-up schedule was selected or launched.
