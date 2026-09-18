# Direct SD1.5 PSP vs 10→2 on all 553 GenEval prompts

## Result

| Method | Schedule | Logical UNet evals | IR ↑ | HPS ↑ | GenEval ↑ | sec/prompt ↓ | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| PSP | 8→4@16→2@32 | 256 | 0.838022 | 0.278026 | 0.522604 | 4.955 | 8.27 GiB |
| 10→2 | 10→2@16 | 256 | 0.829159 | 0.277447 | 0.535262 | 4.991 | 9.27 GiB |

| Metric | PSP | 10→2 | Δ ours−PSP | 95% paired CI |
|---|---:|---:|---:|---:|
| ImageReward | 0.838022 | 0.829159 | -0.008864 | [-0.043690, 0.027428] |
| HPS | 0.278026 | 0.277447 | -0.000580 | [-0.002189, 0.001049] |
| GenEval | 0.522604 | 0.535262 | +0.012658 | [-0.010850, 0.037975] |

Paired inference over 553 prompts: ImageReward is inconclusive (CI contains zero); HPS is inconclusive (CI contains zero); GenEval is inconclusive (CI contains zero). The GenEval paired outcomes are **27 ours wins / 506 ties / 20 PSP wins**.

## GenEval by official task group

| Group | n | PSP | 10→2 | Δ | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| color_attr | 100 | 0.0900 | 0.1400 | +0.0500 | [0.0000, 0.1000] |
| colors | 94 | 0.8511 | 0.8723 | +0.0213 | [-0.0319, 0.0745] |
| counting | 80 | 0.5750 | 0.5500 | -0.0250 | [-0.1250, 0.0750] |
| position | 100 | 0.0900 | 0.1000 | +0.0100 | [-0.0300, 0.0500] |
| single_object | 80 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] |
| two_object | 99 | 0.6566 | 0.6667 | +0.0101 | [-0.0606, 0.0808] |

## Runtime and hardware

| Hardware | Prompts | Total wall time | Avg utilization | Max VRAM |
|---|---:|---:|---:|---:|
| RTX4090 #0 | 277 | 0.788 h (shared wall) | 94.4% | 9.27 GiB |
| RTX4090 #1 | 276 | 0.788 h (shared wall) | 94.0% | 9.27 GiB |
| Combined | 553 | 0.788 h | 94.2% | 9.27 GiB |

Per-method runtime distribution:

| Method | mean s | median s | p90 s | mean peak VRAM | max peak VRAM |
|---|---:|---:|---:|---:|---:|
| PSP | 4.955 | 4.960 | 5.000 | 8.27 GiB | 8.27 GiB |
| 10→2 | 4.991 | 4.990 | 5.040 | 9.27 GiB | 9.27 GiB |


GPU-shard deltas (ours−PSP), used only as a consistency diagnostic:

| Shard | prompts | ΔIR | ΔHPS | ΔGenEval |
|---|---:|---:|---:|---:|
| GPU0 | 277 | +0.011567 | -0.000676 | +0.025271 |
| GPU1 | 276 | -0.029368 | -0.000483 | +0.000000 |

## Protocol audit

- PSP commit: `590f59f58384169c719e431dc01d14006fb0fc0c`.
- Repository status before launch: `captured in metrics/run_info.json`.
- Repository diff summary before launch: `captured in metrics/run_info.json`.
- Checkpoint: `runwayml/stable-diffusion-v1-5`, fp16, native 512×512 resolution.
- Sampler: `DDIMScheduler` from the checkpoint scheduler config, 64 steps, guidance scale 7.5, `eta=0`.
- PSP schedule was exactly `8→4@16→2@32`; ours was exactly `10→2@16`. Both were asserted to use 256 logical UNet evaluations.
- ImageReward was used at live pruning checkpoints and for final selection. The scheduler's existing `pred_original_sample` was scored; no extra UNet call was introduced.
- For every prompt, candidates 0…9 were generated once from deterministic seeds `prompt_id*32 + candidate_id`; PSP received 0…7 and 10→2 received 0…9. SHA-256 hashes were stored, and exact equality of the first eight tensors was asserted.
- Prompt set: all 553 official GenEval prompts in original order, IDs 0…552, recorded in `prompt_ids_553.json`.
- Sharding: the 300 imported prompts retained their actual predecessor GPU; the remaining prompts were assigned greedily to balance the final 277/276 split. Both methods for every prompt ran on the same GPU.
- Software: PyTorch `2.4.0+cu121`, CUDA runtime `12.1`, Diffusers `0.31.0.dev0`, Transformers `4.38.2`, ImageReward `1.5`, HPSv2 `1.2.0`.
- GPU names: `NVIDIA GeForce RTX 4090` and `NVIDIA GeForce RTX 4090`. Peak allocated VRAM remained below the 24 GiB device capacity; no schedule-altering memory workaround was used.
- HPS is HPS v2.1 on final winners only. GenEval is the repository's official Mask2Former/OpenCLIP evaluator on final winners only.
- Statistics: one repetition/base seed 0; 10,000 prompt-level paired percentile bootstrap resamples with bootstrap seed 20260917.

## Published PSP context (reference only)

The paper's full Table-1 SD1.5 values are approximately IR 0.827, HPS 0.278, and GenEval 0.574. This is **reference only**, not an exact Table-1 reproduction: this run uses one repetition and RTX4090 hardware.

## Narrow conclusion

ImageReward is inconclusive (CI contains zero); HPS is inconclusive (CI contains zero); GenEval is inconclusive (CI contains zero). This conclusion is restricted to SD1.5, the full 553-prompt GenEval set, this single repetition, and the two pre-specified schedules. No further schedule was selected.
