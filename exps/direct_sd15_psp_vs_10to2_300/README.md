# Direct SD1.5 PSP vs 10→2

This directory contains the pre-specified, single-run comparison of official PSP
`8→4@16→2@32` against `10→2@16` on a fixed balanced 300-prompt GenEval subset.
Both schedules use 256 logical UNet evaluations, deterministic 64-step DDIM
(`eta=0`), ImageReward pruning, and paired explicit initial latent pools.

The tracked remote entry point is:

```bash
bash exps/direct_sd15_psp_vs_10to2_300/run_remote.sh
```

It performs the one-prompt trace, the eight-prompt two-GPU smoke test, the full
generation, HPS v2.1 scoring, official GenEval evaluation, paired bootstrap, and
final report generation. It is resume-safe at the prompt/method level.

The intended hardware is exactly two NVIDIA GeForce RTX 4090 GPUs. Prompt
ownership is `sorted_subset_position % 2`; both methods for a prompt remain on
the same GPU. No DDP, NCCL, trajectory sharing, offline replay, or schedule
tuning is used.

Primary outputs are `REPORT.md` and `metrics/`. Exact prompt selection is stored
in `prompt_ids_300.json` and `prompts_geneval_balanced_300.jsonl`.
