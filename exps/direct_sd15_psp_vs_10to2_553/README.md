# Direct SD1.5 PSP vs 10→2 on all 553 GenEval prompts

This is the pre-specified full-prompt extension of the completed 300-prompt
experiment. It preserves every generation and evaluation setting and expands
coverage to all official GenEval prompt IDs 0…552.

The 300 already-generated compatible prompts are validated and imported from
the tracked predecessor. Both methods are generated only for the remaining 253
prompts. This reuse is exact because prompt trajectories are independent and
their explicit initial latent seeds depend only on prompt and candidate IDs.

Run with:

```bash
bash exps/direct_sd15_psp_vs_10to2_553/run_remote.sh
```
