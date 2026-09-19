# Running TQK on Kaggle

This directory adds a thin Kaggle layer around the published experiment. It does not
change the scientific protocol: the prompt manifests, schedules, workers, validators and
frozen schedule stay exactly as published.

## Why this layer exists

```text
exps/single_stage_calibration/run_all.sh   asserts exactly 2x "NVIDIA GeForce RTX 4090"
                                           and pre-built venvs under /workspace
Kaggle (GPU T4 x2)                         2x Tesla T4, 14.56 GiB each, sm75, fresh container
```

* The upstream driver is single-machine and 4090-specific, so it cannot be the control
  plane here. `run_all.sh` stays untouched as the paper's reproduction script; the Kaggle
  layer owns code preparation, environment, sharding and artifact transport.
* Both phase workers already support sharding and resume:
  `generate_bank_worker.py` and `run_worker.py` take
  `--worker-index i --num-workers N --resume [--limit-prompts K]`.

## Environment: uv, isolated from the Kaggle image

`Fk-Diffusion-Steering/requirements.txt` is a flat pip freeze (python 3.10, torch 2.4.0,
transformers 4.38.2, numpy 1.26.3, protobuf 3.20.3, of which the Kaggle image already ships
torch 2.10.0+cu128). Installing it into the notebook interpreter would break the image.

Instead `pyproject.toml` + `uv.lock` at the repository root build a project-local
`.venv`, and KJO Job Spec schema v2 runs every step through `.venv/bin/python`:

* `tools/derive_pyproject.py` transcribes the freeze verbatim (76 dependencies, including
  the two editable git requirements) so the environment matches the published one instead
  of re-resolving newer versions;
* `[tool.uv.extra-build-dependencies]` pins `setuptools==69.5.1` for ImageReward, whose
  legacy `setup.py` imports `pkg_resources` — the same workaround as `setup/setup.sh`;
* `mode = "locked"` requires an unchanged `pyproject.toml`/`uv.lock` and runs
  `uv sync --locked`, so a run is reproducible or it fails.

Operational consequences:

* uv installation, python acquisition and package sync need **internet**;
* the uv cache and `.venv` live under `/tmp` and are **not** result artifacts, so every
  session re-syncs (roughly 2.5 GB, a few minutes). Budget the session long enough to
  amortize that, and keep shards sized in hours, not minutes.

## Compatibility gate

`kaggle/check_env.py` is the first step of every Job Spec. It fails the run before any GPU
work when:

1. the interpreter is not 3.10.x;
2. any pin from the published freeze is missing or at a different version;
3. either git requirement resolves to a commit other than the pinned one
   (checked through PEP 610 `direct_url.json`, not just the version string);
4. CUDA devices are missing when `--require-cuda --expect-gpus N` is set;
5. `fkd_diffusers.fkd_pipeline_sd`, `fkd_diffusers.rewards` or `schedules` fail to import;
6. the hardcoded `runwayml/stable-diffusion-v1-5` weights are unreachable.

It writes `env_check.json` and prints a single `KJO_ENV_CHECK {...}` line.

## Phase 1 result on T4 (this deployment)

Independent calibration on 2x T4, 2026-09-18 (kernel `codemaivanngu/tqk-bank-p1-260918-0731`):
120 prompts x 25 trajectories, **11.34 GPU-h** inside a **6.0 h** session, peak **7.41 GiB**,
mean **340 s/prompt** (99.5 s of that decode + ImageReward).

The 80-prompt search split ranked `9→2@17` first (final IR 0.9336) and the untouched
40-prompt validation split selected it again (final IR 0.5881). That is the **same schedule
the published 2x RTX 4090 run froze**, so the committed
`exps/single_stage_calibration/FROZEN_SCHEDULE.json` is left untouched and
`prepare_protocol.py` passes unchanged. Only provenance differs - search rank 1 vs 3 and
validation IR 0.5881 vs 0.5974 - which is what different fp16 kernels on T4 vs 4090 should
produce. `kaggle/evidence/phase1_t4_freeze.json` records this run's numbers, the top-5
search table and the bank SHA-256.

## Phase 2 (GenEval-553 validation)

`exps/single_stage_validation_553/` compares fixed PSP with the frozen schedule produced
by phase 1. `kaggle/make_job_spec.py --phase eval` renders the whole chain as one Job
Spec:

| step | what it does |
| --- | --- |
| `env-check` | uv environment + CUDA gate |
| `prepare-protocol` | locks the 553 GenEval prompts and copies the frozen schedule |
| `budget-check` | re-asserts `M*checkpoint + K*(64-checkpoint) <= 256` |
| `shard` | `run_shard.sh --phase eval`, same worker-index sharding on 2 GPUs |
| `validate` | `validate_generation.py --expected-prompts N` (N = `--limit-prompts` or 553) |
| `hps` | optional (`--with-hps`), HPS v2.1 on the 1106 final winners, pinned to `CUDA_VISIBLE_DEVICES=0` |
| `export-geneval` | writes the official GenEval input layout |
| `install-geneval` | builds the official GenEval evaluator under `/tmp` and smoke-tests its import chain |
| `evaluate-geneval` | runs `geneval/evaluation/evaluate_images.py` + `summary_scores.py` per method, then deletes the `/tmp` environment |
| `pack-artifacts` | one `artifacts.tar.gz` (outputs, metadata, metrics, geneval inputs/results, manifests) |

The last three steps exist because the evaluation has to happen **inside this session** (see
GenEval below). Nothing bulky is written to `/kaggle/working` except that single archive.

Three operational rules matter:

* **Phase 2 consumes phase 1.** `prepare_protocol.py` copies
  `exps/single_stage_calibration/FROZEN_SCHEDULE.json` and refuses to overwrite a
  different `exps/single_stage_validation_553/FROZEN_SCHEDULE.json`. Commit the schedule
  produced by `analyze_bank.py` (and refresh or delete the stale validation copy) in the
  same commit the Job Spec pins, otherwise that step fails by design.
* **One session may not be enough.** Phase 2 is 553 prompts x 2 methods. Run
  `--limit-prompts 4` first, exactly like the upstream preflight, and shard the full run
  with `--num-workers N` where each session takes two worker indices.
* **HPS needs the whole run.** `evaluate_hps.py` iterates all 553 prompts and opens
  `outputs/<method>/<prompt_id>.png`, so it cannot run on a limited preflight;
  `make_job_spec.py` now rejects `--with-hps --limit-prompts`. The preflight therefore
  covers env-check, prepare-protocol, budget-check, generation and validation only,
  exactly like the `run_all.sh` preflight.

* **GenEval stays in the generating session.** `geneval/evaluation/evaluate_images.py`
  asserts CUDA, and `export_geneval.py` writes `geneval_inputs/<method>/<id>.png` as
  **symlinks** into the session's `outputs/` tree, so the scores cannot be produced from a
  downloaded copy without regenerating that tree. `install-geneval` +
  `evaluate-geneval` therefore run at the end of the same job, and they delete the `/tmp`
  environment again so no artifact grows by a few GB.

## GenEval environment (in-session)

`kaggle/install_geneval_env.sh` mirrors `geneval/environment.yml` on the CUDA 12.1 path,
where `torch==2.1.2` has a **prebuilt** `mmcv-full==1.7.2` wheel, so nothing is compiled:

* the evaluator imports `numpy`, `pandas`, `PIL`, `torch`, `mmdet.apis` (Mask2Former
  Swin-S), `open_clip` and `clip_benchmark.metrics.zeroshot_classification`;
* **numpy must stay on 1.x.** An unpinned resolve picks numpy 2.2.6 + opencv-python 5.x and the
  mmcv C extension then fails at import with `numpy.core.multiarray failed to import`;
* mmdet 2.x is installed as a `.pth` source entry (pure Python; the CUDA ops live in mmcv),
  which skips its `install_requires`, so its runtime requirements (`matplotlib`, `scipy`,
  `six`, `terminaltables`, `pycocotools`) are pinned explicitly;
* `clip-benchmark` must be `>=1.5`: the `environment.yml` pin 1.4.0 declares `torch<2` and
  cannot resolve against torch 2.1.2, while `zeroshot_classification` keeps the same
  five-positional-argument API the evaluator calls;
* everything lives under `/tmp`, installs run with `--no-cache`, and the whole run writes back
  only `geneval_stages.txt` (~2 KB). Each stage line records free space and environment size,
  and a failure appends the tail of `/tmp/geneval_install.log`, so one small fetch is enough to
  tell disk exhaustion apart from an install error.

`kaggle/make_job_spec.py --phase geneval-build` is the **CPU preflight** for that installer:
it builds and smoke-tests the environment without spending GPU quota. Run it before an
evaluation session so a broken evaluator cannot burn six hours of generation time.

Only HPS is covered by the project pins (`hpsv2==1.2.0`, and `evaluate_hps.py` already shims
the headless `turtle` import).

## T4 memory profile

Both workers batch all 25 SMC candidates through the UNet and then decode them in one
VAE call. That fits the 24 GiB card used for the published run, but not a 14.56 GiB T4,
where the decoder's first upsampling activation alone asks for ~3 GiB:

```text
torch.OutOfMemoryError: ... Tried to allocate 3.12 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 3.06 GiB is free.
```

```bash
kaggle/run_shard.sh --phase bank --num-workers 2 --worker-indices 0,1
# exports: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, PSP_VAE_SLICING=1
```

Measured on a T4 by `kaggle/bench_decode.py` (decode of the 25-candidate pool, 512px):

| decode chunk | fits | seconds | peak VRAM |
| ---: | :---: | ---: | ---: |
| 1 | yes | 4.64 | 5.43 GiB |
| 2 | yes | 6.69 | 8.67 GiB |
| 4 | **no** | - | OOM (14.52 GiB in use) |
| 6, 8, 12, 16, 25 | **no** | - | OOM |

So chunk 1 is not a conservative default, it is the fastest fitting option: chunk 2 fits
but is 44% slower, and everything from 4 up dies. A full decode+reward pass for the
pool costs 50.6 s cold, most of which is the one-time ImageReward load.

`PSP_VAE_CHUNK` is read by `score_predicted_clean()` in `generate_bank_worker.py`: it
decodes and scores the candidate batch in slices of that size, and `0` keeps the
published single call. `run_worker.py` needs no knob - phase 2 decodes at most 8
candidates per checkpoint - but `_apply_memory_options()` in both workers still honours
`PSP_VAE_SLICING` and `PSP_ATTENTION_SLICING` for extreme cases.

Chunking changes the execution schedule only: the candidate batch, the seeds and the 14
recorded scores per candidate are unchanged. Decoded pixels can differ in the last bits
because cuDNN may select a different convolution algorithm per slice, so a bank produced
with chunking is not guaranteed bit-identical to one produced on a 24 GiB card. Record
which profile produced a bank when comparing across hosts.

`kaggle/bench_decode.py` (Job Spec phase `bench-decode`) measures decode+reward latency,
peak VRAM and cross-chunk pixel deltas for chunk sizes 1, 2, 4, 6, 8, 12, 16 and 25 on
the real pool, so the chunk size is chosen from data instead of guessed.

## Sharding

Workers partition prompt ids deterministically by `(worker_index, num_workers)`:

```python
owned = [row for row in prompts if int(row["prompt_id"]) % args.num_workers == args.worker_index]
```

Each worker requires exactly one visible GPU (`torch.cuda.device_count() != 1` raises), which
is why the runner sets one `CUDA_VISIBLE_DEVICES` per worker. `--limit-prompts K` is applied
before the partition, so it truncates the shared prompt list, not the shard.

So a shard is exactly `(phase, num_workers, worker_indices)`, and shards from different
sessions are disjoint as long as `num_workers` is identical across them.

```bash
# one Kaggle session with 2x T4 running 2 of 8 shards
kaggle/run_shard.sh --phase bank --num-workers 8 --worker-indices 4,5

# cheap plumbing check, no GPU work
kaggle/run_shard.sh --phase bank --num-workers 4 --worker-indices 0,1 --dry-run
```

## Session sizing and quota

Kaggle counts quota in **wall-clock session time**, not GPU-hours, so a 2x T4 session costs
the same as a 1x T4 session of the same duration.

| phase | work | measured session on 2x T4 |
| --- | --- | --- |
| bank | 120 prompts x 25 trajectories x 64 DDIM steps | 21 559.6 s (5.99 h), 11.34 GPU-h, peak 7.41 GiB |
| eval | 553 prompts x 2 methods + HPS | 24 394.7 s (6.78 h) — `shard` 24 094.9 s, `hps` 211.1 s |

Add the GenEval install and two evaluation passes to an `eval` session: budget ~8 h and keep
the 12 h session ceiling in mind. Estimates scale ~13.3 s per 64-step 512px SD1.5 trajectory
measured on T4 (25 steps = 5.2 s under `attention_slicing`); phase 2 measured 40.5 s per
generation (n=8, 38.0-45.1 s). A larger calibration multiplies the bank phase: raise
`--num-workers` and hand each session the next two worker indices.

## Merge, validate, freeze

```bash
kaggle/merge_shards.py --input run_a/output --input run_b/output \
  --dest exps/single_stage_calibration/bank_raw --expect-prompts 120

python exps/single_stage_calibration/validate_bank.py --expected-prompts 120
python exps/single_stage_calibration/analyze_bank.py | tee exps/single_stage_calibration/logs/analysis.log
python exps/single_stage_calibration/plot_results.py  | tee exps/single_stage_calibration/logs/plots.log
```

`validate_bank.py` asserts the exact prompt set `0..N-1`, so it only passes once every
shard is merged. `merge_shards.py` refuses conflicting payloads for the same prompt id and
reports duplicates instead of silently overwriting.

## KJO Job Spec

```bash
kaggle/make_job_spec.py --phase smoke-cpu --out /tmp/tqk_smoke.json
kaggle/make_job_spec.py --phase bank --num-workers 8 --worker-indices 4,5 --out /tmp/bank_45.json

python3 "$KJO" validate-job-spec --job-spec /tmp/bank_45.json
python3 "$KJO" stage-job-package --job-spec /tmp/bank_45.json --run-dir <run_dir> --owner kieutung \
  --runtime-dataset-source kieutung/kjo-runtime-0-12-0 \
  --runtime-module-sha256 c7935afe2d7fe51afa2e47c93bd7af19d59a1ace51b6cc27686fec86ac6930c3
```

`make_job_spec.py` stamps `code_source.commit` with `git rev-parse HEAD` of the checkout,
so the spec always names the pushed commit that contains the code it will run.

## Downloading artifacts

Bank payloads are plain `.json` (one file per prompt, roughly 25 candidates x 14 scores
plus timing), so the KJO download purge rules for `.pkl/.npz/.dat/.zip` do not touch them.
The default KJO download pattern is diagnostics-only, so pull shard output explicitly:

```bash
python3 "$KJO" download-kernel-output --run-dir <run_dir> --kernel-id <owner>/<slug> \
  --kind all --all --mark-downloaded
```

**Kaggle rate-limits the listing call, per IP, not per credential.**
`kernels.KernelsApiService/ListKernelSessionOutput` starts answering HTTP 429
(`Too Many Requests`) after repeated large fetches, and another account's credential does not
help. So:

* fetch an `eval` run through its single `artifacts.tar.gz` (`pack-artifacts` step) instead of
  the raw tree — an `eval` output holds 1106 PNGs plus metadata;
* narrow every ad-hoc fetch with `--file-pattern` and combine artifacts into one regex
  (for example `'.*(geneval_stages\.txt|[^/]*\.log)$'`) instead of making several calls;
* space attempts out. A tight retry loop keeps the window open; a quiet period of tens of
  minutes usually clears it.

## Caveats

* `runwayml/stable-diffusion-v1-5` was withdrawn from the Hub and now redirects to
  `stable-diffusion-v1-5/stable-diffusion-v1-5`. Both workers hardcode the old id and
  `validate_bank.py` asserts it, so **do not rewrite the string**; `check_env.py` verifies
  the id is still reachable before any GPU time is spent.
* GenEval is a separate environment (`geneval/environment.yml`, mmdetection v2.28.2,
  Mask2Former weights) and is deliberately outside the uv project. `install_geneval_env.sh`
  builds it under `/tmp` on the torch 2.1.2 + prebuilt mmcv-full 1.7.2 path and
  `--phase geneval-build` preflights that build on CPU; it is still not part of the pinned uv
  ``pyproject.toml`/`uv.lock` contract, so treat a change to the evaluator as a change to
  this layer, not to the published environment.
* `run_all.sh` (single machine, 2x RTX 4090) remains the canonical end-to-end reproduction.
  This layer reproduces the same steps as shardable Kaggle jobs.
