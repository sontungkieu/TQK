#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
TEXT = ROOT / "Fk-Diffusion-Steering" / "text_to_image"
for path in (TEXT, TEXT / "fkd_diffusers"):
    sys.path.insert(0, str(path))

from diffusers import DDIMScheduler  # noqa: E402
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion, latent_to_decode  # noqa: E402
from fkd_diffusers.rewards import do_image_reward  # noqa: E402
from schedules import CHECKPOINTS  # noqa: E402

MODEL = "runwayml/stable-diffusion-v1-5"
STEPS = 64
ETA = 0.0
GUIDANCE = 7.5
CANDIDATES = 25
FINAL_STEP = 64
PROMPTS_FILE = EXP / "prompts/calibration_prompts.jsonl"
PROMPT_COUNT = 120
BANK_DIR = EXP / "bank_raw"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def latent_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_prompts(
    path: Path = PROMPTS_FILE, expected_count: int = PROMPT_COUNT
) -> list[dict[str, Any]]:
    """Load a calibration prompt manifest.

    The defaults reproduce the committed 120-prompt phase-1 list exactly. The pre-registered
    200-prompt bank passes prompts200/calibration_prompts.jsonl and 200: that manifest carries a
    deterministic five-fold assignment instead of a search/validation split, so the grouping
    assertions follow whichever grouping key the file actually carries and the groups must be
    equal-sized.
    """
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == expected_count, (len(rows), expected_count)
    assert len({int(row["prompt_id"]) for row in rows}) == expected_count
    assert {int(row["prompt_id"]) for row in rows} == set(range(expected_count))
    if "split" in rows[0]:
        assert expected_count == PROMPT_COUNT
        assert sum(row["split"] == "search" for row in rows) == 80
        assert sum(row["split"] == "validation" for row in rows) == 40
    else:
        folds = sorted({int(row["fold"]) for row in rows})
        assert folds == list(range(len(folds))), folds
        sizes = {fold: sum(1 for row in rows if int(row["fold"]) == fold) for fold in folds}
        assert len(set(sizes.values())) == 1, sizes
    return rows


def _apply_memory_options(pipe):
    """Kaggle/T4 memory knobs; the published 2x RTX 4090 run enabled neither.

    PSP_VAE_SLICING=1 makes the VAE decode the candidate batch one image at a time
    (the diffusers default decodes all candidates at once, which needs more than the
    14.56 GiB of a T4). PSP_ATTENTION_SLICING=1 does the same for UNet attention.
    Both are memory/execution-schedule knobs only: the candidate batch, the seeds and
    the recorded checkpoint scores are unchanged, but decoded pixels can differ in the
    last bits because cuDNN may pick a different convolution algorithm per slice.
    """
    if os.environ.get("PSP_ATTENTION_SLICING", "0") == "1":
        pipe.enable_attention_slicing()
    if os.environ.get("PSP_VAE_SLICING", "0") == "1":
        pipe.enable_vae_slicing()


def build_pipeline():
    pipe = FKDStableDiffusion.from_pretrained(MODEL, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    _apply_memory_options(pipe)
    return pipe


def explicit_pool(pipe, prompt_id: int) -> tuple[torch.Tensor, list[int], list[str]]:
    seeds = [prompt_id * 32 + candidate_id for candidate_id in range(CANDIDATES)]
    generators = [torch.Generator(device="cuda:0").manual_seed(seed) for seed in seeds]
    pool = pipe.prepare_latents(
        batch_size=CANDIDATES,
        num_channels_latents=pipe.unet.config.in_channels,
        height=pipe.unet.config.sample_size * pipe.vae_scale_factor,
        width=pipe.unet.config.sample_size * pipe.vae_scale_factor,
        dtype=pipe.unet.dtype,
        device=torch.device("cuda:0"),
        generator=generators,
        latents=None,
    )
    return pool, seeds, [latent_hash(pool[index]) for index in range(CANDIDATES)]


def score_predicted_clean(pipe, prompt: str, x0_preds: torch.Tensor) -> list[float]:
    """Decode and score the whole candidate batch, optionally in chunks.

    The published run decoded all 25 candidates in one VAE call on 24 GiB cards; on a
    14.56 GiB T4 that call needs about 3 GiB more than is free. PSP_VAE_CHUNK decodes a
    smaller slice at a time (PSP_VAE_CHUNK=0 keeps the published single call). Only the
    execution schedule changes: the candidate batch, the seeds and the score order are
    untouched, while decoded pixels can differ in the last bits because cuDNN may select
    a different convolution algorithm per slice.
    """
    chunk = int(os.environ.get("PSP_VAE_CHUNK", "0") or 0)
    total = int(x0_preds.shape[0])
    step = chunk if 0 < chunk < total else total
    scores: list[float] = []
    for start in range(0, total, step):
        decoded = latent_to_decode(
            model=pipe, output_type="pil", latents=x0_preds[start : start + step]
        )
        values = do_image_reward(
            prompts=[prompt] * int(decoded.shape[0]), image_tensors=decoded
        )
        scores.extend(float(value) for value in values)
        del decoded
    return scores


def generate_prompt(pipe, row: dict[str, Any], worker_index: int) -> dict[str, Any]:
    prompt_id = int(row["prompt_id"])
    prompt = row["prompt"]
    pool, seeds, hashes = explicit_pool(pipe, prompt_id)
    generators = [torch.Generator(device="cuda:0").manual_seed(seed) for seed in seeds]
    score_by_step: dict[int, list[float]] = {}
    score_elapsed_by_step: dict[int, float] = {}
    requested = set(CHECKPOINTS) | {FINAL_STEP}

    def callback(_pipe, step_index, _timestep, kwargs):
        step = int(step_index) + 1
        if step not in requested:
            return {"latents": kwargs["latents"]}
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = score_predicted_clean(pipe, prompt, kwargs["x0_preds"])
        torch.cuda.synchronize()
        assert len(scores) == CANDIDATES
        score_by_step[step] = scores
        score_elapsed_by_step[step] = time.perf_counter() - started
        return {"latents": kwargs["latents"]}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        pipe(
            prompt=[prompt] * CANDIDATES,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            eta=ETA,
            generator=generators,
            latents=pool,
            output_type="latent",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "x0_preds"],
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert set(score_by_step) == requested
    candidates = []
    for candidate_id in range(CANDIDATES):
        candidate = {
            "candidate_id": candidate_id,
            "seed": seeds[candidate_id],
            "initial_latent_sha256": hashes[candidate_id],
            **{
                f"IR_t{step}": score_by_step[step][candidate_id]
                for step in CHECKPOINTS
            },
            "IR_final": score_by_step[FINAL_STEP][candidate_id],
        }
        candidates.append(candidate)
    return {
        **row,
        "worker_index": worker_index,
        "model": MODEL,
        "dtype": "torch.float16",
        "scheduler": type(pipe.scheduler).__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "steps": STEPS,
        "eta": ETA,
        "guidance_scale": GUIDANCE,
        "candidate_count": CANDIDATES,
        "candidate_seed_formula": "prompt_id*32 + candidate_id",
        "elapsed_s": elapsed,
        "checkpoint_decode_reward_s": sum(score_elapsed_by_step.values()),
        "score_elapsed_by_step": score_elapsed_by_step,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "candidates": candidates,
    }


def complete(prompt_id: int, worker_index: int, bank_dir: Path = BANK_DIR) -> bool:
    path = bank_dir / f"gpu{worker_index}" / f"{prompt_id:05d}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        requested = {f"IR_t{step}" for step in CHECKPOINTS} | {"IR_final"}
        return (
            int(payload["prompt_id"]) == prompt_id
            and len(payload["candidates"]) == CANDIDATES
            and all(requested <= set(candidate) for candidate in payload["candidates"])
        )
    except Exception:
        return False


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def warm_up(pipe, prompt: str) -> None:
    generator = [torch.Generator(device="cuda:0").manual_seed(987654)]
    with torch.inference_mode():
        pipe(prompt=[prompt], num_inference_steps=2, eta=0.0, generator=generator, output_type="latent")
    do_image_reward(images=[Image.new("RGB", (224, 224))], prompts=[prompt])
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prompts", type=Path, default=PROMPTS_FILE,
        help="prompt manifest jsonl; defaults to the committed 120-prompt phase-1 list",
    )
    parser.add_argument(
        "--expected-prompts", type=int, default=PROMPT_COUNT,
        help="exact prompt count the manifest must contain (200 for the pre-registered bank)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=BANK_DIR,
        help="bank root receiving gpu<worker>/<prompt_id>.json; defaults to bank_raw",
    )
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")
    prompts = load_prompts(args.prompts, args.expected_prompts)
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    owned = [row for row in prompts if int(row["prompt_id"]) % args.num_workers == args.worker_index]
    if not owned:
        print(json.dumps({"worker": args.worker_index, "completed": 0, "owned": 0}))
        return
    pipe = build_pipeline()
    warm_up(pipe, owned[0]["prompt"])
    import diffusers
    import transformers

    hardware = {
        "worker": args.worker_index,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "image_reward": package_version("image-reward"),
        "python": platform.python_version(),
        "owned_prompt_ids": [int(row["prompt_id"]) for row in owned],
    }
    atomic_json(args.out_dir / f"gpu{args.worker_index}" / "hardware.json", hardware)
    completed = 0
    for row in owned:
        prompt_id = int(row["prompt_id"])
        if args.resume and complete(prompt_id, args.worker_index, args.out_dir):
            completed += 1
            continue
        result = generate_prompt(pipe, row, args.worker_index)
        atomic_json(args.out_dir / f"gpu{args.worker_index}" / f"{prompt_id:05d}.json", result)
        print(json.dumps({
            "phase": "calibration_bank",
            "prompt_id": prompt_id,
            "elapsed_s": result["elapsed_s"],
            "scoring_s": result["checkpoint_decode_reward_s"],
            "peak_vram_gib": result["peak_vram_gib"],
        }), flush=True)
        completed += 1
    print(json.dumps({"worker": args.worker_index, "completed": completed, "owned": len(owned)}), flush=True)


if __name__ == "__main__":
    main()

