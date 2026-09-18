#!/usr/bin/env python3
"""Measure VAE-decode + ImageReward cost against decode chunk size on this GPU.

The bank worker decodes all 25 SMC candidates in one VAE call. That fits the 24 GiB
card used for the published run but not a 14.56 GiB T4, where the decoder's first
upsampling activation alone asks for ~3 GiB more than is free. Decoding in chunks is a
pure execution-schedule knob - the candidate batch, the seeds and the recorded scores
are unchanged - so the only open questions are how large a chunk still fits and what it
costs in latency. This script answers both with measurements instead of a guess.

Run it through kaggle/make_job_spec.py --phase bench-decode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "exps" / "single_stage_calibration"
TEXT = ROOT / "Fk-Diffusion-Steering" / "text_to_image"
for path in (CAL, TEXT, TEXT / "fkd_diffusers"):
    sys.path.insert(0, str(path))

from fkd_diffusers.fkd_pipeline_sd import latent_to_decode  # noqa: E402
from fkd_diffusers.rewards import do_image_reward  # noqa: E402
import generate_bank_worker as bank_worker  # noqa: E402

CHUNKS = (1, 2, 4, 6, 8, 12, 16, 25)
PROMPT = "a red apple on a wooden table, product photo, soft light"


def one_pass(pipe, pool, chunk):
    total = int(pool.shape[0])
    step = chunk if 0 < chunk < total else total
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    started = time.perf_counter()
    scores: list[float] = []
    first_pixels = None
    for start in range(0, total, step):
        piece = pool[start : start + step]
        decoded = latent_to_decode(model=pipe, output_type="pil", latents=piece)
        values = do_image_reward(prompts=[PROMPT] * int(decoded.shape[0]), image_tensors=decoded)
        scores.extend(float(value) for value in values)
        if start == 0:
            first_pixels = decoded[0].detach().float().cpu().clone()
        del decoded
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30
    return elapsed, peak, scores, first_pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="decode_bench.json")
    parser.add_argument("--chunks", default="")
    args = parser.parse_args()
    chunks = tuple(int(item) for item in args.chunks.split(",") if item.strip()) if args.chunks else CHUNKS

    print("cuda", torch.cuda.is_available(), "| devices", torch.cuda.device_count(), flush=True)
    pipe = bank_worker.build_pipeline()
    pool, seeds, hashes = bank_worker.explicit_pool(pipe, 0)
    print("pool", tuple(pool.shape), "| first seeds", seeds[:4], flush=True)

    # Warm up outside the measurements (the published call may not fit here).
    one_pass(pipe, pool[:4], 4)
    torch.cuda.empty_cache()

    reference_pixels = None
    reference_scores = None
    rows = []
    for chunk in chunks:
        try:
            elapsed, peak, scores, pixels = one_pass(pipe, pool, chunk)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            row = {"chunk": chunk, "ok": False, "error": f"OutOfMemoryError: {str(exc)[:160]}"}
            rows.append(row)
            print("KJO_DECODE_BENCH " + json.dumps(row), flush=True)
            continue
        if reference_pixels is None:
            reference_pixels, reference_scores = pixels, scores
            diff = 0.0
        else:
            diff = float((pixels - reference_pixels).abs().max())
        row = {
            "chunk": chunk,
            "ok": True,
            "seconds": round(elapsed, 3),
            "peak_vram_gib": round(peak, 2),
            "candidate_0_reward": round(scores[0], 6),
            "max_abs_pixel_diff_vs_first_row": round(diff, 6),
            "reward_delta_vs_first_row": round(scores[0] - reference_scores[0], 6),
        }
        rows.append(row)
        print("KJO_DECODE_BENCH " + json.dumps(row), flush=True)

    Path(args.out).write_text(json.dumps({"pool_shape": list(pool.shape), "rows": rows}, indent=2) + "\n")
    print("KJO_DECODE_BENCH_SUMMARY " + json.dumps({
        "fits": [row["chunk"] for row in rows if row.get("ok")],
        "fastest_fitting": min((row for row in rows if row.get("ok")), key=lambda row: row["seconds"])["chunk"] if any(row.get("ok") for row in rows) else None,
    }), flush=True)


if __name__ == "__main__":
    main()
