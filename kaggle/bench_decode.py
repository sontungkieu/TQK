#!/usr/bin/env python3
"""Measure VAE-decode cost against decode chunk size on this GPU.

The bank worker decodes all 25 SMC candidates in one VAE call. That fits the 24 GiB
card of the published run but not a 14.56 GiB T4, where the decoder's first upsampling
activation alone asks for ~3 GiB more than is free. Decoding in chunks is a pure
execution-schedule knob - the candidate batch and the seeds are unchanged - so the only
open questions are how large a chunk still fits and what it costs in latency.

This measures decode alone first (no ImageReward, so a reward-side problem cannot hide
the memory answer), then optionally re-runs the full decode+reward path for the fastest
fitting chunk. Results are written after every row, so a hard failure still leaves
evidence behind.

Run it through kaggle/make_job_spec.py --phase bench-decode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "exps" / "single_stage_calibration"
TEXT = ROOT / "Fk-Diffusion-Steering" / "text_to_image"
for path in (CAL, TEXT, TEXT / "fkd_diffusers"):
    sys.path.insert(0, str(path))

from fkd_diffusers.fkd_pipeline_sd import latent_to_decode  # noqa: E402
import generate_bank_worker as bank_worker  # noqa: E402

CHUNKS = (1, 2, 4, 6, 8, 12, 16, 25)
PROMPT = "a red apple on a wooden table, product photo, soft light"


def decode_only(pipe, pool, chunk):
    total = int(pool.shape[0])
    step = chunk if 0 < chunk < total else total
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    first = None
    for start in range(0, total, step):
        decoded = latent_to_decode(model=pipe, output_type="pil", latents=pool[start : start + step])
        if start == 0:
            first = decoded[0].detach().float().cpu().clone()
        del decoded
    torch.cuda.synchronize()
    return time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**30, first


def decode_and_reward(pipe, pool, chunk):
    from fkd_diffusers.rewards import do_image_reward

    total = int(pool.shape[0])
    step = chunk if 0 < chunk < total else total
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    scores: list[float] = []
    for start in range(0, total, step):
        decoded = latent_to_decode(model=pipe, output_type="pil", latents=pool[start : start + step])
        values = do_image_reward(prompts=[PROMPT] * int(decoded.shape[0]), image_tensors=decoded)
        scores.extend(float(value) for value in values)
        del decoded
    torch.cuda.synchronize()
    return time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**30, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="decode_bench.json")
    parser.add_argument("--chunks", default="")
    parser.add_argument("--with-reward", action="store_true")
    args = parser.parse_args()
    chunks = tuple(int(item) for item in args.chunks.split(",") if item.strip()) if args.chunks else CHUNKS
    report: dict = {"cuda": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "rows": [], "traceback": ""}

    def flush():
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    try:
        print("phase: build pipeline", flush=True)
        pipe = bank_worker.build_pipeline()
        pool, seeds, _hashes = bank_worker.explicit_pool(pipe, 0)
        report["pool_shape"] = list(pool.shape)
        report["first_seeds"] = seeds[:4]
        flush()
        print("phase: warmup", flush=True)
        decode_only(pipe, pool[:2], 2)
        report["warmup"] = "ok"
        flush()

        reference = None
        for chunk in chunks:
            print("phase: decode chunk=" + str(chunk), flush=True)
            try:
                seconds, peak, first = decode_only(pipe, pool, chunk)
                diff = None
                if reference is not None and first is not None:
                    diff = round(float((first - reference["pixels"]).abs().max()), 6)
                elif first is not None:
                    reference = {"chunk": chunk, "pixels": first}
                row = {
                    "chunk": chunk,
                    "ok": True,
                    "seconds": round(seconds, 3),
                    "peak_vram_gib": round(peak, 2),
                    "pixel_delta_vs_chunk" + str(reference["chunk"]): diff,
                }
            except Exception as exc:  # noqa: BLE001 - record and keep going
                torch.cuda.empty_cache()
                row = {"chunk": chunk, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            report["rows"].append(row)
            print("KJO_DECODE_BENCH " + json.dumps(row), flush=True)
            flush()

        fitting = [row for row in report["rows"] if row.get("ok")]
        if fitting:
            best = min(fitting, key=lambda row: row["seconds"])
            report["fastest_fitting_chunk"] = best["chunk"]
            if args.with_reward:
                print("phase: decode+reward chunk=" + str(best["chunk"]), flush=True)
                try:
                    seconds, peak, scores = decode_and_reward(pipe, pool, best["chunk"])
                    report["reward_check"] = {
                        "chunk": best["chunk"], "seconds": round(seconds, 3),
                        "peak_vram_gib": round(peak, 2), "candidate_0_reward": round(scores[0], 6),
                        "candidate_count": len(scores),
                    }
                except Exception as exc:  # noqa: BLE001
                    report["reward_check"] = {"chunk": best["chunk"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            flush()
    except Exception:
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        flush()
        print("KJO_DECODE_BENCH_SUMMARY " + json.dumps({
            "fits": [row["chunk"] for row in report["rows"] if row.get("ok")],
            "fastest_fitting_chunk": report.get("fastest_fitting_chunk"),
            "failed": [row["chunk"] for row in report["rows"] if not row.get("ok")],
            "traceback": bool(report["traceback"]),
        }), flush=True)
    raise SystemExit(0 if report["rows"] and any(row.get("ok") for row in report["rows"]) else 1)


if __name__ == "__main__":
    main()
