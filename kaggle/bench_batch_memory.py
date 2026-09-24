#!/usr/bin/env python3
"""Does torch.compile raise the batch sizes that had to be reduced? (T4, SD1.5 fp16)

Two memory facts from the project: the bank failed to decode the 25-candidate pool in one VAE
call (activation ~3.12 GiB with 3.06 GiB free) and was chunked to 1, while the UNet runs the
whole pool at a 7.41 GiB peak. The speedup bench showed compile_reduce cutting UNet peak VRAM
from 3.04 to 2.03 GiB via a static CUDA-graph pool.

This measures, per configuration, the largest batch that still fits:
  part A - VAE decode N in {1,2,4,8,16,25}: baseline / tiling / compiled decoder / tiling+compiled
  part B - UNet forward N in {8,16,25,32}: baseline / compile_reduce
Each run reports seconds, peak allocated VRAM, and OOM as a recorded outcome rather than a
crash; results are written after every configuration.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

MODEL = "runwayml/stable-diffusion-v1-5"
PROMPT = "a photo of a red bench and a blue car"
OUT = Path("/kaggle/working/batch_memory.json")
DECODE_BATCHES = (1, 2, 4, 8, 16, 25)
UNET_BATCHES = (8, 16, 25, 32)
LATENT_SHAPE = (4, 64, 64)


def measure(fn):
    """Run fn once, reporting seconds, peak VRAM or the OOM/failure as data."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - started
        return {"ok": True, "elapsed_s": elapsed,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3)}
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "error": "OutOfMemoryError"}
    except Exception:
        return {"ok": False, "error": traceback.format_exc().strip().splitlines()[-1][:160]}


def build():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    )
    return pipe.to('cuda')


def main() -> None:
    print('[batch] torch {0} | {1}'.format(torch.__version__, torch.cuda.get_device_name(0)))
    pipe = build()
    results: dict = {}

    with torch.no_grad():
        embeds, _ = pipe.encode_prompt(PROMPT, 'cuda', 1, False)
        for batch in DECODE_BATCHES:
            latents = torch.randn((batch,) + LATENT_SHAPE, device='cuda', dtype=torch.float16)
            for label, setup in (
                ('baseline', lambda: None),
                ('tiling', lambda: pipe.vae.enable_tiling()),
                ('compiled', lambda: setattr(pipe.vae, 'decoder', torch.compile(pipe.vae.decoder, mode='reduce-overhead'))),
                ('tiling+compiled', lambda: (pipe.vae.enable_tiling(), setattr(pipe.vae, 'decoder', torch.compile(pipe.vae.decoder, mode='reduce-overhead')))),
            ):
                key = 'decode_n{0}/{1}'.format(batch, label)
                try:
                    pipe.vae.disable_tiling()
                    pipe.vae.decoder = pipe.vae._orig_decoder if hasattr(pipe.vae, '_orig_decoder') else pipe.vae.decoder
                    if not hasattr(pipe.vae, '_orig_decoder'):
                        pipe.vae._orig_decoder = pipe.vae.decoder
                    setup()
                    scale = getattr(pipe.vae.config, 'scaling_factor', 0.18215)
                    record = measure(lambda: pipe.vae.decode(latents / scale).sample)
                except Exception:
                    record = {'ok': False, 'error': traceback.format_exc().strip().splitlines()[-1][:160]}
                results[key] = record
                print('[batch] {0}: {1}'.format(key, json.dumps(record)[:160]))
                OUT.write_text(json.dumps(results, indent=2) + chr(10))
        for batch in UNET_BATCHES:
            sample = torch.randn((batch,) + LATENT_SHAPE, device='cuda', dtype=torch.float16)
            hidden = embeds.repeat(batch, 1, 1)
            timestep = torch.tensor(500, device='cuda')
            for label, setup in (
                ('baseline', lambda: setattr(pipe.unet, 'forward', pipe.unet._orig_forward)),
                ('compile_reduce', lambda: setattr(pipe.unet, 'forward', torch.compile(pipe.unet._orig_forward, mode='reduce-overhead'))),
            ):
                if not hasattr(pipe.unet, '_orig_forward'):
                    pipe.unet._orig_forward = pipe.unet.forward
                key = 'unet_n{0}/{1}'.format(batch, label)
                try:
                    setup()
                    record = measure(lambda: pipe.unet(sample, timestep, encoder_hidden_states=hidden).sample)
                except Exception:
                    record = {'ok': False, 'error': traceback.format_exc().strip().splitlines()[-1][:160]}
                results[key] = record
                print('[batch] {0}: {1}'.format(key, json.dumps(record)[:160]))
                OUT.write_text(json.dumps(results, indent=2) + chr(10))
    lines = ['# Batch/VRAM ceiling on T4 (SD1.5 512px fp16)', '']
    for key in sorted(results):
        rec = results[key]
        if rec.get('ok'):
            lines.append('- {0}: ok {1:.2f} s, peak {2:.2f} GiB'.format(key, rec['elapsed_s'], rec['peak_vram_gib']))
        else:
            lines.append('- {0}: FAILED {}'.format(key).replace('{}', rec.get('error', '?')))
    Path('/kaggle/working/batch_memory.md').write_text(chr(10).join(lines) + chr(10))
    print('[batch] wrote', OUT)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('[batch] FAILED' + chr(10) + traceback.format_exc())
        raise
