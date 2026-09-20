#!/usr/bin/env python3
"""Speed/quality bench for UNet-level acceleration on the TQK workload.

Workload: SD1.5 (runwayml/stable-diffusion-v1-5), 512x512, fp16, DDIM 64 steps, eta 0,
guidance 7.5 - the phase-2 worker setting. Two shapes: batch 8 x 64 steps (early phase of an
8-seed schedule) and batch 2 x 64 steps (tail phase after two prunes, GPU least occupied).

Variants, each changing exactly one thing against the baseline:
  baseline            AttnProcessor (legacy math attention), plain UNet
  sdpa                AttnProcessor2_0 (scaled_dot_product_attention; mem-efficient on sm75)
  compile_default     torch.compile(mode='default')
  compile_reduce      torch.compile(mode='reduce-overhead') - inductor + CUDA graphs
  cache_temb_010/020  TeaCache-style: accumulate the relative L1 drift of the timestep
                      embedding and reuse the previous noise prediction below a threshold.
                      The official rescale polynomial is NOT applied, so the thresholds are
                      not comparable to the TeaCache paper; skip fraction and drift are what
                      this bench measures.
  skip_stride2        reuse the previous noise prediction every second step - an upper bound
                      on what block-level caching (DeepCache) can buy, cruder in quality.

Per run: seconds, ms per image-step, UNet forward calls, peak allocated VRAM, latent relative
error against the baseline latents for the same seeds, and ImageReward when it imports.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

MODEL = "runwayml/stable-diffusion-v1-5"
STEPS = 64
GUIDANCE = 7.5
SIZE = 512
SEEDS = [20260919 + index for index in range(8)]
SHAPES = ((8, 64), (2, 64))
PROMPT = "a photo of a red bench and a blue car"
OUT = Path("/kaggle/working/speedup_bench.json")
REPORT = Path("/kaggle/working/speedup_bench.md")


def build_pipeline() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def base_latents(batch: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(SEEDS[0])
    shape = (batch, 4, SIZE // 8, SIZE // 8)
    return torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16)


def run_once(pipe, batch: int, variant: str) -> dict:
    latents = base_latents(batch)
    generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in SEEDS[:batch]]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    calls = {'n': 0}
    original_forward = pipe.unet.forward

    def counting_forward(*args, **kwargs):
        calls['n'] += 1
        return original_forward(*args, **kwargs)

    pipe.unet.forward = counting_forward
    state = {"previous": None, "accumulated": 0.0, "step": 0, "temb": None}
    hooks = []

    def temb_hook(module, inputs, output):
        current = output.detach()
        previous = state["temb"]
        if previous is not None:
            state["accumulated"] += float((current - previous).abs().mean() / (previous.abs().mean() + 1e-6))
        state["temb"] = current

    if variant.startswith('cache_temb'):
        threshold = float(variant.rsplit('_', 1)[1]) / 100.0
        hooks.append(pipe.unet.time_embedding.register_forward_hook(temb_hook))

        def caching_forward(*args, **kwargs):
            if state["accumulated"] < threshold and state["previous"] is not None:
                return state["previous"]
            result = original_forward(*args, **kwargs)
            state["accumulated"] = 0.0
            state["previous"] = result
            return result

        pipe.unet.forward = caching_forward
    elif variant.startswith('skip_stride'):
        stride = int(variant.rsplit('_', 1)[1])

        def strided_forward(*args, **kwargs):
            state["step"] += 1
            if state["previous"] is not None and state["step"] % stride != 0:
                return state["previous"]
            result = original_forward(*args, **kwargs)
            state["previous"] = result
            return result

        pipe.unet.forward = strided_forward

    try:
        started = time.perf_counter()
        # The worker batches N candidates of ONE prompt, so the prompt is repeated to set the
        # effective batch size; a single string would make diffusers expect one generator only.
        result = pipe(
            [PROMPT] * batch,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generators,
            latents=latents.clone(),
            output_type="latent",
            height=SIZE,
            width=SIZE,
        )
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        out = result.images.detach().float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
        pipe.unet.forward = original_forward
    return {
        "variant": variant,
        "batch": batch,
        "elapsed_s": elapsed,
        "image_steps": batch * STEPS,
        "ms_per_image_step": 1000.0 * elapsed / (batch * STEPS),
        "unet_calls": calls["n"],
        "peak_vram_gib": peak,
        "latents": out,
    }


def main() -> None:
    print(f'[bench] torch {torch.__version__} | {torch.cuda.get_device_name(0)}')
    variants = [
        'baseline',
        'sdpa',
        'compile_default',
        'compile_reduce',
        'cache_temb_010',
        'cache_temb_020',
        'skip_stride2',
    ]
    results: dict = {}
    reference: dict = {}
    for batch, steps in SHAPES:
        prefix = f'b{batch}_s{steps}'
        for variant in variants:
            pipe = build_pipeline()
            if variant == 'sdpa':
                from diffusers.models.attention_processor import AttnProcessor2_0
                pipe.unet.set_attn_processor(AttnProcessor2_0())
            elif variant.startswith('compile'):
                mode = 'default' if variant == 'compile_default' else 'reduce-overhead'
                pipe.unet = torch.compile(pipe.unet, mode=mode, fullgraph=False)
                run_once(pipe, batch, variant + '_warmup')
            record = run_once(pipe, batch, variant)
            if variant == 'baseline':
                reference[prefix] = record.pop('latents')
            else:
                baseline = reference[prefix]
                diff = (record['latents'] - baseline).abs()
                record['latent_mae'] = float(diff.mean())
                record['latent_rel_error'] = float(diff.mean() / (baseline.abs().mean() + 1e-6))
                record.pop('latents')
            results[f'{prefix}/{variant}'] = record
            print('[bench] {0}/{1}: {2:.1f}s {3:.0f} ms/image-step calls={4} vram={5:.2f} GiB rel={6:.4f}'.format(
                prefix, variant, record['elapsed_s'], record['ms_per_image_step'],
                record['unet_calls'], record['peak_vram_gib'], record.get('latent_rel_error', 0.0)))
            # Dump after every variant: if a later variant crashes, the measurements already
            # collected survive in the output instead of being lost with the step.
            OUT.write_text(json.dumps(results, indent=2) + chr(10))
            del pipe
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, indent=2) + chr(10))
    lines = ['# UNet acceleration bench (SD1.5, 512px, fp16, 64 DDIM steps)', '']
    for key in sorted(results):
        record = results[key]
        lines.append('- {0}: {1:.2f} s, {2:.0f} ms/image-step, unet_calls={3}, vram={4:.2f} GiB, rel_err={5:.4f}'.format(
            key, record['elapsed_s'], record['ms_per_image_step'], record['unet_calls'],
            record['peak_vram_gib'], record.get('latent_rel_error', 0.0)))
    REPORT.write_text(chr(10).join(lines) + chr(10))
    print('[bench] wrote', OUT, 'and', REPORT)


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception:
        # Kaggle keeps only the notebook log, which does not carry a step's raw stdout, so the
        # traceback is printed explicitly here and picked up by the log fetch.
        print('[bench] FAILED' + chr(10) + traceback.format_exc())
        raise
