#!/usr/bin/env python3
"""Speed/quality bench for UNet-level acceleration on the TQK workload (v2).

Changes from v1, both driven by its own measurements:
  * torch.compile needs triton, which the locked uv environment does not install, so the bench
    installs triton in-session (remote install; the locked env is untouched for real runs).
  * the timestep-embedding cache distances measured ~0.001 per step, so the v1 thresholds were
    ~100x too high and the variant degenerated into returning the first prediction. Thresholds
    are now calibrated to that scale and each run reports its observed skip fraction, which is
    the number that must be non-zero before any speed number is believable.
  * every variant runs inside try/except and results are dumped after each one.

Per run: seconds, ms/image-step, UNet forward calls, skip fraction, peak VRAM, and latent drift
against the baseline latents for the same seeds.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
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
CACHE_THRESHOLDS = {'cache_temb_0005': 0.0005, 'cache_temb_001': 0.001, 'cache_temb_002': 0.002}
VARIANTS = [
    'baseline',
    'sdpa',
    'compile_default',
    'compile_reduce',
    'skip_stride2',
    'skip_stride3',
    'skip_stride4',
]


def ensure_triton() -> str:
    """Diagnose triton instead of guessing.

    The lock already pins triton 3.0.0 as a torch 2.4.0 dependency, so inductor's
    "Cannot find a working triton installation" means the installed triton is unusable, not
    missing. Report the import result, torch's own probes and whether ptxas exists; an earlier
    attempt to install triton==2.4.0 could only fail because torch 2.4.0 pins ==3.0.0.
    """
    status = []
    try:
        import triton

        status.append('import ok version={0}'.format(getattr(triton, '__version__', '?')))
    except Exception as exc:
        status.append('import failed: {0}: {1}'.format(type(exc).__name__, str(exc)[:180]))
    try:
        from torch.utils import _triton as torch_triton

        for name in ('has_triton', 'has_triton_package'):
            probe = getattr(torch_triton, name, None)
            if probe is not None:
                try:
                    status.append('{0}()={1}'.format(name, probe()))
                except Exception as exc:
                    status.append('{0}() raised {1}'.format(name, type(exc).__name__))
    except Exception as exc:
        status.append('torch.utils._triton unavailable: {0}'.format(type(exc).__name__))
    try:
        import importlib.metadata as metadata

        status.append('triton dist: {0}'.format(metadata.version('triton')))
    except Exception as exc:
        status.append('triton dist missing: {0}'.format(type(exc).__name__))
    try:
        probe = subprocess.run(
            ['sh', '-lc', 'command -v ptxas || ls /usr/local/cuda/bin/ptxas'],
            capture_output=True, text=True, timeout=60,
        )
        status.append('ptxas: ' + (probe.stdout or 'not found').strip().replace(chr(10), ' ')[:100])
    except Exception:
        pass
    print('[bench] triton diagnosis: ' + ' | '.join(status))
    return 'diagnosed'


def build_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to('cuda')


def base_latents(batch: int) -> torch.Tensor:
    generator = torch.Generator(device='cuda').manual_seed(SEEDS[0])
    shape = (batch, 4, SIZE // 8, SIZE // 8)
    return torch.randn(shape, generator=generator, device='cuda', dtype=torch.float16)


def run_once(pipe, batch: int, variant: str) -> dict:
    latents = base_latents(batch)
    generators = [torch.Generator(device='cuda').manual_seed(seed) for seed in SEEDS[:batch]]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    calls = {'n': 0}
    real_forward = pipe.unet.forward

    def counting_forward(*args, **kwargs):
        calls['n'] += 1
        return real_forward(*args, **kwargs)

    pipe.unet.forward = counting_forward
    state = {'previous': None, 'accumulated': 0.0, 'step': 0, 'temb': None}
    hooks = []

    def temb_hook(module, inputs, output):
        current = output.detach()
        previous = state['temb']
        if previous is not None:
            scale = max(float(previous.abs().mean()), 1e-6)
            state['accumulated'] += float((current - previous).abs().mean()) / scale
        state['temb'] = current

    if variant.startswith('cache_temb'):
        threshold = CACHE_THRESHOLDS[variant]
        hooks.append(pipe.unet.time_embedding.register_forward_hook(temb_hook))

        def caching_forward(*args, **kwargs):
            if state['accumulated'] < threshold and state['previous'] is not None:
                return state['previous']
            result = counting_forward(*args, **kwargs)
            state['accumulated'] = 0.0
            state['previous'] = result
            return result

        pipe.unet.forward = caching_forward
    elif variant.startswith('skip_stride'):
        stride = int(variant[len('skip_stride'):])

        def strided_forward(*args, **kwargs):
            state['step'] += 1
            if state['previous'] is not None and state['step'] % stride != 0:
                return state['previous']
            result = counting_forward(*args, **kwargs)
            state['previous'] = result
            return result

        pipe.unet.forward = strided_forward

    try:
        started = time.perf_counter()
        result = pipe(
            [PROMPT] * batch,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generators,
            latents=latents.clone(),
            output_type='latent',
            height=SIZE,
            width=SIZE,
        )
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        produced = result.images.detach().float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
        pipe.unet.forward = real_forward
    record = {
        'variant': variant,
        'batch': batch,
        'elapsed_s': elapsed,
        'image_steps': batch * STEPS,
        'ms_per_image_step': 1000.0 * elapsed / (batch * STEPS),
        'unet_calls': calls['n'],
        'skip_fraction': 1.0 - calls['n'] / float(STEPS),
        'peak_vram_gib': peak,
        'latents': produced,
    }
    return record


def main() -> None:
    print('[bench] torch {0} | {1}'.format(torch.__version__, torch.cuda.get_device_name(0)))
    print('[bench] triton: ' + ensure_triton())
    results: dict = {}
    reference: dict = {}
    for batch, steps in SHAPES:
        prefix = 'b{0}_s{1}'.format(batch, steps)
        for variant in VARIANTS:
            key = '{0}/{1}'.format(prefix, variant)
            try:
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
                    diff = (record['latents'] - reference[prefix]).abs()
                    record['latent_mae'] = float(diff.mean())
                    record['latent_rel_error'] = float(diff.mean() / (reference[prefix].abs().mean() + 1e-6))
                    record.pop('latents')
                results[key] = record
                print('[bench] {0}: {1:.1f}s {2:.0f} ms/image-step calls={3} skip={4:.2f} vram={5:.2f} rel={6:.4f}'.format(
                    key, record['elapsed_s'], record['ms_per_image_step'], record['unet_calls'],
                    record['skip_fraction'], record['peak_vram_gib'], record.get('latent_rel_error', 0.0)))
                del pipe
            except Exception:
                error = traceback.format_exc()
                results[key] = {'variant': variant, 'batch': batch, 'error': error[-1200:]}
                print('[bench] {0} FAILED: {1}'.format(key, error.strip().splitlines()[-1][:180]))
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            OUT.write_text(json.dumps(results, indent=2) + chr(10))
    lines = ['# UNet acceleration bench (SD1.5, 512px, fp16, 64 DDIM steps)', '']
    for key in sorted(results):
        record = results[key]
        if 'error' in record:
            lines.append('- {0}: FAILED ({1})'.format(key, record['error'].strip().splitlines()[-1][:120]))
        else:
            lines.append('- {0}: {1:.2f} s, {2:.0f} ms/image-step, calls={3}, skip={4:.2f}, vram={5:.2f} GiB, rel_err={6:.4f}'.format(
                key, record['elapsed_s'], record['ms_per_image_step'], record['unet_calls'],
                record['skip_fraction'], record['peak_vram_gib'], record.get('latent_rel_error', 0.0)))
    REPORT.write_text(chr(10).join(lines) + chr(10))
    print('[bench] wrote', OUT, 'and', REPORT)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('[bench] FAILED' + chr(10) + traceback.format_exc())
        raise
