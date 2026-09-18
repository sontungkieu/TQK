#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap(frame: pd.DataFrame, metric: str, seed: int = 20260917, samples: int = 10_000):
    wide = frame.pivot(index="prompt_id", columns="method", values=metric)
    delta = (wide["early10to2"] - wide["psp"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(samples, len(delta)), replace=True).mean(axis=1)
    return {
        "psp": float(wide["psp"].mean()),
        "early10to2": float(wide["early10to2"].mean()),
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "samples": samples,
        "seed": seed,
    }
