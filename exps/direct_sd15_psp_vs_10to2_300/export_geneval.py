#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

EXP = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (EXP / "prompts_geneval_balanced_300.jsonl").read_text().splitlines()]
for method in ("psp", "early10to2"):
    root = EXP / "geneval_inputs" / method
    for row in rows:
        prompt_id = int(row["prompt_id"])
        folder = root / f"{prompt_id:05d}"
        samples = folder / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        (folder / "metadata.jsonl").write_text(json.dumps({k: v for k, v in row.items() if k != "prompt_id"}) + "\n")
        target = EXP / "outputs" / method / f"{prompt_id:05d}.png"
        link = samples / "00000.png"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(target.resolve(), link)
print("geneval inputs exported")
