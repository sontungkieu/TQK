#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

EXP = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (EXP / "prompts_geneval_all_553.jsonl").read_text().splitlines()]
assignments = {
    int(k): int(v)
    for k, v in json.loads((EXP / "worker_assignments_553.json").read_text())["assignments"].items()
}
smoke = rows[:8]
elapsed_by_worker = {0: 0.0, 1: 0.0}
prompts_by_worker = {0: set(), 1: set()}

for position, prompt in enumerate(smoke):
    worker = assignments[int(prompt["prompt_id"])]
    prompt_id = prompt["prompt_id"]
    method_rows = {}
    for method in ("psp", "early10to2"):
        path = EXP / "metadata" / f"gpu{worker}" / f"{prompt_id:05d}_{method}.json"
        image = EXP / "outputs" / method / f"{prompt_id:05d}.png"
        assert path.exists() and image.exists(), (path, image)
        row = json.loads(path.read_text())
        assert row["logical_unet_evals"] == 256
        assert row["winner_id"] is not None
        assert math.isfinite(row["final_image_reward"])
        expected = (
            [(16, 8, 4), (32, 4, 2), (64, 2, 1)]
            if method == "psp"
            else [(16, 10, 2), (64, 2, 1)]
        )
        actual = [(x["step"], x["batch_before"], x["batch_after"]) for x in row["trace"]]
        assert actual == expected, (method, actual, expected)
        method_rows[method] = row
        elapsed_by_worker[worker] += row["elapsed_s"]
        prompts_by_worker[worker].add(prompt_id)
    assert method_rows["psp"]["initial_latent_hashes"] == method_rows["early10to2"]["initial_latent_hashes"][:8]

throughputs = {
    worker: len(prompts_by_worker[worker]) / elapsed_by_worker[worker]
    for worker in (0, 1)
}
estimate = 553 / sum(throughputs.values())
payload = {
    "status": "PASS",
    "smoke_prompts": [row["prompt_id"] for row in smoke],
    "completed_by_worker": {str(k): len(v) for k, v in prompts_by_worker.items()},
    "seconds_by_worker": elapsed_by_worker,
    "prompt_pairs_per_second": throughputs,
    "estimated_full_generation_seconds_if_run_from_scratch": estimate,
    "estimated_full_generation_hours": estimate / 3600,
}
(EXP / "metrics").mkdir(exist_ok=True)
(EXP / "metrics" / "smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
