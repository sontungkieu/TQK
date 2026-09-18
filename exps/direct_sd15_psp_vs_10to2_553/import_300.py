#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SOURCE = Path("/workspace/psp_direct_sd15_resume_cache")
OLD_MANIFEST = ROOT / "exps/direct_sd15_psp_vs_10to2_300/prompt_ids_300.json"


def validate_row(row: dict, method: str, prompt_id: int) -> None:
    assert row["prompt_id"] == prompt_id and row["method"] == method
    assert row["model"] == "runwayml/stable-diffusion-v1-5"
    assert row["scheduler"] == "DDIMScheduler"
    assert row["steps"] == 64 and row["eta"] == 0.0 and row["guidance_scale"] == 7.5
    assert row["logical_unet_evals"] == 256 and row["winner_id"] is not None
    expected = (
        [(16, 8, 4), (32, 4, 2), (64, 2, 1)]
        if method == "psp"
        else [(16, 10, 2), (64, 2, 1)]
    )
    actual = [(x["step"], x["batch_before"], x["batch_after"]) for x in row["trace"]]
    assert actual == expected


def main() -> None:
    ids = json.loads(OLD_MANIFEST.read_text())["prompt_ids"]
    assert len(ids) == 300 and len(set(ids)) == 300
    assignments: dict[str, int] = {}
    imported = 0
    for prompt_id in ids:
        found = {}
        for method in ("psp", "early10to2"):
            candidates = list((SOURCE / "metadata").glob(f"gpu*/{prompt_id:05d}_{method}.json"))
            assert len(candidates) == 1, (prompt_id, method, candidates)
            path = candidates[0]
            worker = int(path.parent.name.removeprefix("gpu"))
            row = json.loads(path.read_text())
            validate_row(row, method, prompt_id)
            found[method] = (worker, row, path)
        assert found["psp"][0] == found["early10to2"][0]
        assert found["psp"][1]["initial_latent_hashes"] == found["early10to2"][1]["initial_latent_hashes"][:8]
        worker = found["psp"][0]
        assignments[str(prompt_id)] = worker
        for method in ("psp", "early10to2"):
            source_image = SOURCE / "outputs" / method / f"{prompt_id:05d}.png"
            assert source_image.is_file()
            target_image = EXP / "outputs" / method / source_image.name
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target_image)
            target_metadata = EXP / "metadata" / f"gpu{worker}" / found[method][2].name
            target_metadata.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found[method][2], target_metadata)
        imported += 1

    counts = {0: sum(worker == 0 for worker in assignments.values()), 1: sum(worker == 1 for worker in assignments.values())}
    for prompt_id in range(553):
        key = str(prompt_id)
        if key in assignments:
            continue
        worker = 0 if counts[0] <= counts[1] else 1
        assignments[key] = worker
        counts[worker] += 1
    assert counts == {0: 277, 1: 276}
    (EXP / "worker_assignments_553.json").write_text(
        json.dumps(
            {
                "policy": "preserve actual GPU for 300 imported prompts; greedily balance remaining prompts",
                "counts": {str(k): v for k, v in counts.items()},
                "assignments": assignments,
                "imported_prompt_ids": ids,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"imported_prompts": imported, "worker_counts": counts}))


if __name__ == "__main__":
    main()
