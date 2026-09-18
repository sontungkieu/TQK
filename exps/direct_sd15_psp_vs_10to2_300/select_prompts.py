#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"
OUT_DIR = Path(__file__).resolve().parent
SEED = 20260917


def selected() -> list[dict]:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    groups: dict[str, list[int]] = defaultdict(list)
    for prompt_id, row in enumerate(rows):
        groups[row["tag"]].append(prompt_id)
    if len(groups) != 6 or any(len(ids) < 50 for ids in groups.values()):
        raise AssertionError({tag: len(ids) for tag, ids in groups.items()})
    rng = random.Random(SEED)
    prompt_ids = []
    for tag in sorted(groups):
        prompt_ids.extend(rng.sample(groups[tag], 50))
    prompt_ids.sort()
    if len(prompt_ids) != 300 or len(set(prompt_ids)) != 300:
        raise AssertionError("subset must contain exactly 300 unique prompts")
    return [{"prompt_id": idx, **rows[idx]} for idx in prompt_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    rows = selected()
    jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    manifest = {
        "selection_seed": SEED,
        "source": str(SOURCE.relative_to(ROOT)),
        "prompt_ids": [row["prompt_id"] for row in rows],
        "counts_by_tag": {
            tag: sum(row["tag"] == tag for row in rows) for tag in sorted({r["tag"] for r in rows})
        },
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    jsonl_path = OUT_DIR / "prompts_geneval_balanced_300.jsonl"
    manifest_path = OUT_DIR / "prompt_ids_300.json"
    if args.verify_existing:
        assert jsonl_path.read_text() == jsonl, "locked JSONL subset differs from deterministic selection"
        assert manifest_path.read_text() == manifest_text, "locked prompt manifest differs from selection"
        print("locked 300-prompt subset verified")
    else:
        jsonl_path.write_text(jsonl)
        manifest_path.write_text(manifest_text)


if __name__ == "__main__":
    main()
