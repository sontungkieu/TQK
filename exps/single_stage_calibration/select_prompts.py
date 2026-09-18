#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SOURCE = ROOT / "Fk-Diffusion-Steering/text_to_image/prompt_files/test_ir.json"
GENEVAL = ROOT / "Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"
SELECTION_SEED = 20260918
SPLIT_SEED = 20260918
PROMPT_COUNT = 120


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected() -> tuple[list[dict], dict]:
    source_rows = json.loads(SOURCE.read_text())
    unique = []
    seen = set()
    for source_index, row in enumerate(source_rows):
        prompt = row["prompt"].strip()
        key = prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append({"source_index": source_index, "source_id": row["id"], "prompt": prompt})
    geneval_prompts = {
        json.loads(line)["prompt"].strip().casefold()
        for line in GENEVAL.read_text().splitlines()
        if line.strip()
    }
    assert not ({row["prompt"].casefold() for row in unique} & geneval_prompts)
    indices = list(range(len(unique)))
    random.Random(SELECTION_SEED).shuffle(indices)
    selected = [unique[index] for index in indices[:PROMPT_COUNT]]
    split_order = list(range(PROMPT_COUNT))
    random.Random(SPLIT_SEED).shuffle(split_order)
    search = set(split_order[:80])
    rows = []
    for prompt_id, row in enumerate(selected):
        rows.append({
            "prompt_id": prompt_id,
            "prompt": row["prompt"],
            "source_corpus": "ImageReward test_ir.json",
            "source_index": row["source_index"],
            "source_id": row["source_id"],
            "split": "search" if prompt_id in search else "validation",
        })
    manifest = {
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "geneval_path": str(GENEVAL.relative_to(ROOT)),
        "geneval_sha256": sha256(GENEVAL),
        "selection_seed": SELECTION_SEED,
        "split_seed": SPLIT_SEED,
        "prompt_count": PROMPT_COUNT,
        "search_count": 80,
        "validation_count": 40,
        "exact_geneval_prompt_overlap": 0,
        "prompt_ids_search": [row["prompt_id"] for row in rows if row["split"] == "search"],
        "prompt_ids_validation": [row["prompt_id"] for row in rows if row["split"] == "validation"],
    }
    return rows, manifest


def render_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    rows, manifest = expected()
    prompt_path = EXP / "prompts/calibration_prompts.jsonl"
    manifest_path = EXP / "prompts/prompt_manifest.json"
    split_path = EXP / "prompts/prompt_split.json"
    expected_prompts = render_jsonl(rows)
    expected_manifest = json.dumps(manifest, indent=2) + "\n"
    expected_split = json.dumps(
        {str(row["prompt_id"]): row["split"] for row in rows}, indent=2
    ) + "\n"
    if args.verify_existing:
        assert prompt_path.read_text() == expected_prompts
        assert manifest_path.read_text() == expected_manifest
        assert split_path.read_text() == expected_split
        print(json.dumps({"status": "verified", **manifest}))
        return
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(expected_prompts)
    manifest_path.write_text(expected_manifest)
    split_path.write_text(expected_split)
    print(json.dumps({"status": "written", **manifest}))


if __name__ == "__main__":
    main()

