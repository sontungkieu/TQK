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


def expected(
    count: int = PROMPT_COUNT,
    selection_seed: int = SELECTION_SEED,
    split_seed: int = SPLIT_SEED,
    folds: int = 0,
) -> tuple[list[dict], dict]:
    """Select the calibration prompts.

    Defaults reproduce the committed 120-prompt selection exactly, so --verify-existing and the
    committed artefacts stay valid. The pre-registered next experiment selects 200 prompts with a
    new fixed seed and a deterministic five-fold split for out-of-fold selection; that is requested
    explicitly with --count/--seed/--folds and written to its own directory.
    """
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
    assert count <= len(unique), f"only {len(unique)} unique prompts available"
    indices = list(range(len(unique)))
    random.Random(selection_seed).shuffle(indices)
    selected = [unique[index] for index in indices[:count]]
    search = set()
    fold_of: dict[int, int] = {}
    if folds:
        fold_order = list(range(count))
        random.Random(split_seed).shuffle(fold_order)
        for position, prompt_id in enumerate(fold_order):
            fold_of[prompt_id] = position % folds
    else:
        split_order = list(range(count))
        random.Random(split_seed).shuffle(split_order)
        search = set(split_order[:80])
    rows = []
    for prompt_id, row in enumerate(selected):
        entry = {
            "prompt_id": prompt_id,
            "prompt": row["prompt"],
            "source_corpus": "ImageReward test_ir.json",
            "source_index": row["source_index"],
            "source_id": row["source_id"],
        }
        if folds:
            entry["fold"] = fold_of[prompt_id]
        else:
            entry["split"] = "search" if prompt_id in search else "validation"
        rows.append(entry)
    manifest = {
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "geneval_path": str(GENEVAL.relative_to(ROOT)),
        "geneval_sha256": sha256(GENEVAL),
        "selection_seed": selection_seed,
        "split_seed": split_seed,
        "prompt_count": count,
    }
    # Key order is part of the committed artefact: the default path must reproduce the existing
    # 120-prompt manifest byte for byte, so the fold variant only appends its own keys.
    if folds:
        manifest["folds"] = folds
        manifest["fold_counts"] = {
            str(fold): sum(1 for row in rows if row["fold"] == fold) for fold in range(folds)
        }
    else:
        manifest["search_count"] = 80
        manifest["validation_count"] = 40
    manifest["exact_geneval_prompt_overlap"] = 0
    if folds:
        manifest["prompt_ids_by_fold"] = {
            str(fold): [row["prompt_id"] for row in rows if row["fold"] == fold]
            for fold in range(folds)
        }
    else:
        manifest["prompt_ids_search"] = [row["prompt_id"] for row in rows if row["split"] == "search"]
        manifest["prompt_ids_validation"] = [row["prompt_id"] for row in rows if row["split"] == "validation"]
    return rows, manifest


def render_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--count", type=int, default=PROMPT_COUNT)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--folds", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=EXP / "prompts")
    args = parser.parse_args()
    if args.verify_existing:
        rows, manifest = expected()
    else:
        rows, manifest = expected(args.count, args.seed, args.split_seed, args.folds)
    prompt_path = args.out_dir / "calibration_prompts.jsonl"
    manifest_path = args.out_dir / "prompt_manifest.json"
    split_path = args.out_dir / "prompt_split.json"
    expected_prompts = render_jsonl(rows)
    expected_manifest = json.dumps(manifest, indent=2) + "\n"
    # The split file maps prompt id to its group: search/validation for the committed 120-prompt
    # selection, fold index when the pre-registered five-fold selection is requested.
    expected_split = json.dumps(
        {
            str(row["prompt_id"]): (row["fold"] if args.folds else row["split"])
            for row in rows
        },
        indent=2,
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

