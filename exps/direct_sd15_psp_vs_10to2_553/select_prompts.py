#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"
EXP = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    source_rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    assert len(source_rows) == 553
    rows = [{"prompt_id": idx, **row} for idx, row in enumerate(source_rows)]
    jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    manifest = {
        "selection": "all official GenEval prompts in original order",
        "source": str(SOURCE.relative_to(ROOT)),
        "prompt_ids": list(range(553)),
        "counts_by_tag": dict(sorted(Counter(row["tag"] for row in rows).items())),
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    jsonl_path = EXP / "prompts_geneval_all_553.jsonl"
    manifest_path = EXP / "prompt_ids_553.json"
    if args.verify_existing:
        assert jsonl_path.read_text() == jsonl
        assert manifest_path.read_text() == manifest_text
        print("locked full 553-prompt set verified")
    else:
        jsonl_path.write_text(jsonl)
        manifest_path.write_text(manifest_text)


if __name__ == "__main__":
    main()
