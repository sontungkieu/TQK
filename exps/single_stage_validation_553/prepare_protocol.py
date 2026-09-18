#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
CAL = ROOT / "exps/single_stage_calibration"
SOURCE_PROMPTS = ROOT / "exps/direct_sd15_psp_vs_10to2_553/prompts_geneval_all_553.jsonl"
SOURCE_FROZEN = CAL / "FROZEN_SCHEDULE.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_exact(source: Path, target: Path) -> None:
    if target.exists() and sha256(target) != sha256(source):
        raise RuntimeError(f"refusing to overwrite mismatched protocol artifact: {target}")
    if not target.exists():
        shutil.copy2(source, target)
    assert sha256(target) == sha256(source)


def main() -> None:
    prompts = [json.loads(line) for line in SOURCE_PROMPTS.read_text().splitlines() if line.strip()]
    assert len(prompts) == 553
    frozen = json.loads(SOURCE_FROZEN.read_text())
    compute = frozen["M"] * frozen["checkpoint"] + frozen["K"] * (64 - frozen["checkpoint"])
    assert compute == frozen["logical_compute"] <= 256
    copy_exact(SOURCE_PROMPTS, EXP / "prompts_geneval_all_553.jsonl")
    copy_exact(SOURCE_FROZEN, EXP / "FROZEN_SCHEDULE.json")
    manifest = {
        "prompt_count": 553,
        "prompts_sha256": sha256(EXP / "prompts_geneval_all_553.jsonl"),
        "source_prompts_sha256": sha256(SOURCE_PROMPTS),
        "frozen_schedule_sha256": sha256(EXP / "FROZEN_SCHEDULE.json"),
        "source_frozen_schedule_sha256": sha256(SOURCE_FROZEN),
        "fresh_seed_base": 20260919,
        "schedule": frozen,
    }
    (EXP / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "protocol_locked", **manifest}, indent=2))


if __name__ == "__main__":
    main()

