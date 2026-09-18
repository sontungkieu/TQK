#!/usr/bin/env python3
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
frozen = json.loads((EXP / "FROZEN_SCHEDULE.json").read_text())
ours = frozen["M"] * frozen["checkpoint"] + frozen["K"] * (64 - frozen["checkpoint"])
psp = 8 * 16 + 4 * 16 + 2 * 32
assert ours == frozen["logical_compute"] <= 256
assert psp == 256
assert frozen["K"] in (1, 2, 3)
assert frozen["M"] > frozen["K"]
print(json.dumps({"status": "PASS", "psp_evals": psp, "ours_evals": ours, "frozen": frozen}, indent=2))

