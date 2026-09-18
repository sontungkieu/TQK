#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "phase2"), required=True)
    parser.add_argument("--started", type=int)
    parser.add_argument("--ended", type=int)
    args = parser.parse_args()
    common = {
        "official_psp_commit": "590f59f58384169c719e431dc01d14006fb0fc0c",
        "parent_direct_experiment_commit": "4d6e8a44401b507c3c78a57ba922a432a4c9887f",
        "orx_source_snapshot": "recorded by OpenResearch run metadata",
        "source_git_status_at_launch": "clean committed OpenResearch snapshot",
        "source_git_diff_at_launch": "empty; experiment changes are committed",
    }
    if args.phase == "phase1":
        path = ROOT / "exps/single_stage_calibration/metrics/run_info.json"
        payload = {**common, "host_started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    else:
        assert args.started is not None and args.ended is not None and args.ended >= args.started
        path = ROOT / "exps/single_stage_validation_553/metrics/run_info.json"
        payload = {
            **common,
            "phase2_started_unix": args.started,
            "phase2_ended_unix": args.ended,
            "generation_wall_s": args.ended - args.started,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

