#!/usr/bin/env python3
from schedules import CHECKPOINTS, SURVIVORS, valid_schedules


rows = valid_schedules()
assert len(rows) == len(CHECKPOINTS) * len(SURVIVORS) == 39
assert {(row["checkpoint"], row["K"], row["M"]) for row in rows} >= {
    (8, 1, 25), (16, 2, 10), (32, 3, 5)
}
assert all(row["logical_compute"] <= 256 for row in rows)
print(f"budget check PASS: {len(rows)} fixed schedules")

