from __future__ import annotations

CHECKPOINTS = (8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 28, 32)
SURVIVORS = (1, 2, 3)
STEPS = 64
BUDGET = 256


def initial_pool(t: int, k: int) -> int:
    return (BUDGET - k * (STEPS - t)) // t


def logical_compute(t: int, k: int, m: int | None = None) -> int:
    if m is None:
        m = initial_pool(t, k)
    return m * t + k * (STEPS - t)


def valid_schedules() -> list[dict[str, int]]:
    rows = []
    for t in CHECKPOINTS:
        for k in SURVIVORS:
            m = initial_pool(t, k)
            compute = logical_compute(t, k, m)
            if m <= k or m > 25 or compute > BUDGET:
                continue
            rows.append({
                "checkpoint": t,
                "K": k,
                "M": m,
                "logical_compute": compute,
                "unused_compute": BUDGET - compute,
                "verifier_calls": 2,
                "verifier_candidate_scores": m + k,
            })
    return rows

