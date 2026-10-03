#!/usr/bin/env python3
"""Re-derive every machine-readable number in the bank200 evidence notes.

The four notes under ``kaggle/evidence/`` were written by hand from
``replay200/schedule_ranking.csv``, ``schedule_ranking_3round.csv`` and the merged
bank. Hand-copying is exactly how four digits went wrong when
``bank200_frontier.md`` was first written, and nothing in the repository would have
caught it. This script closes that loop: it *parses the markdown* and compares it
against the data, rather than asserting constants that could drift away from the prose
just as easily.

Run it directly (``python3 check_evidence_numbers.py``); it exits non-zero on the first
batch of disagreements and prints every one it finds. ``tests/test_bank200_evidence.py``
runs it as a gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]

INT_COLS = ("n", "t1", "k1", "t2", "k2", "t3", "k3", "unet", "scores", "rounds")
FLOAT_COLS = ("mean_delta_ir", "lcb80_delta_ir", "equiv")
SHAPE_RE = re.compile(r"\d+->\d+@\d+(?:->\d+@\d+)*")

failures: list[str] = []
checked = 0


def check(label: str, got: object, want: object, tol: float = 5e-7) -> None:
    global checked
    checked += 1
    if isinstance(want, float) or isinstance(got, float):
        try:
            ok = abs(float(got) - float(want)) <= tol
        except (TypeError, ValueError):
            ok = False
    else:
        ok = got == want
    if not ok:
        failures.append("%-46s note=%r data=%r" % (label, got, want))


def num(cell: str) -> float:
    """Markdown table cell -> number. Tolerates bold, code spans, %, and thin-space
    thousands separators (``16 817``)."""
    cleaned = cell.replace("*", "").replace("`", "").replace("%", "")
    for space in (" ", "\u2009", "\u00a0", ","):
        cleaned = cleaned.replace(space, "")
    return float(cleaned.lstrip("+"))


def shape(cell: str) -> str:
    match = SHAPE_RE.search(cell.replace("*", "").replace("`", "").replace(" ", ""))
    assert match, "no schedule in cell %r" % cell
    return match.group(0)


def table_rows(text: str, header: str) -> list[list[str]]:
    """Rows of the first pipe table whose header row contains ``header``."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and header in line:
            following = lines[i + 1] if i + 1 < len(lines) else ""
            if set(following.replace("|", "").replace(" ", "")) <= {"-", ":"} and following:
                rows = []
                for row in lines[i + 2:]:
                    if not row.startswith("|"):
                        break
                    rows.append([c.strip() for c in row.strip().strip("|").split("|")])
                return rows
    raise AssertionError("table with header %r not found" % header)


def load_rankings(replay: Path) -> tuple[list[dict], list[dict]]:
    def load(path: Path) -> list[dict]:
        rows = []
        for row in csv.DictReader(open(path)):
            for key in INT_COLS:
                if row.get(key) not in (None, ""):
                    row[key] = int(row[key])
            for key in FLOAT_COLS:
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
            rows.append(row)
        return rows

    return load(replay / "schedule_ranking.csv"), load(replay / "schedule_ranking_3round.csv")


def label(row: dict) -> str:
    if int(row["rounds"]) == 1:
        return "%d->%d@%d" % (row["n"], row["k1"], row["t1"])
    if int(row["rounds"]) == 2:
        return "%d->%d@%d->%d@%d->1@64" % (row["n"], row["k1"], row["t1"], row["k2"], row["t2"])
    return "%d->%d@%d->%d@%d->%d@%d->1@64" % (
        row["n"], row["k1"], row["t1"], row["k2"], row["t2"], row["k3"], row["t3"])


def thousands(value: int) -> str:
    """Match the notes' style: ``1403`` stays bare, ``16 817`` is separated."""
    return f"{value:,}".replace(",", " ") if value >= 10000 else str(value)


def flat(text: str) -> str:
    """Collapse newlines so a prose claim survives Markdown hard-wrapping."""
    return re.sub(r"\s+", " ", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=EXP / "replay200")
    parser.add_argument("--evidence-dir", type=Path, default=REPO / "kaggle" / "evidence")
    parser.add_argument("--bank-dir", type=Path, default=EXP / "bank200")
    parser.add_argument("--expect-prompts", type=int, default=200)
    parser.add_argument("--scorer-cost", type=float, default=1.9,
                        help="scorer cost as a multiple of one UNet evaluation")
    args = parser.parse_args()

    # replay200/*.csv is git-ignored, so a fresh clone cannot check anything without
    # regenerating the two selectors first. Say so instead of silently passing.
    missing_csv = [name for name in ("schedule_ranking.csv", "schedule_ranking_3round.csv")
                   if not (args.replay_dir / name).exists()]
    if missing_csv:
        print("SKIP: cannot check, missing %s" % ", ".join(missing_csv))
        print("      replay200/*.csv is git-ignored. Regenerate with")
        print("      python3 select_fold_schedule.py && python3 select_three_round.py")
        return 2

    one_two, three = load_rankings(args.replay_dir)
    union = sorted(one_two + three, key=lambda r: -r["mean_delta_ir"])
    rank_of = {id(row): i + 1 for i, row in enumerate(union)}
    by_n: dict[int, list[dict]] = defaultdict(list)
    for row in union:
        by_n[int(row["n"])].append(row)
    two_by_n: dict[int, list[dict]] = defaultdict(list)
    for row in one_two:
        if int(row["rounds"]) == 2:
            two_by_n[int(row["n"])].append(row)
    best_three = max(three, key=lambda r: r["mean_delta_ir"])

    frontier = (args.evidence_dir / "bank200_frontier.md").read_text(encoding="utf-8")
    three_round = (args.evidence_dir / "bank200_three_round.md").read_text(encoding="utf-8")
    provenance = (args.evidence_dir / "bank200_provenance.md").read_text(encoding="utf-8")
    fold_selection = (args.evidence_dir / "bank200_fold_selection.md").read_text(encoding="utf-8")

    # --- family counts, frontier note -------------------------------------
    families = {"one/two-discard": one_two, "three-discard": three}
    for cells in table_rows(frontier, "Family"):
        name = cells[0].strip()
        rows = families[name]
        best = max(rows, key=lambda r: r["mean_delta_ir"])
        check("frontier/%s shapes" % name, num(cells[1]), len(rows))
        check("frontier/%s best rank" % name, num(cells[2]), rank_of[id(best)])
        check("frontier/%s best delta" % name, num(cells[3]), best["mean_delta_ir"])
        check("frontier/%s worst delta" % name, num(cells[4]),
              min(r["mean_delta_ir"] for r in rows))

    # --- union leaderboard, frontier note ---------------------------------
    expected_rows = [(i, union[i - 1]) for i in range(1, 6)]
    expected_rows.append((rank_of[id(best_three)], best_three))
    for cells, (rank, row) in zip(table_rows(frontier, "Union rank"), expected_rows):
        check("frontier/rank %d rank" % rank, num(cells[0]), rank)
        check("frontier/rank %d shape" % rank, shape(cells[1]), label(row))
        check("frontier/rank %d delta" % rank, num(cells[2]), row["mean_delta_ir"])
        check("frontier/rank %d lcb80" % rank, num(cells[3]), row["lcb80_delta_ir"])
        check("frontier/rank %d unet" % rank, num(cells[4]), row["unet"])
        check("frontier/rank %d scores" % rank, num(cells[5]), row["scores"])
        check("frontier/rank %d equiv" % rank, num(cells[6]), row["equiv"])

    # --- per-N frontier, frontier note ------------------------------------
    cells_by_n: dict[int, list[str]] = {}
    for cells in table_rows(frontier, "Best shape"):
        if len(cells) == 4:  # left-hand column of the two-up table
            cells_by_n[int(num(cells[0]))] = cells
        elif len(cells) == 8:
            cells_by_n[int(num(cells[0]))] = cells[:4]
            cells_by_n[int(num(cells[4]))] = cells[4:]
    for n, best in ((n, max(rows, key=lambda r: r["mean_delta_ir"])) for n, rows in by_n.items()):
        cells = cells_by_n[n]
        check("frontier/N=%d shapes" % n, num(cells[1]), len(by_n[n]))
        check("frontier/N=%d delta" % n, num(cells[2]), best["mean_delta_ir"])
        check("frontier/N=%d shape" % n, shape(cells[3]), label(best))

    # --- budget mechanism, frontier note ----------------------------------
    budget_rows = table_rows(frontier, "Max affordable t1")
    if len(budget_rows) != len(by_n):
        failures.append("budget table has %d rows, data has %d N values"
                        % (len(budget_rows), len(by_n)))
    for cells in budget_rows:
        n = int(num(cells[0]))
        best = max(by_n[n], key=lambda r: r["mean_delta_ir"])
        affordable = two_by_n.get(n, [])
        if cells[1].strip() in {"none", "-"}:
            if affordable:
                failures.append("frontier/N=%d says no two-discard shape, data has %d"
                                % (n, len(affordable)))
        else:
            max_t1 = max(int(r["t1"]) for r in affordable)
            check("frontier/N=%d max t1" % n, num(cells[1]), max_t1)
            check("frontier/N=%d t1 percent" % n, num(cells[2]), round(100 * max_t1 / 64), 0.51)
        check("frontier/N=%d budget delta" % n, num(cells[3]), best["mean_delta_ir"])
        check("frontier/N=%d budget shape" % n, shape(cells[4]), label(best))

    # --- prose counts, frontier note --------------------------------------
    tied = [r for r in one_two if abs(r["mean_delta_ir"]) < 0.006]
    prose = [
        "Union = %s shapes." % thousands(len(union)),
        "Three-discard shapes with a positive mean Delta IR: **0 of %s**."
        % thousands(len(three)),
        "(%d%%)" % round(100 * sum(1 for r in one_two
                                   if r["mean_delta_ir"] > best_three["mean_delta_ir"])
                         / len(one_two)),
        "**%d** fall inside that band" % len(tied),
        "**%d** of them beat PSP" % sum(1 for r in tied if r["mean_delta_ir"] > 0),
    ]
    for claim in prose:
        if claim not in flat(frontier):
            failures.append("frontier prose claim missing or wrong: %r" % claim)

    # --- the rho cross-check, three_round note ----------------------------
    psp = union[0]
    check("three_round/PSP unet", psp["unet"] + args.scorer_cost * psp["scores"], psp["equiv"])
    fitting = [r for r in one_two if r["equiv"] <= 256]
    fitting_two = [r for r in one_two if int(r["rounds"]) == 2 and r["equiv"] <= 256]
    fitting_three = [r for r in three if r["equiv"] <= 256]
    check("three_round/fit one+two", len(fitting), 6)
    check("three_round/fit two only", len(fitting_two), 1)
    check("three_round/fit three", len(fitting_three), 0)
    for claim in ("only %d of %s one/two-discard shapes fit" % (len(fitting),
                                                              thousands(len(one_two))),
                  "**no** three-discard shape fits (%d of %s)"
                  % (len(fitting_three), thousands(len(three)))):
        if claim not in flat(three_round):
            failures.append("three_round budget claim missing or wrong: %r" % claim)

    best_fit = max(fitting, key=lambda r: r["mean_delta_ir"])
    for cells in table_rows(three_round, "Accounting"):
        if "NFE-equivalent" not in cells[0]:
            continue
        check("three_round/NFE best shape", shape(cells[1]), label(best_fit))
        check("three_round/NFE best delta", num(cells[2]), best_fit["mean_delta_ir"])

    # --- union top-3, three_round note ------------------------------------
    for cells, row in zip(table_rows(three_round, "Rank (measured mean Delta IR)"), union[:3]):
        check("three_round/top%d shape" % int(num(cells[0])), shape(cells[1]), label(row))
        check("three_round/top%d delta" % int(num(cells[0])), num(cells[2]),
              row["mean_delta_ir"])
        check("three_round/top%d equiv" % int(num(cells[0])), num(cells[3]), row["equiv"])

    # --- bank digest, provenance and fold-selection notes ------------------
    bank_files = sorted(p for p in (args.bank_dir / "gpu0").glob("*.json") if p.stem.isdigit())
    if not bank_files:
        print("skipped 4 check(s): no merged bank at %s (git-ignored)" % args.bank_dir)
    else:
        digest = hashlib.sha256()
        for path in bank_files:
            digest.update(path.name.encode())
            digest.update(b":")
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
            digest.update(b"\n")
        check("bank/prompts", len(bank_files), args.expect_prompts)
        for note, name in ((provenance, "provenance"), (fold_selection, "fold_selection")):
            if digest.hexdigest() not in note:
                failures.append("%s note does not carry the recomputed bank sha256 %s"
                                % (name, digest.hexdigest()))
        machine = json.loads((args.replay_dir / "three_round.json").read_text())
        check("three_round.json bank sha256", machine.get("bank_sha256"), digest.hexdigest())

    print("checked %d values across %d evidence notes" % (checked, 4))
    if failures:
        print("FAIL: %d disagreement(s)" % len(failures))
        for line in failures:
            print("  - " + line)
        return 1
    print("evidence numbers agree with replay200/ and bank200/")
    return 0


if __name__ == "__main__":
    sys.exit(main())