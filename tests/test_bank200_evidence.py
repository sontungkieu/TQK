"""Gate the hand-written bank200 evidence notes against the data they describe.

The four notes in ``kaggle/evidence/`` were typed out of ``replay200/*.csv`` by hand,
and that is how four digits were wrong the first time. ``check_evidence_numbers.py``
is the correction; these tests keep the correction honest.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CAL = REPO / "exps" / "single_stage_calibration"
CHECKER = CAL / "check_evidence_numbers.py"
EVIDENCE = REPO / "kaggle" / "evidence"
RANKING_CSV = CAL / "replay200" / "schedule_ranking.csv"


def run_checker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=str(CAL), capture_output=True, text=True)


def test_evidence_notes_agree_with_the_data():
    result = run_checker()
    if result.returncode == 2:
        pytest.skip("replay200/*.csv is git-ignored and absent: " + result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "agree with replay200/" in result.stdout


def test_checker_rejects_a_doctored_note(tmp_path):
    """A gate that cannot go red is decoration. Perturb one digit, demand a failure."""
    if not RANKING_CSV.exists():
        pytest.skip("replay200/*.csv is git-ignored and absent")
    doctored = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, doctored)
    target = doctored / "bank200_frontier.md"
    text = target.read_text(encoding="utf-8")
    assert "-0.008636" in text, "fixture drifted; pick another anchor digit"
    target.write_text(text.replace("-0.008636", "-0.008641"), encoding="utf-8")

    result = run_checker("--evidence-dir", str(doctored))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "frontier/N=12 delta" in result.stdout


def test_checker_reports_a_doctored_prose_count(tmp_path):
    """Prose counts drift the same way table cells do."""
    if not RANKING_CSV.exists():
        pytest.skip("replay200/*.csv is git-ignored and absent")
    doctored = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, doctored)
    target = doctored / "bank200_three_round.md"
    text = target.read_text(encoding="utf-8")
    assert "only 6 of 1403" in text, "fixture drifted; pick another anchor phrase"
    target.write_text(text.replace("only 6 of 1403", "only 7 of 1403"), encoding="utf-8")

    result = run_checker("--evidence-dir", str(doctored))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "budget claim missing or wrong" in result.stdout