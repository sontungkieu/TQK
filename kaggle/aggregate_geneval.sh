#!/usr/bin/env bash
# Compute the phase-2 report inside the session that owns the data, so pack_artifacts.sh can
# archive it with the run.
#
# Deliberately not `set -e`: this step runs after several hours of generation and behind an
# expensive GenEval pass, so a defect in the report script must not turn a finished run into a
# failed job. The measurements are always packed; the report is best effort and its failure is
# visible in the step log.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${AGGREGATE_EXP_DIR:-$ROOT/exps/single_stage_validation_553}"

echo "[aggregate] python=$(command -v python) $("python" -V 2>&1)"
echo "[aggregate] exp=$EXP"
echo "[aggregate] metrics present: $(find "$EXP/metrics" -type f 2>/dev/null | wc -l) | geneval results: $(find "$EXP/geneval_results" -type f 2>/dev/null | wc -l)"

"python" "$EXP/aggregate.py"
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$EXP/FINAL_REPORT.md" ]; then
  echo "[aggregate] report written: $EXP/FINAL_REPORT.md"
  head -n 40 "$EXP/FINAL_REPORT.md"
else
  echo "[aggregate] FAILED rc=$rc; continuing so the measured data is still packed"
fi
exit 0
