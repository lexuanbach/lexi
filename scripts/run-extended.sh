#!/usr/bin/env bash
# Regenerate the extended-version stress results (RQ6/RQ7, Table 9 and Sect. 12):
# robustness_results.json through the sharded runner run_incremental.py, and
# deployment_results.json in one pass. About 10 minutes on a laptop.
#
# The released results/ folder (including results/shards) is restored on exit. The
# regenerated files are kept in regenerated/ (or in $LEXI_REGEN_DIR).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
OUT="${LEXI_REGEN_DIR:-$ROOT/regenerated}"
export PYTHONDONTWRITEBYTECODE=1

SNAP="$(mktemp -d)"
cp -R "$ROOT/results" "$SNAP/results"
restore() {
  mkdir -p "$OUT"
  rm -rf "$OUT/results-extended"
  cp -R "$ROOT/results" "$OUT/results-extended"
  rm -rf "$ROOT/results"
  cp -R "$SNAP/results" "$ROOT/results"
  rm -rf "$SNAP"
}
trap restore EXIT

run() {
  local t0=$SECONDS
  "$PY" "$@" > /dev/null
  echo "   $* ($((SECONDS - t0)) s)"
}

cd "$ROOT/lexi"
echo "== robustness_results.json (run_incremental.py, shards then merge)"
for r in "0 3" "3 6" "6 9" "9 12" "12 15"; do run run_incremental.py rx1 $r; done
run run_incremental.py rx1_merge
for r in "0 3" "3 6" "6 9" "9 12" "12 15"; do run run_incremental.py rx2 $r; done
run run_incremental.py rx2_merge
for r in "0 55" "55 95" "95 135" "135 175" "175 220"; do run run_incremental.py rx3w $r; done
for w in lexi cr tc; do run run_incremental.py rx3b $w; done
run run_incremental.py rx3_merge
for r in "0 3" "3 6" "6 9" "9 12" "12 15"; do run run_incremental.py rx4 $r; done
run run_incremental.py rx4_merge
run run_incremental.py rx5
run run_incremental.py rx6
run run_incremental.py rx7
run run_incremental.py meta
echo "== deployment_results.json (deployment_experiments.py, one pass)"
run deployment_experiments.py
cd "$ROOT"
"$PY" scripts/compare-results.py "$SNAP/results" "$ROOT/results" robustness_results.json \
  deployment_results.json
"$PY" scripts/compare-results.py "$SNAP/results/shards" "$ROOT/results/shards"
echo "PASS: extended stress results regenerated. Released results restored."
