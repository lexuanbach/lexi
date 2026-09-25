#!/usr/bin/env bash
# Regenerate the results behind every camera-ready number, Fig. 2 and the tables, check
# them against the papers and compare them with the released result files.
#
#   scripts/run-full.sh           all steps, 11 to 16 minutes on a laptop
#   scripts/run-full.sh --quick   skip experiments_core.py (the 10 to 15 minute step), 2 minutes
#
# The released results/*.json files are restored when the script exits. This keeps the
# manifest valid. The regenerated files and figures are kept in regenerated/ (or in
# $LEXI_REGEN_DIR) for inspection.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
OUT="${LEXI_REGEN_DIR:-$ROOT/regenerated}"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1
export PYTHONDONTWRITEBYTECODE=1

SNAP="$(mktemp -d)"
cp "$ROOT"/results/*.json "$SNAP"/
restore() {
  mkdir -p "$OUT/results"
  cp "$ROOT"/results/*.json "$OUT/results/"
  cp "$SNAP"/*.json "$ROOT/results/"
  rm -rf "$SNAP"
}
trap restore EXIT

step() {
  local t0=$SECONDS
  echo "== $*"
  "$@"
  echo "   ($((SECONDS - t0)) s)"
}

cd "$ROOT/lexi"
if [ "$QUICK" -eq 0 ]; then
  step "$PY" experiments_core.py          # core_results.json: Table 3, Fig. 2, RQ1 to RQ5
fi
step "$PY" real_data_experiment.py        # real_data_results.json: real-trace rerun
step "$PY" extended_experiments.py        # extended_results.json: Table 1, menu, Constr-WS scale
step "$PY" decision_checks.py             # decision_checks.json: 99.7%, worked-decision scores
export LEXI_FIGURE_DIR="$OUT/figures"
step "$PY" make_camera_figures.py         # Fig. 2 of both papers
step "$PY" make_real_figures.py           # supplementary figures, which neither paper uses
step "$PY" make_tables.py                 # Tables 1 to 3 (camera-ready), Tables 1 to 9 (extended)
cd "$ROOT"
step "$PY" scripts/check-claims.py
FILES="real_data_results.json extended_results.json decision_checks.json"
[ "$QUICK" -eq 0 ] && FILES="core_results.json $FILES"
step "$PY" scripts/compare-results.py "$SNAP" "$ROOT/results" $FILES
echo "PASS: regeneration finished. Regenerated files: $OUT. Released results restored."
