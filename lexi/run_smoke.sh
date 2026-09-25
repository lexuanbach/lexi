#!/usr/bin/env bash
# run_smoke.sh: a fast end-to-end wiring check of the Lexi artifact (a few seconds).
#
# It runs a small subset of the E1 priority-satisfaction experiment (3 methods, 3 seeds,
# horizon 400) through the real code paths, meaning the simulator, the outcome model, the
# lexicographic controller and the weighted-sum baselines. It then checks the qualitative
# ordering that the paper relies on (Table 3, RQ2).
#
#     Lexi (no weights) reaches a low SLO and PII operating point, the untuned weighted
#     sum does not, and the tuned weighted sum sits in between.
#
# The subset is a sanity check and not the measured result. The full 15-seed run is
# `python3 experiments_core.py` (10 to 15 minutes on a laptop). The committed results are verified
# against the manuscript by `python3 check_results.py`.
#
# Exit status 0 means the smoke test passed. Any other status means that something is broken.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Lexi smoke test (subset: 3 methods x 3 seeds x horizon 400) =="
python3 - <<'PY'
import time
t0 = time.perf_counter()
from experiments_core import _eval_methods, _curated
import numpy as np

METHODS = ["Lexi-strict", "ws_tuned", "ws_default"]
SEEDS = [0, 1, 2]
HORIZON = 400

out = _eval_methods(METHODS, SEEDS, _curated, horizon=HORIZON)
mean = lambda m, k: float(np.mean(out[m][k]))

print(f"\n  {'method':14s} {'SLO%':>7s} {'PII':>7s}")
for m in METHODS:
    print(f"  {m:14s} {mean(m,'slo'):7.2f} {mean(m,'privacy'):7.3f}")

lexi_slo, lexi_pii = mean("Lexi-strict", "slo"), mean("Lexi-strict", "privacy")
wsd_slo,  wsd_pii  = mean("ws_default", "slo"),  mean("ws_default", "privacy")
wst_slo            = mean("ws_tuned",   "slo")

checks = [
    ("Lexi reaches a low SLO operating point (< 10%)",      lexi_slo < 10.0),
    ("Lexi keeps PII exposure low (< 0.5 of 2 stages)",     lexi_pii < 0.5),
    ("untuned weighted sum is far worse on SLO than Lexi",  wsd_slo > lexi_slo + 5.0),
    ("untuned weighted sum leaks much more PII than Lexi",  wsd_pii > lexi_pii + 0.5),
    ("tuned weighted sum sits between Lexi and untuned",    lexi_slo <= wst_slo <= wsd_slo),
]
print()
ok_all = True
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    ok_all &= ok

dt = time.perf_counter() - t0
print(f"\n  ran in {dt:.1f}s")
if not ok_all:
    raise SystemExit("SMOKE FAILED: ordering/sanity check did not hold")
print("SMOKE PASSED -- artifact wiring is healthy.")
print("Next: `python3 experiments_core.py` (full run) then `python3 check_results.py`.")
PY
