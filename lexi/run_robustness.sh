#!/usr/bin/env bash
# Run the additional experiments of robustness_experiments.py, then regenerate the figures
# of make_real_figures.py. The results go to ../results/robustness_results.json. These
# measurements belong to the extended version (see the header of robustness_experiments.py).
# This one-pass run completes, but it does not reproduce the committed robustness_results.json
# because it uses five seeds for the RX3 sweeps where the committed file uses two and three.
# To reproduce the committed file use ../scripts/run-extended.sh, which follows
# run_incremental.py.
# Usage: bash run_robustness.sh
set -e
cd "$(dirname "$0")"
echo "[run_robustness] starting $(date)"
python3 robustness_experiments.py
echo "[run_robustness] experiments done $(date); regenerating figures"
python3 make_real_figures.py
echo "[run_robustness] DONE $(date)"
