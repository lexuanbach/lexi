# Lexi: artifact of the ICSOC 2026 paper

This package accompanies **Lexi: Priority-First Service Orchestration** (ICSOC 2026). It
holds the camera-ready paper, the extended version, the simulator and controller code,
the real traces with their provenance, every result file behind the numbers of both
papers, and scripts that verify and regenerate them. Public repository:
https://github.com/lexuanbach/lexi.

All experiments run in a discrete-event simulator on one CPU core. No GPU, cluster,
network access, credentials or language model is needed.

## Contents

| Path | Content |
| :-- | :-- |
| `paper/camera-ready.pdf` | The 10-page ICSOC 2026 paper. |
| `paper/extended.pdf` | The extended version (40 pages) with proofs, RQ5 to RQ7 and the real-trace rerun. |
| `lexi/` | Python code: simulator, outcome model, controller, baselines, experiments, figure and table scripts. See `lexi/README.md`. |
| `lexi/data/` | UK Carbon Intensity API and Azure Functions 2019 trace excerpts, with `PROVENANCE.md`. |
| `results/` | Committed result files (JSON). `results/shards/` and `results/_dx*.json` are the pieces from which `robustness_results.json` and `deployment_results.json` were assembled. `results/raw/` holds two per-step traces from the first code version, which no paper number uses. |
| `scripts/` | Verification and regeneration entry points (below). |
| `CLAIM_MATRIX.md` | Every number of the camera-ready and the headline numbers of the extended version, with result file, key and command. |
| `EXPECTED_RESULTS.md` | What each command prints when it succeeds. |
| `REPRODUCIBILITY.md` | Environment, determinism, runtimes and known limits in detail. |
| `requirements.txt` | Pinned Python packages. |
| `MANIFEST.sha256` | SHA-256 digest of every file. |

Names used in the code: the controller classes `CausalContinuum*` (the project's earlier
name) implement Lexi. The result key `constrained_rl` is the baseline the papers call
Constr-WS. Node `edge_A` is E0 in the papers, `edge_B` is E1, `edge_C` is E2, `cloud_A`
is C0, `cloud_C` is C2, `srvl_B` is S1 and `srvl_C` is S2.

## Requirements

- Tested on macOS 26.5 (Apple M5 Max, arm64) with Python 3.13.5 in a fresh virtual
  environment, and with Python 3.14.6. Python 3.11 or newer is required by numpy 2.4.
  The code is plain Python and numpy and should run on Linux and Windows, which were not
  tested.
- Packages: `requirements.txt` pins numpy 2.4.6 and matplotlib 3.10.9 with their
  dependencies. With these versions every result file regenerates byte for byte and
  Fig. 2 renders pixel-identical to the paper. numpy 2.5.3 with matplotlib 3.11.2 gives
  the same result files and a figure that differs only in anti-aliasing.
- Hardware: any laptop. One core, under 1 GB of memory, about 5 MB of disk.
- Optional: `pdfinfo` (poppler) for the page-count check in `verify.sh`.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONDONTWRITEBYTECODE=1   # keeps __pycache__ out of the package
```

## Verify, then reproduce

| Step | Command | Time (M5 Max) | What it does |
| :-- | :-- | :-- | :-- |
| 1 | `scripts/verify.sh` | 2 s | Checks the manifest and package hygiene, then checks 97 code-base claims (`lexi/check_results.py`) and 214 paper claims (`scripts/check-claims.py`) against the committed results. |
| 2 | `scripts/run-smoke.sh` | 3 s | Runs 3 methods over 3 seeds for 400 steps through the real code and checks the method ordering of Table 3. |
| 3 | `scripts/run-full.sh --quick` | 2 min | Regenerates the real-trace, extended and decision-check results, Fig. 2 and all tables, runs the claim checks on the regenerated files and compares them with the released ones. |
| 4 | `scripts/run-full.sh` | 11 to 16 min | Step 3 plus `experiments_core.py`, which regenerates `core_results.json` (Table 3, Fig. 2, RQ1 to RQ5). |
| 5 | `scripts/run-extended.sh` | 10 min | Regenerates the extended-version stress study (Table 9, Sect. 12): `robustness_results.json` through its shards, and `deployment_results.json`. |

Steps 3 to 5 restore the released `results/` folder on exit, which keeps the manifest
valid. The regenerated files and figures stay in `regenerated/` for inspection.
`scripts/compare-results.py` reports, for each regenerated file, which top-level keys equal
the released ones. Only wall-clock timing keys may differ.

## Paper item to command

Every command below runs from `lexi/`. The figure scripts write to `$LEXI_FIGURE_DIR`, or
to `figures/` when it is not set.

| Paper item | Result file (key) | Regenerate | Render |
| :-- | :-- | :-- | :-- |
| Camera-ready Fig. 1 counts (40, 11, 4, 1, 1) | `extended_results.json` (`D_worked_example.band_trace`) | `python3 extended_experiments.py D` | `python3 make_tables.py camera` |
| Camera-ready Table 1, extended Table 1 | `extended_results.json` (`D_worked_example`) | `python3 extended_experiments.py D` | `python3 make_tables.py` |
| Camera-ready Sect. 3 weighted-sum scores, extended Sect. 3.3 and 4.5 | `decision_checks.json` (`worked_decision`) | `python3 decision_checks.py` | printed by the same command |
| Table 2 (both papers) | `sim.py` (`build_nodes`) | configuration | `python3 make_tables.py camera` |
| Camera-ready Table 3, extended Table 3 | `core_results.json` (`e1_satisfaction`, `e1_inversion`) | `python3 experiments_core.py` | `python3 make_tables.py` |
| Fig. 2 (both papers) | `core_results.json` (`e2_weight_sweep`, `e3_scale_invariance`, `e7_misspecification`), `extended_results.json` (`B_constrrl_scale`) | `python3 experiments_core.py`, `python3 extended_experiments.py B` | `python3 make_camera_figures.py` |
| Camera-ready Sect. 6.3, extended Table 5 | `extended_results.json` (`A_menu`), `core_results.json` (`e5_candidate`, `e6_generalization`) | `python3 extended_experiments.py A`, `python3 experiments_core.py` | `python3 make_tables.py extended` |
| 99.7% optimum agreement (camera-ready Sect. 6.3, extended Sect. 9.4, 9.5) | `decision_checks.json` (`oracle_agreement`) | `python3 decision_checks.py` | printed by the same command |
| Prediction accuracy (camera-ready Sect. 6.3, extended Sect. 9.5) | `extended_results.json` (`C_pred_acc`) | `python3 extended_experiments.py C` | `python3 make_tables.py extended` |
| Extended Table 4 | `core_results.json` (`e3_scale_invariance`), `extended_results.json` (`B_constrrl_scale`) | as Fig. 2 | `python3 make_tables.py extended` |
| Extended Table 6 | `core_results.json` (`e7_misspecification`), `robustness_results.json` (`rx1_contention`) | `python3 experiments_core.py`, `scripts/run-extended.sh` | `python3 make_tables.py extended` |
| Extended Table 7 | `core_results.json` (`e11_conflict_stress`) | `python3 experiments_core.py` | `python3 make_tables.py extended` |
| Extended Table 8, Constr-WS 14.3% (camera-ready Sect. 6.2) | `real_data_results.json` | `python3 real_data_experiment.py` | `python3 make_tables.py extended` |
| Extended Sect. 9.6 (margin, stability, decision time) | `core_results.json` (`e10_margin`, `e4_stability`, `scalability`) | `python3 experiments_core.py` | `python3 make_tables.py extended` |
| Extended Table 9 and Sect. 12 | `robustness_results.json`, `deployment_results.json` | `scripts/run-extended.sh` | `python3 make_tables.py extended` |

`CLAIM_MATRIX.md` lists each individual number.

## Determinism

Every result is a deterministic function of the seeds and reproduces bit for bit. Seeds
0 to 14 drive the main runs, seeds 0 to 5 the weight sweep and 0 to 7 the other sweeps.
Execution noise uses seed `1000 + s`. The only values that change between runs are
wall-clock timings: the decision-latency microbenchmark (`scalability` in
`core_results.json`), the scaling study (`rx5_scaling` in `robustness_results.json`) and the
`runtime_s` fields. The papers state these as machine-dependent, and the claim checks
print them as INFO without judging them.

## External resources

None are needed. The two real traces are included as small JSON excerpts:
`lexi/data/real_carbon_uk.json` (UK Carbon Intensity API, regional half-hourly forecasts,
1 to 15 January 2024) and `lexi/data/real_arrivals_azure.json` (Azure Functions 2019
trace, day 1, per-minute invocation totals). `lexi/data/PROVENANCE.md` gives the sources
and the aggregation. Re-downloading them is optional and needs no account.

## Notes on code and papers

None of these changes a result or a conclusion of either paper. Details are in
`REPRODUCIBILITY.md`.

- The evaluated controller applies a stability hold before the cost tie-break: it keeps
  the previous placement when that placement survives all three bands. The camera-ready
  reports this in Sect. 5: Lex-LP, the same rule without the hold, matches Lexi on all
  24,000 main-run decisions (`core_results.json`, `lexlp_agreement`). The hold never
  changes a strict decision in the main run (`decision_checks.json`,
  `hysteresis_effect`).
- `results/improve_lexi.json`, row `Lexi-thr(ship)`, was produced with an earlier default
  slack that had 0.04 on the SLO hinge. The current code gives 1.425% SLO for that row.
  No paper number uses this file.
- `lexi/robustness_experiments.py` run on its own completes, but it uses five seeds for the
  RX3 sweeps. `scripts/run-extended.sh` uses the committed seed counts and reproduces the
  committed file.

## License and citation

Code: Apache-2.0 (`LICENSE`, `LICENSES/Apache-2.0.txt`). Documentation and result data:
CC-BY-4.0 (`LICENSES/CC-BY-4.0.txt`). Third-party traces and the paper PDFs keep their own
terms (`THIRD_PARTY_NOTICES.md`). Please cite the ICSOC 2026 paper (`CITATION.cff`).
