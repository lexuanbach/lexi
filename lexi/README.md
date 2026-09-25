# Lexi code base

Python code of the simulator, the Lexi controller, the baselines and every experiment of
"Lexi: Priority-First Service Orchestration". It needs Python 3.11 or newer with numpy,
and matplotlib for the figures. It uses no GPU, no network access and no language model.
All randomness comes from seeded `numpy.random.default_rng` generators.

The class and file names keep the project's earlier name, CausalContinuum. The classes
`CausalContinuum*` in `controller.py` are the controller the paper calls Lexi. The result
key `constrained_rl` is the baseline the paper calls Constr-WS.

The priority order of the paper is SLO, then privacy, then carbon, then cost.

## Files

| File | Role |
| :-- | :-- |
| `sim.py` | Continuum simulator. Config A (7 nodes, 6-stage DAG, 2 PII stages, Table 2), config B (10 nodes, 8 stages, 3 PII stages), synthetic scaled fleets, the geo-diurnal workload, the noise-free outcome `expected_all` and the executed outcome `realize`. |
| `causal.py` | Counterfactual outcome model: per-node online ridge regressions for latency and carbon, running means for cost, placement metadata for privacy. Also the contention-aware and doubly-robust variants of the extended version. |
| `controller.py` | Thresholded lexicographic selection (Def. 2), the stability hold, the 40-candidate menu, the priority oracle, and the auto-slack, margin and order variants. |
| `baselines.py` | WS-default, WS-tuned, Rank-WS, Norm-WS, Tchebycheff, Lex-LP, Constr-WS, SO-RL, static-greedy and binpack. |
| `experiments_core.py` | Experiments E1 to E11 of both papers. Writes `../results/core_results.json`. |
| `real_data_experiment.py` | The Table 3 comparison under the UK carbon and Azure load traces. Writes `../results/real_data_results.json`. |
| `extended_experiments.py` | Worked decision (Table 1), menu anatomy with the oracle, Constr-WS privacy-scale sweep, prediction accuracy. Writes `../results/extended_results.json`. |
| `decision_checks.py` | Oracle agreement (99.7%), the weighted-sum scores on the Table 1 matrix, the effect of the stability hold, the menu carbon range. Writes `../results/decision_checks.json`. |
| `robustness_experiments.py`, `run_incremental.py` | Extended-version stress study RX1 to RX7 (Table 9). `run_incremental.py` runs it in shards and merges them into `../results/robustness_results.json`. |
| `deployment_experiments.py` | Extended-version stress study DX1 to DX7 (Table 9, there is no DX3). Writes `../results/deployment_results.json`, whose keys keep the older prefix `r2q` (for example `r2q1_safe_probe` for DX1). |
| `improve_lexi.py`, `improve_lexi2.py` | Diagnosis study behind the SLO safety margin. Writes `../results/improve_lexi*.json`. No paper number depends on it. |
| `make_camera_figures.py` | Fig. 2 of both papers (`fig_results.pdf`). |
| `make_tables.py` | Prints every results table of both papers and Table 2 from the committed files. |
| `make_real_figures.py` | Supplementary figures that appear in neither paper. |
| `check_results.py` | The original 97-claim gate on `core_results.json` and `real_data_results.json`. |
| `make_manifest.py` | Digest list of `../results/*.json` and `data/*` for the working repository. In this package use `../scripts/check-manifest.py` instead. Running `make_manifest.py` without `--check` would overwrite the package manifest with a subset. |
| `run_smoke.sh` | Wiring check in a few seconds. |
| `run_robustness.sh` | One-pass RX1 to RX7 run, then the supplementary figures. It completes. Its `robustness_results.json` differs from the committed one because it runs the RX3 sweeps on five seeds where the committed file uses two and three. `../scripts/run-extended.sh` reproduces the committed file. |
| `data/` | Real traces and `PROVENANCE.md`. |

## Commands

The package-level scripts in `../scripts/` wrap these commands and are the recommended
entry points. See `../README.md` for runtimes and the claim-by-claim list.

```bash
export PYTHONDONTWRITEBYTECODE=1
python3 check_results.py          # 97 claims on the committed results, seconds
bash run_smoke.sh                 # wiring check, seconds
python3 experiments_core.py       # 10 to 15 minutes
python3 real_data_experiment.py   # about 1 minute
python3 extended_experiments.py   # about 30 seconds
python3 decision_checks.py        # about 20 seconds
LEXI_FIGURE_DIR=/tmp/lexi-figs python3 make_camera_figures.py   # Fig. 2, seconds
python3 make_tables.py            # Tables 1 to 9, seconds
```

## Determinism

Every quantity is reproducible bit for bit from the seeds: the workload uses `seed`, the
execution noise uses `1000 + seed`, the menu uses seed 0 and the random menu uses
`100 + seed`. The only values that change between runs are wall-clock timings: the
decision-latency microbenchmark (`scalability` in `core_results.json`), the scaling study
(`rx5_scaling` in `robustness_results.json`) and the `runtime_s` fields.
