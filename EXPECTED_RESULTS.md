# Expected results

Each command below exits with status 0 on success. The listed lines are the last lines it
prints. Numbers in the tables are the committed values and appear in `CLAIM_MATRIX.md`.

## `scripts/verify.sh`

```text
PASS: 89 manifest entries
PASS: package hygiene, local-path and credential checks
...
97/97 claims match the manuscript.
RESULT: PASS -- artifact results are consistent with the paper.
...
214/214 paper claims match (117 camera-ready, 97 extended). 3 machine-dependent timing values printed as INFO.
PASS: camera-ready PDF has 10 pages
PASS: artifact verification complete
```

## `scripts/run-smoke.sh`

```text
  method            SLO%     PII
  Lexi-strict       1.50   0.050
  ws_tuned          2.67   0.017
  ws_default       28.17   1.913
...
SMOKE PASSED -- artifact wiring is healthy.
```

## `scripts/run-full.sh --quick` and `scripts/run-full.sh`

Each step prints its runtime. `make_tables.py` prints the tables of both papers, for
example camera-ready Table 3:

```text
  Method               SLO%   PII Carbon  Cost Churn  Inv%
  Lexi (no weights)    1.46  0.05   0.98  3.21  1.35   0.0
  Constr-WS            1.32  0.05   0.98  3.19  1.32   0.7
  WS-tuned             3.19  0.01   0.99  3.20  1.28   3.7
  WS-default          23.03  1.94   0.45  1.22  0.65  94.7
```

`make_tables.py` rounds half up on the stored three-decimal value. Where the stored value
ends in exactly 5 the paper sometimes rounds down (23.025 printed as 23.02, 0.995 as 0.99).
Both are within the tolerance of the claim checks.

The run ends with the claim checks on the regenerated files and the comparison with the
released files:

```text
214/214 paper claims match (117 camera-ready, 97 extended). 3 machine-dependent timing values printed as INFO.
  SAME             core_results.json : e10_margin     (full run only, one line per key)
  TIMING           core_results.json : _meta          (full run only)
  TIMING           core_results.json : scalability    (full run only)
  IDENTICAL        real_data_results.json (byte for byte)
  IDENTICAL        extended_results.json (byte for byte)
  IDENTICAL        decision_checks.json (byte for byte)
PASS: regenerated results equal the released ones (timing keys excepted)
PASS: regeneration finished. Regenerated files: .../regenerated. Released results restored.
```

`regenerated/figures/fig_results.pdf` is Fig. 2 of both papers.

## `scripts/run-extended.sh`

```text
  SAME             robustness_results.json : rx1_contention
  ...
  TIMING           robustness_results.json : rx5_scaling
  ...
  SAME             deployment_results.json : r2q1_safe_probe
  ...
  TIMING           deployment_results.json : _meta
PASS: regenerated results equal the released ones (timing keys excepted)
  IDENTICAL        rx1_0_3.json (byte for byte)
  ...
PASS: regenerated results equal the released ones (timing keys excepted)
PASS: extended stress results regenerated. Released results restored.
```

## Values that change between runs

Only wall-clock timings: the decision-latency points and slope in `core_results.json`
(`scalability`), the scaling study in `robustness_results.json` (`rx5_scaling`) and the
`runtime_s` fields. The papers quote them as machine-dependent: under 0.1 ms per decision
at 40 candidates and at most 0.75 ms at 2000 candidates on the authors' laptop.
