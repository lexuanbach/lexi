#!/usr/bin/env python3
"""
Real-trace check of RQ2 (external validity).

The primary priority-satisfaction comparison (Table 3) is repeated with two of the
simulator's signals replaced by public traces.
  carbon intensity  the UK grid regional traces of the Carbon Intensity API (14 days at
                    half-hourly resolution, three contrasting regions), one per simulator
                    region.
  request load      the Azure Functions 2019 invocation trace (Shahrad et al., ATC 2020),
                    day 1, per-minute arrivals, rescaled to the simulator's load range.
The service DAG keeps the DeathStarBench-style shape of Sect. 5. Both traces are loaded by
sim.load_real_traces and described in data/PROVENANCE.md. Nothing is re-tuned. Lexi has
no weights, and the baselines keep the weights calibrated on the synthetic signals. The
question is whether the priority-first conclusions survive real carbon and load.

Output is ../results/real_data_results.json, with the per-method table (15 seeds, 95
percent intervals), the paired test of Lexi-strict against WS-tuned, the priority-inversion
rates and the trace provenance. The camera-ready paper cites the inversion result for
Constr-WS under the real Azure load in Sect. 6.2. The full real-trace table is in the
extended version, Sect. "Real-Trace External Validity (RQ2)". check_results.py verifies
the file against those numbers. The run takes about a minute, and everything is seeded.
"""
import os, json
import numpy as np
import experiments_core as E
from sim import Continuum, make_workload, load_real_traces
from causal import CausalModel
from controller import CausalContinuum, candidate_set, thresholded_lex_select
from baselines import (WeightedSum, ConstrainedRL, RankWeightedSum,
                       NormWeightedSum, Tchebycheff, TUNED_WS_W)

# The methods reported in the real-trace table, the same rows as the primary comparison
# (Table 3).
REPORT = ["Lexi-strict", "constrained_rl", "ws_tuned",
          "rank_ws", "norm_ws", "tcheby", "ws_default"]
NAME = {"Lexi-strict": "Lexi (no weights)", "constrained_rl": "Constr-WS",
        "ws_tuned": "WS-tuned", "rank_ws": "Rank-WS", "norm_ws": "Norm-WS",
        "tcheby": "Tcheby", "ws_default": "WS-default"}


def rq2_real(seeds=E.SEEDS):
    """Operator-facing outcomes of RQ2 under the real traces, with the protocol of
    experiments_core.e1_priority_satisfaction. Returns the per-method table and the paired
    test of Lexi-strict against WS-tuned on SLO percent."""
    res = E._eval_methods(E.MAIN_METHODS, seeds, E._curated, real_traces=True)
    table = {}
    for m in E.MAIN_METHODS:
        table[m] = {}
        for k in ("slo", "privacy", "carbon", "cost", "churn"):
            mean, lo, hi = E.boot_ci(res[m][k])
            table[m][k] = {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
    p, eff = E.paired_boot_p(res["Lexi-strict"]["slo"], res["ws_tuned"]["slo"])
    tests = {"lexi_vs_wstuned_slo": {"p": round(p, 4), "median_diff": round(eff, 4)}}
    return table, tests


def inversion_real(seeds=E.SEEDS):
    """Priority-inversion percent under the real traces.

    The protocol is that of experiments_core.e1_inversion. It is the share of decisions
    that are lexicographically worse than the strict optimum on the same shared,
    warmed-up estimates, with hysteresis off. The value is 0 for a rule that honours the
    strict order."""
    cand = candidate_set(Continuum(real_traces=True), cap=E.CAND_CAP, seed=0)
    builders = {
        "Lexi-strict":    lambda s, m: CausalContinuum(s, m, eta=E.STRICT_ETA),
        "constrained_rl": lambda s, m: ConstrainedRL(s, m),
        "ws_tuned":       lambda s, m: WeightedSum(s, m, TUNED_WS_W),
        "rank_ws":        lambda s, m: RankWeightedSum(s, m),
        "norm_ws":        lambda s, m: NormWeightedSum(s, m),
        "tcheby":         lambda s, m: Tchebycheff(s, m),
    }
    per = {n: [] for n in builders}
    for seed in seeds:
        sim = Continuum(seed=seed, real_traces=True)
        reqs = make_workload(E.HORIZON, seed=seed, real_traces=True)
        diag = CausalModel(sim, seed=seed)                 # the shared, warmed-up model
        warm = CausalContinuum(sim, diag, eta=E.STRICT_ETA)
        rng = np.random.default_rng(2000 + seed); prev = None
        for req in reqs[:E.WARMUP]:
            c = warm.act(req, cand, None)
            _, _, _, obs = sim.realize(cand[c], req, prev, rng)
            diag.update(obs, req); prev = cand[c]
        flags = {n: [] for n in builders}
        for req in reqs[E.WARMUP:]:
            est = diag.predict_all(cand, req)
            o = thresholded_lex_select(est, E.STRICT_ETA, prev_idx=None)
            for n, build in builders.items():
                c = build(sim, diag).act(req, cand, None)
                flags[n].append(1.0 if E._lex_worse(est[c], est[o]) else 0.0)
        for n in builders:
            per[n].append(100.0 * float(np.mean(flags[n])))
    out = {}
    for n in builders:
        mean, lo, hi = E.boot_ci(per[n])
        out[n] = {"inversion_pct": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
    return out


def main():
    table, tests = rq2_real()
    inv = inversion_real()
    carbon, load = load_real_traces()
    prov = {
        "carbon_regions": ["NorthScotland(r0)", "London(r1)", "Yorkshire(r2)"],
        "carbon_mean_gco2kwh": [round(float(carbon[i].mean()), 1) for i in range(3)],
        "carbon_source": "UK Carbon Intensity API (regional), 14 days half-hourly, 2024-01",
        "load_source": "Azure Functions 2019 trace (Shahrad et al. ATC'20), day-1 per-minute",
        "load_mean_std": [round(float(load.mean()), 3), round(float(load.std()), 3)],
    }
    res = {"n_seeds": len(E.SEEDS), "table": table, "paired_tests": tests,
           "inversion": inv, "provenance": prov}
    here = os.path.dirname(os.path.abspath(__file__))
    outp = os.path.join(here, "..", "results", "real_data_results.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(res, f, indent=2)

    print("RQ2 under REAL traces (carbon: UK grid regional; load: Azure Functions):")
    print(f"  {'method':20s} {'SLO%':>7} {'PII':>6} {'carbon':>7} {'cost':>6} "
          f"{'churn':>6} {'inv%':>6}")
    for m in REPORT:
        t = table[m]; iv = inv.get(m, {}).get("inversion_pct")
        ivs = f"{iv:6.2f}" if iv is not None else "   n/a"
        print(f"  {NAME[m]:20s} {t['slo']['mean']:7.2f} {t['privacy']['mean']:6.3f} "
              f"{t['carbon']['mean']:7.3f} {t['cost']['mean']:6.3f} "
              f"{t['churn']['mean']:6.3f} {ivs}")
    print("  paired Lexi vs WS-tuned on SLO:", tests["lexi_vs_wstuned_slo"])
    print("  carbon region means (gCO2/kWh):", prov["carbon_mean_gco2kwh"],
          " load mean/std:", prov["load_mean_std"])
    print("  -> written", os.path.normpath(outp))


if __name__ == "__main__":
    main()
