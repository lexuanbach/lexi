"""
Sharded runner for the experiments of robustness_experiments.py (RX1 to RX7).

The directory keeps its earlier name, causalcontinuum. Each command runs one experiment,
or one seed range of it, and writes a shard under ../results/shards. The merge commands
combine the shards and store the result under its key in ../results/robustness_results.json,
which is the file the RQ6/RQ7 table of the extended version (measured robustness and
deployment extensions) is drawn from. The camera-ready paper does not use it. The
shards exist so that a long experiment can be run in short calls and resumed.

Aggregation across shards. The per-shard results are already means over their own seeds,
so the merge commands average them weighted by the number of seeds of each shard. The
paired tests and bootstrap intervals of a full single run are therefore not recomputed for
merged experiments, and the merged files carry means only. The rx3 result is assembled
from two seeds for the finer weight sweep and three seeds for the baseline sweeps.

Usage:
  python3 run_incremental.py rx1 0 5      # rx1 on seeds [0, 5), writes a shard
  python3 run_incremental.py rx1_merge    # merge the rx1 shards into robustness_results.json
  python3 run_incremental.py rx2 0 15     # rx2 on seeds [0, 15), then rx2_merge
  python3 run_incremental.py rx3w LO HI   # weight vectors LO to HI of the finer grid
  python3 run_incremental.py rx3b lexi    # Lexi reference for rx3 (also: cr, tc)
  python3 run_incremental.py rx3_merge    # merge the rx3 shards
  python3 run_incremental.py rx4 0 5      # then rx4_merge
  python3 run_incremental.py rx5          # unsharded
  python3 run_incremental.py rx6          # unsharded, 15 seeds
  python3 run_incremental.py rx7          # unsharded, 15 seeds
  python3 run_incremental.py meta         # write the _meta record
"""
import sys, os, json
import numpy as np
import robustness_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "robustness_results.json")
SHARD = os.path.join(HERE, "..", "results", "shards")
os.makedirs(SHARD, exist_ok=True)


def load_out():
    """Current robustness_results.json as a dict, or an empty dict if it does not exist."""
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save_out(d):
    """Write the whole results dict back to robustness_results.json."""
    json.dump(d, open(OUT, "w"), indent=2)


def merge(key, val):
    """Store val under key in robustness_results.json and leave the other keys as they are."""
    d = load_out(); d[key] = val; save_out(d)
    print("merged", key)


def main():
    """Dispatch on the command in sys.argv[1]. See the module docstring for the commands."""
    cmd = sys.argv[1]
    if cmd == "rx1":
        a, b = int(sys.argv[2]), int(sys.argv[3])
        r = R.rx1_contention_aware(seeds=list(range(a, b)))
        json.dump(r, open(os.path.join(SHARD, f"rx1_{a}_{b}.json"), "w"))
        print("rx1 shard", a, b, "done")
    elif cmd == "rx1_merge":
        # The shard rows are already means over their own seeds. They are combined by a
        # mean weighted by the number of seeds per shard. The per-seed values are not
        # stored, which means the merged file has no confidence intervals or paired test.
        import glob
        shards = sorted(glob.glob(os.path.join(SHARD, "rx1_*.json")))
        agg = {}
        rows0 = None; tot = 0
        for sf in shards:
            s = json.load(open(sf)); n = s["n_seeds"]; tot += n
            if rows0 is None:
                rows0 = {r["contention"]: {"add": 0.0, "aware": 0.0, "add_pii": 0.0,
                                           "aware_pii": 0.0} for r in s["rows"]}
            for r in s["rows"]:
                c = r["contention"]
                rows0[c]["add"] += n * r["additive_slo"]
                rows0[c]["aware"] += n * r["aware_slo"]
                rows0[c]["add_pii"] += n * r["additive_pii"]
                rows0[c]["aware_pii"] += n * r["aware_pii"]
        rows = []
        nom = None
        for c in sorted(rows0):
            v = rows0[c]
            row = {"contention": c, "additive_slo": round(v["add"] / tot, 3),
                   "aware_slo": round(v["aware"] / tot, 3),
                   "additive_pii": round(v["add_pii"] / tot, 3),
                   "aware_pii": round(v["aware_pii"] / tot, 3)}
            if nom is None:
                nom = row["additive_slo"]
            degr = row["additive_slo"] - nom
            row["recovery_frac"] = round((row["additive_slo"] - row["aware_slo"]) / degr, 3) if degr > 1e-6 else 0.0
            rows.append(row)
        merge("rx1_contention", {"n_seeds": tot, "rows": rows,
                                 "note": "M/G/1-flavoured online contention term; recovery_frac "
                                         "= share of contention-induced SLO degradation recovered "
                                         "vs the additive model (seed-weighted over shards)"})
    elif cmd == "rx2":
        a, b = int(sys.argv[2]), int(sys.argv[3])
        r = R.rx2_doubly_robust(seeds=list(range(a, b)))
        json.dump(r, open(os.path.join(SHARD, f"rx2_{a}_{b}.json"), "w"))
        print("rx2 shard", a, b, "done")
    elif cmd == "rx2_merge":
        import glob
        shards = sorted(glob.glob(os.path.join(SHARD, "rx2_*.json")))
        agg = {}; tot = 0
        for sf in shards:
            s = json.load(open(sf)); n = s["n_seeds"]; tot += n
            for r in s["rows"]:
                e = r["epsilon"]
                a_ = agg.setdefault(e, {k: 0.0 for k in ("dm_esterr", "dr_esterr",
                     "dm_slo", "dr_slo", "dm_pii", "dr_pii")})
                for k in a_:
                    a_[k] += n * r[k]
        rows = []
        for e in sorted(agg):
            v = {k: agg[e][k] / tot for k in agg[e]}
            v["esterr_reduction_pct"] = round(100.0 * (v["dm_esterr"] - v["dr_esterr"]) / v["dm_esterr"], 2) if v["dm_esterr"] > 1e-9 else 0.0
            rows.append({"epsilon": e, **{k: round(v[k], 4) for k in v}})
        merge("rx2_doubly_robust", {"n_seeds": tot, "rows": rows,
              "note": "DR (carbon-only, evidence-gated residual, Dudik et al.) vs direct "
                      "method; esterr = mean |hat g - g| on SLO-hinge+carbon over candidates; "
                      "DR is decision-safe (SLO/PII == DM) and the additive DM is already "
                      "well-specified for the priority-driving objectives (seed-weighted)"})
    elif cmd == "rx3w":
        lo, hi = int(sys.argv[2]), int(sys.argv[3])
        r = R.rx3_weight_shard(lo, hi, seeds=(0, 1))
        json.dump(r, open(os.path.join(SHARD, f"rx3w_{lo}_{hi}.json"), "w"))
        print("rx3w shard", lo, hi, "of", r["n_total"], "done")
    elif cmd == "rx3b":
        which = sys.argv[2] if len(sys.argv) > 2 else "cr"
        seeds = (0, 1, 2)
        import robustness_experiments as RE
        cand = RE.candidate_set(RE.Continuum(), cap=RE.CAND_CAP, seed=0)

        def eval_ctrl(build):
            slo, priv = [], []
            for seed in seeds:
                sim = RE.Continuum(seed=seed); reqs = RE.make_workload(RE.HORIZON, seed=seed)
                tn, orc = RE._precompute(sim, cand, reqs)
                rr = RE.run_method(build(sim, RE.CausalModel(sim, seed=seed)), sim, cand, reqs,
                                   tn, orc, realize_seed=1000 + seed)
                slo.append(rr["slo"]); priv.append(rr["privacy"])
            return round(float(np.mean(slo)), 2), round(float(np.mean(priv)), 3)
        if which == "lexi":
            s, p = eval_ctrl(lambda s, m: RE.CausalContinuum(s, m, eta=RE.STRICT_ETA))
            json.dump({"slo": s, "privacy": p}, open(os.path.join(SHARD, "rx3b_lexi.json"), "w"))
            print("lexi", s, p)
        elif which == "cr":
            grid = []
            for tol in (0.0, 0.02, 0.05, 0.1):
                for wp in (0.34, 0.6, 0.8):
                    wrest = np.array([wp, (1 - wp) / 2, (1 - wp) / 2])
                    def build(s, m, tol=tol, wrest=wrest):
                        c = RE.ConstrainedRL(s, m, tol=tol); c.W_REST = wrest; return c
                    s, p = eval_ctrl(build)
                    grid.append({"tol": tol, "w_priv": round(wp, 2), "slo": s, "privacy": p})
            json.dump({"grid": grid, "best": min(grid, key=lambda r: (r["slo"], r["privacy"]))},
                      open(os.path.join(SHARD, "rx3b_cr.json"), "w"))
            print("cr done")
        elif which == "tc":
            grid = []
            for rho in (1e-4, 1e-3, 1e-2):
                for wp in (0.10, 0.4, 0.7):
                    W = np.array([0.85, wp, 0.0, 0.05]); W = W / W.sum()
                    def build(s, m, rho=rho, W=W):
                        t = RE.Tchebycheff(s, m, W); t.RHO = rho; return t
                    s, p = eval_ctrl(build)
                    grid.append({"rho": rho, "w_priv": round(wp, 2), "slo": s, "privacy": p})
            json.dump({"grid": grid, "best": min(grid, key=lambda r: (r["slo"], r["privacy"]))},
                      open(os.path.join(SHARD, "rx3b_tc.json"), "w"))
            print("tc done")
    elif cmd == "rx3_merge":
        import glob
        lexi = json.load(open(os.path.join(SHARD, "rx3b_lexi.json")))
        cr = json.load(open(os.path.join(SHARD, "rx3b_cr.json")))
        tc = json.load(open(os.path.join(SHARD, "rx3b_tc.json")))
        vecs = []; ntot = 0
        for sf in sorted(glob.glob(os.path.join(SHARD, "rx3w_*.json"))):
            s = json.load(open(sf)); ntot = s["n_total"]; vecs.extend(s["vectors"])
        lexi_slo, lexi_priv = lexi["slo"], lexi["privacy"]
        meet = sum(1 for v in vecs if v["slo"] <= 8.0 and v["privacy"] <= 0.1)
        dom = sum(1 for v in vecs if v["slo"] <= lexi_slo + 1e-6 and v["privacy"] <= lexi_priv + 1e-6)
        best = min(vecs, key=lambda v: (v["slo"], v["privacy"]))
        merge("rx3_expanded_sweeps", {
            "n_seeds_sweep": 2, "n_seeds_baseline": 3,
            "weight_sweep": {"n_weights": len(vecs), "n_grid_total": ntot,
                             "frac_meeting_target": round(meet / len(vecs), 4),
                             "frac_dominating_lexi": round(dom / len(vecs), 4),
                             "best_ws": best, "lexi": lexi},
            "constrained_rl_sweep": cr, "tchebycheff_sweep": tc,
            "note": "finer simplex (step 0.05, SLO-weight>=0.4 region, >84 vectors) + baseline "
                    "hyperparameter sweeps; best-case baselines still price the objectives"})
    elif cmd == "rx4":
        a, b = int(sys.argv[2]), int(sys.argv[3])
        r = R.rx4_hinge_margin_sensitivity(seeds=list(range(a, b)))
        json.dump(r, open(os.path.join(SHARD, f"rx4_{a}_{b}.json"), "w"))
        print("rx4 shard", a, b, "done")
    elif cmd == "rx4_merge":
        import glob
        shards = sorted(glob.glob(os.path.join(SHARD, "rx4_*.json")))
        tot = 0; hinge = {}; man = {}; auto = {}
        for sf in shards:
            s = json.load(open(sf)); n = s["n_seeds"]; tot += n
            for x in s["hinge_scale_invariance"]:
                h = hinge.setdefault(x["rho"], {"slo": 0.0, "privacy": 0.0})
                h["slo"] += n * x["slo"]; h["privacy"] += n * x["privacy"]
            for x in s["manual_margin"]:
                mm = man.setdefault(x["margin_ms"], {"slo": 0.0, "privacy": 0.0})
                mm["slo"] += n * x["slo"]; mm["privacy"] += n * x["privacy"]
            for x in s["auto_margin"]:
                au = auto.setdefault(x["k"], {"slo": 0.0, "privacy": 0.0, "mean_delta_ms": 0.0})
                au["slo"] += n * x["slo"]; au["privacy"] += n * x["privacy"]
                au["mean_delta_ms"] += n * x["mean_delta_ms"]
        merge("rx4_hinge_margin", {"n_seeds": tot,
              "hinge_scale_invariance": [{"rho": r_, "slo": round(hinge[r_]["slo"] / tot, 3),
                                          "privacy": round(hinge[r_]["privacy"] / tot, 3)} for r_ in sorted(hinge)],
              "manual_margin": [{"margin_ms": m_, "slo": round(man[m_]["slo"] / tot, 3),
                                 "privacy": round(man[m_]["privacy"] / tot, 3)} for m_ in sorted(man)],
              "auto_margin": [{"k": k_, "slo": round(auto[k_]["slo"] / tot, 3),
                               "privacy": round(auto[k_]["privacy"] / tot, 3),
                               "mean_delta_ms": round(auto[k_]["mean_delta_ms"] / tot, 2)} for k_ in sorted(auto)],
              "note": "hinge rho is a monotone reparam of obj 0 => strict Lexi physically invariant; "
                      "auto delta = k*std(latency residual), self-calibrating, no units (seed-weighted)"})
    elif cmd == "rx5":
        merge("rx5_scaling", R.rx5_scaling())
    elif cmd == "rx6":
        merge("rx6_priority_orders", R.rx6_priority_orders(seeds=list(range(15))))
    elif cmd == "rx7":
        merge("rx7_compliance", R.rx7_compliance(seeds=list(range(15))))
    elif cmd == "meta":
        d = load_out(); d["_meta"] = {"n_seeds": 15, "note": "robustness experiments RX1 to RX7, "
                                      "merged from shards by run_incremental.py"}; save_out(d)
        print("meta set; keys:", list(d.keys()))
    else:
        print("unknown", cmd)


if __name__ == "__main__":
    main()
