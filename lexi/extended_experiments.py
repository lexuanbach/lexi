#!/usr/bin/env python3
"""
Four measurements that back the menu, Constr-WS, prediction-accuracy and worked-decision
claims of the paper. Their results are in ../results/extended_results.json.

The directory keeps its earlier name, causalcontinuum. Nothing here is new modelling.
Every function drives the same simulator, outcome model, controller and baselines as
experiments_core.py, with the same HORIZON, WARMUP, CAND_CAP and per-seed realisation
seeds (1000 + seed), which means the numbers compose with core_results.json. Runs are
deterministic.

  A  measure_menu                 key A_menu. Menu anatomy on the curated and on a uniformly
     random menu, with a per-step oracle that plays the priority-optimal candidate on true
     outcomes. It separates two explanations for the rise in SLO violations under a random
     menu. If the oracle on that menu rises as much as Lexi does, the menu limits what
     any rule can achieve and the rule itself is not degrading. Backs the menu paragraph
     of RQ4 (Sect. 6.3, 8 seeds).
  B  measure_constrrl_scale       key B_constrrl_scale. The privacy-scale sweep for Constr-WS,
     the one baseline besides Lexi that needs no tuning of weights for the SLO. Backs the
     Constr-WS comparison of Sect. 6.2 and the Constr-WS curve of Fig. 2(b).
  C  measure_prediction_accuracy  key C_pred_acc. Mean absolute error of the outcome model
     in physical units over the whole 40-candidate menu, including the 39 candidates not
     executed at each step, which is the counterfactual claim itself. Backs the accuracy
     sentence at the end of RQ4 (15 seeds).
  D  worked_example               key D_worked_example. One decision replayed level by level,
     seed 0 and step 904, from the signals through the predicted objective matrix and the
     filtering to the placement. Backs the worked decision of Sect. 3 and Table 1.

The full run takes under a minute on a laptop (about 30 s on an Apple M5).

Usage:
    python3 extended_experiments.py          # all four
    python3 extended_experiments.py A        # one part (A, B, C or D)
"""
import json, os, sys
import numpy as np

# Resolve imports against this file's own directory so that the script gives the same
# result from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sim import Continuum, make_workload, LAT, PRIV, CARB, COST, SCALES, SLO_MARGIN
from causal import CausalModel
from controller import CausalContinuum, candidate_set
from baselines import ConstrainedRL
from experiments_core import (_precompute, run_method, _curated, _random_cand,
                              _Rescaled, HORIZON, WARMUP, CAND_CAP)

OUTPATH = os.path.join(_HERE, "..", "results", "extended_results.json")

# The decision step of the worked example (Sect. 3, "A Worked Decision"). Any step after
# the warm-up would do, and the paper quotes this one.
WORKED_SEED, WORKED_STEP = 0, 904


class OracleMethod:
    """Plays the per-step priority-optimal candidate computed on true outcomes.

    It is the best that any selection rule can do on a given menu, because it sees the
    noise-free outcomes and takes the lexicographic optimum (pi*_t of Thm. 1). Part A uses
    it to tell the effect of the menu from the effect of the rule. oracle is the
    per-step index array returned by experiments_core._precompute.
    """
    name = "oracle"

    def __init__(self, sim, oracle):
        self.oracle = oracle
        self.i = 0

    def reset(self):
        self.i = 0

    def act(self, req, cand, prev_idx):
        j = int(self.oracle[self.i]); self.i += 1
        return j

    def learn(self, *a):
        pass


# A : menu anatomy
def measure_menu(seeds=list(range(8))):
    """Menu anatomy and the oracle-against-Lexi gap, on the curated and random menus.

    For the post-warmup steps it records the share of candidates that meet the SLO, the
    share that keep both PII stages private and the share that do both, all on noise-free
    outcomes. It then records the realised SLO percent and PII exposure of the oracle and
    of strict Lexi, and the mean normalised gap of Lexi to the oracle on the hinge and
    privacy coordinates."""
    rows = {}
    for label, fn in (("curated", _curated), ("random", _random_cand)):
        feas, clean, both, orc_slo, lexi_slo, lexi_priv = [], [], [], [], [], []
        gap0, gap1 = [], []
        orc_priv = []
        for seed in seeds:
            sim = Continuum(seed=seed)
            cand = fn(sim, seed)
            reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            # How much of the menu is admissible on each objective, per step.
            f, c, b = [], [], []
            for step in range(WARMUP, HORIZON):
                raw = sim.expected_all(cand, reqs[step])
                ok_slo = raw[:, LAT] <= sim.slo
                ok_pii = raw[:, PRIV] <= 0.0
                f.append(ok_slo.mean()); c.append(ok_pii.mean())
                b.append((ok_slo & ok_pii).mean())
            feas.append(float(np.mean(f)) * 100)
            clean.append(float(np.mean(c)) * 100)
            both.append(float(np.mean(b)) * 100)
            # What the best possible choice from this menu achieves once realised.
            ro = run_method(OracleMethod(sim, orc), sim, cand, reqs, tn, orc,
                            realize_seed=1000 + seed)
            orc_slo.append(ro["slo"]); orc_priv.append(ro["privacy"])
            # What Lexi achieves, and its gap to that optimum.
            rl = run_method(CausalContinuum(sim, CausalModel(sim, seed=seed),
                                            eta=np.zeros(3)),
                            sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            lexi_slo.append(rl["slo"]); lexi_priv.append(rl["privacy"])
            g = rl["obj_gap"][WARMUP:]
            gap0.append(float(np.mean(g[:, 0]))); gap1.append(float(np.mean(g[:, 1])))
        rows[label] = {
            "pct_cand_slo_feasible": round(float(np.mean(feas)), 1),
            "pct_cand_pii_clean": round(float(np.mean(clean)), 1),
            "pct_cand_both": round(float(np.mean(both)), 1),
            "oracle_slo_pct": round(float(np.mean(orc_slo)), 2),
            "oracle_pii": round(float(np.mean(orc_priv)), 3),
            "lexi_slo_pct": round(float(np.mean(lexi_slo)), 2),
            "lexi_pii": round(float(np.mean(lexi_priv)), 3),
            "lexi_mean_gap_obj0_norm": round(float(np.mean(gap0)), 5),
            "lexi_mean_gap_obj1_norm": round(float(np.mean(gap1)), 5),
        }
    return {"n_seeds": len(seeds), "sets": rows}


# B : Constr-WS scale sweep
def measure_constrrl_scale(factors=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
                           seeds=list(range(8))):
    """Constr-WS with the privacy coordinate rescaled as the optimiser sees it.

    A factor below 1 under-scales privacy. The other columns and the true outcomes are
    unchanged. The keys constr_rl_slo and constr_rl_priv keep the historical class
    name."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    rows = []
    for c in factors:
        slo, priv = [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            base = CausalModel(sim, seed=seed)
            model = _Rescaled(base, 1, c) if c != 1.0 else base
            r = run_method(ConstrainedRL(sim, model), sim, cand, reqs, tn, orc,
                           realize_seed=1000 + seed)
            slo.append(r["slo"]); priv.append(r["privacy"])
        rows.append({"factor": c,
                     "constr_rl_slo": round(float(np.mean(slo)), 2),
                     "constr_rl_priv": round(float(np.mean(priv)), 3)})
    return {"n_seeds": len(seeds), "coord": "privacy", "rows": rows}


# C : prediction accuracy
def measure_prediction_accuracy(seeds=list(range(15))):
    """Mean absolute error of the counterfactual model over the whole candidate menu.

    Errors are measured in physical units (ms, gCO2, dollars) after the warm-up, along the
    trajectory of strict Lexi, which means the model is scored on the states it actually visits.
    The latency error uses the raw predicted latency, and the hinge error is on the
    normalised scale and also converted to ms through SLO_MARGIN."""
    lat_mae, carb_mae, cost_mae, hinge_mae = [], [], [], []
    lat_scale = []
    for seed in seeds:
        sim = Continuum(seed=seed)
        cand = candidate_set(sim, cap=CAND_CAP, seed=0)
        reqs = make_workload(HORIZON, seed=seed)
        model = CausalModel(sim, seed=seed)
        ctrl = CausalContinuum(sim, model, eta=np.zeros(3))
        rng = np.random.default_rng(1000 + seed)
        prev_idx = prev_place = None
        el, ec, eo, eh, ls = [], [], [], [], []
        for step, req in enumerate(reqs):
            est = model.predict_all(cand, req)          # normalised, [n, 4]
            truth_raw = sim.expected_all(cand, req)     # physical units
            truth = sim.lex_normalize(truth_raw)
            if step >= WARMUP:
                lat_hat = model.predict_raw_latency(cand, req)
                el.append(float(np.mean(np.abs(lat_hat - truth_raw[:, LAT]))))
                ls.append(float(np.mean(truth_raw[:, LAT])))
                ec.append(float(np.mean(np.abs(est[:, CARB] * SCALES[CARB]
                                               - truth_raw[:, CARB]))))
                eo.append(float(np.mean(np.abs(est[:, COST] * SCALES[COST]
                                               - truth_raw[:, COST]))))
                eh.append(float(np.mean(np.abs(est[:, LAT] - truth[:, LAT]))))
            c_idx = ctrl.act(req, cand, prev_idx); place = cand[c_idx]
            _, _, _, obs = sim.realize(place, req, prev_place, rng)
            ctrl.learn(c_idx, place, obs, req)
            prev_idx, prev_place = c_idx, place
        lat_mae.append(np.mean(el)); carb_mae.append(np.mean(ec))
        cost_mae.append(np.mean(eo)); hinge_mae.append(np.mean(eh))
        lat_scale.append(np.mean(ls))
    return {"n_seeds": len(seeds),
            "latency_mae_ms": round(float(np.mean(lat_mae)), 3),
            "mean_true_latency_ms": round(float(np.mean(lat_scale)), 1),
            "latency_mape_pct": round(100 * float(np.mean(lat_mae))
                                      / float(np.mean(lat_scale)), 2),
            "slo_hinge_mae_norm": round(float(np.mean(hinge_mae)), 4),
            "slo_hinge_mae_ms": round(float(np.mean(hinge_mae)) * SLO_MARGIN, 3),
            "carbon_mae_g": round(float(np.mean(carb_mae)), 4),
            "cost_mae_usd": round(float(np.mean(cost_mae)), 4)}


# D : worked decision
def worked_example(seed=WORKED_SEED, step=WORKED_STEP):
    """Replay one decision of strict Lexi and return its full trace.

    The controller runs from step 0 up to step so that the model is in the state the paper
    describes. At step the function records the predicted and true objectives of the
    level-0 survivors and of a few reference candidates, the surviving candidate count
    after each of the three thresholded levels, and the chosen placement. The result is
    the source of Table 1."""
    sim = Continuum(seed=seed)
    cand = candidate_set(sim, cap=CAND_CAP, seed=0)
    reqs = make_workload(HORIZON, seed=seed)
    model = CausalModel(sim, seed=seed)
    ctrl = CausalContinuum(sim, model, eta=np.zeros(3))
    rng = np.random.default_rng(1000 + seed)
    prev_idx = prev_place = None
    for s in range(step):
        req = reqs[s]
        c_idx = ctrl.act(req, cand, prev_idx); place = cand[c_idx]
        _, _, _, obs = sim.realize(place, req, prev_place, rng)
        ctrl.learn(c_idx, place, obs, req)
        prev_idx, prev_place = c_idx, place
    req = reqs[step]
    est = model.predict_all(cand, req)
    raw_hat_lat = model.predict_raw_latency(cand, req)
    truth_raw = sim.expected_all(cand, req)
    # Replay the filtering level by level with slack 0.
    idx = np.arange(len(cand)); trace = []
    for k in range(3):
        best = est[idx, k].min()
        keep = idx[est[idx, k] <= best + 1e-12]
        trace.append({"level": k, "best": float(best),
                      "n_before": int(len(idx)), "n_after": int(len(keep)),
                      "survivors": [int(i) for i in keep][:12]})
        idx = keep
    chosen = int(idx[np.argmin(est[idx, COST])])
    names = [n.name for n in sim.nodes]
    stages = [s.name for s in sim.dag]

    def describe(i):
        return {"idx": int(i),
                "placement": [names[j] for j in cand[i]],
                "pred_lat_ms": round(float(raw_hat_lat[i]), 1),
                "true_lat_ms": round(float(truth_raw[i, LAT]), 1),
                "pred_norm": [round(float(x), 4) for x in est[i]],
                "true_norm": [round(float(x), 4) for x in sim.lex_normalize(truth_raw)[i]],
                "true_raw": [round(float(x), 3) for x in truth_raw[i]]}

    # The panel lists every level-0 survivor plus the cheapest, lowest-carbon and fastest
    # candidates, which a priority-first rule has to reject.
    priv_clean = np.where(truth_raw[:, PRIV] == 0)[0]
    panel = sorted(set(list(trace[0]["survivors"])
                       + [chosen,
                          int(np.argmin(truth_raw[:, COST])),
                          int(np.argmin(truth_raw[:, CARB])),
                          int(np.argmin(truth_raw[:, LAT]))]))
    return {"seed": seed, "step": step,
            "stages": stages, "nodes": names, "slo_ms": sim.slo,
            "request": {"t": int(req.t), "origin": int(req.origin),
                        "load": round(float(req.load), 3),
                        "tod_bucket": int(req.tod_bucket),
                        "time_of_day_frac": round((req.t % 240) / 240.0, 3)},
            "n_cand": int(len(cand)),
            "n_slo_feasible": int((truth_raw[:, LAT] <= sim.slo).sum()),
            "n_pii_clean": int(len(priv_clean)),
            "band_trace": trace,
            "chosen": chosen,
            "panel": [describe(i) for i in panel],
            "scales": [float(SLO_MARGIN)] + [float(x) for x in SCALES[1:]]}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    # Merge into an existing file so that runs of a single part accumulate.
    if os.path.exists(OUTPATH):
        with open(OUTPATH) as f:
            out = json.load(f)
    if which in ("all", "D"):
        out["D_worked_example"] = worked_example(); print("D done", flush=True)
    if which in ("all", "A"):
        out["A_menu"] = measure_menu(); print("A done", flush=True)
    if which in ("all", "B"):
        out["B_constrrl_scale"] = measure_constrrl_scale(); print("B done", flush=True)
    if which in ("all", "C"):
        out["C_pred_acc"] = measure_prediction_accuracy(); print("C done", flush=True)
    os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
    with open(OUTPATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.normpath(OUTPATH)}")
