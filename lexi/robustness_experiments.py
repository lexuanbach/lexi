"""
Model-stress and deployment experiments RX1 to RX7 of Lexi (extended version, RQ6 and RQ7).

The directory keeps its earlier name, causalcontinuum. The camera-ready paper omits these
measurements for space. The extended version reports them in the section "RQ6/RQ7: Model
and deployment stress" (its table of measured robustness and deployment extensions). The
first eight rows of that table correspond to RX1 to RX7 below. The Sect. 6 results of the
camera-ready do not depend on this file.

Everything is simulation only. The experiments reuse the simulator, controllers,
baselines and statistics of experiments_core.py (15 seeds, 95 percent bootstrap
intervals, paired bootstrap), and every stochastic input is seeded. The committed
../results/robustness_results.json was assembled in shards by run_incremental.py, which is the
command sequence that reproduces it (scripts/run-extended.sh). main() here runs the same
experiments in one pass and completes, but its file differs from the committed one. It runs
the RX3 sweeps on five seeds where run_incremental.py uses two (weight sweep) and three
(baseline sweeps), and it keeps the bootstrap intervals and paired tests that the shard
merge of RX1, RX2 and RX4 does not store. Its RX1, RX2 and RX4 means match the committed
ones up to the last rounded digit, and RX6 and RX7 are identical.

  RX1  Contention-aware outcome model. A learned M/G/1-like term is compared with the
       additive model as co-location contention grows, and the script reports the share of
       the contention-induced SLO degradation that it recovers. Assumption 1 fails under
       contention and this term repairs part of it.
  RX2  Doubly-robust (DR) estimator against the direct method under parametric
       misspecification. The quantity is the estimation error epsilon_t of Thm. 1.
  RX3  A finer weighted-sum sweep (step 0.05, weighting of the SLO at least 0.4, 220
       vectors) and hyperparameter sweeps of Constr-WS and Tchebycheff, which means the baselines are
       as competitive as the protocol allows.
  RX4  Hinge scale rho (a monotone reparameterisation of objective 0, which means strict Lexi
       should be invariant to it) and the SLO margin delta, with an online auto-calibrated
       delta = k * standard deviation of the latency residual.
  RX5  Decision and update time for larger DAGs (K up to 30) and candidate menus (C up to
       2000), with the resulting SLO percent and PII. The update is O(K) and independent of C.
  RX6  Alternative static priority orders and a dynamic switch from SLO-first to
       privacy-first in mid-run. The outcome model is order-agnostic, which means no retraining is
       needed.
  RX7  Hard residency pre-filter. Inadmissible placements are removed before Lexi ranks the
       rest, which means the lexicographic order over the survivors keeps its scale invariance.

Output: ../results/robustness_results.json, with keys rx1_contention, rx2_doubly_robust,
rx3_expanded_sweeps, rx4_hinge_margin, rx5_scaling, rx6_priority_orders and
rx7_compliance. The figure make_real_figures.py draws (fig_stress.pdf) reads three of
them.
"""
from __future__ import annotations
import json, os, time, itertools
import numpy as np

from sim import (Continuum, make_workload, build_dag_scaled, build_nodes_scaled,
                 LAT, PRIV, CARB, COST, SCALES)
from causal import (CausalModel, ContentionCausalModel, DoublyRobustCausalModel)
from controller import (CausalContinuum, CausalContinuumMargin, CausalContinuumOrder,
                        candidate_set, oracle_index, lex_scalar,
                        thresholded_lex_select, thresholded_lex_select_order,
                        DEFAULT_ETA)
from baselines import (WeightedSum, ConstrainedRL, Tchebycheff,
                       RankWeightedSum, NormWeightedSum, TUNED_WS_W, DEFAULT_WS_W)
from experiments_core import (boot_ci, paired_boot_p, run_method, _precompute,
                              HORIZON, SEEDS, WARMUP, CAND_CAP)

STRICT_ETA = np.zeros(3)


# RX1 contention
def rx1_contention_aware(seeds=SEEDS, conts=(0.0, 0.1, 0.2, 0.4, 0.6)):
    """RX1, additive model against the contention-aware model as contention grows.

    For each contention level the function reports the SLO percent of strict Lexi with
    the additive CausalModel and with ContentionCausalModel. The recovery is
    (additive SLO - aware SLO) / (additive SLO - nominal SLO), the share of the
    contention-induced degradation that the M/G/1 term takes back. Both arms use the
    identical strict controller, which means any difference comes from the outcome model alone."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    rows = []
    add_raw, aware_raw = {}, {}
    for cont in conts:
        add_slo, aw_slo, add_pii, aw_pii = [], [], [], []
        for seed in seeds:
            sim = Continuum(seed=seed, contention=cont)
            reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            ra = run_method(CausalContinuum(sim, CausalModel(sim, seed=seed), eta=STRICT_ETA),
                            sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            rc = run_method(CausalContinuum(sim, ContentionCausalModel(sim, seed=seed), eta=STRICT_ETA),
                            sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            add_slo.append(ra["slo"]); aw_slo.append(rc["slo"])
            add_pii.append(ra["privacy"]); aw_pii.append(rc["privacy"])
        am, al, ah = boot_ci(add_slo); wm, wl, wh = boot_ci(aw_slo)
        rows.append({"contention": cont,
                     "additive_slo": round(am, 3), "additive_slo_ci": [round(al, 3), round(ah, 3)],
                     "aware_slo": round(wm, 3), "aware_slo_ci": [round(wl, 3), round(wh, 3)],
                     "additive_pii": round(float(np.mean(add_pii)), 3),
                     "aware_pii": round(float(np.mean(aw_pii)), 3)})
        add_raw[cont] = add_slo; aware_raw[cont] = aw_slo
    nom = rows[0]["additive_slo"]
    for r in rows:
        degr = r["additive_slo"] - nom
        r["recovery_frac"] = round((r["additive_slo"] - r["aware_slo"]) / degr, 3) if degr > 1e-6 else 0.0
    # Paired test at the strongest contention level, where the recovery is largest.
    c_hi = conts[-1]
    p, md = paired_boot_p(aware_raw[c_hi], add_raw[c_hi])
    return {"n_seeds": len(seeds), "rows": rows,
            "sig_aware_vs_additive_at_max": {"contention": c_hi, "p": round(p, 4),
                                             "median_diff": round(md, 4)},
            "note": "M/G/1-flavoured online contention term; recovery_frac = share of "
                    "contention-induced SLO degradation clawed back vs the additive model"}


# RX2 doubly robust
def _run_est_error(model, sim, cand, reqs, realize_seed):
    """Run strict Lexi with the given outcome model. Returns (estimation error, SLO percent,
    PII exposure) over the post-warmup window.

    The estimation error is the mean of |estimate - truth| over all candidates and over
    the two decision-relevant normalised columns, the hinge and carbon (privacy and cost
    are exact). It plays the role of epsilon_t in Thm. 1."""
    ctrl = CausalContinuum(sim, model, eta=STRICT_ETA)
    rng = np.random.default_rng(realize_seed)
    prev_idx = prev_place = None
    est_err = []; slo_hits = pii_sum = n = 0
    for step, req in enumerate(reqs):
        est = model.predict_all(cand, req)                 # normalised estimates
        true = sim.lex_normalize(sim.expected_all(cand, req))    # normalised noise-free truth
        c_idx = thresholded_lex_select(est, STRICT_ETA, prev_idx)
        place = cand[c_idx]
        if step >= WARMUP:
            # Error on the hinge and carbon columns, the two that the misspecification eps
            # perturbs. It is averaged over the whole menu.
            err = np.abs(est[:, [LAT, CARB]] - true[:, [LAT, CARB]]).mean()
            est_err.append(float(err))
        outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
        model.update(obs, req)
        if step >= WARMUP:
            slo_hits += int(slo_v); pii_sum += outcome[PRIV]; n += 1
        prev_idx, prev_place = c_idx, place
    return float(np.mean(est_err)), 100.0 * slo_hits / n, pii_sum / n


def rx2_doubly_robust(seeds=SEEDS, epss=(0.0, 0.1, 0.2, 0.3, 0.4)):
    """RX2, the direct method (base CausalModel) against the doubly-robust estimator
    (DoublyRobustCausalModel) under increasing parametric misspecification eps.

    For each eps the function reports the estimation error of _run_est_error (lower is
    better) and the realised SLO percent and PII. A doubly-robust estimator should lower
    the error under misspecification without hurting the nominal case. The DR correction
    touches carbon only, which means the top priorities and the decisions are expected to match
    those of the direct method."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    rows = []
    dm_raw, dr_raw = {}, {}
    for eps in epss:
        dm_e, dr_e, dm_slo, dr_slo, dm_pii, dr_pii = [], [], [], [], [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            e1, s1, p1 = _run_est_error(CausalModel(sim, eps=eps, seed=seed), sim, cand, reqs, 1000 + seed)
            e2, s2, p2 = _run_est_error(DoublyRobustCausalModel(sim, eps=eps, seed=seed), sim, cand, reqs, 1000 + seed)
            dm_e.append(e1); dr_e.append(e2); dm_slo.append(s1); dr_slo.append(s2)
            dm_pii.append(p1); dr_pii.append(p2)
        dmm, dml, dmh = boot_ci(dm_e); drm, drl, drh = boot_ci(dr_e)
        rows.append({"epsilon": eps,
                     "dm_esterr": round(dmm, 4), "dm_esterr_ci": [round(dml, 4), round(dmh, 4)],
                     "dr_esterr": round(drm, 4), "dr_esterr_ci": [round(drl, 4), round(drh, 4)],
                     "esterr_reduction_pct": round(100.0 * (dmm - drm) / dmm, 2) if dmm > 1e-9 else 0.0,
                     "dm_slo": round(float(np.mean(dm_slo)), 3), "dr_slo": round(float(np.mean(dr_slo)), 3),
                     "dm_pii": round(float(np.mean(dm_pii)), 3), "dr_pii": round(float(np.mean(dr_pii)), 3)})
        dm_raw[eps], dr_raw[eps] = dm_e, dr_e
    e_hi = epss[-1]
    p, md = paired_boot_p(dr_raw[e_hi], dm_raw[e_hi])
    return {"n_seeds": len(seeds), "rows": rows,
            "sig_dr_vs_dm_esterr_at_max_eps": {"epsilon": e_hi, "p": round(p, 4), "median_diff": round(md, 6)},
            "note": "DR = direct method + evidence-gated per-node residual correction (Dudik et al.); "
                    "esterr = mean |hat g_0 - g_0| on the executed placement (Thm.1 epsilon_t)"}


# RX3 (shardable)
def _rx3_grid():
    """Weight grid of RX3. Entries are positive multiples of 0.05 that sum to 1, keeping
    those whose SLO weight is at least 0.4. Weightings with a low SLO weight cannot honour
    an SLO-first priority and would fail the target, which means the pruning keeps the sweep
    tractable and leaves 220 vectors."""
    step = 0.05
    ks = [round(x, 2) for x in np.arange(step, 1.0, step)]
    return [np.array(w) for w in itertools.product(ks, repeat=4)
            if abs(sum(w) - 1.0) < 1e-9 and w[0] >= 0.4]


def rx3_weight_shard(lo, hi, seeds=(0, 1)):
    """Evaluate the weight vectors lo to hi of the finer grid on the given seeds and return
    the per-vector (SLO, PII). run_incremental.py runs the grid in shards and merges the
    meet and dominate fractions afterwards."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    grid = _rx3_grid()
    out = []
    for W in grid[lo:hi]:
        slo, priv = [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(WeightedSum(sim, CausalModel(sim, seed=seed), W),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo.append(r["slo"]); priv.append(r["privacy"])
        out.append({"w": [round(float(x), 2) for x in W],
                    "slo": round(float(np.mean(slo)), 2), "privacy": round(float(np.mean(priv)), 3)})
    return {"lo": lo, "hi": hi, "n_total": len(grid), "seeds": list(seeds), "vectors": out}


def rx3_baseline_sweeps(seeds=(0, 1, 2, 3, 4)):
    """Lexi reference point and hyperparameter sweeps for Constr-WS (SLO tolerance and the
    split of the remaining weights) and for Tchebycheff (rho and the privacy weight)."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)

    def eval_ctrl(build):
        slo, priv = [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(build(sim, CausalModel(sim, seed=seed)),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo.append(r["slo"]); priv.append(r["privacy"])
        return float(np.mean(slo)), float(np.mean(priv))

    lexi_slo, lexi_priv = eval_ctrl(lambda s, m: CausalContinuum(s, m, eta=STRICT_ETA))
    cr_grid = []
    for tol in (0.0, 0.02, 0.05, 0.1):
        for wp in (0.34, 0.6, 0.8):
            wrest = np.array([wp, (1 - wp) / 2, (1 - wp) / 2])
            def build(s, m, tol=tol, wrest=wrest):
                c = ConstrainedRL(s, m, tol=tol); c.W_REST = wrest; return c
            s, p = eval_ctrl(build)
            cr_grid.append({"tol": tol, "w_priv": round(wp, 2), "slo": round(s, 2), "privacy": round(p, 3)})
    cr_best = min(cr_grid, key=lambda r: (r["slo"], r["privacy"]))
    tc_grid = []
    for rho in (1e-4, 1e-3, 1e-2):
        for wp in (0.10, 0.4, 0.7):
            W = np.array([0.85, wp, 0.0, 0.05]); W = W / W.sum()
            def build(s, m, rho=rho, W=W):
                t = Tchebycheff(s, m, W); t.RHO = rho; return t
            s, p = eval_ctrl(build)
            tc_grid.append({"rho": rho, "w_priv": round(wp, 2), "slo": round(s, 2), "privacy": round(p, 3)})
    tc_best = min(tc_grid, key=lambda r: (r["slo"], r["privacy"]))
    return {"n_seeds": len(seeds), "lexi": {"slo": round(lexi_slo, 2), "privacy": round(lexi_priv, 3)},
            "constrained_rl_sweep": {"grid": cr_grid, "best": cr_best},
            "tchebycheff_sweep": {"grid": tc_grid, "best": tc_best}}


def rx3_expanded_sweeps(seeds=list(range(5)), sweep_seeds=list(range(5))):
    """RX3, an expanded scalarisation sweep.

    (a) A weighted-sum sweep on a finer simplex grid than the 84 vectors of RQ1, with
    step 0.05 and an SLO weight of at least 0.4 (220 vectors). It reports the number of
    vectors, the fraction that meets the priority target (SLO at most 8 percent and PII at
    most 0.1) and the fraction that dominates Lexi.
    (b) A hyperparameter sweep for Constr-WS (SLO tolerance and the split of the
    remaining weights) and for Tchebycheff (rho and the privacy weight). Each reports its
    best SLO percent and PII, which gives the baselines their best case."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)

    def eval_ctrl(build, sds=seeds):
        slo, priv = [], []
        for seed in sds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(build(sim, CausalModel(sim, seed=seed)),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo.append(r["slo"]); priv.append(r["privacy"])
        return float(np.mean(slo)), float(np.mean(priv))

    # (a) Finer simplex grid with step 0.05, pruned to an SLO weight of at least 0.4. This is
    # the only region that can honour an SLO-first priority, and the vectors dropped by the
    # pruning fail the target trivially.
    step = 0.05
    ks = [round(x, 2) for x in np.arange(step, 1.0, step)]
    grid = [np.array(w) for w in itertools.product(ks, repeat=4)
            if abs(sum(w) - 1.0) < 1e-9 and w[0] >= 0.4]
    lexi_slo, lexi_priv = eval_ctrl(lambda s, m: CausalContinuum(s, m, eta=STRICT_ETA),
                                    sds=sweep_seeds)
    meet, dom, best = 0, 0, None
    for W in grid:
        s, p = eval_ctrl(lambda si, m, W=W: WeightedSum(si, m, W), sds=sweep_seeds)
        if s <= 8.0 and p <= 0.1:
            meet += 1
        if s <= lexi_slo + 1e-6 and p <= lexi_priv + 1e-6:
            dom += 1
        if best is None or (s, p) < (best["slo"], best["privacy"]):
            best = {"w": [round(float(x), 2) for x in W], "slo": round(s, 2), "privacy": round(p, 3)}
    sweep = {"n_weights": len(grid), "frac_meeting_target": round(meet / len(grid), 4),
             "frac_dominating_lexi": round(dom / len(grid), 4), "best_ws": best,
             "lexi": {"slo": round(lexi_slo, 2), "privacy": round(lexi_priv, 3)}}

    # (b) Constr-WS hyperparameters: the hinge tolerance and the sub-weights.
    cr_grid = []
    for tol in (0.0, 0.02, 0.05, 0.1):
        for wp in (0.34, 0.6, 0.8):                # weight of privacy among the remaining objectives
            wrest = np.array([wp, (1 - wp) / 2, (1 - wp) / 2])
            def build(s, m, tol=tol, wrest=wrest):
                c = ConstrainedRL(s, m, tol=tol); c.W_REST = wrest; return c
            s, p = eval_ctrl(build)
            cr_grid.append({"tol": tol, "w_priv": round(wp, 2), "slo": round(s, 2), "privacy": round(p, 3)})
    cr_best = min(cr_grid, key=lambda r: (r["slo"], r["privacy"]))

    # (c) Tchebycheff hyperparameters: rho and the privacy weight.
    tc_grid = []
    for rho in (1e-4, 1e-3, 1e-2):
        for wp in (0.10, 0.4, 0.7):
            W = np.array([0.85 - wp * 0.0, wp, 0.0, 0.05]); W = W / W.sum()
            def build(s, m, rho=rho, W=W):
                t = Tchebycheff(s, m, W); t.RHO = rho; return t
            s, p = eval_ctrl(build)
            tc_grid.append({"rho": rho, "w_priv": round(wp, 2), "slo": round(s, 2), "privacy": round(p, 3)})
    tc_best = min(tc_grid, key=lambda r: (r["slo"], r["privacy"]))

    return {"n_seeds": len(seeds), "weight_sweep": sweep,
            "constrained_rl_sweep": {"grid": cr_grid, "best": cr_best},
            "tchebycheff_sweep": {"grid": tc_grid, "best": tc_best},
            "note": "finer simplex + baseline hyperparameter sweeps; best-case baselines "
                    "still price the objectives (positive inversion) -- see paper"}


# RX4 p / delta sensitivity
def rx4_hinge_margin_sensitivity(seeds=SEEDS):
    """RX4, sensitivity to the hinge scale rho and to the SLO margin delta.

    (a) Hinge scale. The hinge divisor rho (SLO_MARGIN) is swept over 30 to 240 ms. The
    hinge is a strictly increasing reparameterisation of objective 0, which means the physical
    outcomes of strict Lexi should not change (Prop. 1), while the margin delta stays in
    milliseconds.
    (b) SLO margin. delta is swept over a manual grid. An online auto-calibrated delta
    equal to k times the standard deviation of the recent latency residuals is also run,
    which needs no manual tuning of units. The auto delta's SLO percent and PII can be
    compared with those of the manual values."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)

    # (a) Invariance of strict Lexi's physical outcomes to the hinge scale rho. The module
    # constant is replaced for the run and restored afterwards.
    import sim as simmod
    orig = simmod.SCALES.copy()
    hinge_rows = []
    for rho in (30.0, 60.0, 120.0, 240.0):
        simmod.SCALES = orig.copy(); simmod.SCALES[LAT] = rho
        slo, priv = [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(CausalContinuum(sim, CausalModel(sim, seed=seed), eta=STRICT_ETA),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo.append(r["slo"]); priv.append(r["privacy"])
        hinge_rows.append({"rho": rho, "slo": round(float(np.mean(slo)), 3),
                           "privacy": round(float(np.mean(priv)), 3)})
    simmod.SCALES = orig                       # restore the original scales

    # (b) Manual delta sweep.
    margin_rows = []
    for mg in (0.0, 2.0, 4.0, 8.0, 12.0):
        slo, priv = [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(CausalContinuumMargin(sim, CausalModel(sim, seed=seed), margin_ms=mg),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo.append(r["slo"]); priv.append(r["privacy"])
        sm, sl, sh = boot_ci(slo)
        margin_rows.append({"margin_ms": mg, "slo": round(sm, 3), "slo_ci": [round(sl, 3), round(sh, 3)],
                            "privacy": round(float(np.mean(priv)), 3)})

    # (b') Online auto-calibrated delta, k times the standard deviation of the last 200
    # latency residuals once at least 21 are available.
    auto_rows = []
    for kcal in (1.0, 2.0, 3.0):
        slo, priv, deltas = [], [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            model = CausalModel(sim, seed=seed)
            rng = np.random.default_rng(1000 + seed)
            prev_idx = prev_place = None
            resid = []                      # latency residuals, realised minus predicted
            slo_hits = 0; pii_sum = 0.0; n_eval = 0
            for step, req in enumerate(reqs):
                raw = model.predict_raw_latency(cand, req)
                sigma = float(np.std(resid[-200:])) if len(resid) > 20 else 0.0
                delta = kcal * sigma
                ctrl = CausalContinuumMargin(sim, model, margin_ms=delta)
                c_idx = ctrl.act(req, cand, prev_idx); place = cand[c_idx]
                outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
                # Residual of the executed placement's critical-path latency.
                resid.append(outcome[LAT] - float(raw[c_idx]))
                model.update(obs, req)
                if step >= WARMUP:
                    slo_hits += int(slo_v); pii_sum += outcome[PRIV]; n_eval += 1
                prev_idx, prev_place = c_idx, place
            slo.append(100.0 * slo_hits / n_eval); priv.append(pii_sum / n_eval)
            deltas.append(kcal * (np.std(resid[-200:]) if len(resid) > 20 else 0.0))
        sm, sl, sh = boot_ci(slo)
        auto_rows.append({"k": kcal, "slo": round(sm, 3), "slo_ci": [round(sl, 3), round(sh, 3)],
                          "privacy": round(float(np.mean(priv)), 3),
                          "mean_delta_ms": round(float(np.mean(deltas)), 2)})
    return {"n_seeds": len(seeds), "hinge_scale_invariance": hinge_rows,
            "manual_margin": margin_rows, "auto_margin": auto_rows,
            "note": "hinge rho is a monotone reparam of obj 0 => strict Lexi physically "
                    "invariant; auto delta = k*std(latency residual), self-calibrating, no units"}


# RX5 scaling K, C
def rx5_scaling(seeds=list(range(5))):
    """RX5, scaling to larger DAGs (K in 6, 12, 20, 30) and candidate menus (C in 40, 250,
    1000, 2000).

    It reports the median wall time of a decision (act) and of an update, together with
    SLO percent and PII so that a loss of quality would be visible. The question is
    whether the off-policy update path is a bottleneck. Timings are machine-specific.
    The function corresponds to Prop. "Efficient decisions" of the extended version."""
    # (a) DAG size K on the scaled synthetic continuum, with C fixed at 250 to isolate K.
    k_rows = []
    for K in (6, 12, 20, 30):
        M = max(9, K // 2)
        act_t, upd_t, slos, piis = [], [], [], []
        for seed in seeds:
            sim = Continuum(config="scaled", n_nodes=M, n_stages=K, topo_seed=seed, seed=seed)
            cand = candidate_set(sim, cap=250, seed=0)
            reqs = make_workload(400, seed=seed)
            # Deeper DAGs have longer critical paths. The SLO is set to 1.3 times the median
            # of the best candidate's latency over 20 requests so that SLO percent reflects
            # the rule and not an arbitrarily tight threshold. Timing is what this
            # experiment isolates.
            med_lat = float(np.median([sim.expected_all(cand, r)[:, LAT].min()
                                       for r in reqs[:20]]))
            sim.slo = 1.3 * med_lat
            model = CausalModel(sim, seed=seed)
            ctrl = CausalContinuum(sim, model, eta=STRICT_ETA)
            rng = np.random.default_rng(1000 + seed)
            prev_idx = prev_place = None
            slo_hits = pii_sum = n = 0
            for step, req in enumerate(reqs):
                t0 = time.perf_counter()
                c_idx = ctrl.act(req, cand, prev_idx)
                act_t.append((time.perf_counter() - t0) * 1e3)
                place = cand[c_idx]
                outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
                t1 = time.perf_counter()
                ctrl.learn(c_idx, place, obs, req)
                upd_t.append((time.perf_counter() - t1) * 1e3)
                if step >= 200:
                    slo_hits += int(slo_v); pii_sum += outcome[PRIV]; n += 1
                prev_idx, prev_place = c_idx, place
            slos.append(100.0 * slo_hits / n); piis.append(pii_sum / n)
        k_rows.append({"K": K, "M": M, "C": 250,
                       "act_ms_median": round(float(np.median(act_t)), 4),
                       "update_ms_median": round(float(np.median(upd_t)), 4),
                       "slo": round(float(np.mean(slos)), 2),
                       "pii": round(float(np.mean(piis)), 3)})
    # (b) Candidate-menu size C, with K fixed at 20.
    c_rows = []
    for C in (40, 250, 1000, 2000):
        act_t, upd_t = [], []
        for seed in seeds[:3]:
            sim = Continuum(config="scaled", n_nodes=12, n_stages=20, topo_seed=seed, seed=seed)
            cand = candidate_set(sim, cap=C, seed=0)
            reqs = make_workload(200, seed=seed)
            model = CausalModel(sim, seed=seed)
            ctrl = CausalContinuum(sim, model, eta=STRICT_ETA)
            rng = np.random.default_rng(1000 + seed)
            prev_idx = prev_place = None
            for step, req in enumerate(reqs):
                t0 = time.perf_counter()
                c_idx = ctrl.act(req, cand, prev_idx)
                act_t.append((time.perf_counter() - t0) * 1e3)
                place = cand[c_idx]
                _, _, _, obs = sim.realize(place, req, prev_place, rng)
                t1 = time.perf_counter()
                ctrl.learn(c_idx, place, obs, req)
                upd_t.append((time.perf_counter() - t1) * 1e3)
                prev_idx, prev_place = c_idx, place
        c_rows.append({"C": int(len(cand)),
                       "act_ms_median": round(float(np.median(act_t)), 4),
                       "update_ms_median": round(float(np.median(upd_t)), 4)})
    return {"n_seeds": len(seeds), "by_K": k_rows, "by_C": c_rows,
            "note": "wall-clock per decision (act) and per off-policy update; update cost is "
                    "O(K) (per-node estimators only), independent of C -- no update bottleneck"}


# RX6 priority orders
def rx6_priority_orders(seeds=SEEDS):
    """RX6, alternative and dynamic priority orders.

    Lexi is run under four strict orders and the operator-facing outcomes are reported.
    A dynamic run then switches the order at the midpoint of the horizon, from SLO-first
    to privacy-first, and compares the 400 steps before the switch with the 400 steps
    after it. The outcome model is order-agnostic, which means the switch needs no retraining and
    should take effect at once."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    names = {(0, 1, 2, 3): "SLO>-priv>-carbon>-cost (default)",
             (1, 0, 2, 3): "priv>-SLO>-carbon>-cost",
             (2, 0, 1, 3): "carbon>-SLO>-priv>-cost",
             (0, 2, 1, 3): "SLO>-carbon>-priv>-cost"}
    static = []
    for order, label in names.items():
        slo, priv, carb, cost = [], [], [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(CausalContinuumOrder(sim, CausalModel(sim, seed=seed), order=order),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo.append(r["slo"]); priv.append(r["privacy"]); carb.append(r["carbon"]); cost.append(r["cost"])
        static.append({"order": list(order), "label": label,
                       "slo": round(float(np.mean(slo)), 3), "privacy": round(float(np.mean(priv)), 3),
                       "carbon": round(float(np.mean(carb)), 3), "cost": round(float(np.mean(cost)), 3)})

    # Dynamic reordering: switch the order at the midpoint and measure the windows around it.
    dyn = {"first_half": {}, "second_half": {}}
    fh_slo = sh_slo = fh_priv = sh_priv = []
    fh_slo, sh_slo, fh_priv, sh_priv = [], [], [], []
    for seed in seeds:
        sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
        tn, orc = _precompute(sim, cand, reqs)
        model = CausalModel(sim, seed=seed)
        ctrl = CausalContinuumOrder(sim, model, order=(0, 1, 2, 3))   # starts SLO-first
        rng = np.random.default_rng(1000 + seed)
        prev_idx = prev_place = None
        half = HORIZON // 2
        f_slo = f_priv = s_slo = s_priv = 0.0; fn = sn = 0
        for step, req in enumerate(reqs):
            if step == half:
                ctrl.set_order((1, 0, 2, 3))     # privacy-first from here on, no retraining
            c_idx = ctrl.act(req, cand, prev_idx); place = cand[c_idx]
            outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
            model.update(obs, req)
            if half - 400 <= step < half:        # window just before the switch
                f_slo += int(slo_v); f_priv += outcome[PRIV]; fn += 1
            if half <= step < half + 400:        # window just after the switch
                s_slo += int(slo_v); s_priv += outcome[PRIV]; sn += 1
            prev_idx, prev_place = c_idx, place
        fh_slo.append(100.0 * f_slo / fn); fh_priv.append(f_priv / fn)
        sh_slo.append(100.0 * s_slo / sn); sh_priv.append(s_priv / sn)
    dyn = {"before_switch_SLOfirst": {"slo": round(float(np.mean(fh_slo)), 3),
                                      "privacy": round(float(np.mean(fh_priv)), 3)},
           "after_switch_privfirst": {"slo": round(float(np.mean(sh_slo)), 3),
                                      "privacy": round(float(np.mean(sh_priv)), 3)}}
    return {"n_seeds": len(seeds), "static_orders": static, "dynamic_switch": dyn,
            "note": "order is a constructor/set_order argument; the counterfactual model is "
                    "order-agnostic so reordering needs no retraining and takes effect at once"}


# RX7 compliance filter
class ComplianceContinuum(Continuum):
    """Continuum with a data-residency rule for PII stages (RX7, extended version).

    A hard admissibility filter removes every candidate that places a PII stage outside
    the private tier, before Lexi ranks the survivors. Compliance is therefore a
    pre-filter and not a priced term, which is the treatment that Sect. 2 recommends for
    hard prohibitions, and the lexicographic order over the survivors stays scale
    invariant."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # The residency rule is "PII stages run on a private (on-prem edge) node", the
        # familiar personal-data-stays-on-prem constraint. A stricter variant pinned to a
        # region would be a small change. The tier rule is used here because it leaves
        # admissible placements for every request origin.

    def admissible(self, cand: np.ndarray) -> np.ndarray:
        """Boolean mask [n_cand], True where every PII stage of the candidate is on a
        private-tier node."""
        ok = np.ones(cand.shape[0], dtype=bool)
        for s in range(self.K):
            if self.pii[s]:
                ok &= (self.priv_tier[cand[:, s]] == 1)
        return ok

    def residency_violation(self, place: np.ndarray) -> bool:
        """True if some PII stage of this executed placement is not on a private node."""
        return any(self.pii[s] and self.priv_tier[place[s]] != 1 for s in range(self.K))


def rx7_compliance(seeds=SEEDS):
    """RX7, Lexi with and without the hard residency filter.

    It reports the residency-violation rate, SLO percent and PII exposure for both arms.
    With the filter, Lexi ranks only the admissible candidates. If fewer than four are
    admissible, the full menu is kept. The filter acts before selection, which means the priority
    order and its scale invariance are unchanged for the survivors."""
    rows = []
    for use_filter in (False, True):
        slo, resid_viol, pii = [], [], []
        for seed in seeds:
            sim = ComplianceContinuum(seed=seed)
            cand_all = candidate_set(sim, cap=CAND_CAP, seed=0)
            if use_filter:
                mask = sim.admissible(cand_all)
                cand = cand_all[mask] if mask.sum() >= 4 else cand_all   # fall back if fewer than 4 remain
            else:
                cand = cand_all
            reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            model = CausalModel(sim, seed=seed)
            ctrl = CausalContinuum(sim, model, eta=STRICT_ETA)
            rng = np.random.default_rng(1000 + seed)
            prev_idx = prev_place = None
            slo_hits = viol = pii_sum = n = 0
            for step, req in enumerate(reqs):
                c_idx = ctrl.act(req, cand, prev_idx); place = cand[c_idx]
                outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
                model.update(obs, req)
                if step >= WARMUP:
                    bad = sim.residency_violation(place)
                    slo_hits += int(slo_v); viol += int(bad); pii_sum += outcome[PRIV]; n += 1
                prev_idx, prev_place = c_idx, place
            slo.append(100.0 * slo_hits / n); resid_viol.append(100.0 * viol / n); pii.append(pii_sum / n)
        rows.append({"filter": use_filter,
                     "slo": round(float(np.mean(slo)), 3),
                     "residency_violation_pct": round(float(np.mean(resid_viol)), 3),
                     "pii_exposure": round(float(np.mean(pii)), 3)})
    return {"n_seeds": len(seeds), "rows": rows,
            "note": "hard residency/jurisdiction pre-filter removes inadmissible placements "
                    "before Lexi ranks survivors; order + scale-invariance unchanged"}


def main():
    """Run RX1 to RX7, write ../results/robustness_results.json and print a summary."""
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    res = {}; t0 = time.perf_counter()
    print("RX1 contention-aware model ...");   res["rx1_contention"] = rx1_contention_aware()
    print("RX2 doubly-robust estimator ...");  res["rx2_doubly_robust"] = rx2_doubly_robust()
    print("RX3 expanded sweeps ...");          res["rx3_expanded_sweeps"] = rx3_expanded_sweeps()
    print("RX4 hinge/margin sensitivity ...."); res["rx4_hinge_margin"] = rx4_hinge_margin_sensitivity()
    print("RX5 scaling K,C ...");              res["rx5_scaling"] = rx5_scaling()
    print("RX6 priority orders ...");          res["rx6_priority_orders"] = rx6_priority_orders()
    print("RX7 compliance filter ...");        res["rx7_compliance"] = rx7_compliance()
    res["_meta"] = {"n_seeds": len(SEEDS), "runtime_s": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(here, "..", "results", "robustness_results.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[done in {res['_meta']['runtime_s']}s]\n")
    print("RX1 contention recovery (SLO% additive -> aware, recovery_frac):")
    for r in res["rx1_contention"]["rows"]:
        print(f"  c={r['contention']}: additive {r['additive_slo']:.2f}  aware {r['aware_slo']:.2f}  "
              f"recovery {r['recovery_frac']}")
    print("  sig:", res["rx1_contention"]["sig_aware_vs_additive_at_max"])
    print("RX2 DR vs DM (estimation error):")
    for r in res["rx2_doubly_robust"]["rows"]:
        print(f"  eps={r['epsilon']}: DM {r['dm_esterr']:.4f}  DR {r['dr_esterr']:.4f}  "
              f"reduction {r['esterr_reduction_pct']}%")
    print("  sig:", res["rx2_doubly_robust"]["sig_dr_vs_dm_esterr_at_max_eps"])
    sw = res["rx3_expanded_sweeps"]["weight_sweep"]
    print(f"RX3 expanded sweep: {sw['n_weights']} vectors, "
          f"{sw['frac_meeting_target']*100:.1f}% meet target, "
          f"{sw['frac_dominating_lexi']*100:.2f}% dominate Lexi; best WS "
          f"{sw['best_ws']['slo']}/{sw['best_ws']['privacy']}")
    print("  Constr-WS best:", res["rx3_expanded_sweeps"]["constrained_rl_sweep"]["best"])
    print("  Tcheby best:", res["rx3_expanded_sweeps"]["tchebycheff_sweep"]["best"])
    print("RX4 hinge-scale invariance (rho -> SLO/PII):")
    for r in res["rx4_hinge_margin"]["hinge_scale_invariance"]:
        print(f"  rho={r['rho']}: SLO {r['slo']:.2f} PII {r['privacy']:.3f}")
    print("  auto-margin:", res["rx4_hinge_margin"]["auto_margin"])
    print("RX5 scaling by K:")
    for r in res["rx5_scaling"]["by_K"]:
        print(f"  K={r['K']} M={r['M']}: act {r['act_ms_median']:.3f}ms  "
              f"update {r['update_ms_median']:.4f}ms  SLO {r['slo']:.2f} PII {r['pii']:.3f}")
    print("RX5 scaling by C:")
    for r in res["rx5_scaling"]["by_C"]:
        print(f"  C={r['C']}: act {r['act_ms_median']:.3f}ms  update {r['update_ms_median']:.4f}ms")
    print("RX6 priority orders:")
    for r in res["rx6_priority_orders"]["static_orders"]:
        print(f"  {r['label']}: SLO {r['slo']:.2f} PII {r['privacy']:.3f} "
              f"carbon {r['carbon']:.3f} cost {r['cost']:.3f}")
    print("  dynamic:", res["rx6_priority_orders"]["dynamic_switch"])
    print("RX7 compliance filter:")
    for r in res["rx7_compliance"]["rows"]:
        print(f"  filter={r['filter']}: SLO {r['slo']:.2f}  residency-viol "
              f"{r['residency_violation_pct']:.2f}%  PII {r['pii_exposure']:.3f}")


if __name__ == "__main__":
    main()
