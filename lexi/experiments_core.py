"""
Core measured experiments of Lexi (Priority-First Service Orchestration), E1 to E11.

The directory keeps its earlier name, causalcontinuum. This script drives the simulator
(sim.py), the outcome model (causal.py), the selector variants (controller.py) and the
baselines (baselines.py) over 15 seeds and writes ../results/core_results.json. Running it
takes 10 to 15 minutes on a laptop. Everything is seeded. Each seed fixes the workload
(make_workload(seed)), the model's misspecification signs and the execution noise
(realize seed 1000 + seed), which means the results are reproducible bit for bit. Only the
wall-clock scalability timings vary between runs.

Protocol shared by all experiments. A run has HORIZON = 1600 steps and every method
chooses among the same fixed 40-candidate menu (Sect. 5). Metrics are the physical
outcomes averaged over the post-warmup window, steps 800 to 1600, with 95 percent
bootstrap confidence intervals over seeds and paired-bootstrap tests behind the
headline comparisons. run_method scores one method, and _eval_methods repeats it over
seeds.

Experiments and where their results appear in the paper. The paper's RQ labels are those
of the camera-ready, and rows marked "extended" appear only in the extended version.
    E1   priority satisfaction of every method: SLO percent, PII exposure, carbon, cost,
         churn, with the paired test against WS-tuned. Table 3 (RQ2), Sect. 6.1.
    E1b  priority-inversion rate on one shared converged model, the Inv% column of
         Table 3 (Sect. 6.1 and 6.2).
    E2   sweep of the 84 weightings of the weighted sum, and the fraction that meets the
         target (SLO at most 8 percent and PII at most 0.1). Fig. 2(a) and RQ1.
    E3   rescaling of one objective's units as the optimiser sees it. Prop. 1, Fig. 2(b)
         and RQ3, including Constr-WS through extended_experiments.py.
    E4   churn against the carbon slack, and decision latency against the candidate
         count (extended version, RQ5).
    E5   curated against uniformly random candidate menus (RQ4, Sect. 6.3). The menu
         anatomy with the per-step oracle is in extended_experiments.py.
    E6   SLO-threshold sweep, the independent 10-node configuration B and a serverless
         cold-start check (RQ4, Sect. 6.3).
    E7   misspecification, parametric (bounded by construction) and structural
         (co-location contention, which breaks Assumption 1). Fig. 2(c) and RQ4.
    E8   convergence of the estimation gap over several horizons. This gap is a
         diagnostic and it is not a regret, because the aggregate used is an
         epsilon-lexicographic scalar and the signed gap can be negative.
    E9   automatic unit-free carbon slack (extended version, RQ5).
    E10  SLO safety margin delta (extended version, RQ5, "a safety margin protects the
         SLO"), nominal and under contention 0.4.
    E11  priority-conflict stress with serverless cold starts (extended version,
         Sect. "RQ2 Under Priority Conflict").
    lexlp_agreement  the exact lexicographic optimiser against strict Lexi on the same
         estimates (the equivalence between Lex-LP, listed among the baselines of Sect. 5, and strict Lexi).
check_results.py compares the numbers in core_results.json with the values quoted in the
paper.
"""
from __future__ import annotations
import json, os, time, itertools
import numpy as np

from sim import Continuum, make_workload, LAT, PRIV, CARB, COST
from causal import CausalModel
from controller import (CausalContinuum, CausalContinuumAutoSlack,
                        CausalContinuumMargin, candidate_set,
                        oracle_index, lex_scalar, thresholded_lex_select,
                        DEFAULT_ETA, LEX_W)
from baselines import (StaticGreedy, BinPack, SingleObjRL, WeightedSum,
                       ConstrainedRL, RankWeightedSum, NormWeightedSum,
                       Tchebycheff, LexicographicLP, DEFAULT_WS_W, TUNED_WS_W)

HORIZON = 1600
SEEDS = list(range(15))          # 15 seeds for the confidence intervals and paired tests
WARMUP = 800
CAND_CAP = 40
STRICT_ETA = np.zeros(3)         # slack 0 on every level: strict selection, the rule of Prop. 1
AUTO_FRAC = 0.5                  # default interquartile fraction of the auto-slack variant
MARGIN_MS = 4.0                  # default SLO safety margin delta of the margin variant, in ms


# Statistics without scipy: percentile bootstrap interval and paired bootstrap p-value.
def boot_ci(x, n_boot=5000, alpha=0.05, seed=0):
    """Mean and percentile-bootstrap (1 - alpha) interval of a sample over seeds."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_boot_p(a, b, n_boot=10000, seed=0):
    """Two-sided p-value for mean(a) != mean(b) on paired samples.

    The p-value is twice the smaller share of bootstrap-resampled mean differences on
    either side of 0, capped at 1. Pairing is by seed, since both arms of a seed see the
    same workload. Returns (p, median paired difference), the second value being an
    effect size."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    md = d[idx].mean(axis=1)
    p = 2.0 * min((md >= 0).mean(), (md <= 0).mean())
    return float(min(1.0, p)), float(np.median(d))


def _precompute(sim, cand, requests):
    """True normalised outcomes of every candidate at every step, and the per-step
    strict priority optimum (the oracle). Both are noise-free and independent of the method."""
    true_norm = [sim.lex_normalize(sim.expected_all(cand, req)) for req in requests]
    oracle = np.array([oracle_index(tn) for tn in true_norm])
    return true_norm, oracle


def run_method(method, sim, cand, requests, true_norm, oracle, realize_seed):
    """Run one controller through the request stream and return its metrics.

    Each step the method chooses a candidate, the simulator realises it with seeded noise
    and the method learns from the stage observations. step_regret is the gap in the
    geometric scalar between the chosen candidate and the oracle candidate on true
    outcomes, which is the estimation-gap diagnostic of E8. The physical metrics
    (SLO violation percent, carbon, cost, PII exposure, churn in stages per request)
    are averaged over the post-warmup window.
    """
    rng = np.random.default_rng(realize_seed)
    method.reset()
    H = len(requests)
    step_regret = np.zeros(H); obj_gap = np.zeros((H, 4))
    slo = np.zeros(H); carbon = np.zeros(H); cost = np.zeros(H)
    privacy = np.zeros(H); churn = np.zeros(H)
    prev_idx = prev_place = None
    for step, req in enumerate(requests):
        tn = true_norm[step]; o_idx = int(oracle[step])
        c_idx = method.act(req, cand, prev_idx); place = cand[c_idx]
        step_regret[step] = float(lex_scalar(tn[c_idx]) - lex_scalar(tn[o_idx]))
        obj_gap[step] = tn[c_idx] - tn[o_idx]
        outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
        method.learn(c_idx, place, obs, req)
        slo[step] = 1.0 if slo_v else 0.0
        carbon[step] = outcome[CARB]; cost[step] = outcome[COST]
        privacy[step] = outcome[PRIV]; churn[step] = ch
        prev_idx, prev_place = c_idx, place
    # Post-warmup evaluation window. For horizons of at most 2 * WARMUP steps, such as the
    # smoke test, it falls back to the second half so that the window is never empty. For
    # the full runs (H = 1600) it is exactly steps WARMUP to H.
    w = min(WARMUP, H // 2)
    ev = slice(w, H)
    return {"regret_cum": np.cumsum(step_regret), "obj_gap": obj_gap,
            "slo": 100.0 * slo[ev].mean(), "carbon": carbon[ev].mean(),
            "cost": cost[ev].mean(), "privacy": privacy[ev].mean(),
            "churn": churn[ev].mean(),
            "slo_full": 100.0 * slo.mean(), "privacy_full": privacy.mean(),
            "regret_curve": np.cumsum(step_regret)}


def _factory(name):
    """Return a constructor make(sim, cand, seed) for a method name. Every model-based
    method gets its own fresh CausalModel with the same seed, which means all of them start from
    identical priors."""
    def make(sim, cand, seed):
        if name == "Lexi-strict":
            return CausalContinuum(sim, CausalModel(sim, seed=seed), eta=STRICT_ETA)
        if name == "Lexi":
            return CausalContinuum(sim, CausalModel(sim, seed=seed), eta=DEFAULT_ETA)
        if name == "ws_default":
            return WeightedSum(sim, CausalModel(sim, seed=seed), DEFAULT_WS_W)
        if name == "ws_tuned":
            return WeightedSum(sim, CausalModel(sim, seed=seed), TUNED_WS_W)
        if name == "constrained_rl":
            return ConstrainedRL(sim, CausalModel(sim, seed=seed))
        if name == "rank_ws":
            return RankWeightedSum(sim, CausalModel(sim, seed=seed))
        if name == "norm_ws":
            return NormWeightedSum(sim, CausalModel(sim, seed=seed))
        if name == "tcheby":
            return Tchebycheff(sim, CausalModel(sim, seed=seed))
        if name == "lex_lp":
            return LexicographicLP(sim, CausalModel(sim, seed=seed))
        if name == "lexi_auto":
            return CausalContinuumAutoSlack(sim, CausalModel(sim, seed=seed), frac=AUTO_FRAC)
        if name == "Lexi-margin":
            return CausalContinuumMargin(sim, CausalModel(sim, seed=seed), margin_ms=MARGIN_MS)
        if name == "single_obj_rl":
            return SingleObjRL(sim, len(cand), seed=seed)
        if name == "static_greedy":
            return StaticGreedy(sim)
        if name == "binpack":
            return BinPack(sim)
        raise ValueError(name)
    return make


MAIN_METHODS = ["Lexi-strict", "Lexi", "Lexi-margin", "lex_lp", "ws_default",
                "ws_tuned", "rank_ws", "norm_ws", "tcheby", "constrained_rl",
                "single_obj_rl", "static_greedy", "binpack"]


def _eval_methods(methods, seeds, cand_fn, contention=0.0, slo=260.0,
                  horizon=HORIZON, config="A", cold_penalty=0.0, real_traces=False):
    """Return {method: {metric: [per-seed values]}} of realised physical metrics.

    contention, slo, config, cold_penalty and real_traces are passed to the Continuum.
    cand_fn(sim, seed) builds the candidate menu."""
    out = {m: {k: [] for k in ("slo", "privacy", "carbon", "cost", "churn",
                               "regret")} for m in methods}
    for seed in seeds:
        sim = Continuum(seed=seed, contention=contention, slo=slo, config=config,
                        cold_penalty=cold_penalty, real_traces=real_traces)
        cand = cand_fn(sim, seed)
        requests = make_workload(horizon, seed=seed, real_traces=real_traces)
        tn, orc = _precompute(sim, cand, requests)
        for m in methods:
            r = run_method(_factory(m)(sim, cand, seed), sim, cand, requests,
                           tn, orc, realize_seed=1000 + seed)
            for k in ("slo", "privacy", "carbon", "cost", "churn"):
                out[m][k].append(r[k])
            out[m]["regret"].append(r["regret_cum"][-1])
    return out


def _curated(sim, seed):
    """The curated 40-candidate menu of Sect. 5, identical for every seed."""
    return candidate_set(sim, cap=CAND_CAP, seed=0)


# E1
def e1_priority_satisfaction(seeds=SEEDS):
    """E1, the operator-facing outcomes of Table 3 for every method in MAIN_METHODS, with
    bootstrap intervals, paired tests of Lexi-strict against WS-tuned (SLO and privacy)
    and against the three scale-robust scalarisers (SLO), and the largest per-metric gap
    between Lex-LP and Lexi-strict."""
    res = _eval_methods(MAIN_METHODS, seeds, _curated)
    table = {}
    for m in MAIN_METHODS:
        table[m] = {}
        for k in ("slo", "privacy", "carbon", "cost", "churn"):
            mean, lo, hi = boot_ci(res[m][k])
            table[m][k] = {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
    # Paired tests of Lexi-strict against the tuned weighted sum, and against the
    # scale-robust scalarisers, to see whether they trade away the top-priority SLO.
    tests = {}
    for k in ("slo", "privacy"):
        p, eff = paired_boot_p(res["Lexi-strict"][k], res["ws_tuned"][k])
        tests[f"lexi_vs_wstuned_{k}"] = {"p": round(p, 4), "median_diff": round(eff, 4)}
    for opp in ("rank_ws", "norm_ws", "tcheby"):
        p, eff = paired_boot_p(res["Lexi-strict"]["slo"], res[opp]["slo"])
        tests[f"lexi_vs_{opp}_slo"] = {"p": round(p, 4), "median_diff": round(eff, 4)}
    # Lex-LP should coincide with Lexi-strict. Report the largest per-metric gap.
    lp_gap = {k: round(float(np.max(np.abs(np.array(res["Lexi-strict"][k])
                                           - np.array(res["lex_lp"][k])))), 6)
              for k in ("slo", "privacy", "carbon", "cost", "churn")}
    return {"n_seeds": len(seeds), "table": table, "paired_tests": tests,
            "lexlp_vs_strict_max_gap": lp_gap}


# E1b priority inversion
def _lex_worse(a, b, tol=1e-9):
    """True if objective vector a is lexicographically worse than b, meaning strictly
    greater at the first coordinate (hinge, privacy, carbon, cost) where they differ
    by more than tol."""
    for k in range(4):
        if a[k] > b[k] + tol:
            return True
        if a[k] < b[k] - tol:
            return False
    return False


def e1_inversion(seeds=SEEDS):
    """E1b, the priority-inversion rate (the Inv% column of Table 3).

    It is the share of decisions whose chosen placement is lexicographically worse than
    the strict priority optimum on the same estimates. The rate is zero for a rule that
    honours the strict order and positive for a scalariser that prices the objectives and
    can outvote a higher priority with a lower one. To separate the decision rule from
    estimation noise, every rule is scored on one shared model per seed. The model is
    warmed up by a strict Lexi run for WARMUP steps, and the rules are then compared on
    the post-warmup requests with hysteresis off, which means each is the pure rule."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    builders = {
        "Lexi-strict":    lambda s, m: CausalContinuum(s, m, eta=STRICT_ETA),
        "constrained_rl": lambda s, m: ConstrainedRL(s, m),
        "ws_tuned":       lambda s, m: WeightedSum(s, m, TUNED_WS_W),
        "rank_ws":        lambda s, m: RankWeightedSum(s, m),
        "norm_ws":        lambda s, m: NormWeightedSum(s, m),
        "tcheby":         lambda s, m: Tchebycheff(s, m),
        "ws_default":     lambda s, m: WeightedSum(s, m, DEFAULT_WS_W),
    }
    per = {n: [] for n in builders}
    for seed in seeds:
        sim = Continuum(seed=seed)
        reqs = make_workload(HORIZON, seed=seed)
        diag = CausalModel(sim, seed=seed)                 # the shared, warmed-up model
        warm = CausalContinuum(sim, diag, eta=STRICT_ETA)
        rng = np.random.default_rng(2000 + seed); prev = None
        for req in reqs[:WARMUP]:
            c = warm.act(req, cand, None)
            _, _, _, obs = sim.realize(cand[c], req, prev, rng)
            diag.update(obs, req); prev = cand[c]
        flags = {n: [] for n in builders}
        for req in reqs[WARMUP:]:
            est = diag.predict_all(cand, req)
            o = thresholded_lex_select(est, STRICT_ETA, prev_idx=None)
            for n, build in builders.items():
                c = build(sim, diag).act(req, cand, None)  # pure rule, no hysteresis
                flags[n].append(1.0 if _lex_worse(est[c], est[o]) else 0.0)
        for n in builders:
            per[n].append(100.0 * float(np.mean(flags[n])))
    out = {}
    for n in builders:
        mean, lo, hi = boot_ci(per[n])
        out[n] = {"inversion_pct": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
    return {"n_seeds": len(seeds), "inversion": out,
            "note": "fraction of decisions lexicographically worse than the strict "
                    "priority optimum on the same shared estimates; hysteresis off"}


# E11 priority-conflict stress
def e11_conflict_stress(cold_penalties=(0.0, 15.0, 30.0, 45.0, 60.0),
                        cold_prob=0.7, seeds=SEEDS):
    """E11, how the priority advantage changes with operational conflict (extended version).

    Serverless nodes are the cheap and green option that a pricing scalariser is drawn
    to, but a cold start makes a newly placed serverless stage risky for latency. As the
    cold-start penalty grows, the conflict between the SLO and cost or carbon sharpens,
    and a rule that prices the objectives is increasingly tempted to trade the SLO away,
    which is a priority inversion. Strict Lexi has zero inversions by construction. For
    each penalty level the function reports (a) the inversion rate, scored on one shared
    warmed-up model per seed so the estimator is held fixed across rules, and (b) the
    realised SLO-violation rate. The rules are Lexi and the priced scalarisers, including
    the two scale-invariant ones (rank and min-max) and augmented Tchebycheff. The
    result shows that the effect comes from pricing itself and not from a miscalibrated
    weight. The extended-version conflict table is built from these levels."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    inv_builders = {
        "Lexi-strict": lambda s, m: CausalContinuum(s, m, eta=STRICT_ETA),
        "ws_tuned":    lambda s, m: WeightedSum(s, m, TUNED_WS_W),
        "rank_ws":     lambda s, m: RankWeightedSum(s, m),
        "norm_ws":     lambda s, m: NormWeightedSum(s, m),
        "tcheby":      lambda s, m: Tchebycheff(s, m),
    }
    slo_methods = ["Lexi-strict", "Lexi-margin", "ws_tuned", "rank_ws", "norm_ws", "tcheby"]
    levels = []
    for cp in cold_penalties:
        # (a) Priority inversion on one shared, warmed-up model per seed.
        per_inv = {n: [] for n in inv_builders}
        for seed in seeds:
            sim = Continuum(seed=seed, cold_penalty=cp, cold_prob=cold_prob)
            reqs = make_workload(HORIZON, seed=seed)
            diag = CausalModel(sim, seed=seed)
            warm = CausalContinuum(sim, diag, eta=STRICT_ETA)
            rng = np.random.default_rng(2000 + seed); prev = None
            for req in reqs[:WARMUP]:
                c = warm.act(req, cand, None)
                _, _, _, obs = sim.realize(cand[c], req, prev, rng)
                diag.update(obs, req); prev = cand[c]
            flags = {n: [] for n in inv_builders}
            for req in reqs[WARMUP:]:
                est = diag.predict_all(cand, req)
                o = thresholded_lex_select(est, STRICT_ETA, prev_idx=None)
                for n, build in inv_builders.items():
                    c = build(sim, diag).act(req, cand, None)  # pure rule, no hysteresis
                    flags[n].append(1.0 if _lex_worse(est[c], est[o]) else 0.0)
            for n in inv_builders:
                per_inv[n].append(100.0 * float(np.mean(flags[n])))
        # (b) Realised SLO-violation percent at this cold-start level.
        ev = _eval_methods(slo_methods, seeds, _curated, cold_penalty=cp)
        lvl = {"cold_penalty": cp, "inversion": {}, "slo_viol": {}}
        for n in inv_builders:
            mean, lo, hi = boot_ci(per_inv[n])
            lvl["inversion"][n] = {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
        for n in slo_methods:
            mean, lo, hi = boot_ci(ev[n]["slo"])
            lvl["slo_viol"][n] = {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
        levels.append(lvl)
    return {"n_seeds": len(seeds), "cold_prob": cold_prob,
            "cold_penalties": list(cold_penalties), "levels": levels,
            "note": "serverless cold-start priority-conflict stress: priority-inversion% "
                    "and realized SLO-violation% vs cold-start penalty; Lexi inversion=0 "
                    "by construction, priced scalarisers (incl. scale-invariant) grow"}


# E10 SLO safety margin
def e10_margin_sweep(margins=(0.0, 2.0, 4.0, 6.0, 8.0, 12.0), seeds=SEEDS):
    """E10, sweep of the SLO safety margin delta of CausalContinuumMargin (extended
    version, RQ5).

    The margin reserves delta ms of latency headroom against the queueing noise. For each
    delta the function reports realised SLO percent, PII exposure and churn with 95
    percent bootstrap intervals, in the nominal regime and under co-location contention
    0.4, where the model underestimates latency and the margin matters most. Delta = 0 is
    strict feasibility. It also reports the paired-bootstrap significance of the default
    margin against strict Lexi and against Constr-WS on SLO percent."""
    def _sweep(contention):
        cand = candidate_set(Continuum(contention=contention), cap=CAND_CAP, seed=0)
        pts = []
        raw = {}
        for mg in margins:
            slo_r, priv_r, churn_r = [], [], []
            for seed in seeds:
                sim = Continuum(seed=seed, contention=contention)
                reqs = make_workload(HORIZON, seed=seed)
                tn, orc = _precompute(sim, cand, reqs)
                r = run_method(CausalContinuumMargin(sim, CausalModel(sim, seed=seed),
                                                     margin_ms=mg),
                               sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
                slo_r.append(r["slo"]); priv_r.append(r["privacy"]); churn_r.append(r["churn"])
            sm, sl, sh = boot_ci(slo_r); pm, pl, ph = boot_ci(priv_r)
            cm, cl, chh = boot_ci(churn_r)
            pts.append({"margin_ms": mg,
                        "slo": round(sm, 3), "slo_ci": [round(sl, 3), round(sh, 3)],
                        "privacy": round(pm, 3), "privacy_ci": [round(pl, 3), round(ph, 3)],
                        "churn": round(cm, 4), "churn_ci": [round(cl, 4), round(chh, 4)]})
            raw[mg] = slo_r
        return pts, raw

    nom, raw_nom = _sweep(0.0)
    cont, _ = _sweep(0.4)
    # Significance of the default margin against strict (margin 0) and against constrained_rl.
    strict_slo = raw_nom[0.0]
    default_slo = raw_nom[MARGIN_MS] if MARGIN_MS in raw_nom else raw_nom[margins[0]]
    cr_slo = []
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    for seed in seeds:
        sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
        tn, orc = _precompute(sim, cand, reqs)
        r = run_method(ConstrainedRL(sim, CausalModel(sim, seed=seed)),
                       sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
        cr_slo.append(r["slo"])
    p_strict, md_strict = paired_boot_p(default_slo, strict_slo)
    p_cr, md_cr = paired_boot_p(default_slo, cr_slo)
    return {"n_seeds": len(seeds), "default_margin_ms": MARGIN_MS,
            "nominal": nom, "contention": cont, "contention_level": 0.4,
            "sig_default_vs_strict_slo": {"p": round(p_strict, 4), "median_diff": round(md_strict, 4)},
            "sig_default_vs_constrained_slo": {"p": round(p_cr, 4), "median_diff": round(md_cr, 4)},
            "note": "delta ms of reserved SLO headroom; delta=0 == strict feasibility"}


# E2 weight sweep
def e2_weight_sweep(seeds=list(range(6)), step=0.1):
    """E2, the weight sweep behind Fig. 2(a) and RQ1.

    The grid holds every weighting of the four objectives with strictly positive entries
    in steps of 0.1 that sums to 1, which gives 84 vectors. Each is run as a fixed-weight
    sum with a fresh outcome model per seed and the mean SLO percent and PII exposure are
    recorded. The function then reports the best weighting, the fraction of the grid that
    meets the target (SLO at most 8 percent and PII at most 0.1) and the fraction that
    dominates Lexi-strict on both coordinates. Lexi reaches its point without any
    weights. The sweep uses 6 seeds."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    grid = []
    ks = [round(x, 2) for x in np.arange(step, 1.0, step)]
    for w in itertools.product(ks, repeat=4):
        if abs(sum(w) - 1.0) < 1e-9:
            grid.append(np.array(w))
    pts = []
    for W in grid:
        slo, priv = [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(WeightedSum(sim, CausalModel(sim, seed=seed), W),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo.append(r["slo"]); priv.append(r["privacy"])
        pts.append({"w": [round(float(x), 2) for x in W],
                    "slo": round(float(np.mean(slo)), 2),
                    "privacy": round(float(np.mean(priv)), 3)})
    # Reference point of Lexi-strict on the same seeds.
    lx = _eval_methods(["Lexi-strict"], seeds, _curated)["Lexi-strict"]
    lexi_slo, lexi_priv = float(np.mean(lx["slo"])), float(np.mean(lx["privacy"]))
    target = [(p["slo"] <= 8.0 and p["privacy"] <= 0.1) for p in pts]
    dominates_lexi = [(p["slo"] <= lexi_slo + 1e-6 and p["privacy"] <= lexi_priv + 1e-6)
                      for p in pts]
    best = min(pts, key=lambda p: (p["slo"], p["privacy"]))
    return {"n_weights": len(grid), "n_seeds": len(seeds),
            "lexi_strict": {"slo": round(lexi_slo, 2), "privacy": round(lexi_priv, 3)},
            "frac_meeting_target": round(float(np.mean(target)), 3),
            "frac_dominating_lexi": round(float(np.mean(dominates_lexi)), 3),
            "best_ws": best,
            "pareto": [p for p in pts if p["slo"] <= 25 and p["privacy"] <= 2.0][:200]}


# E3 scale-invariance
class _Rescaled:
    """Wrap an outcome model so that one normalised objective column is multiplied by c
    on the way to the optimiser. True outcomes are untouched. This is the unit change
    of the scale study (a strictly increasing reparameterisation in Prop. 1). The
    column keeps its order, which means a rule that reads only the order is unaffected."""
    def __init__(self, model, coord, c): self.m = model; self.coord = coord; self.c = c
    def predict_all(self, cand, req):
        e = self.m.predict_all(cand, req).copy(); e[:, self.coord] *= self.c; return e
    def update(self, *a): self.m.update(*a)


def e3_scale_invariance(coords=(2, 1), factors=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
                        seeds=list(range(8))):
    """E3, the scale-invariance study behind Prop. 1, Fig. 2(b) and RQ3.

    The units of one objective (carbon by default, and privacy) are multiplied by each
    factor as seen by the optimiser, with true outcomes unchanged. SLO percent and PII
    exposure are recorded for a panel of rules. Strict Lexi and its auto-slack variant
    depend only on the order of each column and are invariant. The rank and min-max
    weighted sums are also invariant but still price the objectives. The tuned linear sum
    and augmented Tchebycheff depend on the scale. Result keys prefixed lexi_ and ws_ are
    read by the figure script. The prefixes rank_ws_, norm_ws_, tcheby_ and lexi_auto_
    belong to the scale-robust scalarisers. Uses 8 seeds."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    coord_names = {1: "privacy", 2: "carbon", 3: "cost"}
    builders = {
        "lexi":      lambda s, m: CausalContinuum(s, m, eta=STRICT_ETA),
        "lexi_auto": lambda s, m: CausalContinuumAutoSlack(s, m, frac=AUTO_FRAC),
        "ws":        lambda s, m: WeightedSum(s, m, TUNED_WS_W),
        "rank_ws":   lambda s, m: RankWeightedSum(s, m),
        "norm_ws":   lambda s, m: NormWeightedSum(s, m),
        "tcheby":    lambda s, m: Tchebycheff(s, m),
    }
    out = {}
    for coord in coords:
        rows = []
        for c in factors:
            def run(build):
                slo, priv = [], []
                for seed in seeds:
                    sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
                    tn, orc = _precompute(sim, cand, reqs)
                    base = CausalModel(sim, seed=seed)
                    model = _Rescaled(base, coord, c) if c != 1.0 else base
                    r = run_method(build(sim, model), sim, cand, reqs, tn, orc,
                                   realize_seed=1000 + seed)
                    slo.append(r["slo"]); priv.append(r["privacy"])
                return round(float(np.mean(slo)), 2), round(float(np.mean(priv)), 3)
            row = {"factor": c}
            for key, build in builders.items():
                s, p = run(build)
                row[f"{key}_slo"] = s
                row[f"{key}_priv"] = p
            rows.append(row)
        out[coord_names[coord]] = rows
    return {"n_seeds": len(seeds), "objectives": out,
            "methods": list(builders.keys())}


def e9_autoslack(fracs=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0), seeds=SEEDS):
    """E9, the automatic carbon slack of CausalContinuumAutoSlack (extended version, RQ5).

    Sweeps the unit-free knob frac and reports the trade between stability and the SLO
    (SLO percent, PII exposure, churn) with 95 percent bootstrap intervals. Frac = 0
    is strict Lexi."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    points = []
    for frac in fracs:
        slo_r, priv_r, churn_r = [], [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(CausalContinuumAutoSlack(sim, CausalModel(sim, seed=seed),
                                                    frac=frac),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo_r.append(r["slo"]); priv_r.append(r["privacy"]); churn_r.append(r["churn"])
        sm, sl, sh = boot_ci(slo_r); pm, pl, ph = boot_ci(priv_r)
        cm, cl, chh = boot_ci(churn_r)
        points.append({"frac": frac,
                       "slo": round(sm, 3), "slo_ci": [round(sl, 3), round(sh, 3)],
                       "privacy": round(pm, 3), "privacy_ci": [round(pl, 3), round(ph, 3)],
                       "churn": round(cm, 4), "churn_ci": [round(cl, 4), round(chh, 4)]})
    return {"points": points, "n_seeds": len(seeds), "default_frac": AUTO_FRAC,
            "note": "IQR-fraction slack; unit-free knob; frac=0 == strict Lexi"}


def lexlp_agreement(seeds=SEEDS):
    """Per-step decision agreement between the exact lexicographic optimiser
    (LexicographicLP) and strict Lexi on identical estimates.

    The trajectory is driven by the Lexi-strict controller with hysteresis on one shared
    model. At each step the LP choice is compared with (a) the pure strict rule without
    hysteresis and (b) the deployed rule with hysteresis. Agreement (a) should be 1, and
    the gap in (b) is due to the hysteresis alone."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    agree_pure = agree_hyst = total = 0
    for seed in seeds:
        sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
        model = CausalModel(sim, seed=seed)
        lp = LexicographicLP(sim, model)
        rng = np.random.default_rng(1000 + seed)
        prev_idx = prev_place = None
        for req in reqs:
            est = model.predict_all(cand, req)
            b = lp.act(req, cand, prev_idx)                                # exact lexicographic optimiser
            pure = thresholded_lex_select(est, STRICT_ETA, prev_idx=None)  # strict rule, no hysteresis
            hyst = thresholded_lex_select(est, STRICT_ETA, prev_idx)       # strict rule with hysteresis
            agree_pure += int(pure == b); agree_hyst += int(hyst == b)
            total += 1
            place = cand[hyst]
            _, _, _, obs = sim.realize(place, req, prev_place, rng)
            model.update(obs, req)
            prev_idx, prev_place = hyst, place
    return {"n_seeds": len(seeds), "n_steps": total,
            "agreement_pure": round(agree_pure / total, 4),
            "agreement_hysteresis": round(agree_hyst / total, 4),
            "note": "LexicographicLP == strict Lexi on identical estimates; the only "
                    "gap is Lexi's stability hysteresis among candidates tied on "
                    "latency, privacy and carbon"}


# E4 stability + scalability
def e4_stability(mult_grid=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0), seeds=SEEDS):
    """E4a, churn and SLO against the carbon slack (extended version, RQ5). The multiplier
    scales the carbon slack of DEFAULT_ETA. The hinge and privacy entries of DEFAULT_ETA
    are 0, which means both stay strict."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    points = []
    for mult in mult_grid:
        eta = DEFAULT_ETA * np.array([mult, 1.0, mult])
        slo_r, churn_r = [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(CausalContinuum(sim, CausalModel(sim, seed=seed), eta=eta),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo_r.append(r["slo"]); churn_r.append(r["churn"])
        sm, sl, sh = boot_ci(slo_r); cm, cl, ch = boot_ci(churn_r)
        points.append({"eta_mult": mult, "slo": round(sm, 3), "slo_ci": [round(sl, 3), round(sh, 3)],
                       "churn": round(cm, 4), "churn_ci": [round(cl, 4), round(ch, 4)]})
    return {"points": points, "n_seeds": len(seeds), "base_eta": DEFAULT_ETA.tolist()}


def scalability(caps=(8, 16, 24, 32, 48, 64, 96, 128), reps=500, repeats=15, seed=0):
    """E4b, in-process decision latency against the candidate count (extended version, RQ5,
    Prop. "Efficient decisions").

    For each menu size the controller is warmed on 20 requests. The mean time of act over
    500 requests is then measured 15 times. The result reports the median and
    interquartile range per size and a bootstrap confidence interval for the slope in
    ms per candidate. Only the timing of the decision rule is measured, and not
    reconciliation, admission or migration. Wall-clock values vary between runs."""
    sim = Continuum(seed=seed); requests = make_workload(reps, seed=seed)
    out = []
    for cap in caps:
        cand = candidate_set(sim, cap=cap, seed=0)
        ctrl = CausalContinuum(sim, CausalModel(sim, seed=seed))
        for req in requests[:20]:
            c = ctrl.act(req, cand, None)
            _, _, _, obs = sim.realize(cand[c], req, None, np.random.default_rng(1))
            ctrl.learn(c, cand[c], obs, req)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for req in requests:
                ctrl.act(req, cand, 0)
            times.append((time.perf_counter() - t0) / len(requests) * 1e3)
        times = np.array(times)
        out.append({"n_cand": int(len(cand)), "ms_median": round(float(np.median(times)), 5),
                    "ms_iqr": [round(float(np.quantile(times, .25)), 5),
                               round(float(np.quantile(times, .75)), 5)]})
    xs = np.array([o["n_cand"] for o in out]); ys = np.array([o["ms_median"] for o in out])
    # Confidence interval of the slope by bootstrap over the (size, time) points.
    rng = np.random.default_rng(0); slopes = []
    for _ in range(2000):
        idx = rng.integers(0, len(xs), len(xs))
        if len(np.unique(xs[idx])) < 2:
            continue
        slopes.append(np.polyfit(xs[idx], ys[idx], 1)[0])
    slo, shi = np.quantile(slopes, [0.025, 0.975])
    return {"points": out, "dag_stages": sim.K,
            "slope_ms_per_cand": round(float(np.polyfit(xs, ys, 1)[0]), 6),
            "slope_ci": [round(float(slo), 6), round(float(shi), 6)]}


# E5 candidate robustness
def _random_cand(sim, seed):
    """Uniformly random menu of CAND_CAP distinct placements, the uncurated generator of RQ4."""
    rng = np.random.default_rng(100 + seed)
    cands = set(); out = []
    while len(out) < CAND_CAP:
        p = tuple(int(rng.integers(0, sim.M)) for _ in range(sim.K))
        if p not in cands:
            cands.add(p); out.append(list(p))
    return np.array(out)


def e5_candidate_robustness(seeds=list(range(8))):
    """E5, curated against random menus for Lexi-strict and the two weighted sums (RQ4,
    Sect. 6.3). Absolute SLO and PII depend on the menu, and the method ordering is the
    quantity to compare. Uses 8 seeds."""
    methods = ["Lexi-strict", "ws_tuned", "ws_default"]
    res = {}
    for label, fn in (("curated", _curated), ("random", _random_cand)):
        r = _eval_methods(methods, seeds, fn)
        res[label] = {m: {"slo": round(float(np.mean(r[m]["slo"])), 2),
                          "privacy": round(float(np.mean(r[m]["privacy"])), 3),
                          "regret": round(float(np.mean(r[m]["regret"])), 2)}
                      for m in methods}
    return {"n_seeds": len(seeds), "sets": res}


# E6 generalization
def e6_generalization(seeds=list(range(8))):
    """E6, generalisation for RQ4 (Sect. 6.3).

    (a) An SLO-threshold sweep from 220 to 340 ms on config A. (b) The independent config
    B of 10 nodes, an 8-stage DAG with 3 PII stages and different economics. (c) A
    serverless cold-start setting that the additive model can only average. The
    question is whether the ordering of the methods is stable. Uses 8 seeds."""
    methods = ["Lexi-strict", "ws_tuned", "ws_default", "constrained_rl"]
    slo_sweep = []
    for slo in (220.0, 260.0, 300.0, 340.0):
        r = _eval_methods(methods, seeds, _curated, slo=slo)
        slo_sweep.append({"slo_ms": slo,
                          **{m: {"slo": round(float(np.mean(r[m]["slo"])), 2),
                                 "privacy": round(float(np.mean(r[m]["privacy"])), 3)}
                             for m in r}})
    # (b) Config B, with its own candidate menu built by the same generator.
    def cand_b(sim, seed):
        return candidate_set(sim, cap=CAND_CAP, seed=0)
    rb = _eval_methods(methods, seeds, cand_b, config="B")
    config_b = {m: {"slo": round(float(np.mean(rb[m]["slo"])), 2),
                    "privacy": round(float(np.mean(rb[m]["privacy"])), 3),
                    "carbon": round(float(np.mean(rb[m]["carbon"])), 3),
                    "cost": round(float(np.mean(rb[m]["cost"])), 3)} for m in methods}
    # (c) Serverless cold start, a deployment effect that the additive model only averages.
    rc = _eval_methods(methods, seeds, _curated, cold_penalty=90.0)
    cold = {m: {"slo": round(float(np.mean(rc[m]["slo"])), 2),
                "privacy": round(float(np.mean(rc[m]["privacy"])), 3)} for m in methods}
    return {"n_seeds": len(seeds), "slo_sweep": slo_sweep, "config_b": config_b,
            "config_b_desc": "10 nodes (4 edge/3 cloud/3 serverless), 8-stage DAG, 3 PII stages",
            "cold_start": cold, "cold_penalty_ms": 90.0}


# E7 misspecification
def e7_misspecification(seeds=SEEDS):
    """E7, model misspecification (RQ4, Fig. 2(c)).

    (a) Parametric. The model's bounded perturbation eps (causal.py) is set to 0 to 0.4
    and the final estimation gap is fitted with a line. This is a consistency check of
    the base + C * eps behaviour and not independent validation, because the perturbation
    is bounded by construction. (b) Structural. Co-location contention from 0 to 0.6
    breaks Assumption 1 and every additive-model method degrades, which is the stated
    limit of the identification assumption. The SLO percent of Lexi-strict, WS-tuned and
    Constr-WS is recorded."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    # (a) Parametric.
    param = []
    for eps in (0.0, 0.10, 0.20, 0.30, 0.40):
        finals = []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(CausalContinuum(sim, CausalModel(sim, eps=eps, seed=seed)),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            finals.append(r["regret_cum"][-1])
        m, lo, hi = boot_ci(finals)
        param.append({"epsilon": eps, "regret": round(m, 3), "ci": [round(lo, 3), round(hi, 3)]})
    xs = np.array([p["epsilon"] for p in param]); ys = np.array([p["regret"] for p in param])
    slope, intercept = np.polyfit(xs, ys, 1); r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)
    # (b) Structural: non-additive co-location contention that the additive model cannot
    # represent. Record SLO percent as contention grows.
    struct = []
    for cont in (0.0, 0.1, 0.2, 0.4, 0.6):
        row = {"contention": cont}
        r = _eval_methods(["Lexi-strict", "ws_tuned", "constrained_rl"], seeds,
                          _curated, contention=cont)
        for m in r:
            row[m] = round(float(np.mean(r[m]["slo"])), 2)
        struct.append(row)
    return {"parametric": {"curve": param, "slope": round(float(slope), 2),
                           "intercept": round(float(intercept), 3), "r2": round(r2, 4),
                           "note": "bounded-by-design injection; a consistency check, not independent validation"},
            "structural_contention": {"slo_pct": struct,
                                      "note": "non-additive; ALL additive-model methods degrade -- a stated identification-assumption limit"}}


# E8 regret + horizons + on/off-policy
def e8_convergence(seeds=list(range(8))):
    """E8, convergence diagnostics, secondary to the physical metrics.

    (a) The cumulative estimation gap of Lexi at horizons 400 to 3200. The aggregate is
    an epsilon-lexicographic scalar, which means the signed gap can be negative and is not a
    regret. (b) Mean cumulative-gap curves for Lexi-strict and the two weighted sums over
    the 1600-step horizon, sampled at 80 points. Uses 8 seeds."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    # (a) Estimation gap at several horizons.
    horizons = [400, 800, 1600, 3200]
    hz = []
    for H in horizons:
        finals = []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(H, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(CausalContinuum(sim, CausalModel(sim, seed=seed)),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            finals.append(r["regret_cum"][-1])
        hz.append({"horizon": H, "regret": round(float(np.mean(finals)), 3)})
    # (b) Curves over the 1600-step horizon for Lexi-strict, ws_tuned and ws_default.
    curves = {}
    for m in ("Lexi-strict", "ws_tuned", "ws_default"):
        cc = []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(_factory(m)(sim, cand, seed), sim, cand, reqs, tn, orc,
                           realize_seed=1000 + seed)
            cc.append(r["regret_cum"])
        C = np.array(cc); idxs = np.linspace(0, HORIZON - 1, 80).astype(int)
        curves[m] = [round(float(x), 4) for x in C.mean(axis=0)[idxs]]
    idxs = np.linspace(0, HORIZON - 1, 80).astype(int)
    return {"n_seeds": len(seeds), "horizons": hz,
            "curve_steps": idxs.tolist(), "curves": curves}


def main():
    """Run every experiment, write ../results/core_results.json and print a summary."""
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "..", "results", "raw"), exist_ok=True)
    res = {}; t0 = time.perf_counter()
    print("E1 priority satisfaction ..."); res["e1_satisfaction"] = e1_priority_satisfaction()
    print("E1b priority inversion ...");   res["e1_inversion"] = e1_inversion()
    print("E11 priority-conflict stress ..."); res["e11_conflict_stress"] = e11_conflict_stress()
    print("E1 lex-LP agreement ...");      res["lexlp_agreement"] = lexlp_agreement()
    print("E10 SLO safety margin ...");    res["e10_margin"] = e10_margin_sweep()
    print("E2 weight sweep ...");          res["e2_weight_sweep"] = e2_weight_sweep()
    print("E3 scale-invariance ...");      res["e3_scale_invariance"] = e3_scale_invariance()
    print("E4 stability ...");             res["e4_stability"] = e4_stability()
    print("E4 scalability ...");           res["scalability"] = scalability()
    print("E5 candidate robustness ...");  res["e5_candidate"] = e5_candidate_robustness()
    print("E6 generalization ...");        res["e6_generalization"] = e6_generalization()
    print("E7 misspecification ...");      res["e7_misspecification"] = e7_misspecification()
    print("E8 convergence ...");           res["e8_convergence"] = e8_convergence()
    print("E9 auto-slack ...");            res["e9_autoslack"] = e9_autoslack()
    res["_meta"] = {"horizon": HORIZON, "n_seeds": len(SEEDS), "cand_cap": CAND_CAP,
                    "warmup": WARMUP, "priority": "latency-SLO>-privacy>-carbon>-cost",
                    "runtime_s": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(here, "..", "results", "core_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    # Console summary of the headline quantities.
    print(f"\n[done in {res['_meta']['runtime_s']}s]")
    print("E1 SLO% / privacy (mean [95% CI]):")
    for m in MAIN_METHODS:
        t = res["e1_satisfaction"]["table"][m]
        print(f"  {m:16s} SLO={t['slo']['mean']:5.2f} {t['slo']['ci']}  "
              f"priv={t['privacy']['mean']:.3f} {t['privacy']['ci']}")
    print("E1 paired (Lexi-strict vs ws_tuned):", res["e1_satisfaction"]["paired_tests"])
    e2 = res["e2_weight_sweep"]
    print(f"E2 weight sweep: {e2['frac_meeting_target']*100:.0f}% of {e2['n_weights']} "
          f"weightings meet target; best WS {e2['best_ws']['slo']}/{e2['best_ws']['privacy']}; "
          f"Lexi {e2['lexi_strict']['slo']}/{e2['lexi_strict']['privacy']}")
    print("E1 lex-LP vs strict-Lexi agreement:", res["lexlp_agreement"]["agreement_pure"],
          "(pure);", res["lexlp_agreement"]["agreement_hysteresis"], "(hysteresis)")
    print("E3 scale-invariance (privacy rescaled): PII exposure of each rule")
    for r_ in res["e3_scale_invariance"]["objectives"]["privacy"]:
        print(f"  factor {r_['factor']:5}: Lexi {r_['lexi_priv']:.3f}  "
              f"Lexi-auto {r_['lexi_auto_priv']:.3f}  rank-WS {r_['rank_ws_priv']:.3f}  "
              f"norm-WS {r_['norm_ws_priv']:.3f}  Tcheby {r_['tcheby_priv']:.3f}  "
              f"WS {r_['ws_priv']:.3f}")
    print("E9 auto-slack (frac -> SLO% / PII / churn):")
    for p in res["e9_autoslack"]["points"]:
        print(f"  frac {p['frac']:4}: SLO {p['slo']:5.2f}  PII {p['privacy']:.3f}  "
              f"churn {p['churn']:.3f}")
    print("E1b priority inversion (% of decisions violating the priority order):")
    for m, v in res["e1_inversion"]["inversion"].items():
        print(f"  {m:16s} inv={v['inversion_pct']:.3f}% {v['ci']}")
    print("E11 priority-conflict stress (cold-start -> priority inversion %):")
    for lvl in res["e11_conflict_stress"]["levels"]:
        iv = lvl["inversion"]
        print(f"  cold {lvl['cold_penalty']:4.0f}ms: Lexi {iv['Lexi-strict']['mean']:5.2f}  "
              f"ws-tuned {iv['ws_tuned']['mean']:5.2f}  rank-WS {iv['rank_ws']['mean']:5.2f}  "
              f"norm-WS {iv['norm_ws']['mean']:5.2f}  Tcheby {iv['tcheby']['mean']:5.2f}")
    print("E10 SLO safety margin (nominal: delta -> SLO% / PII / churn):")
    for p in res["e10_margin"]["nominal"]:
        print(f"  delta {p['margin_ms']:4}ms: SLO {p['slo']:5.2f}  PII {p['privacy']:.3f}  "
              f"churn {p['churn']:.3f}")
    print("  sig default vs strict:", res["e10_margin"]["sig_default_vs_strict_slo"],
          " vs Constr-WS:", res["e10_margin"]["sig_default_vs_constrained_slo"])
    print("E10 SLO safety margin (contention 0.4: delta -> SLO%):")
    for p in res["e10_margin"]["contention"]:
        print(f"  delta {p['margin_ms']:4}ms: SLO {p['slo']:5.2f}  PII {p['privacy']:.3f}")


if __name__ == "__main__":
    main()
