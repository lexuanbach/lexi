#!/usr/bin/env python3
"""
Decision-level quantities that the papers quote and that no other script stores.
Output: ../results/decision_checks.json. The run is deterministic and takes well under a
minute on a laptop.

The script adds no new modelling. It replays the simulator, outcome model, controller and
baselines of experiments_core.py with the same HORIZON, WARMUP, candidate menus and
realisation seeds (1000 + seed), and records four groups of numbers.

  oracle_agreement   Share of post-warmup decisions (steps 800 to 1599, seeds 0 to 7) on
                     which strict Lexi and Constr-WS pick the expected-objective priority
                     optimum (experiments_core._precompute), on the curated and on the
                     uniformly random menu. Camera-ready Sect. 6.3 (last sentence, 99.7%)
                     and extended Sect. 9.4 and 9.5 (99.7%, 98.9%, 99.1%, 87.6%).
  worked_decision    The step-904 estimate matrix of Table 1 (seed 0) handed to the other
                     rules: the choices of WS-default, WS-tuned and Constr-WS, the WS-tuned
                     scores of candidates 21 and 6 at nominal scale and with privacy
                     under-scaled tenfold, the survivors and the choice under a carbon band
                     of 0.08, the estimate row of candidate 18, the predicted carbon of the
                     four level-1 survivors and the incumbent kept by the stability hold.
                     Camera-ready Sect. 3 and extended Sect. 3.3, 4.3 and 4.5.
  hysteresis_effect  Share of decisions on which the stability hold of
                     controller.thresholded_lex_select (keep the previous placement when it
                     survives the three bands) changes the strict choice, measured along the
                     strict Lexi trajectory of the main run (15 seeds), the real-trace run
                     (15 seeds) and the random menu (8 seeds). Def. 2 of the camera-ready ends
                     with a cost tie-break, and the evaluated controller applies the hold
                     first. Camera-ready Sect. 5 reports that Lex-LP, the rule without the
                     hold, matches Lexi on all 24,000 main-run decisions.
  menu_carbon_range  Smallest and largest noise-free carbon (g per request) over the
                     curated menu at the post-warmup steps of the 15 main-run seeds. Extended
                     Sect. 9.5 ("a fleet range of roughly 0.3 to 2.6 g").

Usage:
    python3 decision_checks.py
"""
import json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sim import Continuum, make_workload, PRIV, CARB, COST, SCALES
from causal import CausalModel
from controller import CausalContinuum, candidate_set, thresholded_lex_select
from baselines import ConstrainedRL, DEFAULT_WS_W, TUNED_WS_W
from experiments_core import (_precompute, run_method, _curated, _random_cand,
                              HORIZON, WARMUP, CAND_CAP, SEEDS)

OUTPATH = os.path.join(_HERE, "..", "results", "decision_checks.json")
STRICT = np.zeros(3)
WORKED_SEED, WORKED_STEP = 0, 904


class _Recorder:
    """Wrap a controller and keep the index it chooses at every step."""

    def __init__(self, inner):
        self.inner = inner
        self.chosen = []

    def reset(self):
        self.inner.reset()
        self.chosen = []

    def act(self, req, cand, prev_idx):
        c = self.inner.act(req, cand, prev_idx)
        self.chosen.append(int(c))
        return c

    def learn(self, *a):
        self.inner.learn(*a)


def oracle_agreement(seeds=list(range(8))):
    """Per-step agreement with the expected-objective oracle, post-warmup, by menu."""
    builders = {
        "lexi": lambda sim, seed: CausalContinuum(sim, CausalModel(sim, seed=seed), eta=STRICT),
        "constr_ws": lambda sim, seed: ConstrainedRL(sim, CausalModel(sim, seed=seed)),
    }
    out = {"n_seeds": len(seeds), "window": [WARMUP, HORIZON - 1]}
    for label, cand_fn in (("curated", _curated), ("random", _random_cand)):
        row = {}
        for name, build in builders.items():
            per_seed = []
            for seed in seeds:
                sim = Continuum(seed=seed)
                cand = cand_fn(sim, seed)
                reqs = make_workload(HORIZON, seed=seed)
                tn, orc = _precompute(sim, cand, reqs)
                rec = _Recorder(build(sim, seed))
                run_method(rec, sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
                chosen = np.array(rec.chosen)[WARMUP:]
                per_seed.append(float(np.mean(chosen == np.asarray(orc)[WARMUP:])))
            row[name + "_pct"] = round(100.0 * float(np.mean(per_seed)), 2)
        out[label] = row
    return out


def worked_decision(seed=WORKED_SEED, step=WORKED_STEP):
    """Replay strict Lexi to the worked step and apply the other rules to its matrix."""
    sim = Continuum(seed=seed)
    cand = candidate_set(sim, cap=CAND_CAP, seed=0)
    reqs = make_workload(HORIZON, seed=seed)
    model = CausalModel(sim, seed=seed)
    ctrl = CausalContinuum(sim, model, eta=STRICT)
    rng = np.random.default_rng(1000 + seed)
    prev_idx = prev_place = None
    for s in range(step):
        req = reqs[s]
        c = ctrl.act(req, cand, prev_idx)
        place = cand[c]
        _, _, _, obs = sim.realize(place, req, prev_place, rng)
        ctrl.learn(c, place, obs, req)
        prev_idx, prev_place = c, place
    req = reqs[step]
    est = model.predict_all(cand, req)
    raw_lat = model.predict_raw_latency(cand, req)

    tuned = est @ TUNED_WS_W
    under = est.copy()
    under[:, PRIV] *= 0.1
    tuned_under = under @ TUNED_WS_W

    band = np.array([0.0, 0.0, 0.08])
    idx = np.arange(len(cand))
    for k in range(3):
        best = est[idx, k].min()
        idx = idx[est[idx, k] <= best + band[k] + 1e-12]

    invariant = all(
        thresholded_lex_select(est * np.array([1.0, f, 1.0, 1.0]), STRICT, None)
        == thresholded_lex_select(est, STRICT, None)
        for f in (0.1, 0.3, 3.0, 10.0, 30.0))

    return {
        "seed": seed, "step": step,
        "strict_choice": thresholded_lex_select(est, STRICT, None),
        "strict_choice_with_hold": thresholded_lex_select(est, STRICT, prev_idx),
        "incumbent": int(prev_idx),
        "ws_default_choice": int(np.argmin(est @ DEFAULT_WS_W)),
        "ws_tuned_choice": int(np.argmin(tuned)),
        "ws_tuned_score": {"21": round(float(tuned[21]), 4), "6": round(float(tuned[6]), 4)},
        "ws_tuned_privacy_x0.1_choice": int(np.argmin(tuned_under)),
        "ws_tuned_privacy_x0.1_score": {"21": round(float(tuned_under[21]), 4),
                                        "6": round(float(tuned_under[6]), 4)},
        "constr_ws_choice": int(ConstrainedRL(sim, model).act(req, cand, None)),
        "carbon_band_0.08_survivors": [int(i) for i in idx],
        "carbon_band_0.08_choice": thresholded_lex_select(est, band, None),
        "cand18_pred_norm": [round(float(x), 3) for x in est[18]],
        "cand18_pred_latency_ms": round(float(raw_lat[18]), 2),
        "level1_pred_carbon_g": {str(i): round(float(est[i, CARB] * SCALES[CARB]), 3)
                                 for i in (2, 19, 20, 21)},
        "level1_pred_cost": {str(i): round(float(est[i, COST] * SCALES[COST]), 3)
                             for i in (2, 19, 20, 21)},
        "strict_choice_invariant_to_privacy_rescaling": bool(invariant),
    }


def _hold_changes(sim, cand, reqs, seed):
    """Count the steps on which the hold changes the strict choice along the strict
    Lexi trajectory (all steps, and post-warmup steps)."""
    model = CausalModel(sim, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    prev_idx = prev_place = None
    n_all = n_post = 0
    for step, req in enumerate(reqs):
        est = model.predict_all(cand, req)
        pure = thresholded_lex_select(est, STRICT, None)
        held = thresholded_lex_select(est, STRICT, prev_idx)
        if pure != held:
            n_all += 1
            n_post += int(step >= WARMUP)
        place = cand[held]
        _, _, _, obs = sim.realize(place, req, prev_place, rng)
        model.update(obs, req)
        prev_idx, prev_place = held, place
    return n_all, n_post


def hysteresis_effect():
    """Share of decisions on which the stability hold changes the strict choice."""
    out = {}
    settings = (
        ("main", SEEDS, False, _curated),
        ("real_traces", SEEDS, True, None),
        ("random_menu", list(range(8)), False, _random_cand),
    )
    for label, seeds, real, cand_fn in settings:
        n_all = n_post = 0
        for seed in seeds:
            sim = Continuum(seed=seed, real_traces=real)
            if cand_fn is None:
                cand = candidate_set(Continuum(real_traces=True), cap=CAND_CAP, seed=0)
            else:
                cand = cand_fn(sim, seed)
            reqs = make_workload(HORIZON, seed=seed, real_traces=real)
            a, p = _hold_changes(sim, cand, reqs, seed)
            n_all += a
            n_post += p
        steps = HORIZON * len(seeds)
        post = (HORIZON - WARMUP) * len(seeds)
        out[label] = {"n_seeds": len(seeds), "n_steps": steps,
                      "changed_steps": n_all,
                      "changed_pct": round(100.0 * n_all / steps, 3),
                      "changed_steps_post_warmup": n_post,
                      "changed_pct_post_warmup": round(100.0 * n_post / post, 3)}
    return out


def menu_carbon_range(seeds=SEEDS):
    """Range of the noise-free per-request carbon over the curated menu, post-warmup."""
    lo, hi = np.inf, -np.inf
    for seed in seeds:
        sim = Continuum(seed=seed)
        cand = _curated(sim, seed)
        for req in make_workload(HORIZON, seed=seed)[WARMUP:]:
            carb = sim.expected_all(cand, req)[:, CARB]
            lo, hi = min(lo, float(carb.min())), max(hi, float(carb.max()))
    return {"n_seeds": len(seeds), "min_g": round(lo, 3), "max_g": round(hi, 3)}


if __name__ == "__main__":
    res = {"oracle_agreement": oracle_agreement()}
    print("oracle agreement:", res["oracle_agreement"], flush=True)
    res["worked_decision"] = worked_decision()
    print("worked decision:", res["worked_decision"], flush=True)
    res["hysteresis_effect"] = hysteresis_effect()
    print("hysteresis effect:", res["hysteresis_effect"], flush=True)
    res["menu_carbon_range"] = menu_carbon_range()
    print("menu carbon range:", res["menu_carbon_range"], flush=True)
    os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
    with open(OUTPATH, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {os.path.normpath(os.path.relpath(OUTPATH, os.getcwd()))}")
