"""
Diagnosis of the residual SLO violations of strict Lexi, and the selector variants that
address them. This is the study behind the safety-margin controller
CausalContinuumMargin in controller.py, which the extended version reports as RQ5
("a safety margin protects the SLO"). The camera-ready paper does not include it.

The directory keeps its earlier name, causalcontinuum.

Question. In the E1 operator table (Table 3), Constr-WS has a slightly lower SLO
violation rate than strict Lexi, at equal privacy. Where does the difference come from,
and can it be removed without giving up what defines Lexi, namely no weights, invariance
to units and a strict priority order?

Working hypothesis. The top objective is a threshold, the SLO hinge. Once several candidates are
predicted to meet the SLO, all of them are tied at hinge 0, which means strict Lexi chooses among
them by privacy, then carbon, then cost. It never looks at how much latency headroom
each has below the SLO. Realised latency carries queueing noise whose scale grows with
each node's queue sensitivity, which means a placement that is feasible but tight crosses the SLO
under noise. Constr-WS prices carbon and cost, and it happens to land on roomier
placements. It would therefore suffer fewer noise-driven violations.

Variants compared, all lexicographic, weight-free and order-only on the lower priorities.
  Lexi-strict      the strict rule, eta = 0.
  Lexi-thr(ship)   the thresholded rule with controller.DEFAULT_ETA.
  Lexi-thr-fixed   slack only on carbon. It isolates the effect of putting slack on the
                   SLO hinge, which admits SLO-overrunning placements into the privacy
                   level. A threshold objective must not be relaxed.
  Lexi-headroom    strict order, with latency headroom replacing cost as the last
                   tie-break.
  Lexi-robust      strict hinge, a small band on privacy and carbon, then a headroom
                   tie-break with hysteresis. This is a chance-constraint surrogate:
                   among priority-equivalent placements it takes the one least likely
                   to breach the SLO. Headroom is measured on the top objective, whose
                   priority is already fixed, which means the order-only treatment of privacy,
                   carbon and cost is untouched.
  constrained_rl   Constr-WS, the reference.

The simulator, outcome model and workload are the ones of the paper. The script uses 15
seeds, the post-warmup window of experiments_core.py, and prints a table. It writes
../results/improve_lexi.json (per-method means of SLO percent, PII, carbon, cost, churn
and mean chosen headroom in ms). improve_lexi2.py continues the study under larger noise.
"""
from __future__ import annotations
import json, os
import numpy as np

from sim import Continuum, make_workload, LAT, PRIV, CARB, COST
from causal import CausalModel
from controller import (CausalContinuum, candidate_set, oracle_index, lex_scalar,
                        thresholded_lex_select, DEFAULT_ETA)
from baselines import ConstrainedRL, WeightedSum, TUNED_WS_W

HORIZON = 1600
WARMUP = 800
SEEDS = list(range(15))
CAND_CAP = 40
STRICT = np.zeros(3)


def est_raw_latency(model: CausalModel, cand: np.ndarray, req) -> np.ndarray:
    """Predicted critical-path latency in ms per candidate, before hinge and normalisation.

    The headroom SLO - E[latency] needs the raw value, which the hinge clips to 0 for
    every feasible placement. The computation is the latency path of
    CausalModel.predict_all. CausalModel.predict_raw_latency computes the same quantity for the
    nominal model."""
    theta_lat, _, _ = model._thetas()
    same = (model.sim.region == req.origin).astype(float)
    K, M = model.sim.K, model.M
    latKM = np.empty((K, M))
    for s in range(K):
        d = model.sim.demand[s]
        X = np.stack([np.ones(M), np.full(M, d), np.full(M, req.load), same], axis=1)
        latKM[s] = (X * theta_lat).sum(axis=1)
    stage_lat = latKM[np.arange(K)[None, :], cand]
    return model.sim._critical_path(stage_lat)          # [n_cand], ms


class LexiRobust:
    """Strict SLO level, a slack band on the lower priorities, and a headroom tie-break.

    Order: SLO hinge, privacy, carbon, cost.
      Level 0 (SLO), strict. Keep the candidates with minimal estimated hinge. Every
        placement predicted feasible ties at hinge 0, which means this is a hard feasibility
        filter and never carries slack.
      Levels 1 and 2. Keep the candidates within eta_priv of the best privacy and then
        within eta_carb of the best carbon, a small band of priority-equivalent placements.
      Final tie-break. Within the band, maximise the predicted latency headroom
        SLO - E[latency], which picks the least risky placement. The previous placement is
        kept only if its headroom is within hold_ms of the roomiest option. This limits
        churn without chasing sub-millisecond differences in headroom.
    """
    name = "Lexi-robust"

    def __init__(self, sim, model, eta_priv=0.0, eta_carb=0.06, hold_ms=8.0):
        self.sim = sim
        self.model = model
        self.eta_priv = float(eta_priv)
        self.eta_carb = float(eta_carb)
        self.hold_ms = float(hold_ms)

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)          # normalised, [n, 4]
        idx = np.arange(est.shape[0])
        # Level 0: strict SLO feasibility, no slack.
        best_h = est[idx, LAT].min()
        idx = idx[est[idx, LAT] <= best_h + 1e-12]
        # Levels 1 and 2 with slack give the band of priority-equivalent placements.
        for k, eta in ((PRIV, self.eta_priv), (CARB, self.eta_carb)):
            best = est[idx, k].min()
            idx = idx[est[idx, k] <= best + eta + 1e-12]
        # Tie-break on headroom SLO - E[latency], the least SLO-risky choice.
        raw_lat = est_raw_latency(self.model, cand[idx], req)
        headroom = self.sim.slo - raw_lat
        roomiest = idx[int(np.argmax(headroom))]
        if prev_idx is not None and prev_idx in idx:
            prev_room = self.sim.slo - est_raw_latency(self.model, cand[[prev_idx]], req)[0]
            best_room = headroom.max()
            if best_room - prev_room <= self.hold_ms:
                return int(prev_idx)
        return int(roomiest)

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class LexiHeadroomOnly:
    """Ablation with the strict order of Lexi-strict, where the last tie-break after the
    carbon level is latency headroom instead of cost. It isolates the value of the
    risk-aware tie-break when the lower priorities carry no slack. Its row in the
    committed result file is identical to that of Lexi-strict."""
    name = "Lexi-headroom"

    def __init__(self, sim, model):
        self.sim = sim
        self.model = model

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)
        idx = np.arange(est.shape[0])
        for k in range(3):                               # hinge, privacy, carbon, all strict
            best = est[idx, k].min()
            idx = idx[est[idx, k] <= best + 1e-12]
        if prev_idx is not None and prev_idx in idx:
            return int(prev_idx)
        raw_lat = est_raw_latency(self.model, cand[idx], req)
        return int(idx[int(np.argmax(self.sim.slo - raw_lat))])

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class ThreshLexiFixed:
    """Thresholded Lexi with all slack moved off the top priorities, eta = [0, 0, eta_carb].

    Comparing it with a rule that also slacks the SLO hinge shows that the rise in SLO
    violations comes from slack on the hinge and not from thresholding as such."""
    name = "Lexi-thr-fixed"

    def __init__(self, sim, model, eta_carb=0.08):
        self.sim = sim
        self.model = model
        self.eta = np.array([0.0, 0.0, float(eta_carb)])

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)
        return thresholded_lex_select(est, self.eta, prev_idx)

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


def run(method, sim, cand, requests, model_for_diag, realize_seed):
    """Run one method over one seed and return realised metrics for the post-warmup window.

    Besides the usual metrics it records the headroom SLO - E[latency] of each chosen
    placement, read from model_for_diag, and returns its mean in ms. A larger value means
    safer decisions."""
    rng = np.random.default_rng(realize_seed)
    method.reset()
    H = len(requests)
    slo = np.zeros(H); carbon = np.zeros(H); cost = np.zeros(H)
    privacy = np.zeros(H); churn = np.zeros(H); headroom = np.full(H, np.nan)
    prev_idx = prev_place = None
    for step, req in enumerate(requests):
        c_idx = method.act(req, cand, prev_idx)
        place = cand[c_idx]
        # Headroom of the chosen placement according to the diagnostic model.
        raw = est_raw_latency(model_for_diag, cand[[c_idx]], req)[0]
        headroom[step] = sim.slo - raw
        outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
        method.learn(c_idx, place, obs, req)
        slo[step] = 1.0 if slo_v else 0.0
        carbon[step] = outcome[CARB]; cost[step] = outcome[COST]
        privacy[step] = outcome[PRIV]; churn[step] = ch
        prev_idx, prev_place = c_idx, place
    ev = slice(WARMUP, H)
    return dict(slo=100.0 * slo[ev].mean(), privacy=privacy[ev].mean(),
                carbon=carbon[ev].mean(), cost=cost[ev].mean(),
                churn=churn[ev].mean(), headroom=float(np.nanmean(headroom[ev])))


# Constructors of the compared variants, keyed by the row names of the result table.
METHODS = {
    "Lexi-strict":   lambda s, m: CausalContinuum(s, m, eta=STRICT),
    "Lexi-thr(ship)":lambda s, m: CausalContinuum(s, m, eta=DEFAULT_ETA),
    "Lexi-thr-fixed":lambda s, m: ThreshLexiFixed(s, m, eta_carb=0.08),
    "constrained_rl":lambda s, m: ConstrainedRL(s, m),
    "Lexi-headroom": lambda s, m: LexiHeadroomOnly(s, m),
    "Lexi-robust":   lambda s, m: LexiRobust(s, m, eta_priv=0.0, eta_carb=0.06, hold_ms=8.0),
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    agg = {name: {k: [] for k in ("slo", "privacy", "carbon", "cost", "churn", "headroom")}
           for name in METHODS}
    for seed in SEEDS:
        sim = Continuum(seed=seed)
        cand = candidate_set(sim, cap=CAND_CAP, seed=0)
        requests = make_workload(HORIZON, seed=seed)
        # diag_model is built here but not used. Each method's headroom is read from its
        # own model, which is what run receives below.
        diag_model = CausalModel(sim, seed=seed)
        for name, make in METHODS.items():
            model = CausalModel(sim, seed=seed)
            r = run(make(sim, model), sim, cand, requests, model, realize_seed=1000 + seed)
            for k in agg[name]:
                agg[name][k].append(r[k])
    # Per-method means over seeds.
    def stat(xs):
        x = np.asarray(xs); return float(x.mean())
    table = {name: {k: round(stat(v), 4) for k, v in d.items()} for name, d in agg.items()}
    with open(os.path.join(here, "..", "results", "improve_lexi.json"), "w") as f:
        json.dump({"n_seeds": len(SEEDS), "horizon": HORIZON, "warmup": WARMUP,
                   "table": table}, f, indent=2)
    # Console table.
    cols = ("slo", "privacy", "carbon", "cost", "churn", "headroom")
    print(f"{'method':16s} " + " ".join(f"{c:>9}" for c in cols))
    for name in METHODS:
        t = table[name]
        print(f"{name:16s} " + " ".join(f"{t[c]:>9.3f}" for c in cols))
    print("\n(lower is better for slo/privacy/carbon/cost/churn; headroom = mean "
          "ms below SLO of the chosen placement, higher = safer)")


if __name__ == "__main__":
    main()
