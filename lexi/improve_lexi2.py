"""
Second part of the safety-margin study: larger noise regimes and a priority-inversion
diagnostic. The extended version reports the margin result as RQ5 ("a safety margin
protects the SLO"). The camera-ready paper does not include it. The first part is
improve_lexi.py, and the production version of the margin controller is
CausalContinuumMargin in controller.py.

The directory keeps its earlier name, causalcontinuum.

Two questions.
  (a) Does an SLO-margin-aware top level help where latency noise is large, namely
      serverless cold starts (90 ms penalty) and co-location contention (coefficient 0.4)?
      regime() runs Lexi-strict, Constr-WS (constrained_rl), WS-tuned and LexiMargin for
      margins 0, 4, 8 and 12 ms in one regime over 15 seeds, and gives paired-bootstrap
      tests of the best margin against Constr-WS and against Lexi-strict. The nominal
      regime reproduces the margin values quoted in RQ5.
  (b) What is strict Lexi better at than Constr-WS, given that their nominal means are
      close? inversion_and_conflict() scores every rule on one shared, warmed-up model
      per seed. It reports the priority-inversion rate (the Inv% notion of Table 3) and,
      on the steps where privacy and carbon truly conflict (an SLO-feasible private
      candidate exists and an SLO-feasible public candidate has lower carbon), how often
      the rule's pick keeps PII private. The conflict-step analysis is a diagnostic. The
      paper does not report it as a table.

The simulator, outcome model, workload and window are those of experiments_core.py.
Writes ../results/improve_lexi2.json with the keys nominal, cold, contention and inversion.
"""
from __future__ import annotations
import json, os
import numpy as np

from sim import Continuum, make_workload, LAT, PRIV, CARB, COST
from causal import CausalModel
from controller import (CausalContinuum, candidate_set, thresholded_lex_select)
from baselines import ConstrainedRL, WeightedSum, TUNED_WS_W
from improve_lexi import est_raw_latency

HORIZON = 1600
WARMUP = 800
SEEDS = list(range(15))
CAND_CAP = 40
STRICT = np.zeros(3)


class LexiMargin:
    """Safety-margin controller of the study, equal in decision logic to
    controller.CausalContinuumMargin.

    It keeps the candidates predicted to meet the SLO with headroom margin, meaning
    E[latency] <= SLO - margin. If none qualifies it relaxes to E[latency] <= SLO and then
    to the least-overrunning candidate. Privacy (slack eta_priv) and carbon (slack eta_carb)
    follow, and the last tie-break maximises headroom. Margin = 0 is strict feasibility.
    A positive margin reserves headroom for the queueing noise and lowers realised
    violations."""
    name = "Lexi-margin"

    def __init__(self, sim, model, margin_ms=0.0, eta_priv=0.0, eta_carb=0.06, hold_ms=8.0):
        self.sim = sim; self.model = model
        self.margin = float(margin_ms); self.eta_priv = float(eta_priv)
        self.eta_carb = float(eta_carb); self.hold_ms = float(hold_ms)

    def reset(self): pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)
        raw = est_raw_latency(self.model, cand, req)
        idx = np.where(raw <= self.sim.slo - self.margin)[0]
        if len(idx) == 0:
            idx = np.where(raw <= self.sim.slo)[0]
        if len(idx) == 0:
            return int(np.argmin(raw))
        for k, eta in ((PRIV, self.eta_priv), (CARB, self.eta_carb)):
            best = est[idx, k].min()
            idx = idx[est[idx, k] <= best + eta + 1e-12]
        headroom = self.sim.slo - raw[idx]
        roomiest = idx[int(np.argmax(headroom))]
        if prev_idx is not None and prev_idx in idx:
            prev_room = self.sim.slo - raw[prev_idx]
            if headroom.max() - prev_room <= self.hold_ms:
                return int(prev_idx)
        return int(roomiest)

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


def lex_worse(a, b, tol=1e-9):
    """True if vector a is lexicographically worse than b, meaning strictly greater at the
    first coordinate (hinge, privacy, carbon, cost) where they differ by more than tol."""
    for k in range(4):
        if a[k] > b[k] + tol:
            return True
        if a[k] < b[k] - tol:
            return False
    return False


def run(method, sim, cand, requests, diag_model, realize_seed, want_inv=False):
    """Run one method over one seed and return realised metrics for the post-warmup window.

    With want_inv set, each decision is also scored for a priority inversion against the
    strict optimum on the estimates of diag_model. The regime study leaves it off."""
    rng = np.random.default_rng(realize_seed)
    method.reset()
    H = len(requests)
    slo = np.zeros(H); privacy = np.zeros(H); carbon = np.zeros(H)
    cost = np.zeros(H); churn = np.zeros(H); inv = np.zeros(H)
    prev_idx = prev_place = None
    for step, req in enumerate(requests):
        c_idx = method.act(req, cand, prev_idx)
        if want_inv:
            est = diag_model.predict_all(cand, req)      # the shared model's estimates
            o = thresholded_lex_select(est, STRICT, prev_idx=None)  # strict lexicographic optimum
            inv[step] = 1.0 if lex_worse(est[c_idx], est[o]) else 0.0
        place = cand[c_idx]
        outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
        method.learn(c_idx, place, obs, req)
        slo[step] = 1.0 if slo_v else 0.0
        privacy[step] = outcome[PRIV]; carbon[step] = outcome[CARB]
        cost[step] = outcome[COST]; churn[step] = ch
        prev_idx, prev_place = c_idx, place
    ev = slice(WARMUP, H)
    return dict(slo=100.0 * slo[ev].mean(), privacy=privacy[ev].mean(),
                carbon=carbon[ev].mean(), cost=cost[ev].mean(),
                churn=churn[ev].mean(), inv=100.0 * inv[ev].mean())


def paired_p(a, b, n_boot=10000, seed=0):
    """Paired-bootstrap two-sided p-value and median difference, as
    experiments_core.paired_boot_p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    md = d[idx].mean(axis=1)
    return float(min(1.0, 2.0 * min((md >= 0).mean(), (md <= 0).mean()))), float(np.median(d))


def regime(cold_penalty=0.0, contention=0.0, margins=(0.0, 4.0, 8.0, 12.0)):
    """Compare Lexi-strict, Constr-WS, WS-tuned and LexiMargin over the margin sweep in one
    noise regime, given by cold_penalty (ms) and contention. Returns the per-method mean
    metrics, the margin with the lowest SLO percent and its paired tests on SLO."""
    methods = {
        "Lexi-strict":    lambda s, m: CausalContinuum(s, m, eta=STRICT),
        "constrained_rl": lambda s, m: ConstrainedRL(s, m),
        "ws_tuned":       lambda s, m: WeightedSum(s, m, TUNED_WS_W),
    }
    for mg in margins:
        methods[f"Lexi-margin({mg:g})"] = (lambda s, m, mg=mg: LexiMargin(s, m, margin_ms=mg))
    agg = {n: {k: [] for k in ("slo", "privacy", "carbon", "cost", "churn")} for n in methods}
    for seed in SEEDS:
        sim = Continuum(seed=seed, cold_penalty=cold_penalty, contention=contention)
        cand = candidate_set(sim, cap=CAND_CAP, seed=0)
        requests = make_workload(HORIZON, seed=seed)
        for n, make in methods.items():
            model = CausalModel(sim, seed=seed)
            r = run(make(sim, model), sim, cand, requests, None, realize_seed=1000 + seed)
            for k in agg[n]:
                agg[n][k].append(r[k])
    table = {n: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for n, d in agg.items()}
    # Paired tests of the best margin against Constr-WS and against Lexi-strict on SLO.
    best_mg = min((f"Lexi-margin({mg:g})" for mg in margins),
                  key=lambda nm: table[nm]["slo"])
    sig = {}
    for ref in ("constrained_rl", "Lexi-strict"):
        p, md = paired_p(agg[best_mg]["slo"], agg[ref]["slo"])
        sig[f"{best_mg}_vs_{ref}_slo"] = {"p": round(p, 4), "median_diff": round(md, 4)}
    return {"table": table, "best_margin": best_mg, "sig": sig}


def inversion_and_conflict():
    """Priority-inversion rate plus a conflict-subset analysis, on one shared model per seed.

    The inversion rate is the share of decisions that are lexicographically worse than
    the strict optimum on the estimates the rule saw. The conflict subset holds the steps
    where privacy and carbon really pull apart. Among the SLO-feasible candidates there is
    a zero-PII option and a public option with lower carbon. On those steps the function
    reports how often each rule keeps PII private."""
    methods = {
        "Lexi-strict":    lambda s, m: CausalContinuum(s, m, eta=STRICT),
        "constrained_rl": lambda s, m: ConstrainedRL(s, m),
        "ws_tuned":       lambda s, m: WeightedSum(s, m, TUNED_WS_W),
        "Lexi-margin(8)": lambda s, m: LexiMargin(s, m, margin_ms=8.0),
    }
    inv = {n: [] for n in methods}
    # Conflict-subset privacy. On steps with an SLO-feasible zero-PII candidate and an
    # SLO-feasible but public candidate of strictly lower carbon, count how often the
    # method's pick keeps PII private.
    conf_priv = {n: [] for n in methods}
    conf_frac = []
    for seed in SEEDS:
        sim = Continuum(seed=seed)
        cand = candidate_set(sim, cap=CAND_CAP, seed=0)
        requests = make_workload(HORIZON, seed=seed)
        # Shared model, trained once on a strict-Lexi rollout of WARMUP steps.
        diag = CausalModel(sim, seed=seed)
        warm = CausalContinuum(sim, diag, eta=STRICT)
        rngw = np.random.default_rng(2000 + seed)
        prev = None
        for req in requests[:WARMUP]:
            c = warm.act(req, cand, None)
            _, _, _, obs = sim.realize(cand[c], req, prev, rngw)
            diag.update(obs, req); prev = cand[c]
        # Score the decisions on this model's estimates over the evaluation window.
        per = {n: [] for n in methods}
        cpriv = {n: [] for n in methods}
        cflags = []
        for req in requests[WARMUP:]:
            est = diag.predict_all(cand, req)
            raw = est_raw_latency(diag, cand, req)
            feas = np.where(raw <= sim.slo)[0]
            if len(feas) == 0:
                feas = np.array([int(np.argmin(raw))])
            o = thresholded_lex_select(est, STRICT, prev_idx=None)
            # A conflict step has a zero-PII option and a lower-carbon public option among
            # the feasible candidates, which means privacy and carbon genuinely compete.
            pii_feas = est[feas, PRIV]
            has_private = (pii_feas <= 1e-9).any()
            has_greener_public = False
            if has_private:
                minc_private = est[feas][pii_feas <= 1e-9, CARB].min()
                pub = est[feas][pii_feas > 1e-9]
                if len(pub) and pub[:, CARB].min() < minc_private - 1e-9:
                    has_greener_public = True
            conflict = has_private and has_greener_public
            cflags.append(1.0 if conflict else 0.0)
            for n, make in methods.items():
                model = make(sim, diag)          # the rule, reading the shared model
                c = model.act(req, cand, None)
                per[n].append(1.0 if lex_worse(est[c], est[o]) else 0.0)
                if conflict:
                    cpriv[n].append(1.0 if est[c, PRIV] <= 1e-9 else 0.0)
        for n in methods:
            inv[n].append(100.0 * np.mean(per[n]))
            if cpriv[n]:
                conf_priv[n].append(100.0 * np.mean(cpriv[n]))
        conf_frac.append(100.0 * np.mean(cflags))
    return {
        "inversion_pct": {n: round(float(np.mean(v)), 3) for n, v in inv.items()},
        "conflict_step_pct": round(float(np.mean(conf_frac)), 3),
        "conflict_keeps_private_pct": {n: (round(float(np.mean(v)), 2) if v else None)
                                       for n, v in conf_priv.items()},
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    print("=== nominal ===")
    out["nominal"] = regime()
    for n, t in out["nominal"]["table"].items():
        print(f"  {n:18s} SLO={t['slo']:5.2f} PII={t['privacy']:.3f} carb={t['carbon']:.3f} "
              f"cost={t['cost']:.3f} churn={t['churn']:.3f}")
    print("  sig:", out["nominal"]["sig"])
    print("=== cold-start (90ms) ===")
    out["cold"] = regime(cold_penalty=90.0)
    for n, t in out["cold"]["table"].items():
        print(f"  {n:18s} SLO={t['slo']:5.2f} PII={t['privacy']:.3f} churn={t['churn']:.3f}")
    print("  sig:", out["cold"]["sig"])
    print("=== contention (0.4) ===")
    out["contention"] = regime(contention=0.4)
    for n, t in out["contention"]["table"].items():
        print(f"  {n:18s} SLO={t['slo']:5.2f} PII={t['privacy']:.3f} churn={t['churn']:.3f}")
    print("  sig:", out["contention"]["sig"])
    print("=== priority inversion + conflict ===")
    out["inversion"] = inversion_and_conflict()
    print("  inversion %:", out["inversion"]["inversion_pct"])
    print("  conflict-step %:", out["inversion"]["conflict_step_pct"])
    print("  conflict keeps-private %:", out["inversion"]["conflict_keeps_private_pct"])
    with open(os.path.join(here, "..", "results", "improve_lexi2.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
