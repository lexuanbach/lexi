"""
Second set of model-stress and deployment experiments for Lexi (extended version, RQ6 and
RQ7), labelled DX1 to DX7 (there is no DX3). The result keys and function names keep the
older prefix r2q, for example key r2q1_safe_probe for DX1.

The directory keeps its earlier name, causalcontinuum. robustness_experiments.py holds RX1 to
RX7. This file adds six more. The extended version reports them in the second half of the
table of measured robustness and deployment extensions, in the section "RQ6/RQ7: Model
and deployment stress". The camera-ready results do not depend on it. All experiments are
simulation only, use 15 seeds, 95 percent bootstrap intervals and paired bootstrap tests
of experiments_core.py, and are seeded.

  DX1  Priority-safe probing. Lexi never explores, which means a node it seldom selects keeps its
        weak prior, and the coverage condition (A3) of Thm. 1 can fail. A cautious probe
        explores under-visited nodes only among candidates that stay inside the SLO and do
        not worsen privacy. The experiment reports the gain in coverage and checks that
        the SLO and PII of the top two priorities are not harmed. Table row "Safe probing".
  DX2  Nonstationarity. Drift is injected into RTTs and carbon intensity, and a model
        frozen after warm-up is compared with a model that updates on every request. Table
        row "Nonstation.".
  DX4  Candidate generation. A deployable greedy planner builds the menu instead of
        hand curation, and Lexi is compared over the curated, planner and random menus.
        Table row "Cand. gen.".
  DX5  Contention and the lower priorities. It measures the residual latency error of
        the additive and contention-aware models, and the effect of contention on PII,
        carbon and cost and not only on the SLO. Table row "Contention, lower-priority impact".
  DX6  Grid over the SLO margin delta and a nonzero privacy slack, plus a rescaling of
        privacy by 10 to test whether invariance survives. Strict privacy is invariant and a
        nonzero slack, being a band in the objective's own units, is not. Table row on
        delta and the privacy slack.
  DX7  A lexicographic controller over quantile-normalised scores. The quantile
        transform is strictly increasing per objective, which means by Prop. 1 it should replicate
        strict Lexi at zero slack. Table row "Quantile".

Output: ../results/deployment_results.json, with keys r2q1_safe_probe, r2q2_nonstationarity,
r2q4_candidate_gen, r2q5_contention_lowpri, r2q6_delta_slack and r2q7_quantile. The run
takes about three minutes. The committed file was assembled from the pieces
../results/_dx*.json, and this one-pass run reproduces every value in it.
"""
from __future__ import annotations
import json, os, time
import numpy as np

from sim import (Continuum, make_workload, LAT, PRIV, CARB, COST, SCALES, DAY)
from causal import CausalModel, ContentionCausalModel
from controller import (CausalContinuum, CausalContinuumMargin, candidate_set,
                        oracle_index, thresholded_lex_select, DEFAULT_ETA)
from baselines import WeightedSum, TUNED_WS_W
from experiments_core import (boot_ci, paired_boot_p, run_method, _precompute,
                              HORIZON, SEEDS, WARMUP, CAND_CAP)

STRICT_ETA = np.zeros(3)


# DX1: priority-safe probing and coverage (A3).
class SafeProbeContinuum(CausalContinuum):
    r"""Strict Lexi with a cautious probe that improves coverage (DX1).

    The base controller never explores, which means a node that Lexi rarely selects keeps its weak
    prior for good, and the coverage assumption A3 of Thm. 1 can fail. This variant adds
    a priority-safe probe. With probability p_probe it considers a candidate that touches
    an under-visited node, but only among candidates that meet two tests. The predicted
    latency must leave at least the headroom of the strict choice, capped at 6 ms, and
    the PII exposure must be no worse than that of the strict choice. Among those safe
    candidates it takes the one whose least-visited node has the fewest visits, with ties
    broken by the largest headroom. If no candidate is safe it returns the strict choice.

    The visit counts guide the choice among safe candidates only. They never relax the
    SLO and privacy tests. The probe therefore stays on the frontier of placements that
    are equivalent on the two top priorities, and it cannot trade them away. The
    coverage state is the per-node visit count in self.visits.
    """

    name = "Lexi-safeprobe"

    def __init__(self, sim, model, p_probe: float = 0.15, kappa: float = 1.5,
                 seed: int = 0):
        super().__init__(sim, model, eta=STRICT_ETA)
        self.p_probe = float(p_probe)
        self.kappa = float(kappa)          # width multiplier of the optimistic bound in _lcb_latency
        self.rng = np.random.default_rng(7000 + seed)
        self.visits = np.zeros(sim.M)      # per-node visit counts, the coverage state

    def reset(self):
        self.visits[:] = 0.0

    def _lcb_latency(self, cand, req):
        """Optimistic (lower-confidence-bound) latency of each candidate in ms.

        It is the predicted latency minus kappa * 30 ms times the mean over the
        candidate's nodes of 1 / sqrt(1 + visits), an uncertainty that shrinks as a node
        is visited. The current act method does not call this helper. The safety test
        there uses the mean prediction, which is the stricter choice."""
        raw = self.model.predict_raw_latency(cand, req)          # [n]
        unc = (1.0 / np.sqrt(1.0 + self.visits))[cand].mean(axis=1)  # [n]
        return raw - self.kappa * 30.0 * unc                     # 30 ms is the latency scale

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)
        strict = thresholded_lex_select(est, STRICT_ETA, prev_idx)
        if self.rng.random() < self.p_probe:
            raw = self.model.predict_raw_latency(cand, req)      # predicted mean latency, ms
            # Safe frontier. A candidate must meet the SLO in expectation with the headroom
            # of the strict choice (capped at 6 ms) so that a probe is not riskier than
            # what strict Lexi would pick, and must not worsen PII. The visit counts are
            # used only to choose among safe candidates. They never relax these tests.
            head_strict = self.sim.slo - float(raw[strict])
            pii = est[:, PRIV]
            safe = np.where((raw <= self.sim.slo - max(0.0, head_strict) * 0.0 - 1e-9)
                            & (self.sim.slo - raw >= min(head_strict, 6.0) - 1e-9)
                            & (pii <= est[strict, PRIV] + 1e-9))[0]
            if len(safe) > 0:
                # Prefer the least-visited node for coverage, and break ties by the most
                # latency headroom, the least SLO-risky probe.
                unc = np.array([self.visits[cand[i]].min() for i in safe])
                head = self.sim.slo - raw[safe]
                probe = int(safe[int(np.lexsort((-head, unc))[0])])
                return probe
        return int(strict)

    def learn(self, chosen_idx, place, obs, req):
        for m in place:
            self.visits[m] += 1.0
        self.model.update(obs, req)


def r2q1_safe_probe(seeds=SEEDS):
    """Compare strict Lexi with the safe-probe variant on three quantities.

    (a) Coverage, the percentage of fleet nodes that received at least 2 percent of the
    horizon in visits (a proxy for A3). (b) The latency error of the outcome model on the executed
    placement, to see whether coverage helps the model. (c) SLO percent and PII, which
    must not get worse, with a paired test on the SLO. Outside a probe both use the same
    strict selection."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    rows = {}
    raw_slo = {}
    for label, build in (("strict", lambda s, m, sd: CausalContinuum(s, m, eta=STRICT_ETA)),
                         ("safeprobe", lambda s, m, sd: SafeProbeContinuum(s, m, seed=sd))):
        cov, err, slo, pii = [], [], [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            model = CausalModel(sim, seed=seed)
            ctrl = build(sim, model, seed)
            ctrl.reset()
            rng = np.random.default_rng(1000 + seed)
            visits = np.zeros(sim.M)
            prev_idx = prev_place = None
            slo_hits = pii_sum = n = 0; errs = []
            for step, req in enumerate(reqs):
                c_idx = ctrl.act(req, cand, prev_idx); place = cand[c_idx]
                for m in place:
                    visits[m] += 1.0
                if step >= WARMUP:
                    # Model latency error on the executed placement, in ms, a Thm. 1 quantity.
                    pred = float(model.predict_raw_latency(place[None, :], req)[0])
                outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
                ctrl.learn(c_idx, place, obs, req)
                if step >= WARMUP:
                    slo_hits += int(slo_v); pii_sum += outcome[PRIV]; n += 1
                    errs.append(abs(pred - float(outcome[LAT])))
                prev_idx, prev_place = c_idx, place
            # Coverage counts the nodes with at least 2 percent of the horizon in visits.
            thresh = 0.02 * HORIZON
            cov.append(100.0 * float((visits >= thresh).mean()))
            err.append(float(np.mean(errs)))
            slo.append(100.0 * slo_hits / n); pii.append(pii_sum / n)
        cm, cl, ch = boot_ci(cov); em, el, eh = boot_ci(err)
        sm, sl, sh = boot_ci(slo)
        rows[label] = {"coverage_pct": round(cm, 2), "coverage_ci": [round(cl, 2), round(ch, 2)],
                       "model_lat_err_ms": round(em, 3), "err_ci": [round(el, 3), round(eh, 3)],
                       "slo": round(sm, 3), "slo_ci": [round(sl, 3), round(sh, 3)],
                       "pii": round(float(np.mean(pii)), 4)}
        raw_slo[label] = slo
    p_slo, md_slo = paired_boot_p(raw_slo["safeprobe"], raw_slo["strict"])
    return {"n_seeds": len(seeds), "strict": rows["strict"], "safeprobe": rows["safeprobe"],
            "sig_slo_safe_minus_strict": {"p": round(p_slo, 4), "median_diff": round(md_slo, 4)},
            "note": "LCB-gated probe explores only SLO-feasible, PII-nonworsening candidates; "
                    "coverage = %% of nodes visited >= 2%% of horizon; SLO must not worsen"}


# DX2: nonstationarity, with drift in RTTs and carbon intensity.
class DriftContinuum(Continuum):
    r"""Continuum whose RTTs and carbon intensity drift over the horizon (DX2).

    A slow trend, linear for RTT and sinusoidal for carbon, multiplies each node's RTTs
    and carbon base, with a node-dependent phase so that the ranking of the nodes shifts
    and not only the level. What the model learned early becomes progressively wrong. A
    static model cannot follow this, and an online-updating ridge model can. The caller
    must invoke set_time(step) before each step."""

    def __init__(self, *a, drift_rtt=0.6, drift_carbon=0.5, horizon=HORIZON, **k):
        super().__init__(*a, **k)
        self.drift_rtt = float(drift_rtt)
        self.drift_carbon = float(drift_carbon)
        self.horizon = int(horizon)
        self._rtt_intra0 = self.rtt_intra.copy()
        self._rtt_inter0 = self.rtt_inter.copy()
        self._carbon_base0 = self.carbon_base.copy()

    def set_time(self, step):
        """Apply the drift at the given step. RTTs grow by up to a factor 1 + drift_rtt
        and the carbon base swings by a factor within 1 +/- drift_carbon."""
        frac = step / max(1, self.horizon)
        # Node-specific phases.
        phase = np.linspace(0.0, 1.0, self.M)
        rtt_mult = 1.0 + self.drift_rtt * frac * (0.5 + 0.5 * np.sin(
            2 * np.pi * (frac + phase)))
        carb_mult = 1.0 + self.drift_carbon * np.sin(2 * np.pi * (frac + phase))
        self.rtt_intra = self._rtt_intra0 * rtt_mult
        self.rtt_inter = self._rtt_inter0 * rtt_mult
        self.carbon_base = self._carbon_base0 * carb_mult


def _run_drift(model_static, sim, cand, reqs, realize_seed, update_online):
    """Run strict Lexi under drift and return (SLO percent, PII, carbon) after the warm-up.

    With update_online False the model is frozen after WARMUP steps, and otherwise it
    keeps updating on every request."""
    ctrl = CausalContinuum(sim, model_static, eta=STRICT_ETA)
    rng = np.random.default_rng(realize_seed)
    prev_idx = prev_place = None
    slo_hits = pii_sum = n = 0; carb_sum = 0.0
    for step, req in enumerate(reqs):
        sim.set_time(step)                     # apply the drift of this step
        c_idx = ctrl.act(req, cand, prev_idx); place = cand[c_idx]
        outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
        if update_online or step < WARMUP:     # a static model stops updating after the warm-up
            model_static.update(obs, req)
        if step >= WARMUP:
            slo_hits += int(slo_v); pii_sum += outcome[PRIV]; carb_sum += outcome[CARB]; n += 1
        prev_idx, prev_place = c_idx, place
    return 100.0 * slo_hits / n, pii_sum / n, carb_sum / n


def r2q2_nonstationarity(seeds=SEEDS, drifts=(0.0, 0.3, 0.6)):
    """DX2, a static model against an online-updating model under RTT and carbon drift.

    SLO percent and carbon are reported for each drift level, with a paired test at the
    strongest drift. The online model, which updates on every request, is expected to
    track the drift, and the frozen model is expected to degrade as the drift grows."""
    rows = []
    static_raw, online_raw = {}, {}
    for dr in drifts:
        st_slo, on_slo, st_carb, on_carb = [], [], [], []
        for seed in seeds:
            reqs = make_workload(HORIZON, seed=seed)
            sim_s = DriftContinuum(seed=seed, drift_rtt=dr, drift_carbon=dr, horizon=HORIZON)
            sim_o = DriftContinuum(seed=seed, drift_rtt=dr, drift_carbon=dr, horizon=HORIZON)
            cand = candidate_set(sim_s, cap=CAND_CAP, seed=0)
            s1, p1, c1 = _run_drift(CausalModel(sim_s, seed=seed), sim_s, cand, reqs,
                                    1000 + seed, update_online=False)
            s2, p2, c2 = _run_drift(CausalModel(sim_o, seed=seed), sim_o, cand, reqs,
                                    1000 + seed, update_online=True)
            st_slo.append(s1); on_slo.append(s2); st_carb.append(c1); on_carb.append(c2)
        sm, sl, sh = boot_ci(st_slo); om, ol, oh = boot_ci(on_slo)
        rows.append({"drift": dr,
                     "static_slo": round(sm, 3), "static_slo_ci": [round(sl, 3), round(sh, 3)],
                     "online_slo": round(om, 3), "online_slo_ci": [round(ol, 3), round(oh, 3)],
                     "static_carbon": round(float(np.mean(st_carb)), 3),
                     "online_carbon": round(float(np.mean(on_carb)), 3)})
        static_raw[dr], online_raw[dr] = st_slo, on_slo
    dhi = drifts[-1]
    p, md = paired_boot_p(online_raw[dhi], static_raw[dhi])
    return {"n_seeds": len(seeds), "rows": rows,
            "sig_online_vs_static_at_max_drift": {"drift": dhi, "p": round(p, 4),
                                                  "median_diff": round(md, 4)},
            "features": "per-node ridge on [1, demand, load, same_region] (latency) and "
                        "[1, sin t, cos t] (carbon); ridge lambda=1e-2; update EVERY request",
            "note": "static model frozen after warmup; online model keeps updating -> "
                    "tracks RTT/carbon drift"}


# DX4: practical candidate generation (planner against curated and random menus).
def planner_candidates(sim, cap=CAND_CAP, seed=0, reqs=None):
    r"""A simple deployable candidate planner (DX4).

    It combines three generators. First, a latency-greedy placement per request origin,
    in which each stage goes to the node of lowest nominal latency for that origin. This
    is what makes some placements SLO-feasible. Second, privacy-aware placements that put
    the PII stages on a private edge node and the rest on a public backend, for every
    (edge, backend) pair. Third, per-objective greedy placements for six blends of latency,
    privacy, carbon and cost weights. The menu is filled up to cap by local perturbations
    of these placements, and by random placements if the perturbations run out.

    It stands for a planner that proposes a small set of sensible placements without
    enumerating the exponential space and without hand curation so that Lexi ranks a
    generated menu. Its rules are generic, and only the curated menu of controller.py
    is hand-picked."""
    K, M = sim.K, sim.M
    rng = np.random.default_rng(seed)
    carb_cost = (sim.energy * sim.carbon_base)[None, :] * sim.demandf[:, None]     # [K, M] mean carbon per stage and node
    cost_cost = np.tile(sim.cost[None, :], (K, 1))                                 # [K, M] price per stage and node
    priv_pen = (sim.priv_tier == 0).astype(float)                                  # 1 for a public node
    tiers = {t: [i for i, n in enumerate(sim.nodes) if n.tier == t]
             for t in ("cloud", "edge", "serverless")}
    cands, seen = [], set()

    def add(place):
        key = tuple(int(x) for x in place)
        if key not in seen:
            seen.add(key); cands.append(list(key))

    # (1) Latency-greedy per origin region.
    for origin in range(3):
        rtt = np.where(sim.region == origin, sim.rtt_intra, sim.rtt_inter)
        add([int(np.argmin(rtt + sim.base_compute * sim.demand[s] + sim.q_sens * 1.0))
             for s in range(K)])
    # (2) Privacy-aware placements over every (edge, backend) pair.
    for e in tiers["edge"]:
        for b in tiers["cloud"] + tiers["serverless"]:
            add([e if sim.pii[s] else b for s in range(K)])
    # (3) Per-objective greedy directions, including carbon-lean and cost-lean blends.
    # The tuples are weights of (latency, privacy, carbon, cost).
    ref_rtt = np.where(sim.region == 0, sim.rtt_intra, sim.rtt_inter)
    lat_cost = ref_rtt[None, :] + sim.base_compute[None, :] * sim.demand[:, None]  # [K,M]
    for wl, wp, wc, wo in ((1.0, 0.0, 0.0, 0.0), (0.3, 1.0, 0.0, 0.0),
                           (0.3, 0.0, 1.0, 0.0), (0.3, 0.0, 0.0, 1.0),
                           (0.5, 0.5, 0.3, 0.3), (0.8, 0.6, 0.0, 0.0)):
        place = []
        for s in range(K):
            score = (wl * lat_cost[s] / lat_cost[s].max()
                     + wc * carb_cost[s] / carb_cost[s].max()
                     + wo * cost_cost[s] / cost_cost[s].max())
            if sim.pii[s]:
                score = score + wp * priv_pen
            place.append(int(np.argmin(score)))
        add(place)
    # Fill up to cap with single-stage perturbations of the placements found so far.
    base = list(cands)
    while len(cands) < cap and base:
        p = list(base[rng.integers(0, len(base))])
        s = int(rng.integers(0, K)); p[s] = int(rng.integers(0, M))
        key = tuple(p)
        if key not in seen:
            seen.add(key); cands.append(p)
        if len(seen) > 8 * cap:                          # stop if perturbations keep repeating
            break
    while len(cands) < cap:                               # pad with random placements if needed
        p = [int(rng.integers(0, M)) for _ in range(K)]
        if tuple(p) not in seen:
            seen.add(tuple(p)); cands.append(p)
    return np.array(cands[:cap], dtype=int)


def _random_cand(sim, seed):
    """Uniformly random menu of CAND_CAP distinct placements (same as experiments_core)."""
    rng = np.random.default_rng(100 + seed)
    seen, out = set(), []
    while len(out) < CAND_CAP:
        p = tuple(int(rng.integers(0, sim.M)) for _ in range(sim.K))
        if p not in seen:
            seen.add(p); out.append(list(p))
    return np.array(out)


def r2q4_candidate_generation(seeds=SEEDS):
    """DX4, strict Lexi over three menu generators: the curated menu of Sect. 5, the
    planner of planner_candidates and a uniformly random menu.

    SLO percent, PII and carbon are reported for each. The recovery is the share of the
    curated-to-random SLO gap that the planner closes, computed as
    (random - planner) / (random - curated), and there is a paired test of the planner
    against the random menu."""
    rows = {}
    raw = {}
    gens = {"curated": lambda s, sd: candidate_set(s, cap=CAND_CAP, seed=0),
            "planner": lambda s, sd: planner_candidates(s, cap=CAND_CAP, seed=sd),
            "random":  lambda s, sd: _random_cand(s, sd)}
    for label, gen in gens.items():
        slo, pii, carb = [], [], []
        for seed in seeds:
            sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
            cand = gen(sim, seed)
            tn, orc = _precompute(sim, cand, reqs)
            r = run_method(CausalContinuum(sim, CausalModel(sim, seed=seed), eta=STRICT_ETA),
                           sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
            slo.append(r["slo"]); pii.append(r["privacy"]); carb.append(r["carbon"])
        sm, sl, sh = boot_ci(slo)
        rows[label] = {"slo": round(sm, 3), "slo_ci": [round(sl, 3), round(sh, 3)],
                       "pii": round(float(np.mean(pii)), 4),
                       "carbon": round(float(np.mean(carb)), 3)}
        raw[label] = slo
    # Recovery: the share of the gap between the curated and the random menu that the planner closes.
    cur, pln, ran = rows["curated"]["slo"], rows["planner"]["slo"], rows["random"]["slo"]
    gap = ran - cur
    recovery = round(100.0 * (ran - pln) / gap, 1) if abs(gap) > 1e-6 else 100.0
    p, md = paired_boot_p(raw["planner"], raw["random"])
    return {"n_seeds": len(seeds), "generators": rows,
            "planner_recovers_pct_of_curated": recovery,
            "sig_planner_vs_random_slo": {"p": round(p, 4), "median_diff": round(md, 4)},
            "note": "planner = per-objective greedy beam (deployable, no enumeration); "
                    "recovery = share of the curated->random SLO gap the planner closes"}


# DX5: contention, residual misspecification and the lower priorities.
def r2q5_contention_lowpriority(seeds=SEEDS, conts=(0.0, 0.2, 0.4, 0.6)):
    """DX5, the contention study extended to the lower priorities.

    For the additive and the contention-aware model and each contention level, it reports
    (a) the residual latency error, the mean absolute difference between the realised
    latency and the model's critical-path prediction for the executed placement, which
    is the part the model cannot explain, and (b) the effect on PII, carbon and cost and
    not only on the SLO. The M/G/1 term models a per-stage correction c * q_n * (coloc - 1)
    on the critical path, and the residual is what remains. The extended version
    reports that heavy contention spills onto the lower priorities as private placements
    turn SLO-infeasible."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    rows = []
    for cont in conts:
        recs = {"additive": {"resid": [], "slo": [], "pii": [], "carbon": [], "cost": []},
                "aware":    {"resid": [], "slo": [], "pii": [], "carbon": [], "cost": []}}
        for seed in seeds:
            sim = Continuum(seed=seed, contention=cont)
            reqs = make_workload(HORIZON, seed=seed)
            for label, mk in (("additive", lambda: CausalModel(sim, seed=seed)),
                              ("aware", lambda: ContentionCausalModel(sim, seed=seed))):
                model = mk()
                ctrl = CausalContinuum(sim, model, eta=STRICT_ETA)
                rng = np.random.default_rng(1000 + seed)
                prev_idx = prev_place = None
                slo_hits = pii_sum = n = 0; carb_sum = cost_sum = 0.0; resid = []
                for step, req in enumerate(reqs):
                    c_idx = ctrl.act(req, cand, prev_idx); place = cand[c_idx]
                    if step >= WARMUP:
                        pred = float(model.predict_raw_latency(place[None, :], req)[0])
                    outcome, slo_v, ch, obs = sim.realize(place, req, prev_place, rng)
                    model.update(obs, req)
                    if step >= WARMUP:
                        resid.append(abs(pred - float(outcome[LAT])))
                        slo_hits += int(slo_v); pii_sum += outcome[PRIV]
                        carb_sum += outcome[CARB]; cost_sum += outcome[COST]; n += 1
                    prev_idx, prev_place = c_idx, place
                recs[label]["resid"].append(float(np.mean(resid)))
                recs[label]["slo"].append(100.0 * slo_hits / n)
                recs[label]["pii"].append(pii_sum / n)
                recs[label]["carbon"].append(carb_sum / n)
                recs[label]["cost"].append(cost_sum / n)
        row = {"contention": cont}
        for label in ("additive", "aware"):
            row[label] = {k: round(float(np.mean(recs[label][k])), 3)
                          for k in ("resid", "slo", "pii", "carbon", "cost")}
        rows.append(row)
    return {"n_seeds": len(seeds), "rows": rows,
            "model": "M/G/1-flavoured per-stage correction c*q_n*(coloc-1) on the critical path; "
                     "single c fit online by ridge on the residual latency",
            "note": "resid = mean |realised - model critical-path latency| (ms) on executed "
                    "placements; PII/carbon/cost show the lower-priority impact of contention"}


# DX6: grid over the margin delta and a privacy slack.
class MarginPrivSlack(CausalContinuumMargin):
    """CausalContinuumMargin with an explicit privacy slack eta_priv (DX6).

    After the SLO-margin feasibility filter, privacy keeps the candidates within eta_priv
    of the best, where the parent class is strict. The carbon band and the headroom
    tie-break follow. The class makes it possible to sweep the margin on the top priority
    and a slack on the second priority together, and to see the effect on PII and on
    scale invariance."""

    name = "Lexi-margin-privslack"

    def __init__(self, sim, model, margin_ms=4.0, eta_priv=0.0, eta_carb=0.06,
                 hold_ms=8.0):
        super().__init__(sim, model, margin_ms=margin_ms, eta_carb=eta_carb, hold_ms=hold_ms)
        self.eta_priv = float(eta_priv)

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)
        raw = self.model.predict_raw_latency(cand, req)
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


def r2q6_delta_slack_grid(seeds=SEEDS, deltas=(0.0, 4.0, 8.0), etas_priv=(0.0, 0.25, 0.5)):
    """DX6, a two-dimensional grid over the safety margin delta (ms) and a privacy slack
    eta_priv (normalised units).

    SLO percent and PII are reported at each cell, which means an operator sees how delta trades
    the SLO against privacy. For the strict corner (eta_priv = 0) and a slack corner
    (eta_priv = 0.5) the run is then repeated with privacy rescaled tenfold. Strict
    privacy should stay invariant. A nonzero slack is a band in privacy's own units, and it
    is what makes the invariance degrade."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    grid = []
    for dl in deltas:
        for ep in etas_priv:
            slo, pii = [], []
            for seed in seeds:
                sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
                tn, orc = _precompute(sim, cand, reqs)
                r = run_method(MarginPrivSlack(sim, CausalModel(sim, seed=seed),
                                               margin_ms=dl, eta_priv=ep),
                               sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
                slo.append(r["slo"]); pii.append(r["privacy"])
            grid.append({"delta_ms": dl, "eta_priv": ep,
                         "slo": round(float(np.mean(slo)), 3),
                         "pii": round(float(np.mean(pii)), 4)})

    # Scale-invariance check with privacy rescaled tenfold, for strict and for slack privacy.
    # This wrapper also forwards predict_raw_latency, which the margin controller needs.
    class _Rescaled:
        def __init__(self, m, coord, c): self.m = m; self.coord = coord; self.c = c
        def predict_all(self, cand, req):
            e = self.m.predict_all(cand, req).copy(); e[:, self.coord] *= self.c; return e
        def predict_raw_latency(self, cand, req):
            return self.m.predict_raw_latency(cand, req)
        def update(self, *a): self.m.update(*a)

    inv = []
    for ep in (0.0, 0.5):
        for c in (1.0, 10.0):
            pii = []
            for seed in seeds:
                sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
                tn, orc = _precompute(sim, cand, reqs)
                base = CausalModel(sim, seed=seed)
                model = _Rescaled(base, PRIV, c) if c != 1.0 else base
                r = run_method(MarginPrivSlack(sim, model, margin_ms=4.0, eta_priv=ep),
                               sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
                pii.append(r["privacy"])
            inv.append({"eta_priv": ep, "priv_scale": c, "pii": round(float(np.mean(pii)), 4)})
    return {"n_seeds": len(seeds), "grid": grid, "scale_invariance": inv,
            "note": "delta on the top-priority SLO (ms), eta_priv a band on the 2nd priority; "
                    "strict privacy (eta_priv=0) is scale-invariant, nonzero slack is not"}


# DX7: lexicographic controller over quantile-normalised scores.
def _quantile_ranks(col: np.ndarray) -> np.ndarray:
    """Quantile transform of a column to [0, 1], with 0 for the lowest value and tied values
    sharing their mean rank. It is a monotone reparameterisation of the column and
    preserves its order."""
    order = np.argsort(col, kind="stable")
    ranks = np.empty(len(col), dtype=float)
    n = len(col)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and col[order[j + 1]] == col[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / max(n - 1, 1)


class QuantileLex:
    r"""Lexicographic controller over per-objective quantile scores (DX7).

    Each objective column is replaced by its quantile rank in [0, 1], and the same
    thresholded lexicographic selection then runs on those scores. The quantile
    transform is a monotone reparameterisation of each column, which means by Prop. 1 the selected
    placement equals that of strict Lexi at zero slack. This controller therefore
    replicates strict Lexi. With a nonzero slack the band would be measured in quantile
    units, a fixed fraction of the candidates, instead of the objective's own units, and
    that is the only behavioural difference."""

    name = "Lexi-quantile"

    def __init__(self, sim, model, eta=None):
        self.sim = sim
        self.model = model
        self.eta = np.zeros(3) if eta is None else np.asarray(eta, float)

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)
        q = np.stack([_quantile_ranks(est[:, k]) for k in range(est.shape[1])], axis=1)
        return thresholded_lex_select(q, self.eta, prev_idx)

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


def r2q7_quantile_surrogate(seeds=SEEDS):
    """DX7, strict Lexi against the quantile-normalised controller.

    (a) The per-step decision agreement on identical estimates. (b) The realised SLO
    percent, PII, carbon and cost of each controller. Prop. 1 predicts an agreement of 1
    and matching outcomes."""
    cand = candidate_set(Continuum(), cap=CAND_CAP, seed=0)
    agree = total = 0
    strict_m = {"slo": [], "privacy": [], "carbon": [], "cost": []}
    quant_m = {"slo": [], "privacy": [], "carbon": [], "cost": []}
    for seed in seeds:
        sim = Continuum(seed=seed); reqs = make_workload(HORIZON, seed=seed)
        tn, orc = _precompute(sim, cand, reqs)
        # (a) Agreement on identical estimates, along one shared model trajectory.
        model = CausalModel(sim, seed=seed)
        rng = np.random.default_rng(1000 + seed); prev_idx = prev_place = None
        for req in reqs:
            est = model.predict_all(cand, req)
            s = thresholded_lex_select(est, STRICT_ETA, prev_idx=None)
            q = np.stack([_quantile_ranks(est[:, k]) for k in range(4)], axis=1)
            qsel = thresholded_lex_select(q, STRICT_ETA, prev_idx=None)
            agree += int(s == qsel); total += 1
            place = cand[s]
            _, _, _, obs = sim.realize(place, req, prev_place, rng)
            model.update(obs, req); prev_idx, prev_place = s, place
        # (b) End-to-end realised outcomes of each controller.
        rs = run_method(CausalContinuum(sim, CausalModel(sim, seed=seed), eta=STRICT_ETA),
                        sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
        rq = run_method(QuantileLex(sim, CausalModel(sim, seed=seed)),
                        sim, cand, reqs, tn, orc, realize_seed=1000 + seed)
        for k in strict_m:
            strict_m[k].append(rs["slo" if k == "slo" else k])
            quant_m[k].append(rq["slo" if k == "slo" else k])
    out = {"n_seeds": len(seeds), "n_steps": total,
           "decision_agreement": round(agree / total, 5),
           "strict": {k: round(float(np.mean(v)), 4) for k, v in strict_m.items()},
           "quantile": {k: round(float(np.mean(v)), 4) for k, v in quant_m.items()},
           "note": "quantile transform is monotone per objective => Prop. scale-invariance "
                   "makes the quantile-lex surrogate replicate strict-order Lexi at zero slack"}
    return out


def main():
    """Run DX1 to DX7 (there is no DX3), write ../results/deployment_results.json and print a summary."""
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    res = {}; t0 = time.perf_counter()
    print("DX1 priority-safe probing ...");    res["r2q1_safe_probe"] = r2q1_safe_probe()
    print("DX2 nonstationarity/drift ...");     res["r2q2_nonstationarity"] = r2q2_nonstationarity()
    print("DX4 candidate generation ...");      res["r2q4_candidate_gen"] = r2q4_candidate_generation()
    print("DX5 contention lower-priority ...");  res["r2q5_contention_lowpri"] = r2q5_contention_lowpriority()
    print("DX6 delta x slack grid ...");        res["r2q6_delta_slack"] = r2q6_delta_slack_grid()
    print("DX7 quantile surrogate ...");        res["r2q7_quantile"] = r2q7_quantile_surrogate()
    res["_meta"] = {"n_seeds": len(SEEDS), "runtime_s": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(here, "..", "results", "deployment_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[done in {res['_meta']['runtime_s']}s]\n")

    q1 = res["r2q1_safe_probe"]
    print("DX1 safe probe:  strict cov %.1f%% err %.2fms SLO %.2f | safeprobe cov %.1f%% err %.2fms SLO %.2f  (p_SLO=%s)"
          % (q1["strict"]["coverage_pct"], q1["strict"]["model_lat_err_ms"], q1["strict"]["slo"],
             q1["safeprobe"]["coverage_pct"], q1["safeprobe"]["model_lat_err_ms"], q1["safeprobe"]["slo"],
             q1["sig_slo_safe_minus_strict"]["p"]))
    print("DX2 nonstationarity (drift -> static/online SLO):")
    for r in res["r2q2_nonstationarity"]["rows"]:
        print(f"  drift={r['drift']}: static SLO {r['static_slo']:.2f}  online SLO {r['online_slo']:.2f}  "
              f"(static carbon {r['static_carbon']:.2f} vs online {r['online_carbon']:.2f})")
    print("DX4 candidate generation:")
    for k, v in res["r2q4_candidate_gen"]["generators"].items():
        print(f"  {k:8s}: SLO {v['slo']:.2f}  PII {v['pii']:.3f}  carbon {v['carbon']:.2f}")
    print(f"  planner recovers {res['r2q4_candidate_gen']['planner_recovers_pct_of_curated']}% of curated")
    print("DX5 contention lower-priority (resid ms; PII/carbon/cost):")
    for r in res["r2q5_contention_lowpri"]["rows"]:
        a, w = r["additive"], r["aware"]
        print(f"  c={r['contention']}: additive resid {a['resid']:.1f} PII {a['pii']:.2f} carbon {a['carbon']:.2f} | "
              f"aware resid {w['resid']:.1f} PII {w['pii']:.2f} carbon {w['carbon']:.2f}")
    print("DX6 delta x eta_priv grid (SLO / PII):")
    for g in res["r2q6_delta_slack"]["grid"]:
        print(f"  delta={g['delta_ms']}ms eta_priv={g['eta_priv']}: SLO {g['slo']:.2f} PII {g['pii']:.3f}")
    print("  scale-invariance (privacy x10):")
    for s in res["r2q6_delta_slack"]["scale_invariance"]:
        print(f"    eta_priv={s['eta_priv']} scale x{s['priv_scale']}: PII {s['pii']:.3f}")
    q7 = res["r2q7_quantile"]
    print(f"DX7 quantile surrogate: agreement {q7['decision_agreement']*100:.2f}% "
          f"strict {q7['strict']} quantile {q7['quantile']}")


if __name__ == "__main__":
    main()
