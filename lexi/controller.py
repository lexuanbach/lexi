"""
The Lexi selector: thresholded lexicographic selection over the predicted objective
matrix (Def. 2, "Priority order and selection rule", Sect. 2), the oracle used to
measure it, and the controller variants of the extended version.

The directory keeps its earlier name, causalcontinuum, which means the controller classes are
still called CausalContinuum*. They are the framework that the paper calls Lexi.

Selection rule. The objectives are ordered SLO hinge, privacy, carbon, cost. Given the
[n_cand, 4] normalised estimate matrix from causal.py, for k = 0, 1, 2 in turn the
candidates within slack eta_k of the current best on objective k are kept, and the
survivors are tie-broken on cost. With eta = 0 this is the strict lexicographic optimum
and it reads only the order of each column, which is why Prop. 1 (scale invariance)
holds for it. Objective k is consulted only on the set that objectives 0 to k - 1 did
not separate, which means a lower priority cannot cause a higher-priority regression.

Stability. The controllers also implement a hysteresis rule. If the last executed
placement is still among the survivors of the thresholds, it is kept and the cost
tie-break is skipped. This trades a little of the last objective for fewer migrations
(the churn column of Table 3). The camera-ready evaluates the strict rule with the
carbon slack disabled. In the extended version the carbon slack and the hysteresis are
studied in RQ5 and controlled by DEFAULT_ETA and the auto-slack variant.

Oracle. oracle_index applies the same rule with eta = 0 to the true, noise-free
normalised outcomes. It defines the priority optimum pi*_t of Thm. 1, and it is the
reference for the priority-inversion rate and for the "selects the true priority
optimum" result of RQ4.

Cost per decision is O(n_cand * K) for prediction plus O(n_cand) for selection (Prop.
"Efficient decisions" of the extended version). All functions are deterministic.
"""
from __future__ import annotations
import numpy as np
from sim import Continuum, Request, PRIV, CARB, COST

# Geometric priority weights on normalised objectives in [0, 1]. With base 8 a unit gap
# on objective k outweighs the largest combined gap of all lower objectives. The weights
# give a scalar aggregate that respects the priority order and stays numerically well
# scaled, which is what the cumulative-regret plots need. Selection itself never uses them.
LEX_W = np.array([1.0, 1.0 / 8, 1.0 / 64, 1.0 / 512])

# Default slack thresholds in normalised units, one per thresholded level (latency,
# privacy, carbon). Slack sits only on the lower priorities. The SLO hinge and privacy
# stay strict at 0, and a small carbon band of 0.08 buys placement stability through the
# hysteresis rule. Slack on the SLO is avoided on purpose, because on a threshold
# objective it converts directly into SLO violations by admitting overrunning placements
# into the privacy level. The SLO is protected by an explicit safety margin instead
# (CausalContinuumMargin). The strict rule of the camera-ready sets all three to 0.
DEFAULT_ETA = np.array([0.0, 0.0, 0.08])    # [latency, privacy, carbon]


def lex_scalar(norm_scores: np.ndarray) -> np.ndarray:
    """Scalar aggregate of normalised objective vectors under the geometric weights
    LEX_W. It is used to plot convergence curves and never to select a placement."""
    return norm_scores @ LEX_W


def oracle_index(true_norm: np.ndarray) -> int:
    """Index of the strict-lexicographic optimum over the candidates, on true outcomes.

    true_norm is the normalised noise-free matrix from Continuum.expected_all. The
    choice comes from thresholded_lex_select with eta = 0 and no previous placement, the
    same rule the controller applies to its estimates, and not from the geometric
    scalar. Regret therefore measures estimation error, the distance between the
    controller's choice and the priority optimum pi*_t of Thm. 1, and does not depend on
    any scalarisation."""
    return thresholded_lex_select(true_norm, np.zeros(3), prev_idx=None)


def thresholded_lex_select(norm_scores: np.ndarray, eta: np.ndarray,
                           prev_idx: int | None) -> int:
    """Thresholded lexicographic selection (Def. 2) over normalised candidate scores.

    norm_scores is [n_cand, 4] with lower better. eta holds the slack of the first three
    objectives (latency hinge, privacy, carbon). prev_idx is the last executed
    candidate for the hysteresis rule and may be None. Returns the chosen row index.
    Ties inside the tolerance 1e-12 count as equal, which means a plateau such as the all-zero
    hinge of the SLO-feasible set is kept whole for the next level.
    """
    idx = np.arange(norm_scores.shape[0])
    for k in range(3):                       # levels 0 to 2: hinge, privacy, carbon
        best = norm_scores[idx, k].min()
        idx = idx[norm_scores[idx, k] <= best + eta[k] + 1e-12]
    # Hysteresis: keep the previous placement if it survived every threshold.
    if prev_idx is not None and prev_idx in idx:
        return int(prev_idx)
    # Otherwise break the remaining tie on cost, the last objective.
    return int(idx[np.argmin(norm_scores[idx, COST])])


def candidate_set(sim: Continuum, cap: int = 40, seed: int = 0) -> np.ndarray:
    """Build the fixed candidate menu (int array [n_cand, K], node index per stage).

    This is the 40-candidate menu of Sect. 5. Structured placements come first. They
    are the homogeneous placements (all stages on one node), one latency-greedy
    placement per origin region, the twelve privacy-aware placements that put the PII
    stages on one edge node and the rest on one public backend for every (edge,
    backend) pair, and the all-cheapest placement. After deduplication, seeded random
    placements fill the menu up to cap. In config A this gives 22 structured and 18
    random candidates. The privacy-aware placements guarantee that the privacy objective
    always has something to choose, which means a privacy failure is attributable to the rule.
    The menu does not change over steps, which means the tabular baselines can learn over it and
    the oracle is defined over the same action set. RQ4 varies how the menu is built
    (a uniformly random menu against this curated one).
    """
    K, M = sim.K, sim.M
    nodes = sim.nodes
    idx_by = {n.name: i for i, n in enumerate(nodes)}
    tiers = {t: [i for i, n in enumerate(nodes) if n.tier == t]
             for t in ("cloud", "edge", "serverless")}

    structured = []
    # Homogeneous placements, one per node.
    for m in range(M):
        structured.append([m] * K)
    # Latency-greedy placement for each origin: per stage, the node of least nominal latency.
    for origin in range(3):
        rtt = np.where(sim.region == origin, sim.rtt_intra, sim.rtt_inter)
        greedy = [int(np.argmin(rtt + sim.base_compute * sim.demand[s]
                                + sim.q_sens * 1.0)) for s in range(K)]
        structured.append(greedy)
    # Privacy-aware placements: PII stages on a private edge node, the rest on one public
    # backend. Enumerating every (edge, backend) pair means that for any request origin
    # there is a region-matched private option that can still meet the SLO.
    backends = tiers["cloud"] + tiers["serverless"]
    for e in tiers["edge"]:
        for b in backends:
            structured.append([e if sim.pii[s] else b for s in range(K)])
    # Cost-minimal placement: everything on the cheapest node.
    cheapest = int(np.argmin(sim.cost))
    structured.append([cheapest] * K)

    seen, cands = set(), []
    for p in structured:
        key = tuple(p)
        if key not in seen:
            seen.add(key); cands.append(list(p))
    rng = np.random.default_rng(seed)
    while len(cands) < cap:
        p = [int(rng.integers(0, M)) for _ in range(K)]
        key = tuple(p)
        if key not in seen:
            seen.add(key); cands.append(p)
    return np.array(cands[:cap], dtype=int)


def thresholded_lex_select_order(norm_scores: np.ndarray, order: tuple,
                                 eta_map: dict | None = None,
                                 prev_idx: int | None = None) -> int:
    """Thresholded lexicographic selection under an arbitrary priority order
    (extended version, RQ6/RQ7 study of alternative and dynamic orders).

    order is a permutation of (0, 1, 2, 3), most important objective first. The first
    three levels are thresholded and the last is the tie-break. eta_map maps an
    objective index to its slack and defaults to 0, which means the rule is strict unless told
    otherwise. Selection reads only the order of each column, which means Prop. 1 holds for
    every priority order and not only the default one.
    """
    eta_map = eta_map or {}
    idx = np.arange(norm_scores.shape[0])
    for k in order[:-1]:
        best = norm_scores[idx, k].min()
        idx = idx[norm_scores[idx, k] <= best + eta_map.get(k, 0.0) + 1e-12]
    last = order[-1]
    if prev_idx is not None and prev_idx in idx:
        return int(prev_idx)
    return int(idx[np.argmin(norm_scores[idx, last])])


class CausalContinuumOrder:
    """Lexi under an operator-specified priority order (extended version).

    The controller equals CausalContinuum except that the order is a constructor
    argument, which means a workload can rank, for example, privacy before the SLO. The outcome
    model never sees the order, hence changing it needs no retraining. set_order swaps it
    in O(1) at run time."""

    name = "Lexi-order"

    def __init__(self, sim: Continuum, model, order=(0, 1, 2, 3),
                 eta_map: dict | None = None):
        self.sim = sim
        self.model = model
        self.order = tuple(order)
        self.eta_map = eta_map or {}

    def reset(self):
        pass

    def set_order(self, order):
        """Change the priority order for all later decisions. No state is rebuilt."""
        self.order = tuple(order)

    def act(self, req: Request, cand: np.ndarray, prev_idx: int | None) -> int:
        est = self.model.predict_all(cand, req)
        return thresholded_lex_select_order(est, self.order, self.eta_map, prev_idx)

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class CausalContinuum:
    """The Lexi controller: counterfactual estimates from the outcome model followed by
    thresholded lexicographic selection with hysteresis.

    Each step calls act to choose a candidate index from the menu and learn to feed the
    executed placement's observations back to the model. With eta = 0 it is the strict
    rule evaluated in the camera-ready. The default eta is DEFAULT_ETA.
    """

    name = "CausalContinuum"

    def __init__(self, sim: Continuum, model, eta: np.ndarray = DEFAULT_ETA):
        self.sim = sim
        self.model = model                      # a CausalModel or one of its subclasses
        self.eta = np.asarray(eta, dtype=float)

    def reset(self):
        pass

    def act(self, req: Request, cand: np.ndarray, prev_idx: int | None) -> int:
        est = self.model.predict_all(cand, req)          # the interface matrix of Fig. 1
        return thresholded_lex_select(est, self.eta, prev_idx)

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class CausalContinuumAutoSlack:
    """Lexi with an automatic, unit-free carbon slack (extended version, RQ5).

    The carbon slack is set to a fraction frac of that objective's interquartile range
    over the current candidate estimates, eta_2 = frac * (Q75 - Q25). The SLO hinge
    (eta_0) and privacy (eta_1) stay strict, as in DEFAULT_ETA, which means neither the SLO nor
    privacy is traded for stability. The interquartile range rescales with the
    objective's units. The surviving carbon band is therefore unchanged by any positive
    rescaling of that objective, and the controller keeps the scale invariance of strict
    selection while exposing a single unit-free stability knob. A larger frac widens the
    carbon band and lowers churn."""

    name = "Lexi-auto"

    def __init__(self, sim: Continuum, model, frac: float = 0.5):
        self.sim = sim
        self.model = model
        self.frac = float(frac)

    def reset(self):
        pass

    def _auto_eta(self, est: np.ndarray) -> np.ndarray:
        """Interquartile-fraction slack for the three thresholded levels. Only the
        carbon entry is nonzero. The first two are reset to 0 below."""
        q75 = np.quantile(est[:, :3], 0.75, axis=0)
        q25 = np.quantile(est[:, :3], 0.25, axis=0)
        eta = self.frac * (q75 - q25)
        eta[0] = 0.0                      # the hinge stays strict
        eta[1] = 0.0                      # privacy stays strict
        return eta

    def act(self, req: Request, cand: np.ndarray, prev_idx: int | None) -> int:
        est = self.model.predict_all(cand, req)
        return thresholded_lex_select(est, self._auto_eta(est), prev_idx)

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class CausalContinuumMargin:
    """Lexi with an SLO safety margin on the top priority (extended version, RQ5).

    Strict selection treats the SLO as a threshold on the expected latency. A placement
    predicted to just meet it enters the privacy level even though realised queueing
    noise often pushes it over, which is where the strict rule's residual SLO violations
    come from (Table 3). The margin variant keeps the placements whose predicted latency is at
    most SLO - delta. If none qualifies it relaxes to plain feasibility (latency at most
    the SLO), and then to the least-overrunning candidate. Privacy is applied strictly,
    carbon takes a small band, and the final tie-break maximises latency headroom, which
    picks the least risky of the priority-equivalent placements. With delta = 0 this is
    strict feasibility. A positive delta reserves headroom for the noise and lowers
    realised violations at a small privacy cost. That is the second priority being
    spent to protect the first, as the order dictates.

    delta is one knob in milliseconds on a threshold objective. It does not touch the
    order-only treatment of privacy, carbon and cost, which means their scale invariance
    survives. hold_ms bounds how much headroom the hysteresis rule may give up to keep
    the previous placement."""

    name = "Lexi-margin"

    def __init__(self, sim: Continuum, model, margin_ms: float = 4.0,
                 eta_carb: float = 0.06, hold_ms: float = 8.0):
        self.sim = sim
        self.model = model
        self.margin = float(margin_ms)
        self.eta_carb = float(eta_carb)
        self.hold_ms = float(hold_ms)

    def reset(self):
        pass

    def act(self, req: Request, cand: np.ndarray, prev_idx: int | None) -> int:
        est = self.model.predict_all(cand, req)              # normalised matrix (privacy and carbon columns)
        raw = self.model.predict_raw_latency(cand, req)      # predicted latency in ms, for headroom
        idx = np.where(raw <= self.sim.slo - self.margin)[0]
        if len(idx) == 0:
            idx = np.where(raw <= self.sim.slo)[0]           # relax to plain feasibility
        if len(idx) == 0:
            return int(np.argmin(raw))                       # least overrunning candidate
        for k, eta in ((PRIV, 0.0), (CARB, self.eta_carb)):  # strict privacy, then the carbon band
            best = est[idx, k].min()
            idx = idx[est[idx, k] <= best + eta + 1e-12]
        headroom = self.sim.slo - raw[idx]
        roomiest = idx[int(np.argmax(headroom))]
        if prev_idx is not None and prev_idx in idx:         # hysteresis, unless it costs too much headroom
            prev_room = self.sim.slo - raw[prev_idx]
            if headroom.max() - prev_room <= self.hold_ms:
                return int(prev_idx)
        return int(roomiest)

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)
