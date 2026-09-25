"""
Baseline decision rules that Lexi is compared against (the "Baselines" paragraph of
Sect. 5, and the results of Sect. 6.1, 6.2 and 6.3).

The directory keeps its earlier name, causalcontinuum. Every rule follows the same
controller interface as the Lexi controller in controller.py:
    reset(), act(req, cand, prev_idx) -> index into the shared candidate menu, and
    learn(chosen_idx, place, obs, req).
All rules choose from the same fixed 40-candidate menu, which means any difference in outcome
comes from the decision rule. The rules that use predictions share one outcome model
(causal.py) with Lexi, which means they see the same [n_cand, 4] estimate matrix.

Rules of the camera-ready comparison (Table 3, Fig. 2):
    WeightedSum          "WS-default" with DEFAULT_WS_W and "WS-tuned" with TUNED_WS_W.
                         The fixed-weight scalariser that Prop. 1 shows is not
                         scale invariant.
    RankWeightedSum,     the three scale-robust scalarisers of Sect. 5 (rank, min-max
    NormWeightedSum,     and augmented Tchebycheff). They reuse the WS-tuned weights.
    Tchebycheff          The first two are invariant to rescaling and still price the
                         objectives, which means they can outvote the top priority. Tchebycheff
                         is only approximately robust to a unit change.
    LexicographicLP      exact discrete lexicographic optimisation with no slack. On
                         identical estimates it returns the same placement as strict
                         Lexi, and it is included to make that equivalence measurable.
    ConstrainedWeightedSum  "Constr-WS", the SLO-filtered weighted sum of Sect. 6.2.
Rules of the extended version (RQ2 table, discussion):
    StaticGreedy, BinPack fixed heuristic placements that ignore the drifting workload.
    SingleObjRL          a latency-only epsilon-greedy contextual bandit with a tabular
                         estimate per (context, placement).

Everything is deterministic given the seeds. Cost per decision is O(n_cand) on top of
the prediction, except for the rank version which sorts each objective column.
"""
from __future__ import annotations
import numpy as np
from sim import Continuum, LAT


def _fixed_index(cand: np.ndarray, place: list) -> int:
    """Index of a target placement in the candidate menu. If it is missing, return the
    candidate that differs from it in the fewest stages."""
    key = tuple(place)
    for i, c in enumerate(cand):
        if tuple(c.tolist()) == key:
            return i
    # The placement is missing from the menu. Take the nearest candidate by Hamming distance.
    diffs = (cand != np.array(place)).sum(axis=1)
    return int(np.argmin(diffs))


class StaticGreedy:
    """A fixed latency-greedy placement. Each stage goes to the node of lowest nominal
    latency for a reference origin (region 0). It ignores every other objective and
    never adapts to the drifting origin and load."""
    name = "static_greedy"

    def __init__(self, sim: Continuum):
        # Per stage, the node of lowest nominal latency for a request from region 0.
        ref_rtt = np.where(sim.region == 0, sim.rtt_intra, sim.rtt_inter)
        self.place = [int(np.argmin(ref_rtt + sim.base_compute * sim.demand[s]
                                    + sim.q_sens * 1.0)) for s in range(sim.K)]
        self._idx = None

    def reset(self):
        self._idx = None

    def act(self, req, cand, prev_idx):
        if self._idx is None:
            self._idx = _fixed_index(cand, self.place)
        return self._idx

    def learn(self, *a):
        pass


class BinPack:
    """A fixed cost-driven placement, every stage on the cheapest node. It has good cost
    and poor latency and privacy."""
    name = "binpack"

    def __init__(self, sim: Continuum):
        cheapest = int(np.argmin(sim.cost))
        self.place = [cheapest] * sim.K
        self._idx = None

    def reset(self):
        self._idx = None

    def act(self, req, cand, prev_idx):
        if self._idx is None:
            self._idx = _fixed_index(cand, self.place)
        return self._idx

    def learn(self, *a):
        pass


class SingleObjRL:
    """Latency-only epsilon-greedy contextual bandit with a tabular on-policy update.

    The context is (origin region, load bucket). The estimate is a running mean of the
    realised critical-path latency per (context, candidate), with no generalisation
    across candidates. It converges slowly and is blind to privacy, carbon and cost.
    Despite its class name it is a bandit and not a reinforcement-learning system."""
    name = "single_obj_rl"

    def __init__(self, sim: Continuum, n_cand: int, eps: float = 0.10, seed: int = 0):
        self.sim = sim
        self.n_cand = n_cand
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        # Table of latency estimates with shape [3 origins, 3 load buckets, n_cand].
        # The initial value of 50 ms is far below any real latency. Unexplored
        # placements therefore look attractive and get tried (optimistic initialisation).
        self.q = np.full((3, 3, n_cand), 50.0)
        self.n = np.zeros((3, 3, n_cand))

    def reset(self):
        self.q[:] = 50.0
        self.n[:] = 0.0

    def act(self, req, cand, prev_idx):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_cand))
        return int(np.argmin(self.q[req.origin, req.load_bucket]))

    def learn(self, chosen_idx, place, obs, req):
        # The realised end-to-end latency is the critical path over the stage latencies in obs.
        stage_lat = np.zeros(self.sim.K)
        for (s, m, lat, carb, cost) in obs:
            stage_lat[s] = lat
        longest = np.zeros(self.sim.K)
        for i in range(self.sim.K):
            base = max((longest[p] for p in self.sim.preds[i]), default=0.0)
            longest[i] = stage_lat[i] + base
        latency = float(longest.max())
        o, l = req.origin, req.load_bucket
        self.n[o, l, chosen_idx] += 1
        a = 1.0 / self.n[o, l, chosen_idx]
        self.q[o, l, chosen_idx] += a * (latency - self.q[o, l, chosen_idx])


# WS-default: an efficiency-leaning weighting that an operator who has not tuned to the
# priority order might pick. It weighs the SLO but will trade it, and privacy, for carbon
# and cost savings. The weight sweep of RQ1 shows that a calibrated weighting can match
# Lexi. Lexi's advantage is that it needs no per-deployment tuning and is scale
# invariant.
DEFAULT_WS_W = np.array([0.30, 0.12, 0.30, 0.28])
# WS-tuned: a hand-calibrated, SLO-heavy weighting whose strongly separated weights
# approximate the lexicographic order. It is not a point of the 84-weighting sweep grid
# (strictly positive weights in steps of 0.1) and it beats the best point of that sweep
# on SLO. Such weights have to be known in advance and they depend on the units.
TUNED_WS_W = np.array([0.85, 0.10, 0.00, 0.05])


class WeightedSum:
    """Fixed-weight linear scalarisation of the model estimates, argmin of est @ W.

    W is a constructor argument, which means one class serves WS-default (the default weights)
    and WS-tuned (TUNED_WS_W). The score depends on the units. Rescaling one objective
    can change the argmin, which is the failure that Prop. 1 states and that RQ3
    measures."""
    name = "weighted_sum"

    def __init__(self, sim: Continuum, model, W=None):
        self.sim = sim
        self.model = model
        self.W = np.asarray(DEFAULT_WS_W if W is None else W, dtype=float)

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)      # normalised estimates
        return int(np.argmin(est @ self.W))

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


def _avg_ranks(col: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks of a 1-D array, with 0 for the lowest value.

    Tied values share the mean of the ranks they span, which means equal objective values are
    not ordered arbitrarily. The result depends only on the order of the values and is
    therefore unchanged by any strictly increasing rescaling of the column.
    """
    order = np.argsort(col, kind="stable")
    sorted_vals = col[order]
    ranks = np.empty(len(col), dtype=float)
    i, n = 0, len(col)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


class RankWeightedSum:
    """Weighted sum over per-objective ranks, one of the scale-robust scalarisers of Sect. 5.

    Each objective column is replaced by the tie-aware rank of each candidate, the ranks
    are combined with the TUNED_WS_W weights, and the argmin is taken. Ranks depend only
    on the order of each column, which means the rule is invariant to monotone rescaling, as
    strict Lexi is. It does not honour the strict priority, however. It still prices the
    ranks, and a large rank advantage on a lower objective can outvote a small rank
    deficit on the top one. This is the reason for its higher SLO violations and
    priority-inversion rate in Sect. 6.1 and 6.2.
    """
    name = "rank_ws"

    def __init__(self, sim: Continuum, model, W=None):
        self.sim = sim
        self.model = model
        self.W = np.asarray(TUNED_WS_W if W is None else W, dtype=float)

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)          # [n, 4] normalised estimates
        ranks = np.stack([_avg_ranks(est[:, k]) for k in range(est.shape[1])], axis=1)
        return int(np.argmin(ranks @ self.W))

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class NormWeightedSum:
    """Min-max normalised weighted sum, a scale-robust scalariser of Sect. 5.

    Each objective column is rescaled to [0, 1] over the candidates as (x - min) /
    (max - min), and then combined with the TUNED_WS_W weights. Min-max normalisation
    is unchanged by a positive affine rescaling. Under the linear unit swings of the
    scale study (RQ3) the rule is therefore also invariant. Like the rank version it still prices
    the normalised objectives, which means it can trade the top priority away."""
    name = "norm_ws"

    def __init__(self, sim: Continuum, model, W=None):
        self.sim = sim
        self.model = model
        self.W = np.asarray(TUNED_WS_W if W is None else W, dtype=float)

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)          # [n, 4] normalised estimates
        lo = est.min(axis=0)
        hi = est.max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)   # a constant column would divide by zero
        norm = (est - lo) / span
        return int(np.argmin(norm @ self.W))

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class Tchebycheff:
    """Augmented weighted Chebyshev (Tchebycheff) scalarisation, a scale-robust
    scalariser of Sect. 5 (Miettinen, Nonlinear Multiobjective Optimization).

    z* is the ideal point, the column minimum over the candidates. With weights w (the
    TUNED_WS_W vector renormalised to sum 1) the score of candidate i is
        max_k w_k (est_ik - z*_k) + rho * sum_k w_k (est_ik - z*_k),   rho = 1e-3,
    and the argmin is chosen. The reference point makes the rule robust to a shift of
    an objective. The weighted deviations are not normalised by the range, however.
    Multiplying one objective's units by c multiplies its Chebyshev term by c. The
    rule is therefore not strictly scale invariant, and the scale study of RQ3 measures this."""
    name = "tcheby"

    RHO = 1e-3

    def __init__(self, sim: Continuum, model, W=None):
        self.sim = sim
        self.model = model
        w = np.asarray(TUNED_WS_W if W is None else W, dtype=float)
        self.w = w / w.sum()

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)          # [n, 4] normalised estimates
        z = est.min(axis=0)                              # ideal point
        d = self.w * (est - z)                           # weighted deviations, [n, 4]
        score = d.max(axis=1) + self.RHO * d.sum(axis=1)
        return int(np.argmin(score))

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class LexicographicLP:
    """Exact discrete lexicographic optimiser ("Lex-LP" in Sect. 5).

    It minimises the SLO hinge, then privacy, then carbon, then cost over the model
    estimates. Each level keeps exactly the argmin set of the previous one, with no
    slack and no hysteresis. On identical estimates it returns a placement that ties
    with strict Lexi on every level, which means Lexi is an online, linear-time realisation of
    hierarchical lexicographic optimisation and no LP solver is needed. The class is here
    to make that equivalence measurable and is not an independent competitor. Ties on the
    last level resolve to the lowest candidate index."""
    name = "lex_lp"

    def __init__(self, sim: Continuum, model):
        self.sim = sim
        self.model = model

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)          # [n, 4] normalised estimates
        idx = np.arange(est.shape[0])
        for k in range(4):                               # hinge, privacy, carbon, cost
            best = est[idx, k].min()
            idx = idx[est[idx, k] <= best + 1e-12]
        return int(idx[0])

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


class ConstrainedWeightedSum:
    """SLO-constrained weighted sum, shown as "Constr-WS" in the paper (Sect. 5, 6.2).

    The class name and the result key constrained_rl are historical. The rule is a
    one-step selector and not a reinforcement-learning or constrained-MDP learner. It
    learns no policy, value function or Lagrange multiplier. Among the candidates whose
    estimated hinge is at most tol it minimises a fixed weighted sum of the remaining
    objectives (privacy, carbon, cost). If no candidate is feasible it minimises the
    hinge.

    The feasibility filter is level 0 of the lexicographic rule with eta = 0, which means on the
    top priority Constr-WS and Lexi coincide by construction. Below that level it
    applies a price where Lexi applies three more levels, and it therefore inherits the
    scale dependence of the blend, which Sect. 6.2 measures as priority inversions and
    as a rise in PII exposure under a mis-scaled privacy coordinate."""
    name = "constrained_rl"

    # Weights over the non-SLO objectives [privacy, carbon, cost] once a candidate is feasible.
    W_REST = np.array([0.34, 0.33, 0.33])

    def __init__(self, sim: Continuum, model, tol: float = 0.0):
        self.sim = sim
        self.model = model
        self.tol = float(tol)

    def reset(self):
        pass

    def act(self, req, cand, prev_idx):
        est = self.model.predict_all(cand, req)       # [n, 4] normalised estimates
        feas = np.where(est[:, LAT] <= self.tol + 1e-9)[0]
        if len(feas) == 0:
            return int(np.argmin(est[:, LAT]))        # no SLO-feasible option
        scores = est[feas, 1:] @ self.W_REST
        return int(feas[int(np.argmin(scores))])

    def learn(self, chosen_idx, place, obs, req):
        self.model.update(obs, req)


# Historical import name used by the result-generation scripts.
ConstrainedRL = ConstrainedWeightedSum
