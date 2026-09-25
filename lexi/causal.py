"""
Counterfactual outcome model of Lexi (the outcome-model interface of Sect. 2, resting on
Assumption 1, and the online-ridge tracking argument of Thm. 1).

The directory keeps its earlier name, causalcontinuum. This file is the "model" half of
the decision loop of Fig. 1. For a request it returns a [n_cand, 4] matrix of normalised
predicted objectives for every candidate placement, including the ones that were never
executed. The selector in controller.py reads only this matrix. The model never sees the
priority order and the selector never sees units or uncertainty, which means the two share nothing
but the matrix.

Identification. Under Assumption 1 each objective is a sum over stages (latency: along the
critical path) of a per-(stage, node) term that depends on the stage's assignment and the
exogenous request state, whichever other stages are co-placed. One executed placement
therefore yields K stage observations spread over the fleet, and because every candidate
reuses the same nodes, one execution informs the counterfactual value of many
un-executed candidates. That is why the controller needs no exploratory migrations.

Per-objective estimators, all online and pure numpy:
    latency  one ridge regression per node on [1, demand, load, same_region]. The
             coefficients recover the inter-region RTT, the demand slope (compute per
             unit demand), the queueing sensitivity and the RTT difference between
             intra- and inter-region requests. Stage predictions are combined by the
             critical path of the DAG (Continuum._critical_path).
    carbon   one ridge regression per node on [1, sin, cos] of the time of day. It
             learns the rate energy * intensity(t), which is scaled by demand / 20.
    cost     per-node running mean.
    privacy  policy metadata (node privacy tier times stage PII tag). It is read from
             the placement and never learned, as in the topological privacy coordinate
             of Sect. 2.
The ridge solve is a d x d system per node and per decision (d is 3 or 4), and the
update touches K terms, which means a step costs O(K) in the update and O(n_cand * K) in the
prediction.

The eps argument injects a controlled, bounded misspecification (a fixed per-node bias
sign plus proportional noise) into the normalised latency and carbon estimates. It is
used for the parametric misspecification sweep, where the model error eps_t of Thm. 1
is set by hand instead of measured.

Two subclasses at the end of the file are extensions of the additive model. The
contention-aware and doubly-robust variants back the model-stress rows of the extended
version (RQ6/RQ7, Table "Measured robustness and deployment extensions"). The camera-ready
paper uses only CausalModel.
"""
from __future__ import annotations
import numpy as np
from sim import Continuum, Request, DAY, LAT, PRIV, CARB, COST, SCALES, SLO_MARGIN

# Scale of the normalised SLO hinge (objective 0). A raw-millisecond correction such as
# the learned contention term is divided by this before it enters the hinge.
SCALES_LAT = SLO_MARGIN


class CausalModel:
    """Additive per-(stage, node) outcome model with online ridge estimators.

    sim is the Continuum whose fleet and DAG define the model dimensions. ridge is the
    regulariser of the normal equations. The model starts from weak nominal priors.
    Predictions are poor early on and improve as executed placements arrive through
    update. predict_all returns the [n_cand, 4] normalised matrix that the selector
    consumes.
    """

    # Gains of the bounded misspecification: per unit of eps, a systematic bias and a
    # proportional-noise scale added to the normalised estimates. They are kept modest
    # so that model error degrades the outcome gradually, roughly base + C * eps, which
    # is the graceful degradation that the tracking bound of Thm. 1 predicts.
    BIAS_GAIN = 0.12
    NOISE_SCALE = 0.20

    def __init__(self, sim: Continuum, eps: float = 0.0, seed: int = 0,
                 ridge: float = 1e-2):
        self.sim = sim
        self.M = sim.M
        self.eps = float(eps)
        self.ridge = float(ridge)
        rng = np.random.default_rng(seed)
        # Fixed per-node misspecification signs in {-1, +1}, with independent patterns
        # for the latency and carbon estimators.
        self.bias_sign = rng.choice([-1.0, 1.0], size=self.M)
        self.bias_sign2 = rng.choice([-1.0, 1.0], size=self.M)
        self.mnoise = rng  # the same generator draws the proportional eps noise

        d_lat, d_carb = 4, 3
        # Sufficient statistics of the ridge normal equations, per node and objective.
        self.A_lat = np.stack([self.ridge * np.eye(d_lat) for _ in range(self.M)])
        self.b_lat = np.zeros((self.M, d_lat))
        self.A_carb = np.stack([self.ridge * np.eye(d_carb) for _ in range(self.M)])
        self.b_carb = np.zeros((self.M, d_carb))
        self.cost_sum = np.zeros(self.M)
        self.cost_n = np.zeros(self.M)

        # Weak nominal priors, standing in for an operator's rough spec sheet. They are
        # imperfect on purpose. They assume an average load and ignore the diurnal
        # carbon cycle. Early predictions are therefore wrong and online data visibly improves
        # them (the convergence measurement).
        self._seed_priors()

    def _feat_lat(self, demand: float, load: float, same_region: float) -> np.ndarray:
        return np.array([1.0, demand, load, same_region])

    def _feat_carb(self, t: int) -> np.ndarray:
        theta = 2.0 * np.pi * (t % DAY) / DAY
        return np.array([1.0, np.sin(theta), np.cos(theta)])

    def _seed_priors(self):
        """Add a few low-weight pseudo-observations per node, drawn from the nominal
        spec (load 1.0, flat carbon at the regional mean). Latency uses two demands and
        both region cases, carbon uses two times of day, and cost starts at the listed
        price."""
        w = 0.4  # pseudo-observation weight, small against the real data
        for m in range(self.M):
            nd = self.sim.nodes[m]
            for demand in (10.0, 25.0):
                for same in (0.0, 1.0):
                    rtt = nd.rtt_intra if same else nd.rtt_inter
                    lat = rtt + nd.base_compute * demand + nd.q_sens * 1.0  # load~1
                    x = self._feat_lat(demand, 1.0, same)
                    self.A_lat[m] += w * np.outer(x, x)
                    self.b_lat[m] += w * lat * x
            # Carbon prior: flat intensity at the regional mean.
            for t in (0, DAY // 2):
                x = self._feat_carb(t)
                rate = nd.energy * nd.carbon_base
                self.A_carb[m] += w * np.outer(x, x)
                self.b_carb[m] += w * rate * x
            self.cost_sum[m] += w * nd.cost
            self.cost_n[m] += w

    def update(self, obs, req: Request):
        """Fold the K stage observations of one executed placement into the estimators.

        obs is the list of (stage_idx, node_idx, latency, carbon, cost) tuples that
        Continuum.realize returns. Each tuple adds one row to its node's latency and
        carbon regressions, which means the cost is O(K) per step.
        """
        for (s, m, lat, carb, cost) in obs:
            same = 1.0 if self.sim.region[m] == req.origin else 0.0
            x = self._feat_lat(self.sim.demand[s], req.load, same)
            self.A_lat[m] += np.outer(x, x)
            self.b_lat[m] += lat * x
            # The carbon regression target is the rate energy * intensity. Demand / 20 is divided out.
            xc = self._feat_carb(req.t)
            rate = carb / max(self.sim.demandf[s], 1e-9)
            self.A_carb[m] += np.outer(xc, xc)
            self.b_carb[m] += rate * xc
            self.cost_sum[m] += cost
            self.cost_n[m] += 1.0

    def _thetas(self):
        """Solve the ridge normal equations of every node. Returns the latency
        coefficients [M, 4], the carbon coefficients [M, 3] and the mean cost [M]."""
        theta_lat = np.stack([np.linalg.solve(self.A_lat[m], self.b_lat[m])
                              for m in range(self.M)])          # [M, 4]
        theta_carb = np.stack([np.linalg.solve(self.A_carb[m], self.b_carb[m])
                               for m in range(self.M)])         # [M, 3]
        cost_mean = self.cost_sum / np.maximum(self.cost_n, 1e-9)
        return theta_lat, theta_carb, cost_mean

    def predict_raw_latency(self, cand: np.ndarray, req: Request) -> np.ndarray:
        """Estimated critical-path latency in ms, before hinging and normalising.

        The SLO safety-margin controller (extended version, RQ5) needs the predicted
        headroom SLO - latency. The normalised hinge clips that to 0 for every feasible
        placement, which means it cannot expose it. The computation is the latency path of
        predict_all without the misspecification perturbation. It is meant
        for the nominal model (eps = 0).
        """
        theta_lat, _, _ = self._thetas()
        same = (self.sim.region == req.origin).astype(float)
        K, M = self.sim.K, self.M
        latKM = np.empty((K, M))
        for s in range(K):
            d = self.sim.demand[s]
            X = np.stack([np.ones(M), np.full(M, d), np.full(M, req.load), same], axis=1)
            latKM[s] = (X * theta_lat).sum(axis=1)
        stage_lat = latKM[np.arange(K)[None, :], cand]
        return self.sim._critical_path(stage_lat)

    def predict_all(self, cand: np.ndarray, req: Request) -> np.ndarray:
        """Predict the normalised objective matrix [n_cand, 4] for a batch of candidates.

        Column 0 is the estimated SLO hinge, and columns 1 to 3 are privacy, carbon and
        cost, all in the normalised space of Continuum.lex_normalize. This is the
        matrix drawn as the only interface in Fig. 1. Latency and carbon come from the
        per-node regressions, cost from the running mean and privacy from the placement.

        With eps > 0 a bounded perturbation is added to the normalised latency and
        carbon columns. It is a fixed per-node bias (the mean of the node signs over
        the candidate) plus Gaussian noise, both linear in eps. Because it is bounded,
        a wrong selection costs at most O(eps) in true objective, which is the base +
        C * eps degradation used in the misspecification sweep. An unbounded bias on raw
        latency would instead collapse the selector abruptly.
        """
        theta_lat, theta_carb, cost_mean = self._thetas()
        same = (self.sim.region == req.origin).astype(float)     # [M]
        xc = self._feat_carb(req.t)                              # [3]
        carb_rate = theta_carb @ xc                              # [M] = energy*intensity

        K, M = self.sim.K, self.M
        # Per-(stage, node) latency and carbon estimates as [K, M] tables.
        latKM = np.empty((K, M))
        carbKM = np.empty((K, M))
        for s in range(K):
            d = self.sim.demand[s]
            X = np.stack([np.ones(M), np.full(M, d),
                          np.full(M, req.load), same], axis=1)   # [M, 4]
            latKM[s] = (X * theta_lat).sum(axis=1)
            carbKM[s] = self.sim.demandf[s] * carb_rate

        stage_lat  = latKM[np.arange(K)[None, :], cand]
        stage_carb = carbKM[np.arange(K)[None, :], cand]
        latency = self.sim._critical_path(stage_lat)
        carbon = stage_carb.sum(axis=1)
        cost = cost_mean[cand].sum(axis=1)
        pub = (self.sim.priv_tier[cand] == 0).astype(float)
        privacy = (pub * self.sim.pii[None, :]).sum(axis=1)

        out = np.empty((cand.shape[0], 4))
        out[:, LAT], out[:, PRIV], out[:, CARB], out[:, COST] = \
            latency, privacy, carbon, cost
        out = self.sim.lex_normalize(out)                        # -> [n_cand, 4]

        if self.eps > 0.0:
            n = cand.shape[0]
            # Per-candidate systematic bias is the mean of the fixed node signs.
            b_lat = self.bias_sign[cand].mean(axis=1)            # [n_cand] in [-1,1]
            b_carb = self.bias_sign2[cand].mean(axis=1)
            e = self.eps
            out[:, LAT] = np.maximum(0.0, out[:, LAT]
                + e * (self.BIAS_GAIN * b_lat
                       + self.NOISE_SCALE * self.mnoise.normal(0.0, 1.0, n)))
            out[:, CARB] = np.maximum(0.0, out[:, CARB]
                + e * (self.BIAS_GAIN * b_carb
                       + self.NOISE_SCALE * self.mnoise.normal(0.0, 1.0, n)))
        return out


# Extensions of the additive model. Both subclass CausalModel and reuse its per-node ridge
# estimators, which means the base model is untouched. They belong to the extended version.


class ContentionCausalModel(CausalModel):
    r"""Additive model plus a learned co-location contention term (extended version, RQ6).

    The base CausalModel assumes latency is additive per (stage, node), which is
    Assumption 1. It therefore cannot represent shared-node queueing, where several of a
    candidate's stages land on the same node and wait behind one another. This is an
    M/G/1-like effect. The extra wait grows with the number of co-located jobs and with
    the node's queue sensitivity. The subclass adds a per-stage correction
    :math:`c\cdot q_n\cdot(\text{co-located stages}-1)`, the same structural form as the
    simulator's contention term, and recomputes the critical path with it, which means the
    correction lengthens exactly the co-located stages on the busiest path.

    The single coefficient :math:`c` is fitted online by a one-dimensional ridge
    regression of the residual (realised latency minus the additive prediction) on the
    co-location feature. It uses the same executed observations as the base model and no
    knowledge of the simulator's true coefficient. An evidence gate keeps :math:`c` at 0
    until it exceeds GAIN_GATE after 100 observations. Without true contention the gate
    stays shut and the model equals the additive one.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # One global ridge for the contention coefficient, with the co-location feature
        # as regressor and the residual latency as target. A single scalar matches the
        # simulator's single contention coefficient, and the per-node q_sens already
        # scales the effect per stage.
        self.cg_A = 20.0        # weak ridge prior: the coefficient starts near 0
        self.cg_b = 0.0
        self.cg_n = 0.0         # observation count (for the evidence gate)

    def _coloc(self, cand: np.ndarray) -> np.ndarray:
        """Per-(candidate, stage) co-location count, the number of other stages of the
        same candidate on that stage's node. Shape [n_cand, K], zero for a lone stage."""
        same = (cand[:, :, None] == cand[:, None, :])      # [n, K, K]
        return same.sum(axis=2) - 1.0                       # [n, K]

    def update(self, obs, req: Request):
        super().update(obs, req)
        place = np.array([m for (_, m, _, _, _) in obs])
        if len(place) == 0:
            return
        coloc = self._coloc(place[None, :])[0]              # [K]
        for idx, (s, m, lat, carb, cost) in enumerate(obs):
            same = 1.0 if self.sim.region[m] == req.origin else 0.0
            rtt = self.sim.rtt_intra[m] if same else self.sim.rtt_inter[m]
            add_pred = rtt + self.sim.base_compute[m] * self.sim.demand[s] \
                + self.sim.q_sens[m] * req.load
            resid = lat - add_pred                          # what the additive model leaves: contention plus noise
            x = self.sim.q_sens[m] * coloc[idx]             # regressor of the contention term
            self.cg_A += x * x
            self.cg_b += x * resid
            self.cg_n += 1.0

    GAIN_GATE = 0.02     # smallest coefficient treated as real, which keeps the model from fitting noise

    def _coef(self) -> float:
        c = self.cg_b / max(self.cg_A, 1e-9)
        # Evidence gate. A coefficient fitted from queueing noise alone is small and
        # unstable. It is applied only after it clears the threshold on enough
        # observations so that with zero true contention the correction is exactly 0.
        if c <= self.GAIN_GATE or self.cg_n < 100:
            return 0.0
        return float(c)

    def _extra_path_ms(self, cand: np.ndarray, req: Request) -> np.ndarray:
        """Increase in critical-path latency (ms, [n_cand]) caused by the learned
        contention term. It adds c * q_n * coloc to each stage, re-runs the critical path
        and subtracts the additive critical path, which means contention only counts where it
        lengthens the longest path."""
        c = self._coef()
        if c == 0.0:
            return np.zeros(cand.shape[0])
        # Additive per-(stage, node) latency, as in predict_raw_latency.
        base = super().predict_raw_latency(cand, req)       # [n] additive crit path
        # Per-stage extra, then the critical path with the extra included.
        coloc = self._coloc(cand)                           # [n, K]
        extra_stage = c * self.sim.q_sens[cand] * coloc     # [n, K]
        # Rebuild the per-stage additive latency so that the extra can be added before re-pathing.
        same = (self.sim.region == req.origin).astype(float)
        theta_lat, _, _ = self._thetas()
        K, M = self.sim.K, self.M
        latKM = np.empty((K, M))
        for s in range(K):
            X = np.stack([np.ones(M), np.full(M, self.sim.demand[s]),
                          np.full(M, req.load), same], axis=1)
            latKM[s] = (X * theta_lat).sum(axis=1)
        stage_lat = latKM[np.arange(K)[None, :], cand] + extra_stage   # [n, K]
        path_with = self.sim._critical_path(stage_lat)      # [n]
        return np.maximum(0.0, path_with - base)

    def predict_all(self, cand: np.ndarray, req: Request) -> np.ndarray:
        out = super().predict_all(cand, req)
        extra = self._extra_path_ms(cand, req)              # [n] ms
        out[:, LAT] = np.clip(out[:, LAT] + extra / SCALES_LAT, 0.0, 1.0)
        return out

    def predict_raw_latency(self, cand: np.ndarray, req: Request) -> np.ndarray:
        return super().predict_raw_latency(cand, req) + self._extra_path_ms(cand, req)


class DoublyRobustCausalModel(CausalModel):
    r"""Doubly-robust (DR) variant of the outcome model (extended version, RQ6).

    The base model is a direct method (DM). It regresses each objective on per-(stage,
    node) features and reads off the counterfactual. The DR construction of Dudik et al.
    adds a correction built from the residual of the observed action, realised outcome
    minus DM prediction. Here that residual is kept per node as an exponentially weighted
    sum, and it is transferred to an un-executed candidate through the nodes it shares
    with the executed placements (the mean of the node residuals over the candidate's
    stages). The correction cancels in expectation when the DM is right. With a
    perfect DM the residuals vanish and DR equals DM. The usual DR guarantee, that the
    estimate is unbiased if either the outcome model or the propensity is right, applies
    here with an implicit uniform propensity over candidates.

    The correction is applied to carbon only (see predict_all). The extended-version
    table reports that the DM is already near unbiased, and DR changes no decision.
    """

    DECAY = 0.98            # EW decay for the per-node residual memory
    # The correction is shrunk by how consistent the per-node residual is, measured as
    # mean^2 / (mean^2 + variance). Pure noise (mean near 0, large spread) is shrunk to
    # about 0, which means DR adds no noise. A systematic bias (mean much larger than the spread,
    # the misspecified-model case) is corrected almost in full. This is the
    # self-normalised DR estimator with the bias correction gated by an evidence ratio.
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.res_lat = np.zeros(self.M)     # exponentially weighted sum of the per-node latency residual
        self.res_carb = np.zeros(self.M)
        self.res2_lat = np.zeros(self.M)    # the same for the squared residual, which gives the spread
        self.res2_carb = np.zeros(self.M)
        self.res_w = np.zeros(self.M)       # exponentially weighted count, for normalisation

    def update(self, obs, req: Request):
        # Residuals are taken before the DM update, which means each one is out of sample: the
        # realised per-stage latency and carbon are compared with the DM prediction
        # that did not yet see them.
        theta_lat, theta_carb, _ = self._thetas()
        xc = self._feat_carb(req.t)
        for (s, m, lat, carb, cost) in obs:
            same = 1.0 if self.sim.region[m] == req.origin else 0.0
            x = self._feat_lat(self.sim.demand[s], req.load, same)
            dm_lat = float(x @ theta_lat[m])
            rate = float(theta_carb[m] @ xc)
            dm_carb = self.sim.demandf[s] * rate
            rl, rc = lat - dm_lat, carb - dm_carb
            self.res_lat[m] = self.DECAY * self.res_lat[m] + rl
            self.res_carb[m] = self.DECAY * self.res_carb[m] + rc
            self.res2_lat[m] = self.DECAY * self.res2_lat[m] + rl * rl
            self.res2_carb[m] = self.DECAY * self.res2_carb[m] + rc * rc
            self.res_w[m] = self.DECAY * self.res_w[m] + 1.0
        super().update(obs, req)          # the direct model keeps learning as well

    def _res(self, s1, s2):
        """Shrunk per-node residual, mean * mean^2 / (mean^2 + var). A consistent
        systematic bias survives and pure noise shrinks toward zero. s1 and s2 are the
        weighted sums of the residual and of its square."""
        w = np.maximum(self.res_w, 1e-9)
        mean = s1 / w
        var = np.maximum(s2 / w - mean * mean, 0.0)
        shrink = mean * mean / (mean * mean + var + 1e-9)
        return mean * shrink

    def predict_all(self, cand: np.ndarray, req: Request) -> np.ndarray:
        out = super().predict_all(cand, req)               # DM normalised
        rc = self._res(self.res_carb, self.res2_carb)      # [M] shrunk carbon residual
        # The correction is applied to carbon only. Under strict lexicographic selection
        # a correction to the top-priority hinge would change which candidates survive
        # the SLO and privacy levels, and it would add variance to a latency model that is
        # nearly unbiased under Assumption 1. Carbon is a lower priority. A correction
        # there cannot cause a higher-priority regression, and carbon is where a
        # misspecification bias inflates the estimate. This is the decision-safe DR variant.
        corr_carb = rc[cand].mean(axis=1)                  # [n]
        out[:, CARB] = np.clip(out[:, CARB] + corr_carb / SCALES[CARB], 0.0, 1.0)
        return out
