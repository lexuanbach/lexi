"""
Simulator of a fog/edge/cloud continuum for Lexi (Sect. 2 and Sect. 5 of the paper).

The directory keeps its earlier name, causalcontinuum. The framework is called Lexi
in the paper, and the code names CausalContinuum* refer to it.

What this file implements
    Def. 1 (continuum, service DAG, placement, objectives). A Continuum holds a fleet
    of Nodes and a service DAG of Stages, and evaluates a placement (one node index per
    stage) into the raw objective vector [latency, privacy, carbon, cost], all lower
    is better. Latency is the critical path over the DAG. Privacy counts PII-tagged
    stages that sit on a public node. Carbon and cost are sums over stages.
    Normalisation (paragraph "Normalisation" of Sect. 2). lex_normalize maps the raw
    vector to [0,1]^4. Objective 0 becomes the SLO hinge max(0, latency - SLO) divided
    by SLO_MARGIN (rho = 60 ms in the paper). The other three divide by the
    operator-declared magnitudes in SCALES and are clamped. Both the true outcomes and
    the model estimates pass through this one map, which means the selector and the evaluation
    share a bounded scale.
    Sect. 5 (setup and inputs, Table 2 for config A). build_nodes and build_dag are the
    7-node, 6-stage configuration of the primary comparison (260 ms SLO, auth and pay
    tagged PII). build_nodes_b and build_dag_b are the independent 10-node, 8-stage,
    3-PII configuration used for the generalisation result of RQ4. build_nodes_scaled
    and build_dag_scaled generate synthetic larger fleets for the scalability
    measurement (extended version, RQ6/RQ7 stress study).
    make_workload draws the geo-diurnal request stream. load_real_traces and the
    real_traces switch replace carbon and load by the real UK grid and Azure Functions
    traces in data/ (the real-trace check of RQ2, extended version).

Two evaluation entry points
    expected_all evaluates a batch of candidate placements without noise. It defines the
    per-step priority-optimal oracle that the regret and priority-inversion measures use.
    realize executes one placement with seeded queueing and carbon noise and returns the
    per-stage observations that the outcome model learns from (causal.py).

The optional contention parameter adds a non-additive co-location term to latency. It
breaks Assumption 1 (additive no-interference) on purpose, for the RQ4 contention
study (Fig. 2c). Everything is deterministic given the seeds passed in. Cost is
O(n_cand * K) numpy work per batch.

Objective vector convention, used throughout the package:
    index 0 = latency (ms, critical path over the DAG)
    index 1 = privacy usage (number of PII stages placed on a public node)
    index 2 = carbon (gCO2 per request)
    index 3 = cost (simulator dollars per request)
The strict priority of the paper (SLO, then privacy, then carbon, then cost) is the
index order.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import json, os
import numpy as np

DAY = 240                 # simulation steps in one diurnal cycle (the 240-step cycle of Sect. 5)
LAT, PRIV, CARB, COST = 0, 1, 2, 3

# Fixed normalisation scales, the operator-declared "typical" magnitudes of Sect. 2.
# True outcomes and model estimates are both mapped through them so that the slack
# thresholds and the selector see the same bounded space. Objective 0 is the SLO hinge
# max(0, latency - SLO). Latency acts as a threshold, which means every SLO-feasible placement
# ties at 0 and privacy then decides among them. SLO_MARGIN is the hinge divisor rho.
SLO_MARGIN = 60.0
SCALES = np.array([SLO_MARGIN, 2.0, 2.5, 5.0])   # [SLO hinge, privacy, carbon, cost]


@dataclass
class Node:
    """One compute node of the continuum (a row of Table 2 in the paper).

    Stage s placed on this node has latency rtt + base_compute * demand_s + q_sens * load,
    where rtt is rtt_intra if the request originates in the node's region and rtt_inter
    otherwise. Carbon of the stage is energy * (demand_s / 20) * intensity(t). The two
    priv_tier values encode the paper's privacy tier (1 for edge, 0 for public nodes).
    """
    name: str
    tier: str                 # "cloud" | "edge" | "serverless"
    region: int               # 0,1,2  (a geographic region tag)
    base_compute: float       # ms per unit of stage demand (cloud small = powerful)
    rtt_intra: float          # ms network RTT when request origin == node region
    rtt_inter: float          # ms network RTT when origin != node region
    q_sens: float             # queueing sensitivity: added ms per unit load level
    cost: float               # $ per invocation
    energy: float             # kWh for a stage of demand 20 (scaled by demand / 20 per stage)
    carbon_base: float        # gCO2/kWh mean grid intensity of the node's region
    carbon_amp: float         # diurnal amplitude of the carbon intensity (gCO2/kWh)
    carbon_phase: float       # diurnal phase of the carbon intensity
    priv_tier: int            # 1 = private/on-prem (edge), 0 = public region


@dataclass
class Stage:
    """One stage of the service DAG. preds lists predecessor indices, and stages are
    stored in topological order so a single forward pass computes the critical path."""
    name: str
    demand: float             # compute demand (drives compute latency + energy)
    pii: bool                 # data-sensitivity tag: True => carries PII
    preds: Tuple[int, ...]    # predecessor stage indices (DAG edges)


@dataclass
class Request:
    """The exogenous signals of one decision (origin region, load, time). The paper's
    worked decision of Sect. 3 conditions on exactly these three quantities."""
    step: int
    t: int                    # wall time in steps, and t % DAY is the time of day
    origin: int               # request origin region (0,1,2)
    load: float               # continuous instantaneous load level (>~0)
    load_bucket: int          # 0,1,2 discretised load (context for baselines)
    tod_bucket: int           # 0..5 discretised time-of-day (context for carbon)


def build_nodes() -> List[Node]:
    """Config A fleet: seven nodes over three regions (Table 2 of the paper).

    The tiers are built to disagree, because a dominant tier would make every decision
    rule pick the same placement.
      cloud       compute-strong, but far away (high RTT) and public.
      edge        compute-weak, dearest, most queueing-sensitive and on the dirtiest
                  grid. It is fast only for requests from its own region and it is
                  the only private tier.
      serverless  cheapest and most energy-efficient, but public and middling on
                  compute and RTT.
    Keeping PII on edge therefore costs latency, carbon and money, which is the
    priority trade-off the controller has to resolve. Magnitudes are simulator units
    and do not track provider prices.
    """
    return [
        # name          tier          reg  bcomp  rttI rttO  qsen  cost  energy  cbase camp cphase priv
        Node("edge_A",  "edge",        0,  2.20,   4,   70,  9.0,  1.15, 0.62e-3, 520, 70, 0.5, 1),
        Node("edge_B",  "edge",        1,  2.00,   5,   72,  8.5,  1.10, 0.58e-3, 540, 80, 2.0, 1),
        Node("edge_C",  "edge",        2,  2.10,   5,   71,  8.8,  1.12, 0.60e-3, 430, 70, 3.8, 1),
        Node("cloud_A", "cloud",       0,  0.50,  35,   46,  2.0,  0.55, 0.95e-3, 360,120, 1.0, 0),
        Node("cloud_C", "cloud",       2,  0.55,  38,   50,  2.2,  0.50, 0.90e-3, 190, 90, 3.5, 0),
        Node("srvl_B",  "serverless",  1,  1.10,  18,   30,  4.5,  0.20, 0.36e-3, 260, 70, 2.5, 0),
        Node("srvl_C",  "serverless",  2,  1.20,  20,   32,  4.8,  0.18, 0.34e-3, 175, 60, 4.0, 0),
    ]


def build_dag() -> List[Stage]:
    """Config A service DAG with K = 6 stages (Sect. 5):

        frontend -> auth -> {search, recommend} -> order -> pay

    The shape follows DeathStarBench-style request graphs. auth and pay carry the PII
    tag, which means the privacy objective takes values in {0, 1, 2}. The parallel branches
    search and recommend carry most of the compute demand and decide latency, while the
    PII stages decide privacy, which means no single stage controls both objectives.
    """
    return [
        Stage("frontend",  8.0,  False, ()),
        Stage("auth",     16.0,  True,  (0,)),
        Stage("search",   30.0,  False, (1,)),
        Stage("recommend",25.0,  False, (1,)),
        Stage("order",    20.0,  False, (2, 3)),
        Stage("pay",      18.0,  True,  (4,)),
    ]


def build_nodes_b() -> List[Node]:
    """Config B fleet: ten nodes with a different tier mix and different economics
    (cheaper and greener edge, dirtier cloud, a serverless node per region). The
    trade-off structure therefore differs from config A. It backs the independent
    configuration result of RQ4, which asks whether the method ordering survives a
    change of topology.
    """
    return [
        # name          tier          reg  bcomp  rttI rttO  qsen  cost  energy   cbase camp cphase priv
        Node("edgeA1", "edge",        0,  1.80,   3,   66,  7.0,  0.85, 0.50e-3, 300, 60, 0.3, 1),
        Node("edgeA2", "edge",        0,  1.90,   4,   68,  7.5,  0.80, 0.52e-3, 320, 70, 0.6, 1),
        Node("edgeB1", "edge",        1,  1.70,   4,   65,  6.8,  0.88, 0.48e-3, 280, 65, 2.2, 1),
        Node("edgeC1", "edge",        2,  1.85,   4,   67,  7.2,  0.82, 0.51e-3, 260, 60, 3.6, 1),
        Node("cloudA", "cloud",       0,  0.45,  40,   52,  1.8,  0.62, 1.10e-3, 470,130, 1.1, 0),
        Node("cloudB", "cloud",       1,  0.48,  42,   54,  1.9,  0.60, 1.05e-3, 450,120, 2.4, 0),
        Node("cloudC", "cloud",       2,  0.50,  44,   56,  2.0,  0.58, 1.00e-3, 300,100, 3.6, 0),
        Node("srvlA",  "serverless",  0,  1.05,  16,   28,  4.2,  0.22, 0.34e-3, 240, 65, 1.5, 0),
        Node("srvlB",  "serverless",  1,  1.10,  17,   29,  4.4,  0.20, 0.33e-3, 230, 60, 2.6, 0),
        Node("srvlC",  "serverless",  2,  1.15,  19,   31,  4.6,  0.19, 0.32e-3, 165, 55, 4.1, 0),
    ]


def build_dag_b() -> List[Stage]:
    """Config B service DAG: eight stages and three PII stages (authN, profile, pay),
    against config A's two:

        gateway -> authN -> {profile, catalog, inventory} -> cart -> checkout -> pay
    """
    return [
        Stage("gateway",   7.0,  False, ()),
        Stage("authN",    15.0,  True,  (0,)),
        Stage("profile",  12.0,  True,  (1,)),
        Stage("catalog",  28.0,  False, (1,)),
        Stage("inventory",22.0,  False, (1,)),
        Stage("cart",     14.0,  False, (2, 3, 4)),
        Stage("checkout", 18.0,  False, (5,)),
        Stage("pay",      16.0,  True,  (6,)),
    ]


def build_nodes_scaled(M: int, seed: int = 0) -> List[Node]:
    """Synthetic fleet of M nodes for the scalability measurement (extended version).

    Tiers cycle through edge, cloud and serverless and regions through 0, 1, 2. The
    per-tier economics are jittered around the hand-built fleets by a seeded generator,
    which keeps the same tension (edge private and dirty, cloud powerful and public,
    serverless cheap and green). The generator seed is 1000 + seed, which means the fleet is
    reproducible."""
    rng = np.random.default_rng(1000 + seed)
    nodes = []
    for i in range(M):
        tier = ("edge", "cloud", "serverless")[i % 3]
        region = i % 3
        if tier == "edge":
            n = Node(f"edge{i}", tier, region, 2.0 + 0.2 * rng.random(), 4 + rng.random(),
                     70 + 3 * rng.random(), 8.5 + rng.random(), 1.1 + 0.1 * rng.random(),
                     0.6e-3, 480 + 80 * rng.random(), 70, float(rng.random() * 4), 1)
        elif tier == "cloud":
            n = Node(f"cloud{i}", tier, region, 0.5 + 0.05 * rng.random(), 35 + 4 * rng.random(),
                     48 + 4 * rng.random(), 2.0 + 0.3 * rng.random(), 0.5 + 0.1 * rng.random(),
                     0.95e-3, 300 + 120 * rng.random(), 110, float(rng.random() * 4), 0)
        else:
            n = Node(f"srvl{i}", tier, region, 1.1 + 0.1 * rng.random(), 18 + 3 * rng.random(),
                     30 + 3 * rng.random(), 4.5 + 0.5 * rng.random(), 0.20 + 0.05 * rng.random(),
                     0.35e-3, 200 + 90 * rng.random(), 65, float(rng.random() * 4), 0)
        nodes.append(n)
    return nodes


def build_dag_scaled(K: int, seed: int = 0) -> List[Stage]:
    """Synthetic K-stage service DAG for the scalability measurement (extended version).

    The DAG is layered with width about sqrt(K). Each stage depends on one or two
    stages of the previous layer, roughly a quarter of the stages carry PII (never the
    first) and demands are drawn from the range of the hand-built DAGs. Stages come out
    in topological order. The generator seed is 2000 + seed."""
    rng = np.random.default_rng(2000 + seed)
    stages = []
    layer_w = max(2, int(round(K ** 0.5)))
    prev_layer = []
    cur_layer = []
    for k in range(K):
        demand = float(8.0 + 24.0 * rng.random())
        pii = bool(rng.random() < 0.25) and k > 0
        if k == 0:
            preds = ()
        else:
            src = prev_layer if prev_layer else [k - 1]
            npred = 1 if rng.random() < 0.6 else 2
            preds = tuple(sorted(set(int(rng.choice(src)) for _ in range(npred))))
        stages.append(Stage(f"s{k}", demand, pii, preds))
        cur_layer.append(k)
        if len(cur_layer) >= layer_w:
            prev_layer, cur_layer = cur_layer, []
    return stages


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_REAL_CACHE = {}


def load_real_traces(regions=("NorthScotland", "London", "Yorkshire")):
    """Load the real carbon and workload traces described in data/PROVENANCE.md.

    Returns (carbon_by_region, arrival_load).
      carbon_by_region  float array [n_regions, T] of gCO2/kWh, one UK Carbon
                        Intensity API regional series per simulator region 0, 1, 2.
      arrival_load      float array [Ta], a load multiplier with mean 1.0 that keeps
                        the per-minute shape of the Azure Functions 2019 trace and
                        stretches its amplitude to the simulator's load range.
    The result is cached at module level. These traces back the real-trace check of
    RQ2 (real_data_experiment.py).
    """
    if _REAL_CACHE:
        return _REAL_CACHE["carbon"], _REAL_CACHE["load"]
    with open(os.path.join(_DATA_DIR, "real_carbon_uk.json")) as f:
        cj = json.load(f)
    carbon = np.array([cj[r] for r in regions], dtype=float)          # [3, T]
    with open(os.path.join(_DATA_DIR, "real_arrivals_azure.json")) as f:
        aj = json.load(f)
    a = np.array(aj["per_min"], dtype=float)
    # Keep the real arrival shape and centre it at 1.0. The factor 2.5 matches the
    # amplitude of the synthetic load (which swings about 0.8 around 1.0), which means the
    # latency tension between tiers is comparable in the two settings.
    a = a / a.mean()
    a = 1.0 + 2.5 * (a - 1.0)
    a = np.clip(a, 0.1, None)
    _REAL_CACHE["carbon"], _REAL_CACHE["load"] = carbon, a
    return carbon, a


class Continuum:
    """The node fleet, the service DAG and the physics that evaluate placements (Def. 1).

    Two entry points share one set of formulas.
      expected_all  noise-free objective vectors for a batch of candidates. It defines
                    the per-step priority optimum (the oracle) and the true outcome
                    of every candidate.
      realize       one executed placement with seeded queueing and carbon noise. Its
                    result feeds the satisfaction metrics of RQ2 and its per-stage
                    observations feed the outcome model.
    Parameters: slo is the latency SLO in ms (260 in the primary comparison), config
    selects the fleet ("A", "B" or "scaled"), contention is the co-location coefficient
    c of RQ4 (0 keeps latency additive), cold_penalty and cold_prob add a serverless
    cold start, and real_traces swaps in the real carbon series.
    """

    def __init__(self, slo: float = 260.0, seed: int = 0, contention: float = 0.0,
                 config: str = "A", cold_penalty: float = 0.0, cold_prob: float = 0.5,
                 n_nodes: int = 12, n_stages: int = 20, topo_seed: int = 0,
                 real_traces: bool = False):
        # With real_traces, carbon_intensity() reads the UK grid regional series
        # instead of the synthetic diurnal signal (real-trace check of RQ2).
        self.real_traces = bool(real_traces)
        self.cold_penalty = float(cold_penalty)   # serverless cold-start latency (ms)
        self.cold_prob = float(cold_prob)         # prob a new serverless placement is cold
        if config == "A":
            self.nodes, self.dag = build_nodes(), build_dag()
        elif config == "B":
            self.nodes, self.dag = build_nodes_b(), build_dag_b()
        elif config == "scaled":                  # synthetic larger fleet and DAG
            self.nodes = build_nodes_scaled(n_nodes, seed=topo_seed)
            self.dag = build_dag_scaled(n_stages, seed=topo_seed)
        else:
            raise ValueError(config)
        self.config = config
        self.M = len(self.nodes)
        self.K = len(self.dag)
        self.slo = float(slo)
        self.n_pii = sum(1 for s in self.dag if s.pii)
        # contention adds latency to a stage in proportion to the number of other
        # stages of the same candidate on its node. A value above 0 makes latency
        # non-additive across stages, which the per-(stage, node) decomposition of
        # Assumption 1 cannot represent. This is the structural misspecification
        # of the RQ4 contention study.
        self.contention = float(contention)

        # Node parameters as arrays of length M, for batch evaluation by fancy indexing.
        self.region      = np.array([n.region for n in self.nodes])
        self.base_compute= np.array([n.base_compute for n in self.nodes])
        self.rtt_intra   = np.array([n.rtt_intra for n in self.nodes])
        self.rtt_inter   = np.array([n.rtt_inter for n in self.nodes])
        self.q_sens      = np.array([n.q_sens for n in self.nodes])
        self.cost        = np.array([n.cost for n in self.nodes])
        self.energy      = np.array([n.energy for n in self.nodes])
        self.carbon_base = np.array([n.carbon_base for n in self.nodes])
        self.carbon_amp  = np.array([n.carbon_amp for n in self.nodes])
        self.carbon_phase= np.array([n.carbon_phase for n in self.nodes])
        self.priv_tier   = np.array([n.priv_tier for n in self.nodes])

        # One real regional series per simulator region, indexed by t. Carbon then
        # differs between two nodes of a region only through their energy per stage
        # (edge is less efficient than serverless), which is where intra-grid
        # variation comes from.
        if self.real_traces:
            carbon_by_region, _ = load_real_traces()          # [n_reg, T]
            self._real_carbon = carbon_by_region
            self._real_T = carbon_by_region.shape[1]

        # DAG parameters as arrays of length K.
        self.demand  = np.array([s.demand for s in self.dag])
        self.demandf = self.demand / 20.0          # energy scaling factor
        self.pii     = np.array([1 if s.pii else 0 for s in self.dag])
        self.preds   = [s.preds for s in self.dag]

    def _rtt(self, origin: int) -> np.ndarray:
        """Per-node round-trip time (length M) for a request from the given origin region."""
        return np.where(self.region == origin, self.rtt_intra, self.rtt_inter)

    def carbon_intensity(self, t: int) -> np.ndarray:
        """Per-node grid carbon intensity (gCO2/kWh, length M) at time t.

        The default is a diurnal sinusoid around each node's regional mean with its own
        amplitude and phase, on the 240-step cycle. With real_traces=True the value is
        the UK regional trace of the node's region (data/real_carbon_uk.json), cycled
        over t.
        """
        if self.real_traces:
            return self._real_carbon[self.region, t % self._real_T]
        theta = 2.0 * np.pi * (t % DAY) / DAY
        return self.carbon_base + self.carbon_amp * np.sin(theta + self.carbon_phase)

    def _critical_path(self, stage_lat: np.ndarray) -> np.ndarray:
        """Critical-path latency over the DAG (the latency of Def. 1).

        stage_lat has shape [n_cand, K] with stages in topological order. The longest
        path to each stage is its own latency plus the largest longest-path value among
        its predecessors, and the end-to-end latency is the maximum over stages. The
        outcome model reuses this method on its predicted stage latencies.
        Returns a vector of length n_cand.
        """
        n = stage_lat.shape[0]
        longest = np.zeros((n, self.K))
        for i in range(self.K):
            if self.preds[i]:
                base = np.max([longest[:, p] for p in self.preds[i]], axis=0)
            else:
                base = np.zeros(n)
            longest[:, i] = stage_lat[:, i] + base
        return longest.max(axis=1)

    def lex_normalize(self, raw: np.ndarray) -> np.ndarray:
        """Map raw objectives [..., 4] into the normalised space [0,1]^4 of Sect. 2.

        Objective 0 becomes the SLO hinge max(0, latency - SLO) / SLO_MARGIN and the
        rest are divided by SCALES. Every coordinate is clamped to [0,1], which gives
        the boundedness that the analysis of Sect. 4 assumes. The clamp is
        non-decreasing, which means strict selection is unchanged as long as no candidate
        saturates it (Prop. 1). The same map is applied to the true outcomes and to the
        model estimates.
        """
        out = np.array(raw, dtype=float, copy=True)
        out[..., LAT] = np.maximum(0.0, out[..., LAT] - self.slo)
        out = out / SCALES
        return np.clip(out, 0.0, 1.0)

    def expected_all(self, cand: np.ndarray, req: Request) -> np.ndarray:
        """Noise-free raw objective matrix for a batch of placements.

        cand is an int array [n_cand, K] holding a node index per stage. The return
        value has shape [n_cand, 4] in the LAT, PRIV, CARB, COST order and is not yet
        normalised. Includes the expected cold-start penalty and the contention term
        when those are switched on, which means it is the noise-free truth that realize samples
        around.
        """
        rtt = self._rtt(req.origin)                       # [M]
        intensity = self.carbon_intensity(req.t)          # [M]
        # Latency of stage s on node m is rtt + base_compute[m] * demand[s] + q_sens[m] * load.
        comp = np.outer(self.demand, self.base_compute)   # [K, M]
        queue = np.outer(np.ones(self.K), self.q_sens * req.load)  # [K, M]
        latKM = rtt[None, :] + comp + queue               # [K, M]
        # Carbon of stage s on node m is energy[m] * (demand[s] / 20) * intensity[m].
        carbKM = (self.energy * intensity)[None, :] * self.demandf[:, None]  # [K, M]

        # Gather the [K, M] tables by each candidate's node choices.
        stage_lat  = latKM[np.arange(self.K)[None, :], cand]
        stage_carb = carbKM[np.arange(self.K)[None, :], cand]

        # Expected cold-start cost: cold_prob times the penalty on every serverless stage.
        if self.cold_penalty > 0.0:
            srvl = np.array([n.tier == "serverless" for n in self.nodes])   # [M]
            stage_lat = stage_lat + self.cold_prob * self.cold_penalty * srvl[cand]
        # Contention: extra queueing per stage grows with the number of other stages of
        # the same candidate that share its node.
        if self.contention > 0.0:
            same = (cand[:, :, None] == cand[:, None, :])          # [n, K, K]
            coloc = same.sum(axis=2) - 1.0                          # [n, K] others on node
            stage_lat = stage_lat + self.contention * self.q_sens[cand] * coloc

        latency = self._critical_path(stage_lat)                       # [n_cand]
        carbon  = stage_carb.sum(axis=1)                               # [n_cand]
        cost    = self.cost[cand].sum(axis=1)                          # [n_cand]
        # Privacy usage counts PII stages on a public (priv_tier == 0) node. It reads
        # the placement and the static tag only, never the data.
        pub = (self.priv_tier[cand] == 0).astype(float)                # [n_cand, K]
        privacy = (pub * self.pii[None, :]).sum(axis=1)                # [n_cand]

        out = np.empty((cand.shape[0], 4))
        out[:, LAT] = latency
        out[:, PRIV] = privacy
        out[:, CARB] = carbon
        out[:, COST] = cost
        return out

    def realize(self, place: np.ndarray, req: Request, prev: np.ndarray | None,
                rng: np.random.Generator):
        """Execute one placement and return (outcome, slo_violation, churn, obs).

        Queueing noise is Gaussian with a scale that grows with the mean queue, and
        carbon carries 3 percent multiplicative noise. Both come from the supplied
        generator, which means a run is stochastic but reproducible. outcome is the raw
        [latency, privacy, carbon, cost] vector, churn counts the stages whose node
        differs from prev, and obs holds one (stage, node, latency, carbon, cost)
        tuple per stage for the outcome model to fit online (Assumption 1 lets one
        execution identify these per-(stage, node) terms).
        """
        rtt = self._rtt(req.origin)
        intensity = self.carbon_intensity(req.t)
        nodes = place                                     # length K node indices

        # Stage latency with queueing noise that grows with load and q_sens.
        comp = self.base_compute[nodes] * self.demand
        queue_mean = self.q_sens[nodes] * req.load
        queue = queue_mean + rng.normal(0.0, 0.10 * (1.0 + queue_mean))
        stage_lat = rtt[nodes] + comp + np.maximum(0.0, queue)
        # Serverless cold start. A stage that was not on a serverless node in the previous
        # placement pays the penalty with probability cold_prob. The additive model can
        # only average this effect.
        if self.cold_penalty > 0.0:
            srvl = np.array([self.nodes[nd].tier == "serverless" for nd in nodes])
            newly = srvl if prev is None else (srvl & (place != prev))
            hit = rng.random(self.K) < self.cold_prob
            stage_lat = stage_lat + self.cold_penalty * (newly & hit)
        # Co-location contention, as in expected_all.
        if self.contention > 0.0:
            coloc = np.array([(nodes == nodes[k]).sum() - 1.0 for k in range(self.K)])
            stage_lat = stage_lat + self.contention * self.q_sens[nodes] * coloc

        # Critical path, scalar version of _critical_path.
        longest = np.zeros(self.K)
        for i in range(self.K):
            base = max((longest[p] for p in self.preds[i]), default=0.0)
            longest[i] = stage_lat[i] + base
        latency = float(longest.max())

        # Carbon with small multiplicative grid noise.
        cnoise = 1.0 + rng.normal(0.0, 0.03, size=self.K)
        stage_carb = self.energy[nodes] * self.demandf * intensity[nodes] * cnoise
        carbon = float(stage_carb.sum())
        cost = float(self.cost[nodes].sum())
        privacy = float(((self.priv_tier[nodes] == 0) * self.pii).sum())

        slo_viol = latency > self.slo
        churn = 0 if prev is None else int((place != prev).sum())

        outcome = np.array([latency, privacy, carbon, cost])
        obs = [(i, int(nodes[i]), float(stage_lat[i]), float(stage_carb[i]),
                float(self.cost[nodes[i]])) for i in range(self.K)]
        return outcome, bool(slo_viol), churn, obs


def make_workload(n_steps: int, seed: int, real_traces: bool = False) -> List[Request]:
    """Deterministic geo-diurnal request stream of n_steps requests (Sect. 5).

    Load follows a diurnal sinusoid with Gaussian jitter. The origin region is drawn
    with time-of-day-dependent weights (region 0 peaks in its morning, region 1 at
    mid-day and region 2 in the evening). As a result the latency-optimal and
    carbon-optimal nodes drift over the day and any fixed placement is wrong for part
    of it. With real_traces=True the load comes from the Azure Functions arrival shape.
    The stream depends only on seed, which means all methods see the same requests.
    """
    rng = np.random.default_rng(seed)
    region_peak = np.array([0.20, 0.50, 0.80])       # time-of-day peak of each origin, as a fraction of the day
    real_load = load_real_traces()[1] if real_traces else None
    reqs: List[Request] = []
    for step in range(n_steps):
        t = step
        tod = (t % DAY) / DAY
        if real_traces:
            base_load = float(real_load[t % len(real_load)])
        else:
            base_load = 1.0 + 0.8 * np.sin(2.0 * np.pi * (tod - 0.25))
        load = float(max(0.1, base_load + rng.normal(0.0, 0.10)))
        load_bucket = 0 if load < 0.9 else (1 if load < 1.6 else 2)
        # Origin weights are a softmax of a cosine around each region's peak.
        w = np.exp(3.0 * np.cos(2.0 * np.pi * (tod - region_peak)))
        w = w / w.sum()
        origin = int(rng.choice(3, p=w))
        tod_bucket = int(tod * 6) % 6
        reqs.append(Request(step, t, origin, load, load_bucket, tod_bucket))
    return reqs
