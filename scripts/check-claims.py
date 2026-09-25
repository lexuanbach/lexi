#!/usr/bin/env python3
"""Check every number of the camera-ready paper that comes from a result file, and the
headline numbers and tables of the extended version, against the committed results.

Step 1 runs lexi/check_results.py, the original 97-claim gate of the code base.
Step 2 walks the claim table below. Each entry names the paper location (CR = camera-ready,
EXT = extended version, with the label of the current PDFs), the value printed in the
paper, the result file and key it comes from, and the tolerance, which is half a unit of
the last printed digit unless the paper rounds more coarsely. CLAIM_MATRIX.md lists the
same claims with the command that regenerates each file.

Timing values (decision latency) depend on the machine. They are printed as INFO and are
never a failure.

Exit status 0 means that every claim matches.
"""
from pathlib import Path
import json, os, subprocess, sys

sys.dont_write_bytecode = True   # keep __pycache__ out of the package
ROOT = Path(__file__).resolve().parents[1]
LEXI = ROOT / "lexi"
RES = ROOT / "results"

r = subprocess.run([sys.executable, "check_results.py"], cwd=LEXI,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
if r.returncode:
    raise SystemExit(r.returncode)

sys.path.insert(0, str(LEXI))
import sim as SIM  # noqa: E402
from baselines import DEFAULT_WS_W, TUNED_WS_W, ConstrainedWeightedSum  # noqa: E402
from controller import candidate_set  # noqa: E402


def load(name):
    return json.loads((RES / name).read_text())


C = load("core_results.json")
X = load("extended_results.json")
RD = load("real_data_results.json")
ROB = load("robustness_results.json")
DEP = load("deployment_results.json")
DC = load("decision_checks.json")

T = C["e1_satisfaction"]["table"]
INV = C["e1_inversion"]["inversion"]
D = X["D_worked_example"]
PANEL = {p["idx"]: p for p in D["panel"]}
E3P = {r["factor"]: r for r in C["e3_scale_invariance"]["objectives"]["privacy"]}
BCR = {r["factor"]: r for r in X["B_constrrl_scale"]["rows"]}
MENU = X["A_menu"]["sets"]
E5 = C["e5_candidate"]["sets"]
CB = C["e6_generalization"]["config_b"]
SWEEP = {r["slo_ms"]: r for r in C["e6_generalization"]["slo_sweep"]}
CONT = {r["contention"]: r for r in C["e7_misspecification"]["structural_contention"]["slo_pct"]}
RX1 = {r["contention"]: r for r in ROB["rx1_contention"]["rows"]}
COLD = {int(l["cold_penalty"]): l for l in C["e11_conflict_stress"]["levels"]}
MG = {p["margin_ms"]: p for p in C["e10_margin"]["nominal"]}
MGC = {p["margin_ms"]: p for p in C["e10_margin"]["contention"]}
E4 = C["e4_stability"]["points"]
PA = X["C_pred_acc"]
OA = DC["oracle_agreement"]
WD = DC["worked_decision"]
HY = DC["hysteresis_effect"]

CONT_SIM = SIM.Continuum(seed=0)
TIER = {"edge": "E", "cloud": "C", "serverless": "S"}
PID = {n.name: f"{TIER[n.tier]}{n.region}" for n in CONT_SIM.nodes}
NODE = {PID[n.name]: n for n in CONT_SIM.nodes}

claims = []   # (location, description, paper value, measured value, passed)
infos = []


def num(loc, desc, paper, measured, tol):
    ok = abs(float(measured) - float(paper)) <= tol + 1e-9
    claims.append((loc, desc, paper, measured, ok))


def same(loc, desc, paper, measured):
    claims.append((loc, desc, paper, measured, paper == measured))


def true(loc, desc, cond, detail=""):
    claims.append((loc, desc, "holds", detail or bool(cond), bool(cond)))


def nums(loc, desc, paper, measured, tol):
    ok = len(paper) == len(measured) and all(abs(float(a) - float(b)) <= tol + 1e-9
                                             for a, b in zip(paper, measured))
    claims.append((loc, desc, paper, measured, ok))


def mean(key, metric):
    return T[key][metric]["mean"]


# ---------------------------------------------------------------- camera-ready (CR)
# Abstract and Sect. 1
num("CR Abstract", "seeds of the main run", 15, C["e1_satisfaction"]["n_seeds"], 0)
num("CR Abstract, Sect. 6.1", "Lexi SLO violations 1.46%", 1.46, mean("Lexi-strict", "slo"), 0.005)
num("CR Abstract, Sect. 6.1", "Lexi PII exposure 0.05", 0.05, mean("Lexi-strict", "privacy"), 0.005)
num("CR Abstract", "untuned sum SLO violations 23%", 23, mean("ws_default", "slo"), 0.5)
num("CR Abstract", "untuned sum PII exposure 1.94", 1.94, mean("ws_default", "privacy"), 0.005)
num("CR Abstract, Sect. 1, Sect. 6.1", "weightings tested", 84, C["e2_weight_sweep"]["n_weights"], 0)
num("CR Abstract, Sect. 1, Sect. 6.1", "weightings meeting the target (one of 84)", 1,
    round(C["e2_weight_sweep"]["frac_meeting_target"] * C["e2_weight_sweep"]["n_weights"]), 0)
num("CR Abstract, Sect. 6.2", "WS-tuned PII at privacy x0.1: 1.85", 1.85, E3P[0.1]["ws_priv"], 0.005)
num("CR Abstract, Sect. 6.2", "Constr-WS PII at privacy x0.1: 1.84", 1.84, BCR[0.1]["constr_rl_priv"], 0.005)
true("CR Abstract, Sect. 6.2", "Lexi unchanged across the privacy-scale sweep",
     len({r["lexi_priv"] for r in E3P.values()}) == 1)

# Sect. 2
num("CR Sect. 2", "hinge scale rho = 60 ms", 60, SIM.SLO_MARGIN, 0)
num("CR Sect. 2", "hinge saturates at 320 ms", 320, CONT_SIM.slo + SIM.SLO_MARGIN, 0)

# Fig. 1 and Sect. 3, Table 1
same("CR Fig. 1", "survivor counts 40, 11, 4, 1",
     [40, 11, 4, 1], [D["band_trace"][0]["n_before"]] + [b["n_after"] for b in D["band_trace"]])
# Corrected in this revision: the model is fitted on steps 0-903 (904 updates).
num("CR Sect. 3, EXT Sect. 4.2", "model fitted on steps 0-903, decision at step 904", 904, D["step"], 0)
same("CR Sect. 3", "request signals: origin 2, load 0.765, t 904",
     [2, 0.765, 904], [D["request"]["origin"], D["request"]["load"], D["request"]["t"]])
num("CR Sect. 3", "candidates in the matrix (39 never executed)", 40, D["n_cand"], 0)
lvl0 = D["band_trace"][0]["survivors"]
num("CR Sect. 3", "9 of the 11 level-0 survivors truly meet the SLO", 9,
    sum(PANEL[i]["true_lat_ms"] <= D["slo_ms"] for i in lvl0), 0)
same("CR Sect. 3, Table 1", "admitted by prediction error: #24 and one other (#33)",
     [24, 33], [i for i in lvl0 if PANEL[i]["true_lat_ms"] > D["slo_ms"]])
same("CR Sect. 3", "level-1 survivors {2, 19, 20, 21}", [2, 19, 20, 21], D["band_trace"][1]["survivors"])
same("CR Sect. 3, Table 1", "chosen placement #21 wins carbon outright", [21],
     D["band_trace"][2]["survivors"])
same("CR Sect. 3", "band 0.08 keeps #20 and #21, cost picks #21", [[20, 21], 21],
     [WD["carbon_band_0.08_survivors"], WD["carbon_band_0.08_choice"]])
num("CR Sect. 3", "band check 0.387 <= 0.314 + 0.08", 0.387, PANEL[20]["pred_norm"][2], 0.0005)
num("CR Sect. 3", "untuned sum picks #6", 6, WD["ws_default_choice"], 0)
same("CR Sect. 3", "tuned sum and Constr-WS pick #21", [21, 21],
     [WD["ws_tuned_choice"], WD["constr_ws_choice"]])
num("CR Sect. 3", "tuned-sum score of #21: 0.030", 0.030, WD["ws_tuned_score"]["21"], 0.0005)
num("CR Sect. 3", "tuned-sum score of #6: 0.111", 0.111, WD["ws_tuned_score"]["6"], 0.0005)
num("CR Sect. 3", "privacy x0.1 switches the tuned sum to #6", 6, WD["ws_tuned_privacy_x0.1_choice"], 0)
num("CR Sect. 3", "privacy x0.1 score of #6: 0.021", 0.021, WD["ws_tuned_privacy_x0.1_score"]["6"], 0.0005)
num("CR Sect. 3", "privacy x0.1 score of #21: 0.030", 0.030, WD["ws_tuned_privacy_x0.1_score"]["21"], 0.0005)
true("CR Sect. 3", "no privacy rescaling changes the lexicographic choice",
     WD["strict_choice_invariant_to_privacy_rescaling"])

TABLE1 = {  # row: (placement, pred ms, true ms, g0, g1, g2, g3, eliminated at)
    2: ("E2/E2/E2/E2/E2/E2", 252.5, 251.9, 0, 0.0, 0.673, 1.000, "L2"),
    4: ("C2/C2/C2/C2/C2/C2", 222.7, 249.0, 0, 1.0, 0.395, 0.600, "L1"),
    6: ("S2/S2/S2/S2/S2/S2", 228.9, 228.8, 0, 1.0, 0.166, 0.216, "L1"),
    9: ("E2/S2/C2/C2/S2/S2", 215.6, 220.5, 0, 1.0, 0.309, 0.532, "L1"),
    19: ("C2/E2/C2/C2/C2/E2", 230.3, 245.8, 0, 0.0, 0.476, 0.848, "L2"),
    20: ("S1/E2/S1/S1/S1/E2", 259.1, 259.0, 0, 0.0, 0.387, 0.608, "L2"),
    21: ("S2/E2/S2/S2/S2/E2", 235.7, 235.5, 0, 0.0, 0.314, 0.592, "-"),
    24: ("C0/C2/S2/S1/C2/C0", 256.2, 268.2, 0, 1.0, 0.371, 0.496, "L1"),
    31: ("S1/S2/E2/C2/S2/C2", 252.4, 257.1, 0, 1.0, 0.388, 0.536, "L1"),
    33: ("C2/S1/S1/C0/E2/E2", 259.3, 264.8, 0, 0.5, 0.496, 0.738, "L1"),
    36: ("S1/C2/C0/E2/S1/E2", 255.0, 259.9, 0, 0.5, 0.539, 0.738, "L1"),
}
lvl1 = set(D["band_trace"][1]["survivors"])


def table1_row(i, loc):
    p = PANEL[i]
    place, pred, tru, g0, g1, g2, g3, elim = TABLE1[i]
    got_elim = "-" if i == D["chosen"] else ("L2" if i in lvl1 else "L1")
    got = ("/".join(PID[n] for n in p["placement"]), p["pred_lat_ms"], p["true_lat_ms"])
    g = p["pred_norm"]
    ok = (got[0] == place and abs(got[1] - pred) < 0.051 and abs(got[2] - tru) < 0.051
          and abs(g[0] - g0) < 1e-9 and abs(g[1] - g1) < 0.051
          and abs(g[2] - g2) < 0.0006 and abs(g[3] - g3) < 0.0006 and got_elim == elim)
    claims.append((loc, f"Table 1 row #{i}", TABLE1[i],
                   (got[0], got[1], got[2], g, got_elim), ok))


for i in (2, 6, 19, 20, 21, 24):
    table1_row(i, "CR Table 1")

# Sect. 5 and Table 2
num("CR Sect. 5", "nodes M = 7", 7, CONT_SIM.M, 0)
num("CR Sect. 5", "regions", 3, len(set(CONT_SIM.region.tolist())), 0)
num("CR Sect. 5", "stages K = 6", 6, CONT_SIM.K, 0)
num("CR Sect. 5", "SLO 260 ms", 260, CONT_SIM.slo, 0)
num("CR Sect. 5", "menu size 40", 40, C["_meta"]["cand_cap"], 0)
same("CR Sect. 5", "evaluation window steps 800-1599", [800, 1600],
     [C["_meta"]["warmup"], C["_meta"]["horizon"]])
same("CR Sect. 5", "sweeps use 6-8 seeds (E2, E3, E5, E6, menu, Constr-WS scale)",
     [6, 8, 8, 8, 8, 8],
     [C["e2_weight_sweep"]["n_seeds"], C["e3_scale_invariance"]["n_seeds"],
      C["e5_candidate"]["n_seeds"], C["e6_generalization"]["n_seeds"],
      X["A_menu"]["n_seeds"], X["B_constrrl_scale"]["n_seeds"]])
TABLE2 = {  # id: tier, region, b, RTTin, RTTout, q, cost, energy (Wh), mean intensity, private
    "E0": ("edge", 0, 2.20, 4, 70, 9.0, 1.15, 0.62, 520, 1),
    "E1": ("edge", 1, 2.00, 5, 72, 8.5, 1.10, 0.58, 540, 1),
    "E2": ("edge", 2, 2.10, 5, 71, 8.8, 1.12, 0.60, 430, 1),
    "C0": ("cloud", 0, 0.50, 35, 46, 2.0, 0.55, 0.95, 360, 0),
    "C2": ("cloud", 2, 0.55, 38, 50, 2.2, 0.50, 0.90, 190, 0),
    "S1": ("serverless", 1, 1.10, 18, 30, 4.5, 0.20, 0.36, 260, 0),
    "S2": ("serverless", 2, 1.20, 20, 32, 4.8, 0.18, 0.34, 175, 0),
}
same("CR Table 2", "node ids", sorted(TABLE2), sorted(NODE))
for nid, row in TABLE2.items():
    n = NODE[nid]
    got = (n.tier, n.region, n.base_compute, n.rtt_intra, n.rtt_inter, n.q_sens, n.cost,
           round(n.energy * 1000, 2), n.carbon_base, n.priv_tier)
    claims.append(("CR Table 2", f"row {nid} ({n.name})", row, got,
                   all(abs(a - b) < 1e-9 if isinstance(a, (int, float)) else a == b
                       for a, b in zip(row, got))))
same("CR Sect. 5.1", "DAG demands 8, 16, 30, 25, 20, 18",
     [8, 16, 30, 25, 20, 18], [s.demand for s in CONT_SIM.dag])
same("CR Sect. 5.1", "PII-tagged stages auth and pay", ["auth", "pay"],
     [s.name for s in CONT_SIM.dag if s.pii])
same("CR Sect. 5.1", "intensity amplitude range 60-120 gCO2/kWh", [60, 120],
     [min(n.carbon_amp for n in CONT_SIM.nodes), max(n.carbon_amp for n in CONT_SIM.nodes)])
num("CR Sect. 5.1", "diurnal cycle of 240 steps", 240, SIM.DAY, 0)
_a = candidate_set(CONT_SIM, cap=40, seed=0)
_b = candidate_set(CONT_SIM, cap=40, seed=1)
_n = next(k for k in range(40) if tuple(_a[k]) != tuple(_b[k]))
same("CR Sect. 5.1", "menu: 22 structured and 18 random placements", [22, 18], [_n, 40 - _n])
same("CR Sect. 5", "WS-default weights", [0.30, 0.12, 0.30, 0.28], [float(x) for x in DEFAULT_WS_W])
same("CR Sect. 5", "WS-tuned weights", [0.85, 0.10, 0.00, 0.05], [float(x) for x in TUNED_WS_W])
true("CR Sect. 5", "WS-tuned beats the best point of the 84-point sweep on SLO",
     E3P[1.0]["ws_slo"] < C["e2_weight_sweep"]["best_ws"]["slo"],
     f"{E3P[1.0]['ws_slo']} < {C['e2_weight_sweep']['best_ws']['slo']} (8 and 6 seeds)")
same("CR Sect. 5", "Constr-WS equal-weight sum over privacy, carbon, cost",
     [0.34, 0.33, 0.33], [float(x) for x in ConstrainedWeightedSum.W_REST])

# Table 3
TABLE3 = {  # method: SLO %, PII, carbon, cost, churn, Inv %
    "Lexi-strict": (1.46, 0.05, 0.98, 3.21, 1.35, 0.0),
    "constrained_rl": (1.32, 0.05, 0.98, 3.19, 1.32, 0.7),
    "ws_tuned": (3.19, 0.01, 0.99, 3.20, 1.28, 3.7),
    "ws_default": (23.02, 1.94, 0.45, 1.22, 0.65, 94.7),
    "rank_ws": (3.48, 0.00, 0.99, 3.22, 1.29, 4.2),
    "norm_ws": (3.30, 0.01, 0.98, 3.20, 1.29, 3.6),
    "tcheby": (4.87, 0.01, 0.97, 3.20, 1.28, 5.0),
    "single_obj_rl": (11.53, 1.84, 1.22, 2.70, 1.99, None),
}


def table3_row(key, loc):
    paper = TABLE3[key]
    got = [mean(key, m) for m in ("slo", "privacy", "carbon", "cost", "churn")]
    got.append(INV[key]["inversion_pct"] if key in INV else None)
    tols = (0.005, 0.005, 0.005, 0.005, 0.005, 0.05)
    ok = all((p is None and g is None) or abs(p - g) <= t + 1e-9
             for p, g, t in zip(paper, got, tols))
    claims.append((loc, f"Table 3 row {key}", paper, tuple(got), ok))


for k in ("Lexi-strict", "constrained_rl", "ws_tuned", "ws_default"):
    table3_row(k, "CR Table 3")

# Fig. 2
num("CR Fig. 2(a)", "weightings drawn above 25% (27 more)", 27,
    C["e2_weight_sweep"]["n_weights"] - len(C["e2_weight_sweep"]["pareto"]), 0)
same("CR Fig. 2(b)", "factor grid shared by Lexi, WS-tuned and Constr-WS",
     [0.1, 0.3, 1.0, 3.0, 10.0, 30.0], sorted(set(E3P) & set(BCR)))
num("CR Fig. 2(c)", "contention seeds", 15, C["e1_satisfaction"]["n_seeds"], 0)
same("CR Fig. 2(c)", "contention levels 0 to 0.6", [0.0, 0.1, 0.2, 0.4, 0.6], sorted(CONT))

# Sect. 6.1
true("CR Sect. 6.1", "no weighting dominates Lexi", C["e2_weight_sweep"]["frac_dominating_lexi"] == 0)
num("CR Sect. 6.1", "Lexi SLO in the 6-seed sweep 1.42%", 1.42, C["e2_weight_sweep"]["lexi_strict"]["slo"], 0.005)
nums("CR Sect. 6.1", "Lexi SLO 95% CI [1.31, 1.60]", [1.31, 1.60], T["Lexi-strict"]["slo"]["ci"], 0.005)
true("CR Sect. 6.1", "Lexi better on SLO than WS-tuned (paired p < 0.001)",
     C["e1_satisfaction"]["paired_tests"]["lexi_vs_wstuned_slo"]["p"] < 0.001
     and mean("Lexi-strict", "slo") < mean("ws_tuned", "slo"))
true("CR Sect. 6.1", "WS-tuned better on privacy", mean("ws_tuned", "privacy") < mean("Lexi-strict", "privacy"))
ratios = [mean(k, "slo") / mean("Lexi-strict", "slo") for k in ("rank_ws", "norm_ws", "tcheby")]
nums("CR Sect. 6.1, EXT Sect. 9.2", "scale-robust sums at 2.3-3.3x Lexi's SLO",
     [2.3, 3.3], [min(ratios), max(ratios)], 0.05)
_others = ("constrained_rl", "ws_tuned", "rank_ws", "norm_ws", "tcheby")
nums("CR Sect. 6.1, EXT Sect. 9.2", "inversion 0.0% for Lexi against 0.7-5.0% for the others",
     [0.0, 0.7, 5.0], [INV["Lexi-strict"]["inversion_pct"],
                       min(INV[k]["inversion_pct"] for k in _others),
                       max(INV[k]["inversion_pct"] for k in _others)], 0.05)

# Sect. 6.2
num("CR Sect. 6.2", "Lexi PII invariant at 0.052", 0.052, E3P[1.0]["lexi_priv"], 0.0005)
num("CR Sect. 6.2", "300x swing (factor 0.1 to 30)", 300, max(E3P) / min(E3P), 1e-6)
num("CR Sect. 6.2", "WS-tuned PII at nominal scale 0.01", 0.01, E3P[1.0]["ws_priv"], 0.005)
num("CR Sect. 6.2", "Tchebycheff PII at privacy x0.1: 1.86", 1.86, E3P[0.1]["tcheby_priv"], 0.005)
num("CR Sect. 6.2", "Rank-WS SLO about 3.5% (8 seeds)", 3.5, E3P[1.0]["rank_ws_slo"], 0.1)
num("CR Sect. 6.2", "Norm-WS SLO about 3.5% (8 seeds)", 3.5, E3P[1.0]["norm_ws_slo"], 0.1)
num("CR Sect. 6.2", "Constr-WS SLO 1.32%", 1.32, mean("constrained_rl", "slo"), 0.005)
true("CR Sect. 6.2", "identical PII and carbon (2 d.p.), cost within 0.02",
     round(mean("constrained_rl", "privacy"), 2) == round(mean("Lexi-strict", "privacy"), 2)
     and round(mean("constrained_rl", "carbon"), 2) == round(mean("Lexi-strict", "carbon"), 2)
     and abs(mean("constrained_rl", "cost") - mean("Lexi-strict", "cost")) <= 0.02)
num("CR Sect. 6.2", "Constr-WS inverts on 0.68% of decisions", 0.68, INV["constrained_rl"]["inversion_pct"], 0.005)
num("CR Sect. 6.2", "Constr-WS inverts on 14.3% under real Azure load", 14.3,
    RD["inversion"]["constrained_rl"]["inversion_pct"], 0.05)
same("CR Sect. 6.2", "Lexi inverts on 0% in both", [0.0, 0.0],
     [INV["Lexi-strict"]["inversion_pct"], RD["inversion"]["Lexi-strict"]["inversion_pct"]])
num("CR Sect. 6.2", "Constr-WS PII at nominal scale 0.052", 0.052, BCR[1.0]["constr_rl_priv"], 0.0005)

# Sect. 6.3
num("CR Sect. 6.3", "Lexi SLO on the curated menu 1.44% (8 seeds)", 1.44, E5["curated"]["Lexi-strict"]["slo"], 0.005)
num("CR Sect. 6.3", "Lexi SLO on the random menu 43.6%", 43.6, E5["random"]["Lexi-strict"]["slo"], 0.05)
for key, val, desc in (("pct_cand_slo_feasible", 13.7, "SLO-feasible"),
                       ("pct_cand_pii_clean", 45.0, "PII-clean"), ("pct_cand_both", 5.6, "both")):
    num("CR Sect. 6.3", f"curated menu share {desc} {val}%", val, MENU["curated"][key], 0.05)
for key, val, desc in (("pct_cand_slo_feasible", 3.9, "SLO-feasible"),
                       ("pct_cand_pii_clean", 19.4, "PII-clean"), ("pct_cand_both", 0.2, "both")):
    num("CR Sect. 6.3", f"random menu share {desc} {val}%", val, MENU["random"][key], 0.05)
num("CR Sect. 6.3", "oracle SLO on the curated menu 1.48%", 1.48, MENU["curated"]["oracle_slo_pct"], 0.005)
num("CR Sect. 6.3", "Lexi SLO on the curated menu 1.44% (oracle run)", 1.44, MENU["curated"]["lexi_slo_pct"], 0.005)
num("CR Sect. 6.3", "oracle SLO on the random menu 43.53%", 43.53, MENU["random"]["oracle_slo_pct"], 0.005)
num("CR Sect. 6.3", "Lexi SLO on the random menu 43.56%", 43.56, MENU["random"]["lexi_slo_pct"], 0.005)
num("CR Sect. 6.3", "WS-tuned SLO on the random menu 45.8%", 45.8, E5["random"]["ws_tuned"]["slo"], 0.05)
num("CR Sect. 6.3", "WS-default SLO on the random menu 59.8%", 59.8, E5["random"]["ws_default"]["slo"], 0.05)
true("CR Sect. 6.3", "ordering Lexi <= WS-tuned <= WS-default over SLO 220-340 ms",
     all(SWEEP[s]["Lexi-strict"]["slo"] <= SWEEP[s]["ws_tuned"]["slo"] <= SWEEP[s]["ws_default"]["slo"]
         for s in SWEEP) and sorted(SWEEP) == [220.0, 260.0, 300.0, 340.0])
for key, val in (("Lexi-strict", 6.78), ("constrained_rl", 7.17), ("ws_tuned", 8.14), ("ws_default", 13.81)):
    num("CR Sect. 6.3", f"config B {key} SLO {val}%", val, CB[key]["slo"], 0.005)
num("CR Sect. 6.3", "config B description: 10 nodes, 8 stages, 3 PII",
    1, int(C["e6_generalization"]["config_b_desc"].startswith("10 nodes") and "8-stage" in
           C["e6_generalization"]["config_b_desc"] and "3 PII" in C["e6_generalization"]["config_b_desc"]), 0)
num("CR Sect. 6.3", "contention: Lexi 1.46% at c = 0", 1.46, CONT[0.0]["Lexi-strict"], 0.005)
num("CR Sect. 6.3", "contention: Lexi 40.97% at c = 0.6", 40.97, CONT[0.6]["Lexi-strict"], 0.005)
true("CR Sect. 6.3", "contention keeps Lexi and Constr-WS ahead of WS-tuned at every level",
     all(max(r["Lexi-strict"], r["constrained_rl"]) < r["ws_tuned"] for r in CONT.values()))
num("CR Sect. 6.3", "contention: Lexi PII 0.05 at c = 0", 0.05, RX1[0.0]["additive_pii"], 0.005)
num("CR Sect. 6.3", "contention: Lexi PII 1.49 at c = 0.6", 1.49, RX1[0.6]["additive_pii"], 0.005)
num("CR Sect. 6.3", "SLO-hinge MAE 0.0286 (normalised)", 0.0286, PA["slo_hinge_mae_norm"], 0.00005)
num("CR Sect. 6.3", "Lexi picks the expected-objective optimum on 99.7% of decisions", 99.7,
    OA["curated"]["lexi_pct"], 0.05)
infos.append(("CR Sect. 6.3", "decision time under 0.1 ms at 40 candidates (machine-dependent)",
              [p["ms_median"] for p in C["scalability"]["points"] if p["n_cand"] in (32, 48)]))
# Corrected in this revision: Sect. 5 now reports Lex-LP, the rule without Lexi's stability
# hold, and its agreement with Lexi on all 24,000 main-run decisions.
num("CR Sect. 5", "Lex-LP matches Lexi on all decisions (agreement 1.00)", 1.0,
    C["lexlp_agreement"]["agreement_pure"], 0)
num("CR Sect. 5", "decisions compared: 24,000", 24000, C["lexlp_agreement"]["n_steps"], 0)
num("CR Sect. 5, Def. 2", "stability hold changes 0 of the 24,000 main-run strict decisions", 0,
    HY["main"]["changed_steps"], 0)

# ---------------------------------------------------------------- extended version (EXT)
# Corrected in this revision: the intensity phase is set per node.
true("EXT Sect. 8.1", "per-node phase (nodes of one region have different phases)",
     any(len({n.carbon_phase for n in CONT_SIM.nodes if n.region == r}) > 1 for r in (0, 1, 2)))
num("EXT Sect. 3.2", "divisors 2 stages, 2.5 g, 5 cost units", 1,
    int(list(SIM.SCALES[1:]) == [2.0, 2.5, 5.0]), 0)
same("EXT Sect. 3.3", "candidate #18 estimate (0.076, 0, 0.673, 0.888)",
     [0.076, 0.0, 0.673, 0.888], WD["cand18_pred_norm"])
num("EXT Sect. 3.3", "#18 predicted 4.5 ms over the SLO", 4.5, WD["cand18_pred_latency_ms"] - 260, 0.05)
num("EXT Sect. 3.3", "#6 predicted 7 ms faster than #21", 7,
    PANEL[21]["pred_lat_ms"] - PANEL[6]["pred_lat_ms"], 0.5)
num("EXT Sect. 3.3", "#21 was the previous placement", 21, WD["incumbent"], 0)
num("EXT Sect. 4.1", "time-of-day fraction 0.767", 0.767, D["request"]["time_of_day_frac"], 0.0005)
same("EXT Sect. 4.3", "admitted candidates 24 and 33 (true 268.2 and 264.8 ms)", [268.2, 264.8],
     [PANEL[24]["true_lat_ms"], PANEL[33]["true_lat_ms"]])
# Corrected in this revision: the four values are now the predicted carbon.
nums("EXT Sect. 4.3", "predicted carbon of {2, 19, 20, 21}: 1.683, 1.190, 0.968, 0.784 g",
     [1.683, 1.190, 0.968, 0.784], [WD["level1_pred_carbon_g"][str(i)] for i in (2, 19, 20, 21)], 0.0005)
true("EXT Sect. 4.3", "#21 lowest predicted carbon of the four",
     min(WD["level1_pred_carbon_g"], key=WD["level1_pred_carbon_g"].get) == "21")
num("EXT Sect. 4.3", "candidate 21 cost $2.96", 2.96, PANEL[21]["true_raw"][3], 0.005)
same("EXT Sect. 4.3, 4.5", "#21 noise-free objectives: 235.5 ms, PII 0, carbon 0.785 g",
     [235.5, 0.0, 0.785], [PANEL[21]["true_lat_ms"], PANEL[21]["true_raw"][1], PANEL[21]["true_raw"][2]])
same("EXT Sect. 4.5", "#6: latency 228.8 ms, carbon 0.416 g, cost $1.08",
     [228.8, 0.416, 1.08], [PANEL[6]["true_lat_ms"], PANEL[6]["true_raw"][2], PANEL[6]["true_raw"][3]])
for i in TABLE1:
    table1_row(i, "EXT Table 1")
for k in ("Lexi-strict", "constrained_rl", "ws_tuned", "rank_ws", "norm_ws", "tcheby", "ws_default", "single_obj_rl"):
    table3_row(k, "EXT Table 3")
nums("EXT Table 3", "static-greedy / binpack SLO 67.2 / 67.8",
     [67.2, 67.8], [mean("static_greedy", "slo"), mean("binpack", "slo")], 0.05)
nums("EXT Table 3", "static-greedy / binpack carbon 1.95 / 0.36, cost 3.90 / 1.08, churn 0.00",
     [1.95, 0.36, 3.90, 1.08, 0.0, 0.0],
     [mean("static_greedy", "carbon"), mean("binpack", "carbon"), mean("static_greedy", "cost"),
      mean("binpack", "cost"), mean("static_greedy", "churn"), mean("binpack", "churn")], 0.005)
nums("EXT Table 3", "static-greedy / binpack PII 2.00", [2.0, 2.0],
     [mean("static_greedy", "privacy"), mean("binpack", "privacy")], 0.005)
num("EXT Sect. 9.2", "Lex-LP agreement 1.00 over 24k decisions", 1.0, C["lexlp_agreement"]["agreement_pure"], 0)
num("EXT Sect. 9.2", "decisions compared by Lex-LP", 24000, C["lexlp_agreement"]["n_steps"], 0)
TABLE4 = {"lexi": [0.052] * 6, "crl": [1.837, 1.551, 0.052, 0.052, 0.052, 0.052],
          "ws": [1.848, 0.042, 0.012, 0.000, 0.000, 0.000], "tch": [1.855, 0.312, 0.007, 0.000, 0.000, 0.000]}
fs = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
nums("EXT Table 4", "Lexi row", TABLE4["lexi"], [E3P[f]["lexi_priv"] for f in fs], 0.0005)
nums("EXT Table 4", "Constr-WS row", TABLE4["crl"], [BCR[f]["constr_rl_priv"] for f in fs], 0.0005)
nums("EXT Table 4", "WS-tuned row", TABLE4["ws"], [E3P[f]["ws_priv"] for f in fs], 0.0005)
nums("EXT Table 4", "Tchebycheff row", TABLE4["tch"], [E3P[f]["tcheby_priv"] for f in fs], 0.0005)
nums("EXT Table 5", "curated row", [13.7, 45.0, 5.6, 1.48, 0.052, 1.44, 0.052, 0.0],
     [MENU["curated"][k] for k in ("pct_cand_slo_feasible", "pct_cand_pii_clean", "pct_cand_both",
                                   "oracle_slo_pct", "oracle_pii", "lexi_slo_pct", "lexi_pii",
                                   "lexi_mean_gap_obj0_norm")], 0.05)
nums("EXT Table 5", "random row", [3.9, 19.4, 0.2, 43.53, 1.461, 43.56, 1.450, 3e-05],
     [MENU["random"][k] for k in ("pct_cand_slo_feasible", "pct_cand_pii_clean", "pct_cand_both",
                                  "oracle_slo_pct", "oracle_pii", "lexi_slo_pct", "lexi_pii",
                                  "lexi_mean_gap_obj0_norm")], 0.05)
TABLE6 = {0.0: (1.46, 1.32, 3.19, 0.05, None), 0.1: (5.79, 5.77, 8.43, 0.18, 57),
          0.2: (9.50, 9.46, 18.43, 0.65, 19), 0.4: (21.57, 20.97, 26.29, 1.07, 12),
          0.6: (40.97, 45.09, 52.18, 1.49, 2)}
for c, row in TABLE6.items():
    got = (CONT[c]["Lexi-strict"], CONT[c]["constrained_rl"], CONT[c]["ws_tuned"],
           RX1[c]["additive_pii"], None if c == 0 else 100 * RX1[c]["recovery_frac"])
    ok = all((p is None and g is None) or abs(p - g) <= (0.5 if j == 4 else 0.005) + 1e-9
             for j, (p, g) in enumerate(zip(row, got)))
    claims.append(("EXT Table 6", f"row c = {c}", row, got, ok))
TABLE7 = {0: (0, 4, 4, 4, 5, 0, 3), 15: (0, 23, 22, 23, 24, 9, 12), 30: (0, 28, 29, 27, 37, 22, 34),
          45: (0, 24, 24, 24, 35, 32, 36), 60: (0, 24, 24, 24, 35, 32, 38)}
for cp, row in TABLE7.items():
    iv, sv = COLD[cp]["inversion"], COLD[cp]["slo_viol"]
    got = tuple(iv[k]["mean"] for k in ("Lexi-strict", "ws_tuned", "rank_ws", "norm_ws", "tcheby")) + \
        (sv["Lexi-margin"]["mean"], sv["ws_tuned"]["mean"])
    claims.append(("EXT Table 7", f"row cold start {cp} ms", row, got,
                   all(abs(p - g) <= 0.5 + 1e-9 for p, g in zip(row, got))))
TABLE8 = {"Lexi-strict": (2.02, 0.020, 0.432, 3.634, 1.576, 0.0), "constrained_rl": (1.43, 0.020, 0.449, 3.350, 1.421, 14.3),
          "ws_tuned": (2.22, 0.002, 0.453, 3.355, 1.345, 16.3), "rank_ws": (1.86, 0.000, 0.453, 3.366, 1.356, 16.4),
          "norm_ws": (2.43, 0.002, 0.453, 3.351, 1.344, 16.4), "tcheby": (3.82, 0.001, 0.454, 3.340, 1.378, 16.8),
          "ws_default": (24.90, 1.218, 0.359, 1.903, 0.956, None)}
for k, row in TABLE8.items():
    t = RD["table"][k]
    got = tuple(t[m]["mean"] for m in ("slo", "privacy", "carbon", "cost", "churn")) + \
        ((RD["inversion"][k]["inversion_pct"],) if k in RD["inversion"] else (None,))
    tols = (0.005, 0.0005, 0.0005, 0.0005, 0.0005, 0.05)
    claims.append(("EXT Table 8", f"row {k}", row, got,
                   all((p is None and g is None) or abs(p - g) <= tl + 1e-9 for p, g, tl in zip(row, got, tols))))
num("EXT Sect. 11", "paired p Lexi vs WS-tuned 0.15", 0.15, RD["paired_tests"]["lexi_vs_wstuned_slo"]["p"], 0.005)
same("EXT Sect. 8.2, 11", "regional carbon means 92.8, 167.5, 182.9", [92.8, 167.5, 182.9],
     RD["provenance"]["carbon_mean_gco2kwh"])
auto2 = [a for a in ROB["rx4_hinge_margin"]["auto_margin"] if a["k"] == 2.0][0]
comp = {r["filter"]: r for r in ROB["rx7_compliance"]["rows"]}
q1 = DEP["r2q1_safe_probe"]
q2 = {r["drift"]: r for r in DEP["r2q2_nonstationarity"]["rows"]}
q4 = DEP["r2q4_candidate_gen"]
q5 = DEP["r2q5_contention_lowpri"]["rows"][-1]
num("EXT Table 9", "M/G/1 term recovers 57% at light contention", 57, 100 * RX1[0.1]["recovery_frac"], 0.5)
# Corrected in this revision: 15 seeds unless marked, dagger rows use 2 to 5 seeds.
same("EXT Table 9", "15-seed rows (RX1, RX2, RX4, RX6, RX7, DX1-DX7)", [15] * 12,
     [ROB[k]["n_seeds"] for k in ("rx1_contention", "rx2_doubly_robust", "rx4_hinge_margin",
                                 "rx6_priority_orders", "rx7_compliance")]
     + [DEP[k]["n_seeds"] for k in ("r2q1_safe_probe", "r2q2_nonstationarity", "r2q4_candidate_gen",
                                   "r2q5_contention_lowpri", "r2q6_delta_slack", "r2q7_quantile")]
     + [DEP["_meta"]["n_seeds"]])
same("EXT Table 9", "dagger rows: scaling 5 seeds, larger sweep 2 and 3 seeds", [5, 2, 3],
     [ROB["rx5_scaling"]["n_seeds"], ROB["rx3_expanded_sweeps"]["n_seeds_sweep"],
      ROB["rx3_expanded_sweeps"]["n_seeds_baseline"]])
true("EXT Table 9", "DR decision-safe, error reduction <= 0.5%",
     all(r["dm_slo"] == r["dr_slo"] and r["dm_pii"] == r["dr_pii"] for r in ROB["rx2_doubly_robust"]["rows"])
     and max(r["esterr_reduction_pct"] for r in ROB["rx2_doubly_robust"]["rows"]) < 0.55)
true("EXT Table 9", "hinge rho 30-240: SLO 1.46%, PII 0.05 invariant",
     {(r["slo"], r["privacy"]) for r in ROB["rx4_hinge_margin"]["hinge_scale_invariance"]} == {(1.458, 0.054)})
nums("EXT Table 9", "auto margin k = 2: delta 3.7 ms, SLO 0.23%", [3.7, 0.23],
     [auto2["mean_delta_ms"], auto2["slo"]], 0.005)
infos.append(("EXT Table 9", "scaling K <= 30, C <= 2000: at most 0.75 ms per decision (machine-dependent)",
              max(r["act_ms_median"] for r in ROB["rx5_scaling"]["by_C"])))
same("EXT Table 9", "220 weightings, 0% dominate", [220, 0.0],
     [ROB["rx3_expanded_sweeps"]["weight_sweep"]["n_weights"], ROB["rx3_expanded_sweeps"]["weight_sweep"]["frac_dominating_lexi"]])
nums("EXT Table 9", "residency violations 2.7% -> 0%", [2.7, 0.0],
     [comp[False]["residency_violation_pct"], comp[True]["residency_violation_pct"]], 0.05)
nums("EXT Table 9", "safe probe coverage 85.7 -> 100%, SLO 1.46 -> 1.30%", [85.7, 100.0, 1.46, 1.30],
     [q1["strict"]["coverage_pct"], q1["safeprobe"]["coverage_pct"], q1["strict"]["slo"], q1["safeprobe"]["slo"]], 0.05)
nums("EXT Table 9", "drift: static 37.2% vs online 28.9%", [37.2, 28.9],
     [q2[0.6]["static_slo"], q2[0.6]["online_slo"]], 0.05)
nums("EXT Table 9", "planner SLO 1.9%, recovers 99%", [1.9, 99.0],
     [q4["generators"]["planner"]["slo"], q4["planner_recovers_pct_of_curated"]], 0.05)
num("EXT Table 9", "contention c = 0.6: PII 1.49", 1.49, q5["additive"]["pii"], 0.005)
num("EXT Table 9", "quantile surrogate agreement 100%", 1.0, DEP["r2q7_quantile"]["decision_agreement"], 0)
g6 = {(r["delta_ms"], r["eta_priv"]): r for r in DEP["r2q6_delta_slack"]["grid"]}
nums("EXT Sect. 12", "delta 0/4/8 ms -> SLO 1.45/0.16/0.008%", [1.45, 0.16, 0.008],
     [g6[(0.0, 0.0)]["slo"], g6[(4.0, 0.0)]["slo"], g6[(8.0, 0.0)]["slo"]], 0.005)
nums("EXT Sect. 12", "delta 0/4/8 ms -> PII 0.054/0.114/0.278", [0.054, 0.114, 0.278],
     [g6[(d, 0.0)]["pii"] for d in (0.0, 4.0, 8.0)], 0.0005)
true("EXT Sect. 12", "privacy slack 0.5 breaks invariance, strict privacy keeps it",
     DEP["r2q6_delta_slack"]["scale_invariance"][0]["pii"] == DEP["r2q6_delta_slack"]["scale_invariance"][1]["pii"]
     and DEP["r2q6_delta_slack"]["scale_invariance"][2]["pii"] != DEP["r2q6_delta_slack"]["scale_invariance"][3]["pii"])
# Sect. 9.4 and 9.5
nums("EXT Sect. 9.4, 9.5", "optimum picked on 99.7% / 98.9% (Lexi), 99.1% / 87.6% (Constr-WS)",
     [99.7, 98.9, 99.1, 87.6],
     [OA["curated"]["lexi_pct"], OA["random"]["lexi_pct"],
      OA["curated"]["constr_ws_pct"], OA["random"]["constr_ws_pct"]], 0.05)
nums("EXT Sect. 9.4", "SLO sweep at 220 ms: 58.9 / 58.9 / 73.8 / 88.0",
     [58.9, 58.9, 73.8, 88.0], [SWEEP[220.0][k]["slo"] for k in ("Lexi-strict", "constrained_rl", "ws_tuned", "ws_default")], 0.05)
true("EXT Sect. 9.4", "at 340 ms every ranked method 0% and WS-default PII 2",
     all(SWEEP[340.0][k]["slo"] == 0 for k in ("Lexi-strict", "constrained_rl", "ws_tuned"))
     and SWEEP[340.0]["ws_default"]["privacy"] == 2.0)
nums("EXT Sect. 9.4", "config B PII: Lexi 0.00, WS-default 2.73", [0.0, 2.73],
     [CB["Lexi-strict"]["privacy"], CB["ws_default"]["privacy"]], 0.005)
nums("EXT Sect. 9.5", "latency MAE 10.5 ms, mean latency 363.8 ms, 2.9%", [10.5, 363.8, 2.9],
     [PA["latency_mae_ms"], PA["mean_true_latency_ms"], PA["latency_mape_pct"]], 0.05)
nums("EXT Sect. 9.5", "hinge MAE 0.0286 and 1.7 ms", [0.0286, 1.7],
     [PA["slo_hinge_mae_norm"], PA["slo_hinge_mae_ms"]], 0.05)
nums("EXT Sect. 9.5", "carbon MAE 0.053 g, cost MAE $0.16", [0.053, 0.16],
     [PA["carbon_mae_g"], PA["cost_mae_usd"]], 0.005)
nums("EXT Sect. 9.5", "fleet carbon range roughly 0.3-2.6 g (min, max over the menu)", [0.3, 2.6],
     [DC["menu_carbon_range"]["min_g"], DC["menu_carbon_range"]["max_g"]], 0.1)
num("EXT Sect. 9.5", "DR study mean error 0.0246", 0.0246, ROB["rx2_doubly_robust"]["rows"][0]["dm_esterr"], 0.00005)
num("EXT Sect. 9.6", "margin 4 ms: SLO 0.16%", 0.16, MG[4.0]["slo"], 0.005)
num("EXT Sect. 9.6", "margin 8 ms: SLO 0.01%", 0.01, MG[8.0]["slo"], 0.005)
nums("EXT Sect. 9.6", "margin PII 0.05 -> 0.11 -> 0.28", [0.05, 0.11, 0.28],
     [MG[d]["privacy"] for d in (0.0, 4.0, 8.0)], 0.005)
nums("EXT Sect. 9.6", "contention 0.4: margin cuts SLO 18.7% -> 14.6%", [18.7, 14.6],
     [MGC[0.0]["slo"], MGC[12.0]["slo"]], 0.05)
nums("EXT Sect. 5.3, 9.6", "churn 1.35 -> 1.14, SLO 1.46 -> 0.92% over slack 0 -> 8",
     [1.35, 1.14, 1.46, 0.92], [E4[0]["churn"], E4[-1]["churn"], E4[0]["slo"], E4[-1]["slo"]], 0.005)
infos.append(("EXT Sect. 9.6", "slope 1.12e-4 ms/cand, CI [1.05, 1.35]e-4, 0.082 -> 0.096 ms (machine-dependent)",
              (C["scalability"]["slope_ms_per_cand"], C["scalability"]["slope_ci"],
               C["scalability"]["points"][0]["ms_median"], C["scalability"]["points"][-1]["ms_median"])))
_priced = ("ws_tuned", "rank_ws", "norm_ws", "tcheby")
nums("EXT Sect. 9.2, 10", "cold start: priced methods invert 22-37% (15-60 ms), Lexi 0%",
     [22, 37, 0.0], [min(COLD[c]["inversion"][k]["mean"] for c in (15, 30, 45, 60) for k in _priced),
                     max(COLD[c]["inversion"][k]["mean"] for c in (15, 30, 45, 60) for k in _priced),
                     max(COLD[c]["inversion"]["Lexi-strict"]["mean"] for c in COLD)], 0.5)
nums("EXT Sect. 9.2", "30 ms cold start: margin variant 22% SLO, WS-tuned 34%", [22, 34],
     [COLD[30]["slo_viol"]["Lexi-margin"]["mean"], COLD[30]["slo_viol"]["ws_tuned"]["mean"]], 0.5)
true("EXT Sect. 11", "real load: priced scalarisers invert 16-17%",
     all(16 <= RD["inversion"][k]["inversion_pct"] < 17.5 for k in _priced),
     [RD["inversion"][k]["inversion_pct"] for k in _priced])
num("EXT Sect. 9.3", "Constr-WS inversion CI [0.55, 0.83]", 0.55, INV["constrained_rl"]["ci"][0], 0.005)
num("EXT Sect. 9.3", "Constr-WS inversion CI upper 0.83", 0.83, INV["constrained_rl"]["ci"][1], 0.005)

# ---------------------------------------------------------------- report
width = max(len(c[0]) for c in claims)
ok_all = True
for loc, desc, paper, got, ok in claims:
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {loc.ljust(width)}  {desc}: paper {paper}, artifact {got}")
for loc, desc, got in infos:
    print(f"  [INFO] {loc.ljust(width)}  {desc}: artifact {got}")
n_ok = sum(c[4] for c in claims)
n_cr = sum(1 for c in claims if c[0].startswith("CR"))
n_ext = len(claims) - n_cr
print(f"{n_ok}/{len(claims)} paper claims match ({n_cr} camera-ready, {n_ext} extended). "
      f"{len(infos)} machine-dependent timing values printed as INFO.")
raise SystemExit(0 if ok_all else 1)
