#!/usr/bin/env python3
"""
Print every results table of the two papers, and the node table, from the committed
result files and the simulator configuration. No experiment is re-run.

  camera-ready  Fig. 1 counts, Table 1 (worked decision), Table 2 (nodes of config A),
                Table 3 (RQ2 outcomes).
  extended      Table 1 (all eleven level-0 survivors), Table 3 (all methods), Tables 4 to
                9. Table 10 of the extended version is a qualitative comparison and has no
                data behind it. Tables 2 and the camera-ready Fig. 1 counts are shared.

Values are rounded half up on the stored value. Where a stored value ends in exactly 5
the papers sometimes round down (23.025 is printed as 23.02 in Table 3), which is within
half a unit of the last digit. Node ids follow the papers: a tier
letter (E edge, C cloud, S serverless) followed by the region number. For example edge_A
is E0, edge_C is E2, cloud_C is C2, srvl_B is S1 and srvl_C is S2.

Usage:
    python3 make_tables.py              # both papers
    python3 make_tables.py camera       # camera-ready only
    python3 make_tables.py extended     # extended version only
"""
import json, os, sys
from decimal import Decimal, ROUND_HALF_UP

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sim import Continuum  # noqa: E402

RES = os.path.join(_HERE, "..", "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


CORE = load("core_results.json")
EXT = load("extended_results.json")
REAL = load("real_data_results.json")
ROB = load("robustness_results.json")
DEP = load("deployment_results.json")

SIM = Continuum(seed=0)
TIER = {"edge": "E", "cloud": "C", "serverless": "S"}
PID = {n.name: f"{TIER[n.tier]}{n.region}" for n in SIM.nodes}


def fx(x, nd):
    """Round half up on the stored decimal value, the way the papers round (3.65 -> 3.7)."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def rule(title):
    print()
    print(title)
    print("-" * len(title))


def fig1_counts():
    d = EXT["D_worked_example"]
    counts = [d["band_trace"][0]["n_before"]] + [b["n_after"] for b in d["band_trace"]]
    counts.append(1)  # the cost level receives one candidate and returns it
    rule("Camera-ready and extended Fig. 1: survivors per level at step 904 (seed 0)")
    print("  menu -> SLO -> privacy -> carbon -> cost : " + " -> ".join(map(str, counts)))


def table1(rows=None):
    d = EXT["D_worked_example"]
    slo = d["slo_ms"]
    lvl1 = set(d["band_trace"][1]["survivors"])
    lvl2 = set(d["band_trace"][2]["survivors"])
    panel = [p for p in d["panel"] if p["idx"] in d["band_trace"][0]["survivors"]]
    if rows is not None:
        panel = [p for p in panel if p["idx"] in rows]
    print(f"  {'#':>4} {'Placement':20} {'pred.':>6} {'true':>6} {'g0':>3} {'g1':>4} "
          f"{'g2':>6} {'g3':>6}  Elim.")
    for p in panel:
        i = p["idx"]
        dag = "+" if (p["pred_norm"][0] == 0 and p["true_lat_ms"] > slo) else " "
        elim = "-" if i == d["chosen"] else ("L2" if i in lvl1 else "L1")
        if i in lvl2 and i != d["chosen"]:
            elim = "L3"
        place = "/".join(PID[n] for n in p["placement"])
        g = p["pred_norm"]
        g0 = "0" if g[0] == 0 else f"{g[0]:.3f}"
        bold = "*" if i == d["chosen"] else " "
        print(f"  {i:>3}{dag}{bold}{place:20} {p['pred_lat_ms']:6.1f} {p['true_lat_ms']:6.1f} "
              f"{g0:>3} {g[1]:4.1f} {g[2]:6.3f} {g[3]:6.3f}  {elim}")
    n_true = sum(1 for p in d["panel"] if p["idx"] in d["band_trace"][0]["survivors"]
                 and p["true_lat_ms"] <= slo)
    print(f"  (* chosen, + admitted at level 0 by prediction error. {n_true} of "
          f"{len(d['band_trace'][0]['survivors'])} level-0 survivors truly meet the "
          f"{slo:.0f} ms SLO)")


def table2():
    rule("Camera-ready and extended Table 2: nodes of config A (sim.build_nodes)")
    print(f"  {'Id':3} {'Tier':11} {'Reg':>3} {'b':>5} {'RTTin':>5} {'RTTout':>6} {'q':>4} "
          f"{'Cost':>5} {'Energy':>6} {'Ibar':>5} Priv  (code name)")
    for n in SIM.nodes:
        print(f"  {PID[n.name]:3} {n.tier:11} {n.region:>3} {n.base_compute:5.2f} "
              f"{n.rtt_intra:5.0f} {n.rtt_inter:6.0f} {n.q_sens:4.1f} {n.cost:5.2f} "
              f"{n.energy * 1000:6.2f} {n.carbon_base:5.0f} {'yes' if n.priv_tier else '   '}"
              f"   ({n.name})")
    print("  units: b ms/unit demand, RTT ms, q ms/load, cost simulator units per invocation,")
    print("  energy Wh for a stage of demand 20, Ibar gCO2/kWh")


METHODS = [("Lexi-strict", "Lexi (no weights)"), ("constrained_rl", "Constr-WS"),
           ("ws_tuned", "WS-tuned"), ("rank_ws", "Rank-WS"), ("norm_ws", "Norm-WS"),
           ("tcheby", "Tchebycheff"), ("ws_default", "WS-default"),
           ("single_obj_rl", "SO-RL"), ("static_greedy", "static-greedy"),
           ("binpack", "binpack")]


def table3(keys):
    t = CORE["e1_satisfaction"]["table"]
    inv = CORE["e1_inversion"]["inversion"]
    print(f"  {'Method':18} {'SLO%':>6} {'PII':>5} {'Carbon':>6} {'Cost':>5} {'Churn':>5} "
          f"{'Inv%':>5}")
    for k, name in METHODS:
        if k not in keys:
            continue
        r = t[k]
        iv = fx(inv[k]["inversion_pct"], 1) if k in inv else "n/a"
        print(f"  {name:18} {fx(r['slo']['mean'], 2):>6} {fx(r['privacy']['mean'], 2):>5} "
              f"{fx(r['carbon']['mean'], 2):>6} {fx(r['cost']['mean'], 2):>5} "
              f"{fx(r['churn']['mean'], 2):>5} {iv:>5}")
    ci = t["Lexi-strict"]["slo"]["ci"]
    print(f"  (Lexi SLO 95% CI [{ci[0]:.2f}, {ci[1]:.2f}] over 15 seeds, paired p vs WS-tuned "
          f"{CORE['e1_satisfaction']['paired_tests']['lexi_vs_wstuned_slo']['p']})")


def camera():
    fig1_counts()
    rule("Camera-ready Table 1: six of the eleven level-0 survivors at step 904 (seed 0)")
    table1(rows={2, 6, 19, 20, 21, 24})
    table2()
    rule("Camera-ready Table 3: RQ2 outcomes (15 seeds, means)")
    table3({"Lexi-strict", "constrained_rl", "ws_tuned", "ws_default"})


def extended():
    rule("Extended Table 1: all eleven level-0 survivors at step 904 (seed 0)")
    table1()
    rule("Extended Table 3: RQ2 outcomes, all methods (15 seeds, means)")
    table3({k for k, _ in METHODS})

    rule("Extended Table 4: PII exposure under a rescaled privacy view (8 seeds)")
    rows = CORE["e3_scale_invariance"]["objectives"]["privacy"]
    crl = {r["factor"]: r["constr_rl_priv"] for r in EXT["B_constrrl_scale"]["rows"]}
    print("  factor      " + " ".join(f"{r['factor']:>6}" for r in rows))
    for name, fn in (("Lexi", lambda r: r["lexi_priv"]),
                     ("Constr-WS", lambda r: crl[r["factor"]]),
                     ("WS-tuned", lambda r: r["ws_priv"]),
                     ("Tchebycheff", lambda r: r["tcheby_priv"])):
        print(f"  {name:11} " + " ".join(f"{fn(r):6.3f}" for r in rows))

    rule("Extended Table 5: candidate-menu anatomy (8 seeds)")
    print(f"  {'Menu':8} {'SLO-f%':>6} {'clean%':>6} {'both%':>5} {'orSLO%':>6} {'orPII':>6} "
          f"{'LxSLO%':>6} {'LxPII':>6} {'gap':>8}")
    for m, r in EXT["A_menu"]["sets"].items():
        print(f"  {m:8} {r['pct_cand_slo_feasible']:6.1f} {r['pct_cand_pii_clean']:6.1f} "
              f"{r['pct_cand_both']:5.1f} {r['oracle_slo_pct']:6.2f} {r['oracle_pii']:6.3f} "
              f"{r['lexi_slo_pct']:6.2f} {r['lexi_pii']:6.3f} {r['lexi_mean_gap_obj0_norm']:8.0e}")

    rule("Extended Table 6: SLO violation (%) under co-location contention (15 seeds)")
    sc = CORE["e7_misspecification"]["structural_contention"]["slo_pct"]
    rx1 = {r["contention"]: r for r in ROB["rx1_contention"]["rows"]}
    print(f"  {'c':>4} {'Lexi':>6} {'Constr':>6} {'WS-t':>6} {'LexiPII':>7} {'M/G/1 rec.':>10}")
    for r in sc:
        c = r["contention"]
        rec = "-" if c == 0 else fx(100 * rx1[c]["recovery_frac"], 0) + "%"
        print(f"  {c:4.1f} {fx(r['Lexi-strict'], 2):>6} {fx(r['constrained_rl'], 2):>6} "
              f"{fx(r['ws_tuned'], 2):>6} {fx(rx1[c]['additive_pii'], 2):>7} {rec:>10}")

    rule("Extended Table 7: cold-start stress, inversion % and SLO % (15 seeds)")
    print(f"  {'cold ms':>7} {'Lexi':>4} {'WS-t':>4} {'rank':>4} {'norm':>4} {'Tch':>4}  mgn/WS SLO%")
    for lv in CORE["e11_conflict_stress"]["levels"]:
        iv, sv = lv["inversion"], lv["slo_viol"]
        print(f"  {lv['cold_penalty']:7.0f} {fx(iv['Lexi-strict']['mean'], 0):>4} "
              f"{fx(iv['ws_tuned']['mean'], 0):>4} {fx(iv['rank_ws']['mean'], 0):>4} "
              f"{fx(iv['norm_ws']['mean'], 0):>4} {fx(iv['tcheby']['mean'], 0):>4}  "
              f"{fx(sv['Lexi-margin']['mean'], 0)}/{fx(sv['ws_tuned']['mean'], 0)}")

    rule("Extended Table 8: RQ2 under real traces (15 seeds, means)")
    print(f"  {'Method':18} {'SLO%':>6} {'PII':>6} {'Carbon':>6} {'Cost':>6} {'Churn':>6} {'Inv%':>5}")
    for k, name in METHODS[:7]:
        r = REAL["table"][k]
        iv = fx(REAL["inversion"][k]["inversion_pct"], 1) if k in REAL["inversion"] else "n/a"
        print(f"  {name:18} {fx(r['slo']['mean'], 2):>6} {fx(r['privacy']['mean'], 3):>6} "
              f"{fx(r['carbon']['mean'], 3):>6} {fx(r['cost']['mean'], 3):>6} "
              f"{fx(r['churn']['mean'], 3):>6} {iv:>5}")
    print(f"  (paired p Lexi vs WS-tuned on SLO: {REAL['paired_tests']['lexi_vs_wstuned_slo']['p']})")

    rule("Extended Table 9: measured robustness and deployment extensions")
    rx2 = ROB["rx2_doubly_robust"]["rows"]
    rx4 = ROB["rx4_hinge_margin"]
    auto2 = [a for a in rx4["auto_margin"] if a["k"] == 2.0][0]
    rx5 = ROB["rx5_scaling"]
    sw = ROB["rx3_expanded_sweeps"]["weight_sweep"]
    comp = {r["filter"]: r for r in ROB["rx7_compliance"]["rows"]}
    q1 = DEP["r2q1_safe_probe"]
    q2 = {r["drift"]: r for r in DEP["r2q2_nonstationarity"]["rows"]}
    q4 = DEP["r2q4_candidate_gen"]
    q5 = DEP["r2q5_contention_lowpri"]["rows"][-1]
    q7 = DEP["r2q7_quantile"]
    lines = [
        ("Contention, learned M/G/1", f"{100 * rx1[0.1]['recovery_frac']:.0f}% SLO recovery at c=0.1"),
        ("DR estimator", f"max error reduction {max(r['esterr_reduction_pct'] for r in rx2):.2f}%, "
                         f"SLO/PII identical to DM: {all(r['dm_slo'] == r['dr_slo'] for r in rx2)}"),
        ("Hinge rho 30..240", "SLO " + "/".join(f"{r['slo']:.2f}" for r in rx4["hinge_scale_invariance"])
                              + ", PII " + "/".join(f"{r['privacy']:.3f}" for r in rx4["hinge_scale_invariance"])),
        ("Margin auto k=2", f"delta {auto2['mean_delta_ms']} ms, SLO {auto2['slo']}%"),
        ("Scaling (timing)", f"max {max(r['act_ms_median'] for r in rx5['by_C']):.2f} ms/decision "
                             f"at C={max(r['C'] for r in rx5['by_C'])}, {rx5['n_seeds']} seeds"),
        ("Larger sweep", f"{sw['n_weights']} weightings, {100 * sw['frac_dominating_lexi']:.0f}% dominate "
                         f"Lexi ({ROB['rx3_expanded_sweeps']['n_seeds_sweep']} seeds)"),
        ("Residency pre-filter", f"violations {comp[False]['residency_violation_pct']:.1f}% -> "
                                 f"{comp[True]['residency_violation_pct']:.0f}%"),
        ("Safe probing", f"coverage {q1['strict']['coverage_pct']:.1f} -> {q1['safeprobe']['coverage_pct']:.0f}%, "
                         f"SLO {q1['strict']['slo']:.2f} -> {q1['safeprobe']['slo']:.2f}%"),
        ("Nonstationarity 0.6", f"static {q2[0.6]['static_slo']:.1f}% vs online {q2[0.6]['online_slo']:.1f}%"),
        ("Candidate generation", f"planner SLO {q4['generators']['planner']['slo']:.1f}%, recovers "
                                 f"{q4['planner_recovers_pct_of_curated']:.0f}% of curated"),
        ("Contention c=0.6, PII", f"{q5['additive']['pii']:.2f}"),
        ("Quantile surrogate", f"{100 * q7['decision_agreement']:.0f}% agreement"),
    ]
    for a, b in lines:
        print(f"  {a:24} {b}")
    g = {(r["delta_ms"], r["eta_priv"]): r for r in DEP["r2q6_delta_slack"]["grid"]}
    print("  delta x slack grid (eta_priv=0): "
          + ", ".join(f"delta={d:.0f} ms SLO {g[(d, 0.0)]['slo']}% PII {g[(d, 0.0)]['pii']}"
                      for d in (0.0, 4.0, 8.0)))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "camera"):
        print("=== Camera-ready ===")
        camera()
    if which in ("all", "extended"):
        print("\n=== Extended version ===")
        if which == "extended":
            fig1_counts()
            table2()
        extended()
