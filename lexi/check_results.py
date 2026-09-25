#!/usr/bin/env python3
"""
Consistency gate between the committed results and the numbers quoted in the paper
("Lexi: Priority-First Service Orchestration").

The script reads ../results/core_results.json and, when present, real_data_results.json
from the same folder. It asserts claim by claim that each headline number of the paper is
present in the results within a stated tolerance. The claims follow the camera-ready
research questions. RQ1 is the weight sweep, RQ2 the priority satisfaction of Table 3
together with the inversion rates and the real-trace check, RQ3 the scale-invariance
study of Fig. 2(b), and RQ4 the menu, the second configuration and the contention of
Fig. 2(c). The RQ5 claims (safety margin, auto-slack, stability) are those of the
extended version. The script does not re-run the simulator. Use experiments_core.py for a
full run and run_smoke.sh for a fast subset.

Every quantity checked here is deterministic and reproducible bit for bit. The one
machine-specific value, the decision-latency slope, is printed for information and is
not a claim. The claims that come from extended_results.json (menu anatomy, Constr-WS
scale sweep, prediction accuracy, worked decision) are not checked here.

Usage:
    python3 check_results.py            # PASS or FAIL per claim, exit 0 only if all pass
    python3 check_results.py path.json  # check a specific core results file

Exit status 0 means that every claim matches. 1 means at least one mismatch or a missing file.
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "..", "results", "core_results.json")

_results = []  # list of (ok, label, detail), filled by check() and printed at the end


def check(label, ok, detail=""):
    _results.append((bool(ok), label, detail))


def close(actual, expected, tol, label, unit=""):
    """Record a claim that passes when |actual - expected| <= tol. The detail line shows
    the claimed and the measured value. A value that cannot be converted to float fails."""
    try:
        ok = abs(float(actual) - float(expected)) <= tol
    except (TypeError, ValueError):
        ok = False
    check(label, ok, f"claimed {expected}{unit}, measured {actual}{unit} (tol {tol})")


def _find_factor(rows, factor):
    """Row of a scale-study table with the given rescaling factor (KeyError if absent)."""
    for r in rows:
        if abs(float(r["factor"]) - factor) < 1e-9:
            return r
    raise KeyError(f"factor {factor} not found")


def main(path=DEFAULT_JSON):
    if not os.path.exists(path):
        print(f"FATAL: results file not found: {path}")
        print("Run `python3 experiments_core.py` first.")
        return 1
    with open(path) as f:
        R = json.load(f)

    # RQ2, operator-facing outcomes (SLO percent and PII exposure), 15 seeds. The four
    # main rows are Table 3 of the camera-ready. The remaining rows are the scale-robust
    # scalarisers and the heuristics quoted in the text and in the extended version.
    tbl = R["e1_satisfaction"]["table"]
    # Each entry is (method key, claimed SLO percent, claimed PII).
    table_claims = [
        ("Lexi-strict",    1.46, 0.05),
        ("constrained_rl", 1.32, 0.05),
        ("ws_tuned",       3.19, 0.01),
        ("rank_ws",        3.48, 0.00),
        ("norm_ws",        3.30, 0.01),
        ("tcheby",         4.87, 0.01),
        ("ws_default",    23.02, 1.94),
        ("single_obj_rl", 11.53, 1.84),
        ("static_greedy", 67.22, 2.00),
        ("binpack",       67.79, 2.00),
    ]
    for key, slo, pii in table_claims:
        close(tbl[key]["slo"]["mean"], slo, 0.02, f"RQ2 {key} SLO%", "%")
        close(tbl[key]["privacy"]["mean"], pii, 0.01, f"RQ2 {key} PII")
    # Thresholded Lexi keeps its slack on carbon only, which means its SLO stays close to that of
    # strict Lexi.
    close(tbl["Lexi"]["slo"]["mean"], 1.43, 0.05,
          "RQ2 thresholded Lexi SLO (carbon slack, off primary)", "%")

    # RQ2 priority inversions (Inv% of Table 3): Lexi never inverts, the scalarisers do.
    inv = R["e1_inversion"]["inversion"]
    close(inv["Lexi-strict"]["inversion_pct"], 0.0, 1e-6,
          "RQ2 Lexi priority-inversion rate = 0", "%")
    close(inv["constrained_rl"]["inversion_pct"], 0.68, 0.15,
          "RQ2 Constr-WS inverts the priority order", "%")
    close(inv["ws_tuned"]["inversion_pct"], 3.65, 0.30,
          "RQ2 WS-tuned inverts more than Constr-WS", "%")
    check("RQ2 untuned WS inverts almost always (>90%)",
          inv["ws_default"]["inversion_pct"] > 90.0,
          f"ws_default inv={inv['ws_default']['inversion_pct']}%")

    # RQ2 under conflict (E11, extended version): the gap widens with cold starts.
    if "e11_conflict_stress" in R:
        lv = {int(l["cold_penalty"]): l for l in R["e11_conflict_stress"]["levels"]}
        for cp in lv:  # Lexi has no inversions at any stress level
            close(lv[cp]["inversion"]["Lexi-strict"]["mean"], 0.0, 1e-6,
                  f"E11 Lexi inversion = 0 at cold-start {cp}ms", "%")
        hi = lv[30]["inversion"]  # at 30 ms every priced scalariser inverts often
        for m, lo in (("ws_tuned", 20.0), ("rank_ws", 20.0),
                      ("norm_ws", 20.0), ("tcheby", 30.0)):
            check(f"E11 {m} inverts >= {lo:.0f}% under cold-start stress (30ms)",
                  hi[m]["mean"] >= lo, f"{m} inv={hi[m]['mean']:.1f}%")
        check("E11 conflict widens the inversion gap >5x vs benign",
              hi["ws_tuned"]["mean"] > 5 * lv[0]["inversion"]["ws_tuned"]["mean"],
              f"0ms={lv[0]['inversion']['ws_tuned']['mean']:.1f}% -> "
              f"30ms={hi['ws_tuned']['mean']:.1f}%")

    # RQ2 paired bootstrap tests of Lexi-strict against WS-tuned.
    pt = R["e1_satisfaction"]["paired_tests"]
    check("RQ2 Lexi vs WS-tuned SLO p<0.001",
          pt["lexi_vs_wstuned_slo"]["p"] < 0.001,
          f"p={pt['lexi_vs_wstuned_slo']['p']}")
    close(pt["lexi_vs_wstuned_slo"]["median_diff"], -1.9, 0.05,
          "RQ2 Lexi vs WS-tuned SLO median diff", " pts")
    close(pt["lexi_vs_wstuned_privacy"]["median_diff"], 0.04, 0.01,
          "RQ2 Lexi vs WS-tuned PII median diff")

    # RQ2 Lex-LP coincides with strict Lexi on the metrics and on the decisions.
    gap = R["e1_satisfaction"]["lexlp_vs_strict_max_gap"]
    check("RQ2 lex-LP == strict Lexi (max per-metric gap = 0)",
          all(abs(v) < 1e-9 for v in gap.values()), f"max gaps {gap}")
    check("RQ2 lex-LP agreement = 1.00 over decisions",
          abs(R["lexlp_agreement"]["agreement_pure"] - 1.0) < 1e-9,
          f"agreement_pure={R['lexlp_agreement']['agreement_pure']}")

    # RQ1 weight sweep (Fig. 2a): about 1.2 percent of weightings meet the target and
    # none dominate Lexi.
    e2 = R["e2_weight_sweep"]
    check("RQ1 sweep is 84 weightings", e2["n_weights"] == 84,
          f"n_weights={e2['n_weights']}")
    close(e2["frac_meeting_target"] * 100.0, 1.2, 0.2,
          "RQ1 fraction meeting priority target", "%")
    check("RQ1 no weighting dominates Lexi",
          abs(e2["frac_dominating_lexi"]) < 1e-9,
          f"frac_dominating_lexi={e2['frac_dominating_lexi']}")

    # RQ3 scale invariance (Fig. 2b): rescale the units of the privacy objective.
    priv = R["e3_scale_invariance"]["objectives"]["privacy"]
    lexi_pii = [r["lexi_priv"] for r in priv]
    check("RQ3 Lexi PII invariant across the scale swing (=0.05)",
          max(lexi_pii) - min(lexi_pii) < 1e-6 and abs(lexi_pii[0] - 0.05) < 0.01,
          f"lexi_priv values={sorted(set(lexi_pii))}")
    rank_pii = [r["rank_ws_priv"] for r in priv]
    check("RQ3 Rank-WS scale-invariant (PII ~0.00 across swing)",
          max(rank_pii) - min(rank_pii) < 1e-6 and abs(rank_pii[0]) < 0.01,
          f"rank_ws_priv values={sorted(set(rank_pii))}")
    norm_pii = [r["norm_ws_priv"] for r in priv]
    check("RQ3 Norm-WS scale-invariant (PII ~0.01 across swing)",
          max(norm_pii) - min(norm_pii) < 1e-6 and abs(norm_pii[0] - 0.01) < 0.01,
          f"norm_ws_priv values={sorted(set(norm_pii))}")
    # The tuned weighted sum breaks. Its PII exposure rises from 0.01 at nominal scale to
    # 1.85 when privacy is under-scaled tenfold (factor 0.1).
    ws_nom = _find_factor(priv, 1.0)["ws_priv"]
    ws_10x = _find_factor(priv, 0.1)["ws_priv"]
    close(ws_nom, 0.01, 0.01, "RQ3 WS-tuned PII at nominal scale")
    close(ws_10x, 1.85, 0.03, "RQ3 WS-tuned PII under 10x under-weighting (breaks)")
    # Augmented Tchebycheff is not invariant either. It shows the same rise, 0.01 to 1.85.
    tch_nom = _find_factor(priv, 1.0)["tcheby_priv"]
    tch_10x = _find_factor(priv, 0.1)["tcheby_priv"]
    close(tch_nom, 0.01, 0.01, "RQ3 Tchebycheff PII at nominal scale")
    close(tch_10x, 1.85, 0.03, "RQ3 Tchebycheff PII under 10x rescale (not invariant)")

    # RQ4 candidate curation: Lexi's SLO violation rises from about 1.4 to about 44 percent
    # on a uniformly random menu.
    e5 = R["e5_candidate"]["sets"]
    close(e5["curated"]["Lexi-strict"]["slo"], 1.44, 0.05,
          "RQ4 Lexi SLO on curated set", "%")
    close(e5["random"]["Lexi-strict"]["slo"], 43.56, 0.5,
          "RQ4 Lexi SLO on random set (curation matters)", "%")

    # RQ4 second configuration (config B): the method ordering holds.
    cb = R["e6_generalization"]["config_b"]
    for key, slo in [("Lexi-strict", 6.78), ("constrained_rl", 7.17),
                     ("ws_tuned", 8.14), ("ws_default", 13.81)]:
        close(cb[key]["slo"], slo, 0.05, f"RQ4 config-B {key} SLO%", "%")
    check("RQ4 config-B ordering Lexi<=Constr-WS<=WS-tuned<=WS-default",
          cb["Lexi-strict"]["slo"] <= cb["constrained_rl"]["slo"] + 1e-6
          <= cb["ws_tuned"]["slo"] + 1e-6 <= cb["ws_default"]["slo"] + 1e-6,
          f"SLO: Lexi {cb['Lexi-strict']['slo']}, Constr {cb['constrained_rl']['slo']}, "
          f"WS-t {cb['ws_tuned']['slo']}, WS-d {cb['ws_default']['slo']}")

    # RQ4 additivity (Assumption 1, Fig. 2c): non-additive contention degrades all methods.
    sc = R["e7_misspecification"]["structural_contention"]["slo_pct"]
    lexi0 = sc[0]["Lexi-strict"]
    lexi_hi = sc[-1]["Lexi-strict"]
    check("RQ4 contention degrades Lexi (SUTVA limit; ~1.5% -> ~41%)",
          lexi0 < 2.0 and lexi_hi > 35.0,
          f"Lexi SLO {lexi0}% (c=0) -> {lexi_hi}% (c={sc[-1]['contention']})")
    wst_hi = sc[-1]["ws_tuned"]
    check("RQ4 contention degrades WS-tuned too (~3.2% -> ~52%)",
          sc[0]["ws_tuned"] < 4.0 and wst_hi > 45.0,
          f"WS-tuned SLO {sc[0]['ws_tuned']}% -> {wst_hi}%")

    # RQ5 (extended version), SLO safety margin: delta = 4 ms cuts the violations about ninefold.
    mg = {p["margin_ms"]: p for p in R["e10_margin"]["nominal"]}
    close(mg[0.0]["slo"], 1.45, 0.05, "RQ5 margin delta=0 == strict feasibility", "%")
    close(mg[4.0]["slo"], 0.16, 0.05, "RQ5 margin delta=4ms cuts SLO ~9x", "%")
    close(mg[8.0]["slo"], 0.01, 0.03, "RQ5 margin delta=8ms near-eliminates SLO", "%")
    close(mg[4.0]["privacy"], 0.114, 0.02, "RQ5 margin delta=4ms privacy cost")
    sig = R["e10_margin"]["sig_default_vs_strict_slo"]
    check("RQ5 margin(4ms) beats strict Lexi on SLO (p<0.001)",
          sig["p"] < 0.001 and sig["median_diff"] < 0,
          f"p={sig['p']} median_diff={sig['median_diff']}")
    sigc = R["e10_margin"]["sig_default_vs_constrained_slo"]
    check("RQ5 margin(4ms) beats Constr-WS on SLO (p<0.001)",
          sigc["p"] < 0.001 and sigc["median_diff"] < 0,
          f"p={sigc['p']} median_diff={sigc['median_diff']}")
    mgc = {p["margin_ms"]: p for p in R["e10_margin"]["contention"]}
    check("RQ5 margin helps under contention (delta=12 < delta=0)",
          mgc[12.0]["slo"] < mgc[0.0]["slo"],
          f"contention SLO {mgc[0.0]['slo']}% (d=0) -> {mgc[12.0]['slo']}% (d=12)")

    # RQ5 (extended version), auto-slack: frac = 0 recovers strict Lexi and the slack never
    # touches the SLO hinge.
    e9 = {p["frac"]: p for p in R["e9_autoslack"]["points"]}
    close(e9[0.0]["slo"], 1.46, 0.02, "RQ5 auto-slack frac=0 == strict Lexi", "%")
    close(e9[0.0]["privacy"], 0.05, 0.01, "RQ5 auto-slack privacy strict")
    check("RQ5 auto-slack holds SLO as frac grows (strict primary), cuts churn",
          e9[2.0]["slo"] <= e9[0.0]["slo"] + 0.1 and e9[2.0]["churn"] < e9[0.0]["churn"],
          f"frac0 SLO {e9[0.0]['slo']}% churn {e9[0.0]['churn']}; "
          f"frac2 SLO {e9[2.0]['slo']}% churn {e9[2.0]['churn']}")

    # RQ5 (extended version), stability: carbon slack with hysteresis lowers churn without
    # an SLO cost.
    pts = R["e4_stability"]["points"]
    close(pts[0]["churn"], 1.35, 0.02, "RQ5 churn at zero carbon slack")
    close(pts[-1]["churn"], 1.14, 0.03, "RQ5 churn at max carbon slack (stability)")
    check("RQ5 carbon slack holds SLO (no top-priority cost)",
          pts[-1]["slo"] <= pts[0]["slo"] + 0.1,
          f"SLO {pts[0]['slo']}% -> {pts[-1]['slo']}%")

    # Informational only. The decision-latency slope depends on the machine.
    slope = R["scalability"]["slope_ms_per_cand"]
    print(f"[info] decision-latency slope = {slope:.2e} ms/candidate "
          f"(machine-specific microbenchmark; not a PASS/FAIL claim)\n")

    # RQ2 real-trace check (the real-trace table of the extended version). It verifies
    # ../results/real_data_results.json, produced by real_data_experiment.py, against the
    # numbers quoted in the paper.
    rd_path = os.path.join(os.path.dirname(path), "real_data_results.json")
    if os.path.exists(rd_path):
        with open(rd_path) as f:
            RD = json.load(f)
        rt = RD["table"]; ri = RD["inversion"]
        # Each entry is (method, claimed SLO percent, PII, inversion percent), from the
        # real-trace table.
        real_claims = [
            ("Lexi-strict",     2.02, 0.020,  0.0),
            ("constrained_rl",  1.43, 0.020, 14.3),
            ("ws_tuned",        2.22, 0.002, 16.3),
            ("rank_ws",         1.86, 0.000, 16.4),
            ("norm_ws",         2.43, 0.002, 16.4),
            ("tcheby",          3.82, 0.001, 16.8),
            ("ws_default",     24.90, 1.218, None),
        ]
        for key, slo, pii, inv in real_claims:
            close(rt[key]["slo"]["mean"], slo, 0.05, f"RQ2-real {key} SLO%", "%")
            close(rt[key]["privacy"]["mean"], pii, 0.01, f"RQ2-real {key} PII")
            if inv is not None:
                close(ri[key]["inversion_pct"], inv, 0.5, f"RQ2-real {key} inversion%", "%")
        # Headline: Lexi never inverts while every priced scalariser inverts at least 14 percent.
        check("RQ2-real Lexi priority-exact (0% inversion)",
              ri["Lexi-strict"]["inversion_pct"] == 0.0,
              f"measured {ri['Lexi-strict']['inversion_pct']}%")
        check("RQ2-real all scalarisers invert >= 14%",
              min(ri[k]["inversion_pct"] for k in
                  ("ws_tuned", "rank_ws", "norm_ws", "tcheby", "constrained_rl")) >= 14.0,
              "priced scalarisers all >= 14% under real bursty load")
        # Regional carbon means, which the paper rounds to 93, 168 and 183 gCO2/kWh.
        means = RD["provenance"]["carbon_mean_gco2kwh"]
        for got, want in zip(means, (92.8, 167.5, 182.9)):
            close(got, want, 0.1, f"RQ2-real carbon region mean {want}", " gCO2/kWh")
    else:
        print(f"[info] real-data results not found ({rd_path}); "
              f"run `python3 real_data_experiment.py` to check RQ2-real claims.\n")

    # Report.
    npass = sum(1 for ok, _, _ in _results if ok)
    ntot = len(_results)
    width = max(len(lbl) for _, lbl, _ in _results)
    for ok, lbl, detail in _results:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {lbl.ljust(width)}  |  {detail}")
    print(f"\n{npass}/{ntot} claims match the manuscript.")
    if npass == ntot:
        print("RESULT: PASS -- artifact results are consistent with the paper.")
        return 0
    print("RESULT: FAIL -- see mismatches above.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON))
