"""
Figures of the working paper drafts, drawn from the measured results in core_results.json
(and robustness_results.json for the last one). The output folder is the one named by the
environment variable LEXI_FIGURE_DIR, or ../figures/ when the variable is not set. The
script needs numpy and matplotlib only.

Outputs.
  fig_thesis.pdf        E2 weight sweep (RQ1) and E3 scale invariance (RQ3).
  fig_satisfaction.pdf  E1 physical metrics with 95 percent intervals (RQ2).
  fig_robust.pdf        E5 candidate menu and E7 contention (RQ4).
  fig_conflict.pdf      E11 cold-start priority conflict (extended version), when the
                        result key is present.
  fig_stress.pdf        contention recovery, doubly-robust estimation and decision-time
                        scaling (extended version, RQ6 and RQ7), when
                        robustness_results.json is present.

The figure that appears in the camera-ready paper (Fig. 2) and in the extended version is
fig_results.pdf, which make_camera_figures.py draws at the printed size. The figures
written here are supplementary panels and are not included in either paper.
"""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "0.85",
    "grid.linewidth": 0.6, "figure.dpi": 200,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "gray": "#7A7A7A"}
HERE = os.path.dirname(os.path.abspath(__file__))
# Write into $LEXI_FIGURE_DIR when it is set and into ../figures otherwise. The released
# artifact ships no manuscript folder.
FIG = os.environ.get("LEXI_FIGURE_DIR", os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(HERE, "..", "results", "core_results.json")))

PRETTY = {"Lexi-strict": "Lexi", "Lexi": "Lexi($\\eta$)", "ws_default": "WS-default",
          "ws_tuned": "WS-tuned", "constrained_rl": "Constr-WS",
          "single_obj_rl": "SO-RL", "static_greedy": "greedy", "binpack": "binpack"}


# fig_thesis.pdf: the weight sweep (E2, RQ1) beside the privacy-scale study (E3, RQ3).
sw = R["e2_weight_sweep"]
pts = sw["pareto"]
obj = R["e3_scale_invariance"]["objectives"]["privacy"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.9, 1.9), gridspec_kw={"wspace": 0.55})
# (a) Weight sweep: one point per stored weighting, with Lexi as a star.
px = np.array([p["privacy"] for p in pts]); py = np.array([p["slo"] for p in pts])
axa.scatter(px, py, s=10, color=C["red"], alpha=0.55, edgecolor="none",
            label=f"weighted sums ($n{{=}}{sw['n_weights']}$)")
axa.scatter([sw["lexi_strict"]["privacy"]], [sw["lexi_strict"]["slo"]], marker="*",
            s=170, color=C["green"], edgecolor="black", linewidth=0.6, zorder=5,
            label="Lexi (no weights)")
axa.set_xlabel("PII exposure (of 2)"); axa.set_ylabel("SLO-violation (\\%)")
axa.set_ylim(bottom=0)
axa.set_title(f"(a) tuning burden: {sw['frac_meeting_target']*100:.1f}% meet target",
              fontsize=8.5)
axa.legend(fontsize=7, loc="lower right")
# (b) Scale invariance, with the privacy objective rescaled by the factors of E3.
f = np.array([r["factor"] for r in obj])
axb.plot(f, [r["lexi_priv"] for r in obj], "-o", color=C["green"], ms=5, label="Lexi")
axb.plot(f, [r["ws_priv"] for r in obj], "-s", color=C["red"], ms=5, label="WS-tuned")
axb.axvline(1.0, color=C["gray"], ls=":", lw=0.9); axb.set_xscale("log")
# Tick the extreme factors explicitly so that the whole 0.1 to 30 range (a 300-fold swing) is readable.
axb.set_xlim(0.08, 40)
axb.set_xticks([0.1, 1, 10, 30]); axb.set_xticklabels(["0.1", "1", "10", "30"])
axb.set_xlabel("Privacy-scale miscalibration ($300\\times$ swing)"); axb.set_ylabel("PII exposure (of 2)")
axb.set_title("(b) scale-invariance", fontsize=8.5)
axb.legend(fontsize=7, loc="center right")
for ax in (axa, axb):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(FIG, "fig_thesis.pdf"), bbox_inches="tight")
plt.close(fig)


# fig_conflict.pdf: priority conflict with serverless cold starts (E11, extended version).
# As the cold-start risk sharpens the tension between the SLO and cost or carbon, every
# priced scalariser (including the scale-invariant rank and min-max sums and Tchebycheff)
# inverts the priority order on a growing share of decisions. Strict Lexi stays at 0.
if "e11_conflict_stress" in R:
    lv = R["e11_conflict_stress"]["levels"]
    cp = np.array([l["cold_penalty"] for l in lv])
    figc, (cxa, cxb) = plt.subplots(1, 2, figsize=(9.4, 1.15), gridspec_kw={"wspace": 0.32})
    inv_series = [("ws_tuned", C["red"], "-s", "WS-tuned"),
                  ("rank_ws", C["orange"], "-^", "rank-WS"),
                  ("norm_ws", C["purple"], "-D", "norm-WS"),
                  ("tcheby", C["blue"], "-v", "Tcheby")]
    for key, col, mk, lab in inv_series:
        y = np.array([l["inversion"][key]["mean"] for l in lv])
        cxa.plot(cp, y, mk, color=col, ms=4.5, lw=1.3, label=lab)
    cxa.plot(cp, [l["inversion"]["Lexi-strict"]["mean"] for l in lv], "-o",
             color=C["green"], ms=5.5, lw=1.8, label="Lexi (=0)")
    cxa.set_xlabel("serverless cold-start penalty (ms)")
    cxa.set_ylabel("priority inversion (\\%)"); cxa.set_ylim(bottom=-1)
    cxa.set_title("(a) priority order betrayed under conflict", fontsize=8.5)
    cxa.legend(fontsize=7, loc="center right", ncol=1)
    for key, col, mk, lab in [("Lexi-margin", C["green"], "-o", "Lexi-margin"),
                              ("ws_tuned", C["red"], "-s", "WS-tuned"),
                              ("tcheby", C["blue"], "-v", "Tcheby")]:
        y = np.array([l["slo_viol"][key]["mean"] for l in lv])
        cxb.plot(cp, y, mk, color=col, ms=4.5, lw=1.4, label=lab)
    cxb.set_xlabel("serverless cold-start penalty (ms)")
    cxb.set_ylabel("SLO-violation (\\%)"); cxb.set_ylim(bottom=0)
    cxb.set_title("(b) realized SLO violation", fontsize=8.5)
    cxb.legend(fontsize=7, loc="lower right")
    for ax in (cxa, cxb):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    figc.savefig(os.path.join(FIG, "fig_conflict.pdf"), bbox_inches="tight")
    plt.close(figc)
    print("wrote fig_conflict.pdf (E11 cold-start priority-conflict stress)")


# fig_satisfaction.pdf: SLO violation and PII exposure with 95 percent intervals for the
# methods of E1, the graphical form of Table 3 and its extended-version rows.
tab = R["e1_satisfaction"]["table"]
methods = [m for m in ["Lexi-strict", "constrained_rl", "ws_tuned", "ws_default",
                       "single_obj_rl", "static_greedy", "binpack"] if m in tab]
labels = [PRETTY[m] for m in methods]
x = np.arange(len(methods))
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, k, ttl in ((axa, "slo", "(a) SLO-violation (\\%)"),
                   (axb, "privacy", "(b) PII exposure (of 2)")):
    mean = np.array([tab[m][k]["mean"] for m in methods])
    lo = np.array([tab[m][k]["ci"][0] for m in methods])
    hi = np.array([tab[m][k]["ci"][1] for m in methods])
    cols = [C["green"] if m == "Lexi-strict" else
            (C["blue"] if m in ("ws_tuned", "constrained_rl") else C["gray"])
            for m in methods]
    ax.bar(x, mean, yerr=[np.maximum(0, mean - lo), np.maximum(0, hi - mean)],
           color=cols, edgecolor="black", linewidth=0.5, capsize=2.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title(ttl, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_satisfaction.pdf"), bbox_inches="tight")
plt.close(fig)


# fig_robust.pdf: dependence on the candidate menu (E5) and structural misspecification
# by co-location contention (E7), the material of RQ4.
cand = R["e5_candidate"]["sets"]
cont = R["e7_misspecification"]["structural_contention"]["slo_pct"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9))
sets = ["curated", "random"]
lex = [cand[s]["Lexi-strict"]["slo"] for s in sets]
wst = [cand[s]["ws_tuned"]["slo"] for s in sets]
xx = np.arange(2); w = 0.36
axa.bar(xx - w / 2, lex, w, color=C["green"], edgecolor="black", linewidth=0.5, label="Lexi")
axa.bar(xx + w / 2, wst, w, color=C["blue"], edgecolor="black", linewidth=0.5, label="WS-tuned")
axa.set_xticks(xx); axa.set_xticklabels(["curated", "random"])
axa.set_ylabel("SLO-violation (\\%)")
axa.set_title("(a) Candidate-set dependence", fontsize=9)
axa.legend(fontsize=7.5)
cc = np.array([r["contention"] for r in cont])
axb.plot(cc, [r["Lexi-strict"] for r in cont], "-o", color=C["green"], ms=4, label="Lexi")
axb.plot(cc, [r["ws_tuned"] for r in cont], "-s", color=C["blue"], ms=4, label="WS-tuned")
axb.plot(cc, [r["constrained_rl"] for r in cont], "-^", color=C["orange"], ms=4, label="Constr-WS")
axb.set_xlabel("Contention (non-additive)")
axb.set_ylabel("SLO-violation (\\%)")
axb.set_title("(b) Structural misspecification", fontsize=9)
axb.legend(fontsize=7.5, loc="upper left")
for ax in (axa, axb):
    ax.spines["top"].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_robust.pdf"), bbox_inches="tight")
plt.close(fig)

# fig_stress.pdf: recovery of contention by the learned M/G/1 term, doubly-robust against
# direct estimation, and decision and update time against the DAG size. These are the
# model-stress measurements of the extended version (RQ6 and RQ7), produced by
# robustness_experiments.py. The figure is written only if ../results/robustness_results.json exists.
_robust = os.path.join(HERE, "..", "results", "robustness_results.json")
if os.path.exists(_robust):
    ROB = json.load(open(_robust))
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(9.9, 2.7),
                                        gridspec_kw={"wspace": 0.5})
    # (a) SLO violation of the additive model against the contention-aware model.
    rows = ROB["rx1_contention"]["rows"]
    cc = np.array([r["contention"] for r in rows])
    axa.plot(cc, [r["additive_slo"] for r in rows], "-s", color=C["red"], ms=5,
             label="additive model")
    axa.plot(cc, [r["aware_slo"] for r in rows], "-o", color=C["green"], ms=5,
             label="contention-aware")
    axa.set_xlabel("Contention (non-additive)")
    axa.set_ylabel("SLO-violation (\\%)")
    axa.set_title("(a) $M/G/1$ recovery", fontsize=9)
    axa.legend(fontsize=7, loc="upper left")
    # (b) Carbon estimation error of the doubly-robust and the direct estimator under misspecification.
    dr = ROB["rx2_doubly_robust"]["rows"]
    ee = np.array([r["epsilon"] for r in dr])
    axb.plot(ee, [r["dm_esterr"] for r in dr], "-s", color=C["red"], ms=5, label="direct method")
    axb.plot(ee, [r["dr_esterr"] for r in dr], "-o", color=C["green"], ms=5, label="doubly robust")
    axb.set_xlabel("Misspecification $\\varepsilon$")
    axb.set_ylabel("Estimation error")
    axb.set_title("(b) DR vs DM", fontsize=9)
    axb.legend(fontsize=7, loc="upper left")
    # (c) Decision and update latency against the DAG size K.
    kk = ROB["rx5_scaling"]["by_K"]
    K = np.array([r["K"] for r in kk])
    axc.plot(K, [r["act_ms_median"] for r in kk], "-o", color=C["blue"], ms=5, label="decision")
    axc.plot(K, [r["update_ms_median"] for r in kk], "-^", color=C["orange"], ms=5, label="update")
    axc.set_xlabel("DAG size $K$"); axc.set_ylabel("ms / request")
    axc.set_title("(c) scaling ($C{=}250$)", fontsize=9)
    axc.legend(fontsize=7, loc="upper left")
    for ax in (axa, axb, axc):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(FIG, "fig_stress.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_stress.pdf (contention recovery + DR + scaling)")

print("wrote fig_thesis.pdf, fig_satisfaction.pdf and fig_robust.pdf (supplementary)")
