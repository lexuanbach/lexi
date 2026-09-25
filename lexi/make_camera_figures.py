"""
The single multi-panel results figure of the camera-ready paper (Fig. 2).

The figure is drawn at its printed size (12.2 cm text width) from the committed result
files only. No experiment is re-run and no value is interpolated. Output is
fig_results.pdf, with three panels.
  (a) RQ1. The 84-weighting sweep as SLO violation against PII exposure, with Lexi marked.
      Source: core_results.json, key e2_weight_sweep (experiments_core.e2_weight_sweep).
  (b) RQ3. Privacy-scale miscalibration, factors 0.1 to 30, as PII exposure.
      Sources: core_results.json, key e3_scale_invariance.objectives.privacy (Lexi and
      WS-tuned), and extended_results.json, key B_constrrl_scale (Constr-WS, the
      SLO-constrained weighted sum, whose result key keeps its historical name).
  (c) RQ4. Co-location contention c from 0 to 0.6, which breaks Assumption 1, as SLO
      violation. Source: core_results.json, key e7_misspecification.structural_contention.

The figure is written to the folder named by the environment variable LEXI_FIGURE_DIR, or
to ../figures/ when the variable is not set. It needs matplotlib, and the Computer Modern
font ships with it.
"""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.environ.get("LEXI_FIGURE_DIR", os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

core = json.load(open(os.path.join(RES, "core_results.json")))
ext = json.load(open(os.path.join(RES, "extended_results.json")))

PT = 7  # every text element is 7 pt at the printed size (text width 12.2 cm)
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": PT, "axes.labelsize": PT, "xtick.labelsize": PT,
    "ytick.labelsize": PT, "legend.fontsize": PT,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5,
    "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.4,
    "hatch.linewidth": 0.4,
})

# One style per method, identical in every panel. Colours come from the Okabe-Ito palette
# and each method also has its own marker and line style, which means the series stay
# distinguishable in grayscale print. Lexi is drawn first with filled markers and a
# slightly heavier line. The two baselines are drawn over it with hollow markers. Where a
# baseline coincides with Lexi (Constr-WS from factor 1 upwards in (b), and for c <= 0.4
# in (c)) both series therefore stay visible instead of one hiding the other.
STY = {
    "lexi": dict(color="#009E73", marker="o", ls="-", lw=1.4, ms=3.9, mfc=None,
                 z=4, label="Lexi (no weights)"),
    "crl":  dict(color="#0072B2", marker="^", ls="--", lw=1.0, ms=3.9, mfc="none",
                 z=5, label="Constr-WS"),
    "ws":   dict(color="#D55E00", marker="s", ls=":", lw=1.1, ms=3.4, mfc="none",
                 z=5, label="WS-tuned"),
}
GRAY = "0.5"
BOLD = "cmb10"  # Computer Modern bold, shipped with matplotlib (cmr10 has no bold face)

CM = 1 / 2.54
fig, axs = plt.subplots(1, 3, figsize=(12.2 * CM, 3.5 * CM))
fig.subplots_adjust(left=0.07, right=0.978, bottom=0.215, top=0.745, wspace=0.44)
axa, axb, axc = axs


def plot(ax, x, y, key):
    s = STY[key]
    ax.plot(x, y, color=s["color"], marker=s["marker"], ls=s["ls"], lw=s["lw"],
            ms=s["ms"], mfc=s["mfc"] or s["color"], mec=s["color"],
            markeredgewidth=0.7, clip_on=False, zorder=s["z"])


# (a) RQ1 weight sweep.
# The x axis is a square-root scale. A number of the plotted weightings sit at PII = 0,
# and the region that matters (the target and Lexi) spans only 0 to 0.1 of the 0 to 2 range.
sw = core["e2_weight_sweep"]
pts = sw["pareto"]                       # the stored weightings with SLO at most 25 percent
n_hidden = sw["n_weights"] - len(pts)    # weightings above the plotted window, only counted
axa.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)),
                                      lambda v: np.square(v)))
# The target region of RQ1 is SLO <= 8% and PII <= 0.1 (Sect. 6.1). A light hatch keeps it
# readable without competing with the Lexi marker drawn inside it.
axa.add_patch(Rectangle((0, 0), 0.1, 8.0, facecolor="0.95", edgecolor="0.55",
                        lw=0.5, hatch="////", zorder=1))
axa.scatter([p["privacy"] for p in pts], [p["slo"] for p in pts], s=6,
            facecolor="none", edgecolor=GRAY, linewidth=0.45, zorder=3, clip_on=False)
lx = sw["lexi_strict"]
axa.plot([lx["privacy"]], [lx["slo"]], color=STY["lexi"]["color"], marker="o",
         ms=5, mec="black", mew=0.5, ls="none", zorder=5, clip_on=False)
axa.annotate("target", xy=(0.1, 6.5), xytext=(0.2, 8.4), fontsize=PT, va="center",
             arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
axa.text(0.2, 5.5, r"SLO$\leq$8%, PII$\leq$0.1", fontsize=PT - 1, va="center",
         color="0.35")
axa.annotate("Lexi", xy=(lx["privacy"], lx["slo"]), xytext=(0.2, 1.9),
             fontsize=PT, va="center", color=STY["lexi"]["color"],
             arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
# The weightings above the window are counted just above the top edge of the panel, where
# the note cannot collide with any plotted point.
axa.text(1.0, 1.015, f"$\\uparrow$ {n_hidden} more above 25%", transform=axa.transAxes,
         ha="right", va="bottom", fontsize=PT - 1, color="0.35")
axa.set_xlim(0, 2); axa.set_ylim(0, 25)
axa.set_xticks([0, 0.1, 0.5, 1, 2]); axa.set_xticklabels(["0", "0.1", "0.5", "1", "2"])
axa.minorticks_off()
axa.set_yticks([0, 5, 10, 15, 20, 25])
axa.set_xlabel("PII exposure (of 2, sqrt scale)")
axa.set_ylabel("SLO violation (%)")

# (b) RQ3 privacy-scale miscalibration. The three curves must share one factor grid.
rows = core["e3_scale_invariance"]["objectives"]["privacy"]
crl = {r["factor"]: r["constr_rl_priv"] for r in ext["B_constrrl_scale"]["rows"]}
f = [r["factor"] for r in rows]
assert sorted(crl) == sorted(f), "Constr-WS and WS scale grids differ"
plot(axb, f, [r["lexi_priv"] for r in rows], "lexi")
plot(axb, f, [r["ws_priv"] for r in rows], "ws")
plot(axb, f, [crl[x] for x in f], "crl")
axb.axvline(1.0, color="0.35", ls="-", lw=0.5, zorder=2)
axb.text(1.12, 1.95, "nominal", fontsize=PT - 1, va="top", color="0.35")
axb.set_xscale("log"); axb.set_xlim(0.1, 30)
axb.set_xticks([0.1, 0.3, 1, 3, 10, 30])
axb.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
axb.minorticks_off()
axb.set_ylim(0, 2); axb.set_yticks([0, 0.5, 1, 1.5, 2])
axb.set_yticklabels(["0", "0.5", "1", "1.5", "2"])
axb.set_xlabel("privacy-scale factor (log)")
axb.set_ylabel("PII exposure (of 2)")

# (c) RQ4 co-location contention, which breaks Assumption 1.
cont = core["e7_misspecification"]["structural_contention"]["slo_pct"]
c = [r["contention"] for r in cont]
plot(axc, c, [r["Lexi-strict"] for r in cont], "lexi")
plot(axc, c, [r["ws_tuned"] for r in cont], "ws")
plot(axc, c, [r["constrained_rl"] for r in cont], "crl")
axc.set_xlim(0, 0.6); axc.set_xticks([0, 0.2, 0.4, 0.6])
axc.set_xticklabels(["0", "0.2", "0.4", "0.6"])
axc.set_ylim(0, 55); axc.set_yticks([0, 10, 20, 30, 40, 50])
axc.set_xlabel("contention coefficient $c$")
axc.set_ylabel("SLO violation (%)")

for ax, tag in zip(axs, "abc"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # Offsetting the axis lines by a few points keeps markers that sit exactly on an axis
    # (the weightings at PII = 0 in (a), WS-tuned at exposure 0 in (b)) off the spine.
    ax.spines["left"].set_position(("outward", 2.5))
    ax.spines["bottom"].set_position(("outward", 2.5))
    ax.set_axisbelow(True)
    ax.text(-0.30, 1.06, f"({tag})", transform=ax.transAxes, fontsize=PT,
            fontfamily=BOLD, ha="left", va="bottom")

# One legend for all three panels. The sweep entry shows three hollow points, which reads
# as a cloud of weightings and removes the gap a lone marker leaves in a wide handle.
handles = [Line2D([], [], color=STY[k]["color"], marker=STY[k]["marker"],
                  ls=STY[k]["ls"], lw=STY[k]["lw"], ms=STY[k]["ms"],
                  mfc=STY[k]["mfc"] or STY[k]["color"], mec=STY[k]["color"], mew=0.7,
                  label=STY[k]["label"])
           for k in ("lexi", "crl", "ws")]
cloud = Line2D([], [], color=GRAY, marker="o", mfc="none", mew=0.45, ms=2.6, ls="none",
               label=f"{sw['n_weights']} weighted sums (a)")
handles.append(cloud)
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.52, 1.02), handlelength=2.6, columnspacing=1.6,
           handletextpad=0.5, borderaxespad=0.1,
           handler_map={cloud: HandlerLine2D(numpoints=3)})

out = os.path.join(FIG, "fig_results.pdf")
fig.savefig(out)          # the canvas is the printed size, which means LaTeX includes it at natural width
plt.close(fig)
print(f"wrote {os.path.relpath(out, HERE)}")
