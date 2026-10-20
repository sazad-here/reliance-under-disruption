"""Figure 2: discrimination and calibration, drawn at print width (122 mm).

Rebuilt from chart_data.json rather than cropping the supplement's figures, so
it is drawn at the width it will be printed with 6-7 pt text, and so the panels
share one height and one type size.

Why these panels: the paper's validation argument is that the displayed risk
both ranks weeks correctly AND means what it says. ROC shows the first, the
reliability diagram the second.

Redesigned 19 Oct 2026 to match Figs. 1 and 3:
  * LightGBM, the deployed family, is drawn in the blue Fig. 1 uses for the AI
    side, heavier and with the area under its curve shaded; the comparison
    models use Okabe-Ito orange and green and a neutral grey (colour-vision-
    deficiency safe), and every series also differs by line style for greyscale.
  * Panel (b) shows AUROC in each of the five cross-validation folds. The
    leading models move together, and the gap between them (at most 0.005 in
    any fold) is small against the 0.04 swing from fold to fold: the paper's
    equivalence claim, shown rather than asserted. (It replaced a zoom of the
    ROC knee, which only magnified panel (a).)
  * Panel (c) is a reliability diagram: LightGBM's ten equal-count deciles as
    filled points on the diagonal, the comparison models as faint lines, and
    slope and intercept in a callout.
  * One shared legend across the top, so no panel carries a legend over data.
  Nothing is smoothed or re-binned: every point is a value in chart_data.json.

Run:  python figures/make_figure_validation.py
Out:  figures/paper/main_text/Figure_validation.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "paper", "main_text", "Figure_validation.png")
INK, MUTED, DIAG = "#1a1a19", "#6b6a65", "#b9b8b2"
BLUE = "#0b5e99"   # the AI side in Fig. 1

STYLE = {
    "LightGBM":            dict(color=BLUE,      ls="-",                   lw=1.7, z=6),
    "Logistic regression": dict(color="#e69f00", ls=(0, (5, 2)),           lw=1.15, z=5),
    "XGBoost":             dict(color="#009e73", ls=(0, (1.4, 1.3)),       lw=1.3, z=4),
    "Net position only":   dict(color="#8c8b86", ls=(0, (4, 1.5, 1, 1.5)), lw=1.1, z=3),
}
LABEL = {"LightGBM": "LightGBM", "Logistic regression": "Logistic regression",
         "XGBoost": "XGBoost", "Net position only": "Net position alone"}


def main():
    cd = json.load(open(os.path.join(ROOT, "lightgbm-pivot", "chart_data.json"), encoding="utf-8"))
    models = cd["models"]

    with plt.rc_context({"font.size": 6.8, "axes.labelsize": 6.6, "xtick.labelsize": 6.0,
                         "ytick.labelsize": 6.0, "legend.fontsize": 6.2, "axes.titlesize": 6.9,
                         "axes.titleweight": "bold", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "axes.edgecolor": "#8c8b86", "axes.linewidth": 0.7,
                         "xtick.color": MUTED, "ytick.color": MUTED}):
        fig, ax = plt.subplots(1, 3, figsize=(4.8, 1.62), layout="constrained")

        # ----------------------------------------------------- (a) ROC, full range
        a = ax[0]
        a.plot([0, 1], [0, 1], color=DIAG, lw=0.7, ls=(0, (2, 2)), zorder=1)
        a.text(0.64, 0.52, "chance", rotation=37, color=MUTED, fontsize=5.4, ha="center", va="bottom")
        handles = []
        for name, st in STYLE.items():
            m = models[name]
            r = m["roc"]
            h, = a.plot(r["fpr"], r["tpr"], color=st["color"], linestyle=st["ls"], lw=st["lw"],
                        zorder=st["z"], label="%s  %.3f" % (LABEL[name], m["auroc"]))
            handles.append(h)
            if name == "LightGBM":
                a.fill_between(r["fpr"], r["tpr"], color=BLUE, alpha=0.07, lw=0, zorder=2)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_xlabel("False positive rate")
        a.set_ylabel("True positive rate")
        a.set_title("a  ROC curves (AUROC)", loc="left")

        # ----------------------------------------------------- (b) AUROC in each fold
        # The paper's claim in one picture: the leading models move together from fold to fold,
        # and the gap between them (<= 0.005 in every fold) is small against the 0.04 swing.
        z = ax[1]
        folds = [1, 2, 3, 4, 5]
        for name, st in STYLE.items():
            fa = models[name]["fold_auroc"]
            z.plot(folds, fa, color=st["color"], linestyle=st["ls"], lw=st["lw"], zorder=st["z"],
                   marker="o", ms=2.6 if name == "LightGBM" else 2.0, mec="white", mew=0.4)
        lead = ["LightGBM", "Logistic regression", "XGBoost"]
        gap = max(max(models[m]["fold_auroc"][i] for m in lead) - min(models[m]["fold_auroc"][i] for m in lead)
                  for i in range(5))
        f = models["LightGBM"]["fold_auroc"]
        lo, hi = min(f), max(f)
        # bracket for the fold-to-fold swing, at the right edge
        bx = 5.45
        z.plot([bx, bx], [lo, hi], color=INK, lw=0.7, clip_on=False)
        for yy in (lo, hi):
            z.plot([bx - 0.08, bx], [yy, yy], color=INK, lw=0.7, clip_on=False)
        z.text(bx + 0.12, (lo + hi) / 2, "fold\nswing\n%.2f" % (hi - lo), fontsize=5.3, va="center",
               ha="left", color=INK, clip_on=False)
        z.text(0.03, 0.97, "leading models within\n%.3f in every fold" % gap, transform=z.transAxes,
               ha="left", va="top", fontsize=5.5, color=INK, style="italic")
        z.set_xlim(0.7, 5.3)
        z.set_ylim(0.84, 0.93)
        z.set_xticks(folds)
        z.set_yticks([0.85, 0.88, 0.91])
        z.set_xlabel("Cross-validation fold")
        z.set_ylabel("AUROC", labelpad=1)
        z.set_title("b  AUROC in each fold", loc="left")

        # --------------------------------------------- (c) reliability diagram
        b = ax[2]
        b.plot([0, 1], [0, 1], color=DIAG, lw=0.7, ls=(0, (2, 2)), zorder=1)
        for name, st in STYLE.items():
            if name == "LightGBM":
                continue
            pts = models[name]["calibration"]["points"]
            b.plot([q["pred"] for q in pts], [q["obs"] for q in pts], color=st["color"],
                   linestyle=st["ls"], lw=0.9, alpha=0.85, zorder=st["z"])
        lg = models["LightGBM"]["calibration"]
        pts = lg["points"]
        xs, ys = [q["pred"] for q in pts], [q["obs"] for q in pts]
        b.plot(xs, ys, color=BLUE, lw=1.5, zorder=6)
        b.scatter(xs, ys, s=11, color=BLUE, edgecolor="white", linewidth=0.5, zorder=7)
        b.text(0.04, 0.96, "slope %.2f\nintercept %+.2f" % (lg["slope"], lg["intercept"]),
               transform=b.transAxes, va="top", ha="left", fontsize=5.8, color=BLUE, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BLUE, lw=0.6))
        b.text(0.97, 0.04, "points: deciles,\n%s weeks each" % format(pts[0]["n"], ","),
               transform=b.transAxes, va="bottom", ha="right", fontsize=5.3, color=MUTED)
        b.set_xlim(0, 1)
        b.set_ylim(0, 1)
        b.set_xlabel("Predicted stockout risk")
        b.set_ylabel("Observed stockout rate")
        b.set_title("c  Calibration", loc="left")

        for axis in ax:
            for sp in ("top", "right"):
                axis.spines[sp].set_visible(False)
            axis.tick_params(length=2, width=0.6, pad=1.5)
        for axis in (a, b):
            axis.set_xticks([0, 0.5, 1])
            axis.set_yticks([0, 0.5, 1])

        leg = fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
                         handlelength=2.0, columnspacing=1.0, handletextpad=0.45)
        leg.get_texts()[0].set_fontweight("bold")

        fig.savefig(OUT, dpi=600, facecolor="white")
        print("wrote", OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
