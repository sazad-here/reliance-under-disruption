"""A compact two-panel validation figure: discrimination beside calibration.

Rebuilt from chart_data.json rather than cropping the supplement's figures, so
it is drawn at the width it will be printed (122 mm) with 6.5-7 pt text, and so
the two panels share one height and one type size.

Why these two panels: the paper's validation argument is that the displayed
risk both ranks weeks correctly AND means what it says. ROC shows the first,
the calibration curve shows the second, and the paper's claim that three model
families are indistinguishable is visible in the overlapping curves.

Greyscale-safe: every series differs by line style as well as by shade.

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
INK = "#1a1a19"

STYLE = {
    "LightGBM":          dict(color="#1a1a19", ls="-",             lw=1.5, z=5),
    "Logistic regression": dict(color="#6a6a66", ls=(0, (5, 2)),   lw=1.3, z=4),
    "XGBoost":           dict(color="#9a9a95", ls=(0, (1.5, 1.5)), lw=1.3, z=3),
    "Net position only": dict(color="#b8b8b2", ls=(0, (4, 1.5, 1, 1.5)), lw=1.2, z=2),
}
SHORT = {"Logistic regression": "Logistic reg.", "Net position only": "Net position"}


def main():
    cd = json.load(open(os.path.join(ROOT, "lightgbm-pivot", "chart_data.json"),
                        encoding="utf-8"))
    models = cd["models"]

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.2, "axes.titlesize": 7.2,
                         "axes.titleweight": "bold", "font.family": "sans-serif"}):
        fig, ax = plt.subplots(1, 2, figsize=(4.8, 1.12))

        # ----------------------------------------------------- (a) ROC
        a = ax[0]
        a.plot([0, 1], [0, 1], color="#c9c9c4", lw=0.8, zorder=1)
        for name, st in STYLE.items():
            m = models.get(name)
            if not m:
                continue
            r = m["roc"]
            a.plot(r["fpr"], r["tpr"], color=st["color"], linestyle=st["ls"],
                   lw=st["lw"], zorder=st["z"],
                   label="%s %.3f" % (SHORT.get(name, name), m["auroc"]))
        a.set_xlabel("False positive rate")
        a.set_ylabel("True positive rate")
        a.set_title("a   Discrimination (AUROC)", loc="left")
        a.legend(loc="lower right", frameon=False, handlelength=2.1,
                 borderpad=0.2, labelspacing=0.25)

        # --------------------------------------------- (b) calibration
        b = ax[1]
        b.plot([0, 1], [0, 1], color="#c9c9c4", lw=0.8, zorder=1)
        for name, st in STYLE.items():
            m = models.get(name)
            if not m or "calibration" not in m:
                continue
            # chart_data stores calibration as decile points: pred vs obs
            c = m["calibration"]
            pts = c.get("points") or []
            if not pts:
                continue
            xs = [q["pred"] for q in pts]
            ys = [q["obs"] for q in pts]
            b.plot(xs, ys, color=st["color"], linestyle=st["ls"], lw=st["lw"],
                   marker="o", markersize=2.2, zorder=st["z"],
                   label=SHORT.get(name, name))
        b.set_xlabel("Predicted stockout probability")
        b.set_ylabel("Observed frequency")
        lg = models["LightGBM"]["calibration"]
        b.set_title("b   Calibration (slope %.2f, intercept %+.2f)"
                    % (lg["slope"], lg["intercept"]), loc="left")
        # No legend here: the line styles are identical to panel (a), and at this
        # height a second legend overlaps the calibration curve.

        for axis in ax:
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            for sp in ("top", "right"):
                axis.spines[sp].set_visible(False)
            axis.tick_params(length=2.2, width=0.7)

        fig.tight_layout(pad=0.3, w_pad=1.2)
        fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
        print("wrote", OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
