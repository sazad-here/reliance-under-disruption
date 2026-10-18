"""Figure 1: model development and the weekly human-AI decision loop.

Drawn deterministically rather than AI-generated, because the figure carries
exact values (194 trees, AUROC 0.897) that an image model would misspell or
invent, and because the manuscript is unpublished.

Design decisions, and why:

* The two panels are CONNECTED. The frozen model in (a) is the same object that
  produces the risk estimate the participant reads in (b) step 3. The earlier
  version drew two unrelated rows of boxes, hiding the paper's own argument
  that the displayed risk is a validated instrument.
* The week is drawn as a LOOP, not a line, and the return arrow passes BELOW
  the row so it never crosses a box. Inventory, pipeline orders and backorders
  carry into the next week; that persistence is the premise of the study.
* The commitment point between step 2 and step 3 is marked. Weight of advice is
  undefined without an independent order captured before the advice appears, so
  the lock is the measurement mechanism and is drawn as such.
* Greyscale-safe: white and two greys, black borders and black text throughout;
  nothing is distinguished by hue alone.
* Drawn at 4.8 in, the LNCS 122 mm print width, with 6.3-7.2 pt text, so text
  prints at the size it is drawn. Scaling a placed figure down is what shrank an
  earlier figure's labels to roughly 5 pt.

Run:  python figures/make_figure1.py
Out:  figures/paper/main_text/Figure1_merged_workflow.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1a1a19"
FILL_A = "#ecebe5"   # model pipeline
FILL_B = "#ffffff"   # participant-facing steps
FILL_M = "#c9c7c0"   # the frozen model: the object both panels share

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "paper", "main_text", "Figure1_merged_workflow.png")


def box(ax, x, y, w, h, title, lines, fill, ts=6.8, bs=6.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.010",
        linewidth=0.7, edgecolor=INK, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 0.040, title, ha="center", va="top",
            fontsize=ts, fontweight="bold", color=INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.108 - i * 0.068, ln, ha="center", va="top",
                fontsize=bs, color=INK, zorder=3)


def arrow(ax, p0, p1, rad=0.0, dashed=False, lw=0.8):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=6.5, linewidth=lw, color=INK,
        zorder=4, linestyle=(0, (2.2, 1.6)) if dashed else "solid",
        connectionstyle="arc3,rad=%.3f" % rad, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(4.8, 2.25))
    # Margin on every side: the previous version clipped the right-hand box
    # titles and cut the return arrow off at the bottom.
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.175, 1.02)
    ax.axis("off")

    # ---------------------------------------------------------------- panel a
    ax.text(0.0, 0.995, "a", fontsize=8.2, fontweight="bold", va="top", color=INK)
    ax.text(0.026, 0.995, "Model development, once, before any participant",
            fontsize=7.2, fontweight="bold", va="top", color=INK)

    ay, ah = 0.645, 0.265
    N, gap = 5, 0.028
    aw = (1.0 - (N - 1) * gap) / N          # exactly fills the width
    stages = [
        ("Open data", ["Walmart M5,", "Olist, DataCo"], FILL_A),
        ("Digital twin", ["one echelon,", "520 x 100 wk"], FILL_A),
        ("Features", ["30: state +", "4 wk history"], FILL_A),
        ("LightGBM", ["194 trees,", "Platt scaled"], FILL_A),
        ("Frozen model", ["AUROC 0.897", "ECE 0.006"], FILL_M),
    ]
    axs = []
    for i, (t, ls, fl) in enumerate(stages):
        x = i * (aw + gap)
        axs.append(x)
        box(ax, x, ay, aw, ah, t, ls, fl,
            ts=6.0 if t == "Frozen model" else 6.8)
        if i:
            arrow(ax, (x - gap + 0.002, ay + ah / 2), (x - 0.002, ay + ah / 2))

    # ---------------------------------------------------------------- panel b
    ax.text(0.0, 0.525, "b", fontsize=8.2, fontweight="bold", va="top", color=INK)
    ax.text(0.026, 0.525, "One week, repeated 30 times",
            fontsize=7.2, fontweight="bold", va="top", color=INK)

    by, bh = 0.145, 0.285
    M, bgap = 4, 0.034
    bw = (1.0 - (M - 1) * bgap) / M
    steps = [
        ("1  Inspect", ["stock, demand,", "arrivals, backorders"]),
        ("2  Own order", ["entered, then", "locked"]),
        ("3  AI advice", ["order + risk %"]),
        ("4  Final order", ["accept, modify", "or override"]),
    ]
    bxs = []
    for i, (t, ls) in enumerate(steps):
        x = i * (bw + bgap)
        bxs.append(x)
        box(ax, x, by, bw, bh, t, ls, FILL_B)
        if i:
            arrow(ax, (x - bgap + 0.002, by + bh / 2), (x - 0.002, by + bh / 2))

    # the lock: an independent order must exist before any advice is shown
    lx = bxs[2] - bgap / 2
    ax.plot([lx, lx], [by + 0.012, by + bh - 0.012], color=INK, lw=1.0,
            linestyle=(0, (1.3, 1.1)), zorder=5)

    # the frozen model feeds step 3 -- the link the earlier figure omitted
    arrow(ax, (axs[4] + aw / 2, ay - 0.004),
          (bxs[2] + bw * 0.55, by + bh + 0.004), dashed=True, rad=0.22)
    ax.text(0.905, 0.585, "risk shown to\nthe participant", fontsize=6.0,
            ha="center", va="center", color=INK, zorder=5)

    # the loop, routed BELOW the row so it crosses nothing
    y_loop = by - 0.085
    x_from, x_to = bxs[3] + bw / 2, bxs[0] + bw / 2
    ax.plot([x_from, x_from], [by - 0.002, y_loop], color=INK, lw=0.9, zorder=1)
    ax.plot([x_from, x_to], [y_loop, y_loop], color=INK, lw=0.9, zorder=1)
    arrow(ax, (x_to, y_loop), (x_to, by - 0.004), lw=0.9)
    ax.text(0.5, -0.138,
            "inventory, pipeline orders and backorders carry into week t+1",
            fontsize=6.3, ha="center", va="bottom", color=INK, zorder=5)

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.015)
    print("wrote", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
