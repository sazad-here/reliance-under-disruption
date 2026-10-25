"""
Architecture figure: how the AI model is built (a) and how it is used in the experiment (b).

Visual grammar taken from model figures in the literature: a stage-by-stage pipeline
(Ntakolia et al. 2021 Fig. 1; Ahmed et al. 2025 Fig. 1), boosting drawn as a chain of
decision trees summed into one score (the usual depiction of gradient-boosted trees,
Ke et al. 2017), a node-and-arrow supply network for the twin, and concrete sizes written
into the diagram (Nie et al. 2023 Fig. 1).

Every mini-graphic is drawn from real project data, not decoration:
  M5 sparkline      data/processed/m5_weekly.npy (series 0, the session series)
  lead-time bars    fitted Olist delay pmf (fitted_params.json)
  disruption ticks  a calendar drawn at the fitted DataCo rate (make_corpus.random_calendar)
  episode card      the first corpus episode, stockout weeks marked
  feature grid      the live model's 30 inputs for one real week (standardised)
  trees             the first three trees of the frozen live model, leaf-wise structure
  ROC curve         the frozen live model on the two held-out session series (AUROC 0.902)

Layout is in inches and fonts in points, so the figure prints exactly at its width.

Run:  python lightgbm-pivot/make_architecture_figure.py
Out:  figures/paper/main_text/Figure_model_pipeline.svg/.png  (4.8 in print width)
      figures/paper/Fig00_model_pipeline.svg/.png               (6.3 in, supplementary)
"""
from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "figures", "paper")

INK, INK2, MUTE = "#1a1a19", "#52514e", "#9c9b95"
BLUE, BLUE_T = "#2a78d6", "#e4eefb"
VIOLET, VIOLET_T = "#5b4bc4", "#ecebfa"
RED = "#d03b3b"
GREY_T, EDGE = "#f1f0ec", "#8a8984"


def rj(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ real data for the graphics
def gather():
    import live_model
    import make_corpus as mc
    from baselines_strong import load_corpus
    from interface.world import SESSION_SERIES
    from sklearn.metrics import roc_auc_score, roc_curve

    f = rj("data/processed/fitted_params.json")
    meta = rj("models/live_risk_meta.json")
    sim = rj("lightgbm-pivot/simulator_results.json")
    cal = rj("data/processed/calibration_real.json")
    weekly = np.load(os.path.join(ROOT, "data", "processed", "m5_weekly.npy"))

    df = load_corpus()
    ep = df[df["episode"] == df["episode"].iloc[0]]
    rng = np.random.default_rng(3)
    calendar = [e.period for e in mc.random_calendar(rng, 100)]

    booster = lgb.Booster(model_file=live_model.MODEL_PATH)
    info = booster.dump_model()["tree_info"]
    trees = [info[0], info[1], info[-1]]   # first, second and last tree

    data = live_model._build(df)
    data = data[data["_valid"] == 1]
    held = data[data["_series"].isin(SESSION_SERIES)]
    cols = meta["features"]
    raw = booster.predict(held[cols])
    a, b = meta["platt"]
    z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    p = 1 / (1 + np.exp(-(a * z + b)))
    fpr, tpr, _ = roc_curve(held["_y"], p)
    # one real week's inputs, standardised against the training rows
    X = data[cols]
    wk = X.iloc[len(X) // 3]
    zrow = ((wk - X.mean()) / X.std().replace(0, 1)).fillna(0).clip(-2, 2).values

    return dict(
        spark=weekly[SESSION_SERIES[0], :], pmf=f["olist"]["delay_pmf"], calendar=calendar,
        ep_inv=ep["inventory"].values, ep_so=ep["stockout"].values,
        trees=trees, n_trees=meta["rounds"], zrow=zrow, n_feat=len(cols),
        fpr=fpr, tpr=tpr, auc_held=float(roc_auc_score(held["_y"], p)), auc_cv=meta["cv"]["auroc"],
        series=f["m5"]["n_series"], weeks=f["m5"]["n_weeks"],
        olist=f["olist"]["preprocessing"]["rows_after_outlier_cut"],
        dis=100 * f["dataco"]["disruption_rate_per_period"], z=cal["z_calibrated"],
        episodes=sim["profile"]["episodes"], rows=sim["profile"]["rows_total"],
        pos=100 * sim["profile"]["positive_rate"], ece=meta["cv"]["ece"],
        a=meta["platt"][0], b=meta["platt"][1],
        leaves=live_model.PARAMS["num_leaves"], eta=live_model.PARAMS["learning_rate"],
        nominal=f["olist"]["lead_time_nominal"], delta=cal["delta_5pct"],
    )


# ------------------------------------------------------------------ drawing helpers
def rbox(ax, x, y, w, h, fc="white", ec=EDGE, lw=0.7, r=0.03, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%s" % r,
                                facecolor=fc, edgecolor=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, a, b, color=INK2, lw=0.9, rad=0.0, ms=6, z=3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle="arc3,rad=%s" % rad, shrinkA=0, shrinkB=0, zorder=z))


def txt(ax, x, y, s, fs=5.6, **kw):
    kw.setdefault("color", INK)
    ax.text(x, y, s, fontsize=fs, **kw)


def stage(ax, x, top, w, h, title, highlight=False):
    head = 0.17
    ec = BLUE if highlight else EDGE
    rbox(ax, x, top - h, w, h, ec=ec, lw=1.0 if highlight else 0.7)
    rbox(ax, x, top - head, w, head, fc=BLUE if highlight else GREY_T, ec=ec, lw=1.0 if highlight else 0.7)
    txt(ax, x + w / 2, top - head / 2, title, fs=6.1, ha="center", va="center", fontweight="bold",
        color="white" if highlight else INK)
    return top - head  # top of the content area


def sparkline(ax, x, y, w, h, v, color=INK2, lw=0.6):
    v = np.asarray(v, float)
    xs = x + np.linspace(0, w, len(v))
    ys = y + (v - v.min()) / (np.ptp(v) or 1) * h
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=4)


def db_icon(ax, cx, cy, s=0.07, color=MUTE):
    for k in range(3):
        ax.add_patch(matplotlib.patches.Ellipse((cx, cy - k * s * 0.45), s, s * 0.35, fc="white",
                                                ec=color, lw=0.6, zorder=4))
    ax.plot([cx - s / 2, cx - s / 2], [cy, cy - 2 * s * 0.45], color=color, lw=0.6, zorder=4)
    ax.plot([cx + s / 2, cx + s / 2], [cy, cy - 2 * s * 0.45], color=color, lw=0.6, zorder=4)


def tree(ax, node, x, y, dx, dy, depth=0, max_depth=3):
    """Draw a LightGBM tree from its dump: splits as open dots, leaves as blue squares,
    and anything deeper than max_depth as a small grey triangle (a further subtree).
    Leaves that appear early show LightGBM's leaf-wise (unbalanced) growth."""
    if "leaf_index" in node:
        ax.add_patch(Rectangle((x - 0.011, y - 0.011), 0.022, 0.022, fc=BLUE, ec="none", zorder=5))
        return
    if depth >= max_depth:
        ax.add_patch(matplotlib.patches.Polygon([[x, y + 0.014], [x - 0.013, y - 0.011], [x + 0.013, y - 0.011]],
                                                closed=True, fc=MUTE, ec="none", zorder=5))
        return
    ax.add_patch(Circle((x, y), 0.014, fc="white", ec=INK2, lw=0.6, zorder=5))
    for side, child in ((-1, node["left_child"]), (1, node["right_child"])):
        cx, cy = x + side * dx, y - dy
        ax.plot([x, cx], [y, cy], color=MUTE, lw=0.5, zorder=4)
        tree(ax, child, cx, cy, dx * 0.55, dy, depth + 1, max_depth)


def person(ax, cx, cy, s=0.06, color=BLUE):
    ax.add_patch(Circle((cx, cy + s * 0.9), s * 0.45, fc=color, ec="none", zorder=5))
    ax.add_patch(FancyBboxPatch((cx - s * 0.7, cy - s * 0.7), s * 1.4, s * 1.1,
                                boxstyle="round,pad=0,rounding_size=%s" % (s * 0.5), fc=color, ec="none", zorder=5))


# ------------------------------------------------------------------ the figure
def draw(W, path, d):
    M = 0.03
    names = ["1  Open data", "2  Digital twin", "3  Training data", "4  Model inputs",
             "5  LightGBM risk model", "6  Validation"]
    rel = np.array([1.0, 1.0, 1.0, 0.95, 1.55, 1.0])
    gap = 0.075
    widths = rel / rel.sum() * (W - 2 * M - 5 * gap)
    xs = M + np.concatenate([[0], np.cumsum(widths[:-1] + gap)])
    ha = 1.36     # stage height
    hb = 0.78     # loop row height
    H = 0.16 + ha + 0.46 + 0.16 + hb + 0.30

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    top_a = H - 0.16
    txt(ax, M, H - 0.02, "a  Building the AI model", fs=7.5, fontweight="bold", va="top")
    tops = [stage(ax, x, top_a, w, ha, t, highlight=(i == 4)) for i, (x, w, t) in enumerate(zip(xs, widths, names))]
    for i in range(5):
        yy = top_a - ha / 2
        arrow(ax, (xs[i] + widths[i], yy), (xs[i + 1], yy))
    cap = lambda i, lines: [txt(ax, xs[i] + 0.04, top_a - ha + 0.04 + (len(lines) - 1 - k) * 0.095, ln, fs=5.4, va="bottom")
                            for k, ln in enumerate(lines)]

    # 1 open data: three real mini-series
    x, w, t = xs[0], widths[0], tops[0]
    gx, gw = x + 0.16, w - 0.21
    db_icon(ax, x + 0.08, t - 0.10)
    sparkline(ax, gx, t - 0.17, gw, 0.10, d["spark"], color=BLUE)
    txt(ax, gx, t - 0.21, "M5 demand", fs=5.0, color=INK2, va="top")
    db_icon(ax, x + 0.08, t - 0.36)
    for k, pmf in enumerate(d["pmf"]):
        bw = gw / 4 * 0.7
        ax.add_patch(Rectangle((gx + k * gw / 4, t - 0.45), bw, pmf * 0.17, fc=INK2, ec="none", zorder=4))
    txt(ax, gx, t - 0.47, r"$P(L=%d+k)$" % d["nominal"], fs=5.2, color=INK2, va="top")
    db_icon(ax, x + 0.08, t - 0.62)
    ax.plot([gx, gx + gw], [t - 0.66, t - 0.66], color=MUTE, lw=0.5, zorder=4)
    for c in d["calendar"]:
        xx = gx + c / 100 * gw
        ax.plot([xx, xx], [t - 0.69, t - 0.62], color=RED, lw=0.8, zorder=4)
    txt(ax, gx, t - 0.72, "disruptions", fs=5.0, color=INK2, va="top")
    cap(0, ["%d M5 series" % d["series"], "%s orders" % f"{d['olist']:,}", "%.1f %% disrupted" % d["dis"]])

    # 2 digital twin: supply network
    x, w, t = xs[1], widths[1], tops[1]
    cy = t - 0.33
    nodes = [(x + 0.12, "S"), (x + w / 2, "W"), (x + w - 0.12, "C")]
    for (nx, lab) in nodes:
        r = 0.085 if lab == "W" else 0.06
        ax.add_patch(Circle((nx, cy), r, fc=BLUE_T if lab == "W" else "white", ec=BLUE if lab == "W" else INK2, lw=0.8, zorder=5))
        txt(ax, nx, cy, lab, fs=5.8, ha="center", va="center", fontweight="bold", zorder=6)
    arrow(ax, (nodes[0][0] + 0.06, cy), (nodes[1][0] - 0.085, cy), ms=5, lw=0.8)
    arrow(ax, (nodes[1][0] + 0.085, cy), (nodes[2][0] - 0.06, cy), ms=5, lw=0.8)
    # disruption bolt on the supply arc
    bx = (nodes[0][0] + nodes[1][0]) / 2
    ax.plot([bx - 0.015, bx + 0.01, bx - 0.005, bx + 0.02], [cy - 0.02, cy - 0.07, cy - 0.07, cy - 0.12], color=RED, lw=0.9, zorder=6)
    txt(ax, x + w / 2, cy - 0.17, "8-variable state", fs=5.0, ha="center", color=INK2)
    txt(ax, x + w / 2, cy - 0.26, "red = disruption", fs=4.8, ha="center", color=RED)
    txt(ax, x + w / 2, cy - 0.36, r"$Q^*_t = S - IP_t$", fs=5.4, ha="center", color=INK)
    cap(1, ["S supplier", "W warehouse", "C customer"])

    # 3 training data: stacked episode cards, top card = a real episode
    x, w, t = xs[2], widths[2], tops[2]
    for k in (2, 1, 0):
        cx0, cy0 = x + 0.06 + k * 0.035, t - 0.62 + k * 0.05
        rbox(ax, cx0, cy0, w - 0.19, 0.42, fc="white", ec=MUTE, lw=0.6, r=0.015, z=3 + (2 - k))
    cx0, cy0, cw = x + 0.06, t - 0.62, w - 0.19
    inv = d["ep_inv"]
    xs_ep = cx0 + 0.03 + np.linspace(0, cw - 0.06, len(inv))
    ys_ep = cy0 + 0.10 + (inv - inv.min()) / (np.ptp(inv) or 1) * 0.22
    ax.plot(xs_ep, ys_ep, color=INK2, lw=0.6, zorder=9)
    so = np.flatnonzero(d["ep_so"])
    for xx in xs_ep[so]:
        ax.plot([xx, xx], [cy0 + 0.035, cy0 + 0.085], color=RED, lw=0.7, zorder=9)
    txt(ax, cx0 + 0.03, cy0 + 0.39, "stock, 100 weeks", fs=4.8, color=INK2, va="top", zorder=9)
    txt(ax, x + w / 2, cy0 - 0.03, "red = stockout week", fs=4.8, color=RED, ha="center", va="top", zorder=9)
    cap(2, ["%d episodes" % d["episodes"], "= %s weeks" % f"{d['rows']:,}", r"$y_t$: 2-wk stockout"])

    # 4 model inputs: the 30 features of one real week as a grid (6 x 5)
    x, w, t = xs[3], widths[3], tops[3]
    zr = d["zrow"]
    ncol, nrow = 5, int(np.ceil(len(zr) / 5))
    cs = min((w - 0.12) / ncol, 0.54 / nrow)
    gx0 = x + (w - cs * ncol) / 2
    cmap = matplotlib.colormaps["RdBu_r"]
    for i, v in enumerate(zr):
        r_, c_ = divmod(i, ncol)
        ax.add_patch(Rectangle((gx0 + c_ * cs, t - 0.12 - (r_ + 1) * cs), cs * 0.9, cs * 0.9,
                               fc=cmap(0.5 + v / 4.4), ec="none", zorder=4))
    txt(ax, x + w / 2, t - 0.12 - nrow * cs - 0.04, "one real week", fs=4.8, ha="center", va="top", color=INK2)
    cap(3, [r"$\mathbf{x}_t \in \mathbb{R}^{%d}$:" % d["n_feat"], "state + 4-week", "history"])

    # 5 LightGBM: first three real trees, then ..., sum, calibration, risk
    x, w, t = xs[4], widths[4], tops[4]
    slots = [0.18, 0.38, 0.58, 0.82]          # tree 1, tree 2, "...", last tree (fractions of w)
    labels = ["tree 1", "tree 2", None, "tree %d" % d["n_trees"]]
    trees = [d["trees"][0], d["trees"][1], None, d["trees"][2]]
    for f_, lab, tr in zip(slots, labels, trees):
        tx = x + f_ * w
        if tr is None:
            txt(ax, tx, t - 0.22, "+ … +", fs=6.2, ha="center", va="center", color=INK2)
            continue
        tree(ax, tr["tree_structure"], tx, t - 0.10, w * 0.055, 0.085, max_depth=3)
        txt(ax, tx, t - 0.44, lab, fs=4.8, ha="center", color=INK2)
    txt(ax, x + (slots[0] + slots[1]) / 2 * w, t - 0.22, "+", fs=7, ha="center", va="center", color=INK2)
    # the model's probability function: calibrated sum of tree scores
    sy, bh = t - 0.62, 0.27
    rbox(ax, x + 0.05, sy - bh / 2, w - 0.10, bh, fc=BLUE_T, ec=BLUE, lw=0.8, r=0.02)
    txt(ax, x + w / 2, sy + 0.055, r"$\hat{p}_t=\sigma\,(a\,F(\mathbf{x}_t)+b)$", fs=6.6, ha="center", va="center")
    txt(ax, x + w / 2, sy - 0.06, r"$F=f_1+f_2+\dots+f_{%d}$" % d["n_trees"], fs=5.8, ha="center", va="center",
        color=INK2)
    cap(4, ["%d leaves, $\\eta$ = %.2f, log-loss" % (d["leaves"], d["eta"]),
            "each tree fits the residuals",
            "Platt a = %.3f, b = %.3f" % (d["a"], d["b"])])

    # 6 validation: real ROC of the frozen model on the held-out session series
    x, w, t = xs[5], widths[5], tops[5]
    s = min(w - 0.16, 0.55)
    rx, ry = x + (w - s) / 2 + 0.03, t - 0.10 - s
    ax.add_patch(Rectangle((rx, ry), s, s, fc="white", ec=MUTE, lw=0.5, zorder=3))
    ax.plot([rx, rx + s], [ry, ry + s], color=MUTE, lw=0.5, ls=(0, (2, 2)), zorder=4)
    ax.plot(rx + d["fpr"] * s, ry + d["tpr"] * s, color=BLUE, lw=1.0, zorder=5)
    txt(ax, rx + s * 0.95, ry + s * 0.08, "ROC", fs=5.0, ha="right", color=INK2)
    txt(ax, rx - 0.02, ry + s / 2, "TPR", fs=4.6, rotation=90, ha="right", va="center", color=INK2)
    txt(ax, rx + s / 2, ry - 0.02, "FPR", fs=4.6, ha="center", va="top", color=INK2)
    cap(5, ["AUROC CV %.3f" % d["auc_cv"], "unseen %.3f" % d["auc_held"], "ECE %.3f" % d["ece"]])

    # ---------------------------------------------------------------- b: the weekly loop
    top_b = top_a - ha - 0.46 - 0.16
    txt(ax, M, top_b + 0.16 - 0.02, "b  Using it in the experiment (every week of 30)", fs=7.5, fontweight="bold", va="top")
    bw_ = (W - 2 * M - 3 * 0.17) / 4
    bx = [M + i * (bw_ + 0.17) for i in range(4)]
    yb = top_b - hb
    # 1 state
    rbox(ax, bx[0], yb, bw_, hb)
    person(ax, bx[0] + 0.11, yb + hb - 0.17, s=0.055, color=MUTE)
    txt(ax, bx[0] + 0.22, yb + hb - 0.10, "Twin state shown", fs=6.0, fontweight="bold", va="top")
    for k, ln in enumerate(["demand, stock, orders,", "lead time, disruption,", "past-week trends"]):
        txt(ax, bx[0] + 0.06, yb + 0.44 - k * 0.1, ln, fs=5.4, va="top")
    # 2 own order (locked)
    rbox(ax, bx[1], yb, bw_, hb, ec=BLUE, lw=0.9)
    txt(ax, bx[1] + 0.06, yb + hb - 0.10, r"Own order $Q_{own}$", fs=6.0, fontweight="bold", va="top")
    rbox(ax, bx[1] + 0.08, yb + 0.30, bw_ - 0.40, 0.17, fc=GREY_T, ec=MUTE, lw=0.6, r=0.02)
    txt(ax, bx[1] + 0.13, yb + 0.385, "150", fs=6.0, va="center", color=INK2)
    # padlock
    lx, ly = bx[1] + bw_ - 0.22, yb + 0.33
    ax.add_patch(Rectangle((lx, ly), 0.09, 0.07, fc=BLUE, ec="none", zorder=5))
    ax.add_patch(matplotlib.patches.Arc((lx + 0.045, ly + 0.07), 0.06, 0.08, theta1=0, theta2=180, color=BLUE, lw=1.0, zorder=5))
    txt(ax, bx[1] + 0.06, yb + 0.20, "entered first, then locked:", fs=5.4, va="top")
    txt(ax, bx[1] + 0.06, yb + 0.11, "the unaided baseline", fs=5.4, va="top")
    # 3 AI advice with risk meter
    rbox(ax, bx[2], yb, bw_, hb, fc=VIOLET_T, ec=VIOLET, lw=1.0)
    txt(ax, bx[2] + 0.06, yb + hb - 0.10, "AI advice", fs=6.0, fontweight="bold", va="top", color=VIOLET)
    txt(ax, bx[2] + 0.06, yb + 0.47, r"order $Q_{AI}$ = 167", fs=5.6, va="center", fontweight="bold", color=VIOLET)
    mx, my, mw = bx[2] + 0.06, yb + 0.30, bw_ - 0.12
    ax.add_patch(Rectangle((mx, my), mw, 0.05, fc="white", ec=VIOLET, lw=0.5, zorder=4))
    ax.add_patch(Rectangle((mx, my), mw * 0.34, 0.05, fc=VIOLET, ec="none", zorder=5))
    for f_ in (0.25, 0.55):
        ax.plot([mx + mw * f_] * 2, [my - 0.015, my + 0.065], color=VIOLET, lw=0.5, zorder=6)
    txt(ax, mx, my - 0.03, r"$\hat{p}_t$ = 34 %: live model", fs=5.2, va="top", color=VIOLET)
    txt(ax, bx[2] + 0.06, yb + 0.11, r"correct iff $|Q_{AI}-Q^*_t| \leq %d$" % d["delta"], fs=5.2, va="top", color=INK)
    # 4 final order
    rbox(ax, bx[3], yb, bw_, hb, ec=BLUE, lw=0.9)
    txt(ax, bx[3] + 0.06, yb + hb - 0.10, r"Final order $Q_{final}$", fs=6.0, fontweight="bold", va="top")
    rbox(ax, bx[3] + 0.08, yb + 0.36, bw_ - 0.40, 0.15, fc="white", ec=BLUE, lw=0.8, r=0.02)
    txt(ax, bx[3] + 0.13, yb + 0.435, "160", fs=6.0, va="center", color=BLUE)
    txt(ax, bx[3] + 0.06, yb + 0.30, "weight of advice:", fs=5.2, va="top")
    txt(ax, bx[3] + 0.06, yb + 0.14,
        r"$\mathrm{WoA}=\dfrac{Q_{final}-Q_{own}}{Q_{AI}-Q_{own}}$", fs=6.0, va="center")
    for i in range(3):
        arrow(ax, (bx[i] + bw_, yb + hb / 2), (bx[i + 1], yb + hb / 2))
    arrow(ax, (bx[3] + bw_ / 2, yb), (bx[0] + bw_ / 2, yb), rad=-0.07)
    txt(ax, W / 2, 0.03, "next week: orders, stock and backorders carry forward in the twin",
        fs=5.4, ha="center", va="bottom", color=INK2, style="italic")

    # the frozen model feeds the risk estimate every week
    src = (xs[4] + widths[4] * 0.8, top_a - ha)
    dst = (bx[2] + bw_ * 0.65, yb + hb)
    arrow(ax, src, dst, color=BLUE, lw=1.1)
    txt(ax, src[0] + 0.05, (src[1] + dst[1]) / 2 + 0.06, "frozen model runs on each\nparticipant's own history",
        fs=5.4, color=BLUE, va="center", ha="left", linespacing=1.15)
    txt(ax, dst[0] - 0.33, (src[1] + dst[1]) / 2 + 0.06, "AI errors placed on\npropagating or absorbed weeks",
        fs=5.4, color=VIOLET, va="center", ha="right", linespacing=1.15)

    for ext in ("svg", "png"):
        fig.savefig(path + "." + ext, dpi=300)
    plt.close(fig)
    print("  %s  (%.2f x %.2f in)" % (os.path.relpath(path, ROOT), W, H))


if __name__ == "__main__":
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    os.makedirs(os.path.join(OUT, "main_text"), exist_ok=True)
    data = gather()
    print("  live model on held-out session series: AUROC %.3f" % data["auc_held"])
    draw(4.8, os.path.join(OUT, "main_text", "Figure_model_pipeline"), data)
    draw(6.3, os.path.join(OUT, "Fig00_model_pipeline"), data)
