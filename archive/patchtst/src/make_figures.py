"""
Publication-style comparison figures across all ten models tested.

WHY THIS EXISTS

Every number behind these bars is already in results/*.json, produced by
baselines_strong.py, patchtst_v2.py and extra_baselines.py -- this script
does no new computation, it only renders what's already there, the same
"every number traces to a script" discipline RESULTS.md holds everything
else to.

THE ONE THING THIS FIGURE IS BUILT TO SHOW HONESTLY

The ten models cluster inside a band narrower than the noise floor
eval_power.py already measured (~0.079 PR-AUC, series-clustered bootstrap).
That band is drawn explicitly on the PR-AUC panel, not left for the reader to
infer from bar heights alone -- a bar chart with ten similar-looking bars and
no reference to "how much difference would even be real" invites exactly the
"tallest bar wins" misreading this project has spent two sessions correcting.

Colors follow the ECC dataviz skill's validated default palette (categorical
slots assigned by MODEL FAMILY, not cycled per model -- ten models would
blow through the safe categorical range otherwise): blue=linear/kernel,
orange=tree ensemble, aqua=tabular deep net, yellow=convolutional,
magenta=recurrent, green=transformer family.

Run:  python src/make_figures.py
Out:  results/figures/model_comparison.png (and .svg)
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT_DIR = os.path.join(RESULTS, "figures")

# ECC dataviz validated default palette (light mode) -- assigned by family
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7",
)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

FAMILY_COLOR = {
    "linear/kernel": BLUE,
    "tree ensemble": ORANGE,
    "tabular deep net": AQUA,
    "convolutional": YELLOW,
    "recurrent": MAGENTA,
    "transformer": GREEN,
}

# (display name, family, json source, json key or None-for-mean-of-runs)
MODELS = [
    ("Logistic\nregression", "linear/kernel", "baselines_strong", "logistic regression"),
    ("LightGBM", "tree ensemble", "baselines_strong", "LightGBM"),
    ("SVM", "linear/kernel", "extra", "svm"),
    ("MLP\n(tabular)", "tabular deep net", "extra", "mlp_tabular"),
    ("CNN\n(1D)", "convolutional", "extra", "cnn"),
    ("JointPatch\nTransformer", "transformer", "extra", "joint_patch_transformer"),
    ("LSTM", "recurrent", "extra", "lstm"),
    ("Transformer\n(vanilla)", "transformer", "extra", "transformer_vanilla"),
    ("GRU", "recurrent", "extra", "gru"),
    ("PatchTST v2", "transformer", "patchtst_v2", None),
]


def load_all():
    with open(os.path.join(RESULTS, "baselines_strong.json"), encoding="utf-8") as fh:
        strong = {r["model"]: r for r in json.load(fh)["results"]}
    with open(os.path.join(RESULTS, "extra_baselines.json"), encoding="utf-8") as fh:
        extra = json.load(fh)
    with open(os.path.join(RESULTS, "patchtst_v2.json"), encoding="utf-8") as fh:
        pt = json.load(fh)["runs"]

    rows = []
    for name, family, src, key in MODELS:
        if src == "baselines_strong":
            r = strong[key]
        elif src == "extra":
            r = extra[key]
        else:  # patchtst_v2: mean across 5 seeds
            r = {m: float(np.mean([run[m] for run in pt]))
                 for m in ("accuracy", "auroc", "pr_auc", "f1", "ece")}
        rows.append({"name": name, "family": family, **r})
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_all()
    rows.sort(key=lambda r: r["auroc"], reverse=True)  # fixed order, all panels

    names = [r["name"] for r in rows]
    colors = [FAMILY_COLOR[r["family"]] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

    panels = [
        ("auroc", "AUROC", axes[0, 0], True, None),
        ("pr_auc", "PR-AUC", axes[0, 1], True, 0.079),
        ("f1", "F1", axes[1, 0], True, None),
        ("ece", "ECE  (lower is better)", axes[1, 1], False, None),
    ]

    for key, title, ax, higher_better, noise_band in panels:
        vals = [r[key] for r in rows]
        ax.set_facecolor(SURFACE)
        ax.bar(x, vals, color=colors, width=0.62, zorder=3)

        # noise-floor band: PR-AUC only, drawn from the top bar down by the
        # measured series-clustered bootstrap threshold (results/eval_power.json,
        # RESULTS.md §6/§10) -- everything inside it is not distinguishable.
        if noise_band is not None:
            top = max(vals)
            ax.axhspan(top - noise_band, top, color=BLUE, alpha=0.08, zorder=1)
            ax.axhline(top - noise_band, color=BLUE, lw=1, ls=(0, (3, 2)), zorder=2)
            ax.text(len(rows) - 0.4, top - noise_band + 0.003,
                     "within noise floor (±%.3f, series bootstrap)" % noise_band,
                     fontsize=8.5, color=BLUE, ha="right", va="bottom", style="italic")

        for xi, v in zip(x, vals):
            ax.text(xi, v + (0.003 if higher_better else 0.0006), "%.3f" % v,
                     ha="center", va="bottom", fontsize=8, color=INK)

        ax.set_title(title, fontsize=12, color=INK, fontweight="bold", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, color="#52514e")
        ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        lo = min(vals); hi = max(vals)
        pad = (hi - lo) * 0.35 + 0.01
        ax.set_ylim(max(0, lo - pad * 0.4), hi + pad)

    # one shared legend, family colors, order matches the palette's validated slots
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    fig.legend(handles, list(FAMILY_COLOR.keys()), loc="lower center", ncol=6,
               frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Ten architectures, one unified series-held-out test set (fold 0) — "
        "stockout-within-2-periods classification",
        fontsize=13, color=INK, y=1.01, fontweight="bold",
    )
    fig.text(0.5, 0.965,
              "Corpus fingerprint f953645593cab9a1  ·  logistic regression / "
              "LightGBM / PatchTST v2 use their own multi-run protocols; the "
              "rest are single-seed, early-stopped (see RESULTS.md §6, §10)",
              ha="center", fontsize=8.5, color=MUTED, style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 0.955])

    png = os.path.join(OUT_DIR, "model_comparison.png")
    svg = os.path.join(OUT_DIR, "model_comparison.svg")
    fig.savefig(png, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(svg, facecolor=SURFACE, bbox_inches="tight")
    print("written -> %s" % png)
    print("written -> %s" % svg)


if __name__ == "__main__":
    main()
