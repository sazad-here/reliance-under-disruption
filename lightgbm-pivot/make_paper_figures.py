"""
Paper figures for the project, as SVG (vector, for Word) and PNG (300 dpi).

Each figure follows a chart form used by one of the 14 papers in papers/ and
puts OUR results next to the results those papers PUBLISHED. Every number is
read from a result file produced by a script in this folder or src/; the only
hand-entered numbers are the published values, each with its table reference.

    Fig 1  dataset profile ("demographics")      Ahmed 2025 Fig 11-12, Islam & Amin 2020 Fig 1
    Fig 2  feature correlation heatmap            Ahmed 2025 Fig 10
    Fig 3  ROC and precision-recall curves        Ntakolia 2021 Fig 5
    Fig 4  calibration curves, slope, intercept   van den Goorbergh 2022 Fig 1, 4, 5; Li 2026 Fig 1
    Fig 5  confusion matrices, ours vs published  Ahmed 2025 Fig 14; Ntakolia 2021 Table 3
    Fig 6  SHAP importance and beeswarm           Ntakolia 2021 Fig 6-7; Ahmed 2025 Fig 15-16
    Fig 7  ten models, 95 % CI, epoch 50 vs 100   Makridakis 2022 Fig 3
    Fig 8  validation AUROC by epoch              Chen 2016 Fig 3; Nie 2023 Fig 7
    Fig 9  backorder benchmark vs published       Ntakolia 2021 Table 3
    Fig 10 M5 leaderboard                         Makridakis 2022 Fig 2
    Fig 11 Capture@10 %, DataCo and Olist         Li 2026 Table 2
    Fig 12 published accuracy vs no-skill floor   Thaker 2026, Ahmed 2025, Islam & Amin 2020, Ntakolia 2021

Inputs: chart_data.json + chart_oof.npz (run chart_data.py first), epoch_study.json,
backorder_benchmark.json, xgb_gpu_benchmark.json, m5_results.json,
li2026_results.json, simulator_results.json, models/live_risk_*, the corpus.

Run:  python lightgbm-pivot/make_paper_figures.py
Out:  figures/paper/FigNN_*.svg and .png
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "figures", "paper")

# Validated categorical palette (dataviz validator: CVD and normal-vision pass).
# Colour follows the model on every figure; line style and marker repeat the
# identity for greyscale print.
C = {"LightGBM": "#2a78d6", "Logistic regression": "#eb6834",
     "XGBoost": "#1baf7a", "Net position only": "#eda100"}
LS = {"LightGBM": "-", "Logistic regression": "--", "XGBoost": "-.", "Net position only": ":"}
MK = {"LightGBM": "o", "Logistic regression": "s", "XGBoost": "^", "Net position only": "D"}
PUB = "#8c8b86"      # published values
MUTED = "#b9b8b2"    # other models
INK, INK2, GRID = "#1a1a19", "#52514e", "#e1e0d9"
FULL, HALF = 6.3, 3.1  # inches: full text width, one column

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "svg.fonttype": "none", "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  %s" % name)


def panel(ax, letter):
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


# --------------------------------------------------------------- published values
# Ntakolia et al. 2021, Sensors 21:7926, Table 3 (balanced test set, 5,588 rows).
# Confusion matrices are [[TN, FP], [FN, TP]]; accuracy re-derived from them matches the table.
NTAKOLIA = {
    "RF": (88.82, 89.94, 89.01, 88.10, [[2432, 342], [283, 2531]]),
    "KNN": (75.93, 79.82, 76.96, 74.30, [[1997, 777], [568, 2246]]),
    "NN (MLP)": (85.68, 85.54, 85.75, 85.96, [[2381, 393], [407, 2407]]),
    "LR": (70.22, 74.09, 71.48, 69.04, [[1839, 935], [729, 2085]]),
    "SVM": (72.39, 85.86, 75.80, 67.85, [[1629, 1145], [398, 2416]]),
    "XGBoost": (88.53, 90.26, 88.80, 87.38, [[2407, 367], [274, 2540]]),
    "LightGBM": (87.78, 89.02, 88.00, 87.01, [[2400, 374], [309, 2505]]),
    "Balanced Bagging": (88.85, 90.69, 89.12, 87.61, [[2413, 361], [262, 2552]]),
}  # (accuracy, recall, F1, precision, CM)


# =================================================================== figures
def fig1_profile():
    import live_model
    from baselines_strong import load_corpus
    df = load_corpus()
    d = live_model._build(df)
    d = d[d["_valid"] == 1]
    sim = load("simulator_results.json")["profile"]
    fig, ax = plt.subplots(2, 2, figsize=(FULL, 4.3))

    mix = sim["policy_mode_share"]
    order = sorted(mix, key=mix.get)
    a = ax[0, 0]
    a.barh(order, [100 * mix[k] for k in order], color=C["LightGBM"], height=0.55)
    for i, k in enumerate(order):
        a.text(100 * mix[k] + 0.8, i, "%.0f %%" % (100 * mix[k]), va="center", fontsize=7, color=INK2)
    a.set_xlabel("Share of 520 episodes (%)"); a.set_title("Ordering policy that generated each episode", loc="left")
    a.grid(axis="y", visible=False); a.set_xlim(0, 40); panel(a, "a")

    a = ax[0, 1]
    hi = float(df["demand"].quantile(0.998))
    a.hist(df["demand"].clip(upper=hi), bins=np.linspace(0, hi, 41), color=C["LightGBM"], edgecolor="white", linewidth=0.4)
    a.axvline(df["demand"].mean(), color=INK, lw=1)
    a.text(df["demand"].mean() + 8, a.get_ylim()[1] * 0.92, "mean %.0f" % df["demand"].mean(), fontsize=7)
    a.set_xlim(0, hi)
    a.set_xlabel("Weekly demand (units, M5 rescaled)"); a.set_ylabel("Weeks")
    a.set_title("Demand distribution, 52,000 weeks", loc="left"); panel(a, "b")

    rate = d.groupby("_series")["_y"].mean().sort_values().values * 100
    a = ax[1, 0]
    a.bar(np.arange(len(rate)), rate, color=C["LightGBM"], width=0.8)
    a.axhline(rate.mean(), color=INK, lw=1)
    a.text(1, rate.mean() + 2, "mean %.1f %%" % rate.mean(), fontsize=7)
    a.set_xlabel("M5 store-department series (sorted)"); a.set_ylabel("Stockout within 2 weeks (%)")
    a.set_title("Stockout rate differs by series (mean of 70)", loc="left"); a.set_xticks([0, 34, 69], ["1", "35", "70"])
    panel(a, "c")

    a = ax[1, 1]
    groups = [d.loc[d["_y"] == 0, "net_position"].values, d.loc[d["_y"] == 1, "net_position"].values]
    bp = a.boxplot(groups, vert=False, widths=0.5, showfliers=False, patch_artist=True,
                   medianprops={"color": INK, "lw": 1.2})
    for patch, col in zip(bp["boxes"], (MUTED, C["LightGBM"])):
        patch.set_facecolor(col); patch.set_edgecolor(INK2)
    a.set_yticks([1, 2], ["No stockout\n(n=%s)" % f"{len(groups[0]):,}", "Stockout\n(n=%s)" % f"{len(groups[1]):,}"])
    a.axvline(0, color=INK2, lw=0.8)
    a.set_xlabel("Net position (units)")
    a.set_title("Net position separates the two outcomes", loc="left"); a.grid(axis="y", visible=False)
    panel(a, "d")
    fig.tight_layout(h_pad=2.2)
    save(fig, "Fig01_dataset_profile")


def fig2_correlation():
    import live_model
    from baselines_strong import load_corpus
    d = live_model._build(load_corpus())
    d = d[d["_valid"] == 1]
    cols = {"demand": "Demand", "inventory": "Inventory", "on_order": "On order", "backorders": "Backorders",
            "lead_time": "Lead time", "supplier_reliability": "Supplier reliability",
            "transport_ok": "Transport available", "disruption_flag": "Disruption",
            "net_position": "Net position", "cover_ratio": "Cover ratio", "demand_std4": "Demand SD (4 wk)",
            "_y": "Stockout within 2 wk"}
    m = d[list(cols)].corr(method="spearman").values
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, a = plt.subplots(figsize=(4.6, 4.0))
    im = a.imshow(m, cmap=cmap, vmin=-1, vmax=1)
    a.set_xticks(range(len(cols)), cols.values(), rotation=45, ha="right")
    a.set_yticks(range(len(cols)), cols.values()); a.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            a.text(j, i, "%.2f" % m[i, j], ha="center", va="center", fontsize=5.6,
                   color="white" if abs(m[i, j]) > 0.6 else INK)
    for s in a.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("Spearman correlation")
    save(fig, "Fig02_feature_correlation")


def fig3_roc_pr(cd):
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.9))
    order = ["LightGBM", "Logistic regression", "XGBoost", "Net position only"]
    for n in order:
        m = cd["models"][n]
        ax[0].plot(m["roc"]["fpr"], m["roc"]["tpr"], color=C[n], ls=LS[n], lw=1.6,
                   label="%s (AUROC %.3f)" % (n, m["auroc"]))
        ax[1].plot(m["pr"]["recall"], m["pr"]["precision"], color=C[n], ls=LS[n], lw=1.6,
                   label="%s  %.3f" % (n.replace("Logistic regression", "Logistic reg."), m["pr_auc"]))
    ax[0].plot([0, 1], [0, 1], color=MUTED, lw=0.8)
    ax[0].text(0.62, 0.55, "chance", color=INK2, fontsize=7, rotation=38)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax[0].set_title("ROC, 24,960 out-of-fold weeks", loc="left")
    base = cd["positive_rate"]
    ax[1].axhline(base, color=MUTED, lw=0.8)
    ax[1].text(0.02, base + 0.02, "no-skill (%.1f %% stockouts)" % (100 * base), fontsize=7, color=INK2)
    ax[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    ax[1].set_title("Precision-recall", loc="left")
    for a, L in zip(ax, "ab"):
        if L == "a":
            a.legend(loc="lower right")
        else:
            a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.25), title="PR-AUC", fontsize=6.8,
                     title_fontsize=6.8, alignment="left")
        a.set_aspect("equal"); panel(a, L)
    fig.tight_layout()
    save(fig, "Fig03_roc_pr_curves")


def fig4_calibration(cd):
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
    a.text(0.7, 0.62, "perfect", color=INK2, fontsize=7, rotation=38)
    for n in ["LightGBM", "Logistic regression", "XGBoost", "Net position only"]:
        pts = cd["models"][n]["calibration"]["points"]
        a.plot([p["pred"] for p in pts], [p["obs"] for p in pts], color=C[n], ls=LS[n], marker=MK[n],
               ms=4, lw=1.2, mec="white", mew=0.6, label=n)
    a.set(xlabel="Predicted probability (decile mean)", ylabel="Observed stockout rate", xlim=(0, 1), ylim=(0, 1))
    a.set_aspect("equal"); a.legend(loc="upper left"); a.set_title("Calibration by decile", loc="left")
    panel(a, "a")

    a = ax[1]
    names = ["Net position only", "Logistic regression", "XGBoost", "LightGBM"]
    y = np.arange(len(names))
    sl = [cd["models"][n]["calibration"]["slope"] for n in names]
    it = [cd["models"][n]["calibration"]["intercept"] for n in names]
    a.axvline(1, color=INK2, lw=0.8)
    a.scatter(sl, y + 0.12, color=[C[n] for n in names], marker="o", s=26, zorder=3, label="slope (ideal 1)")
    a.scatter(np.array(it) + 1, y - 0.12, facecolor="white", edgecolor=[C[n] for n in names],
              marker="s", s=24, lw=1.2, zorder=3, label="intercept + 1 (ideal 1)")
    for yy, s_, i_ in zip(y, sl, it):
        a.text(1.075, yy, "slope %.2f   int. %+.3f" % (s_, i_), va="center", fontsize=6.8, color=INK2)
    a.set_yticks(y, names); a.set_xlim(0.9, 1.26); a.grid(axis="y", visible=False)
    a.set_xticks([0.9, 0.95, 1.0, 1.05])
    a.set_title("Calibration slope and intercept", loc="left")
    a.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.14), ncol=2, fontsize=6.8)
    a.set_ylim(-0.6, len(names) - 0.4)
    panel(a, "b")
    fig.tight_layout()
    save(fig, "Fig04_calibration")


def _cm(a, m, title, sub):
    m = np.asarray(m, float)
    pct = m / m.sum(axis=1, keepdims=True) * 100
    cmap = LinearSegmentedColormap.from_list("seq", ["#eaf2fd", "#2a78d6", "#104281"])
    a.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            a.text(j, i, "%s\n%.1f %%" % (f"{int(m[i, j]):,}", pct[i, j]), ha="center", va="center",
                   fontsize=7.5, color="white" if pct[i, j] > 55 else INK)
    a.set_xticks([0, 1], ["No stockout", "Stockout"]); a.set_yticks([0, 1], ["No stockout", "Stockout"])
    a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title(title, loc="left"); a.text(0, 1.02, sub, transform=a.transAxes, fontsize=6.8, color=INK2)


def fig5_confusion(cd):
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.7))
    cm = cd["confusion_lightgbm"]
    _cm(ax[0], cm["matrix"], "Ours: LightGBM, simulator\n", "threshold %.2f (max F1), 24,960 weeks" % cm["threshold"])
    _cm(ax[1], NTAKOLIA["LightGBM"][4], "Published: LightGBM, Ntakolia 2021\n",
        "Table 3, balanced Kaggle test set, 5,588 SKUs")
    ax[1].set_xticks([0, 1], ["No backorder", "Backorder"]); ax[1].set_yticks([0, 1], ["No backorder", "Backorder"])
    panel(ax[0], "a"); panel(ax[1], "b")
    fig.tight_layout(w_pad=3)
    save(fig, "Fig05_confusion_matrices")


LABELS = {"net_position": "Net position", "cover_ratio": "Cover ratio", "inventory": "Inventory on hand",
          "demand_mean4": "Mean demand, 4 wk", "ip_lag2": "Inventory position, 2 wk ago",
          "disruption_flag": "Disruption this week", "lead_time": "Lead time", "demand_lag2": "Demand, 2 wk ago",
          "ip_change": "Change in inventory position", "demand_lag3": "Demand, 3 wk ago",
          "ip_lag3": "Inventory position, 3 wk ago", "ip_lag1": "Inventory position, 1 wk ago"}


def fig6_shap(cd):
    s = cd["shap_live"]
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    imp = s["importance"][::-1]
    a = ax[0]
    a.barh([LABELS.get(r["feature"], r["feature"]) for r in imp], [r["mean_abs"] for r in imp],
           color=C["LightGBM"], height=0.6)
    a.set_xlabel("Mean |SHAP| (log-odds)"); a.grid(axis="y", visible=False)
    a.set_title("Feature importance, live model", loc="left"); panel(a, "a")

    a = ax[1]
    rng = np.random.default_rng(0)
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#a9a8a3", "#e34948"])
    sw = s["swarm"][::-1]
    for i, row in enumerate(sw):
        v = np.array(row["shap"]); r = np.array(row["value_rank"])
        hist, edges = np.histogram(v, bins=40)
        dens = hist[np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)] / max(hist.max(), 1)
        jit = rng.uniform(-1, 1, len(v)) * 0.38 * dens
        sc = a.scatter(v, i + jit, c=r, cmap=cmap, vmin=0, vmax=1, s=5, lw=0, alpha=0.9)
    a.set_yticks(range(len(sw)), [LABELS.get(r["feature"], r["feature"]) for r in sw])
    a.axvline(0, color=INK2, lw=0.8); a.grid(axis="y", visible=False)
    a.set_xlabel("SHAP value (effect on log-odds of stockout)")
    a.set_title("Beeswarm, %d weeks never trained on" % len(s["swarm"][0]["shap"]), loc="left")
    cb = fig.colorbar(sc, ax=a, fraction=0.035, pad=0.02, ticks=[0, 1]); cb.outline.set_visible(False)
    cb.ax.set_yticklabels(["low", "high"]); cb.set_label("Feature value")
    panel(a, "b")
    fig.tight_layout()
    save(fig, "Fig06_shap")


SHORT = {"logistic regression": "Logistic regression", "XGBoost (CUDA)": "XGBoost",
         "Transformer (vanilla)": "Transformer", "JointPatchTransformer": "JointPatch Transformer"}


def fig7_ten_models():
    e = load("epoch_study.json")["results"]
    rows = sorted(e.items(), key=lambda kv: kv[1]["auroc_val_best"])
    names = [SHORT.get(k, k) for k, _ in rows]
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    a = ax[0]
    ref = e["LightGBM"]["auroc_val_best"]
    a.axvline(ref, color=C["LightGBM"], lw=0.8, alpha=0.6)
    for i, (k, v) in enumerate(rows):
        n = SHORT.get(k, k)
        col = C.get(n, MUTED if n != "LightGBM" else C["LightGBM"])
        tie = v["diff_vs_lightgbm_ci"] is not None and v["diff_vs_lightgbm_ci"][0] <= 0 <= v["diff_vs_lightgbm_ci"][1]
        a.plot(v["auroc_ci"], [i, i], color=col if n in C else INK2, lw=1.4)
        a.scatter(v["auroc_val_best"], i, color=col if n in C else INK2, s=28, zorder=3,
                  marker=MK.get(n, "o"), edgecolor="white", lw=0.6)
        tag = "reference" if k == "LightGBM" else ("tie" if tie else ("higher" if v["diff_vs_lightgbm_ci"][0] > 0 else "lower"))
        a.text(0.9115, i, tag, va="center", fontsize=6.5, color=INK2)
    a.set_yticks(range(len(rows)), names); a.set_xlim(0.83, 0.925); a.grid(axis="y", visible=False)
    a.set_xlabel("AUROC, series-clustered 95 % CI")
    a.set_title("Ten models, identical test weeks", loc="left"); panel(a, "a")

    a = ax[1]
    for i, (k, v) in enumerate(rows):
        n = SHORT.get(k, k)
        col = C.get(n, INK2)
        a.plot([v["auroc_ep50"], v["auroc_ep100"]], [i, i], color=col, lw=1.2)
        a.scatter(v["auroc_ep50"], i, facecolor="white", edgecolor=col, s=22, zorder=3, lw=1.1)
        a.scatter(v["auroc_ep100"], i, color=col, s=22, zorder=3)
    a.set_yticks(range(len(rows)), [""] * len(rows)); a.grid(axis="y", visible=False)
    a.scatter([], [], facecolor="white", edgecolor=INK2, s=22, label="epoch 50 (50 trees)")
    a.scatter([], [], color=INK2, s=22, label="epoch 100 (100 trees)")
    a.legend(loc="upper left"); a.set_xlabel("AUROC"); a.set_xlim(0.81, 0.9)
    a.set_title("More training hurt the neural models", loc="left"); panel(a, "b")
    fig.tight_layout()
    save(fig, "Fig07_ten_model_comparison")


def fig8_epochs():
    e = load("epoch_study.json")["results"]
    nets = [k for k, v in e.items() if v["val_curves"]]
    fig, axs = plt.subplots(2, 4, figsize=(FULL, 3.8), sharex=True, sharey=True)
    for a, k in zip(axs.flat, nets):
        curves = np.array(e[k]["val_curves"])
        x = np.arange(1, curves.shape[1] + 1)
        for c in curves:
            a.plot(x, c, color=MUTED, lw=0.6)
        mean = curves.mean(axis=0)
        a.plot(x, mean, color=C["LightGBM"], lw=1.5)
        b = int(np.argmax(mean))
        a.scatter(x[b], mean[b], color=C["LightGBM"], s=16, zorder=3, edgecolor="white", lw=0.6)
        a.axvline(50, color=GRID, lw=0.8)
        a.set_title("%s\nbest epoch %d" % (SHORT.get(k, k).replace("JointPatch Transformer", "JointPatch"), x[b]),
                    loc="left", fontsize=7.2, linespacing=1.3)
    last = axs.flat[len(nets)]
    last.axis("off")
    last.plot([], [], color=MUTED, lw=0.6, label="each of 5 folds")
    last.plot([], [], color=C["LightGBM"], lw=1.5, label="mean over folds")
    last.scatter([], [], color=C["LightGBM"], s=16, label="best epoch")
    last.legend(loc="center")
    for a in list(axs[1]) + [axs[0, 3]]:
        a.set_xlabel("Epoch"); a.tick_params(labelbottom=True)
    for a in axs[:, 0]:
        a.set_ylabel("Validation AUROC")
    axs[0, 0].set_ylim(0.78, 0.9)
    fig.tight_layout()
    save(fig, "Fig08_validation_by_epoch")


def fig9_backorder():
    b = load("backorder_benchmark.json")["runs"]
    g = load("xgb_gpu_benchmark.json")["runs"]
    ours = [("XGBoost, paper settings", g["F_xgb_paper_cuda"], "XGBoost"),
            ("XGBoost EasyEnsemble", g["G_xgb_easyensemble_cuda"], "XGBoost"),
            ("LightGBM, paper settings (A)", b["A_lgbm_paper"], "LightGBM"),
            ("+ position features (B)", b["B_lgbm_position"], "LightGBM"),
            ("+ stacked with LR (C)", b["C_stack_lr_lgbm"], "LightGBM"),
            ("C as EasyEnsemble (D)", b["D_stack_easyensemble"], "LightGBM"),
            ("LightGBM EasyEnsemble (E)", b["E_lgbm_easyensemble"], "LightGBM")]
    pub = sorted(NTAKOLIA.items(), key=lambda kv: kv[1][0])
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 3.6), gridspec_kw={"width_ratios": [1.45, 1]})
    a = ax[0]
    labels, y = [], 0
    for n, v in pub:
        a.barh(y, v[0], color=PUB, height=0.55)
        a.text(v[0] + 0.3, y, "%.2f" % v[0], va="center", fontsize=6.5, color=INK2)
        labels.append("Ntakolia: " + n); y += 1
    for n, runs, fam in ours:
        acc = np.array([r["accuracy"] for r in runs]) * 100
        a.scatter(acc, np.full(len(acc), y), color=C[fam], s=12, alpha=0.8, zorder=3, marker=MK[fam], lw=0)
        a.plot([acc.mean()] * 2, [y - 0.3, y + 0.3], color=INK, lw=1.4)
        a.text(acc.max() + 0.3, y, "%.2f" % acc.mean(), va="center", fontsize=6.5, color=INK2)
        labels.append("Ours: " + n); y += 1
    a.axhline(len(pub) - 0.5, color=INK2, lw=0.6)
    a.set_yticks(range(len(labels)), labels); a.set_xlim(68, 93.5); a.grid(axis="y", visible=False)
    a.set_xlabel("Accuracy on the balanced test set (%)")
    a.set_title("Ntakolia 2021 (grey) vs our 5 seeds", loc="left"); panel(a, "a")

    a = ax[1]
    A = np.array([r["accuracy"] for r in b["A_lgbm_paper"]]) * 100
    E = np.array([r["accuracy"] for r in b["E_lgbm_easyensemble"]]) * 100
    for s_, (x0, x1) in enumerate(zip(A, E)):
        a.plot([0, 1], [x0, x1], color=C["LightGBM"], lw=1, marker="o", ms=3.5, mec="white", mew=0.5)
    a.axhline(NTAKOLIA["LightGBM"][0], color=PUB, lw=1)
    a.text(0.5, NTAKOLIA["LightGBM"][0] + 0.15, "published LightGBM 87.78", ha="center", fontsize=6.8, color=INK2)
    a.axhline(NTAKOLIA["Balanced Bagging"][0], color=PUB, lw=1, ls="--")
    a.text(0.5, NTAKOLIA["Balanced Bagging"][0] + 0.15, "published best 88.85", ha="center", fontsize=6.8, color=INK2)
    a.set_xticks([0, 1], ["paper settings (A)", "EasyEnsemble (E)"]); a.set_xlim(-0.25, 1.25)
    a.set_ylabel("Accuracy (%)"); a.set_ylim(87, 91.5)
    a.set_title("E beats A on all 5 splits", loc="left"); panel(a, "b")
    fig.tight_layout()
    save(fig, "Fig09_backorder_benchmark")


def fig10_m5():
    m = load("m5_results.json")
    pub = dict(m["published"])
    # 38th and 39th: official M5_Scores_and_Ranks.xlsx (papers/), as in report.html
    pub.update({"38th place": 0.5683, "39th place": 0.5698})
    rows = sorted(list(pub.items()) + [("Ours: LightGBM per store", m["wrmsse"])], key=lambda kv: kv[1])
    fig, a = plt.subplots(figsize=(FULL, 2.8))
    for i, (n, v) in enumerate(rows):
        ours = n.startswith("Ours")
        a.barh(i, v, color=C["LightGBM"] if ours else PUB, height=0.6)
        a.text(v + 0.006, i, "%.4f" % v, va="center", fontsize=6.8, color=INK if ours else INK2,
               fontweight="bold" if ours else "normal")
    a.set_yticks(range(len(rows)), [n.replace("(best benchmark)", "(best statistical)") for n, _ in rows])
    a.invert_yaxis(); a.set_xlim(0.45, 0.9); a.grid(axis="y", visible=False)
    a.set_xlabel("WRMSSE on the private leaderboard (lower is better)")
    a.set_title("M5 Accuracy: equivalent to 39th of 5,507 teams, 15.2 % better than ES_bu", loc="left")
    save(fig, "Fig10_m5_leaderboard")


def fig11_li():
    l = load("li2026_results.json")
    rankers = [("VALUE_ONLY", "Value only (no model)"), ("PROB_ONLY", "Late probability"),
               ("M0", "Severity (M0)"), ("BINARYxVALUE", "Late prob. x value"), ("M1", "Severity x value (M1)")]
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 3.0), sharey=True)
    w = 0.26
    for a, ds, L in zip(ax, ("dataco", "olist"), "ab"):
        y = np.arange(len(rankers))
        p = [l["published"][ds][k] for k, _ in rankers]
        xg = [None if k == "VALUE_ONLY" else l[ds]["capture"]["xgb_" + k] for k, _ in rankers]
        lg = [None if k == "VALUE_ONLY" else l[ds]["capture"]["lgbm_" + k] for k, _ in rankers]
        a.barh(y - w, p, height=w * 0.9, color=PUB, label="Li 2026, published")
        a.barh([yy for yy, v in zip(y, xg) if v is not None], [v for v in xg if v is not None],
               height=w * 0.9, color=C["XGBoost"], label="Ours, XGBoost")
        a.barh(y[0], l[ds]["capture"]["VALUE_ONLY"], height=w * 0.9, color=INK2, label="Ours, value only (no model)")
        a.barh([yy + w for yy, v in zip(y, lg) if v is not None], [v for v in lg if v is not None],
               height=w * 0.9, color=C["LightGBM"], label="Ours, LightGBM")
        vo = l[ds]["capture"]["VALUE_ONLY"]
        a.axvline(vo, color=INK2, lw=0.8)
        a.set_yticks(y, [n for _, n in rankers]); a.invert_yaxis(); a.grid(axis="y", visible=False)
        a.set_xlabel("Capture@10 % of delay exposure"); a.set_xlim(0, 0.56)
        a.set_title("DataCo" if ds == "dataco" else "Olist", loc="left"); panel(a, L)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "Fig11_li2026_capture")


def fig12_floor():
    sim = {r["model"]: r for r in load("simulator_results.json")["results"]}
    b = load("backorder_benchmark.json")["runs"]["E_lgbm_easyensemble"]
    isl_pos = 11915 + 3265
    isl_floor = 100 * (1 - isl_pos / (11915 + 3265 + 1518 + 426522))
    ah = load("ahmed_leakage_results.json")
    ah_all = 100 * ah["feature_sets"]["all_columns"]["accuracy_mean"]
    ah_pre = 100 * ah["feature_sets"]["pre_dispatch"]["accuracy_mean"]
    ah_floor = 100 * ah["majority_floor"]
    rows = [  # label, floor, reported, source colour, note
        ("Thaker 2026: PatchTST, 6 h horizon", None, 91.6, PUB, "dataset and class balance not reported"),
        ("Ahmed 2025: SOM+ANN, DataCo", ah_floor, 96.0, PUB, "input columns encode the label"),
        ("Our LightGBM, Ahmed's columns", ah_floor, ah_all, C["LightGBM"], "leakage reproduced, 5 splits"),
        ("Our LightGBM, pre-dispatch columns only", ah_floor, ah_pre, C["LightGBM"], "no post-delivery columns"),
        ("Islam & Amin 2020: GBM, ranged data", isl_floor, 98.92, PUB, "floor from their Table 5 counts"),
        ("Ntakolia 2021: LightGBM", 50.0, 87.78, PUB, "balanced test set"),
        ("Our LightGBM EasyEnsemble, same protocol", 50.0, 100 * np.mean([r["accuracy"] for r in b]),
         C["LightGBM"], "mean of 5 seeds"),
        ("Our LightGBM, study simulator", 100 * load("simulator_results.json")["majority_accuracy"],
         100 * sim["LightGBM"]["accuracy"], C["LightGBM"], "series cross-validation"),
    ]
    fig, a = plt.subplots(figsize=(FULL, 3.3))
    for i, (n, f, r, col, note) in enumerate(rows):
        if f is not None:
            a.plot([f, r], [i, i], color=col, lw=1.6, alpha=0.8)
            a.scatter(f, i, marker="|", s=90, color=INK2, zorder=3, lw=1.4)
        a.scatter(r, i, color=col, s=30, zorder=4, edgecolor="white", lw=0.6)
        gap = "" if f is None else "  +%.1f pt" % (r - f)
        a.text(101.5, i, "%.1f %%%s  (%s)" % (r, gap, note), va="center", fontsize=6.6, color=INK2)
    a.scatter([], [], marker="|", s=90, color=INK2, lw=1.4, label="no-skill floor (always predict the majority class)")
    a.scatter([], [], color=PUB, s=30, label="published")
    a.scatter([], [], color=C["LightGBM"], s=30, label="ours")
    a.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    a.set_yticks(range(len(rows)), [r[0] for r in rows]); a.invert_yaxis()
    a.set_xlim(45, 101); a.grid(axis="y", visible=False)
    a.set_xlabel("Accuracy (%)")
    a.set_title("Reported accuracy measured against each dataset's no-skill floor", loc="left")
    save(fig, "Fig12_accuracy_vs_floor")


# ================================================================ robustness and design
def _rows_robust():
    r = load("robustness_results.json")
    meta = json.load(open(os.path.join(ROOT, "models", "live_risk_meta.json"), encoding="utf-8"))
    rows = [("Series cross-validation (reported)", meta["cv"]["auroc"], None, None, "scheme")]
    for st, m in r["B_leave_one_state_out"].items():
        v = m["LightGBM"]
        rows.append(("Held-out state: %s" % st, v["auroc"], v["auroc_ci"], v["cal_intercept"], "scheme"))
    v = r["A_out_of_time"]["models"]["LightGBM"]
    rows.append(("Later M5 weeks 173-272", v["auroc"], v["auroc_ci"], v["cal_intercept"], "scheme"))
    names = {"Baseline": "no shift", "Demand x0.7": "demand x0.7", "Demand x1.3": "demand x1.3",
             "Disruptions x2": "disruptions x2", "Disruptions x3": "disruptions x3",
             "Heavier lead-time delays": "heavier delays"}
    for name, v in r["C_stress_frozen_model"].items():
        rows.append(("Stress: " + names.get(name, name), v["auroc"], v["auroc_ci"], v["cal_intercept"], "stress"))
    return rows


def fig13_robustness(width=FULL, name="Fig13_robustness"):
    rows = _rows_robust()
    fig, ax = plt.subplots(1, 2, figsize=(width, 0.22 * len(rows) + 0.9), sharey=True,
                           gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(rows))
    for i, (n, auc, ci, icpt, kind) in enumerate(rows):
        col = C["LightGBM"] if kind == "scheme" else INK2
        if ci:
            ax[0].plot(ci, [i, i], color=col, lw=1.3)
        ax[0].scatter(auc, i, color=col, s=20, zorder=3, edgecolor="white", lw=0.5,
                      marker="o" if kind == "scheme" else "s")
        if icpt is not None:
            ax[1].barh(i, icpt, color=col, height=0.55)
            ax[1].text(icpt + (0.03 if icpt >= 0 else -0.03), i, "%+.2f" % icpt, va="center",
                       ha="left" if icpt >= 0 else "right", fontsize=6.3, color=INK2)
    split = sum(1 for r in rows if r[4] == "scheme") - 0.5
    for a in ax:
        a.axhline(split, color=INK2, lw=0.6)
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, [r[0] for r in rows])
    ax[0].invert_yaxis()
    ax[0].set_xlabel("AUROC (95 % CI)")
    ax[0].set_xlim(0.8, 1.0)
    ax[0].set_title("Discrimination", loc="left")
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_xlim(-1.05, 1.05)
    ax[1].set_xlabel("Calibration intercept")
    ax[1].set_title("Calibration (0 = ideal)", loc="left")
    panel(ax[0], "a")
    panel(ax[1], "b")
    fig.tight_layout()
    save(fig, name)


def fig14_decision():
    r = load("robustness_results.json")["D_decision_value"]
    pol = r["policies"]
    names = ["Base-stock + top-up when model risk >= 50 %",
             "Base-stock + top-up when net position < 0 (no model)",
             "Reference policy Q* (upper bound)"]
    labels = ["Rule + live risk model", "Rule + net position (no model)", "Reference policy Q*"]
    cols = [C["LightGBM"], C["Net position only"], MUTED]
    fig, a = plt.subplots(figsize=(FULL, 1.9))
    for i, (n, lab, c) in enumerate(zip(names, labels, cols)):
        v = pol[n]
        a.barh(i, v["cost_vs_base"], color=c, height=0.55)
        a.plot(v["cost_vs_base_ci"], [i, i], color=INK, lw=1)
        a.text(v["cost_vs_base_ci"][0] - 400, i, "{:+,.0f}".format(v["cost_vs_base"]), va="center", ha="right",
               fontsize=7, color=INK2)
    a.set_yticks(range(3), labels)
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.axvline(0, color=INK2, lw=0.8)
    a.set_xlim(-42000, 1000)
    a.set_xlabel("Change in 52-week cost vs the base-stock rule alone (mean of %d paired episodes, 95 %% CI)"
                 % r["settings"]["episodes"])
    d = pol["model_minus_no_model_cost"]
    a.set_title("Acting on the risk %% beats the best no-model rule by %.0f (95 %% CI %.0f to %.0f)"
                % (-d["mean"], -d["ci"][1], -d["ci"][0]), loc="left")
    save(fig, "Fig14_decision_value")


def fig15_placement(width=FULL, name="Fig15_error_placement"):
    with open(os.path.join(ROOT, "results", "placement_design.json"), encoding="utf-8") as fh:
        pdj = json.load(fh)
    per = pdj["selection"]["per_week"]
    weeks = sorted(int(w) for w in per)
    prop, absb = set(pdj["propagating"]), set(pdj["absorbed"])
    # Redrawn 19 Oct 2026, then polished to share Fig. 2's exact style (type sizes,
    # grey axes, no grid, 600 dpi). Vermillion = an error that propagates, blue
    # hatching = one that is absorbed, as in Fig. 1(c) (Okabe-Ito, so it survives
    # colour-vision deficiency; the hatching survives greyscale). Triangles on the
    # week axis mark the six fixed disruptions and have their own legend entry.
    PROPC, ABSC, NEVER = "#d55e00", "#0072b2", "#dcdad3"
    MUTEDC = "#6b6a65"
    disruptions = (6, 9, 13, 17, 21, 26)   # the fixed calendar, src/simulator.py
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    style = {"font.size": 6.8, "axes.labelsize": 6.6, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
             "legend.fontsize": 6.0, "axes.titlesize": 6.9, "axes.titleweight": "bold",
             "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
             "axes.edgecolor": "#8c8b86", "axes.linewidth": 0.7, "axes.grid": False,
             "xtick.color": MUTEDC, "ytick.color": MUTEDC, "hatch.linewidth": 0.7}
    with plt.rc_context(style):
        fig, ax = plt.subplots(1, 2, figsize=(width, 1.22), gridspec_kw={"width_ratios": [2.35, 1]},
                               layout="constrained")
        a = ax[0]
        for w in weeks:
            v = per[str(w)]["dcost"]
            if w in prop:
                a.bar(w, v, color=PROPC, width=0.74, zorder=3, lw=0)
            elif w in absb:
                a.bar(w, v, color="white", edgecolor=ABSC, hatch="//////", lw=0.7, width=0.74, zorder=3)
            else:
                a.bar(w, v, color=NEVER, width=0.74, zorder=3, lw=0)
        a.axhline(0, color=INK2, lw=0.7, zorder=4)
        lo, hi = a.get_ylim()
        # disruptions sit on the week axis: no bar reaches the floor at those weeks
        for d in disruptions:
            a.plot(d, lo + 0.05 * (hi - lo), marker="^", color=INK, ms=3.4, mew=0, zorder=5)
        a.set_xlim(0.8, 26.5)
        a.set_xticks([2, 5, 10, 15, 20, 25])
        a.set_yticks([0, 250, 500, 750])
        a.set_xlabel("Week of the session")
        a.set_ylabel("Extra cost (units)")
        a.legend(handles=[Patch(facecolor=PROPC, label="error propagates"),
                          Patch(facecolor="white", edgecolor=ABSC, hatch="//////", label="error absorbed"),
                          Patch(facecolor=NEVER, label="never an error week"),
                          Line2D([], [], marker="^", color=INK, ls="none", ms=3.4, mew=0, label="disruption")],
                 loc="upper left", ncol=1, handlelength=1.1, labelspacing=0.22, borderaxespad=0.15,
                 frameon=False)
        a.set_title("a  Extra cost of one accepted error, by week", loc="left")
        b = ax[1]
        ref, noi = pdj["validation_reference_follower"], pdj["validation_noisy_follower"]
        x = np.arange(2)
        w = 0.36
        b.bar(x - w / 2, [ref["prop_stock"], noi["prop_stock"]], w, color=PROPC, zorder=3, lw=0)
        b.bar(x + w / 2, [ref["abs_stock"], noi["abs_stock"]], w, color="white", edgecolor=ABSC,
              hatch="//////", lw=0.7, zorder=3)
        for xi, rr in zip(x, (ref, noi)):
            b.text(xi, max(rr["prop_stock"], rr["abs_stock"]) + 0.012, "%.2f\u00d7" % rr["stock_ratio"],
                   ha="center", fontsize=6.4, fontweight="bold", color=INK)
        b.set_xticks(x, ["reference\nfollower", "noisy\nfollower"])
        b.set_ylim(0, 0.36)
        b.set_yticks([0, 0.1, 0.2, 0.3])
        b.set_ylabel("Extra stockout weeks")
        b.set_title("b  Held-out draws 100\u2013299", loc="left")
        for axis in ax:
            for sp in ("top", "right"):
                axis.spines[sp].set_visible(False)
            axis.tick_params(length=2, width=0.6, pad=1.5)
            axis.tick_params(axis="x", length=0)
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, facecolor="white")
        fig.savefig(os.path.join(OUT, name + ".svg"), facecolor="white")
        plt.close(fig)
        print("  %s" % name)


def main_text_figures():
    """Figures printed in the paper: 4.8 in wide (print area 122 mm), 7-7.5 pt text."""
    global OUT
    keep = OUT
    OUT = os.path.join(keep, "main_text")
    with plt.rc_context({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}):
        fig15_placement(width=4.8, name="Figure2_error_placement")
        fig13_robustness(width=4.8, name="Figure3_model_robustness")
    OUT = keep


def main():
    cd = load("chart_data.json")
    print("writing figures to %s" % os.path.normpath(OUT))
    fig1_profile(); fig2_correlation(); fig3_roc_pr(cd); fig4_calibration(cd)
    fig5_confusion(cd); fig6_shap(cd); fig7_ten_models(); fig8_epochs()
    fig9_backorder(); fig10_m5(); fig11_li(); fig12_floor()
    fig13_robustness(); fig14_decision(); fig15_placement()
    main_text_figures()


if __name__ == "__main__":
    main()
