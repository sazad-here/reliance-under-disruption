"""
Train the baseline models and measure whether 80% accuracy is realistic.

This is the step that ANCHORS THE EXPERIMENT. The reliability conditions
(80% / 60%) are only defensible if a real model actually reaches roughly that
on this task. If the ceiling here were 65%, the high condition would be a
fantasy and the plan would need to change.

Models, per plan section 05:
    logistic regression   the floor -- how much signal is trivially linear
    gradient boosting     the strong tabular baseline the reference paper omits

PatchTST comes next and has to beat these to justify itself.

Calibration is reported alongside accuracy and matters more than usual here:
participants are shown "Shortage risk: 82%", so that number has to be honest or
we are measuring reactions to a distorted stimulus (plan section 05).

Run:  python src/train_baselines.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

HORIZON = 2          # the horizon the reliability levels are anchored to
LAGS = (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48)
BASE_COLS = [
    "demand", "inventory", "on_order", "lead_time", "backorders",
    "supplier_reliability", "transport_ok", "disruption_flag",
]


def load_corpus():
    d = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    for name in ("corpus.parquet", "corpus.csv"):
        path = os.path.join(d, name)
        if os.path.exists(path):
            return pd.read_parquet(path) if name.endswith("parquet") else pd.read_csv(path)
    raise SystemExit("No corpus found. Run:  python src/make_corpus.py")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged and rolling features, built strictly within each episode.

    Crossing an episode boundary would leak one warehouse's history into
    another's -- the sort of silent defect that inflates accuracy and is
    invisible in the score.
    """
    out = [df[["episode", "period"]].copy()]
    g = df.groupby("episode", sort=False)

    for col in BASE_COLS:
        out.append(df[[col]])
        for lag in LAGS:
            out.append(g[col].shift(lag).rename("%s_lag%d" % (col, lag)))

    for col in ("demand", "inventory", "backorders"):
        for win in (4, 12, 48):
            out.append(
                g[col].rolling(win, min_periods=2).mean()
                .reset_index(level=0, drop=True).rename("%s_mean%d" % (col, win))
            )
            out.append(
                g[col].rolling(win, min_periods=2).std()
                .reset_index(level=0, drop=True).rename("%s_std%d" % (col, win))
            )

    # inventory position and its trend -- what a base-stock rule actually watches
    ip = df["inventory"] + df["on_order"] - df["backorders"]
    out.append(ip.rename("inventory_position"))
    out.append(g.apply(lambda x: (x["inventory"] + x["on_order"] - x["backorders"]).diff(4),
                       include_groups=False).reset_index(level=0, drop=True).rename("ip_trend"))
    return pd.concat(out, axis=1)


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return ece


def evaluate(name, y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "auroc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
        "ece": expected_calibration_error(y, prob),
        "positive_rate": float(y.mean()),
    }


def main():
    df = load_corpus()
    label = "stockout_within_%d" % HORIZON
    df = df[df["label_valid"] == 1].reset_index(drop=True)

    print("=" * 74)
    print("TRAINING BASELINES  (horizon h=%d, %d rows)" % (HORIZON, len(df)))
    print("=" * 74)

    X = build_features(df)
    y = df[label].values

    # Split BY EPISODE, not by row: rows within an episode are autocorrelated,
    # so a random row split would leak the future into the training set.
    episodes = np.sort(df["episode"].unique())
    n = len(episodes)
    train_ep = set(episodes[: int(0.70 * n)])
    val_ep = set(episodes[int(0.70 * n): int(0.85 * n)])
    test_ep = set(episodes[int(0.85 * n):])

    feat_cols = [c for c in X.columns if c not in ("episode", "period")]
    m_tr = df["episode"].isin(train_ep).values
    m_va = df["episode"].isin(val_ep).values
    m_te = df["episode"].isin(test_ep).values

    Xv = X[feat_cols].values.astype(np.float32)
    Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)

    print("  episodes: %d train / %d val / %d test" % (len(train_ep), len(val_ep), len(test_ep)))
    print("  features: %d" % len(feat_cols))
    print("  positive rate: %.1f%% train, %.1f%% test\n"
          % (100 * y[m_tr].mean(), 100 * y[m_te].mean()))

    results = []

    # ---- floor: logistic regression ----
    scaler = StandardScaler().fit(Xv[m_tr])
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(scaler.transform(Xv[m_tr]), y[m_tr])
    results.append(evaluate("logistic regression", y[m_te], lr.predict_proba(scaler.transform(Xv[m_te]))[:, 1]))

    # ---- strong tabular: gradient boosting ----
    gb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        early_stopping=True, validation_fraction=0.15, random_state=0,
    )
    gb.fit(Xv[m_tr], y[m_tr])
    results.append(evaluate("gradient boosting", y[m_te], gb.predict_proba(Xv[m_te])[:, 1]))

    # ---- gradient boosting + calibration on the validation split ----
    cal = CalibratedClassifierCV(gb, method="isotonic", cv="prefit")
    cal.fit(Xv[m_va], y[m_va])
    results.append(evaluate("gradient boosting (calibrated)", y[m_te], cal.predict_proba(Xv[m_te])[:, 1]))

    print("  %-32s %8s %7s %7s %7s %7s" % ("model", "acc", "AUROC", "PR-AUC", "Brier", "ECE"))
    print("  " + "-" * 72)
    for r in results:
        print("  %-32s %7.1f%% %7.3f %7.3f %7.3f %7.3f"
              % (r["model"], 100 * r["accuracy"], r["auroc"], r["pr_auc"], r["brier"], r["ece"]))

    majority = max(y[m_te].mean(), 1 - y[m_te].mean())
    print("  %-32s %7.1f%%  %s" % ("(always predict majority)", 100 * majority, "<- the floor to beat"))

    best = max(results, key=lambda r: r["accuracy"])
    print("\n" + "=" * 74)
    print("WHAT THIS MEANS FOR THE RELIABILITY LEVELS")
    print("=" * 74)
    print("  best baseline accuracy at h=%d:  %.1f%%" % (HORIZON, 100 * best["accuracy"]))
    if best["accuracy"] >= 0.78:
        print("  -> 80%% is a realistic 'achievable system'. The planned levels hold.")
    elif best["accuracy"] >= 0.70:
        print("  -> 80%% is optimistic for a BASELINE. PatchTST must close the gap,")
        print("     or anchor the high condition at %d%% instead." % (round(best['accuracy'] * 20) * 5))
    else:
        print("  -> WARNING: 80%% is not supportable. Re-anchor the conditions.")
    print("\n  Calibration note: ECE %.3f. Participants are shown a risk %%," % best["ece"])
    print("  so this must stay low or the stimulus itself is misleading.")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "baselines.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"horizon": HORIZON, "n_features": len(feat_cols), "results": results}, fh, indent=2)
    print("\n  written -> results/baselines.json")


if __name__ == "__main__":
    main()
