"""
Like-for-like comparison: every model scored on IDENTICAL test rows.

The earlier comparison was invalid. PatchTST only produces a prediction for rows
that have LOOKBACK=48 periods of history behind them; the tabular baselines were
scored on every valid row, including the first 48 of each episode. Different row
sets give different base rates (0.629 vs 0.684) and therefore incomparable
accuracies.

This script restricts every model to the same windowed rows and the same
episode-level split, so the numbers can actually be placed side by side in a
paper.

Run:  python src/compare_fair.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from train_baselines import BASE_COLS, LAGS, build_features, expected_calibration_error  # noqa: E402

def corpus_fingerprint():
    """Hash of the corpus, so results can never be compared across corpora."""
    d = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    for n in ("corpus.parquet", "corpus.csv"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
            return h.hexdigest()[:16]
    return None


HORIZON = 2
LOOKBACK = 48  # must match patchtst_v2.LOOKBACK
USE_SPLIT_COL = True  # corpus now carries a leakage-free series-level split


def load_corpus():
    d = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    for n in ("corpus.parquet", "corpus.csv"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return pd.read_parquet(p) if n.endswith("parquet") else pd.read_csv(p)
    raise SystemExit("No corpus.")


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: accuracy_score(y, (p >= t).astype(int))))


def evaluate(name, y, prob, thr):
    pred = (prob >= thr).astype(int)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "f1": float(f1_score(y, pred)),
        "brier": float(brier_score_loss(y, prob)),
        "ece": float(expected_calibration_error(y, prob)),
    }


def main():
    label = "stockout_within_%d" % HORIZON
    df = load_corpus()
    df = df[df["label_valid"] == 1].reset_index(drop=True)

    # THE FIX: keep only rows PatchTST can also score, i.e. those with a full
    # 48-period look-back inside their own episode.
    #
    # The subsetting is deferred until AFTER build_features below. Dropping the
    # early periods first would leave every *_lag48 column pointing at rows that
    # no longer exist -- measured 100% NaN on lag48, 30.7% across all lag
    # features, then silently zeroed by nan_to_num. The scored row set is
    # identical either way; only the feature values change.
    _keep = (df["period"] >= LOOKBACK).values

    print("=" * 76)
    print("LIKE-FOR-LIKE COMPARISON  (identical rows, identical splits)")
    print("=" * 76)
    print("  rows scored by every model: %d" % int(_keep.sum()))

    X = build_features(df)
    df = df[_keep].reset_index(drop=True)
    X = X[_keep].reset_index(drop=True)
    y = df[label].values
    feat_cols = [c for c in X.columns if c not in ("episode", "period")]
    Xv = np.nan_to_num(X[feat_cols].values.astype(np.float32),
                       nan=0.0, posinf=0.0, neginf=0.0)

    if USE_SPLIT_COL and "split" in df.columns:
        m_tr = (df["split"] == "train").values
        m_va = (df["split"] == "val").values
        m_te = (df["split"] == "test").values
        ns = df.groupby("split")["m5_series"].nunique().to_dict()
        print("  split by M5 SERIES (leakage-free): %s" % ns)
    else:
        eps = np.sort(df["episode"].unique())
        n = len(eps)
        tr = set(eps[: int(0.70 * n)])
        va = set(eps[int(0.70 * n): int(0.85 * n)])
        m_tr = df["episode"].isin(tr).values
        m_va = df["episode"].isin(va).values
        m_te = ~(m_tr | m_va)
    print("  episodes: %d train / %d val / %d test"
          % (df.loc[m_tr, "episode"].nunique(), df.loc[m_va, "episode"].nunique(),
             df.loc[m_te, "episode"].nunique()))
    print("  test rows: %d, positive rate %.1f%%" % (m_te.sum(), 100 * y[m_te].mean()))
    majority = float(max(y[m_te].mean(), 1 - y[m_te].mean()))
    print("  majority-class accuracy: %.3f\n" % majority)

    results = []

    # ---- logistic regression ----
    sc = StandardScaler().fit(Xv[m_tr])
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xv[m_tr]), y[m_tr])
    p_va = lr.predict_proba(sc.transform(Xv[m_va]))[:, 1]
    p_te = lr.predict_proba(sc.transform(Xv[m_te]))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y[m_va])
    results.append(evaluate("logistic regression", y[m_te], iso.predict(p_te),
                            best_threshold(y[m_va], iso.predict(p_va))))

    # ---- gradient boosting ----
    gb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        early_stopping=True, validation_fraction=0.15, random_state=0,
    ).fit(Xv[m_tr], y[m_tr])
    p_va = gb.predict_proba(Xv[m_va])[:, 1]
    p_te = gb.predict_proba(Xv[m_te])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y[m_va])
    results.append(evaluate("gradient boosting", y[m_te], iso.predict(p_te),
                            best_threshold(y[m_va], iso.predict(p_va))))

    # ---- PatchTST v2, from its own run ----
    # GUARD: refuse to compare results produced on a DIFFERENT corpus. This has
    # now bitten three times -- stale numbers from an earlier corpus silently
    # produced an invalid comparison, once flagging a PatchTST "win" that was
    # pure leakage.
    pt_path = os.path.join(os.path.dirname(__file__), "..", "results", "patchtst_v2.json")
    pt = None
    if os.path.exists(pt_path):
        blob = json.load(open(pt_path, encoding="utf-8"))
        if blob.get("corpus_fingerprint") != corpus_fingerprint():
            print("\n  [!] results/patchtst_v2.json was produced on a DIFFERENT corpus.")
            print("      Ignoring it. Re-run: python src/patchtst_v2.py\n")
            blob = None
        runs = blob["runs"] if blob else []
    else:
        runs = []
    if runs:
        pt = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
              for k in ("accuracy", "auroc", "pr_auc", "f1", "brier", "ece")}

    print("  %-24s %9s %9s %9s %9s %9s" % ("model", "acc", "AUROC", "PR-AUC", "F1", "ECE"))
    print("  " + "-" * 74)
    for r in results:
        print("  %-24s %9.3f %9.3f %9.3f %9.3f %9.3f"
              % (r["model"], r["accuracy"], r["auroc"], r["pr_auc"], r["f1"], r["ece"]))
    if pt:
        print("  %-24s %9.3f %9.3f %9.3f %9.3f %9.3f"
              % ("PatchTST v2 (5 seeds)", pt["accuracy"][0], pt["auroc"][0],
                 pt["pr_auc"][0], pt["f1"][0], pt["ece"][0]))
        print("  %-24s %9s %9s" % ("  +/- SD",
                                   "%.3f" % pt["accuracy"][1], "%.3f" % pt["auroc"][1]))
    print("  %-24s %9.3f %9.3f" % ("majority class", majority, 0.500))

    best = max(results, key=lambda r: r["auroc"])
    print("\n" + "=" * 76)
    if pt and pt["auroc"][0] > best["auroc"]:
        print("  PatchTST WINS: AUROC %.3f vs %.3f (%s)"
              % (pt["auroc"][0], best["auroc"], best["model"]))
    elif pt:
        print("  PatchTST loses: AUROC %.3f vs %.3f (%s), a gap of %.3f"
              % (pt["auroc"][0], best["auroc"], best["model"], best["auroc"] - pt["auroc"][0]))
        print("\n  The baselines receive 116 hand-built lag and rolling features -- the")
        print("  temporal structure pre-extracted. PatchTST must learn the same")
        print("  relationships from raw windows. At %d training rows, explicit" % m_tr.sum())
        print("  beats learned. That is the finding, not 'transformers are bad'.")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "comparison_fair.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows_scored": len(df), "majority": majority,
                   "baselines": results, "patchtst_v2": pt,
                   "corpus_fingerprint": corpus_fingerprint()}, fh, indent=2)
    print("\n  written -> results/comparison_fair.json")


if __name__ == "__main__":
    main()
