"""
The real baselines, with correct thresholding and honest error bars.

Two research surveys of the literature independently concluded the same thing:
we did not know what we were trying to beat, and our threshold selection was
wrong. Both are fixed here.

WHAT CHANGED FROM train_baselines.py

1. LightGBM added. Grinsztajn et al. (NeurIPS 2022, arXiv:2207.08815) find tree
   ensembles remain state of the art at ~10k samples, which is our regime. A
   gradient-boosted tree on the 116 engineered features is the honest reference
   model, and quite possibly the strongest thing here.

2. THRESHOLD TUNED FOR THE METRIC BEING REPORTED. Previously the threshold was
   chosen to maximise accuracy and then F1 was reported at that threshold. At
   17.4% positive those optima are far apart. Lipton, Elkan & Narayanaswamy
   (ECML PKDD 2014, arXiv:1402.1892) prove that for calibrated probabilities the
   F1-optimal threshold equals half the optimal F1 -- roughly 0.30 here, not 0.5.
   We now report accuracy at its own threshold and F1 at its own, and print the
   Lipton sanity check.

3. CALIBRATE, THEN THRESHOLD. Temperature/Platt scaling is strictly monotone, so
   it leaves AUROC and PR-AUC mathematically unchanged while improving ECE.
   Calibration is free; there is no trade-off to accept.

4. BOOTSTRAP CONFIDENCE INTERVALS, RESAMPLED BY EPISODE. With ~640 positives the
   noise band on PR-AUC and F1 is roughly +/-0.03. Differences smaller than that
   are not differences. Resampling by episode rather than by row respects the
   autocorrelation of overlapping windows.

Run:  python src/baselines_strong.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from compare_fair import corpus_fingerprint  # noqa: E402
from train_baselines import build_features, expected_calibration_error  # noqa: E402

HORIZON = 2
LOOKBACK = 48
N_BOOT = 400


N_FOLDS = 5


def series_folds(m5_series, fold, n_folds=N_FOLDS, seed=7):
    """Train/val/test masks for one series-level CV fold.

    Every M5 series takes a turn in the test fold, so the test estimate is
    backed by all 70 series instead of a fixed 11. That matters here because
    the series -- not the episode -- is the independent unit: episodes are
    overlapping 100-week windows of the same demand, so 78 test episodes drawn
    from 11 series carry roughly 11 series worth of evidence, which is what
    makes the series-clustered interval 1.20x wider than the episode one.

    The split is BY SERIES at every level, so no series appears in two splits
    and the leakage the corpus was rebuilt to avoid cannot reappear here.
    """
    uniq = np.unique(m5_series)
    order = np.random.default_rng(seed).permutation(uniq)
    chunks = np.array_split(order, n_folds)
    test_s = set(chunks[fold].tolist())
    # the next fold round-robins into validation; the rest train
    val_s = set(chunks[(fold + 1) % n_folds].tolist())
    train_s = set(order.tolist()) - test_s - val_s

    in_ = lambda pool: np.isin(m5_series, list(pool))
    m_tr, m_va, m_te = in_(train_s), in_(val_s), in_(test_s)
    assert not (m_tr & m_va).any() and not (m_tr & m_te).any() and not (m_va & m_te).any()
    return m_tr, m_va, m_te


def load_corpus():
    d = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    for n in ("corpus.parquet", "corpus.csv"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return pd.read_parquet(p) if n.endswith("parquet") else pd.read_csv(p)
    raise SystemExit("No corpus.")


def platt(p_va, y_va, p_te):
    """Monotone 1-D logistic recalibration. Preserves AUROC and PR-AUC exactly."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    eps = 1e-6
    z_va = np.log(np.clip(p_va, eps, 1 - eps) / np.clip(1 - p_va, eps, 1 - eps))
    z_te = np.log(np.clip(p_te, eps, 1 - eps) / np.clip(1 - p_te, eps, 1 - eps))
    lr.fit(z_va.reshape(-1, 1), y_va)
    return (lr.predict_proba(z_va.reshape(-1, 1))[:, 1],
            lr.predict_proba(z_te.reshape(-1, 1))[:, 1])


def thr_for(metric, y, p):
    grid = np.linspace(0.02, 0.98, 97)
    fn = f1_score if metric == "f1" else accuracy_score
    return float(max(grid, key=lambda t: fn(y, (p >= t).astype(int))))


def boot_ci(y, p, fn, episodes, n=N_BOOT, seed=0):
    """Bootstrap by EPISODE -- rows within an episode are autocorrelated."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(episodes)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(episodes == e)[0] for e in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(fn(yy, p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def _fit_models(Xv, y, m_tr, m_va):
    """The same three baselines, fitted on one fold. Returns name -> (p_va, p_te)."""
    out = {}
    sc = StandardScaler().fit(Xv[m_tr])
    lr = LogisticRegression(max_iter=4000, C=1.0).fit(sc.transform(Xv[m_tr]), y[m_tr])
    out["logistic regression"] = lambda M, _sc=sc, _m=lr: _m.predict_proba(_sc.transform(Xv[M]))[:, 1]

    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        early_stopping=True, validation_fraction=0.15, random_state=0,
    ).fit(Xv[m_tr], y[m_tr])
    out["sklearn HistGB"] = lambda M, _m=hgb: _m.predict_proba(Xv[M])[:, 1]

    try:
        import lightgbm as lgb
        g = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1,
        )
        g.fit(Xv[m_tr], y[m_tr], eval_set=[(Xv[m_va], y[m_va])],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        out["LightGBM"] = lambda M, _m=g: _m.predict_proba(Xv[M])[:, 1]
    except ImportError:
        pass
    return out


def run_cv(n_folds=N_FOLDS):
    """Series-level cross-validation: every M5 series takes a turn as test.

    Reports metrics on the POOLED out-of-fold predictions -- each row scored
    exactly once, by a model that never saw its series -- plus the spread
    across folds, which is the honest uncertainty for a claim that one model
    beats another.
    """
    label = "stockout_within_%d" % HORIZON
    df = load_corpus()
    df = df[df["label_valid"] == 1].reset_index(drop=True)
    X = build_features(df)
    keep = (df["period"] >= LOOKBACK).values
    df, X = df[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    y = df[label].values
    feat = [c for c in X.columns if c not in ("episode", "period")]
    Xv = np.nan_to_num(X[feat].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ms = df["m5_series"].values

    print("=" * 82)
    print("SERIES CROSS-VALIDATION  (%d folds, every series tested once)" % n_folds)
    print("=" * 82)
    print("  %d rows | %d series | %d features" % (len(y), len(np.unique(ms)), len(feat)))

    oof = {}
    per_fold = {}
    for k in range(n_folds):
        m_tr, m_va, m_te = series_folds(ms, k, n_folds)
        models = _fit_models(Xv, y, m_tr, m_va)
        for name, predict in models.items():
            p_va, p_te = predict(m_va), predict(m_te)
            _, cal_te = platt(p_va, y[m_va], p_te)
            oof.setdefault(name, np.full(len(y), np.nan))[m_te] = cal_te
            per_fold.setdefault(name, []).append(
                (average_precision_score(y[m_te], cal_te),
                 roc_auc_score(y[m_te], cal_te)))
        print("  fold %d done  (train %d / val %d / test %d)"
              % (k, m_tr.sum(), m_va.sum(), m_te.sum()))

    print()
    print("  %-22s %7s %7s %7s %7s %7s %7s %8s" %
          ("model", "acc", "AUROC", "PR-AUC", "prec", "recall", "F1", "PRAUC sd"))
    print("  " + "-" * 74)
    rows = []
    for name, p in oof.items():
        assert not np.isnan(p).any(), "every row must be scored out-of-fold exactly once"
        thr = thr_for("f1", y, p)
        pr = average_precision_score(y, p)
        sd = float(np.std([f[0] for f in per_fold[name]]))
        r = {"model": name, "accuracy": float(accuracy_score(y, (p >= thr).astype(int))),
             "auroc": float(roc_auc_score(y, p)), "pr_auc": float(pr),
             "f1": float(f1_score(y, (p >= thr).astype(int))),
             "precision": float(precision_score(y, (p >= thr).astype(int), zero_division=0)),
             "recall": float(recall_score(y, (p >= thr).astype(int))),
             "ece": float(expected_calibration_error(y, p)),
             "pr_auc_fold_sd": sd,
             "per_fold": [{"pr_auc": a, "auroc": b} for a, b in per_fold[name]]}
        rows.append(r)
        print("  %-22s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %8.3f"
              % (name, r["accuracy"], r["auroc"], r["pr_auc"], r["precision"],
                 r["recall"], r["f1"], sd))

    out = os.path.join(os.path.dirname(__file__), "..", "results", "baselines_cv.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"corpus_fingerprint": corpus_fingerprint(),
                   "n_folds": n_folds, "results": rows}, fh, indent=2)
    print("\n  written -> %s" % out)
    return rows


def main():
    label = "stockout_within_%d" % HORIZON
    df = load_corpus()
    # Build features on the FULL per-episode history, then subset to the scored
    # rows. Filtering on period first left every *_lag48 column referencing rows
    # that had already been dropped: measured 100% NaN on lag48 and 30.7% across
    # all lag features, silently zeroed by nan_to_num below. The scored row set
    # is identical either way, so this only restores information.
    df = df[df["label_valid"] == 1].reset_index(drop=True)
    X = build_features(df)
    _keep = (df["period"] >= LOOKBACK).values
    df = df[_keep].reset_index(drop=True)
    X = X[_keep].reset_index(drop=True)
    y = df[label].values
    feat = [c for c in X.columns if c not in ("episode", "period")]
    Xv = np.nan_to_num(X[feat].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # series_folds(), NOT df["split"]. The corpus's baked-in split (70/15/15,
    # fixed at generation time) and patchtst_v2.py's series_folds() rotation
    # (60/20/20) are two DIFFERENT partitions of the same 70 M5 series -- their
    # test sets turned out to share zero series in common when checked. Every
    # torch model in extra_baselines.py, and PatchTST v2 itself, is scored with
    # series_folds(fold=0); this file used to be the one exception, which made
    # every "tabular vs temporal" comparison an apples-to-oranges one even
    # though both halves individually were leakage-free. FOLD is exposed the
    # same way patchtst_v2.py exposes PATCHTST_FOLD, so a future CV sweep can
    # still rotate it.
    FOLD = int(os.environ.get("BASELINES_FOLD", "0"))
    ser = df["m5_series"].values
    m_tr, m_va, m_te = series_folds(ser, FOLD)
    assert not set(ser[m_tr]) & set(ser[m_te]), "series leaked across train/test"
    assert not set(ser[m_va]) & set(ser[m_te]), "series leaked across val/test"
    ep_te = df.loc[m_te, "episode"].values
    se_te = df.loc[m_te, "m5_series"].values

    print("=" * 82)
    print("STRONG BASELINES  (series-level split, correct thresholds, bootstrap CIs)")
    print("=" * 82)
    print("  features: %d   train %d / val %d / test %d rows"
          % (len(feat), m_tr.sum(), m_va.sum(), m_te.sum()))
    print("  test positive rate: %.1f%%   majority accuracy: %.3f\n"
          % (100 * y[m_te].mean(), max(y[m_te].mean(), 1 - y[m_te].mean())))

    models = {}

    sc = StandardScaler().fit(Xv[m_tr])
    lr = LogisticRegression(max_iter=4000, C=1.0).fit(sc.transform(Xv[m_tr]), y[m_tr])
    models["logistic regression"] = (lr.predict_proba(sc.transform(Xv[m_va]))[:, 1],
                                     lr.predict_proba(sc.transform(Xv[m_te]))[:, 1])

    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        early_stopping=True, validation_fraction=0.15, random_state=0,
    ).fit(Xv[m_tr], y[m_tr])
    models["sklearn HistGB"] = (hgb.predict_proba(Xv[m_va])[:, 1],
                                hgb.predict_proba(Xv[m_te])[:, 1])

    try:
        import lightgbm as lgb
        g = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1,
        )
        g.fit(Xv[m_tr], y[m_tr], eval_set=[(Xv[m_va], y[m_va])],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        models["LightGBM"] = (g.predict_proba(Xv[m_va])[:, 1], g.predict_proba(Xv[m_te])[:, 1])
        print("  LightGBM stopped at %s trees\n" % g.best_iteration_)
    except ImportError:
        print("  [!] lightgbm not installed\n")

    rows = []
    print("  %-22s %7s %7s %7s %7s %7s %7s" %
          ("model", "acc", "AUROC", "PR-AUC", "F1", "ECE", "thr_F1"))
    print("  " + "-" * 78)

    for name, (p_va, p_te) in models.items():
        cal_va, cal_te = platt(p_va, y[m_va], p_te)
        t_acc = thr_for("acc", y[m_va], cal_va)
        t_f1 = thr_for("f1", y[m_va], cal_va)
        r = {
            "model": name,
            "accuracy": accuracy_score(y[m_te], (cal_te >= t_acc).astype(int)),
            "auroc": roc_auc_score(y[m_te], cal_te),
            "pr_auc": average_precision_score(y[m_te], cal_te),
            "f1": f1_score(y[m_te], (cal_te >= t_f1).astype(int)),
            "ece": expected_calibration_error(y[m_te], cal_te),
            "brier": brier_score_loss(y[m_te], cal_te),
            "thr_acc": t_acc, "thr_f1": t_f1,
        }
        r["pr_auc_ci"] = boot_ci(y[m_te], cal_te, average_precision_score, ep_te)
        # Episodes are overlapping 100-week windows of only 11 M5 series, so
        # resampling episodes treats correlated windows as independent draws.
        # The series-clustered interval is the honest one. See eval_power.py.
        r["pr_auc_ci_series"] = boot_ci(y[m_te], cal_te, average_precision_score, se_te)
        r["auroc_ci_series"] = boot_ci(y[m_te], cal_te, roc_auc_score, se_te)
        r["auroc_ci"] = boot_ci(y[m_te], cal_te, roc_auc_score, ep_te)
        rows.append(r)
        print("  %-22s %7.3f %7.3f %7.3f %7.3f %7.3f %7.2f"
              % (name, r["accuracy"], r["auroc"], r["pr_auc"], r["f1"], r["ece"], t_f1))

    print("\n  95%% bootstrap CIs (resampled by episode, %d reps):" % N_BOOT)
    for r in rows:
        print("    %-22s AUROC [%.3f, %.3f]   PR-AUC [%.3f, %.3f]"
              % (r["model"], r["auroc_ci"][0], r["auroc_ci"][1],
                 r["pr_auc_ci"][0], r["pr_auc_ci"][1]))

    print("\n  95% bootstrap CIs (resampled by SERIES, n=11 -- report THIS one):")
    for r in rows:
        print("    %-22s AUROC [%.3f, %.3f]   PR-AUC [%.3f, %.3f]"
              % (r["model"], r["auroc_ci_series"][0], r["auroc_ci_series"][1],
                 r["pr_auc_ci_series"][0], r["pr_auc_ci_series"][1]))

    best = max(rows, key=lambda r: r["pr_auc"])
    width = best["pr_auc_ci"][1] - best["pr_auc_ci"][0]
    print("\n" + "=" * 82)
    print("  STRONGEST BASELINE: %s  (PR-AUC %.3f, AUROC %.3f)"
          % (best["model"], best["pr_auc"], best["auroc"]))
    print("  Its 95%% CI spans %.3f PR-AUC. Any model must clear roughly" % width)
    print("  +%.3f PR-AUC before the difference is real." % (width / 2))

    lip = best["f1"] / 2
    print("\n  Lipton check: F1-optimal threshold should be ~F1/2 = %.2f; grid found %.2f."
          % (lip, best["thr_f1"]))
    print("  %s" % ("consistent -- probabilities look calibrated"
                    if abs(lip - best["thr_f1"]) < 0.10 else
                    "DIVERGENT -- probabilities may be poorly calibrated"))

    out = os.path.join(os.path.dirname(__file__), "..", "results", "baselines_strong.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"corpus_fingerprint": corpus_fingerprint(), "results": rows}, fh,
                  indent=2, default=float)
    print("\n  written -> results/baselines_strong.json")


if __name__ == "__main__":
    if "--cv" in sys.argv:
        run_cv()
        sys.exit(0)
    main()
