"""
Train the recommended model on the SIMULATOR corpus and profile the data.

WHAT IS COMPARED (5-fold series cross-validation -- every one of the 70 M5
series is a test series exactly once, never seen in training or validation):

  logistic regression        reference, CPU
  LightGBM                   the model validated in backorder_benchmark.py, CPU
  LightGBM EasyEnsemble      the benchmark's winning recipe, CPU
  XGBoost (CUDA)             same model family, trained on the GPU
  XGBoost EasyEnsemble (CUDA)

WHY LightGBM IS ON CPU. The pip LightGBM wheel is built without its GPU/CUDA
tree learners, and rebuilding needs the CUDA toolkit + CMake, neither of which
is installed. XGBoost's wheel ships CUDA support, so it carries the GPU run.

WHY EasyEnsemble MAY NOT HELP HERE. On the Kaggle data it won by using the
~1.9M majority rows that undersampling threw away. The simulator corpus
discards nothing -- every row already trains the plain model -- so the bags
here only re-partition data the plain model already sees. Tested, not assumed.

All probabilities are Platt-calibrated on the validation series before
scoring, because participants will see risk percentages. Intervals are
series-clustered bootstrap (the series is the independent unit; RESULTS.md s6).

Run:  python lightgbm-pivot/simulator_training.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from baselines_strong import (HORIZON, LOOKBACK, load_corpus, platt,  # noqa: E402
                              series_folds, thr_for)
from compare_fair import corpus_fingerprint  # noqa: E402
from train_baselines import build_features, expected_calibration_error  # noqa: E402

OUT = os.path.join(HERE, "simulator_results.json")
N_FOLDS, N_BOOT = 5, 2000


# ------------------------------------------------------------------ models
def fit_lr(X, y, tr, va):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
    return lambda M: m.predict_proba(sc.transform(X[M]))[:, 1]


def fit_lgbm(X, y, tr, va, seed=0):
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=1.0,
                           random_state=seed, verbose=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return lambda M: m.predict_proba(X[M])[:, 1]


def fit_xgb(X, y, tr, va, seed=0):
    m = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2000,
                          learning_rate=0.03, max_depth=6, min_child_weight=5,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          eval_metric="aucpr", early_stopping_rounds=100,
                          random_state=seed)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    assert m.get_booster().save_config().find('"device":"cuda') >= 0, "XGBoost did not run on CUDA"
    return lambda M: m.predict_proba(X[M])[:, 1]


def easy_ensemble(fit, X, y, tr, va):
    """All training positives + a disjoint slice of negatives per bag."""
    idx = np.flatnonzero(tr)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    neg = np.random.default_rng(0).permutation(neg)
    k = int(np.ceil(len(neg) / len(pos)))
    preds = []
    for b, chunk in enumerate(np.array_split(neg, k)):
        bag = np.zeros(len(y), bool); bag[pos] = True; bag[chunk] = True
        preds.append(fit(X, y, bag, va, seed=b))
    return lambda M: np.mean([p(M) for p in preds], axis=0), k


# ------------------------------------------------------------------ metrics
def point(y, p):
    thr = thr_for("f1", y, p)
    yh = (p >= thr).astype(int)
    acc_thr = thr_for("accuracy", y, p)
    return {"accuracy": float(accuracy_score(y, (p >= acc_thr).astype(int))),
            "auroc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "f1": float(f1_score(y, yh)),
            "precision": float(precision_score(y, yh, zero_division=0)),
            "recall": float(recall_score(y, yh)),
            "brier": float(brier_score_loss(y, p)),
            "ece": float(expected_calibration_error(y, p))}


def cluster_boot(y, preds, series, ref):
    """Series-clustered bootstrap: CIs per model and paired diffs vs `ref`."""
    rng = np.random.default_rng(0)
    uniq = np.unique(series)
    rows = {s: np.flatnonzero(series == s) for s in uniq}
    stats = {n: {"auroc": [], "pr_auc": []} for n in preds}
    diffs = {n: {"auroc": [], "pr_auc": []} for n in preds if n != ref}
    for _ in range(N_BOOT):
        ix = np.concatenate([rows[s] for s in rng.choice(uniq, len(uniq))])
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        cur = {n: (roc_auc_score(yy, p[ix]), average_precision_score(yy, p[ix]))
               for n, p in preds.items()}
        for n, (a, pr) in cur.items():
            stats[n]["auroc"].append(a); stats[n]["pr_auc"].append(pr)
            if n != ref:
                diffs[n]["auroc"].append(a - cur[ref][0])
                diffs[n]["pr_auc"].append(pr - cur[ref][1])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return ({n: {m: ci(v) for m, v in s.items()} for n, s in stats.items()},
            {n: {m: ci(v) for m, v in s.items()} for n, s in diffs.items()})


# ------------------------------------------------------------------ profile
def profile(df_all, df, y):
    """Descriptive profile of the corpus -- the dataset's 'demographics'."""
    ep = df_all.groupby("episode")
    per_series = df.groupby("m5_series")["stockout_within_%d" % HORIZON].mean()
    return {
        "corpus_fingerprint": corpus_fingerprint(),
        "episodes": int(df_all["episode"].nunique()),
        "m5_series": int(df_all["m5_series"].nunique()),
        "periods_per_episode": int(ep.size().median()),
        "rows_total": int(len(df_all)),
        "rows_modelled": int(len(df)),
        "policy_mode_share": {k: round(float(v), 3) for k, v in
                              df_all.groupby("episode")["policy_mode"].first()
                              .value_counts(normalize=True).items()},
        "label": "stockout_within_%d" % HORIZON,
        "positive_rate": float(y.mean()),
        "positive_rate_by_series": {"min": float(per_series.min()),
                                    "median": float(per_series.median()),
                                    "max": float(per_series.max())},
        "demand": {"mean": float(df_all["demand"].mean()),
                   "sd": float(df_all["demand"].std()),
                   "p5": float(df_all["demand"].quantile(.05)),
                   "p95": float(df_all["demand"].quantile(.95))},
        "inventory_mean": float(df_all["inventory"].mean()),
        "backorders_mean": float(df_all["backorders"].mean()),
        "lead_time_mean": float(df_all["lead_time"].mean()),
        "disruption_period_rate": float(df_all["disruption_flag"].mean()),
        "transport_down_rate": float(1 - df_all["transport_ok"].mean()),
        "supplier_reliability_mean": float(df_all["supplier_reliability"].mean()),
        "stockout_period_rate": float(df_all["stockout"].mean()),
        "service_level_mean": float(df_all["service_level"].mean()),
        "cost_per_period_mean": float(df_all["cost"].mean()),
    }


# ------------------------------------------------------------------ main
def main():
    label = "stockout_within_%d" % HORIZON
    df_all = load_corpus()
    df = df_all[df_all["label_valid"] == 1].reset_index(drop=True)
    X = build_features(df)
    keep = (df["period"] >= LOOKBACK).values
    df, X = df[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    y = df[label].values.astype(int)
    feat = [c for c in X.columns if c not in ("episode", "period")]
    Xv = np.nan_to_num(X[feat].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ser = df["m5_series"].values

    prof = profile(df_all, df, y)
    print("corpus %s | %d rows | %d series | %d features | positive %.3f"
          % (prof["corpus_fingerprint"], len(y), len(np.unique(ser)), len(feat), y.mean()))

    specs = {
        "logistic regression": ("cpu", lambda tr, va: (fit_lr(Xv, y, tr, va), 1)),
        "LightGBM": ("cpu", lambda tr, va: (fit_lgbm(Xv, y, tr, va), 1)),
        "LightGBM EasyEnsemble": ("cpu", lambda tr, va: easy_ensemble(fit_lgbm, Xv, y, tr, va)),
        "XGBoost (CUDA)": ("cuda", lambda tr, va: (fit_xgb(Xv, y, tr, va), 1)),
        "XGBoost EasyEnsemble (CUDA)": ("cuda", lambda tr, va: easy_ensemble(fit_xgb, Xv, y, tr, va)),
    }
    oof = {n: np.full(len(y), np.nan) for n in specs}
    per_fold = {n: [] for n in specs}
    seconds = {n: 0.0 for n in specs}
    bags = {}
    for k in range(N_FOLDS):
        tr, va, te = series_folds(ser, k, N_FOLDS)
        for name, (_, fit) in specs.items():
            t0 = time.time()
            predict, nb = fit(tr, va)
            _, cal = platt(predict(va), y[va], predict(te))
            seconds[name] += time.time() - t0
            bags[name] = nb
            oof[name][te] = cal
            per_fold[name].append({"auroc": float(roc_auc_score(y[te], cal)),
                                   "pr_auc": float(average_precision_score(y[te], cal))})
        print("fold %d  train %d / val %d / test %d series-disjoint rows   "
              % (k, tr.sum(), va.sum(), te.sum()) +
              "  ".join("%s %.3f" % (n.split()[0][:4], per_fold[n][-1]["auroc"]) for n in specs))

    for n, p in oof.items():
        assert not np.isnan(p).any(), "every row must be scored out-of-fold exactly once"

    ref = "LightGBM"
    ci, diff = cluster_boot(y, oof, ser, ref)
    results = []
    print("\n%-28s %6s %7s %7s %7s %6s %6s %7s %7s" %
          ("model", "device", "acc", "AUROC", "PR-AUC", "F1", "ECE", "train s", "bags"))
    for n, (dev, _) in specs.items():
        r = {"model": n, "device": dev, **point(y, oof[n]),
             "auroc_ci": ci[n]["auroc"], "pr_auc_ci": ci[n]["pr_auc"],
             "auroc_fold_sd": float(np.std([f["auroc"] for f in per_fold[n]])),
             "pr_auc_fold_sd": float(np.std([f["pr_auc"] for f in per_fold[n]])),
             "per_fold": per_fold[n], "train_seconds": seconds[n], "bags": bags[n]}
        if n != ref:
            r["diff_vs_lightgbm_ci"] = diff[n]
        results.append(r)
        print("%-28s %6s %7.3f %7.3f %7.3f %6.3f %6.3f %7.0f %7d"
              % (n, dev, r["accuracy"], r["auroc"], r["pr_auc"], r["f1"], r["ece"],
                 seconds[n], bags[n]))

    majority = float(max(y.mean(), 1 - y.mean()))
    print("\nmajority-class accuracy %.3f" % majority)
    for n, d in diff.items():
        print("  %-28s dAUROC vs LightGBM 95%% CI [%+.4f, %+.4f]   dPR-AUC [%+.4f, %+.4f]"
              % (n, *d["auroc"], *d["pr_auc"]))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"corpus_fingerprint": corpus_fingerprint(), "n_folds": N_FOLDS,
                   "n_boot": N_BOOT, "majority_accuracy": majority,
                   "profile": prof, "results": results}, fh, indent=2)
    print("\nwritten -> %s" % OUT)


if __name__ == "__main__":
    main()
