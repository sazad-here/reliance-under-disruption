"""
Curve-level data for the model-comparison charts (charts.html).

simulator_results.json keeps only summary numbers. The charts papers show --
ROC and precision-recall curves, calibration plots, a confusion matrix,
per-fold spread, SHAP -- need every row's out-of-fold prediction. This script
re-runs the SAME 5-fold series cross-validation as simulator_training.py
(same model settings, same folds, same Platt step, imported, not copied) and
keeps those predictions. It checks its AUROCs against simulator_results.json
before writing anything, so the charts cannot drift from the reported numbers.

It also answers two checks from validation-plan.html:
  #2 calibration intercept and slope (logistic recalibration of y on logit p)
  #3 SHAP for the LIVE model, on the two session series it never trained on
     (LightGBM's own pred_contrib gives exact TreeSHAP values; no extra package)

Run:  python lightgbm-pivot/chart_data.py
Out:  lightgbm-pivot/chart_data.json and chart_oof.npz (row-level predictions)
"""
from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from simulator_training import fit_lgbm, fit_lr, fit_xgb, N_FOLDS  # noqa: E402
from baselines_strong import LOOKBACK, HORIZON, load_corpus, platt, series_folds, thr_for  # noqa: E402
from compare_fair import corpus_fingerprint  # noqa: E402
from train_baselines import build_features  # noqa: E402

OUT = os.path.join(HERE, "chart_data.json")
SAVED = os.path.join(HERE, "simulator_results.json")


def thin(xs, ys, n=120):
    """Keep a curve's shape with ~n points (charts don't need 25k vertices)."""
    idx = np.unique(np.linspace(0, len(xs) - 1, n).round().astype(int))
    return [round(float(xs[i]), 4) for i in idx], [round(float(ys[i]), 4) for i in idx]


def calibration(y, p, bins=10):
    """Quantile-binned reliability points plus logistic intercept and slope."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    pts = [{"pred": float(p[b == i].mean()), "obs": float(y[b == i].mean()), "n": int((b == i).sum())}
           for i in range(bins) if (b == i).any()]
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1)
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(z, y).coef_[0, 0])
    # calibration-in-the-large: intercept with the slope fixed at 1 (offset model)
    off = z.ravel()
    a = 0.0
    for _ in range(50):  # Newton steps on the single intercept parameter
        q = 1 / (1 + np.exp(-(a + off)))
        a += float((y - q).sum() / max((q * (1 - q)).sum(), 1e-9))
    return {"points": pts, "slope": slope, "intercept": a}


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
    # net position = inventory position minus demand over the lead time, as in
    # live_model.features(); the one-number baseline every model has to beat
    net = (df["inventory"] + df["on_order"] - df["backorders"]
           - df["demand"] * (df["lead_time"] + 1)).values.astype(np.float32).reshape(-1, 1)

    specs = {
        "Net position only": lambda tr, va: fit_lr(net, y, tr, va),
        "Logistic regression": lambda tr, va: fit_lr(Xv, y, tr, va),
        "LightGBM": lambda tr, va: fit_lgbm(Xv, y, tr, va),
        "XGBoost": lambda tr, va: fit_xgb(Xv, y, tr, va),
    }
    oof = {n: np.full(len(y), np.nan) for n in specs}
    folds = {n: [] for n in specs}
    for k in range(N_FOLDS):
        tr, va, te = series_folds(ser, k, N_FOLDS)
        for name, fit in specs.items():
            predict = fit(tr, va)
            _, cal = platt(predict(va), y[va], predict(te))
            oof[name][te] = cal
            folds[name].append(float(roc_auc_score(y[te], cal)))
        print("fold %d  " % k + "  ".join("%s %.3f" % (n.split()[0], folds[n][-1]) for n in specs))

    # --- guard: must match the numbers already reported -----------------------
    saved = {r["model"]: r["auroc"] for r in json.load(open(SAVED, encoding="utf-8"))["results"]}
    for mine, theirs in (("Logistic regression", "logistic regression"),
                         ("LightGBM", "LightGBM"), ("XGBoost", "XGBoost (CUDA)")):
        got = roc_auc_score(y, oof[mine])
        print("check %-20s AUROC %.4f vs saved %.4f" % (mine, got, saved[theirs]))
        assert abs(got - saved[theirs]) < 0.002, "charts would not match simulator_results.json"

    models = {}
    for n, p in oof.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        o = np.argsort(rec)
        rx, ry = thin(rec[o], prec[o])
        fx, fy = thin(fpr, tpr)
        models[n] = {"auroc": float(roc_auc_score(y, p)),
                     "pr_auc": float(average_precision_score(y, p)),
                     "roc": {"fpr": fx, "tpr": fy}, "pr": {"recall": rx, "precision": ry},
                     "calibration": calibration(y, p), "fold_auroc": folds[n]}
    thr = thr_for("f1", y, oof["LightGBM"])
    cm = confusion_matrix(y, (oof["LightGBM"] >= thr).astype(int)).tolist()

    # --- SHAP for the live model on the unseen session series -----------------
    import live_model
    from interface.world import SESSION_SERIES
    data = live_model._build(df_all)
    data = data[(data["_valid"] == 1) & data["_series"].isin(SESSION_SERIES)].reset_index(drop=True)
    meta = json.load(open(live_model.META_PATH, encoding="utf-8"))
    cols = meta["features"]
    booster = lgb.Booster(model_file=live_model.MODEL_PATH)
    contrib = booster.predict(data[cols], pred_contrib=True)[:, :-1]  # last column = bias
    mean_abs = np.abs(contrib).mean(axis=0)
    order = np.argsort(-mean_abs)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(data), size=min(300, len(data)), replace=False)
    top = [int(i) for i in order[:8]]
    shap = {"n_rows": int(len(data)), "series": list(SESSION_SERIES),
            "importance": [{"feature": cols[i], "mean_abs": float(mean_abs[i])} for i in order[:12]],
            "swarm": [{"feature": cols[i],
                       "shap": [round(float(v), 4) for v in contrib[sample, i]],
                       "value_rank": [round(float(v), 3) for v in
                                      (data[cols[i]].rank(pct=True).fillna(0.5).values[sample])]}
                      for i in top]}
    print("live-model SHAP top 5:", ", ".join(cols[i] for i in order[:5]))

    blob = {"corpus_fingerprint": corpus_fingerprint(), "n_rows": int(len(y)),
            "positive_rate": float(y.mean()), "models": models,
            "confusion_lightgbm": {"threshold": float(thr), "matrix": cm}, "shap_live": shap}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    # row-level out-of-fold predictions, for distribution figures
    np.savez_compressed(os.path.join(HERE, "chart_oof.npz"), y=y, series=ser,
                        **{n.replace(" ", "_"): p for n, p in oof.items()})
    for n, m in models.items():
        c = m["calibration"]
        print("%-24s AUROC %.3f  PR-AUC %.3f  cal slope %.2f  intercept %+.3f"
              % (n, m["auroc"], m["pr_auc"], c["slope"], c["intercept"]))
    print("written -> %s" % OUT)


if __name__ == "__main__":
    main()
