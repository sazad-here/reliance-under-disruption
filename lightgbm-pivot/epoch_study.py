"""
Every model, 50 vs 100 epochs, identical test series -- does LightGBM win?

The question asked: does LightGBM outperform all other models, and what
changes between 50 and 100 epochs of training? Answered on the simulator
corpus with 5-fold series cross-validation (every M5 series is a test series
exactly once, never seen in training), all models scored on identical rows.

Models
  tabular (116 engineered features): logistic regression, LightGBM, XGBoost
      (CUDA), MLP (torch, epochs)
  sequence (48-week window, 22 channels): CNN, LSTM, GRU, vanilla
      Transformer, JointPatchTransformer, PatchTST v2
"Epochs" for neural nets: each trains a fixed 100 epochs (no early stopping)
and is scored on the test fold at epoch 50 AND epoch 100, plus at the epoch
validation AUROC picks. Boosted trees have no epochs; the analogue is the
number of trees, so they are scored at 50 and 100 trees and at their
validation-early-stopped count (the configuration validated elsewhere).
Rank metrics (AUROC, PR-AUC) are unaffected by monotone calibration, so none
is applied. Single seed; the series-clustered bootstrap supplies uncertainty.

Run:  python lightgbm-pivot/epoch_study.py     (GPU; roughly an hour)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "archive", "patchtst", "src"))

import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from baselines_strong import load_corpus, series_folds  # noqa: E402
from extra_baselines import (CNN1D, GRUClf, JointPatchTransformer, LSTMClf,  # noqa: E402
                             VanillaTransformer, build_tabular)
from patchtst_v2 import CHANNELS, LOOKBACK, PatchTSTv2, add_derived, build_windows  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL, EPOCHS, CHECK, BATCH, LR = "stockout_within_2", 100, 50, 256, 1e-3
N_FOLDS, N_BOOT = 5, 2000
OUT = os.path.join(HERE, "epoch_study.json")


class TabMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict(model, X):
    model.eval()
    with torch.no_grad(), torch.autocast(DEVICE.type, torch.bfloat16, enabled=DEVICE.type == "cuda"):
        out = torch.cat([model(X[i:i + 4096]).float() for i in range(0, len(X), 4096)])
    return torch.sigmoid(out).cpu().numpy()


def train_epochs(make, Xtr, ytr, Xva, yva, Xte, seed=0):
    """Fixed 100 epochs; test predictions at epoch 50, 100 and the val-best epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = make().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    yva_np = yva.cpu().numpy()
    best, snaps, curve = -1.0, {}, []
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            with torch.autocast(DEVICE.type, torch.bfloat16, enabled=DEVICE.type == "cuda"):
                loss = lossf(model(Xtr[idx]).float(), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = float(roc_auc_score(yva_np, predict(model, Xva)))
        curve.append(v)
        if v > best:
            best, snaps["val_best"], snaps["val_best_epoch"] = v, predict(model, Xte), ep
        if ep == CHECK:
            snaps["ep50"] = predict(model, Xte)
        if ep == EPOCHS:
            snaps["ep100"] = predict(model, Xte)
    snaps["val_curve"] = curve
    return snaps


def boosted(kind, Xv, y, tr, va, te):
    out = {}
    if kind == "lgbm":
        base = dict(learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1)
        for n in (50, 100):
            out["ep%d" % n] = lgb.LGBMClassifier(n_estimators=n, **base).fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
        m = lgb.LGBMClassifier(n_estimators=2000, **base).fit(
            Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)])
        out["val_best"], out["val_best_epoch"] = m.predict_proba(Xv[te])[:, 1], int(m.best_iteration_)
    else:
        base = dict(device="cuda", tree_method="hist", learning_rate=0.03, max_depth=6, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, random_state=0)
        for n in (50, 100):
            out["ep%d" % n] = xgb.XGBClassifier(n_estimators=n, **base).fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
        m = xgb.XGBClassifier(n_estimators=2000, eval_metric="aucpr", early_stopping_rounds=100, **base)
        m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], verbose=False)
        out["val_best"], out["val_best_epoch"] = m.predict_proba(Xv[te])[:, 1], int(m.best_iteration)
    return out


def cluster_ci(y, p, ser, ref=None, rng=np.random.default_rng(0)):
    uniq = np.unique(ser)
    rows = {s: np.flatnonzero(ser == s) for s in uniq}
    a, d = [], []
    for _ in range(N_BOOT):
        ix = np.concatenate([rows[s] for s in rng.choice(uniq, len(uniq))])
        if y[ix].min() == y[ix].max():
            continue
        v = roc_auc_score(y[ix], p[ix]); a.append(v)
        if ref is not None:
            d.append(v - roc_auc_score(y[ix], ref[ix]))
    ci = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    dci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d else None
    return ci, dci


def main():
    t0 = time.time()
    df = load_corpus()
    Xv, y, _, _, _, ser_t = build_tabular(df, LABEL, fold=0)
    Xw, yw, _, ser_w = build_windows(add_derived(df), LABEL)
    assert np.array_equal(y, yw) and np.array_equal(ser_t, ser_w), "tabular and window rows disagree"
    ser = ser_t
    print("rows %d | series %d | tabular features %d | windows %s | device %s"
          % (len(y), len(np.unique(ser)), Xv.shape[1], Xw.shape[1:], DEVICE), flush=True)

    MODELS = ["logistic regression", "LightGBM", "XGBoost (CUDA)", "MLP", "CNN", "LSTM", "GRU",
              "Transformer (vanilla)", "JointPatchTransformer", "PatchTST v2"]
    oof = {m: {k: np.full(len(y), np.nan) for k in ("ep50", "ep100", "val_best")} for m in MODELS}
    best_ep = {m: [] for m in MODELS}
    curves = {m: [] for m in MODELS}

    for k in range(N_FOLDS):
        tr, va, te = series_folds(ser, k)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=DEVICE)
        yv = torch.tensor(y[va], dtype=torch.float32, device=DEVICE)
        sc = StandardScaler().fit(Xv[tr])
        # logistic regression: no epochs; the same prediction fills every column
        p = LogisticRegression(max_iter=4000).fit(sc.transform(Xv[tr]), y[tr]).predict_proba(sc.transform(Xv[te]))[:, 1]
        for c in oof["logistic regression"]:
            oof["logistic regression"][c][te] = p
        for name, kind in (("LightGBM", "lgbm"), ("XGBoost (CUDA)", "xgb")):
            r = boosted(kind, Xv, y, tr, va, te)
            for c in ("ep50", "ep100", "val_best"):
                oof[name][c][te] = r[c]
            best_ep[name].append(r["val_best_epoch"])
        # torch models
        Tt = lambda A, m: torch.tensor(A[m], dtype=torch.float32, device=DEVICE)
        tab = (Tt(sc.transform(Xv).astype(np.float32), tr), Tt(sc.transform(Xv).astype(np.float32), va),
               Tt(sc.transform(Xv).astype(np.float32), te))
        win = (Tt(Xw, tr), Tt(Xw, va), Tt(Xw, te))
        C = len(CHANNELS)
        specs = [("MLP", lambda: TabMLP(Xv.shape[1]), tab),
                 ("CNN", lambda: CNN1D(n_channels=C), win),
                 ("LSTM", lambda: LSTMClf(n_channels=C), win),
                 ("GRU", lambda: GRUClf(n_channels=C), win),
                 ("Transformer (vanilla)", lambda: VanillaTransformer(n_channels=C, lookback=LOOKBACK), win),
                 ("JointPatchTransformer", lambda: JointPatchTransformer(n_channels=C, lookback=LOOKBACK), win),
                 ("PatchTST v2", lambda: PatchTSTv2(), win)]
        for name, make, (a, b, c) in specs:
            t1 = time.time()
            r = train_epochs(make, a, yt, b, yv, c)
            for col in ("ep50", "ep100", "val_best"):
                oof[name][col][te] = r[col]
            best_ep[name].append(r["val_best_epoch"]); curves[name].append(r["val_curve"])
            print("  fold %d  %-22s AUROC ep50 %.3f | ep100 %.3f | val-best %.3f (epoch %d)  %4.0fs"
                  % (k, name, roc_auc_score(y[te], r["ep50"]), roc_auc_score(y[te], r["ep100"]),
                     roc_auc_score(y[te], r["val_best"]), r["val_best_epoch"], time.time() - t1), flush=True)
        print("fold %d done  (%.0f min elapsed)" % (k, (time.time() - t0) / 60), flush=True)

    ref = oof["LightGBM"]["val_best"]
    res = {}
    print("\n%-24s %9s %9s %9s   %-17s %s" % ("model", "ep50", "ep100", "val-best", "95% CI (val-best)",
                                              "diff vs LightGBM, 95% CI"))
    for m in MODELS:
        a50, a100, ab = (float(roc_auc_score(y, oof[m][c])) for c in ("ep50", "ep100", "val_best"))
        ci, dci = cluster_ci(y, oof[m]["val_best"], ser, None if m == "LightGBM" else ref)
        res[m] = {"auroc_ep50": a50, "auroc_ep100": a100, "auroc_val_best": ab,
                  "pr_auc_ep50": float(average_precision_score(y, oof[m]["ep50"])),
                  "pr_auc_ep100": float(average_precision_score(y, oof[m]["ep100"])),
                  "pr_auc_val_best": float(average_precision_score(y, oof[m]["val_best"])),
                  "auroc_ci": ci, "diff_vs_lightgbm_ci": dci,
                  "val_best_epochs": best_ep[m], "val_curves": curves[m]}
        print("%-24s %9.3f %9.3f %9.3f   [%.3f, %.3f]   %s" % (
            m, a50, a100, ab, ci[0], ci[1], "reference" if dci is None else "[%+.4f, %+.4f]" % tuple(dci)))
    rank = sorted(MODELS, key=lambda m: -res[m]["auroc_val_best"])
    print("\nranking by AUROC (val-best):", " > ".join(rank))
    json.dump({"n_rows": int(len(y)), "folds": N_FOLDS, "epochs": [CHECK, EPOCHS],
               "results": res, "ranking": rank, "minutes": (time.time() - t0) / 60},
              open(OUT, "w"), indent=2)
    print("written ->", OUT)


if __name__ == "__main__":
    main()
