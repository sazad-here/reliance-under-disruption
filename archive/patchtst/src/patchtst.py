"""
PatchTST for stockout classification.

Architecture follows Nie, Nguyen, Sinthong & Kalagnanam (2023), "A Time Series
is Worth 64 Words", ICLR, arXiv:2211.14730 -- patching, channel independence and
RevIN instance normalisation, with a classification head in place of the
forecasting head.

DEVIATION FROM THE PLAN, worth stating in the method: section 05 says to use
Nixtla neuralforecast or the official yuqinie98 implementation rather than
reimplement. Both target forecasting (continuous multi-step output). Our target
is binary -- does a stockout occur within h periods -- so the encoder is
reproduced here with a classification head. The encoder itself is unchanged.

Hyperparameters are exactly those specified in plan section 05.

Run:  python src/patchtst.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(__file__))

# ---- config, per plan section 05 ----
LOOKBACK = 48        # L
PATCH_LEN = 8        # P
STRIDE = 4           # S  -> 11 overlapping patches
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
FFN = 128
DROPOUT = 0.2
LR = 1e-3
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 10
MC_PASSES = 50       # Monte-Carlo dropout passes at inference
HORIZON = 2
SEEDS = (0, 1, 2, 3, 4)

CHANNELS = [
    "demand", "inventory", "on_order", "lead_time", "backorders",
    "supplier_reliability", "transport_ok", "disruption_flag",
]


class RevIN(nn.Module):
    """Reversible instance normalisation: per-instance, per-channel.

    Removes the level and scale of each window so the encoder sees shape rather
    than magnitude. This is what lets one set of weights serve a warehouse
    holding 40 units and one holding 600.
    """

    def __init__(self, n_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x):                      # x: (B, L, C)
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / std
        return x * self.weight + self.bias


class PatchTSTClassifier(nn.Module):
    def __init__(self, n_channels=len(CHANNELS), lookback=LOOKBACK,
                 patch_len=PATCH_LEN, stride=STRIDE):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1

        self.revin = RevIN(n_channels)
        # "A time series is worth 64 words": each patch of `patch_len` readings
        # becomes one d_model-dimensional token.
        self.embed = nn.Linear(patch_len, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, D_MODEL) * 0.02)
        self.drop = nn.Dropout(DROPOUT)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN,
            dropout=DROPOUT, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(n_channels * D_MODEL, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):                      # x: (B, L, C)
        B = x.shape[0]
        x = self.revin(x)
        x = x.permute(0, 2, 1)                 # (B, C, L)
        # cut the series into overlapping patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # (B, C, n_patches, patch_len) -> channel independence: fold C into batch,
        # so every variable is read separately with SHARED weights.
        x = x.reshape(B * self.n_channels, self.n_patches, self.patch_len)
        x = self.embed(x) + self.pos
        x = self.drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)                      # pool over patches
        x = x.reshape(B, self.n_channels * D_MODEL)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------- data
def load_corpus():
    d = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    for name in ("corpus.parquet", "corpus.csv"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return pd.read_parquet(p) if name.endswith("parquet") else pd.read_csv(p)
    raise SystemExit("No corpus. Run:  python src/make_corpus.py")


def build_windows(df, label_col):
    """(N, LOOKBACK, C) windows, never crossing an episode boundary."""
    Xs, ys, eps = [], [], []
    for ep, g in df.groupby("episode", sort=False):
        g = g.sort_values("period")
        arr = g[CHANNELS].values.astype(np.float32)
        lab = g[label_col].values.astype(np.float32)
        valid = g["label_valid"].values
        for t in range(LOOKBACK, len(g)):
            if not valid[t]:
                continue
            Xs.append(arr[t - LOOKBACK:t])
            ys.append(lab[t])
            eps.append(ep)
    return np.stack(Xs), np.array(ys, dtype=np.float32), np.array(eps)


def mc_dropout_predict(model, X, passes=MC_PASSES, batch=512):
    """Keep dropout ON at inference: spread across passes = model uncertainty.

    Plan section 05 -- this is what turns one number into a confidence, and what
    Step 5 of the explainer describes.
    """
    model.train()  # deliberately: dropout active
    outs = []
    with torch.no_grad():
        for _ in range(passes):
            preds = []
            for i in range(0, len(X), batch):
                preds.append(torch.sigmoid(model(X[i:i + batch])).cpu().numpy())
            outs.append(np.concatenate(preds))
    arr = np.stack(outs)
    return arr.mean(axis=0), arr.std(axis=0)


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            ece += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def train_one(seed, Xtr, ytr, Xva, yva, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PatchTSTClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss, best_state, bad = math.inf, None, 0
    n = len(Xtr)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva), yva).item()
        if vl < best_loss - 1e-4:
            best_loss, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if verbose and epoch % 5 == 0:
            print("      epoch %3d  val loss %.4f" % (epoch, vl))

    model.load_state_dict(best_state)
    return model, best_loss, epoch + 1


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    label = "stockout_within_%d" % HORIZON
    df = load_corpus()

    print("=" * 74)
    print("PATCHTST  --  L=%d, patch=%d, stride=%d, %d patches, d_model=%d"
          % (LOOKBACK, PATCH_LEN, STRIDE, (LOOKBACK - PATCH_LEN) // STRIDE + 1, D_MODEL))
    print("=" * 74)

    X, y, eps = build_windows(df, label)
    print("  windows: %d   shape %s" % (len(X), (X.shape[1], X.shape[2])))

    uniq = np.sort(np.unique(eps))
    n = len(uniq)
    tr_e = set(uniq[: int(0.70 * n)])
    va_e = set(uniq[int(0.70 * n): int(0.85 * n)])
    te_e = set(uniq[int(0.85 * n):])
    m_tr = np.isin(eps, list(tr_e))
    m_va = np.isin(eps, list(va_e))
    m_te = np.isin(eps, list(te_e))

    Xtr = torch.tensor(X[m_tr]); ytr = torch.tensor(y[m_tr])
    Xva = torch.tensor(X[m_va]); yva = torch.tensor(y[m_va])
    Xte = torch.tensor(X[m_te]); yte = y[m_te]
    print("  split:   %d train / %d val / %d test windows" % (len(Xtr), len(Xva), len(Xte)))
    print("  params:  %d\n" % sum(p.numel() for p in PatchTSTClassifier().parameters()))

    runs = []
    for seed in SEEDS:
        t0 = time.time()
        model, vl, epochs = train_one(seed, Xtr, ytr, Xva, yva)
        prob, unc = mc_dropout_predict(model, Xte)
        r = {
            "seed": seed,
            "epochs": epochs,
            "accuracy": float(accuracy_score(yte, (prob >= 0.5).astype(int))),
            "auroc": float(roc_auc_score(yte, prob)),
            "pr_auc": float(average_precision_score(yte, prob)),
            "brier": float(brier_score_loss(yte, prob)),
            "ece": expected_calibration_error(yte, prob),
            "mean_uncertainty": float(unc.mean()),
        }
        runs.append(r)
        print("  seed %d  %2d epochs  %5.1fs   acc %.3f  AUROC %.3f  PR-AUC %.3f  ECE %.3f"
              % (seed, epochs, time.time() - t0, r["accuracy"], r["auroc"], r["pr_auc"], r["ece"]))

    print("\n" + "=" * 74)
    print("RESULT  (mean +/- SD over %d seeds, per plan section 05)" % len(SEEDS))
    print("=" * 74)
    for k in ("accuracy", "auroc", "pr_auc", "brier", "ece"):
        vals = [r[k] for r in runs]
        print("  %-10s %.3f +/- %.3f" % (k, np.mean(vals), np.std(vals)))

    majority = max(yte.mean(), 1 - yte.mean())
    print("\n  majority-class accuracy: %.3f   <- accuracy must be read against this" % majority)
    print("  mean MC-dropout spread:  %.3f   <- the confidence shown to participants"
          % np.mean([r["mean_uncertainty"] for r in runs]))

    out = os.path.join(os.path.dirname(__file__), "..", "results", "patchtst.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"config": {"lookback": LOOKBACK, "patch_len": PATCH_LEN,
                              "stride": STRIDE, "d_model": D_MODEL,
                              "horizon": HORIZON}, "runs": runs}, fh, indent=2)
    print("\n  written -> results/patchtst.json")


if __name__ == "__main__":
    main()
