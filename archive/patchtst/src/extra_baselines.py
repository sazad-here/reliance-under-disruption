"""
The models from the reference paper we haven't actually run: SVM, CNN, LSTM,
vanilla Transformer.

WHY THESE FOUR

AI-Driven_Predictive_Analytics_Framework_for_Proactive_Supply_Chain_Risk_
Identification_and_Mitigation.pdf (the IEEE CCGE 2026 paper Document (14).docx
cites as its technical reference for PatchTST) compares its PatchTST-based
framework against exactly these four baselines and reports PatchTST winning
on every metric. Our own pipeline never ran three of them -- it compares
PatchTST against tabular models (logistic regression, HistGB, LightGBM), a
different and arguably harder baseline family (Grinsztajn et al. 2022: tree
ensembles are SOTA at our data scale). This script fills the gap: the same
four architectures the reference paper used, run on our own leak-free,
series-split corpus, so the comparison is against models from the paper we're
citing, not just the ones we happened to build first.

WHAT'S DELIBERATELY DIFFERENT FROM patchtst_v2.py's RIGOR

  * SINGLE SEED, not 5. This is an exploratory extension, not a headline
    result -- disclosed here and in the printed output, not hidden.
  * EARLY STOPPING (patience=8) rather than PatchTST's full-budget-then-
    checkpoint-select. CNN/LSTM/Transformer here are much smaller and
    converge faster; PatchTST's own code has a long comment on why full-budget
    was chosen for IT specifically (OneCycleLR peaks mid-schedule). No such
    finding was made for these three, so ordinary early stopping is used
    rather than copying a decision that was justified for a different model.
  * SVM uses the 116 hand-built tabular features (same ones LR/HistGB/
    LightGBM get), not the windowed sequence -- flattening a (48, 22) window
    into a 1056-dim vector for an SVM is a much weaker representation than
    either the engineered features or a model built to exploit sequence
    structure, so it would not be a fair test of SVM. CNN/LSTM/Transformer
    get the same windowed input as PatchTST, because exploiting sequence
    structure is the whole point of comparing them to it.

Run:  python src/extra_baselines.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(__file__))

from baselines_strong import series_folds  # noqa: E402
from compare_fair import corpus_fingerprint  # noqa: E402
from patchtst_v2 import CHANNELS, LOOKBACK, add_derived, build_windows, ece  # noqa: E402
from train_baselines import build_features  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZON = 2
SEED = 0
MAX_EPOCHS, PATIENCE, BATCH, LR = 40, 8, 128, 1e-3


def load_corpus():
    d = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    for n in ("corpus.parquet", "corpus.csv"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return pd.read_parquet(p) if n.endswith("parquet") else pd.read_csv(p)
    raise SystemExit("No corpus. Run: python src/make_corpus.py")


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: accuracy_score(y, (p >= t).astype(int))))


def score(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "threshold": round(float(thr), 3),
    }


# --------------------------------------------------------------- tabular ---
def build_tabular(df, label_col, fold=0):
    """The 116 hand-built features plus a series_folds() split.

    Shared by SVM and the new tabular MLP so both see identical rows and
    identical splits -- and, since this is the same series_folds() call
    patchtst_v2.py and this file's temporal models use, the SAME held-out
    series as the CNN/LSTM/GRU/Transformer/PatchTST results too. This used
    to call df["split"] instead (the corpus's baked-in 70/15/15 partition),
    which is a DIFFERENT split of the same 70 M5 series than series_folds()
    produces -- checked directly, the two test sets shared zero series in
    common. baselines_strong.py's main() had the same bug; fixed there too.
    """
    dfl = df[df["label_valid"] == 1].reset_index(drop=True)
    X = build_features(dfl)
    keep = (dfl["period"] >= LOOKBACK).values
    dfl, X = dfl[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    y = dfl[label_col].values
    feat = [c for c in X.columns if c not in ("episode", "period")]
    Xv = np.nan_to_num(X[feat].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    ser = dfl["m5_series"].values
    m_tr, m_va, m_te = series_folds(ser, fold)
    assert not set(ser[m_tr]) & set(ser[m_te]), "series leaked across train/test"
    return Xv, y, m_tr, m_va, m_te, ser


def run_svm(Xv, y, m_tr, m_va, m_te):
    """Linear SVM on the 116 hand-built tabular features, Platt-calibrated.

    LinearSVC has no predict_proba; Platt's original method (1999) fits a
    1-D logistic regression directly on the raw decision_function score,
    which is exactly what's done here -- not the logit-of-probability
    version baselines_strong.platt() uses, because there is no probability
    to take a logit of yet.
    """
    sc = StandardScaler().fit(Xv[m_tr])
    svm = LinearSVC(C=1.0, max_iter=10000, dual="auto", random_state=SEED)
    svm.fit(sc.transform(Xv[m_tr]), y[m_tr])

    d_va = svm.decision_function(sc.transform(Xv[m_va]))
    d_te = svm.decision_function(sc.transform(Xv[m_te]))
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(d_va.reshape(-1, 1), y[m_va])
    p_va = cal.predict_proba(d_va.reshape(-1, 1))[:, 1]
    p_te = cal.predict_proba(d_te.reshape(-1, 1))[:, 1]

    thr = best_threshold(y[m_va], p_va)
    return score(y[m_te], p_te, thr)


def run_mlp_tabular(Xv, y, m_tr, m_va, m_te):
    """Plain feedforward net on the same 116 features LR/SVM/HistGB/LightGBM
    get. The gap this fills: every tabular model tried so far is either
    linear (LR, SVM) or a tree ensemble (HistGB, LightGBM) -- nothing tests
    whether a plain deep net on the SAME engineered features does any
    better. sklearn's MLPClassifier keeps this consistent with the rest of
    the tabular family's sklearn-based fit/predict_proba pattern rather than
    hand-rolling another torch training loop for a two-hidden-layer net.
    """
    from sklearn.neural_network import MLPClassifier

    sc = StandardScaler().fit(Xv[m_tr])
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
        max_iter=200, random_state=SEED,
    ).fit(sc.transform(Xv[m_tr]), y[m_tr])

    p_va = mlp.predict_proba(sc.transform(Xv[m_va]))[:, 1]
    p_te = mlp.predict_proba(sc.transform(Xv[m_te]))[:, 1]
    p_va_cal, p_te_cal = platt_torch(p_va, y[m_va], p_te)
    thr = best_threshold(y[m_va], p_va_cal)
    return score(y[m_te], p_te_cal, thr)


# --------------------------------------------------------- shared torch ---
def platt_torch(p_va, y_va, p_te):
    eps = 1e-6
    z_va = np.log(np.clip(p_va, eps, 1 - eps) / np.clip(1 - p_va, eps, 1 - eps))
    z_te = np.log(np.clip(p_te, eps, 1 - eps) / np.clip(1 - p_te, eps, 1 - eps))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z_va.reshape(-1, 1), y_va)
    return (lr.predict_proba(z_va.reshape(-1, 1))[:, 1],
            lr.predict_proba(z_te.reshape(-1, 1))[:, 1])


class CNN1D(nn.Module):
    """Small 1D-CNN over the window: the architecture family paper1 tested."""

    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):              # x: (B, L, C)
        h = self.net(x.permute(0, 2, 1))
        return self.head(h).squeeze(-1)


class LSTMClf(nn.Module):
    """Single-layer LSTM, last hidden state -> classifier head."""

    def __init__(self, n_channels, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):              # x: (B, L, C)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


class GRUClf(nn.Module):
    """Same job as LSTMClf, lighter gate: GRU has no separate cell state,
    roughly 25% fewer recurrent parameters. Fills the gap next to LSTM in
    the recurrent family -- cheap to add, a different bias-variance point,
    not a new idea."""

    def __init__(self, n_channels, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):              # x: (B, L, C)
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class VanillaTransformer(nn.Module):
    """Token-per-timestep Transformer encoder -- no patching, no cross-
    channel fusion stage. The point of comparison against PatchTST v2:
    everything PatchTST adds (patches, cross-channel attention, attention
    pooling) is absent here, so the gap between this and PatchTST estimates
    what those additions are worth on this task."""

    def __init__(self, n_channels, lookback, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.2, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):              # x: (B, L, C)
        h = self.encoder(self.proj(x) + self.pos)
        return self.head(h[:, -1, :]).squeeze(-1)   # last token, like an RNN readout


class JointPatchTransformer(nn.Module):
    """Patching WITHOUT channel-independence -- the architecture the
    reference paper's own equations (III-C, eq. 11-20) actually describe.

    Its P_k = [F(t_k), ..., F(t_k+P-1)] patches the ALREADY-FUSED multivariate
    vector F(t) directly: one token per time-patch, each token carrying all
    channels at once. That is neither patchtst_v2.PatchTSTv2 (patches each
    channel separately, THEN fuses across channels in a second stage) nor
    VanillaTransformer above (no patching at all, one token per timestep).
    It's the missing middle: patching's effect on its own, without
    channel-independence's effect mixed in.

    Built by flattening each (patch_len, n_channels) patch into one vector
    and embedding it with a single linear layer -- structurally identical to
    PatchTSTv2's per-channel patch embedding, just applied to all channels
    at once instead of one at a time.
    """

    def __init__(self, n_channels, lookback, patch_len=8, stride=4,
                 d_model=64, nhead=4, layers=3):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (lookback - patch_len) // stride + 1
        self.embed = nn.Sequential(
            nn.Linear(patch_len * n_channels, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.2, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):              # x: (B, L, C)
        B = x.shape[0]
        x = x.permute(0, 2, 1)                              # (B, C, L)
        x = x.unfold(-1, self.patch_len, self.stride)        # (B, C, P, patch_len)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, -1)  # (B, P, C*patch_len)
        h = self.encoder(self.embed(x) + self.pos)
        return self.head(h[:, -1, :]).squeeze(-1)


def train_torch_model(model_cls, Xtr, ytr, Xva, yva, **kwargs):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = model_cls(**kwargs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    yva_np = yva.detach().cpu().numpy()

    best_auroc, best_state, patience_left = -1.0, None, PATIENCE
    epochs_run = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva)).cpu().numpy()
        auroc = roc_auc_score(yva_np, p)
        epochs_run = epoch + 1
        if auroc > best_auroc + 1e-6:
            best_auroc, best_state, patience_left = auroc, \
                {k: v.detach().clone() for k, v in model.state_dict().items()}, PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return model, epochs_run, best_auroc


def run_temporal(model_name, model_cls, X, y, m_tr, m_va, m_te, **kwargs):
    Xtr = torch.tensor(X[m_tr]).to(DEVICE); ytr = torch.tensor(y[m_tr]).to(DEVICE)
    Xva = torch.tensor(X[m_va]).to(DEVICE); yva = torch.tensor(y[m_va]).to(DEVICE)
    Xte = torch.tensor(X[m_te]).to(DEVICE)

    t0 = time.time()
    model, epochs, val_auroc = train_torch_model(model_cls, Xtr, ytr, Xva, yva, **kwargs)
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(Xva)).cpu().numpy()
        pte = torch.sigmoid(model(Xte)).cpu().numpy()
    pva_cal, pte_cal = platt_torch(pva, yva.cpu().numpy(), pte)
    thr = best_threshold(yva.cpu().numpy(), pva_cal)
    r = score(y[m_te], pte_cal, thr)
    r["epochs"] = epochs
    r["val_auroc"] = round(float(val_auroc), 4)
    print("  %-20s %2de  %5.0fs  acc %.3f  AUROC %.3f  PR-AUC %.3f  F1 %.3f  ECE %.3f"
          % (model_name, epochs, time.time() - t0, r["accuracy"], r["auroc"],
             r["pr_auc"], r["f1"], r["ece"]))
    return r


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    label = "stockout_within_%d" % HORIZON
    FOLD = int(os.environ.get("BASELINES_FOLD", "0"))

    print("=" * 78)
    print("EXTRA BASELINES -- unified split (series_folds, fold %d) across every model" % FOLD)
    print("=" * 78)

    print("\n[tabular]  116 hand-built features, same ones LR/HistGB/LightGBM get")
    Xv, yt, tm_tr, tm_va, tm_te, tab_ser = build_tabular(load_corpus(), label, fold=FOLD)
    print("  rows: %d train / %d val / %d test\n" % (tm_tr.sum(), tm_va.sum(), tm_te.sum()))

    svm_result = run_svm(Xv, yt, tm_tr, tm_va, tm_te)
    print("  %-24s          acc %.3f  AUROC %.3f  PR-AUC %.3f  F1 %.3f  ECE %.3f"
          % ("SVM (linear)", svm_result["accuracy"], svm_result["auroc"],
             svm_result["pr_auc"], svm_result["f1"], svm_result["ece"]))

    mlp_result = run_mlp_tabular(Xv, yt, tm_tr, tm_va, tm_te)
    print("  %-24s          acc %.3f  AUROC %.3f  PR-AUC %.3f  F1 %.3f  ECE %.3f"
          % ("MLP (tabular)", mlp_result["accuracy"], mlp_result["auroc"],
             mlp_result["pr_auc"], mlp_result["f1"], mlp_result["ece"]))

    print("\n[windowed]  same 22-channel sequence input PatchTST v2 gets")
    df = add_derived(load_corpus())
    X, y, eps, ser = build_windows(df, label)
    m_tr, m_va, m_te = series_folds(ser, fold=FOLD)
    assert not set(ser[m_tr]) & set(ser[m_te]), "series leaked across train/test"
    # This must be the SAME test set the tabular block just used -- both come
    # from series_folds(fold=FOLD) on the same corpus, so the series identity
    # (not just the count) has to match, or the "unified split" claim is false.
    assert set(ser[m_te]) == set(tab_ser[tm_te]), \
        "tabular and windowed test sets disagree -- unified split failed"
    print("  windows: %d train / %d val / %d test  (%d channels, lookback %d)\n"
          % (m_tr.sum(), m_va.sum(), m_te.sum(), len(CHANNELS), LOOKBACK))

    cnn_result = run_temporal("CNN (1D)", CNN1D, X, y, m_tr, m_va, m_te,
                               n_channels=len(CHANNELS))
    lstm_result = run_temporal("LSTM", LSTMClf, X, y, m_tr, m_va, m_te,
                                n_channels=len(CHANNELS))
    gru_result = run_temporal("GRU", GRUClf, X, y, m_tr, m_va, m_te,
                               n_channels=len(CHANNELS))
    txf_result = run_temporal("Transformer (vanilla)", VanillaTransformer, X, y,
                               m_tr, m_va, m_te, n_channels=len(CHANNELS), lookback=LOOKBACK)
    joint_result = run_temporal("JointPatchTransformer", JointPatchTransformer, X, y,
                                 m_tr, m_va, m_te, n_channels=len(CHANNELS), lookback=LOOKBACK)

    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                            "baselines_strong.json"), encoding="utf-8") as fh:
        strong = json.load(fh)
    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                            "patchtst_v2.json"), encoding="utf-8") as fh:
        pt = json.load(fh)
    by = {r["model"]: r for r in strong["results"]}
    pt_runs = pt["runs"]

    def ms(runs, k):
        v = [r[k] for r in runs]
        return float(np.mean(v)), float(np.std(v))

    print("\n" + "=" * 78)
    print("FULL COMPARISON  (fingerprint %s, unified split, fold %d)" % (corpus_fingerprint(), FOLD))
    print("=" * 78)
    print("  %-24s %8s %8s %8s %8s %8s" % ("model", "acc", "AUROC", "PR-AUC", "F1", "ECE"))
    print("  " + "-" * 68)
    rows = [
        ("logistic regression", by["logistic regression"]),
        ("LightGBM", by["LightGBM"]),
        ("SVM (linear)", svm_result),
        ("MLP (tabular)", mlp_result),
        ("CNN (1D)", cnn_result),
        ("LSTM", lstm_result),
        ("GRU", gru_result),
        ("Transformer (vanilla)", txf_result),
        ("JointPatchTransformer", joint_result),
    ]
    for name, r in rows:
        print("  %-24s %8.3f %8.3f %8.3f %8.3f %8.3f" % (
            name, r["accuracy"], r["auroc"], r["pr_auc"], r["f1"], r["ece"]))
    pt_acc, _ = ms(pt_runs, "accuracy"); pt_auroc, _ = ms(pt_runs, "auroc")
    pt_prauc, _ = ms(pt_runs, "pr_auc"); pt_f1, _ = ms(pt_runs, "f1"); pt_ece, _ = ms(pt_runs, "ece")
    print("  %-24s %8.3f %8.3f %8.3f %8.3f %8.3f  (5-seed mean)" % (
        "PatchTST v2", pt_acc, pt_auroc, pt_prauc, pt_f1, pt_ece))
    print("\n  NOTE: logistic regression and LightGBM are re-loaded from")
    print("  baselines_strong.json, which now ALSO uses series_folds(fold=%d) --" % FOLD)
    print("  re-run src/baselines_strong.py first if that file predates this fix.")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "extra_baselines.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "corpus_fingerprint": corpus_fingerprint(),
            "fold": FOLD,
            "note": "All models -- tabular and windowed alike -- now share the "
                    "IDENTICAL series_folds(fold=%d) test set, verified by "
                    "assertion at runtime, not just by construction. SVM/MLP "
                    "are single deterministic fits; CNN/LSTM/GRU/Transformer/"
                    "JointPatchTransformer are single seed, early-stopped "
                    "(patience=8) -- not PatchTST v2's 5-seed full-budget "
                    "protocol. Exploratory, not headline." % FOLD,
            "svm": svm_result, "mlp_tabular": mlp_result,
            "cnn": cnn_result, "lstm": lstm_result, "gru": gru_result,
            "transformer_vanilla": txf_result,
            "joint_patch_transformer": joint_result,
        }, fh, indent=2)
    print("\n  written -> %s" % out)


if __name__ == "__main__":
    main()
