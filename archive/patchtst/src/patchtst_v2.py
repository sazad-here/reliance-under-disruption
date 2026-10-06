"""
PatchTST v2 -- patching kept, cross-channel reasoning restored.

WHY v1 LOST TO GRADIENT BOOSTING (v1: AUROC 0.809, GB: 0.847)

PatchTST is channel-independent by design: every variable is encoded separately
through shared weights, and they only meet at the final flatten. For long-horizon
forecasting that is the paper's key insight and it works. For OUR target it is a
handicap, because a stockout is an INTERACTION:

        will (inventory + on_order) cover (demand over the lead time)?

That question spans channels. Gradient boosting evaluates all variables jointly at
every split; v1 could not until the very last layer. Four changes address it:

  1. CROSS-CHANNEL FUSION. Keep per-channel patch encoding, then add a second
     transformer that attends ACROSS channel tokens. Channel independence for the
     temporal pattern, joint reasoning for the inventory relationship.

  2. DERIVED CHANNELS. Feed inventory position (I + O - B), net position against
     demand, and cover ratio as explicit inputs. The reference policy is a closed-
     form function of exactly these, so the signal is real domain knowledge, not
     leakage: they use only information the participant also sees.

  3. ATTENTION POOLING + RECENCY. v1 mean-pooled over patches, which treats a
     48-week-old patch like last week's. Replaced with learned attention pooling
     concatenated with the final patch.

  4. HONEST CALIBRATION. v1 used pos_weight in the loss, which inflates predicted
     probabilities -- that is why its ECE was 0.189 against GB's 0.016. Removed,
     and isotonic calibration is fitted on the validation split as plan section 05
     requires. v1 was an uncalibrated model judged against a calibrated one.

Run:  python src/patchtst_v2.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOOKBACK, PATCH_LEN, STRIDE = 48, 8, 4
D_MODEL, N_HEADS, FFN, DROPOUT = 64, 4, 128, 0.2
N_TEMPORAL_LAYERS, N_FUSION_LAYERS = 3, 2
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 128, 60, 8
MC_PASSES, HORIZON = 50, 2
# Uniform average of the last K weight sets (cheap SWA).
AVG_LAST_K = 5
SEEDS = tuple(range(int(os.environ.get("PATCHTST_SEEDS", "5"))))
# Which series fold is held out. Cross-validation sweeps this 0..N_FOLDS-1 so
# every M5 series is tested exactly once.
FOLD = int(os.environ.get("PATCHTST_FOLD", "0"))

RAW_CHANNELS = [
    "demand", "inventory", "on_order", "lead_time", "backorders",
    "supplier_reliability", "transport_ok", "disruption_flag",
]
# Derived channels: the quantities the base-stock rule actually acts on.
# Built only from fields the participant can see, so no leakage.
DERIVED = ["inventory_position", "net_position", "cover_ratio"]
# Rolling statistics, mirroring what build_features gives the tabular models.
ROLLING = [
    "demand_mean4", "demand_std4", "demand_mean12", "demand_std12",
    "inventory_mean4", "inventory_std4", "inventory_mean12", "inventory_std12",
    "backorders_mean4", "backorders_std4", "ip_trend",
]
CHANNELS = RAW_CHANNELS + DERIVED + ROLLING
# Indicator-like channels that must skip RevIN (see class docstring).
FLAG_CHANNEL_IDX = tuple(
    CHANNELS.index(c)
    for c in ("supplier_reliability", "transport_ok", "disruption_flag")
)


class RevIN(nn.Module):
    """Reversible instance norm, with an opt-out for flag-like channels.

    Dividing by sqrt(var + eps) is right for a continuous series and wrong for
    a near-constant indicator: across a 48-week window with no disruption, the
    std collapses, the eps floor dominates, and a single rare 1 is amplified
    into an enormous normalized magnitude. Those channels pass through.
    """

    def __init__(self, n_channels, eps=1e-5, passthrough=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))
        mask = torch.zeros(n_channels, dtype=torch.bool)
        for i in (passthrough or ()):
            mask[i] = True
        self.register_buffer("passthrough", mask)

    def forward(self, x):                        # (B, L, C)
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        out = (x - mean) / std * self.weight + self.bias
        if bool(self.passthrough.any()):
            out = torch.where(self.passthrough, x, out)
        return out


class AttentionPool(nn.Module):
    """Learned pooling: let the model decide which patches matter.

    Mean pooling weights a 48-week-old patch the same as last week's. For a
    2-period-ahead prediction that is plainly wrong.
    """

    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x):                       # (B, P, D)
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)


class PatchTSTv2(nn.Module):
    def __init__(self, n_channels=len(CHANNELS), lookback=LOOKBACK,
                 patch_len=PATCH_LEN, stride=STRIDE):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (lookback - patch_len) // stride + 1

        # supplier_reliability, transport_ok and disruption_flag are indicators,
        # not series: normalising them per-window destroys their scale.
        self.revin = RevIN(n_channels, passthrough=FLAG_CHANNEL_IDX)
        # MLP patch embedding: one linear layer is a weak encoder for a
        # patch that has to carry a shape, not just a level.
        self.embed = nn.Sequential(
            nn.Linear(patch_len, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, D_MODEL) * 0.02)
        self.drop = nn.Dropout(DROPOUT)

        def block(layers):
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN,
                dropout=DROPOUT, batch_first=True, norm_first=True,
                activation="gelu",
            )
            return nn.TransformerEncoder(layer, num_layers=layers)

        # STAGE 1 -- per channel, over patches (PatchTST proper)
        self.temporal = block(N_TEMPORAL_LAYERS)
        self.pool = AttentionPool(D_MODEL)
        self.tnorm = nn.LayerNorm(D_MODEL)

        # STAGE 2 -- across channels. This is the addition.
        self.chan_pos = nn.Parameter(torch.randn(1, n_channels, D_MODEL) * 0.02)
        self.fusion = block(N_FUSION_LAYERS)
        self.cls = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.fnorm = nn.LayerNorm(D_MODEL)

        # pooled channel summary + last-patch recency + fused CLS
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 3, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1),
        )

    def forward(self, x):                        # (B, L, C)
        B, C = x.shape[0], self.n_channels
        x = self.revin(x).permute(0, 2, 1)       # (B, C, L)
        x = x.unfold(-1, self.patch_len, self.stride)     # (B, C, P, patch_len)
        x = x.reshape(B * C, self.n_patches, self.patch_len)

        h = self.drop(self.embed(x) + self.pos)
        h = self.tnorm(self.temporal(h))         # (B*C, P, D)

        pooled = self.pool(h)                    # (B*C, D)  attention over patches
        last = h[:, -1, :]                       # (B*C, D)  most recent patch

        tokens = pooled.reshape(B, C, -1) + self.chan_pos
        tokens = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)
        fused = self.fnorm(self.fusion(tokens))  # (B, 1+C, D)

        feat = torch.cat([
            fused[:, 0, :],                      # CLS: cross-channel summary
            fused[:, 1:, :].mean(dim=1),         # channel-average
            last.reshape(B, C, -1).mean(dim=1),  # recency
        ], dim=-1)
        return self.head(feat).squeeze(-1)


# ---------------------------------------------------------------- data
from baselines_strong import platt  # noqa: E402  -- one implementation, shared


def load_corpus():
    d = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    for n in ("corpus.parquet", "corpus.csv"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return pd.read_parquet(p) if n.endswith("parquet") else pd.read_csv(p)
    raise SystemExit("No corpus. Run: python src/make_corpus.py")


def add_derived(df):
    ip = df["inventory"] + df["on_order"] - df["backorders"]
    df = df.copy()
    df["inventory_position"] = ip
    # demand it must cover over the lead time, and whether it does
    need = df["demand"] * (df["lead_time"] + 1)
    df["net_position"] = ip - need
    df["cover_ratio"] = ip / need.replace(0, np.nan)
    df["cover_ratio"] = df["cover_ratio"].fillna(0).clip(-5, 5)

    # ---- rolling-statistic channels -------------------------------------
    # The tabular baselines receive 116 engineered lag/rolling features built
    # from exactly this raw data; the transformer received 11 raw channels and
    # was expected to rediscover the same structure from 17k training windows.
    # That is the real asymmetry left in the comparison. These channels hand it
    # the same derived signal.
    #
    # Strictly backward-looking and computed per episode, the same discipline
    # train_baselines.build_features uses -- rolling() looks only at the current
    # row and earlier, and groupby("episode") stops any window spanning two
    # warehouses. Computed on the FULL episode before any period filtering, so
    # the long windows are not silently truncated the way the lag48 features
    # were.
    g = df.groupby("episode", sort=False)
    for col, wins in (("demand", (4, 12)), ("inventory", (4, 12)),
                      ("backorders", (4,))):
        for w in wins:
            r = g[col].rolling(w, min_periods=1)
            df["%s_mean%d" % (col, w)] = (
                r.mean().reset_index(level=0, drop=True).astype("float32"))
            df["%s_std%d" % (col, w)] = (
                r.std().reset_index(level=0, drop=True).fillna(0.0).astype("float32"))
    # Trend in the quantity a base-stock rule actually watches.
    df["ip_trend"] = (g["inventory_position"].diff(4)
                      .reset_index(level=0, drop=True).fillna(0.0).astype("float32"))
    return df


def build_windows(df, label_col):
    """Windows plus their episode AND M5 series id.

    The series id is what splits are made on. Splitting on episode index, as
    this function's caller used to, only matched the series split by accident
    of how the corpus happened to be ordered -- on a pooled corpus it puts the
    same demand series in train and test, which is the leakage that cost this
    project every pre-fix model number once already.
    """
    Xs, ys, eps, ser = [], [], [], []
    for ep, g in df.groupby("episode", sort=False):
        g = g.sort_values("period")
        arr = g[CHANNELS].values.astype(np.float32)
        lab = g[label_col].values.astype(np.float32)
        valid = g["label_valid"].values
        sid = int(g["m5_series"].iloc[0])
        for t in range(LOOKBACK, len(g)):
            if valid[t]:
                # INCLUSIVE of period t. The old slice was arr[t - LOOKBACK:t],
                # which stops at t-1 and never shows the model period t's own
                # state -- while the tabular baselines get that same period
                # unlagged (train_baselines.build_features appends df[[col]]
                # alongside lags 1..48). Period t is pre-decision state captured
                # by twin.observe() before twin.step(), and the label looks
                # strictly forward to t+1..t+h, so this is legitimate input, not
                # leakage. Window length is unchanged at LOOKBACK, and the set of
                # sampled t is unchanged, so this is a clean A/B against the
                # stored baseline.
                Xs.append(arr[t - LOOKBACK + 1:t + 1])
                ys.append(lab[t]); eps.append(ep); ser.append(sid)
    return (np.stack(Xs), np.array(ys, np.float32),
            np.array(eps), np.array(ser))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    return float(sum((m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
                     for m in (idx == b for b in range(bins)) if m.sum()))


def best_threshold(y, p):
    """Accuracy at 0.5 is arbitrary for an imbalanced label. Pick it on val."""
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: accuracy_score(y, (p >= t).astype(int))))


def mc_predict(model, X, passes=MC_PASSES, batch=1024):
    model.train()
    outs = []
    with torch.no_grad():
        for _ in range(passes):
            outs.append(np.concatenate([
                torch.sigmoid(model(X[i:i + batch])).cpu().numpy()
                for i in range(0, len(X), batch)
            ]))
    a = np.stack(outs)
    return a.mean(0), a.std(0)


def train_one(seed, Xtr, ytr, Xva, yva):
    torch.manual_seed(seed); np.random.seed(seed)
    # Without these, CUDA attention/matmul kernels are not bit-reproducible
    # even at a fixed seed, so RESULTS.md section 9's regeneration promise
    # would hold only in distribution.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = PatchTSTv2().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    overstepped = False
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR * 3, total_steps=MAX_EPOCHS * max(1, len(Xtr) // BATCH + 1),
    )
    # plain BCE: no pos_weight. Calibration is handled properly, afterwards.
    lossf = nn.BCEWithLogitsLoss()

    # ---- why there is no early stopping here any more ---------------------
    # Measured: OneCycleLR(pct_start=0.3) over 60 epochs peaks at epoch 18.
    # With PATIENCE=8, observed stops of 16/17/35/39/40 imply best epochs of
    # 8/9/27/31/32 -- two of them DURING the warmup ramp, at 44-52% of peak LR,
    # after which the run burned its patience while the LR was still climbing
    # to maximum. At epoch 40 the LR is still 46% of peak, so no run ever saw
    # this model at a settled learning rate: the schedule never completed.
    #
    # Official PatchTST does not early-stop at all on its small datasets
    # (ETTh1/ETTh2/ILI): full budget, then the best-validation checkpoint.
    # Google's tuning playbook calls the same thing retrospective checkpoint
    # selection and notes prospective early stopping is usually unnecessary
    # when the budget is fixed. Prechelt (1998) shows validation curves have
    # many near-equal local minima far apart in epoch, which is exactly the
    # erratic-stopping signature we measured.
    #
    # Selection signal is validation AUROC, not PR-AUC. PR-AUC is a rank
    # statistic whose value depends on prevalence, and our fold base rates
    # range 0.182-0.275, so a checkpoint chosen to maximise validation PR-AUC
    # is chosen under a prevalence that does not hold on test. AUROC is
    # prevalence-invariant. PR-AUC remains the REPORTED metric.
    yva_np = yva.detach().cpu().numpy()
    best, state = -math.inf, None
    trajectory, tail = [], []
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            # bf16 on Ada tensor cores: measured 1.57x, and no GradScaler is
            # needed because bf16 keeps fp32 dynamic range.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                enabled=(DEVICE.type == "cuda")):
                loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # OneCycleLR raises once total_steps is exceeded. Swallowing that
            # silently would leave the LR pinned with nobody the wiser, so
            # absorb only that specific condition and record it.
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            else:
                overstepped = True
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(Xva[i:i + 2048])
                                for i in range(0, len(Xva), 2048)])
            vprob = torch.sigmoid(logits.float()).cpu().numpy()
        v_auroc = float(roc_auc_score(yva_np, vprob))
        v_ap = float(average_precision_score(yva_np, vprob))
        trajectory.append({"epoch": epoch + 1, "val_auroc": v_auroc, "val_ap": v_ap})

        if v_auroc > best + 1e-6:
            best = v_auroc
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Keep the last K weight sets for a uniform average. By then OneCycle
        # has annealed, so this is a cheap SWA: one set of weights and one
        # forward pass, NOT an ensemble, so it needs no ensemble disclosure.
        tail.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        if len(tail) > AVG_LAST_K:
            tail.pop(0)

    # Average the tail, then keep whichever of {best checkpoint, tail average}
    # validation AUROC prefers. That is one binary choice on validation, not a
    # sweep, so it adds negligible selection variance.
    avg = {}
    for k in tail[0]:
        vs = [t[k] for t in tail]
        avg[k] = (torch.stack([v.float() for v in vs]).mean(0).to(vs[0].dtype)
                  if vs[0].is_floating_point() else vs[-1])

    def _val_auroc(sd):
        model.load_state_dict(sd); model.eval()
        with torch.no_grad():
            lg = torch.cat([model(Xva[i:i + 2048]) for i in range(0, len(Xva), 2048)])
        return float(roc_auc_score(yva_np, torch.sigmoid(lg.float()).cpu().numpy()))

    a_best, a_avg = _val_auroc(state), _val_auroc(avg)
    chosen = "avg" if a_avg >= a_best else "best"
    model.load_state_dict(avg if chosen == "avg" else state)
    best_epoch = max(trajectory, key=lambda r: r["val_auroc"])["epoch"]
    return model, MAX_EPOCHS, {"best_epoch": best_epoch, "chosen": chosen,
                               "val_auroc_best": a_best, "val_auroc_avg": a_avg,
                               "trajectory": trajectory, "overstepped": overstepped}


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    label = "stockout_within_%d" % HORIZON
    df = add_derived(load_corpus())

    print("=" * 78)
    print("PATCHTST v2 -- cross-channel fusion, attention pooling, honest calibration")
    print("=" * 78)

    X, y, eps, ser = build_windows(df, label)
    # Split BY SERIES, never by episode index. FOLD selects which series are
    # held out; see baselines_strong.series_folds for the rotation.
    from baselines_strong import series_folds
    m_tr, m_va, m_te = series_folds(ser, FOLD)
    assert not set(ser[m_tr]) & set(ser[m_te]), "series leaked across train/test"
    assert not set(ser[m_va]) & set(ser[m_te]), "series leaked across val/test"

    Xtr, ytr = torch.tensor(X[m_tr]).to(DEVICE), torch.tensor(y[m_tr]).to(DEVICE)
    Xva, yva = torch.tensor(X[m_va]).to(DEVICE), torch.tensor(y[m_va]).to(DEVICE)
    Xte, yte = torch.tensor(X[m_te]).to(DEVICE), y[m_te]
    yva_np = y[m_va]

    print("  channels: %d raw + %d derived + %d rolling = %d"
          % (len(RAW_CHANNELS), len(DERIVED), len(ROLLING), len(CHANNELS)))
    print("  windows:  %d train / %d val / %d test" % (len(Xtr), len(Xva), len(Xte)))
    print("  params:   %d  (v1 had 134,737)\n" % sum(p.numel() for p in PatchTSTv2().parameters()))

    runs = []
    for seed in SEEDS:
        t0 = time.time()
        model, epochs, diag = train_one(seed, Xtr, ytr, Xva, yva)

        pva, _ = mc_predict(model, Xva)
        pte, unc = mc_predict(model, Xte)

        # Platt, not isotonic -- matching the tabular baselines exactly.
        # RESULTS.md leans on "calibration is strictly monotone, so it preserves
        # AUROC and PR-AUC exactly" to justify comparing calibrated numbers.
        # That holds for Platt. Isotonic is only weakly monotone: PAV emits tied
        # probabilities for distinct scores, and ties count as half-concordant,
        # so it can move the very metrics the comparison rests on.
        pva_cal, pte_cal = platt(pva, yva_np, pte)
        thr = best_threshold(yva_np, pva_cal)

        r = {
            "seed": seed, "epochs": epochs, "threshold": round(thr, 3),
            "best_epoch": diag["best_epoch"], "checkpoint": diag["chosen"],
            "val_auroc_best": diag["val_auroc_best"],
            "val_auroc_avg": diag["val_auroc_avg"],
            "trajectory": diag["trajectory"],
            "accuracy": float(accuracy_score(yte, (pte_cal >= thr).astype(int))),
            "auroc": float(roc_auc_score(yte, pte_cal)),
            "pr_auc": float(average_precision_score(yte, pte_cal)),
            "f1": float(f1_score(yte, (pte_cal >= thr).astype(int))),
            # Precision and recall separately, not just their harmonic mean.
            # F1 alone hides WHICH way a model is wrong, and for a stockout
            # warning those two errors have very different costs: a missed
            # stockout (low recall) is a lost sale, a false alarm (low
            # precision) is only wasted inventory.
            "precision": float(precision_score(yte, (pte_cal >= thr).astype(int),
                                               zero_division=0)),
            "recall": float(recall_score(yte, (pte_cal >= thr).astype(int))),
            "brier": float(brier_score_loss(yte, pte_cal)),
            "ece": ece(yte, pte_cal),
            "ece_uncalibrated": ece(yte, pte),
            "mc_spread": float(unc.mean()),
        }
        runs.append(r)
        print("  seed %d  %2de (best@%2d, %s)  %5.0fs  acc %.3f  AUROC %.3f  PR-AUC %.3f  F1 %.3f  ECE %.3f"
              % (seed, epochs, diag["best_epoch"], diag["chosen"], time.time() - t0,
                 r["accuracy"], r["auroc"], r["pr_auc"], r["f1"], r["ece"]))

    print("\n" + "=" * 78)
    print("COMPARISON  (same corpus, same episode-level splits)")
    print("=" * 78)

    def ms(k):
        v = [r[k] for r in runs]
        return np.mean(v), np.std(v)

    # Baselines are LOADED, never hardcoded. Inline literals are how a stale
    # number from an earlier corpus silently produces a false "win" -- which is
    # exactly what this block did before: it printed "beats GB" against LogReg
    # 0.691 while the live baseline was 0.700. Fingerprint-checked, like
    # compare_fair.py does.
    from compare_fair import corpus_fingerprint
    # Prefer the cross-validated baselines when they exist: on a pooled corpus
    # the single-split file is scored on a different design and is not
    # comparable.
    rdir = os.path.join(os.path.dirname(__file__), "..", "results")
    bpath = os.path.join(rdir, "baselines_cv.json")
    if not os.path.exists(bpath):
        bpath = os.path.join(rdir, "baselines_strong.json")
    if not os.path.exists(bpath):
        raise SystemExit("run src/baselines_strong.py first -- no baselines to compare against")
    with open(bpath, encoding="utf-8") as fh:
        bl = json.load(fh)
    if bl.get("corpus_fingerprint") != corpus_fingerprint():
        raise SystemExit(
            "baselines_strong.json was built on corpus %s but this is %s -- "
            "re-run src/baselines_strong.py" % (bl.get("corpus_fingerprint"),
                                                corpus_fingerprint()))
    by = {r["model"]: r for r in bl["results"]}
    v1 = by["logistic regression"]
    gb = by.get("LightGBM") or by["sklearn HistGB"]

    print("  %-12s %16s %12s %12s" % ("metric", "PatchTST v2", "LogReg", "GradBoost"))
    print("  " + "-" * 60)
    for k in ("accuracy", "auroc", "pr_auc", "ece"):
        m, s = ms(k)
        flag = ""
        best_base = max(v1[k], gb[k]) if k != "ece" else min(v1[k], gb[k])
        if k != "ece" and m > best_base:
            flag = "  <-- beats BOTH baselines"
        if k == "ece" and m < best_base:
            flag = "  <-- beats BOTH baselines"
        print("  %-12s %9.3f+/-%.3f %12.3f %12.3f%s" % (k, m, s, v1[k], gb[k], flag))

    print("\n  majority-class accuracy: %.3f" % max(yte.mean(), 1 - yte.mean()))
    print("  MC-dropout spread:       %.3f" % ms("mc_spread")[0])

    from compare_fair import corpus_fingerprint
    payload = {"fold": FOLD, "runs": runs, "baseline_gb": gb, "baseline_lr": v1,
               "corpus_fingerprint": corpus_fingerprint()}

    rdir = os.path.join(os.path.dirname(__file__), "..", "results")
    out = os.path.join(rdir, "patchtst_v2_fold%d.json" % FOLD)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\n  written -> %s" % out)

    # FOLD 0 is the default single-run comparison (compare_fair.py and
    # RESULTS.md's regeneration instructions both read the un-suffixed name,
    # not a fold-specific one). Without this, running `python
    # src/patchtst_v2.py` with no PATCHTST_FOLD set silently stops updating
    # results/patchtst_v2.json the moment fold support is added -- which is
    # exactly what happened: this file went stale for a full day while every
    # consumer kept reading it unaware nothing was writing to it any more.
    if FOLD == 0:
        main_out = os.path.join(rdir, "patchtst_v2.json")
        with open(main_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("  written -> %s  (default fold, kept in sync)" % main_out)


if __name__ == "__main__":
    main()
