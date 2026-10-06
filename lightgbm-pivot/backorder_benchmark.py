"""
Backorder benchmark -- reproduce Ntakolia et al. (2021), then test hybrids.

ANCHOR PAPER
  Ntakolia, Kokkotis, Karlsson & Moustakidis (2021). An Explainable Machine
  Learning Model for Material Backorder Prediction in Inventory Management.
  Sensors 21(23), 7926. doi:10.3390/s21237926 (open access, PMC8659943).
  Reported (Table 3, balanced test set): LightGBM acc 87.78 %, best model
  (Balanced Bagging) acc 88.85 %; RF/XGB/LGBM/BB AUC 0.95.

THEIR PROTOCOL, AS RECONSTRUCTED
  Kaggle "Can you predict product backorders?" train + test files combined,
  majority class randomly undersampled to 1:1, then an 80/20 split. Their
  confusion matrices sum to 5,588 test rows; combined positives are 13,981,
  so a balanced 27,962-row set gives a 5,592-row 20 % split -- consistent.
  The paper does not state a seed, so we report the mean and s.d. over
  SEEDS, never a single favourable draw.

VARIANTS (all scored on the identical balanced test set per seed)
  A  LightGBM, raw features, the paper's selected hyper-parameters
  B  A + inventory-position features (the same quantity that alone matched
     PatchTST on the simulator corpus -- see RESULTS.md)
  C  B, stacked with logistic regression (out-of-fold meta-learner)
  D  C, but each base model trained as an EasyEnsemble over K balanced bags,
     so the majority rows the paper discarded are used for TRAINING only.
     The test set is untouched and identical to A-C.
  E  ABLATION: LightGBM alone as an EasyEnsemble (no LR, no stacking), to
     separate what the extra training rows buy from what stacking buys.

Run:  python lightgbm-pivot/backorder_benchmark.py
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw", "backorder")
OUT = os.path.join(HERE, "backorder_benchmark.json")
SEEDS = tuple(range(int(os.environ.get("BO_SEEDS", "5"))))
K_BAGS = 5
PAPER_LGBM = dict(n_estimators=520, num_leaves=50)   # paper Table 3
REPORTED = {"LightGBM (paper)": 0.8778, "Balanced Bagging (paper best)": 0.8885}


def load():
    df = pd.concat([pd.read_csv(os.path.join(RAW, f), low_memory=False)
                    for f in ("Kaggle_Training_Dataset_v2.csv",
                              "Kaggle_Test_Dataset_v2.csv")], ignore_index=True)
    df = df.dropna(subset=["went_on_backorder"])          # trailing summary rows
    for c in df.columns:
        if df[c].dtype == object and c != "sku":
            df[c] = (df[c] == "Yes").astype(np.int8)
    df["perf_6_month_avg"] = df["perf_6_month_avg"].replace(-99, np.nan)
    df["perf_12_month_avg"] = df["perf_12_month_avg"].replace(-99, np.nan)
    return df.drop(columns=["sku"])


def add_position(X):
    """Inventory-position features: what is on hand + coming vs. what is due."""
    X = X.copy()
    pos = X["national_inv"] + X["in_transit_qty"]
    X["inv_position"] = pos
    X["net_vs_fc3"] = pos - X["forecast_3_month"]
    X["net_vs_sales1"] = pos - X["sales_1_month"]
    X["cover_fc3"] = pos / (X["forecast_3_month"] + 1)
    X["cover_sales3"] = pos / (X["sales_3_month"] / 3 + 1)
    X["below_min_bank"] = (X["national_inv"] < X["min_bank"]).astype(np.int8)
    X["past_due_ratio"] = X["pieces_past_due"] / (X["sales_1_month"] + 1)
    return X


def lgbm(seed):
    return lgb.LGBMClassifier(**PAPER_LGBM, random_state=seed, verbose=-1)


def logreg():
    return make_pipeline(QuantileTransformer(output_distribution="normal",
                                             n_quantiles=1000, subsample=200_000),
                         LogisticRegression(max_iter=3000, C=1.0))


def fit_proba(make, Xtr, ytr, Xte):
    m = make().fit(Xtr.fillna(Xtr.median()) if make is logreg else Xtr, ytr)
    Xq = Xte.fillna(Xtr.median()) if make is logreg else Xte
    return m.predict_proba(Xq)[:, 1]


def easy_lgbm(Xtr, ytr, Xte, seed, bags):
    """EasyEnsemble of LightGBM only: the same bags as D, no meta-learner."""
    pos = ytr == 1
    ps = [fit_proba(lambda: lgbm(seed), Xtr, ytr, Xte)]
    for Nb in bags:
        Xbag = pd.concat([Xtr[pos], Nb])
        ybag = np.r_[np.ones(int(pos.sum())), np.zeros(len(Nb))]
        ps.append(fit_proba(lambda: lgbm(seed), Xbag, ybag, Xte))
    return np.mean(ps, axis=0)


def stack(Xtr, ytr, Xte, seed, bags=None):
    """OOF-stacked LightGBM + LR, meta = LR on the two logits.

    If `bags` is given (list of extra negative frames), each base model is an
    EasyEnsemble: trained on the balanced train set with its negatives swapped
    for a fresh draw per bag, probabilities averaged.
    """
    makes = {"gbm": lambda: lgbm(seed), "lr": logreg}

    def base(name, Xa, ya, Xb):
        mk = makes[name]
        if name == "lr":
            return fit_proba(logreg, Xa, ya, Xb)
        return fit_proba(mk, Xa, ya, Xb)

    def ens(name, Xa, ya, Xb):
        if not bags:
            return base(name, Xa, ya, Xb)
        pos = ya == 1
        ps = []
        for Nb in bags:
            Xbag = pd.concat([Xa[pos], Nb.iloc[:int(pos.sum())]])
            ybag = np.r_[np.ones(int(pos.sum())), np.zeros(min(len(Nb), int(pos.sum())))]
            ps.append(base(name, Xbag, ybag, Xb))
        ps.append(base(name, Xa, ya, Xb))
        return np.mean(ps, axis=0)

    oof = np.zeros((len(Xtr), 2))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr):
        for j, n in enumerate(("gbm", "lr")):
            oof[va, j] = base(n, Xtr.iloc[tr], ytr[tr], Xtr.iloc[va])
    test = np.column_stack([ens(n, Xtr, ytr, Xte) for n in ("gbm", "lr")])
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    meta = LogisticRegression().fit(lg(oof), ytr)
    return meta.predict_proba(lg(test))[:, 1]


def score(y, p):
    yh = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, yh), "auroc": roc_auc_score(y, p),
            "f1": f1_score(y, yh), "precision": precision_score(y, yh),
            "recall": recall_score(y, yh)}


def main():
    df = load()
    y_all = df.pop("went_on_backorder").astype(int).values
    X_all = add_position(df)
    raw_cols = list(df.columns)
    print("rows %d  positives %d (%.2f %%)" % (len(y_all), y_all.sum(), 100 * y_all.mean()))

    results = {k: [] for k in ("A_lgbm_paper", "B_lgbm_position",
                               "C_stack_lr_lgbm", "D_stack_easyensemble",
                               "E_lgbm_easyensemble")}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pos_idx = np.flatnonzero(y_all == 1)
        neg_idx = np.flatnonzero(y_all == 0)
        neg_pick = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        bal = np.r_[pos_idx, neg_pick]
        tr_i, te_i = train_test_split(bal, test_size=0.2, random_state=seed,
                                      stratify=y_all[bal])
        ytr, yte = y_all[tr_i], y_all[te_i]
        Xtr, Xte = X_all.iloc[tr_i], X_all.iloc[te_i]

        # negatives never used by the paper's subset -> training-only bags
        spare = np.setdiff1d(neg_idx, neg_pick)
        rng.shuffle(spare)
        n = int((ytr == 1).sum())
        bags = [X_all.iloc[spare[k * n:(k + 1) * n]] for k in range(K_BAGS)]
        assert not set(te_i) & set(np.concatenate([spare[:K_BAGS * n], tr_i]))

        r = {
            "A_lgbm_paper": score(yte, fit_proba(lambda: lgbm(seed), Xtr[raw_cols], ytr, Xte[raw_cols])),
            "B_lgbm_position": score(yte, fit_proba(lambda: lgbm(seed), Xtr, ytr, Xte)),
            "C_stack_lr_lgbm": score(yte, stack(Xtr, ytr, Xte, seed)),
            "D_stack_easyensemble": score(yte, stack(Xtr, ytr, Xte, seed, bags)),
            "E_lgbm_easyensemble": score(yte, easy_lgbm(Xtr, ytr, Xte, seed, bags)),
        }
        for k, v in r.items():
            results[k].append(v)
        print("seed %d  test n=%d  " % (seed, len(yte)) +
              "  ".join("%s %.4f/%.4f" % (k[0], v["accuracy"], v["auroc"]) for k, v in r.items()))

    summary = {}
    print("\n%-24s %16s %16s" % ("variant", "accuracy", "AUROC"))
    for k, runs in results.items():
        s = {m: (float(np.mean([r[m] for r in runs])), float(np.std([r[m] for r in runs])))
             for m in runs[0]}
        summary[k] = s
        print("%-24s %.4f +/- %.4f  %.4f +/- %.4f" % (k, *s["accuracy"], *s["auroc"]))
    for k, v in REPORTED.items():
        print("%-24s %.4f  (reported)" % (k[:24], v))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"anchor": "Ntakolia et al. 2021, Sensors 21(23):7926",
                   "reported_accuracy": REPORTED, "seeds": list(SEEDS),
                   "summary": summary, "runs": results}, fh, indent=2)
    print("\nwritten -> %s" % OUT)


if __name__ == "__main__":
    main()
