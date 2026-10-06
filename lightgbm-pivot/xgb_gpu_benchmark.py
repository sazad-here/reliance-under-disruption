"""
XGBoost on CUDA, on the SAME Kaggle backorder splits as backorder_benchmark.py.

Why: simulator_training.py found XGBoost (CUDA) the best-calibrated model on the
simulator. Recommending it needs the same external check LightGBM passed
against Ntakolia et al. (2021) -- identical seeds, undersample and 80/20 split.

Run:  python lightgbm-pivot/xgb_gpu_benchmark.py
"""
import json
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backorder_benchmark import SEEDS, K_BAGS, add_position, load, score  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb_gpu_benchmark.json")


def model(seed):
    # The paper's selected XGBoost settings (Table 3): max_depth 9, min_child_weight 1, gamma 0.7.
    return xgb.XGBClassifier(device="cuda", tree_method="hist", max_depth=9,
                             min_child_weight=1, gamma=0.7, random_state=seed)


def main():
    df = load()
    y_all = df.pop("went_on_backorder").astype(int).values
    X_all = add_position(df).astype("float32")
    raw = list(df.columns)
    res = {"F_xgb_paper_cuda": [], "G_xgb_easyensemble_cuda": []}
    for seed in SEEDS:
        # identical sampling to backorder_benchmark.main()
        rng = np.random.default_rng(seed)
        pos_idx = np.flatnonzero(y_all == 1)
        neg_idx = np.flatnonzero(y_all == 0)
        neg_pick = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        bal = np.r_[pos_idx, neg_pick]
        tr_i, te_i = train_test_split(bal, test_size=0.2, random_state=seed, stratify=y_all[bal])
        ytr, yte = y_all[tr_i], y_all[te_i]
        spare = np.setdiff1d(neg_idx, neg_pick); rng.shuffle(spare)
        n = int((ytr == 1).sum())
        assert not set(te_i) & set(np.concatenate([spare[:K_BAGS * n], tr_i]))

        Xtr, Xte = X_all.iloc[tr_i], X_all.iloc[te_i]
        pF = model(seed).fit(Xtr[raw], ytr).predict_proba(Xte[raw])[:, 1]
        ps = [model(seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]]
        pos = ytr == 1
        for k in range(K_BAGS):
            Nb = X_all.iloc[spare[k * n:(k + 1) * n]]
            Xb = np.vstack([Xtr[pos].values, Nb.values])
            yb = np.r_[np.ones(int(pos.sum())), np.zeros(len(Nb))]
            ps.append(model(seed).fit(Xb, yb).predict_proba(Xte.values)[:, 1])
        r = {"F_xgb_paper_cuda": score(yte, pF), "G_xgb_easyensemble_cuda": score(yte, np.mean(ps, 0))}
        for k, v in r.items():
            res[k].append(v)
        print("seed %d  F %.4f/%.4f  G %.4f/%.4f" % (seed, r["F_xgb_paper_cuda"]["accuracy"],
              r["F_xgb_paper_cuda"]["auroc"], r["G_xgb_easyensemble_cuda"]["accuracy"],
              r["G_xgb_easyensemble_cuda"]["auroc"]))
    summ = {k: {m: [float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v]))]
                for m in v[0]} for k, v in res.items()}
    for k, s in summ.items():
        print("%-26s acc %.4f +/- %.4f  AUROC %.4f +/- %.4f" % (k, *s["accuracy"], *s["auroc"]))
    print("XGBoost (paper, reported)  acc 0.8853")
    json.dump({"reported_xgb_accuracy": 0.8853, "summary": summ, "runs": res},
              open(OUT, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
