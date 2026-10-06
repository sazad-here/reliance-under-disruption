"""
Reproducible check of the DataCo leakage behind Ahmed et al. (2025), Sci Rep 15:26355.

OVERVIEW.md s6 recorded "LightGBM with all columns 100 %, pre-dispatch columns
77.2 %" without a script. This is that script.

The paper predicts `Late_delivery_risk`. In DataCo that label is a rule on two
columns: actual shipping days > scheduled shipping days. Any column recorded
after delivery (actual shipping days, delivery status, order status, shipping
date) therefore contains the answer.

Three feature sets, same model, same splits (random stratified 80/20, as in the
paper, 5 seeds):
  all_columns        everything except the label and personal/free-text fields
  no_delivery_status all_columns minus `Delivery Status` (leak via actual days only)
  pre_dispatch       only what is known when the order is placed (Li 2026's audit)

Run:  python lightgbm-pivot/ahmed_leakage_check.py
Out:  lightgbm-pivot/ahmed_leakage_results.json
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw", "dataco", "DataCoSupplyChainDataset.csv")
OUT = os.path.join(HERE, "ahmed_leakage_results.json")

LABEL = "Late_delivery_risk"
POST_DISPATCH = ["Days for shipping (real)", "Delivery Status", "Order Status",
                 "shipping date (DateOrders)"]
PERSONAL = ["Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
            "Customer Street", "Product Description", "Product Image", "Customer Id",
            "Order Customer Id", "Order Id", "Order Item Id", "Order Item Cardprod Id",
            "Customer Zipcode", "Order Zipcode"]


def prepare(d, drop):
    X = d.drop(columns=[LABEL] + PERSONAL + drop, errors="ignore").copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    return X


def main():
    d = pd.read_csv(RAW, encoding="latin-1")
    y = d[LABEL].values.astype(int)
    rule = (d["Days for shipping (real)"] > d["Days for shipment (scheduled)"]).astype(int).values
    floor = float(max(y.mean(), 1 - y.mean()))
    print("rows %d | late rate %.3f | majority floor %.3f | rule real>scheduled reproduces label %.2f %%"
          % (len(y), y.mean(), floor, 100 * (rule == y).mean()))
    for c in ("shipping date (DateOrders)", "order date (DateOrders)"):
        d[c] = pd.to_datetime(d[c], format="%m/%d/%Y %H:%M").astype("int64") // 10**9

    sets = {"all_columns": [],
            "no_delivery_status": ["Delivery Status"],
            "pre_dispatch": POST_DISPATCH}
    res = {"rows": int(len(y)), "late_rate": float(y.mean()), "majority_floor": floor,
           "rule_matches_label": float((rule == y).mean()), "feature_sets": {}}
    for name, drop in sets.items():
        X = prepare(d, drop)
        accs, aucs = [], []
        for seed in range(5):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                   random_state=seed, verbose=-1).fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            accs.append(accuracy_score(yte, (p >= 0.5).astype(int))); aucs.append(roc_auc_score(yte, p))
        res["feature_sets"][name] = {"n_features": int(X.shape[1]), "dropped": drop,
                                     "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
                                     "auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs))}
        r = res["feature_sets"][name]
        print("  %-20s %3d features  accuracy %.4f +/- %.4f  AUROC %.4f"
              % (name, r["n_features"], r["accuracy_mean"], r["accuracy_sd"], r["auroc_mean"]))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("written -> %s" % OUT)


if __name__ == "__main__":
    main()
