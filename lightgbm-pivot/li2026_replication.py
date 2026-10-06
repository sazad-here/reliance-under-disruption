"""
DataCo + Olist: replicate Li (2026), then swap in LightGBM.

ANCHOR
  Li, J. (2026). When Does Machine Learning Beat Value Sorting? A Three-Dataset
  Diagnostic of Exposure-Weighted Shipment Prioritization. Proc. AIIIP 2026.
  arXiv:2607.18573 (papers/12_...). Leakage-controlled: only features known at
  purchase, five rolling-origin folds over six chronological blocks.
  Published Capture@10% (Table 4)          DataCo   Olist
      RANDOM                               .100     .097
      VALUE_ONLY                           .222     .464
      PROB_ONLY                            .108     .150
      M0  (severity only)                  .212     .144
      BINARY x VALUE                       .287     .430
      M1  (severity x value)               .323     .415
  Table 2: DataCo rho .386, R2 .270; Olist rho .075, R2 -.019.

DEFINITIONS (paper s3)
  severity d = max(actual - scheduled, 0) days; exposure e = d * value;
  Capture@k = exposure in the top k % by score / total exposure.

WHAT THE PAPER LEAVES UNSTATED, AND HOW IT IS PINNED DOWN HERE
  * DataCo's value column and the block construction. VALUE_ONLY involves no
    model, so each candidate is scored and the one reproducing the published
    VALUE_ONLY capture is used. This is a protocol identification step, not a
    tuning step: it cannot move any model's score.
  * Categorical encoding: frequency/ordinal codes fitted inside each training
    fold (the paper one-hot encodes DataCo). Both learners get the IDENTICAL
    matrix, so any difference between them is the learner, nothing else.

LEARNERS
  XGB   the paper's settings: 180 trees, depth 4, lr .05, subsample .85,
        colsample .85, L1 0, L2 1 -- run on CUDA.
  LGBM  LightGBM, 1,000 trees max, lr .03, 63 leaves; rounds chosen by early
        stopping on the LAST block of each training window (never the test).

Run:  python lightgbm-pivot/li2026_replication.py
"""
from __future__ import annotations

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "li2026_results.json")
K, N_BLOCKS, N_BOOT = 0.10, 6, 1000
PUBLISHED = {
    "dataco": {"RANDOM": .100, "VALUE_ONLY": .222, "PROB_ONLY": .108, "M0": .212,
               "BINARYxVALUE": .287, "M1": .323, "rho": .386, "r2": .270},
    "olist": {"RANDOM": .097, "VALUE_ONLY": .464, "PROB_ONLY": .150, "M0": .144,
              "BINARYxVALUE": .430, "M1": .415, "rho": .075, "r2": -.019},
}


# ------------------------------------------------------------------ data
def load_dataco(value_col):
    d = pd.read_csv(os.path.join(RAW, "dataco", "DataCoSupplyChainDataset.csv"), encoding="latin-1")
    t = pd.to_datetime(d["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
    sev = (d["Days for shipping (real)"] - d["Days for shipment (scheduled)"]).clip(lower=0).astype(float)
    value = d[value_col].astype(float)
    leak = ["Days for shipping (real)", "Delivery Status", "Late_delivery_risk", "Order Status",
            "shipping date (DateOrders)", "order date (DateOrders)"]
    pii_ids = ["Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
               "Customer Street", "Product Description", "Product Image", "Customer Id",
               "Order Customer Id", "Order Id", "Order Item Id", "Order Item Cardprod Id",
               "Customer Zipcode", "Order Zipcode"]
    X = d.drop(columns=leak + pii_ids)
    X["month"], X["dow"], X["hour"] = t.dt.month, t.dt.dayofweek, t.dt.hour
    return X, sev.values, value.values, t.values


def load_olist():
    o = pd.read_csv(os.path.join(RAW, "olist", "olist_orders_dataset.csv"),
                    parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                 "order_estimated_delivery_date"])
    o = o[(o.order_status == "delivered") & o.order_delivered_customer_date.notna()
          & o.order_estimated_delivery_date.notna()].copy()
    sev = ((o.order_delivered_customer_date - o.order_estimated_delivery_date)
           .dt.total_seconds() / 86400).clip(lower=0)
    it = pd.read_csv(os.path.join(RAW, "olist", "olist_order_items_dataset.csv"))
    pr = pd.read_csv(os.path.join(RAW, "olist", "olist_products_dataset.csv"))
    se = pd.read_csv(os.path.join(RAW, "olist", "olist_sellers_dataset.csv"))
    cu = pd.read_csv(os.path.join(RAW, "olist", "olist_customers_dataset.csv"))
    pa = pd.read_csv(os.path.join(RAW, "olist", "olist_order_payments_dataset.csv"))
    geo = (pd.read_csv(os.path.join(RAW, "olist", "olist_geolocation_dataset.csv"))
           .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean())
    it = it.merge(pr, on="product_id", how="left").merge(se, on="seller_id", how="left")
    agg = it.groupby("order_id").agg(
        value=("price", "sum"), freight=("freight_value", "sum"), n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"), seller_id=("seller_id", "first"),
        seller_state=("seller_state", "first"), seller_zip=("seller_zip_code_prefix", "first"),
        category=("product_category_name", "first"), weight=("product_weight_g", "mean"),
        length=("product_length_cm", "mean"), height=("product_height_cm", "mean"),
        width=("product_width_cm", "mean"), photos=("product_photos_qty", "mean"),
        name_len=("product_name_lenght", "mean"), desc_len=("product_description_lenght", "mean"))
    pay = pa.groupby("order_id").agg(pay_type=("payment_type", "first"),
                                     installments=("payment_installments", "max"),
                                     pay_value=("payment_value", "sum"), n_pay=("payment_sequential", "count"))
    o = o.join(agg, on="order_id").join(pay, on="order_id").merge(cu, on="customer_id", how="left")
    for side, z in (("s", "seller_zip"), ("c", "customer_zip_code_prefix")):
        o = o.join(geo.rename(columns=lambda c: side + "_" + c[12:]), on=z)
    t = o["order_purchase_timestamp"]
    X = pd.DataFrame({
        "value": o["value"], "freight": o["freight"], "n_items": o["n_items"],
        "n_sellers": o["n_sellers"], "weight": o["weight"], "length": o["length"],
        "height": o["height"], "width": o["width"], "photos": o["photos"],
        "name_len": o["name_len"], "desc_len": o["desc_len"],
        "installments": o["installments"], "pay_value": o["pay_value"], "n_pay": o["n_pay"],
        "pay_type": o["pay_type"], "seller_id": o["seller_id"], "seller_state": o["seller_state"],
        "category": o["category"], "customer_state": o["customer_state"],
        "s_lat": o["s_lat"], "s_lng": o["s_lng"], "c_lat": o["c_lat"], "c_lng": o["c_lng"],
        "est_interval": (o["order_estimated_delivery_date"] - t).dt.total_seconds() / 86400,
        "month": t.dt.month, "dow": t.dt.dayofweek, "hour": t.dt.hour, "doy": t.dt.dayofyear,
    })
    X["dist"] = np.hypot(X.s_lat - X.c_lat, (X.s_lng - X.c_lng) * np.cos(np.radians(X.c_lat)))
    return X.reset_index(drop=True), sev.values, o["value"].fillna(0).values, t.values


def encode(Xtr, Xte):
    """Fit encodings on the training fold only; frequency codes for strings."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in Xtr.columns:
        if Xtr[c].dtype == object:
            f = Xtr[c].value_counts(normalize=True)
            codes = {k: i for i, k in enumerate(f.index)}
            Xte[c + "_freq"] = Xte[c].map(f).fillna(0).astype(float)
            Xtr[c + "_freq"] = Xtr[c].map(f).astype(float)
            Xtr[c] = Xtr[c].map(codes).fillna(-1).astype(float)
            Xte[c] = Xte[c].map(codes).fillna(-1).astype(float)
    return Xtr.astype(np.float32), Xte.astype(np.float32)


# ------------------------------------------------------------------ learners
def xgb_model(kind):
    cls = xgb.XGBRegressor if kind == "reg" else xgb.XGBClassifier
    return cls(n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85,
               colsample_bytree=0.85, reg_alpha=0, reg_lambda=1, device="cuda",
               tree_method="hist", random_state=42)


def lgbm_fit(kind, Xtr, ytr, Xva, yva):
    cls = lgb.LGBMRegressor if kind == "reg" else lgb.LGBMClassifier
    m = cls(n_estimators=1000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=42, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def capture(score, exposure, k=K):
    n = max(1, int(round(k * len(score))))
    top = np.argsort(-score, kind="stable")[:n]
    return float(exposure[top].sum() / exposure.sum())


def blocks(t, how):
    order = np.argsort(t, kind="stable")
    b = np.empty(len(t), int)
    if how == "count":
        b[order] = np.arange(len(t)) * N_BLOCKS // len(t)
    else:
        tt = t.astype("datetime64[s]").astype(np.int64)
        edges = np.linspace(tt.min(), tt.max(), N_BLOCKS + 1)
        b = np.clip(np.searchsorted(edges, tt, side="right") - 1, 0, N_BLOCKS - 1)
    return b


def value_only_capture(sev, value, b):
    te = b >= 1
    return capture(value[te], (sev * value)[te])


# ------------------------------------------------------------------ run
def run(name, X, sev, value, t, log_target):
    b = blocks(t, BLOCK_HOW[name])
    te_all = b >= 1
    y_reg = np.log1p(sev) if log_target else sev
    y_bin = (sev > 0).astype(int)
    S = {k: np.full(len(sev), np.nan) for k in
         ("xgb_sev", "xgb_prob", "lgbm_sev", "lgbm_prob")}
    for f in range(1, N_BLOCKS):
        tr, te = b < f, b == f
        Xtr, Xte = encode(X[tr], X[te])
        m = xgb_model("reg").fit(Xtr, y_reg[tr]); S["xgb_sev"][te] = m.predict(Xte)
        m = xgb_model("clf").fit(Xtr, y_bin[tr]); S["xgb_prob"][te] = m.predict_proba(Xte)[:, 1]
        # LightGBM early-stops on the last block of the training window
        inner = b[tr] == f - 1
        if f == 1:                           # only one training block: hold out its last 20 %
            idx = np.flatnonzero(tr)
            inner = np.zeros(tr.sum(), bool)
            inner[np.argsort(t[idx])[int(0.8 * len(idx)):]] = True
        Xa, Xv = Xtr[~inner], Xtr[inner]
        m = lgbm_fit("reg", Xa, y_reg[tr][~inner], Xv, y_reg[tr][inner]); S["lgbm_sev"][te] = m.predict(Xte)
        m = lgbm_fit("clf", Xa, y_bin[tr][~inner], Xv, y_bin[tr][inner]); S["lgbm_prob"][te] = m.predict_proba(Xte)[:, 1]
        print("  %s fold %d  train %6d  test %6d" % (name, f, tr.sum(), te.sum()), flush=True)

    sv, vv, ex = sev[te_all], value[te_all], (sev * value)[te_all]
    inv = (lambda p: np.expm1(np.clip(p, 0, None))) if log_target else (lambda p: np.clip(p, 0, None))
    sc = {"RANDOM": np.random.default_rng(42).random(te_all.sum()), "VALUE_ONLY": vv}
    for L in ("xgb", "lgbm"):
        d_hat, p_hat = inv(S[L + "_sev"][te_all]), S[L + "_prob"][te_all]
        sc[L + "_PROB_ONLY"] = p_hat
        sc[L + "_M0"] = d_hat
        sc[L + "_BINARYxVALUE"] = p_hat * vv
        sc[L + "_M1"] = d_hat * vv
    cap = {k: capture(s, ex) for k, s in sc.items()}
    diag = {}
    for L in ("xgb", "lgbm"):
        d_hat = inv(S[L + "_sev"][te_all])
        diag[L] = {"rho": float(spearmanr(d_hat, sv).correlation),
                   "r2": float(r2_score(np.log1p(sv) if log_target else sv, S[L + "_sev"][te_all])),
                   "auroc_late": float(roc_auc_score((sv > 0).astype(int), S[L + "_prob"][te_all]))}

    # paired bootstrap (random_state 42, as in the paper)
    rng = np.random.default_rng(42)
    pairs = [("lgbm_M1", "VALUE_ONLY"), ("xgb_M1", "VALUE_ONLY"), ("lgbm_M1", "xgb_M1"),
             ("lgbm_BINARYxVALUE", "VALUE_ONLY")]
    boot = {f"{a} - {c}": [] for a, c in pairs}
    n = len(ex)
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        for a, c in pairs:
            boot[f"{a} - {c}"].append(capture(sc[a][ix], ex[ix]) - capture(sc[c][ix], ex[ix]))
    ci = {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for k, v in boot.items()}
    return cap, diag, ci


BLOCK_HOW = {}


def main():
    t0 = time.time()
    out = {"published": PUBLISHED}

    # ---- identify unstated protocol choices from the model-free VALUE_ONLY ----
    print("protocol identification (VALUE_ONLY needs no model):")
    best = None
    for col in ("Sales", "Order Item Total", "Order Item Product Price", "Sales per customer"):
        X, sev, val, t = load_dataco(col)
        for how in ("count", "time"):
            v = value_only_capture(sev, val, blocks(t, how))
            print("  DataCo value=%-26s blocks=%-5s VALUE_ONLY %.4f (published .222)" % (col, how, v))
            if best is None or abs(v - .222) < best[0]:
                best = (abs(v - .222), col, how)
    dc_col, BLOCK_HOW["dataco"] = best[1], best[2]
    Xo, so, vo, to = load_olist()
    bo = None
    for how in ("count", "time"):
        v = value_only_capture(so, vo, blocks(to, how))
        print("  Olist  blocks=%-5s VALUE_ONLY %.4f (published .464)" % (how, v))
        if bo is None or abs(v - .464) < bo[0]:
            bo = (abs(v - .464), how)
    BLOCK_HOW["olist"] = bo[1]
    print("  -> DataCo value '%s', blocks by %s; Olist blocks by %s\n"
          % (dc_col, BLOCK_HOW["dataco"], BLOCK_HOW["olist"]))
    out["protocol"] = {"dataco_value": dc_col, "dataco_blocks": BLOCK_HOW["dataco"],
                       "olist_blocks": BLOCK_HOW["olist"]}

    for name, (X, sev, val, t), logt in (("dataco", load_dataco(dc_col), False),
                                         ("olist", (Xo, so, vo, to), True)):
        cap, diag, ci = run(name, X, sev, val, t, logt)
        P = PUBLISHED[name]
        print("\n%s  Capture@10%%          published   XGB(ours)   LightGBM" % name.upper())
        for r in ("PROB_ONLY", "M0", "BINARYxVALUE", "M1"):
            print("  %-18s %10.3f %11.3f %10.3f" % (r, P[r], cap["xgb_" + r], cap["lgbm_" + r]))
        for r in ("RANDOM", "VALUE_ONLY"):
            print("  %-18s %10.3f %11.3f %10s" % (r, P[r], cap[r], "-"))
        print("  severity rho       %10.3f %11.3f %10.3f" % (P["rho"], diag["xgb"]["rho"], diag["lgbm"]["rho"]))
        print("  severity R2        %10.3f %11.3f %10.3f" % (P["r2"], diag["xgb"]["r2"], diag["lgbm"]["r2"]))
        print("  late AUROC         %10s %11.3f %10.3f" % ("-", diag["xgb"]["auroc_late"], diag["lgbm"]["auroc_late"]))
        for k, (m, lo, hi) in ci.items():
            print("  %-34s %+.3f  [%+.3f, %+.3f]%s" % (k, m, lo, hi, "  CLEARS" if lo > 0 else ""))
        out[name] = {"capture": cap, "diagnostics": diag, "paired_ci": ci}

    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print("\nwritten -> %s" % OUT)


if __name__ == "__main__":
    main()
