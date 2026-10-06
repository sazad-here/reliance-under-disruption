"""
LightGBM on Walmart M5, scored exactly like the M5 Accuracy competition.

ANCHOR
  Makridakis, Spiliotis & Assimakopoulos (2022). M5 accuracy competition:
  Results, findings, and conclusions. International Journal of Forecasting
  38(4), 1346-1364. Official private-leaderboard scores are in
  papers/M5_Scores_and_Ranks.xlsx: winner YJ_STU 0.5204, 50th place 0.5760,
  best statistical benchmark ES_bu 0.6710, sNaive 0.8470.

PROTOCOL (the competition's evaluation phase)
  Train on d_1..d_1941, forecast d_1942..d_1969 (28 days), score with WRMSSE
  over all 42,840 series at 12 aggregation levels, using the organisers'
  own weights_evaluation.csv and hidden test file sales_test_evaluation.csv
  (both from the organisers' Drive, data/raw/m5_official/).

SELF-CHECK BEFORE TRAINING
  The scorer must reproduce two published numbers from the official files:
  ES_bu's released forecasts -> 0.670981, and sNaive -> 0.847017. If either
  misses, the script stops: a comparison against the leaderboard is only
  valid if our WRMSSE is the leaderboard's WRMSSE.

MODEL
  One LightGBM per store (10), Tweedie loss, direct 28-day horizon: every
  demand feature is lagged >= 28 days, so one model forecasts all 28 days
  without feeding its own predictions back in. Rounds are chosen by early
  stopping on d_1914..d_1941, then each model is refitted on data through
  d_1941 with that round count. The test window is never used for any choice.

Run:  python lightgbm-pivot/m5_lightgbm.py
"""
from __future__ import annotations

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M5 = os.path.join(HERE, "..", "data", "raw", "m5")
OFF = os.path.join(HERE, "..", "data", "raw", "m5_official")
OUT = os.path.join(HERE, "m5_results.json")

H, LAST_TRAIN = 28, 1941
START = int(os.environ.get("M5_START", "1100"))     # first training day kept
PUBLISHED = {"YJ_STU (1st)": 0.520438, "Matthias (2nd)": 0.528165,
             "mf (3rd)": 0.535717, "10th place": 0.547471, "20th place": 0.556906,
             "50th place": 0.575994, "ES_bu (best benchmark)": 0.670981,
             "sNaive": 0.847017}
ID = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
LEVELS = {  # level -> (keys in sales table, keys as they appear in weights file)
    "Level1": ([], []), "Level2": (["state_id"], ["state_id"]),
    "Level3": (["store_id"], ["store_id"]), "Level4": (["cat_id"], ["cat_id"]),
    "Level5": (["dept_id"], ["dept_id"]),
    "Level6": (["state_id", "cat_id"], ["state_id", "cat_id"]),
    "Level7": (["state_id", "dept_id"], ["state_id", "dept_id"]),
    "Level8": (["store_id", "cat_id"], ["store_id", "cat_id"]),
    "Level9": (["store_id", "dept_id"], ["store_id", "dept_id"]),
    "Level10": (["item_id"], ["item_id"]),
    "Level11": (["state_id", "item_id"], ["state_id", "item_id"]),
    "Level12": (["item_id", "store_id"], ["item_id", "store_id"]),
}


# ------------------------------------------------------------------ scoring
class WRMSSE:
    """The organisers' Estimate WRMSSE.R, vectorised."""

    def __init__(self, train, test, weights):
        self.ids = train[ID].reset_index(drop=True)
        self.tr = train[[f"d_{d}" for d in range(1, LAST_TRAIN + 1)]].values.astype(np.float64)
        self.te = test[[f"d_{d}" for d in range(LAST_TRAIN + 1, LAST_TRAIN + H + 1)]].values.astype(np.float64)
        self.w = weights
        self.levels = {}
        for lvl, (keys, _) in LEVELS.items():
            if keys:
                g = self.ids.groupby(keys, sort=False).ngroup().values
            else:
                g = np.zeros(len(self.ids), int)
            n = g.max() + 1
            # default args bind THIS level's g and n; a bare closure would
            # silently give every level the last loop iteration's grouping
            agg = lambda M, g=g, n=n: np.array([np.bincount(g, weights=M[:, j], minlength=n)
                                                for j in range(M.shape[1])]).T
            ins, outs = agg(self.tr), agg(self.te)
            scale = np.empty(n)
            for i in range(n):
                s = ins[i]
                nz = np.flatnonzero(s > 0)
                s = s[nz[0]:] if len(nz) else s
                scale[i] = np.mean(np.diff(s) ** 2)
            key = (self.ids.groupby(keys, sort=False).first().reset_index()[keys]
                   if keys else pd.DataFrame({"_": [0]}))
            wt = self._weights_for(lvl, key, keys)
            self.levels[lvl] = (g, n, agg, outs, scale, wt)

    def _weights_for(self, lvl, key, keys):
        w = self.w[self.w["Level_id"] == lvl]
        if not keys:
            return w["weight"].values
        _, wkeys = LEVELS[lvl]
        # weights file stores the two keys as Agg_Level_1 / Agg_Level_2 in
        # the order the R code reads them (e.g. Level11: state, item)
        a1 = w["Agg_Level_1"].astype(str).values
        a2 = w["Agg_Level_2"].astype(str).values
        lut = {}
        for x, y, v in zip(a1, a2, w["weight"].values):
            lut[(x, y)] = v; lut[(x,)] = v
        vals = []
        for row in key.itertuples(index=False):
            t = tuple(str(v) for v in row)
            vals.append(lut.get(t) if len(t) == 1 else lut.get(t, lut.get(t[::-1])))
        vals = np.array(vals, dtype=float)
        assert not np.isnan(vals).any(), "unmatched weights at %s" % lvl
        return vals

    def score(self, pred):
        per = {}
        for lvl, (g, n, agg, outs, scale, wt) in self.levels.items():
            f = agg(np.asarray(pred, dtype=np.float64))
            rmsse = np.sqrt(np.mean((f - outs) ** 2, axis=1) / scale)
            per[lvl] = float(np.sum(wt * rmsse))
        return float(np.mean(list(per.values()))), per


# ------------------------------------------------------------------ features
def build_long(train, cal, prices):
    days = [f"d_{d}" for d in range(START, LAST_TRAIN + 1)]
    ext = train[ID + days].copy()
    for d in range(LAST_TRAIN + 1, LAST_TRAIN + H + 1):
        ext[f"d_{d}"] = np.nan
    ext["id"] = np.arange(len(ext))
    df = ext.melt(id_vars=["id"] + ID, var_name="d", value_name="sales")
    df["day"] = df["d"].str[2:].astype(np.int16)
    df = df.drop(columns="d")
    for c in ID:
        df[c] = df[c].astype("category")

    cal = cal.copy()
    cal["day"] = cal["d"].str[2:].astype(np.int16)
    cal["date"] = pd.to_datetime(cal["date"])
    for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        cal[c] = cal[c].astype("category").cat.codes.astype(np.int16)
    cal["mday"] = cal["date"].dt.day.astype(np.int8)
    cal["week"] = cal["date"].dt.isocalendar().week.astype(np.int8)
    keep = ["day", "wm_yr_wk", "wday", "month", "year", "mday", "week",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"]
    df = df.merge(cal[keep], on="day", how="left")
    st = df["state_id"].astype(str).values
    df["snap"] = np.select([st == "CA", st == "TX"], [df["snap_CA"], df["snap_TX"]],
                           df["snap_WI"]).astype(np.int8)
    df = df.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    pr = prices.copy()
    g = pr.groupby(["store_id", "item_id"])["sell_price"]
    pr["price_norm"] = pr["sell_price"] / g.transform("max")
    pr["price_rel"] = pr["sell_price"] / g.transform("mean")
    pr["price_chg"] = pr["sell_price"] / g.shift(1) - 1
    pr["price_nuniq"] = g.transform("nunique")
    for c in ("store_id", "item_id"):
        pr[c] = pr[c].astype(df[c].dtype)
    df = df.merge(pr, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df = df[df["sell_price"].notna()]             # item not yet on sale
    df = df.sort_values(["id", "day"]).reset_index(drop=True)

    gs = df.groupby("id", sort=False)["sales"]
    for L in (28, 29, 30, 35, 42, 49, 56):
        df[f"lag_{L}"] = gs.shift(L).astype(np.float32)
    lag28 = df.groupby("id", sort=False)["lag_28"]
    for w in (7, 14, 30, 60, 180):
        df[f"rmean_{w}"] = lag28.transform(lambda s: s.rolling(w, min_periods=1).mean()).astype(np.float32)
    for w in (7, 30):
        df[f"rstd_{w}"] = lag28.transform(lambda s: s.rolling(w, min_periods=2).std()).astype(np.float32)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


FEATURES = None  # set in main


def fit_store(d, rounds=None):
    # Early stopping monitors RMSE, not the Tweedie deviance LightGBM uses by
    # default. A first run that stopped on deviance picked only 55-66 rounds per
    # store -- far below what M5 LightGBM models need -- and was killed before
    # any test-window score was computed, so this change was decided on the
    # validation window alone.
    params = dict(objective="tweedie", tweedie_variance_power=1.1, metric="rmse",
                  learning_rate=0.03,
                  num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=0.1,
                  max_bin=255, n_jobs=16, verbose=-1, seed=0)
    if rounds is None:
        tr = d[d["day"] <= LAST_TRAIN - H]
        va = d[(d["day"] > LAST_TRAIN - H) & (d["day"] <= LAST_TRAIN)]
        ds_tr = lgb.Dataset(tr[FEATURES], tr["sales"], categorical_feature=["item_id", "dept_id", "cat_id"])
        ds_va = lgb.Dataset(va[FEATURES], va["sales"], reference=ds_tr)
        m = lgb.train(params, ds_tr, 3000, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.best_iteration
    tr = d[d["day"] <= LAST_TRAIN]
    ds = lgb.Dataset(tr[FEATURES], tr["sales"], categorical_feature=["item_id", "dept_id", "cat_id"])
    return lgb.train(params, ds, rounds)


def main():
    global FEATURES
    t0 = time.time()
    train = pd.read_csv(os.path.join(M5, "sales_train_evaluation.csv"))
    test = pd.read_csv(os.path.join(OFF, "sales_test_evaluation.csv"))
    test = train[ID].merge(test, on=ID, how="left")          # same row order
    weights = pd.read_csv(os.path.join(OFF, "weights_evaluation.csv"))
    scorer = WRMSSE(train, test, weights)

    # ---- self-check against two published numbers --------------------------
    es = pd.read_csv(os.path.join(OFF, "ES_bu.csv"))
    es = es[es["id"].str.endswith("_evaluation")].copy()
    es["key"] = es["id"].str.replace("_evaluation", "", regex=False)
    key = train["item_id"] + "_" + train["store_id"]
    es_pred = es.set_index("key").loc[key, [f"F{i}" for i in range(1, H + 1)]].values
    s_es, _ = scorer.score(es_pred)
    snaive = scorer.tr[:, -7:][:, np.arange(H) % 7]
    s_sn, _ = scorer.score(snaive)
    print("self-check  ES_bu %.6f (published 0.670981)   sNaive %.6f (published 0.847017)" % (s_es, s_sn))
    assert abs(s_es - 0.670981) < 5e-4 and abs(s_sn - 0.847017) < 5e-4, "scorer does not match the leaderboard"

    # ---- features -----------------------------------------------------------
    cal = pd.read_csv(os.path.join(M5, "calendar.csv"))
    prices = pd.read_csv(os.path.join(M5, "sell_prices.csv"))
    df = build_long(train, cal, prices)
    FEATURES = [c for c in df.columns if c not in ("id", "sales", "day", "wm_yr_wk",
                                                   "store_id", "state_id")]
    print("long frame %d rows x %d features, built in %.0fs" % (len(df), len(FEATURES), time.time() - t0))

    pred = np.zeros((len(train), H))
    rounds = {}
    for store in train["store_id"].unique():
        t1 = time.time()
        d = df[df["store_id"] == store]
        r = fit_store(d)
        m = fit_store(d, rounds=r)
        te = d[d["day"] > LAST_TRAIN]
        p = m.predict(te[FEATURES])
        pred[te["id"].values, te["day"].values - LAST_TRAIN - 1] = p
        rounds[store] = int(r)
        print("  %s  best rounds %4d  %.0fs" % (store, r, time.time() - t1), flush=True)

    total, per = scorer.score(pred)
    print("\nLightGBM WRMSSE %.6f" % total)
    ranks = sorted(list(PUBLISHED.items()) + [("OURS: LightGBM", total)], key=lambda x: x[1])
    for n, v in ranks:
        print("  %-26s %.4f" % (n, v))
    top50 = pd.read_excel(os.path.join(HERE, "papers", "M5_Scores_and_Ranks.xlsx"),
                          "Accuracy-Top50 (AL)").iloc[1:, 13].astype(float).values
    better = int((top50 < total).sum())
    print("  -> would rank %d of the top 50 (%d top-50 teams scored lower)" % (better + 1, better)
          if better < 50 else "  -> outside the top 50 (50th = 0.5760)")
    imp = 1 - total / PUBLISHED["ES_bu (best benchmark)"]
    print("  improvement over best benchmark ES_bu: %.1f %% (winner: 22.4 %%)" % (100 * imp))

    json.dump({"anchor": "Makridakis et al. 2022, IJF 38(4) -- M5 Accuracy, private leaderboard",
               "self_check": {"ES_bu": s_es, "sNaive": s_sn},
               "wrmsse": total, "per_level": per, "rounds": rounds,
               "train_start_day": START, "top50_better_than_ours": better,
               "improvement_over_ES_bu": imp, "published": PUBLISHED,
               "seconds": time.time() - t0}, open(OUT, "w"), indent=2)
    np.save(os.path.join(HERE, "m5_lightgbm_forecast.npy"), pred.astype(np.float32))
    print("written -> %s" % OUT)


if __name__ == "__main__":
    main()
