"""
Robustness checks for the live stockout-risk model (validation-plan.html, checks 1, 5, 6).

  A  OUT-OF-TIME. Train on episodes whose demand comes from M5 weeks 0-172 only;
     test on episodes built from weeks 173-272, which no training episode saw.
     Series CV (the reported AUROC 0.897) never separates time; this does.
  B  LEAVE-ONE-STATE-OUT. The 70 store-department series come from 10 stores in
     3 US states (series 0-27 CA, 28-48 TX, 49-69 WI; demand_source.load_weekly
     groups by store then department). Train on two states, test on the third.
  C  STRESS. The FROZEN live model (models/live_risk_lgbm.txt, exactly what
     participants see) scored on new episodes under shifted conditions, on the
     two held-out session series it never trained on: demand x0.7 / x1.3,
     disruption rate x2 / x3, heavier lead-time delays.
  D  DECISION VALUE. Does acting on the risk % beat acting without a model?
     A simple base-stock rule (no safety stock) is run three ways on identical
     demand, disruptions and delay draws: alone; plus a top-up when the model's
     risk >= 50 %; plus the same top-up when net position < 0 (the best single
     feature, no model). Threshold and top-up are fixed in advance, not tuned.

A and B compare LightGBM (CPU; the pip wheel has no GPU build), XGBoost on the
GPU (CUDA) and logistic regression on the live model's own features.
Episodes are generated fresh with make_corpus.run_episode; corpus.csv is
never touched.

Run:  python lightgbm-pivot/robustness_study.py
Out:  lightgbm-pivot/robustness_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import live_model  # noqa: E402
import make_corpus as mc  # noqa: E402
from baselines_strong import load_corpus  # noqa: E402
from chart_data import calibration  # noqa: E402
from compare_fair import corpus_fingerprint  # noqa: E402
from demand_source import M5DemandSource, load_weekly  # noqa: E402
from interface.world import SESSION_SERIES  # noqa: E402
from policy import reference_order  # noqa: E402
from simulator import SupplyChainTwin, TwinParams  # noqa: E402
from train_baselines import expected_calibration_error  # noqa: E402

OUT = os.path.join(HERE, "robustness_results.json")
STATES = {"California": range(0, 28), "Texas": range(28, 49), "Wisconsin": range(49, 70)}
CUT = 173            # M5 weeks 0-172 train, 173-272 test (exactly one 100-week window)
N_BOOT = 1000


# ------------------------------------------------------------------ episodes
def episodes(n, seed, weekly, pool, params=None, dis_rate=mc.DISRUPTION_RATE):
    """n fresh corpus-style episodes from `weekly` restricted to series `pool`."""
    params = params or TwinParams.from_fitted()
    rng = np.random.default_rng(seed)
    saved = (mc.M5DemandSource, mc.ALL_SERIES, mc.DISRUPTION_RATE)
    mc.M5DemandSource = lambda r, L, series_pool=None: M5DemandSource(r, L, weekly=weekly, series_pool=pool)
    mc.ALL_SERIES, mc.DISRUPTION_RATE = True, dis_rate
    try:
        frames = [mc.run_episode(seed * 100_000 + i, params, rng) for i in range(n)]
    finally:
        mc.M5DemandSource, mc.ALL_SERIES, mc.DISRUPTION_RATE = saved
    return pd.concat(frames, ignore_index=True)


def build(df):
    d = live_model._build(df)
    return d[d["_valid"] == 1].reset_index(drop=True)


# ------------------------------------------------------------------ metrics
def metrics(y, p, groups):
    rng = np.random.default_rng(0)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    boot = []
    for _ in range(N_BOOT):
        ix = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq))])
        if y[ix].min() != y[ix].max():
            boot.append(roc_auc_score(y[ix], p[ix]))
    cal = calibration(y, p)
    return {"auroc": float(roc_auc_score(y, p)),
            "auroc_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "ece": float(expected_calibration_error(y, p)),
            "cal_slope": cal["slope"], "cal_intercept": cal["intercept"],
            "n": int(len(y)), "positive_rate": float(y.mean())}


def platt(p_va, y_va, p_te):
    a, b = live_model._platt_fit(p_va, y_va)
    return live_model._platt_apply(p_te, a, b)


def fit_three(tr, va, te, cols):
    """LightGBM (CPU), XGBoost (CUDA) and LR on the live features; Platt on val."""
    X = lambda d: d[cols]
    ytr, yva = tr["_y"].values, va["_y"].values
    out, secs = {}, {}

    t = time.time()
    m = lgb.train(live_model.PARAMS, lgb.Dataset(X(tr), ytr), 2000,
                  valid_sets=[lgb.Dataset(X(va), yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    out["LightGBM"] = platt(m.predict(X(va)), yva, m.predict(X(te))); secs["LightGBM"] = time.time() - t

    t = time.time()
    x = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2000, learning_rate=0.03,
                          max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", early_stopping_rounds=100, random_state=0)
    x.fit(X(tr), ytr, eval_set=[(X(va), yva)], verbose=False)
    assert '"device":"cuda' in x.get_booster().save_config(), "XGBoost did not run on the GPU"
    out["XGBoost (GPU)"] = platt(x.predict_proba(X(va))[:, 1], yva, x.predict_proba(X(te))[:, 1])
    secs["XGBoost (GPU)"] = time.time() - t

    t = time.time()
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=4000)).fit(X(tr), ytr)
    out["Logistic regression"] = platt(lr.predict_proba(X(va))[:, 1], yva, lr.predict_proba(X(te))[:, 1])
    secs["Logistic regression"] = time.time() - t
    return out, secs


def split_val(d, frac=0.15, seed=0):
    s = np.unique(d["_series"])
    va = set(np.random.default_rng(seed).choice(s, max(1, int(round(frac * len(s)))), replace=False).tolist())
    m = d["_series"].isin(va).values
    return d[~m].reset_index(drop=True), d[m].reset_index(drop=True)


# ------------------------------------------------------------------ A
def study_out_of_time(weekly, cols):
    pool_tr = [s for s in range(70) if s not in SESSION_SERIES]
    tr_all = build(episodes(420, 11, weekly[:, :CUT], pool_tr))
    te = build(episodes(140, 12, weekly[:, CUT:], range(70)))
    tr, va = split_val(tr_all)
    preds, secs = fit_three(tr, va, te, cols)
    y, g = te["_y"].values.astype(int), te["_series"].values
    res = {n: metrics(y, p, g) for n, p in preds.items()}
    print("A out-of-time  train weeks 0-%d (%d rows) -> test weeks %d-272 (%d rows)"
          % (CUT - 1, len(tr_all), CUT, len(te)))
    for n, r in res.items():
        print("   %-20s AUROC %.3f [%.3f, %.3f]  ECE %.3f  slope %.2f  int %+.3f  (%.1fs)"
              % (n, r["auroc"], *r["auroc_ci"], r["ece"], r["cal_slope"], r["cal_intercept"], secs[n]))
    return {"train_rows": len(tr_all), "test_rows": len(te), "models": res, "seconds": secs}


# ------------------------------------------------------------------ B
def study_states(cols):
    data = build(load_corpus())
    out = {}
    print("B leave-one-state-out (existing corpus, live features)")
    for st, rng_ in STATES.items():
        test_m = data["_series"].isin(list(rng_)).values
        tr, va = split_val(data[~test_m].reset_index(drop=True), seed=1)
        te = data[test_m].reset_index(drop=True)
        preds, secs = fit_three(tr, va, te, cols)
        y, g = te["_y"].values.astype(int), te["_series"].values
        out[st] = {n: metrics(y, p, g) for n, p in preds.items()}
        print("   test %-10s " % st + "  ".join("%s %.3f" % (n.split()[0], r["auroc"]) for n, r in out[st].items()))
    return out


# ------------------------------------------------------------------ C
def study_stress(weekly):
    model = live_model.RiskModel()
    fitted = TwinParams.from_fitted()
    heavy = TwinParams.from_fitted(delay_pmf=(0.30, 0.30, 0.20, 0.20))
    conds = {
        "Baseline": dict(),
        "Demand x0.7": dict(weekly=np.rint(weekly * 0.7).astype(np.int64)),
        "Demand x1.3": dict(weekly=np.rint(weekly * 1.3).astype(np.int64)),
        "Disruptions x2": dict(dis_rate=2 * mc.DISRUPTION_RATE),
        "Disruptions x3": dict(dis_rate=3 * mc.DISRUPTION_RATE),
        "Heavier lead-time delays": dict(params=heavy),
    }
    out = {}
    print("C stress test of the frozen live model (session series %s, never trained on)" % (SESSION_SERIES,))
    for name, kw in conds.items():
        d = build(episodes(120, 21, kw.get("weekly", weekly), SESSION_SERIES,
                           params=kw.get("params", fitted), dis_rate=kw.get("dis_rate", mc.DISRUPTION_RATE)))
        raw = model.booster.predict(d[model.cols])
        z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        p = 1 / (1 + np.exp(-(model.a * z + model.b)))
        y = d["_y"].values.astype(int)
        # cluster by episode: only two series, so episodes are the resampling unit
        # a new episode starts wherever the period counter goes back down
        ep = (d["_period"].diff().fillna(-1) < 0).cumsum().values
        out[name] = metrics(y, p, ep)
        r = out[name]
        print("   %-26s AUROC %.3f [%.3f, %.3f]  ECE %.3f  slope %.2f  int %+.3f  stockouts %.1f %%"
              % (name, r["auroc"], *r["auroc_ci"], r["ece"], r["cal_slope"], r["cal_intercept"],
                 100 * r["positive_rate"]))
    return out


# ------------------------------------------------------------------ D
def study_decision(weekly, n=150, horizon=52, tau=0.5, top_up=60):
    """Paired policies on identical worlds; top_up = delta, the study's own tolerance."""
    model = live_model.RiskModel()
    params = TwinParams.from_fitted()
    rng = np.random.default_rng(31)

    def base(state):
        target = state.demand * (params.lead_time_nominal + 1)
        return max(0, target - state.inventory_position)

    def net_position(state):
        return state.inventory_position - state.demand * (state.lead_time + 1)

    rules = {
        "Base-stock rule alone": lambda s, hist: base(s),
        "Base-stock + top-up when model risk >= 50 %": lambda s, hist: base(s) + (top_up if model.predict(hist) >= tau else 0),
        "Base-stock + top-up when net position < 0 (no model)": lambda s, hist: base(s) + (top_up if net_position(s) < 0 else 0),
        "Reference policy Q* (upper bound)": lambda s, hist: reference_order(s, params),
    }
    res = {k: {"cost": [], "stockout_weeks": []} for k in rules}
    for i in range(n):
        series = SESSION_SERIES[i % len(SESSION_SERIES)]
        start = int(rng.integers(0, weekly.shape[1] - horizon + 1))
        calendar = mc.random_calendar(rng, horizon)
        seed = int(rng.integers(1 << 31))
        for name, rule in rules.items():
            src = M5DemandSource(np.random.default_rng(0), horizon, weekly=weekly[:, start:start + horizon],
                                 series_pool=[series])
            twin = SupplyChainTwin(params=TwinParams(**{**params.__dict__, "horizon": horizon}),
                                   calendar=calendar, seed=seed, demand_source=src)
            state, hist, cost, so = twin.observe(), [], 0.0, 0
            while not twin.done:
                hist.append(live_model.state_row(state))
                q = int(np.clip(rule(state, hist), 0, params.max_order))
                state, c, rec = twin.step(q)
                cost += c; so += int(rec["stockout"])
            res[name]["cost"].append(cost); res[name]["stockout_weeks"].append(so)
    ref = "Base-stock rule alone"
    out = {}
    brng = np.random.default_rng(0)
    idx = [brng.integers(0, n, n) for _ in range(N_BOOT)]
    print("D decision value (%d paired %d-week episodes on the session series)" % (n, horizon))
    for name, r in res.items():
        c, s = np.array(r["cost"]), np.array(r["stockout_weeks"])
        c0, s0 = np.array(res[ref]["cost"]), np.array(res[ref]["stockout_weeks"])
        dc = [np.mean(c[i] - c0[i]) for i in idx]
        ds = [np.mean(s[i] - s0[i]) for i in idx]
        out[name] = {"mean_cost": float(c.mean()), "mean_stockout_weeks": float(s.mean()),
                     "cost_vs_base": float(np.mean(c - c0)), "cost_vs_base_ci": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
                     "stockouts_vs_base": float(np.mean(s - s0)), "stockouts_vs_base_ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]}
        o = out[name]
        print("   %-52s cost %8.0f  (%+7.0f [%+.0f, %+.0f])  stockout wks %5.2f (%+.2f)"
              % (name, o["mean_cost"], o["cost_vs_base"], *o["cost_vs_base_ci"], o["mean_stockout_weeks"], o["stockouts_vs_base"]))
    # the model-vs-no-model contrast itself
    m = np.array(res["Base-stock + top-up when model risk >= 50 %"]["cost"])
    nm = np.array(res["Base-stock + top-up when net position < 0 (no model)"]["cost"])
    d = [np.mean(m[i] - nm[i]) for i in idx]
    out["model_minus_no_model_cost"] = {"mean": float(np.mean(m - nm)),
                                        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    print("   model rule minus no-model rule: cost %+.0f [%+.0f, %+.0f]" % (np.mean(m - nm), *out["model_minus_no_model_cost"]["ci"]))
    return {"settings": {"episodes": n, "horizon": horizon, "risk_threshold": tau, "top_up_units": top_up}, "policies": out}


def main():
    weekly = load_weekly()
    cols = json.load(open(live_model.META_PATH, encoding="utf-8"))["features"]
    t0 = time.time()
    res = {"corpus_fingerprint": corpus_fingerprint(), "live_model_cv_auroc": 0.897}
    res["A_out_of_time"] = study_out_of_time(weekly, cols)
    res["B_leave_one_state_out"] = study_states(cols)
    res["C_stress_frozen_model"] = study_stress(weekly)
    res["D_decision_value"] = study_decision(weekly)
    res["seconds"] = time.time() - t0
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("written -> %s  (%.0f s)" % (OUT, res["seconds"]))


if __name__ == "__main__":
    main()
