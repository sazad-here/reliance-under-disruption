"""
The live LightGBM stockout-risk model: the AI participants actually see.

WHAT IT PREDICTS. P(stockout within the next 2 weeks), from the participant's
own warehouse state, every week of the session. The number shown on screen
("Estimated stockout risk: 34 %") is this model's calibrated output.

WHY A SEPARATE MODEL FROM THE BENCHMARKED ONE. The model validated in
lightgbm-pivot/ uses 116 features reaching 48 weeks back. A session lasts
5 + 30 weeks and the twin restarts between practice and main, so those features
do not exist live. This model uses only what a session has: the current state
plus at most 4 weeks of history. Same learner, same label, same calibration
method; smaller window. Its cross-validated performance is reported next to the
benchmarked model's, so the paper can state both.

NO TRAIN/SERVE SKEW. `features()` is the ONE function that turns a sequence of
weekly states into model inputs. Training applies it per corpus episode; the
app applies it to the participant's history. Missing history (the first weeks)
is NaN in both, and LightGBM handles NaN natively.

HELD-OUT DEMAND. The two M5 series that sessions replay (interface/world.py)
are excluded from training, so the model predicts demand it has never seen.

Run:  python src/live_model.py      (trains, evaluates, writes models/)
"""
from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL_DIR = os.path.join(HERE, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "live_risk_lgbm.txt")
META_PATH = os.path.join(MODEL_DIR, "live_risk_meta.json")

RAW = ["demand", "inventory", "on_order", "lead_time", "backorders",
       "supplier_reliability", "transport_ok", "disruption_flag"]
LAGS = (1, 2, 3, 4)
LABEL = "stockout_within_2"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31,
              min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, seed=0)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for every row of ONE episode/session, oldest first.

    Row t uses only rows <= t. Called unchanged for training and live use.
    """
    d = df[RAW].astype(float).reset_index(drop=True)
    ip = d["inventory"] + d["on_order"] - d["backorders"]
    need = d["demand"] * (d["lead_time"] + 1)
    X = d.copy()
    X["inventory_position"] = ip
    X["net_position"] = ip - need
    X["cover_ratio"] = (ip / need.replace(0, np.nan)).clip(-5, 5)
    for col, series in (("demand", d["demand"]), ("inventory", d["inventory"]),
                        ("backorders", d["backorders"]), ("ip", ip)):
        for k in LAGS:
            X["%s_lag%d" % (col, k)] = series.shift(k)
    X["demand_mean4"] = d["demand"].rolling(4, min_periods=1).mean()
    X["demand_std4"] = d["demand"].rolling(4, min_periods=2).std()
    X["ip_change"] = ip.diff()
    return X


def state_row(state) -> dict:
    """A simulator State as one raw row (the same fields the corpus stores)."""
    v = state.as_vector()
    return dict(zip(RAW, v.tolist()))


class RiskModel:
    """Loaded once per process; `predict(rows)` -> calibrated probability."""

    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "live risk model missing at %s -- run: python src/live_model.py" % model_path)
        self.booster = lgb.Booster(model_file=model_path)
        with open(meta_path, encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.cols = self.meta["features"]
        self.a, self.b = self.meta["platt"]

    def predict(self, rows: list[dict]) -> float:
        """Risk for the LAST row, given the history rows before it, oldest first.

        Rows are validated before use: `pd.DataFrame(rows, columns=RAW)` would
        otherwise fill a missing field with NaN, which LightGBM accepts silently
        and turns into a plausible but wrong risk. Order is the caller's
        contract and cannot be checked here -- reversing the history changes the
        lag features and so the answer.
        """
        if not rows:
            raise ValueError("predict() needs at least one state row")
        missing = sorted(set(RAW) - set(rows[-1]))
        if missing:
            raise ValueError("state row is missing %s" % ", ".join(missing))
        X = features(pd.DataFrame(rows, columns=RAW))[self.cols].tail(1)
        raw = float(self.booster.predict(X)[0])
        z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        return float(1.0 / (1.0 + np.exp(-(self.a * z + self.b))))


_MODEL = None


def get_model() -> RiskModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = RiskModel()
    return _MODEL


# ------------------------------------------------------------------ training
def _build(df):
    parts = []
    for _, g in df.groupby("episode", sort=False):
        g = g.sort_values("period")
        X = features(g)
        X["_y"] = g[LABEL].values
        X["_valid"] = g["label_valid"].values
        X["_series"] = g["m5_series"].values
        X["_period"] = g["period"].values
        parts.append(X)
    return pd.concat(parts, ignore_index=True)


def _platt_fit(p, y):
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def _platt_apply(p, a, b):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def main():
    from sklearn.metrics import average_precision_score, roc_auc_score
    from baselines_strong import load_corpus, series_folds
    from compare_fair import corpus_fingerprint
    from interface.world import SESSION_SERIES
    from train_baselines import expected_calibration_error

    df = load_corpus()
    data = _build(df)
    data = data[data["_valid"] == 1].reset_index(drop=True)
    held = data["_series"].isin(SESSION_SERIES).values
    pool = data[~held].reset_index(drop=True)
    cols = [c for c in data.columns if not c.startswith("_")]
    y, ser = pool["_y"].values.astype(int), pool["_series"].values
    print("rows %d (excluding session series %s) | features %d | positive %.3f"
          % (len(pool), SESSION_SERIES, len(cols), y.mean()))

    # ---- 5-fold series cross-validation: the reported performance ----------
    oof = np.full(len(pool), np.nan)
    rounds = []
    for k in range(5):
        tr, va, te = series_folds(ser, k)
        m = lgb.train(PARAMS, lgb.Dataset(pool.loc[tr, cols], y[tr]), 2000,
                      valid_sets=[lgb.Dataset(pool.loc[va, cols], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds.append(m.best_iteration)
        a, b = _platt_fit(m.predict(pool.loc[va, cols]), y[va])
        oof[te] = _platt_apply(m.predict(pool.loc[te, cols]), a, b)
    late = (pool["_period"] >= 48).values
    cv = {"auroc": float(roc_auc_score(y, oof)), "pr_auc": float(average_precision_score(y, oof)),
          "ece": float(expected_calibration_error(y, oof)),
          "auroc_period_ge_48": float(roc_auc_score(y[late], oof[late])),
          "auroc_first_5_weeks": float(roc_auc_score(y[pool["_period"].values < 5],
                                                     oof[pool["_period"].values < 5]))}
    print("5-fold series CV: AUROC %.3f | PR-AUC %.3f | ECE %.3f | AUROC weeks>=48 %.3f | first 5 weeks %.3f"
          % (cv["auroc"], cv["pr_auc"], cv["ece"], cv["auroc_period_ge_48"], cv["auroc_first_5_weeks"]))

    # ---- held-out check on the session series themselves --------------------
    # ---- final model: all pool series, calibration on a held-out 15 % -------
    rng = np.random.default_rng(0)
    uniq = np.unique(ser)
    cal_series = set(rng.choice(uniq, size=max(1, int(0.15 * len(uniq))), replace=False).tolist())
    cal = np.isin(ser, list(cal_series))
    n_rounds = int(np.median(rounds))
    final = lgb.train(PARAMS, lgb.Dataset(pool.loc[~cal, cols], y[~cal]), n_rounds)
    a, b = _platt_fit(final.predict(pool.loc[cal, cols]), y[cal])

    hs = data[held]
    p_hs = _platt_apply(final.predict(hs[cols]), a, b)
    session = {"auroc": float(roc_auc_score(hs["_y"], p_hs)), "n": int(len(hs)),
               "ece": float(expected_calibration_error(hs["_y"].values, p_hs))}
    print("session series (never trained on): AUROC %.3f | ECE %.3f | n %d"
          % (session["auroc"], session["ece"], session["n"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save_model(MODEL_PATH)
    # `params` and `trained_at` are recorded so the artifact states the config it
    # was built with. LightGBM also writes the full parameter block into
    # live_risk_lgbm.txt, so an older artifact without these keys is still
    # auditable -- read the [seed: ...] / [learning_rate: ...] lines there.
    meta = {"features": cols, "platt": [a, b], "label": LABEL, "lags": list(LAGS),
            "rounds": n_rounds, "cv_rounds": rounds, "cv": cv, "session_series_check": session,
            "excluded_series": list(SESSION_SERIES), "corpus_fingerprint": corpus_fingerprint(),
            "lightgbm_version": lgb.__version__, "params": dict(PARAMS),
            "trained_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")}
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("written -> %s" % MODEL_PATH)


if __name__ == "__main__":
    main()
