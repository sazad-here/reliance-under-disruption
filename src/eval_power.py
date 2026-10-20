"""How much evidence is in the test set, really? A check on the evaluation set's size.

The open item reads: "11 held-out series can yield many more windows, which
would narrow the 0.30-wide PR-AUC interval. Cheap; worth knowing."

This script is the "worth knowing" part, and it is worth running *before*
generating more episodes, because the premise may not hold. The 78 test
episodes are 100-week windows drawn from 11 M5 series of 273 weeks each, so
roughly 7 windows per series with heavy overlap. The existing bootstrap
resamples episodes, which treats those 78 as independent draws. They are not.
Adding more windows from the same 11 series makes the episode-level interval
shrink whether or not any new information has arrived.

So compute both:

    episode bootstrap   resamples the 78 windows      (what the reported intervals use)
    series bootstrap    resamples the 11 series       (the honest clustering)

and a learning curve over the number of test episodes, to show which of the two
responds to more data and which is just counting overlapping windows.

    python src/eval_power.py
    python src/eval_power.py --model hgb --reps 3000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines_strong import HORIZON, LOOKBACK, build_features, load_corpus, platt  # noqa: E402


def cluster_boot_ci(y, p, clusters, fn, n_reps=2000, seed=0):
    """Bootstrap resampling whole clusters, with replacement.

    `clusters` is the unit we are willing to call independent. Pass episode ids
    for the optimistic reading, series ids for the defensible one.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    # Precompute row indices per cluster; the naive version rebuilds this
    # inside the loop and turns 2000 reps into several minutes.
    where = {c: np.where(clusters == c)[0] for c in uniq}
    vals = []
    for _ in range(n_reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([where[c] for c in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(fn(yy, p[idx]))
    if not vals:
        return (np.nan, np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
            float(np.std(vals)))


def fit_predict(df, Xv, y, model="lr"):
    m_tr = (df["split"] == "train").values
    m_va = (df["split"] == "val").values
    m_te = (df["split"] == "test").values

    if model == "lr":
        sc = StandardScaler().fit(Xv[m_tr])
        clf = LogisticRegression(max_iter=4000, C=1.0).fit(sc.transform(Xv[m_tr]), y[m_tr])
        p_va = clf.predict_proba(sc.transform(Xv[m_va]))[:, 1]
        p_te = clf.predict_proba(sc.transform(Xv[m_te]))[:, 1]
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_depth=6,
            early_stopping=True, validation_fraction=0.15, random_state=0,
        ).fit(Xv[m_tr], y[m_tr])
        p_va = clf.predict_proba(Xv[m_va])[:, 1]
        p_te = clf.predict_proba(Xv[m_te])[:, 1]

    _, cal_te = platt(p_va, y[m_va], p_te)
    return m_te, cal_te


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="lr", choices=["lr", "hgb"])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="results/eval_power.json")
    args = ap.parse_args()

    label = "stockout_within_%d" % HORIZON
    df = load_corpus()
    # Build features on the FULL per-episode history, then subset to the scored
    # rows. Filtering on period first left every *_lag48 column referencing rows
    # that had already been dropped: measured 100% NaN on lag48 and 30.7% across
    # all lag features, silently zeroed by nan_to_num below. The scored row set
    # is identical either way, so this only restores information.
    df = df[df["label_valid"] == 1].reset_index(drop=True)
    X = build_features(df)
    _keep = (df["period"] >= LOOKBACK).values
    df = df[_keep].reset_index(drop=True)
    X = X[_keep].reset_index(drop=True)
    y = df[label].values
    feat = [c for c in X.columns if c not in ("episode", "period")]
    Xv = np.nan_to_num(X[feat].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    m_te, p_te = fit_predict(df, Xv, y, args.model)
    y_te = y[m_te]
    ep_te = df.loc[m_te, "episode"].values
    se_te = df.loc[m_te, "m5_series"].values

    n_ep, n_se = len(np.unique(ep_te)), len(np.unique(se_te))
    print("=" * 78)
    print("TEST-SET EVIDENCE  (model: %s)" % args.model)
    print("=" * 78)
    print("  %d test rows | %d episodes | %d independent M5 series"
          % (len(y_te), n_ep, n_se))
    print("  %.1f episodes per series -- overlapping windows of the same demand\n"
          % (n_ep / n_se))

    out = {"model": args.model, "n_rows": int(len(y_te)),
           "n_episodes": int(n_ep), "n_series": int(n_se), "metrics": {}}

    for mname, fn in (("PR-AUC", average_precision_score), ("AUROC", roc_auc_score)):
        point = fn(y_te, p_te)
        lo_e, hi_e, sd_e = cluster_boot_ci(y_te, p_te, ep_te, fn, args.reps, seed=0)
        lo_s, hi_s, sd_s = cluster_boot_ci(y_te, p_te, se_te, fn, args.reps, seed=0)
        print("  %-7s point %.3f" % (mname, point))
        print("     by episode (n=%2d)  [%.3f, %.3f]  width %.3f" % (n_ep, lo_e, hi_e, hi_e - lo_e))
        print("     by series  (n=%2d)  [%.3f, %.3f]  width %.3f" % (n_se, lo_s, hi_s, hi_s - lo_s))
        infl = (hi_s - lo_s) / (hi_e - lo_e) if hi_e > lo_e else float("nan")
        print("     series interval is %.2fx the episode interval\n" % infl)
        out["metrics"][mname] = {
            "point": float(point),
            "ci_episode": [lo_e, hi_e], "width_episode": hi_e - lo_e,
            "ci_series": [lo_s, hi_s], "width_series": hi_s - lo_s,
            "inflation": infl,
        }

    # ---- learning curve: does the interval respond to more episodes? --------
    print("  Interval width vs number of TEST EPISODES (PR-AUC):")
    print("  %-10s %-22s %s" % ("episodes", "by-episode width", "by-series width"))
    rng = np.random.default_rng(0)
    uniq_ep = np.unique(ep_te)
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(4, int(len(uniq_ep) * frac))
        sub = rng.choice(uniq_ep, size=k, replace=False) if k < len(uniq_ep) else uniq_ep
        mask = np.isin(ep_te, sub)
        if y_te[mask].min() == y_te[mask].max():
            continue
        lo_e, hi_e, _ = cluster_boot_ci(y_te[mask], p_te[mask], ep_te[mask],
                                        average_precision_score, 800, seed=1)
        lo_s, hi_s, _ = cluster_boot_ci(y_te[mask], p_te[mask], se_te[mask],
                                        average_precision_score, 800, seed=1)
        print("  %-10d %-22.3f %.3f" % (k, hi_e - lo_e, hi_s - lo_s))
        curve.append({"episodes": int(k), "width_episode": hi_e - lo_e,
                      "width_series": hi_s - lo_s})
    out["curve"] = curve

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n  wrote %s" % args.out)

    pr = out["metrics"]["PR-AUC"]
    print("\n" + "-" * 78)
    if pr["inflation"] > 1.15:
        print("  READ: the by-series interval is materially wider. The 78 episodes")
        print("  are NOT 78 independent observations, and generating more windows")
        print("  from the same 11 series would narrow the by-episode interval")
        print("  without adding evidence. To genuinely narrow it, hold out more")
        print("  M5 SERIES -- not more windows.")
    else:
        print("  READ: episode and series clustering agree, so the windows are")
        print("  behaving close to independently. Generating more test episodes")
        print("  from the held-out series is a legitimate way to narrow the CI.")
    print("-" * 78)


if __name__ == "__main__":
    main()
