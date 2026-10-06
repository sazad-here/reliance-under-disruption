"""
Real demand sequences from M5, replayed into the twin.

WHY THIS EXISTS

The synthetic demand process (negative binomial + weekly seasonality) is
memoryless. Measured autocorrelation of weekly demand:

        lag        real M5     synthetic
        1 week      +0.797       +0.107
        4 weeks     +0.744       -0.162
       13 weeks     +0.562       +0.111
       52 weeks     +0.317       -0.147

Real demand carries structure out to a full year: trends, promotions, holiday
effects. Mine carried none past the weekly cycle. PatchTST exists to find
long-range temporal patterns, so it was handicapped by construction -- which is
why a logistic regression matched it.

GROUND TRUTH SURVIVES. Q* is a function of the inventory position and the demand
FORECAST, not of demand being synthetic. The plan already defines "correct" as
agreeing with a base-stock heuristic, not omniscient optimality (section 06), so
replaying real demand changes nothing about how decisions are graded.

WHAT IS STILL OURS: the disruption calendar, the lead times, the costs, and every
ordering decision. Only the demand stream is real.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M5 = os.path.join(HERE, "..", "data", "raw", "m5", "sales_train_validation.csv")
CACHE = os.path.join(HERE, "..", "data", "processed", "m5_weekly.npy")

TARGET_MEAN = 140.0  # rescale each series to one warehouse's scale


def load_weekly(rescale=True, cache=True):
    """Weekly demand per store-department, rescaled to a common warehouse size.

    Aggregated to store-department because a single SKU is far too intermittent
    to drive one warehouse. Rescaling preserves the SHAPE of each series (its
    autocorrelation, trend and seasonality) while making the level comparable
    across series and to the synthetic baseline.
    """
    if cache and os.path.exists(CACHE):
        # The cache holds the RESCALED array, so honouring a cache hit when the
        # caller asked for rescale=False would hand back the wrong thing. Only
        # serve the cache for the case it was built for.
        if rescale:
            return np.load(CACHE)

    if not os.path.exists(M5):
        raise SystemExit("M5 not found at %s" % M5)

    df = pd.read_csv(M5)
    day_cols = [c for c in df.columns if c.startswith("d_")]
    grouped = df.groupby(["store_id", "dept_id"])[day_cols].sum()
    daily = grouped.values.astype(np.float64)

    n_weeks = daily.shape[1] // 7
    weekly = daily[:, : n_weeks * 7].reshape(daily.shape[0], n_weeks, 7).sum(axis=2)

    if rescale:
        means = weekly.mean(axis=1, keepdims=True)
        # No M5 store-department series is all-zero today (checked: min mean
        # 82.7), so this is insurance for a re-run on different data rather
        # than a live bug -- but a divide-by-zero here would put inf/nan
        # straight into the cached demand every episode is built from.
        if not (means > 0).all():
            raise ValueError("all-zero demand series present; drop before rescaling")
        weekly = weekly / means * TARGET_MEAN

    weekly = np.rint(weekly).astype(np.int64)
    weekly = np.maximum(weekly, 0)
    np.save(CACHE, weekly)
    return weekly


class M5DemandSource:
    """Draws a real demand window and serves it period by period.

    Each episode gets a different (series, start-week) pair, so the corpus sees
    many real demand regimes rather than one.
    """

    def __init__(self, rng, length, weekly=None, min_start=0, series_pool=None):
        """`series_pool` restricts which M5 series this episode may draw from.

        THIS IS NOT OPTIONAL FOR TRAIN/TEST SPLITTING. M5 has 273 weeks. With a
        260-week episode there are only 14 possible start positions per series,
        so two episodes from the same series overlap ~98% in their demand. If
        train and test episodes can share a series, the test set contains demand
        the model already saw -- it recalls rather than predicts.

        Measured cost of getting this wrong: AUROC inflated by roughly 0.14.

        Pass disjoint `series_pool` sets for train / val / test.
        """
        self.weekly = load_weekly() if weekly is None else weekly
        n_series, n_weeks = self.weekly.shape
        if length > n_weeks:
            raise ValueError("requested %d weeks, M5 has %d" % (length, n_weeks))
        pool = list(range(n_series)) if series_pool is None else list(series_pool)
        if not pool:
            raise ValueError("series_pool is empty")
        self.series_idx = int(pool[rng.integers(len(pool))])
        max_start = n_weeks - length
        self.start = int(rng.integers(min_start, max_start + 1)) if max_start > 0 else 0
        self.window = self.weekly[self.series_idx, self.start:self.start + length]
        self.length = length

    def __call__(self, period, multiplier=1.0):
        """Demand for this period. `multiplier` applies a demand-spike disruption."""
        base = self.window[period % self.length]
        return int(round(base * multiplier))

    def stats(self):
        w = self.window
        return {
            "series": self.series_idx,
            "start_week": self.start,
            "mean": round(float(w.mean()), 1),
            "sd": round(float(w.std()), 1),
            "cv": round(float(w.std() / max(w.mean(), 1e-9)), 3),
        }


def autocorrelation(series, lag):
    x = np.asarray(series, dtype=float)
    if x.std() == 0:
        return 0.0
    x = (x - x.mean()) / x.std()
    return float(np.mean(x[lag:] * x[:-lag]))


if __name__ == "__main__":
    w = load_weekly()
    print("M5 weekly demand, rescaled")
    print("  shape: %d series x %d weeks" % w.shape)
    print("  mean %.1f  sd %.1f" % (w.mean(), w.std()))
    print("\n  autocorrelation (mean across series):")
    for lag in (1, 2, 4, 8, 13, 26, 52):
        print("    lag %2d weeks  %+.3f" % (lag, np.mean([autocorrelation(r, lag) for r in w])))
    print("\n  cached -> %s" % os.path.relpath(CACHE))
