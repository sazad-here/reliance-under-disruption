"""
Generate the offline training corpus -- the 50,000 labelled periods PatchTST learns from.

This is the step that answers "how do we train the model if the simulator comes
first". The simulator IS the data source. Because we wrote the generative
process, we can look ahead from every row and record what actually happened,
which no real company dataset can give us.

Two things differ deliberately from the experiment:

  * DISRUPTIONS ARE RANDOM HERE, not the fixed calendar. The model must learn
    the general shape of a building failure, not memorise "week 9 is a spike".
    The experiment uses the fixed calendar so conditions stay comparable.

  * ORDERS COME FROM A MIX OF POLICIES, not just the optimal one. If only a
    good policy generated the data the model would never see a starved
    warehouse, and would be useless exactly when advice matters most.

Run:  python src/make_corpus.py
Out:  data/processed/corpus.parquet  (falls back to .csv without pyarrow)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from demand_source import M5DemandSource  # noqa: E402
from policy import reference_order  # noqa: E402
from simulator import (  # noqa: E402
    Disruption,
    DisruptionEvent,
    SupplyChainTwin,
    TwinParams,
)

TARGET_PERIODS = 52_000
# SHORTER EPISODES, deliberately. M5 has 273 weeks; a 260-week episode leaves
# only 14 start positions, so episodes from one series overlapped ~98%. At 100
# weeks there are 174 start positions, so windows are far more distinct.
EPISODE_LEN = 100

# Disjoint series pools. The split is by SERIES, not by episode -- episodes from
# the same M5 series share demand, so an episode-level split leaks the test set
# into training. Measured cost of getting this wrong: AUROC inflated by ~0.14.
N_SERIES = 70
_perm = np.random.default_rng(7).permutation(N_SERIES)
SERIES_SPLIT = {
    "train": set(_perm[:49].tolist()),   # 70%
    "val":   set(_perm[49:59].tolist()), # 15%
    "test":  set(_perm[59:].tolist()),   # 15%
}
# When True every episode may draw from ANY of the 70 M5 series, and the
# train/val/test division is made afterwards BY SERIES. That is what makes
# series-level cross-validation possible: with the pools baked in at
# generation time, a held-out series only ever appears in its own split, so
# folds cannot be rotated without regenerating. Leakage is still prevented --
# it just moves from generation time to split-assignment time, where every
# consumer enforces it on `m5_series`.
ALL_SERIES = os.environ.get("CORPUS_ALL_SERIES", "0") == "1"

HORIZONS = (1, 2, 3, 4)    # label: does a stockout occur within h periods?
DISRUPTION_RATE = 0.078     # fitted: DataCo severe slip (>=3 days behind schedule)
                           # rate scaled to episode-level shock frequency

KINDS = (
    (Disruption.SUPPLIER, (0.35, 0.75)),
    (Disruption.TRANSPORT, (1.0, 3.0)),
    (Disruption.DEMAND_SPIKE, (1.4, 2.2)),
    (Disruption.LEAD_TIME, (1.0, 3.0)),
    (Disruption.INVENTORY_LOSS, (0.10, 0.35)),
)


def random_calendar(rng, length):
    """Disruptions scattered at random, never two in the same period."""
    events, used = [], set()
    n = rng.binomial(length, DISRUPTION_RATE)
    for _ in range(n):
        period = int(rng.integers(2, length))
        if period in used:
            continue
        used.add(period)
        kind, (lo, hi) = KINDS[rng.integers(len(KINDS))]
        events.append(DisruptionEvent(period, kind, float(rng.uniform(lo, hi))))
    return tuple(events)


def mixed_policy(params, rng):
    """A behaviour mix, so the corpus contains healthy AND starved warehouses."""
    mode = rng.choice(
        ["optimal", "noisy", "under", "over", "naive", "random"],
        p=[0.35, 0.25, 0.12, 0.10, 0.10, 0.08],
    )
    noise_sd = params.demand_sd * 0.5

    def inner(state):
        q_star = reference_order(state, params)
        if mode == "optimal":
            q = q_star
        elif mode == "noisy":
            q = q_star + rng.normal(0, noise_sd)
        elif mode == "under":
            q = q_star * rng.uniform(0.55, 0.80)
        elif mode == "over":
            q = q_star * rng.uniform(1.20, 1.50)
        elif mode == "naive":
            q = state.demand
        else:
            q = rng.integers(0, params.max_order)
        return int(np.clip(q, 0, params.max_order))

    return inner, mode


def run_episode(episode_id, params, rng, split="train"):
    calendar = random_calendar(rng, EPISODE_LEN)
    # REAL demand: each episode replays a different M5 store-department window.
    # Falls back to the synthetic process if M5 is unavailable.
    try:
        pool = range(N_SERIES) if ALL_SERIES else SERIES_SPLIT[split]
        src = M5DemandSource(rng, EPISODE_LEN, series_pool=pool)
    except (SystemExit, FileNotFoundError, ValueError) as exc:
        # load_weekly raises SystemExit, which subclasses BaseException -- a bare
        # `except Exception` never caught it, so this documented fallback was
        # dead code and a missing M5 file crashed corpus generation instead.
        print("  [warn] M5 unavailable (%s); falling back to synthetic demand" % exc)
        src = None
    twin = SupplyChainTwin(
        params=TwinParams(**{**params.__dict__, "horizon": EPISODE_LEN}),
        calendar=calendar,
        seed=int(rng.integers(1 << 31)),
        demand_source=src,
    )
    policy, mode = mixed_policy(params, rng)

    rows, state = [], twin.observe()
    while not twin.done:
        vec = state.as_vector()
        q_star = reference_order(state, params)
        order = policy(state)
        state, cost, rec = twin.step(order)
        rows.append(
            {
                "episode": episode_id,
                "split": "pooled" if ALL_SERIES else split,
                "m5_series": -1 if src is None else src.series_idx,
                "policy_mode": mode,
                "period": rec["period"],
                "demand": vec[0],
                "inventory": vec[1],
                "on_order": vec[2],
                "lead_time": vec[3],
                "backorders": vec[4],
                "supplier_reliability": vec[5],
                "transport_ok": vec[6],
                "disruption_flag": vec[7],
                "q_star": q_star,
                "order": order,
                "cost": cost,
                "stockout": int(rec["stockout"]),
                "service_level": rec["service_level"],
            }
        )

    df = pd.DataFrame(rows)
    # THE LABEL. Look ahead h periods and record whether a stockout followed.
    # Only possible because we generated the process.
    for h in HORIZONS:
        fwd = (
            df["stockout"][::-1]
            .rolling(window=h, min_periods=1)
            .max()[::-1]
            .shift(-1)
        )
        df["stockout_within_%d" % h] = fwd.fillna(0).astype(int)
    # rows too close to the end have an incomplete look-ahead window
    df["label_valid"] = (df["period"] < EPISODE_LEN - max(HORIZONS)).astype(int)
    return df


def main():
    params = TwinParams.from_fitted()
    rng = np.random.default_rng(20260919)
    n_episodes = TARGET_PERIODS // EPISODE_LEN

    print("=" * 66)
    print("BUILDING THE OFFLINE CORPUS")
    print("=" * 66)
    print("  %d episodes x %d periods = %d rows" % (n_episodes, EPISODE_LEN, TARGET_PERIODS))
    print("  random disruptions (NOT the fixed experimental calendar)")
    print("  mixed policies, so the model sees good and bad warehouses\n")

    # allocate episodes to splits in the same 70/15/15 proportion
    frames = []
    for i in range(n_episodes):
        frac = i / n_episodes
        split = "train" if frac < 0.70 else ("val" if frac < 0.85 else "test")
        frames.append(run_episode(i, params, rng, split=split))
        if (i + 1) % 100 == 0:
            print("  %3d / %d episodes" % (i + 1, n_episodes))

    df = pd.concat(frames, ignore_index=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    try:
        path = os.path.join(out_dir, "corpus.parquet")
        df.to_parquet(path, index=False)
    except Exception:
        path = os.path.join(out_dir, "corpus.csv")
        df.to_csv(path, index=False)

    valid = df[df["label_valid"] == 1]
    print("\n" + "=" * 66)
    print("CORPUS SUMMARY")
    print("=" * 66)
    print("  rows total            %8d" % len(df))
    print("  rows with valid label %8d" % len(valid))
    print("  columns               %8d" % df.shape[1])
    print("\n  label balance (this is what PatchTST predicts):")
    for h in HORIZONS:
        rate = valid["stockout_within_%d" % h].mean()
        print("    stockout within %d period(s):  %5.1f%% positive" % (h, 100 * rate))
    print("\n  policy mix:")
    for mode, n in df.groupby("policy_mode")["episode"].nunique().items():
        print("    %-10s %3d episodes" % (mode, n))
    print("\n  disruption periods: %.1f%% of rows" % (100 * df["disruption_flag"].mean()))
    print("  stockout periods:   %.1f%% of rows" % (100 * df["stockout"].mean()))
    print("\n  written -> %s" % os.path.relpath(path))


if __name__ == "__main__":
    main()
