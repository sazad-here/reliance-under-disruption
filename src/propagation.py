"""
Does one misplaced decision actually cascade? Measure it, then fix it.

Section 02 and Figure 1 of the plan assert that a bad order in week 7 produces a
stockout in week 9 and backorders persisting into week 12 -- three to four periods
of consequence from one decision. That claim IS the contribution: it is what
separates this design from every prior reliance study, which uses independent
trials.

A first measurement gave about ONE period. This script measures propagation
properly (many seeds, many injection points, both directions of error) and tests
which parameters actually lengthen it.

Run:  python src/propagation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from policy import ReferencePolicy, reference_order  # noqa: E402
from simulator import SupplyChainTwin, TwinParams  # noqa: E402

N_SEEDS = 150
INJECT_AT = (5, 7, 9, 11, 13, 15, 17, 19)
COST_EPS = 1.0  # a period counts as "affected" if cost differs by more than this


def run(policy, seed, params):
    twin = SupplyChainTwin(params=params, seed=seed)
    state = twin.observe()
    while not twin.done:
        state, _, _ = twin.step(policy(state))
    return twin


def shocked_policy(params, week, shortfall):
    """Base-stock everywhere, except one deliberately bad order."""
    def inner(state):
        q = reference_order(state, params)
        if state.period == week:
            return int(np.clip(q - shortfall, 0, params.max_order))
        return int(np.clip(q, 0, params.max_order))
    return inner


def measure(params, shortfall=150, label=""):
    """Average number of later periods whose cost changed, and the total penalty."""
    spans, penalties, peak_lags = [], [], []
    for seed in range(N_SEEDS):
        clean = run(ReferencePolicy(params), seed, params)
        for week in INJECT_AT:
            if week >= params.horizon - 4:
                continue
            dirty = run(shocked_policy(params, week, shortfall), seed, params)
            affected, peak, peak_lag = 0, 0.0, 0
            for i in range(week + 1, params.horizon):
                d = dirty.history[i]["cost"] - clean.history[i]["cost"]
                if abs(d) > COST_EPS:
                    affected += 1
                if abs(d) > peak:
                    peak, peak_lag = abs(d), i - week
            spans.append(affected)
            penalties.append(dirty.total_cost() - clean.total_cost())
            peak_lags.append(peak_lag)
    return {
        "label": label,
        "mean_periods_affected": round(float(np.mean(spans)), 2),
        "median_periods_affected": float(np.median(spans)),
        "pct_reaching_3_periods": round(float(np.mean(np.array(spans) >= 3) * 100), 1),
        "mean_penalty": round(float(np.mean(penalties)), 1),
        "mean_peak_lag": round(float(np.mean(peak_lags)), 2),
    }


def main():
    base = TwinParams.from_fitted()

    print("=" * 76)
    print("PROPAGATION: how many later periods does ONE bad order change?")
    print("=" * 76)
    print("  %d seeds x %d injection points, shortfall 150 units\n"
          % (N_SEEDS, len(INJECT_AT)))

    results = []

    r = measure(base, label="fitted parameters (current)")
    results.append(r)
    print("  %-38s %6s %8s %9s %8s" % ("configuration", "mean", "median", ">=3 per", "penalty"))
    print("  " + "-" * 74)
    print("  %-38s %6.2f %8.0f %8.1f%% %8.0f"
          % (r["label"], r["mean_periods_affected"], r["median_periods_affected"],
             r["pct_reaching_3_periods"], r["mean_penalty"]))

    # --- candidate levers, each tested in isolation -------------------------
    variants = [
        ("longer lead time (nominal 2)",
         dict(lead_time_nominal=2)),
        ("heavier delay tail",
         dict(delay_pmf=(0.35, 0.30, 0.20, 0.15))),
        ("shortage 10x holding",
         dict(shortage_cost=10.0)),
        ("shortage 20x holding",
         dict(shortage_cost=20.0)),
        ("leaner opening stock",
         dict(initial_inventory=300)),
        ("lead 2 + heavy tail",
         dict(lead_time_nominal=2, delay_pmf=(0.35, 0.30, 0.20, 0.15))),
        ("lead 2 + heavy tail + shortage 10x",
         dict(lead_time_nominal=2, delay_pmf=(0.35, 0.30, 0.20, 0.15),
              shortage_cost=10.0)),
    ]

    for label, kw in variants:
        p = TwinParams.from_fitted(**kw)
        r = measure(p, label=label)
        results.append(r)
        flag = "  <-- reaches the claim" if r["mean_periods_affected"] >= 3 else ""
        print("  %-38s %6.2f %8.0f %8.1f%% %8.0f%s"
              % (label, r["mean_periods_affected"], r["median_periods_affected"],
                 r["pct_reaching_3_periods"], r["mean_penalty"], flag))

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    best = max(results, key=lambda r: r["mean_periods_affected"])
    baseline = results[0]

    print("  Figure 1 claims 3-4 periods of consequence from one decision.")
    print("  Current fitted parameters deliver %.2f." % baseline["mean_periods_affected"])
    if baseline["mean_periods_affected"] >= 3:
        print("  -> The claim already holds. No change needed.")
    elif best["mean_periods_affected"] >= 3:
        print("  -> The claim does NOT hold as configured, but '%s'" % best["label"])
        print("     reaches %.2f periods. Adopt it, or soften Figure 1."
              % best["mean_periods_affected"])
    else:
        print("  -> No tested configuration reaches 3 periods (best: %s at %.2f)."
              % (best["label"], best["mean_periods_affected"]))
        print("     SOFTEN THE CLAIM. Report the measured span honestly; the")
        print("     contribution survives at 2 periods, it just cannot be oversold.")
    print("\n  NOTE: whatever is chosen must be re-fitted against the data, not")
    print("  tuned freely -- a parameter picked to make the story work is not")
    print("  calibration. Any change here reopens z, delta and the corpus.")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "propagation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\n  written -> results/propagation.json")


if __name__ == "__main__":
    main()
