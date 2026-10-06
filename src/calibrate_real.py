"""
Recalibrate z and delta against REAL M5 demand.

Both were fitted on the synthetic negative-binomial process. Real demand has a
different variance (sd 41 vs 36.8) and strong autocorrelation, so the cost-
minimising safety factor moves. These two numbers define Q* for every trial in
the experiment, so they have to be calibrated on the demand the participants
will actually face.

Run:  python src/calibrate_real.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import policy as P  # noqa: E402
from demand_source import M5DemandSource, load_weekly  # noqa: E402
from simulator import SupplyChainTwin, TwinParams  # noqa: E402

N_EPISODES = 250
HORIZON = 30


def episode_cost(z, params, seed, weekly, bias=0):
    """One 30-period episode on a real demand window, base-stock with factor z."""
    rng = np.random.default_rng(seed)
    src = M5DemandSource(rng, HORIZON, weekly=weekly)
    twin = SupplyChainTwin(params=params, seed=seed, demand_source=src)
    state = twin.observe()
    while not twin.done:
        mean_periods, var_lead = P.lead_time_moments(params, state.lead_time)
        level = params.demand_mean * mean_periods + z * math.sqrt(
            mean_periods * params.demand_sd ** 2
            + params.demand_mean ** 2 * var_lead
        )
        q = int(np.clip(round(level - state.inventory_position) + bias,
                        0, params.max_order))
        state, _, _ = twin.step(q)
    return twin.total_cost()


def mean_cost(z, params, weekly, bias=0):
    return float(np.mean([episode_cost(z, params, s, weekly, bias)
                          for s in range(N_EPISODES)]))


def main():
    params = TwinParams.from_fitted()
    weekly = load_weekly()

    print("=" * 70)
    print("RECALIBRATING ON REAL M5 DEMAND  (%d episodes per candidate)" % N_EPISODES)
    print("=" * 70)
    print("  synthetic-demand values were: z = 1.260, delta = +/-45\n")

    grid = [round(z, 2) for z in np.arange(0.6, 2.81, 0.2)]
    results = [(z, mean_cost(z, params, weekly)) for z in grid]
    best_z, best_cost = min(results, key=lambda r: r[1])

    print("  %-8s %12s %10s" % ("z", "mean cost", "vs best"))
    print("  " + "-" * 32)
    for z, c in results:
        mark = "  <-- best" if z == best_z else ""
        print("  %-8.2f %12.0f %9.3fx%s" % (z, c, c / best_cost, mark))

    fine = [round(z, 3) for z in np.arange(max(0.1, best_z - 0.2), best_z + 0.21, 0.05)]
    fine_results = [(z, mean_cost(z, params, weekly)) for z in fine]
    best_z, best_cost = min(fine_results, key=lambda r: r[1])
    myopic = P.myopic_safety_factor(params)
    myopic_cost = mean_cost(myopic, params, weekly)

    print("\n  refined optimum:  z = %.3f   mean cost %.0f" % (best_z, best_cost))
    print("  myopic newsvendor z = %.3f   mean cost %.0f  (%.1f%% worse)"
          % (myopic, myopic_cost, 100 * (myopic_cost / best_cost - 1)))
    print("  previous (synthetic) z = 1.260  mean cost %.0f"
          % mean_cost(1.260, params, weekly))

    # ---- delta, measured the same way -------------------------------------
    print("\n  calibrating delta at the new z ...")
    base = mean_cost(best_z, params, weekly)
    deltas = {}
    for tol in (0.05, 0.10):
        limit = base * (1 + tol)
        lo = hi = 0
        for b in range(5, 161, 5):
            if mean_cost(best_z, params, weekly, bias=b) <= limit:
                hi = b
            else:
                break
        for b in range(5, 161, 5):
            if mean_cost(best_z, params, weekly, bias=-b) <= limit:
                lo = b
            else:
                break
        deltas[tol] = min(lo, hi)
        print("    tolerance %3.0f%%  ->  delta = +/- %3d  (down %d / up %d)"
              % (tol * 100, deltas[tol], lo, hi))

    out = {
        "demand": "real M5 replay",
        "z_calibrated": round(best_z, 3),
        "z_previous_synthetic": 1.260,
        "z_myopic": round(myopic, 3),
        "mean_cost": round(base, 1),
        "delta_5pct": deltas[0.05],
        "delta_10pct": deltas[0.10],
        "n_episodes": N_EPISODES,
    }
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "processed", "calibration_real.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 70)
    print("  ADOPT:  SAFETY_FACTOR_OVERRIDE = %.3f" % best_z)
    print("          DELTA_CALIBRATED = {0.05: %d, 0.10: %d}" % (deltas[0.05], deltas[0.10]))
    print("  written -> data/processed/calibration_real.json")


if __name__ == "__main__":
    main()
