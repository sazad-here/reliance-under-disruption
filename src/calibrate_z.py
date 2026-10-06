"""
Find the safety factor z that actually minimises cost in THIS system.

The textbook z comes from the single-period newsvendor critical ratio
p/(p+h). That is myopic: it assumes each period stands alone. Our system
carries backorders forward, so under-ordering compounds and the true optimal
base-stock level sits above the myopic one.

Leaving the myopic z in place means a participant who consistently over-orders
outperforms the "correct" answer -- which would make decision error, RAIR and
RSR measure the wrong thing. So z is calibrated by grid search on the simulator
and the result is reported in the paper as a deliberate methodological choice.

Run:  python src/calibrate_z.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import policy as P  # noqa: E402
from simulator import SupplyChainTwin, TwinParams  # noqa: E402

N_SEEDS = 300


def cost_at_z(z: float, params: TwinParams) -> float:
    """Mean episode cost for a base-stock agent using safety factor z."""
    total = []
    for seed in range(N_SEEDS):
        twin = SupplyChainTwin(params=params, seed=seed)
        state = twin.observe()
        while not twin.done:
            mean_periods, var_lead = P.lead_time_moments(params, state.lead_time)
            level = params.demand_mean * mean_periods + z * math.sqrt(
                mean_periods * params.demand_sd ** 2
                + params.demand_mean ** 2 * var_lead
            )
            q = int(max(0, min(round(level - state.inventory_position), params.max_order)))
            state, _, _ = twin.step(q)
        total.append(twin.total_cost())
    return float(np.mean(total))


def main():
    params = TwinParams.from_fitted()
    myopic = P.myopic_safety_factor(params)

    print("=" * 66)
    print("CALIBRATING THE SAFETY FACTOR  (%d episodes per candidate)" % N_SEEDS)
    print("=" * 66)
    print("  myopic newsvendor z = %.3f  (critical ratio %.3f)\n"
          % (myopic, params.shortage_cost / (params.shortage_cost + params.holding_cost)))

    grid = [round(z, 2) for z in np.arange(0.6, 3.01, 0.15)]
    results = [(z, cost_at_z(z, params)) for z in grid]
    best_z, best_cost = min(results, key=lambda r: r[1])

    print("  %-8s %12s %10s" % ("z", "mean cost", "vs best"))
    print("  " + "-" * 32)
    for z, c in results:
        mark = "  <-- best" if z == best_z else ""
        print("  %-8.2f %12.0f %9.3fx%s" % (z, c, c / best_cost, mark))

    # refine around the winner
    fine = [round(z, 3) for z in np.arange(best_z - 0.15, best_z + 0.16, 0.03)]
    fine_results = [(z, cost_at_z(z, params)) for z in fine]
    best_z, best_cost = min(fine_results, key=lambda r: r[1])

    myopic_cost = cost_at_z(myopic, params)
    print("\n  refined optimum:  z = %.3f   mean cost %.0f" % (best_z, best_cost))
    print("  myopic z:         z = %.3f   mean cost %.0f  (%.1f%% worse)"
          % (myopic, myopic_cost, 100 * (myopic_cost / best_cost - 1)))

    out = {
        "calibrated_z": round(best_z, 3),
        "myopic_z": round(myopic, 3),
        "mean_cost_calibrated": round(best_cost, 1),
        "mean_cost_myopic": round(myopic_cost, 1),
        "n_seeds": N_SEEDS,
        "note": (
            "z calibrated by grid search on the simulator because the myopic "
            "newsvendor ratio understates the base-stock level in a system that "
            "carries backorders forward. Report this choice in the method."
        ),
    }
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "safety_factor.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n  written -> data/processed/safety_factor.json")
    print("\n  Set SAFETY_FACTOR_OVERRIDE = %.3f in policy.py to adopt it." % best_z)


if __name__ == "__main__":
    main()


def calibrate_delta(params: TwinParams, tolerance: float = 0.05) -> int:
    """Empirical delta: how far from Q* before expected cost rises `tolerance`.

    The analytical newsvendor cost curve no longer applies once z is calibrated,
    and it returns an absurd +/-152 band that would score almost any order as
    correct. Measured on the simulator instead, same as z.
    """
    import policy as pol

    def cost_with_bias(bias):
        total = []
        for seed in range(N_SEEDS):
            twin = SupplyChainTwin(params=params, seed=seed)
            state = twin.observe()
            while not twin.done:
                q = pol.reference_order(state, params) + bias
                state, _, _ = twin.step(int(np.clip(q, 0, params.max_order)))
            total.append(twin.total_cost())
        return float(np.mean(total))

    base = cost_with_bias(0)
    limit = base * (1 + tolerance)
    lo = hi = 0
    for b in range(5, 201, 5):
        if cost_with_bias(b) <= limit:
            hi = b
        else:
            break
    for b in range(5, 201, 5):
        if cost_with_bias(-b) <= limit:
            lo = b
        else:
            break
    return min(lo, hi), lo, hi, base
