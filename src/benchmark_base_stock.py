"""
Does the twin reproduce textbook inventory theory? Check it before anyone plays it.

The reference policy is a periodic-review order-up-to (base-stock) rule with
backorders. For that rule the steady state is known in closed form, given demand
over the risk period R = L + 1 with mean mu_R and standard deviation sigma_R and
the order-up-to level S = mu_R + z sigma_R:

    P(stockout in a week)      = 1 - Phi(z)
    E[backorders at week end]  = sigma_R G(z)            G(z) = phi(z) - z (1 - Phi(z))
    E[on hand at week end]     = sigma_R (z + G(z))
    E[cost per week]           = h E[on hand] + p E[backorders]

These are the standard normal-approximation results for a base-stock policy
(e.g. Axsater, Inventory Control, Springer 2015, doi 10.1007/978-3-319-15729-0). They are exact for normal demand and a
fixed lead time, so a twin whose accounting is right must land on them. With the
fitted Olist lead times orders can cross, and the paper's base-stock formula (Eq. 2) is itself an
approximation; that case measures how good the approximation is, not the twin.

Disruptions are switched off and seasonality flattened: the benchmark has neither.
Demand is the twin's own negative binomial at the M5 mean and SD, so the residual
gap in the fixed-lead-time case is the normal approximation to that skew.

Run:  python src/benchmark_base_stock.py
Out:  results/twin_benchmark.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

import policy  # noqa: E402
from policy import ReferencePolicy, lead_time_moments, order_up_to_level  # noqa: E402
from simulator import SupplyChainTwin, TwinParams  # noqa: E402

N_SEEDS = 20
PERIODS = 2000
WARM_UP = 50
Z_VALUES = (0.967, 1.600)  # the newsvendor value and the calibrated value


def analytic(params: TwinParams, z: float) -> dict:
    mean_periods, var_lead = lead_time_moments(params, params.lead_time_nominal)
    sigma = math.sqrt(mean_periods * params.demand_sd ** 2
                      + params.demand_mean ** 2 * var_lead)
    loss = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
    backorders = sigma * loss
    on_hand = sigma * (z + loss)
    return {
        "stockout_rate": float(1 - stats.norm.cdf(z)),
        "backorders": float(backorders),
        "on_hand": float(on_hand),
        "cost": float(params.holding_cost * on_hand + params.shortage_cost * backorders),
    }


def exact_negbin(params: TwinParams, z: float) -> dict:
    """The same quantities computed exactly for the twin's own demand law.

    Fixed lead time only: risk-period demand is then a sum of L + 1 iid negative
    binomials with a common p, which is itself negative binomial. This removes the
    normal approximation, so any remaining gap would be the twin's accounting.
    """
    mu, var = params.demand_mean, params.demand_sd ** 2
    r = mu * mu / (var - mu)
    d = stats.nbinom((params.lead_time_nominal + 1) * r, r / (r + mu))
    k = np.arange(0, int(d.ppf(1 - 1e-12)) + 1)
    pm = d.pmf(k)
    s = round(order_up_to_level(params, params.lead_time_nominal))  # orders are whole units
    backorders = float((pm * np.maximum(k - s, 0)).sum())
    on_hand = float((pm * np.maximum(s - k, 0)).sum())
    return {
        "stockout_rate": float(pm[k > s].sum()),
        "backorders": backorders,
        "on_hand": on_hand,
        "cost": params.holding_cost * on_hand + params.shortage_cost * backorders,
    }


def simulate(params: TwinParams, seed: int) -> dict:
    twin = SupplyChainTwin(params=params, calendar=(), seed=seed)
    agent = ReferencePolicy(params)
    state = twin.observe()
    while not twin.done:
        state, _, _ = twin.step(agent(state))
    h = twin.history[WARM_UP:]
    if any(r["order"] >= params.max_order for r in h):
        raise RuntimeError("order hit max_order; the benchmark would be censored")
    return {
        "stockout_rate": float(np.mean([r["stockout"] for r in h])),
        "backorders": float(np.mean([r["backorders_close"] for r in h])),
        "on_hand": float(np.mean([r["inventory_close"] for r in h])),
        "cost": float(np.mean([r["cost"] for r in h])),
        "demand": float(np.mean([r["demand"] for r in h])),
    }


def run_case(name: str, params: TwinParams, z: float) -> dict:
    saved = policy.SAFETY_FACTOR_OVERRIDE
    policy.SAFETY_FACTOR_OVERRIDE = z
    try:
        # start at the order-up-to level, so the warm-up only removes the transient
        params = TwinParams(**{**params.__dict__,
                               "initial_inventory": int(round(order_up_to_level(params, params.lead_time_nominal)))})
        runs = [simulate(params, seed) for seed in range(N_SEEDS)]
        theory = analytic(params, z)
        fixed_lead = params.delay_pmf[0] == 1.0
        exact = exact_negbin(params, z) if fixed_lead else None
    finally:
        policy.SAFETY_FACTOR_OVERRIDE = saved
    out = {"case": name, "z": z, "seeds": N_SEEDS, "weeks_per_seed": PERIODS - WARM_UP,
           "lead_time_nominal": params.lead_time_nominal, "delay_pmf": list(params.delay_pmf),
           # every case reuses seeds 0..N-1, so they share demand draws; this is their mean
           "realised_mean_demand": float(np.mean([r["demand"] for r in runs])),
           "metrics": {}}
    for k in theory:
        vals = np.array([r[k] for r in runs])
        se = vals.std(ddof=1) / math.sqrt(len(vals))
        m = {
            "simulated": float(vals.mean()),
            "ci95": [float(vals.mean() - 1.96 * se), float(vals.mean() + 1.96 * se)],
            "normal_approx": theory[k],
            "rel_diff_normal": float(vals.mean() / theory[k] - 1),
        }
        if exact:
            m["exact_negbin"] = exact[k]
            m["rel_diff_exact"] = float(vals.mean() / exact[k] - 1)
        out["metrics"][k] = m
    return out


def main():
    fitted = TwinParams.from_fitted(horizon=PERIODS, seasonality=(1.0,) * 7)
    fixed = TwinParams(**{**fitted.__dict__, "delay_pmf": (1.0, 0.0, 0.0, 0.0)})
    cases = []
    for z in Z_VALUES:
        cases.append(run_case("fixed lead time (exact theory)", fixed, z))
        cases.append(run_case("fitted Olist lead time (Eq. 3 approximation)", fitted, z))

    print("BASE-STOCK BENCHMARK  (%d seeds x %d weeks, no disruptions)" % (N_SEEDS, PERIODS - WARM_UP))
    for c in cases:
        print("\n  %s, z = %.3f   (realised mean demand %.2f)" % (c["case"], c["z"], c["realised_mean_demand"]))
        for k, m in c["metrics"].items():
            line = "    %-14s sim %8.3f  [%8.3f, %8.3f]   normal %8.3f %+6.1f %%" % (
                k, m["simulated"], m["ci95"][0], m["ci95"][1], m["normal_approx"], 100 * m["rel_diff_normal"])
            if "exact_negbin" in m:
                inside = m["ci95"][0] <= m["exact_negbin"] <= m["ci95"][1]
                line += "   exact %8.3f %+6.1f %% %s" % (m["exact_negbin"], 100 * m["rel_diff_exact"],
                                                         "in CI" if inside else "OUTSIDE CI")
            print(line)

    path = os.path.join(os.path.dirname(__file__), "..", "results", "twin_benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"demand_mean": fitted.demand_mean, "demand_sd": fitted.demand_sd,
                   "holding_cost": fitted.holding_cost, "shortage_cost": fitted.shortage_cost,
                   "cases": cases}, fh, indent=2)
    print("\n  written -> results/twin_benchmark.json")


if __name__ == "__main__":
    main()
