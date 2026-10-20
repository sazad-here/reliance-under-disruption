"""
How long does ONE bad order keep costing, depending on how the decision-maker
recovers? The persistence result in docs/RESULTS.md, section 9, made reproducible.

A 150-unit short order is injected at one week. Before and after it, the
decision-maker follows one of three recovery behaviours, identically in the
clean and the shocked run:
    base-stock    orders Q* (self-correcting)
    noisy         Q* + N(0, 0.6 sigma_D)
    under-15      0.85 * Q*  (a persistent bias, e.g. over-relying on an
                  aid that under-orders)
"Weeks affected" = later weeks whose cost differs from the clean run by > 1.

The 4.30 figure was computed in an earlier session with no script kept. This
run reproduces the table on the synthetic demand it was measured on, then
measures it on the real M5 demand participants now play (interface/world.py),
which is the number the paper should cite.

Run:  python src/propagation_recovery.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from interface.world import make_twin  # noqa: E402
from policy import reference_order  # noqa: E402
from simulator import SupplyChainTwin, TwinParams  # noqa: E402

N_SEEDS, SHORT, INJECT_AT = 150, 150, (5, 7, 9, 11, 13, 15, 17, 19)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "propagation_recovery.json")


def behaviour(name, params):
    sd = 0.6 * params.demand_sd

    def act(state, rng):
        q = reference_order(state, params)
        if name == "noisy":
            q += rng.normal(0, sd)
        elif name == "under-15":
            q *= 0.85
        return q
    return act


def run(world, params, seed, act, week=None):
    twin = (make_twin(params, seed, "main") if world == "m5"
            else SupplyChainTwin(params=params, seed=seed))
    st = twin.reset()
    rng = np.random.default_rng(50_000 + seed)
    costs = []
    for t in range(params.horizon):
        q = act(st, rng)
        if t == week:
            q -= SHORT
        st, c, _ = twin.step(int(np.clip(round(q), 0, params.max_order)))
        costs.append(c)
    return np.array(costs)


def main():
    params = TwinParams.from_fitted()
    res = {}
    for world in ("synthetic", "m5"):
        print("%s demand" % ("SYNTHETIC (what the 4.30 was measured on)" if world == "synthetic"
                             else "REAL M5 (what participants play)"))
        for name in ("base-stock", "noisy", "under-15"):
            act = behaviour(name, params)
            spans = []
            for seed in range(N_SEEDS):
                clean = run(world, params, seed, act)
                for w in INJECT_AT:
                    d = run(world, params, seed, act, w) - clean
                    spans.append(int((np.abs(d[w + 1:]) > 1.0).sum()))
            spans = np.array(spans)
            r = {"mean_weeks_affected": round(float(spans.mean()), 2),
                 "pct_3_or_more": round(float((spans >= 3).mean() * 100), 1)}
            res["%s/%s" % (world, name)] = r
            print("  %-11s weeks affected %.2f | >= 3 weeks %5.1f%%"
                  % (name, r["mean_weeks_affected"], r["pct_3_or_more"]))
    json.dump(res, open(OUT, "w"), indent=2)
    print("written ->", OUT)


if __name__ == "__main__":
    main()
