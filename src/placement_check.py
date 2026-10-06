"""
Manipulation-validity check: do "propagating" error positions actually cost more?

H3 assumes an AI error at disruption onset (a PROPAGATING position, advice.py)
does more lasting damage than the same error in a calm week (ABSORBED). This
measures it directly, separating placement from direction -- which the current
advice engine confounds (propagating errors are always under-orders).

For every main-trial week, on the same demand paths, one order is replaced by
Q* - 100 (under) or Q* + 100 (over); every other week follows the reference
policy, i.e. a participant who accepts one bad recommendation and then orders
correctly. 100 units is the mean size of the scripted AI errors.

Run:  python src/placement_check.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from interface.advice import placement_of  # noqa: E402
from interface.world import make_twin  # noqa: E402
from policy import reference_order  # noqa: E402
from simulator import TwinParams  # noqa: E402

N_SEEDS, SIZE, N_TRIALS = 200, 100, 30
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "placement_check.json")


def episode(params, seed, week=None, shift=0):
    # The participants' own world: shared real M5 demand (interface/world.py).
    twin = make_twin(params, seed, "main")
    st = twin.reset()
    costs, stock = [], []
    for t in range(N_TRIALS):
        q = reference_order(st, params)
        if t == week:
            q += shift
        st, c, rec = twin.step(int(np.clip(q, 0, params.max_order)))
        costs.append(c); stock.append(int(rec["stockout"]))
    return np.array(costs), np.array(stock)


def main():
    params = TwinParams.from_fitted()
    rows = []
    for seed in range(N_SEEDS):
        base_c, base_s = episode(params, seed)
        for week in range(N_TRIALS - 4):          # leave room to observe the tail
            for direction, shift in (("under", -SIZE), ("over", SIZE)):
                c, s = episode(params, seed, week, shift)
                dc = c - base_c
                rows.append({
                    "placement": placement_of(week), "direction": direction,
                    "extra_cost": float(dc.sum()),
                    "extra_stockout_weeks": int(s.sum() - base_s.sum()),
                    "span": int((np.abs(dc) > 1.0).sum()),
                })
    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby(["placement", "direction"]).agg(
        n=("extra_cost", "size"), extra_cost=("extra_cost", "mean"),
        extra_stockout_weeks=("extra_stockout_weeks", "mean"), span=("span", "mean"))
    print(tab.round(2).to_string())

    def cell(p, d):
        return tab.loc[(p, d)]
    eff_place_under = cell("propagating", "under").extra_cost - cell("absorbed", "under").extra_cost
    eff_dir_absorbed = cell("absorbed", "under").extra_cost - cell("absorbed", "over").extra_cost
    confounded = cell("propagating", "under").extra_cost - cell("absorbed", "over").extra_cost
    print("\nextra cost, propagating vs absorbed, BOTH under-orders : %+.0f  <- placement alone" % eff_place_under)
    print("extra cost, under vs over, BOTH absorbed               : %+.0f  <- direction alone" % eff_dir_absorbed)
    print("extra cost, current design (prop-under vs abs-over)     : %+.0f  <- what H3 now compares" % confounded)
    json.dump({"size": SIZE, "n_seeds": N_SEEDS,
               "table": {f"{p}/{d}": r for (p, d), r in tab.round(3).to_dict("index").items()},
               "placement_effect_under": eff_place_under,
               "direction_effect_absorbed": eff_dir_absorbed,
               "current_design_contrast": confounded},
              open(OUT, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
