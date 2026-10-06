"""
Choose the PROPAGATING and ABSORBED error weeks by measured consequence.

WHY. The original design called every disruption week and the week before it
"propagating". Measured on the participants' real demand (interface/world.py),
that set is heterogeneous: one under-order in week 16 costs +726, in week 12
-206, and some "absorbed" weeks (7, 18) cause more stockouts than most
propagating ones. H3 is about consequences, so the positions are now defined by
the consequence a single accepted error actually has in this twin.

METHOD (no circularity).
  SELECT on lead-time draws 0-99: for each week 2..25, the mean extra cost and
  extra stockout weeks when one 100-unit under-order is accepted and every
  other week follows the reference policy. Rank weeks by the mean of the two
  ranks. The top 8 are PROPAGATING, the bottom 8 ABSORBED, the rest NEUTRAL
  (never an error week). Weeks 0-1 are excluded: the risk model has almost no
  history there and the warehouse is still settling.
  VALIDATE on draws 100-299, which were never used to select: with a
  reference follower AND with a noisy follower (reference + N(0, 40) units,
  a participant who does not order perfectly). Gate G0 requires propagating
  errors to cost more and cause >= 1.5x the extra stockout weeks.

Why 8 + 8: the 60 % condition needs 12 errors, split 6 + 6, at distinct weeks.

Run:  python src/placement_design.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from interface.world import make_twin  # noqa: E402
from policy import reference_order  # noqa: E402
from simulator import TwinParams  # noqa: E402

N_WEEKS, SIZE, K = 30, 100, 8
CANDIDATES = range(2, 26)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "placement_design.json")


def episode(params, seed, week=None, shift=0, noise_sd=0.0):
    twin = make_twin(params, seed, "main")
    st = twin.reset()
    rng = np.random.default_rng(10_000 + seed)   # follower noise, identical across arms
    costs, stock = [], []
    for t in range(N_WEEKS):
        q = reference_order(st, params) + (rng.normal(0, noise_sd) if noise_sd else 0.0)
        if t == week:
            q += shift
        st, c, rec = twin.step(int(np.clip(round(q), 0, params.max_order)))
        costs.append(c); stock.append(int(rec["stockout"]))
    return np.array(costs), np.array(stock)


def impact(params, seeds, noise_sd=0.0):
    rows = []
    for seed in seeds:
        bc, bs = episode(params, seed, noise_sd=noise_sd)
        for w in CANDIDATES:
            c, s = episode(params, seed, w, -SIZE, noise_sd)
            rows.append((w, float((c - bc).sum()), float(s.sum() - bs.sum())))
    return pd.DataFrame(rows, columns=["week", "dcost", "dstock"]).groupby("week").mean()


def contrast(tab, prop, absb):
    p, a = tab.loc[prop], tab.loc[absb]
    ratio = p.dstock.mean() / a.dstock.mean() if a.dstock.mean() > 0 else float("inf")
    return {"prop_cost": float(p.dcost.mean()), "abs_cost": float(a.dcost.mean()),
            "prop_stock": float(p.dstock.mean()), "abs_stock": float(a.dstock.mean()),
            "stock_ratio": float(ratio),
            "passes": bool(p.dcost.mean() > a.dcost.mean() and ratio >= 1.5)}


def main():
    params = TwinParams.from_fitted()
    sel = impact(params, range(0, 100))
    score = (sel.dcost.rank(ascending=False) + sel.dstock.rank(ascending=False)) / 2
    order = score.sort_values().index.tolist()
    prop, absb = sorted(order[:K]), sorted(order[-K:])
    print("PROPAGATING weeks:", prop)
    print("ABSORBED weeks:   ", absb)

    res = {"propagating": prop, "absorbed": absb, "size": SIZE,
           "selection": {"seeds": "0-99", "per_week": sel.round(3).to_dict("index")}}
    for label, sd in (("reference_follower", 0.0), ("noisy_follower", 40.0)):
        val = impact(params, range(100, 300), noise_sd=sd)
        c = contrast(val, prop, absb)
        res["validation_" + label] = c
        print("validation (%s, draws 100-299): extra cost %+.0f vs %+.0f | extra stockout weeks %.3f vs %.3f"
              " (%.2fx) -> %s" % (label, c["prop_cost"], c["abs_cost"], c["prop_stock"], c["abs_stock"],
                                  c["stock_ratio"], "PASS" if c["passes"] else "FAIL"))
    json.dump(res, open(OUT, "w"), indent=2)
    print("written ->", OUT)


if __name__ == "__main__":
    main()
