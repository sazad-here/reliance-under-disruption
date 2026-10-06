"""
Is the task a real decision problem?

The plan's top risk is that the pilot shows no usable variance -- everyone either
defers to the AI or nobody needs it. That is cheap to check now and expensive to
discover in week 3, so this script checks it against the simulator directly.

Run:  python src/diagnose.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from policy import NaivePolicy, ReferencePolicy, cost_derived_delta, reference_order
from simulator import SupplyChainTwin, TwinParams

N_SEEDS = 200


def episode_cost(policy, seed, params):
    twin = SupplyChainTwin(params=params, seed=seed)
    state = twin.observe()
    while not twin.done:
        state, _, _ = twin.step(policy(state))
    return twin


def biased_policy(params, bias):
    """A participant who systematically under- or over-orders by `bias` units."""

    def inner(state):
        return int(np.clip(reference_order(state, params) + bias, 0, params.max_order))

    return inner


def show_one_trial(params):
    print("=" * 74)
    print("WHAT A PARTICIPANT SEES  (one episode, seed 42)")
    print("=" * 74)
    twin = SupplyChainTwin(params=params, seed=42)
    state = twin.observe()
    delta = cost_derived_delta(params)
    print("  wk  stock  pipe  back    Q*   disruption        demand   cost  note")
    print("  " + "-" * 70)
    while not twin.done and twin.t < 14:
        q_star = reference_order(state, params)
        disruption = state.disruption.value if state.disruption.value != "none" else ""
        row = "  %2d  %5d %5d %5d  %4d   %-16s" % (
            state.period, state.inventory, state.on_order,
            state.backorders, q_star, disruption,
        )
        state, cost, rec = twin.step(q_star)
        note = "STOCKOUT" if rec["stockout"] else ""
        print("%s %6d %6.0f  %s" % (row, rec["demand"], cost, note))
    print("  ...")
    print("\n  Correct band is Q* +/- %d units (5%% cost tolerance).\n" % delta)


def main():
    params = TwinParams.from_fitted()

    show_one_trial(params)

    print("=" * 74)
    print("IS THERE ROOM FOR THE AI TO HELP?  (%d episodes each)" % N_SEEDS)
    print("=" * 74)

    optimal = [episode_cost(ReferencePolicy(params), s, params).total_cost() for s in range(N_SEEDS)]
    naive = [episode_cost(NaivePolicy(params), s, params).total_cost() for s in range(N_SEEDS)]

    print("  %-34s %9s %9s" % ("policy", "mean cost", "sd"))
    print("  " + "-" * 54)
    print("  %-34s %9.0f %9.0f" % ("base-stock (the correct answer)", np.mean(optimal), np.std(optimal)))
    print("  %-34s %9.0f %9.0f" % ("naive (order last week's demand)", np.mean(naive), np.std(naive)))
    print("  %-34s %9.1fx" % ("penalty for being naive", np.mean(naive) / np.mean(optimal)))

    print("\n  Cost of systematic bias (a participant who is consistently off):")
    print("  %-22s %11s %11s" % ("bias vs Q*", "mean cost", "vs optimal"))
    print("  " + "-" * 46)
    for bias in (-80, -40, -20, 0, 20, 40, 80):
        costs = [episode_cost(biased_policy(params, bias), s, params).total_cost() for s in range(N_SEEDS)]
        label = "%+d units" % bias if bias else "0 (perfect)"
        print("  %-22s %11.0f %10.2fx" % (label, np.mean(costs), np.mean(costs) / np.mean(optimal)))

    print("\n" + "=" * 74)
    print("DOES A BAD DECISION ACTUALLY PROPAGATE?")
    print("=" * 74)
    print("  Same seed, same everything, except ONE bad order in week 7.\n")

    def one_bad_order(bad_week, shortfall):
        def inner(state):
            q = reference_order(state, params)
            return max(0, q - shortfall) if state.period == bad_week else q
        return inner

    clean = episode_cost(ReferencePolicy(params), 42, params)
    dirty = episode_cost(one_bad_order(7, 150), 42, params)

    print("  wk   clean cost   after one bad order   difference")
    print("  " + "-" * 52)
    carried = 0
    for i in range(6, 14):
        c, d = clean.history[i]["cost"], dirty.history[i]["cost"]
        flag = "  <-- the bad order" if i == 7 else ""
        if i > 7 and abs(d - c) > 1:
            carried += 1
        print("  %2d %11.0f %21.0f %12.0f%s" % (i, c, d, d - c, flag))
    print("\n  One misplaced decision in week 7 changed costs for %d later weeks." % carried)
    print("  Total episode cost: %.0f clean vs %.0f -- a %.0f penalty from one order."
          % (clean.total_cost(), dirty.total_cost(), dirty.total_cost() - clean.total_cost()))

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    ratio = np.mean(naive) / np.mean(optimal)
    if ratio > 1.5:
        print("  The task HAS usable variance. Getting it right matters %.1fx." % ratio)
        print("  There is room for the AI to help, and room for a participant to")
        print("  be measurably good or bad on their own. Proceed to the model.")
    else:
        print("  WARNING: the gap is only %.1fx. The task may be too easy." % ratio)
        print("  Raise demand variance or lead time before collecting.")


if __name__ == "__main__":
    main()
