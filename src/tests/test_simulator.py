"""
Property tests for the twin.

The point of this file is narrow and important: prove the inventory accounting
cannot double-count shortages. That bug was present in the original proposal
draft, it is silent, it corrupts every cost figure, and it is invisible from the
participant interface. A logging or accounting defect found after 80 sessions
are spent is unrecoverable, so it gets caught here instead.

Run:  python src/tests/test_simulator.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policy import (  # noqa: E402
    NaivePolicy,
    ReferencePolicy,
    cost_derived_delta,
    reference_order,
)
from simulator import (  # noqa: E402
    DEFAULT_CALENDAR,
    Disruption,
    SupplyChainTwin,
    TwinParams,
)

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    line = "  [%s] %s" % (mark, name)
    if detail and not condition:
        line += "\n         %s" % detail
    print(line)


def run_episode(policy, seed: int, params: TwinParams = None):
    params = params or TwinParams()
    twin = SupplyChainTwin(params=params, seed=seed)
    state = twin.observe()
    while not twin.done:
        state, _, _ = twin.step(policy(state))
    return twin


# ---------------------------------------------------------------- invariants
def test_inventory_backorder_identity():
    """The whole point: I and B can never both be positive."""
    bad = []
    for seed in range(40):
        twin = run_episode(ReferencePolicy(TwinParams()), seed)
        for r in twin.history:
            if r["inventory_close"] > 0 and r["backorders_close"] > 0:
                bad.append((seed, r["period"], r["inventory_close"], r["backorders_close"]))
    check(
        "inventory and backorders are never both positive",
        not bad,
        "violations: %s" % bad[:3],
    )


def test_no_negative_stock():
    bad = []
    for seed in range(40):
        twin = run_episode(ReferencePolicy(TwinParams()), seed)
        for r in twin.history:
            if r["inventory_close"] < 0 or r["backorders_close"] < 0:
                bad.append((seed, r["period"]))
    check("inventory and backorders never go negative", not bad, str(bad[:3]))


def test_flow_conservation():
    """Opening stock + nothing else must equal what was served plus what remains.

    inventory_after_loss - (demand + previous backorders)
        == inventory_close - backorders_close

    If shortages were counted twice this identity breaks immediately.
    """
    bad = []
    for seed in range(40):
        twin = run_episode(ReferencePolicy(TwinParams()), seed)
        prev_back = 0
        for r in twin.history:
            avail = r["inventory_open"] - r["lost"]
            lhs = avail - r["demand"] - prev_back
            rhs = r["inventory_close"] - r["backorders_close"]
            if lhs != rhs:
                bad.append((seed, r["period"], lhs, rhs))
            prev_back = r["backorders_close"]
    check("flow conservation holds every period", not bad, "mismatches: %s" % bad[:3])


def test_pipeline_conservation():
    """Every unit the supplier ships either arrives or is still in transit."""
    bad = []
    for seed in range(25):
        twin = run_episode(ReferencePolicy(TwinParams()), seed)
        shipped = sum(r["shipped"] for r in twin.history)
        arrived = sum(r["arrivals"] for r in twin.history)
        in_transit = sum(twin.pipeline.values())
        if shipped != arrived + in_transit:
            bad.append((seed, shipped, arrived, in_transit))
    check("units shipped == units arrived + units in transit", not bad, str(bad[:3]))


def test_double_counting_would_be_caught():
    """Sanity-check the test itself: the OLD buggy formula must fail conservation.

    Old draft:  B_t = max(0, D_t - I_t)   (non-recursive, ignores prior backorders)
                paired with inventory allowed to go negative.
    """
    inv, prev_back, demand = 100, 40, 130
    net = inv - demand - prev_back
    good_inv, good_back = max(0, net), max(0, -net)
    buggy_back = max(0, demand - inv)  # the draft's formula: 30, ignores the 40
    check(
        "the old non-recursive formula really does undercount backorders",
        buggy_back != good_back,
        "good=%d buggy=%d" % (good_back, buggy_back),
    )
    check(
        "good formula carries prior backorders forward",
        good_back == 70 and good_inv == 0,
        "inv=%d back=%d" % (good_inv, good_back),
    )


# ---------------------------------------------------------------- behaviour
def _reference_vs_naive():
    params = TwinParams()
    ref = [run_episode(ReferencePolicy(params), s).total_cost() for s in range(30)]
    naive = [run_episode(NaivePolicy(params), s).total_cost() for s in range(30)]
    return float(np.mean(ref)), float(np.mean(naive))


def test_reference_beats_naive():
    """If a sensible rule cannot beat 'order last week's demand', the twin is wrong."""
    ref, naive = _reference_vs_naive()
    check("base-stock policy beats the naive policy", ref < naive,
          "base-stock %.0f vs naive %.0f" % (ref, naive))


def test_reference_beats_random():
    params = TwinParams()
    rng = np.random.default_rng(0)

    def random_policy(state):
        return int(rng.integers(0, params.max_order))

    ref = [run_episode(ReferencePolicy(params), s).total_cost() for s in range(30)]
    rnd = [run_episode(random_policy, s).total_cost() for s in range(30)]
    check(
        "base-stock policy beats random ordering",
        np.mean(ref) < np.mean(rnd),
        "base-stock %.0f vs random %.0f" % (np.mean(ref), np.mean(rnd)),
    )


def test_determinism():
    a = run_episode(ReferencePolicy(TwinParams()), 7).history
    b = run_episode(ReferencePolicy(TwinParams()), 7).history
    check("same seed gives identical episodes", a == b)
    c = run_episode(ReferencePolicy(TwinParams()), 8).history
    check("different seed gives a different episode", a != c)


def test_disruption_calendar_is_fixed():
    """Identical for every participant -- that is what makes conditions comparable."""
    seen = []
    for seed in range(10):
        twin = run_episode(ReferencePolicy(TwinParams()), seed)
        seen.append([(r["period"], r["disruption"]) for r in twin.history])
    check("disruption calendar is identical across participants", all(s == seen[0] for s in seen))

    expected = {e.period: e.kind.value for e in DEFAULT_CALENDAR}
    actual = {p: d for p, d in seen[0] if d != Disruption.NONE.value}
    check("calendar fires exactly as specified", actual == expected, "%s vs %s" % (actual, expected))


def test_state_diverges_between_participants():
    """The contribution depends on consequences carrying forward, so two different
    order streams MUST end in different states."""
    params = TwinParams()
    a = run_episode(ReferencePolicy(params), 3)
    b = run_episode(NaivePolicy(params), 3)
    closes_a = [r["inventory_close"] for r in a.history]
    closes_b = [r["inventory_close"] for r in b.history]
    check("different decisions produce different trajectories", closes_a != closes_b)


def test_q_star_is_state_dependent():
    """Q* must respond to the participant's own inventory, or advice is nonsense."""
    params = TwinParams()
    twin = SupplyChainTwin(params=params, seed=1)
    lean = twin.observe()
    twin.inventory = 600
    rich = twin.observe()
    q_lean, q_rich = reference_order(lean, params), reference_order(rich, params)
    check(
        "Q* falls when the warehouse is fuller",
        q_rich < q_lean,
        "lean stock -> Q*=%d, full stock -> Q*=%d" % (q_lean, q_rich),
    )


def test_order_bounds_enforced():
    twin = SupplyChainTwin(seed=0)
    ok = False
    try:
        twin.step(-5)
    except ValueError:
        ok = True
    check("negative orders are rejected", ok)
    ok = False
    try:
        twin.step(99999)
    except ValueError:
        ok = True
    check("absurdly large orders are rejected", ok)


def test_params_validation():
    ok = False
    try:
        TwinParams(demand_mean=100, demand_sd=5)  # var < mean, invalid for NB
    except ValueError:
        ok = True
    check("under-dispersed demand parameters are rejected", ok)


if __name__ == "__main__":
    print("\n=== invariants: the accounting cannot double-count ===")
    test_inventory_backorder_identity()
    test_no_negative_stock()
    test_flow_conservation()
    test_pipeline_conservation()
    test_double_counting_would_be_caught()

    print("\n=== behaviour: the twin is a real decision problem ===")
    test_reference_beats_naive()
    ref_cost, naive_cost = _reference_vs_naive()
    test_reference_beats_random()
    test_q_star_is_state_dependent()
    test_state_diverges_between_participants()

    print("\n=== experimental control ===")
    test_determinism()
    test_disruption_calendar_is_fixed()
    test_order_bounds_enforced()
    test_params_validation()

    params = TwinParams()
    print("\n=== calibration readout ===")
    print("  cost-derived delta (5%% tolerance): +/- %d units" % cost_derived_delta(params))
    print("  cost-derived delta (10%% tolerance): +/- %d units" % cost_derived_delta(params, 0.10))
    print("  mean 30-period cost, base-stock:  %8.0f" % ref_cost)
    print("  mean 30-period cost, naive:       %8.0f" % naive_cost)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    sys.exit(1 if FAIL else 0)
