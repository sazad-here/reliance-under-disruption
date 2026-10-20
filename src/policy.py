"""
Ground truth: the correct order quantity Q* for every period.

Every reliance measure in the study depends on a per-trial correct answer.
"Correct" here means AGREEING WITH A BASE-STOCK HEURISTIC, not omniscient
optimality -- the paper must say so plainly rather than let a reviewer find it
(plan section 06).

    S_t  = mu_D E[L+1] + z sqrt(E[L+1] sigma_D^2 + mu_D^2 Var(L))   order-up-to level
    Q*_t = max(0, S_t - IP_t)                   IP = on hand + pipeline - backorders

(The lead time is stochastic, so the risk period has a mean and a variance; see
lead_time_moments and order_up_to_level below. An earlier version of this note
showed the fixed-lead-time form mu_D (L+1) + z sigma_D sqrt(L+1).)
    correct(Q) iff |Q - Q*_t| <= delta

delta is derived from cost sensitivity rather than picked round: it is the band
within which expected period cost rises less than `tolerance` above the optimum.
That is more defensible, and is itself a small methodological contribution.
"""
from __future__ import annotations

import math

from scipy import stats

from simulator import State, TwinParams


# Calibrated by grid search on the simulator (src/calibrate_z.py, 300 episodes
# per candidate) on REAL M5 DEMAND (src/calibrate_real.py). The myopic newsvendor
# ratio gives z = 0.967 and costs 11.2% more. On synthetic memoryless demand the
# penalty was only 3.9% and the optimum was 1.260 -- autocorrelated demand makes
# bad weeks CLUSTER, draining the buffer before it can rebuild, so the safety
# stock must be larger than a memoryless model implies. The myopic ratio costs
# more, because it treats each period as standing alone while this system
# carries backorders forward. Using the myopic value would have meant a
# participant who consistently over-ordered beat the "correct" answer --
# making decision error, RAIR and RSR measure the wrong thing.
# This choice must be stated in the method section.
SAFETY_FACTOR_OVERRIDE = 1.600


def myopic_safety_factor(params: TwinParams) -> float:
    """z from the single-period newsvendor critical ratio p / (p + h).

    Kept for comparison and for the robustness check in the paper. With the
    default costs (h=1, p=5) the critical ratio is 0.833 and z ~ 0.967.
    """
    critical_ratio = params.shortage_cost / (params.shortage_cost + params.holding_cost)
    return float(stats.norm.ppf(critical_ratio))


def safety_factor(params: TwinParams) -> float:
    """The z actually used for Q*. Calibrated, not assumed."""
    if SAFETY_FACTOR_OVERRIDE is not None:
        return SAFETY_FACTOR_OVERRIDE
    return myopic_safety_factor(params)


def lead_time_moments(params: TwinParams, lead_time: int):
    """Mean and variance of the RISK PERIOD (lead time + 1 review period).

    The lead time is not deterministic: `delay_pmf` adds 0-3 extra periods. An
    order-up-to level that ignores that is systematically too low, which makes
    Q* wrong and every reliance measure meaningless. Caught by diagnose.py --
    ordering +80 above the old Q* was 32% CHEAPER than the "correct" answer.
    """
    extras = range(len(params.delay_pmf))
    e_extra = sum(k * p for k, p in zip(extras, params.delay_pmf))
    e_extra_sq = sum(k * k * p for k, p in zip(extras, params.delay_pmf))
    var_extra = e_extra_sq - e_extra ** 2
    return lead_time + 1 + e_extra, var_extra


def order_up_to_level(params: TwinParams, lead_time: int) -> float:
    """S_t: cover expected demand over the risk period, plus a safety buffer.

    With a stochastic lead time the demand-over-lead-time variance has two parts:

        Var = E[L+1] sigma_D^2  +  mu_D^2 Var(L)
              ^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^
              demand varies        the lead time itself varies

    Dropping the second term is the classic safety-stock error. Here it is the
    larger of the two, because a late delivery costs a whole period of demand.
    """
    mean_periods, var_lead = lead_time_moments(params, lead_time)
    mean_over_risk = params.demand_mean * mean_periods
    var_over_risk = (
        mean_periods * params.demand_sd ** 2
        + params.demand_mean ** 2 * var_lead
    )
    return mean_over_risk + safety_factor(params) * math.sqrt(var_over_risk)


def reference_order(state: State, params: TwinParams) -> int:
    """Q*_t -- the correct answer for this period, given this participant's state.

    Note this depends on the LIVE state, so it differs between participants once
    their inventories diverge. That is why advice cannot simply be pre-generated
    from a fixed table.
    """
    level = order_up_to_level(params, state.lead_time)
    return int(max(0, round(level - state.inventory_position)))


# Empirically calibrated on the simulator (src/calibrate_z.py::calibrate_delta,
# 300 episodes per candidate). Once z is calibrated the analytical newsvendor
# cost curve no longer describes this system -- it returns +/-152, a band so wide
# that almost any order would score as "correct". Measured values:
#     5% cost tolerance  -> +/- 60 units
#    10% cost tolerance  -> +/- 85 units
# Measured on REAL M5 demand at z = 1.600. History: +/-50 (invented params),
# +/-45 (fitted params, synthetic demand), +/-60 (real demand).
# Report both in the paper as the robustness check (plan section 06).
DELTA_CALIBRATED = {0.05: 60, 0.10: 85}


def cost_derived_delta(params: TwinParams, tolerance: float = 0.05) -> int:
    """Widest deviation from Q* that keeps expected episode cost within `tolerance`.

    Returns the calibrated value when one exists for this tolerance, else falls
    back to the analytical newsvendor curve (which is only a rough guide here).
    """
    if tolerance in DELTA_CALIBRATED:
        return DELTA_CALIBRATED[tolerance]
    z_star = safety_factor(params)
    mean_periods, var_lead = lead_time_moments(params, params.lead_time_nominal)
    sd = math.sqrt(
        mean_periods * params.demand_sd ** 2 + params.demand_mean ** 2 * var_lead
    )

    def expected_cost(z: float) -> float:
        # standard normal loss function L(z) = phi(z) - z (1 - Phi(z))
        loss = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
        holding = params.holding_cost * sd * (z + loss)
        shortage = params.shortage_cost * sd * loss
        return holding + shortage

    best = expected_cost(z_star)
    limit = best * (1 + tolerance)

    delta = 0
    while delta < 400:
        step = (delta + 1) / sd
        if expected_cost(z_star + step) > limit and expected_cost(z_star - step) > limit:
            break
        delta += 1
    return delta


def is_correct(order: int, q_star: int, delta: int) -> bool:
    return abs(order - q_star) <= delta


class ReferencePolicy:
    """Callable base-stock agent. Used to generate the offline training corpus."""

    def __init__(self, params: TwinParams):
        self.params = params

    def __call__(self, state: State) -> int:
        return min(reference_order(state, self.params), self.params.max_order)


class NaivePolicy:
    """Order exactly last period's demand. The floor any sane policy must beat."""

    def __init__(self, params: TwinParams):
        self.params = params

    def __call__(self, state: State) -> int:
        return min(state.demand, self.params.max_order)
