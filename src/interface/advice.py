"""The AI advice layer: what the participant is shown at stage 2 of each trial.

Two manipulations live here, and nothing else in the interface knows about them.

    Between subjects  reliability   80 % vs 60 % correct recommendations
    Within subjects   placement     each participant's error trials are split
                                    across ABSORBED and PROPAGATING positions

Both are resolved *before* the session starts, by `build_schedule()`, so the
error calendar is deterministic given (condition, participant seed). Advice is
never decided live: a live coin flip would make the realised reliability drift
away from the nominal level, and the manipulation check would fail.

Q_AI itself must be computed live, because Q* depends on the participant's own
inventory position, which diverges from every other participant's by trial 2
(see policy.reference_order). So the *schedule* is fixed and the *quantity* is
derived -- correct trials get Q*, error trials get Q* pushed outside the delta
band in a direction chosen by placement.

The RISK percentage is real model output: a LightGBM stockout model
(live_model.py; cross-validated AUROC 0.897 on held-out series) runs on the
participant's own history every week. The recommendation is scripted so the
reliability manipulation stays exact; the risk is the model's, computed on the
state the AI believes, so the two always agree.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, asdict

import numpy as np

from simulator import DEFAULT_CALENDAR, State, TwinParams
from policy import cost_derived_delta, reference_order
from live_model import get_model, state_row

# Reliability conditions, as the paper reports them. The 80 % level sits below
# the validated LightGBM stockout model (AUROC 0.891 on 70 held-out series,
# accuracy 86.2 % vs a 78.5 % majority floor); see OSF-PREREGISTRATION.md s11
# for the registered wording. Never anchor to raw accuracy alone.
RELIABILITY_LEVELS = {"high": 0.80, "low": 0.60}

# ---- error design -----------------------------------------------------------
# v2 (default, 24 Sep 2026). Fixes three problems found in the design audit:
#   1. Placement was confounded with direction (propagating = always under,
#      absorbed = always over). v2: EVERY error is an under-order, so H3
#      compares placement alone.
#   2. "Propagating" meant "on or one before a disruption", which measured
#      badly on the participants' real demand (week 16: +726 extra cost,
#      week 12: -206). v2 positions are chosen by measured consequence
#      (src/placement_design.py): selected on lead-time draws 0-99, validated
#      on 100-299 -- propagating errors cost +134 vs -62 and cause 3.45x the
#      extra stockout weeks (1.78x for a noisy follower). All other weeks are
#      NEUTRAL: never an error week.
#   3. 17 % of propagating errors recommended 0 units -- implausible and easy
#      to reject. v2 never recommends fewer than MIN_ORDER units.
# v1 is the original design, kept so it can be restored with ADVICE_DESIGN=v1.
ADVICE_DESIGN = os.environ.get("ADVICE_DESIGN", "v2")
if ADVICE_DESIGN not in ("v1", "v2"):
    raise ValueError("ADVICE_DESIGN must be v1 or v2")

V2_PROPAGATING = (5, 7, 8, 9, 14, 15, 16, 19)
V2_ABSORBED = (2, 3, 10, 11, 12, 20, 22, 25)
MIN_ORDER = 10

# v1: trials on or one before a disruption are PROPAGATING, the rest ABSORBED.
_DISRUPTION_PERIODS = {e.period for e in DEFAULT_CALENDAR}
PROPAGATING_PERIODS = sorted(_DISRUPTION_PERIODS | {p - 1 for p in _DISRUPTION_PERIODS})


def placement_of(period: int, design: str | None = None) -> str:
    """'propagating', 'absorbed' or (v2 only) 'neutral' for a trial period."""
    design = design or ADVICE_DESIGN
    if design == "v1":
        return "propagating" if period in PROPAGATING_PERIODS else "absorbed"
    if period in V2_PROPAGATING:
        return "propagating"
    if period in V2_ABSORBED:
        return "absorbed"
    return "neutral"


@dataclass(frozen=True)
class Advice:
    """Exactly what the participant sees at stage 2, plus what we log."""

    q_ai: int
    risk_pct: int           # displayed stockout-risk percentage
    level: str              # displayed confidence band: low / moderate / high
    ai_correct: bool        # ground truth, never shown to the participant
    placement: str          # absorbed | propagating
    q_star: int             # logged, never shown
    delta: int              # logged, never shown

    def public(self) -> dict:
        """The subset that may cross the wire to the browser."""
        return {"q_ai": self.q_ai, "risk_pct": self.risk_pct, "level": self.level}

    def record(self) -> dict:
        return asdict(self)


def participant_seed(participant_id: str) -> int:
    """Stable per-participant seed, so a session can be replayed exactly."""
    digest = hashlib.sha256(participant_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def build_schedule(participant_id: str, condition: str, n_trials: int,
                   design: str | None = None) -> list[bool]:
    """Which trials the AI gets RIGHT. Index = trial number (0-based).

    The error count is exact, not sampled: with 30 trials at 80 % the
    participant sees exactly 6 errors, not "6 on average". Sampling each trial
    independently would give a realised reliability anywhere in roughly
    65-93 %, which is wider than the gap between our two conditions.

    Errors are split as evenly as the calendar allows across absorbed and
    propagating positions, which is the within-subjects manipulation.
    """
    if condition not in RELIABILITY_LEVELS:
        raise ValueError("unknown condition %r" % condition)

    rate = RELIABILITY_LEVELS[condition]
    n_error = int(round(n_trials * (1.0 - rate)))
    rng = np.random.default_rng(participant_seed(participant_id))

    periods = list(range(n_trials))
    prop = [p for p in periods if placement_of(p, design) == "propagating"]
    absorb = [p for p in periods if placement_of(p, design) == "absorbed"]
    if not prop and not absorb:          # e.g. a short practice run under v2
        absorb = periods

    # Half the errors into each bucket; propagating positions are scarcer, so
    # it takes the smaller share when the split cannot be even.
    n_prop = min(len(prop), n_error // 2)
    n_absorb = n_error - n_prop
    if n_absorb > len(absorb):  # degenerate only at very small n_trials
        n_absorb = len(absorb)
        n_prop = min(len(prop), n_error - n_absorb)

    chosen = set(rng.choice(prop, size=n_prop, replace=False).tolist()) if n_prop else set()
    chosen |= set(rng.choice(absorb, size=n_absorb, replace=False).tolist()) if n_absorb else set()

    return [t not in chosen for t in periods]


def _believed_rows(history: list[dict], shift: int) -> list[dict]:
    """The state as the AI 'reads' it on an error week.

    An order-up-to rule orders S - IP. If the AI recommends Q_AI instead of Q*,
    it is acting as if the inventory position were IP + (Q* - Q_AI). Shifting
    the current week's position by that amount and asking the model for the
    risk makes the wrong advice internally consistent: an under-order comes with
    a LOWER risk (it thinks there is more stock), an over-order with a higher
    one. A real model's mistake looks like that; a correct risk shown next to a
    wrong order does not, and gave errors away (audit: detection AUROC
    0.544 -> 0.657).
    """
    rows = [dict(r) for r in history]
    cur = rows[-1]
    if shift >= 0:
        cur["inventory"] += shift
    else:
        take = min(cur["inventory"], -shift)
        cur["inventory"] -= take
        cur["backorders"] += (-shift) - take
    return rows


def _risk_display(history: list[dict], q_star: int, q_ai: int) -> tuple[int, str]:
    """Stockout risk the interface shows: the live LightGBM model's output.

    Calibrated P(stockout within 2 weeks) from live_model.py, computed on the
    participant's own history. On an error week it is computed on the state the
    AI believes (see _believed_rows), so risk and recommendation agree.
    """
    rows = history if q_ai == q_star else _believed_rows(history, q_star - q_ai)
    p = get_model().predict(rows)
    risk = int(np.clip(round(100 * p), 1, 99))
    level = "high" if risk >= 55 else ("moderate" if risk >= 25 else "low")
    return risk, level


def make_advice(
    state: State,
    params: TwinParams,
    trial_index: int,
    correct: bool,
    rng: np.random.Generator,
    tolerance: float = 0.05,
    history: list[dict] | None = None,
    design: str | None = None,
) -> Advice:
    """Build one trial's advice. `correct` comes from the pre-built schedule.

    `history` is this phase's weekly states so far, oldest first, ending with
    `state` (see live_model.state_row). The risk model needs it; without it the
    model sees only the current week.
    """
    q_star = reference_order(state, params)
    delta = cost_derived_delta(params, tolerance)
    design = design or ADVICE_DESIGN
    placement = placement_of(state.period, design)

    if correct:
        # Inside the band, but not pinned exactly to Q* -- a recommendation that
        # is always exactly optimal is detectable and unlike any real model.
        jitter = int(rng.integers(-delta // 2, delta // 2 + 1))
        q_ai = q_star + jitter
    elif design == "v2":
        # Always an under-order, never below MIN_ORDER. Only when Q* is so
        # small that no plausible under-order can leave the band does it fall
        # back to an over-order; the direction stays recoverable from the log.
        magnitude = delta + int(rng.integers(delta // 2, delta + 1))
        if q_star - MIN_ORDER > delta:
            q_ai = max(q_star - magnitude, MIN_ORDER)
        else:
            q_ai = q_star + magnitude
    else:
        # Outside the band. Direction matters: under-ordering is the error that
        # propagates (RESULTS.md section 5 -- persistent under-ordering spans
        # 4.3 periods vs 1.77 for one-off noise), so propagating positions get
        # the under-order and absorbed positions get the over-order.
        magnitude = delta + int(rng.integers(delta // 2, delta + 1))
        q_ai = q_star - magnitude if placement == "propagating" else q_star + magnitude

    q_ai = int(np.clip(q_ai, 0, params.max_order))
    # Edge rule: on a scheduled-correct week whose Q* exceeds the order cap, the
    # best feasible recommendation is the cap. It counts as correct, so the
    # realised reliability stays exactly at the nominal level; the case stays
    # identifiable in the log (q_star > max_order, q_ai == max_order).
    cap_bound = correct and q_star > params.max_order

    # Clipping at 0 can drag an intended error back inside the band when Q* is
    # small. Preserve the DIRECTION first: on a propagating trial the error is
    # supposed to be an under-order, and silently flipping it to an over-order
    # dilutes the placement contrast while the row still reads "propagating".
    # Ordering nothing is the largest under-order available, and it lands
    # outside the band whenever Q* exceeds delta.
    # v1 only: v2 errors leave the band by construction and never reach 0.
    if design == "v1" and not correct and abs(q_ai - q_star) <= delta:
        if placement == "propagating" and q_star > delta:
            q_ai = 0
        else:
            # Only reachable when Q* <= delta, where no under-order can fall
            # outside the band at all. The realised direction stays recoverable
            # in analysis: q_ai and q_star are both logged on every trial.
            q_ai = int(min(params.max_order,
                           q_star + delta + 1 + rng.integers(0, delta)))

    if history is None:
        history = [state_row(state)]
    # Risk follows the recommendation actually shown. On a correct week the AI
    # is right up to jitter inside the band, so it reads the true state.
    risk_pct, level = _risk_display(history, q_star, q_star if correct else q_ai)
    return Advice(
        q_ai=q_ai,
        risk_pct=risk_pct,
        level=level,
        ai_correct=bool(cap_bound or abs(q_ai - q_star) <= delta),
        placement=placement,
        q_star=q_star,
        delta=delta,
    )


def weight_of_advice(q_own: int, q_final: int, q_ai: int) -> float | None:
    """WoA = (final - own) / (AI - own). None when the AI agreed with the human.

    Undefined when q_ai == q_own: the trial carries no information about advice
    taking, and coding it as 0 or 1 would bias the mean. Drop it in analysis.
    """
    denom = q_ai - q_own
    if denom == 0:
        return None
    return (q_final - q_own) / denom


def response_class(q_own: int, q_final: int, q_ai: int, tol: int = 2) -> str:
    """accept / override / modify -- the categorical companion to WoA."""
    if abs(q_final - q_ai) <= tol:
        return "accept"
    if abs(q_final - q_own) <= tol:
        return "override"
    return "modify"
