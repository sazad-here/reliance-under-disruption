"""The participant's world: which real demand a session replays.

One definition, imported by the app, the live risk model's training, and the
design checks, so all three build exactly the same warehouse.

WHY THIS FILE EXISTS. Until 24 Sep 2026 the app built its twin with no demand
source, so participants silently got the SYNTHETIC negative-binomial demand,
which is memoryless (lag-4 autocorrelation -0.16 against +0.74 in real data),
while z = 1.600 and delta = +/-60 had been calibrated on real M5 demand and
The sessions are documented as replaying M5 (docs/RESULTS.md, section 1).

DESIGN. Every participant replays the SAME real M5 window: the plan requires
"comparable difficulty across participants", and a shared demand path is what
delivers it. Only the stochastic lead-time delays, drawn from the participant's
seed, still differ. Practice uses a different series, so practice never
previews the main task's demand.

The two series are drawn once from a fixed seed, not picked by hand, and are
EXCLUDED from the live risk model's training (live_model.py). The AI
participants see is therefore predicting demand it has never seen, as a
deployed model would.
"""

from __future__ import annotations

import numpy as np

from demand_source import M5DemandSource, load_weekly
from simulator import SupplyChainTwin, TwinParams

WORLD_SEED = 2027          # fixed; changing it changes every session's demand
N_SERIES = 70              # M5 store-department series
MAIN_LEN, PRACTICE_LEN = 40, 12   # >= trials + look-ahead margin

_rng = np.random.default_rng(WORLD_SEED)
MAIN_SERIES, PRACTICE_SERIES = (int(s) for s in _rng.choice(N_SERIES, size=2, replace=False))
SESSION_SERIES = (MAIN_SERIES, PRACTICE_SERIES)


def _source(series: int, length: int, salt: int) -> M5DemandSource:
    rng = np.random.default_rng(WORLD_SEED + salt)
    return M5DemandSource(rng, length, weekly=load_weekly(), series_pool=[series])


# Order cap for sessions. The simulator default (600) is below the correct
# order once a participant digs deep into backorders (the live demo hit
# Q* = 731), which made a scheduled-correct week score as an error and broke
# the exact 80/60 reliability. 1500 keeps Q* reachable in practice.
SESSION_MAX_ORDER = 1500


def session_params() -> TwinParams:
    """The parameters every participant session runs on."""
    return TwinParams.from_fitted(max_order=SESSION_MAX_ORDER)


def make_twin(params: TwinParams, seed: int, phase: str = "main") -> SupplyChainTwin:
    """The twin a participant plays in `phase` ('practice' or 'main')."""
    if phase == "main":
        src = _source(MAIN_SERIES, MAIN_LEN, salt=1)
    elif phase == "practice":
        src = _source(PRACTICE_SERIES, PRACTICE_LEN, salt=2)
    else:
        raise ValueError("phase must be 'practice' or 'main'")
    return SupplyChainTwin(params=params, seed=seed, demand_source=src)


def describe() -> dict:
    main = _source(MAIN_SERIES, MAIN_LEN, salt=1)
    prac = _source(PRACTICE_SERIES, PRACTICE_LEN, salt=2)
    return {"main": main.stats(), "practice": prac.stats(), "world_seed": WORLD_SEED}
