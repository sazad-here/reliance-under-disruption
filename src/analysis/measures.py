"""Trial-, participant- and block-level measures (OSF section 9).

Correctness band
----------------
delta = policy.cost_derived_delta(params) = 60 units (5 % cost tolerance).
An order Q is "right" iff |Q - Q*| <= delta. The logged per-trial `delta`
column must agree; `check_delta` raises if it does not.

"AI right" is the logged `ai_correct` flag -- the manipulation's own truth.
It equals |Q_AI - Q*| <= delta except on cap-bound weeks (Q* above the order
cap, AI recommends the cap), which the design counts as correct.

Weight of Advice
----------------
woa        = (Q_final - Q_own) / (Q_AI - Q_own)   primary, untransformed; NaN
             when Q_AI == Q_own (trial dropped from WoA analyses only, OSF s8)
woa_trunc  = woa clipped to [0, 1]                  documented secondary variant
             (the conventional truncation: moving away from the advice -> 0,
             overshooting it -> 1)

Decision error = |Q_final - Q*|.

RAIR / RSR -- Schemmer, Kuehl, Benz, Bartos & Satzger (2023), IUI '23,
doi:10.1145/3581641.3584066, arXiv:2302.02187, section 3.2, eqs. (1)-(2).
Verified against the arXiv v3 text (25 Sep 2026):

    RAIR = sum CAIR_i / sum CA_i
      CA_i   = 1 if the initial human decision is wrong and the AI advice is
               correct (regardless of the final decision)
      CAIR_i = 1 if CA_i and the final decision is correct
    RSR  = sum CSR_i / sum IA_i
      IA_i   = 1 if the initial human decision is correct and the AI advice
               is incorrect
      CSR_i  = 1 if IA_i and the final decision is correct

In the paper's binary task, "final correct" on a CA case can only mean the
human switched to the AI, and on an IA case only that they kept their own
answer. With continuous orders it need not: a participant can end inside the
band without moving toward the advice. So two variants are computed:

  * PRIMARY  (`rair`, `rsr`): the paper's literal formula -- numerator is
    "final within delta of Q*".
  * SWITCH-RULE (`rair_switch`, `rsr_keep`): numerator additionally requires
    the final order to be attributable to the source of the correct answer:
        switched to AI  : |Q_final - Q_AI|  <= delta and |Q_final - Q_AI|  < |Q_final - Q_own|
        kept own        : |Q_final - Q_own| <= delta and |Q_final - Q_own| < |Q_final - Q_AI|
    Ambiguous cases are explicit: equidistant finals, and finals within delta
    of neither anchor ("elsewhere"), count as neither switched nor kept (they
    stay in the denominator). Their counts are reported. Schemmer et al.
    call these collaboration rather than reliance outcomes and set them
    aside; keeping them in the denominator is the conservative choice.

The per-trial binary appropriate-reliance outcome of the OSF s10 logit model is
AR_ij = CAIR_ij on CA trials and CSR_ij on IA trials; it is undefined (NaN) on
the confirmation / both-wrong trials that the Schemmer measure excludes.

Post-error trials (H3)
----------------------
A main trial t >= 1 is post-error if trial t-1 was an AI error
(ai_correct == 0). Its `post_error_placement` is the placement of that error
(propagating / absorbed). This is "the trial AFTER an AI error", as in the
power analysis (src/power_analysis.py). Consecutive error weeks (e.g. 7-8-9)
make some post-error trials error trials themselves; they are kept, and a
sensitivity model adds ai_correct (confirmatory.py).

Block measures
--------------
Trust: TiA items tia_1..tia_6 on 1-7. Items 1-3 are distrust statements
("deceptive", "suspicious", "wary") and are reverse-scored (8 - x); trust =
mean of the six. NASA-TLX: five subscales on 1-20 (physical demand not
administered). "Performance" is asked as "How successful were you" with 20 =
very high, i.e. high = LOW workload, so it is reverse-scored (21 - x); raw
TLX = mean of the five.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analysis  # noqa: F401
from policy import cost_derived_delta

TIA_ITEMS = tuple("tia_%d" % i for i in range(1, 7))
TIA_REVERSED = ("tia_1", "tia_2", "tia_3")
TLX_ITEMS = ("mental", "temporal", "performance", "effort", "frustration")
TLX_REVERSED = ("performance",)


def study_delta() -> int:
    from interface.world import session_params
    return int(cost_derived_delta(session_params()))


def check_delta(trials: pd.DataFrame, delta: int) -> None:
    if len(trials) and not (trials["delta"] == delta).all():
        bad = sorted(trials.loc[trials["delta"] != delta, "delta"].unique().tolist())
        raise ValueError("logged delta %s differs from policy.cost_derived_delta = %d"
                         % (bad, delta))


def trial_measures(trials: pd.DataFrame, delta: int) -> pd.DataFrame:
    t = trials.sort_values(["participant_id", "trial_index"]).copy()
    q_own, q_ai, q_fin, q_star = (t[c].astype(float) for c in ("q_own", "q_ai", "q_final", "q_star"))

    denom = q_ai - q_own
    t["woa_defined"] = denom != 0
    t["woa"] = np.where(t["woa_defined"], (q_fin - q_own) / denom.where(denom != 0), np.nan)
    t["woa_trunc"] = t["woa"].clip(0, 1)
    t["decision_error"] = (q_fin - q_star).abs()
    t["own_error"] = (q_own - q_star).abs()

    t["own_right"] = (q_own - q_star).abs() <= delta
    t["final_right"] = (q_fin - q_star).abs() <= delta
    t["ai_right"] = t["ai_correct"].astype(int) == 1

    t["case_CA"] = ~t["own_right"] & t["ai_right"]
    t["case_IA"] = t["own_right"] & ~t["ai_right"]
    t["CAIR"] = t["case_CA"] & t["final_right"]
    t["CSR"] = t["case_IA"] & t["final_right"]

    d_ai = (q_fin - q_ai).abs()
    d_own = (q_fin - q_own).abs()
    t["switched_to_ai"] = (d_ai <= delta) & (d_ai < d_own)
    t["kept_own"] = (d_own <= delta) & (d_own < d_ai)
    t["anchor_ambiguous"] = ~t["switched_to_ai"] & ~t["kept_own"]
    t["CAIR_switch"] = t["case_CA"] & t["switched_to_ai"]
    t["CSR_keep"] = t["case_IA"] & t["kept_own"]

    t["AR"] = np.where(t["case_CA"], t["CAIR"].astype(float),
                       np.where(t["case_IA"], t["CSR"].astype(float), np.nan))

    # H3: the trial after an AI error, and where that error sat
    g = t.groupby("participant_id", sort=False)
    prev_ai_correct = g["ai_correct"].shift(1)
    prev_idx = g["trial_index"].shift(1)
    prev_place = g["placement"].shift(1)
    contiguous = prev_idx == t["trial_index"] - 1
    t["post_error"] = contiguous & (prev_ai_correct == 0)
    t["post_error_placement"] = np.where(t["post_error"], prev_place, None)
    # outcome of the previous (error) week as the participant saw it
    t["prev_stockout"] = g["stockout"].shift(1)
    t["prev_woa"] = g["woa"].shift(1)
    return t


def participant_measures(t: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("participant_id")
    out = pd.DataFrame({
        "n_trials": g.size(),
        "woa_mean": g["woa"].mean(),
        "woa_trunc_mean": g["woa_trunc"].mean(),
        "decision_error_mean": g["decision_error"].mean(),
        "n_CA": g["case_CA"].sum(), "n_CAIR": g["CAIR"].sum(),
        "n_IA": g["case_IA"].sum(), "n_CSR": g["CSR"].sum(),
        "n_CAIR_switch": g["CAIR_switch"].sum(), "n_CSR_keep": g["CSR_keep"].sum(),
        "n_CA_ambiguous": g.apply(lambda d: int((d["case_CA"] & d["anchor_ambiguous"]).sum()),
                                  include_groups=False),
        "n_IA_ambiguous": g.apply(lambda d: int((d["case_IA"] & d["anchor_ambiguous"]).sum()),
                                  include_groups=False),
        "actual_reliability": g["ai_correct"].mean(),
        "total_cost": g["cost"].sum(),
        "n_stockouts": g["stockout"].sum(),
    })
    out["rair"] = out["n_CAIR"] / out["n_CA"].where(out["n_CA"] > 0)
    out["rsr"] = out["n_CSR"] / out["n_IA"].where(out["n_IA"] > 0)
    out["rair_switch"] = out["n_CAIR_switch"] / out["n_CA"].where(out["n_CA"] > 0)
    out["rsr_keep"] = out["n_CSR_keep"] / out["n_IA"].where(out["n_IA"] > 0)
    out = out.reset_index().merge(
        participants[["participant_id", "condition", "perceived_reliability"]],
        on="participant_id", how="left")
    return out


def block_scores(block: pd.DataFrame) -> pd.DataFrame:
    """One row per participant x block: trust (1-7) and raw TLX (1-20)."""
    if not len(block):
        return pd.DataFrame(columns=["participant_id", "block", "trust", "tlx",
                                     "n_tia_items", "n_tlx_items"])
    b = block.copy()
    tia = b[b["instrument"] == "tia"].copy()
    tia["score"] = np.where(tia["item"].isin(TIA_REVERSED), 8 - tia["response"], tia["response"])
    tia = tia[tia["item"].isin(TIA_ITEMS)]
    tlx = b[b["instrument"] == "tlx"].copy()
    tlx["score"] = np.where(tlx["item"].isin(TLX_REVERSED), 21 - tlx["response"], tlx["response"])
    tlx = tlx[tlx["item"].isin(TLX_ITEMS)]
    trust = tia.groupby(["participant_id", "block"])["score"].agg(["mean", "size"])
    work = tlx.groupby(["participant_id", "block"])["score"].agg(["mean", "size"])
    out = pd.DataFrame({"trust": trust["mean"], "n_tia_items": trust["size"]}).join(
        pd.DataFrame({"tlx": work["mean"], "n_tlx_items": work["size"]}), how="outer")
    # A block score needs every item; a partial questionnaire is not scored.
    out.loc[out["n_tia_items"] != len(TIA_ITEMS), "trust"] = np.nan
    out.loc[out["n_tlx_items"] != len(TLX_ITEMS), "tlx"] = np.nan
    return out.reset_index()


def compute(kept, delta: int | None = None) -> dict:
    delta = study_delta() if delta is None else delta
    check_delta(kept.trials, delta)
    t = trial_measures(kept.trials, delta)
    cond = kept.participants[["participant_id", "condition"]]
    t = t.merge(cond, on="participant_id", how="left")
    p = participant_measures(t, kept.participants)
    b = block_scores(kept.block).merge(cond, on="participant_id", how="left")
    return {"delta": delta, "trials": t, "participants": p, "blocks": b}
