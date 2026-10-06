"""Exploratory analyses (OSF s4 "Exploratory, not confirmatory"). No Holm
correction and no support claims: estimates with 95 % CIs only.

1. RAIR / RSR by condition (Schemmer et al. 2023; definitions in measures.py),
   both the literal variant and the switch-rule variant, with ambiguity
   counts. Also RSR split by the placement of the AI error.

2. Periods-to-recovery after each disruption. Onsets are read from the data
   (main trials whose `disruption` != 'none'; weeks 6, 9, 13, 17, 21, 26 under
   DEFAULT_CALENDAR). Rule, per participant and onset d, looking at weeks
   d .. next onset - 1 (or the last week):
     * impact  = first week >= d with a stockout; none -> "no impact",
                 recovery 0.
     * recovery = (first week after impact with no stockout AND zero closing
                 backorders) - d. If that never happens before the next onset
                 the episode is right-censored and reported as such (value =
                 window length, flagged censored).

3. Perceived vs actual reliability: perceived_reliability (0-100, asked once
   at the end) vs realised AI accuracy on main trials (x100). Calibration
   error = perceived - actual; correlation; by condition.

4. Outcome-bias split (run plan task B4). Take every AI-error trial t that
   the participant FOLLOWED (WoA_t >= 0.5, i.e. the final order moved at least
   half-way to the bad advice). Its outcome is "bad" if the week ended in a
   stockout (shown on the outcome screen immediately after that decision,
   before trial t+1) and "good" otherwise. Because of lead times that week's
   stockout is largely fixed by earlier orders -- the outcome is mostly luck,
   which is exactly what makes a difference in reliance an outcome BIAS.
   Outcome: WoA on trial t+1. Model: WoA_{t+1} ~ bad_outcome + rel_c + (1|pid).
   A complementary descriptive split for NOT-followed error trials is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.confirmatory import _coef, _fit_mixed, _prep


def _mean_ci(x):
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    m = float(x.mean())
    if n < 2:
        return {"n": n, "mean": m, "ci_low": None, "ci_high": None}
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": m, "ci_low": m - h, "ci_high": m + h}


def _diff_ci(a, b):
    """Welch difference a - b with 95 % CI."""
    a, b = pd.Series(a).dropna().astype(float), pd.Series(b).dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dfree = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = float(a.mean() - b.mean())
    h = float(stats.t.ppf(0.975, dfree) * se)
    return {"diff": d, "ci_low": d - h, "ci_high": d + h,
            "p_welch": float(2 * stats.t.sf(abs(d / se), dfree))}


def rair_rsr(measures):
    p, t = measures["participants"], measures["trials"]
    out = {}
    for var in ("rair", "rsr", "rair_switch", "rsr_keep"):
        out[var] = {c: _mean_ci(g[var]) for c, g in p.groupby("condition")}
        out[var]["high_minus_low"] = _diff_ci(p.loc[p.condition == "high", var],
                                              p.loc[p.condition == "low", var])
    # RSR by the placement of the AI error (IA trials)
    ia = t[t["case_IA"]]
    by_place = (ia.groupby(["participant_id", "condition", "placement"])["CSR"].mean()
                  .rename("rsr").reset_index())
    out["rsr_by_error_placement"] = {
        f"{c}/{pl}": _mean_ci(g["rsr"]) for (c, pl), g in by_place.groupby(["condition", "placement"])}
    out["ambiguity"] = {
        "CA_trials": int(t["case_CA"].sum()), "IA_trials": int(t["case_IA"].sum()),
        "CA_final_right_but_not_switched": int((t["CAIR"] & ~t["CAIR_switch"]).sum()),
        "IA_final_right_but_not_kept": int((t["CSR"] & ~t["CSR_keep"]).sum()),
        "CA_anchor_ambiguous": int((t["case_CA"] & t["anchor_ambiguous"]).sum()),
        "IA_anchor_ambiguous": int((t["case_IA"] & t["anchor_ambiguous"]).sum()),
        "excluded_confirmation_or_both_wrong": int((~t["case_CA"] & ~t["case_IA"]).sum()),
    }
    return out


def recovery(measures):
    t = measures["trials"]
    onsets = sorted(t.loc[t["disruption"] != "none", "trial_index"].unique().tolist())
    last = int(t["trial_index"].max()) if len(t) else -1
    rows = []
    for pid, g in t.groupby("participant_id"):
        g = g.set_index("trial_index").sort_index()
        cond = g["condition"].iloc[0]
        for k, d in enumerate(onsets):
            end = (onsets[k + 1] - 1) if k + 1 < len(onsets) else last
            w = g.loc[d:end]
            if w.empty:
                continue
            so = w.index[w["stockout"] == 1]
            if len(so) == 0:
                rows.append((pid, cond, d, 0, False, False))
                continue
            first = so[0]
            ok = w.loc[first:]
            ok = ok.index[(ok["stockout"] == 0) & (ok["backorders_close"] == 0)]
            if len(ok):
                rows.append((pid, cond, d, int(ok[0] - d), True, False))
            else:
                rows.append((pid, cond, d, int(end - d + 1), True, True))
    r = pd.DataFrame(rows, columns=["participant_id", "condition", "onset", "periods_to_recovery",
                                    "impacted", "censored"])
    summary = {}
    for (d, c), g in r.groupby(["onset", "condition"]):
        s = _mean_ci(g["periods_to_recovery"])
        s.update({"share_impacted": float(g["impacted"].mean()),
                  "n_censored": int(g["censored"].sum())})
        summary[f"week{d}/{c}"] = s
    per_p = r.groupby(["participant_id", "condition"])["periods_to_recovery"].mean().reset_index()
    return {"onsets": onsets, "by_onset_condition": summary,
            "mean_per_participant_high_minus_low": _diff_ci(
                per_p.loc[per_p.condition == "high", "periods_to_recovery"],
                per_p.loc[per_p.condition == "low", "periods_to_recovery"]),
            "_frame": r}


def perceived_vs_actual(measures):
    p = measures["participants"].copy()
    p["actual_pct"] = 100 * p["actual_reliability"]
    p["calibration_error"] = p["perceived_reliability"] - p["actual_pct"]
    q = p.dropna(subset=["perceived_reliability"])
    out = {"by_condition": {c: {"perceived": _mean_ci(g["perceived_reliability"]),
                                "actual": _mean_ci(g["actual_pct"]),
                                "calibration_error": _mean_ci(g["calibration_error"])}
                            for c, g in q.groupby("condition")},
           "manipulation_check_high_minus_low": _diff_ci(
               q.loc[q.condition == "high", "perceived_reliability"],
               q.loc[q.condition == "low", "perceived_reliability"])}
    if len(q) > 2 and q["actual_pct"].nunique() > 1:
        r, pv = stats.pearsonr(q["perceived_reliability"], q["actual_pct"])
        out["pearson_r_perceived_actual"] = {"r": float(r), "p": float(pv), "n": int(len(q))}
    return out


def outcome_bias(measures):
    t = measures["trials"]   # post_error, prev_woa, prev_stockout from measures.py
    d = t[t["post_error"] & t["woa"].notna() & t["prev_woa"].notna()].copy()
    d["followed"] = d["prev_woa"] >= 0.5
    d["bad_outcome"] = (d["prev_stockout"] == 1).astype(float)
    out = {"n_post_error_trials": int(len(d)),
           "cells": {f"{'followed' if f else 'not_followed'}/{'bad' if b else 'good'}": _mean_ci(x["woa"])
                     for (f, b), x in d.groupby(["followed", "bad_outcome"])}}
    fol = _prep(d[d["followed"]])
    if fol["bad_outcome"].nunique() == 2 and fol["participant_id"].nunique() > 3:
        try:
            res, label, warns = _fit_mixed("woa ~ bad_outcome + rel_c", fol, try_slope=False)
            out["followed_model"] = {"model": "WoA_{t+1} ~ bad_outcome + rel_c + (1|pid)",
                                     **_coef(res, "bad_outcome"), "n_obs": int(res.nobs)}
        except RuntimeError as exc:
            out["followed_model"] = {"error": str(exc)}
    return out


def workload(measures):
    b = measures["blocks"]
    return {f"block{k + 1}/{c}": _mean_ci(g["tlx"]) for (k, c), g in b.groupby(["block", "condition"])}


def run(measures):
    rec = recovery(measures)
    frame = rec.pop("_frame")
    return {"rair_rsr": rair_rsr(measures), "recovery": rec,
            "perceived_vs_actual": perceived_vs_actual(measures),
            "outcome_bias_split": outcome_bias(measures),
            "tlx_by_block": workload(measures)}, frame
