"""Pre-registered exclusions, OSF section 8, applied exactly.

Participant level -- excluded if ANY holds (all reasons are recorded; the
`primary_reason` is the first in the registered order):

  1. incomplete       the session is not completed through the final trial:
                      fewer than N_MAIN (30) main-phase trial rows.
  2. fast_stage1      more than 20 % of stage-1 responses are faster than
                      500 ms. "Stage-1 responses" is read literally: every
                      recorded stage-1 response, practice and main (the
                      registration does not restrict it to main trials).
                      Strictly greater than 20 %: 7 of 35 (exactly 20 %) is
                      NOT excluded.
  3. constant_own     Q_own identical on every main trial.
  4. withdrawn        withdrawn_utc set, or a withdrawal event in the log,
                      whether or not data were deleted.
  5. technical_fault  either
                      (a) an event_log row whose kind is in
                          TECHNICAL_FAULT_KINDS. NOTE: app 1.3.0 logs no
                          display-fault event yet; the kinds below are the
                          ones this script will honour. `rehydrate_failed` is
                          included because a session that could not be
                          restored was, by definition, not displayed normally.
                      or
                      (b) a browser `client_error` (logged by
                          /api/client-error) that is BOTH during the main
                          phase (detail.phase == "main") AND in a log that
                          fails the replay check of interface/verify_log.py
                          (main trials do not reproduce exactly from the
                          logged orders). A client_error alone is not a fault:
                          one in practice / consent / questionnaires never
                          excludes, and a main-phase one whose log still
                          replays cleanly is FLAGGED (`client_error_flagged`,
                          counted in the report) but the participant is kept.

Failing the numeracy screener is not an exclusion (such people have no
participant row at all).

Trial level: trials with Q_AI == Q_own are dropped from Weight-of-Advice
analyses ONLY (WoA undefined); they stay in every other measure. That drop is
applied in measures.py (woa is NaN there) and counted here.

The analysis does not write `excluded`/`exclusion_reason` back to the
database; it reports them (exclusions.csv) so the database stays read-only.
"""

from __future__ import annotations

import pandas as pd

N_MAIN = 30
FAST_RT_MS = 500
FAST_SHARE_LIMIT = 0.20
TECHNICAL_FAULT_KINDS = frozenset({"technical_fault", "display_fault",
                                   "rehydrate_failed"})
CLIENT_ERROR_KIND = "client_error"
WITHDRAWAL_KINDS = frozenset({"withdrawn_keep_data", "withdrawn"})
REASON_ORDER = ("incomplete", "fast_stage1", "constant_own", "withdrawn",
                "technical_fault")


def participant_flags(data) -> pd.DataFrame:
    p = data.participants[["participant_id", "condition", "withdrawn_utc",
                           "data_deleted"]].copy()
    ta = data.trials_all
    main = data.trials

    n_main = main.groupby("participant_id").size()
    p["n_main_trials"] = p["participant_id"].map(n_main).fillna(0).astype(int)
    p["incomplete"] = p["n_main_trials"] < N_MAIN

    stage1 = ta.groupby("participant_id")["rt_stage1_ms"]
    n_resp = stage1.size()
    n_fast = stage1.apply(lambda s: int((s < FAST_RT_MS).sum()))
    p["n_stage1"] = p["participant_id"].map(n_resp).fillna(0).astype(int)
    p["n_stage1_fast"] = p["participant_id"].map(n_fast).fillna(0).astype(int)
    p["share_fast"] = (p["n_stage1_fast"] / p["n_stage1"].where(p["n_stage1"] > 0)).fillna(0.0)
    p["fast_stage1"] = p["share_fast"] > FAST_SHARE_LIMIT

    n_unique_own = main.groupby("participant_id")["q_own"].nunique()
    # Only defined with at least two main trials; a participant with 0-1 main
    # trials is already caught by `incomplete`.
    p["constant_own"] = (p["participant_id"].map(n_unique_own).eq(1)
                         & (p["n_main_trials"] >= 2))

    ev = data.events
    withdrew_ev = set(ev.loc[ev["kind"].isin(WITHDRAWAL_KINDS), "participant_id"].dropna())
    p["withdrawn"] = p["withdrawn_utc"].notna() | p["participant_id"].isin(withdrew_ev)

    fault_ev = set(ev.loc[ev["kind"].isin(TECHNICAL_FAULT_KINDS), "participant_id"].dropna())

    # Criterion 5(b): a main-phase client_error excludes only if the log also
    # fails replay. Replay is run only for those participants (it is slow).
    cerr = ev[ev["kind"] == CLIENT_ERROR_KIND]
    cerr_main = set(cerr.loc[[isinstance(d, dict) and d.get("phase") == "main"
                              for d in cerr["detail_obj"]], "participant_id"].dropna())
    p["client_error_main"] = p["participant_id"].isin(cerr_main)
    replay_ok = {pid: replays_cleanly(data, pid) for pid in sorted(cerr_main)}
    p["client_error_replay_ok"] = p["participant_id"].map(replay_ok)   # NaN = not checked
    replay_failed = {pid for pid, ok in replay_ok.items() if not ok}
    p["client_error_flagged"] = p["participant_id"].isin(cerr_main - replay_failed)

    p["technical_fault"] = p["participant_id"].isin(fault_ev | replay_failed)

    p["excluded"] = p[list(REASON_ORDER)].any(axis=1)
    p["reasons"] = p.apply(lambda r: ";".join(k for k in REASON_ORDER if r[k]), axis=1)
    p["primary_reason"] = p["reasons"].str.split(";").str[0].where(p["excluded"], "")
    return p


def replays_cleanly(data, participant_id) -> bool:
    """True iff every logged main trial of this participant reproduces exactly
    when the twin is re-run from the logged orders -- the same check as the
    `replay` group of interface/verify_log.py (it calls the same function).
    A log that cannot be replayed at all (no parameter snapshot, missing
    columns) counts as failing."""
    from interface.verify_log import replay_main
    prow = data.participants.loc[data.participants["participant_id"] == participant_id]
    main = data.trials.loc[data.trials["participant_id"] == participant_id].sort_values("trial_index")
    if prow.empty or main.empty:
        return False
    try:
        drift, _, _ = replay_main(prow.iloc[0].to_dict(), main.to_dict("records"))
    except Exception:  # noqa: BLE001 -- an unreplayable log fails the check
        return False
    return not drift


def apply(data):
    """Return (data_kept, flags, report). `data_kept` holds only included
    participants; `report` is JSON-ready counts per condition and reason."""
    flags = participant_flags(data)
    keep = set(flags.loc[~flags["excluded"], "participant_id"])

    from copy import copy
    kept = copy(data)
    kept.participants = data.participants[data.participants["participant_id"].isin(keep)].reset_index(drop=True)
    kept.trials_all = data.trials_all[data.trials_all["participant_id"].isin(keep)].reset_index(drop=True)
    kept.trials = data.trials[data.trials["participant_id"].isin(keep)].reset_index(drop=True)
    kept.block = data.block[data.block["participant_id"].isin(keep)].reset_index(drop=True)
    kept.events = data.events[data.events["participant_id"].isin(keep)].reset_index(drop=True)

    report = {"n_participant_rows": int(len(flags)),
              "n_excluded": int(flags["excluded"].sum()),
              "n_analysed": int(len(keep)),
              # main-phase client_error, log replays cleanly: kept, reported
              "n_client_error_flagged_kept": int((flags["client_error_flagged"] & ~flags["excluded"]).sum()),
              "by_condition": {}}
    for cond, g in flags.groupby("condition"):
        report["by_condition"][cond] = {
            "recruited": int(len(g)),
            "excluded": int(g["excluded"].sum()),
            "analysed": int((~g["excluded"]).sum()),
            # every reason that holds (a participant can meet several)
            "any_reason": {k: int(g[k].sum()) for k in REASON_ORDER},
            # first reason in registered order (these sum to `excluded`)
            "primary_reason": {k: int((g["primary_reason"] == k).sum()) for k in REASON_ORDER},
            "withdrawn_with_deletion": int((g["withdrawn"] & (g["data_deleted"] == 1)).sum()),
            "client_error_flagged_kept": int((g["client_error_flagged"] & ~g["excluded"]).sum()),
        }
    main_kept = kept.trials
    undefined = main_kept["q_ai"] == main_kept["q_own"]
    report["trial_level"] = {
        "main_trials_analysed": int(len(main_kept)),
        "woa_undefined_q_ai_eq_q_own": int(undefined.sum()),
        "woa_undefined_by_condition": {
            c: int(v) for c, v in main_kept.assign(u=undefined)
            .merge(kept.participants[["participant_id", "condition"]], on="participant_id")
            .groupby("condition")["u"].sum().items()},
        "note": "dropped from Weight-of-Advice analyses only; kept for all other measures",
    }
    return kept, flags, report
