"""Read the trial-level study database into tidy pandas DataFrames.

Works on both backends through interface.db.connect: a SQLite file path, or
PostgreSQL when DATABASE_URL is set (pass nothing, or a postgres:// URL).
The analysis never writes to the database.

    participants   one row per participant row (including withdrawn ones)
    trials_all     every trial row, practice and main
    trials         MAIN phase only -- what every confirmatory analysis uses
    block          block_measure rows in long form (instrument, item, response)
    events         event_log rows, `detail` JSON-decoded into `detail_obj`
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

import analysis  # noqa: F401  (puts src/ on sys.path)
from interface import db

SIMULATED_EVENT = "simulated_participant"


@dataclass
class StudyData:
    participants: pd.DataFrame
    trials_all: pd.DataFrame
    trials: pd.DataFrame
    block: pd.DataFrame
    events: pd.DataFrame
    source: str

    @property
    def is_simulated(self) -> bool:
        """True when any participant was written by simulate_participants.py.

        Decided from the data itself (a `simulated_participant` event), not
        from a flag the caller passes, so a real-data run can never be
        labelled simulated by mistake and vice versa.
        """
        return bool((self.events["kind"] == SIMULATED_EVENT).any()) if len(self.events) else False


def _frame(conn, table: str, order_by: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM %s" % table
    if order_by:
        sql += " ORDER BY " + order_by
    rows = conn.execute(sql).fetchall()
    cols = db.table_columns(conn, table)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([tuple(r) for r in rows], columns=list(rows[0].keys()))


def load(target=None) -> StudyData:
    """Load everything. `target`: SQLite path, postgres URL, or None (the
    study database: DATABASE_URL if set, else data/trials.sqlite3)."""
    conn = db.connect(target)
    try:
        participants = _frame(conn, "participant", "started_utc, participant_id")
        trials_all = _frame(conn, "trial", "participant_id, phase, trial_index")
        block = _frame(conn, "block_measure", "participant_id, block, instrument, item")
        events = _frame(conn, "event_log", "id")
    finally:
        conn.close()

    for col in ("q_ai", "q_own", "q_final", "q_star", "delta", "ai_correct",
                "own_correct", "final_correct", "rt_stage1_ms", "rt_stage2_ms",
                "trial_index", "block", "period", "stockout", "backorders_close"):
        if col in trials_all:
            trials_all[col] = pd.to_numeric(trials_all[col])
    for col in ("woa", "cost", "service_level"):
        if col in trials_all:
            trials_all[col] = pd.to_numeric(trials_all[col], errors="coerce")
    if len(block):
        block["response"] = pd.to_numeric(block["response"])
        block["block"] = pd.to_numeric(block["block"])
    events["detail_obj"] = [json.loads(d) if isinstance(d, str) and d else None
                            for d in events.get("detail", pd.Series(dtype=object))]
    for col in ("excluded", "data_deleted", "perceived_reliability", "numeracy_score"):
        if col in participants:
            participants[col] = pd.to_numeric(participants[col], errors="coerce")

    trials = trials_all[trials_all["phase"] == "main"].copy()
    trials = trials.sort_values(["participant_id", "trial_index"]).reset_index(drop=True)
    return StudyData(participants=participants.reset_index(drop=True),
                     trials_all=trials_all.reset_index(drop=True),
                     trials=trials, block=block.reset_index(drop=True),
                     events=events.reset_index(drop=True),
                     # Never echo a postgres URL: it carries the password.
                     source=("postgres" if db.backend(target) == "postgres"
                             else str(target if target is not None else db.DEFAULT_DB)))
