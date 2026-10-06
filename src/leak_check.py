"""
Can a participant tell a wrong recommendation from what the screen shows?

Simulates scripted sessions (the app's own dry run), then asks how well a
simple observer can detect AI-error weeks from the displayed information:
the recommended order, the state panel, and the risk %. AUROC 0.5 means the
display gives nothing away beyond what reasoning about the state provides.

Audit, 24 Sep 2026: with the old formula risk %, adding the risk lifted a
linear observer from 0.544 to 0.657 -- the risk was computed from the CORRECT
order even on error weeks, so it contradicted wrong advice. The live LightGBM
risk is computed on the state the AI believes, and should close that gap.

Run:  python src/leak_check.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from interface.app import dry_run  # noqa: E402

N_SESSIONS = 120
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "leak_check.json")


def main():
    db = os.path.join(tempfile.gettempdir(), "leak_check_v2.sqlite3")
    for f in (db, db + "-wal", db + "-shm"):
        if os.path.exists(f):
            os.remove(f)
    for i in range(N_SESSIONS):
        dry_run(db_path=db, participant_id="LK%03d" % i,
                condition="low" if i % 2 else "high", verbose=False)
    t = pd.read_sql("SELECT * FROM trial WHERE phase='main'", sqlite3.connect(db))
    t["ip"] = t.inventory + t.on_order - t.backorders
    y = 1 - t.ai_correct.values
    groups = t.participant_id.values
    state = ["q_ai", "demand", "inventory", "on_order", "lead_time", "backorders", "ip"]

    def auc(cols):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        p = cross_val_predict(m, t[cols], y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))

    res = {"order_only": auc(["q_ai"]), "order_state": auc(state),
           "order_risk": auc(["q_ai", "ai_risk_pct"]), "order_state_risk": auc(state + ["ai_risk_pct"])}
    e, ok = t[t.ai_correct == 0], t[t.ai_correct == 1]
    res["mean_risk"] = {"correct": float(ok.ai_risk_pct.mean()),
                        "under_order_errors": float(e[e.q_ai < e.q_star].ai_risk_pct.mean()),
                        "over_order_errors": float(e[e.q_ai > e.q_star].ai_risk_pct.mean())}
    res["n_trials"] = int(len(t))
    for k in ("order_only", "order_state", "order_risk", "order_state_risk"):
        print("  detect AI error from %-18s AUROC %.3f" % (k.replace("_", " + "), res[k]))
    print("  mean risk shown: correct %.1f | under-order errors %.1f | over-order errors %.1f"
          % tuple(res["mean_risk"].values()))
    print("  gap added by the risk %% (linear observer): %+.3f   [was +0.113 with the formula]"
          % (res["order_state_risk"] - res["order_state"]))
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
