"""Data-integrity check for a logged session. See docs/RESULTS.md, section 11.

The plan calls this the highest-value hour in the schedule, and the reason is
blunt: with 80 participants run in a single batch there is no second attempt at
the data. A field that is silently wrong is discovered during analysis, when
the money is spent and the participants are gone.

Run after every session, and before collection opens:

    python src/interface/verify_log.py --db data/dryrun.sqlite3 --participant DRYRUN-01

Checks are grouped. A FAIL means the dataset is not analysable as it stands; a
WARN means look at it before trusting the number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import SupplyChainTwin, TwinParams  # noqa: E402
from policy import is_correct, reference_order  # noqa: E402
from interface import db  # noqa: E402
from interface.advice import (  # noqa: E402
    RELIABILITY_LEVELS,
    placement_of,
    response_class,
    weight_of_advice,
)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, group, msg):
        self.rows.append((status, group, msg))

    def ok(self, group, msg):   self.add(PASS, group, msg)
    def warn(self, group, msg): self.add(WARN, group, msg)
    def fail(self, group, msg): self.add(FAIL, group, msg)

    def check(self, cond, group, ok_msg, fail_msg, hard=True):
        if cond:
            self.ok(group, ok_msg)
        else:
            (self.fail if hard else self.warn)(group, fail_msg)

    def render(self) -> int:
        width = max(len(g) for _, g, _ in self.rows) + 2
        last = None
        for status, group, msg in self.rows:
            head = group if group != last else ""
            last = group
            mark = {PASS: "  ok ", WARN: "WARN ", FAIL: "FAIL "}[status]
            print("%s%s%s" % (mark, head.ljust(width), msg))
        n_fail = sum(1 for s, _, _ in self.rows if s == FAIL)
        n_warn = sum(1 for s, _, _ in self.rows if s == WARN)
        print("\n%d checks, %d failed, %d warnings" % (len(self.rows), n_fail, n_warn))
        return n_fail


def replay_main(p, main):
    """Rebuild the episode from the logged orders and compare it with the log.

    `p` is the participant row and `main` the main-phase trial rows in
    trial_index order (anything indexable by column name: sqlite3.Row, the
    Postgres row wrapper, or a dict). Returns (drift, risk_mismatch,
    live_risk): `drift` is empty when every main trial replays exactly;
    `risk_mismatch` lists (trial, logged, recomputed) risk % disagreements
    for app >= 1.2.0 (`live_risk`). Shared with analysis/exclusions.py so the
    exclusion rule and this check can never disagree about "replays".
    """
    params = TwinParams(**json.loads(p["params_json"]))
    # Sessions from app 1.2.0 on replay real M5 demand (interface/world.py);
    # earlier logs were produced on the synthetic process and replay on it.
    version = tuple(int(x) for x in str(p["app_version"]).split(".")[:3])
    if version >= (1, 2, 0):
        from interface.world import make_twin
        twin = make_twin(params, p["seed"], "main")
    else:
        twin = SupplyChainTwin(params=params, seed=p["seed"])
    st = twin.reset()
    drift = []
    # Training/serving skew check (ML Test Score): from app 1.2.0 the risk % on
    # screen is live LightGBM output, so recompute it from the replayed history
    # and require the logged value exactly. Catches a wrong model file or a
    # LightGBM version mismatch in the deployed container.
    live_risk = version >= (1, 2, 0)
    if live_risk:
        from live_model import state_row
        from interface.advice import _risk_display
    history, risk_mismatch = [], []
    for t in main:
        if (st.inventory != t["inventory"] or st.on_order != t["on_order"]
                or st.backorders != t["backorders"] or st.lead_time != t["lead_time"]):
            drift.append(t["trial_index"])
            break
        if reference_order(st, params) != t["q_star"]:
            drift.append(("q_star", t["trial_index"]))
            break
        if st.demand != t["demand"]:
            drift.append(("state_demand", t["trial_index"]))
            break
        if live_risk:
            history.append(state_row(st))
            # Same rule as advice.make_advice: on a correct week the AI reads the
            # true state (its order differs from Q* only by in-band jitter).
            believed = t["q_star"] if t["ai_correct"] else t["q_ai"]
            shown = _risk_display(history, t["q_star"], believed)[0]
            if shown != t["ai_risk_pct"]:
                risk_mismatch.append((t["trial_index"], t["ai_risk_pct"], shown))
        st, cost, rec = twin.step(t["q_final"])
        # `demand` is what the participant saw (D_{t-1}); `demand_realised` is
        # what the step produced (D_t). Comparing the wrong pair is how the
        # missing column stayed invisible until the first dry run.
        if abs(cost - t["cost"]) > 1e-6 or rec["demand"] != t["demand_realised"]:
            drift.append(("outcome", t["trial_index"]))
            break
    return drift, risk_mismatch, live_risk


def verify(db_path, participant_id: str, n_practice=5, n_main=30) -> int:
    """`db_path`: a SQLite file, a postgresql:// URL, or None for the study
    database (DATABASE_URL if set, else the default SQLite file) -- the same
    resolution as db.connect, so the check reads exactly what the app wrote."""
    conn = db.connect(db_path)
    r = Report()

    p = conn.execute("SELECT * FROM participant WHERE participant_id=?",
                     (participant_id,)).fetchone()
    if p is None:
        print("no such participant: %s" % participant_id)
        return 1
    condition = p["condition"]

    trials = conn.execute(
        "SELECT * FROM trial WHERE participant_id=? ORDER BY phase DESC, trial_index",
        (participant_id,)).fetchall()
    main = [t for t in trials if t["phase"] == "main"]
    prac = [t for t in trials if t["phase"] == "practice"]

    # ---------------------------------------------------------- completeness
    g = "completeness"
    r.check(len(prac) == n_practice, g,
            "%d practice trials" % len(prac),
            "expected %d practice trials, found %d" % (n_practice, len(prac)))
    r.check(len(main) == n_main, g,
            "%d main trials" % len(main),
            "expected %d main trials, found %d" % (n_main, len(main)))
    r.check(p["consented"] == 1, g, "consent recorded", "consent flag not set")
    r.check(p["finished_utc"] is not None, g,
            "session marked finished", "session never marked finished", hard=False)
    r.check(p["params_json"], g, "TwinParams snapshot stored for replay",
            "params_json empty -- session cannot be replayed")

    idx = [t["trial_index"] for t in main]
    r.check(idx == list(range(n_main)), g,
            "main trial indices contiguous 0..%d" % (n_main - 1),
            "main trial indices are not contiguous: %s" % idx)

    # NOT NULL is enforced by the schema, but a NULL WoA is legal and must be
    # legal only for the one defined reason.
    bad_null = [t["trial_index"] for t in main
                if t["woa"] is None and t["q_ai"] != t["q_own"]]
    r.check(not bad_null, g, "WoA null only where q_ai == q_own",
            "WoA null on trials where it is defined: %s" % bad_null)

    # ---------------------------------------------------------- manipulation
    g = "manipulation"
    nominal = RELIABILITY_LEVELS[condition]
    realised = np.mean([t["ai_correct"] for t in main])
    r.check(abs(realised - nominal) < 1e-9, g,
            "realised reliability %.0f%% == nominal %.0f%% (%s condition)"
            % (100 * realised, 100 * nominal, condition),
            "realised reliability %.1f%% != nominal %.0f%%"
            % (100 * realised, 100 * nominal))

    # Which error design produced this log (recorded at session start from
    # app 1.3.0; earlier logs are v1).
    ev = conn.execute("SELECT detail FROM event_log WHERE participant_id=? AND kind='session_start' "
                      "ORDER BY id LIMIT 1", (participant_id,)).fetchone()
    design = (json.loads(ev["detail"]).get("design") if ev and ev["detail"] else None) or "v1"

    errs = [t for t in main if not t["ai_correct"]]
    by_place = {"absorbed": 0, "propagating": 0, "neutral": 0}
    for t in errs:
        by_place[t["placement"]] += 1
    r.check(by_place["neutral"] == 0, g, "no error falls on a neutral week (%s design)" % design,
            "%d errors on neutral weeks" % by_place["neutral"])
    by_place.pop("neutral")
    r.check(errs and min(by_place.values()) > 0, g,
            "errors split across placements: %d absorbed / %d propagating"
            % (by_place["absorbed"], by_place["propagating"]),
            "errors not split across both placements: %s" % by_place)

    mismatched = [t["trial_index"] for t in main
                  if t["placement"] != placement_of(t["period"], design)]
    r.check(not mismatched, g, "placement label matches the %s design" % design,
            "placement disagrees with calendar on trials %s" % mismatched)

    # Direction of error encodes the propagation mechanism (docs/RESULTS.md, section 9):
    # propagating positions get the under-order, absorbed get the over-order.
    if design == "v1":
        wrong_dir = [t["trial_index"] for t in errs
                     if (t["placement"] == "propagating" and t["q_ai"] > t["q_star"])
                     or (t["placement"] == "absorbed" and t["q_ai"] < t["q_star"])]
    else:
        # v2: every error is an under-order, never below MIN_ORDER, except the
        # documented fallback when Q* is too small for any plausible under-order.
        from interface.advice import MIN_ORDER
        wrong_dir = [t["trial_index"] for t in errs
                     if (t["q_ai"] > t["q_star"] and t["q_star"] - MIN_ORDER > t["delta"])
                     or t["q_ai"] < MIN_ORDER]
    r.check(not wrong_dir, g, "error direction matches placement on every error trial",
            "error direction wrong on trials %s" % wrong_dir, hard=False)

    # ---------------------------------------------------------- derived fields
    g = "derived fields"
    bad = []
    for t in main:
        expected_woa = weight_of_advice(t["q_own"], t["q_final"], t["q_ai"])
        if expected_woa is None:
            if t["woa"] is not None:
                bad.append(("woa", t["trial_index"]))
        elif abs(t["woa"] - expected_woa) > 1e-9:
            bad.append(("woa", t["trial_index"]))
        if t["response_class"] != response_class(t["q_own"], t["q_final"], t["q_ai"]):
            bad.append(("response_class", t["trial_index"]))
        if t["error_own"] != abs(t["q_own"] - t["q_star"]):
            bad.append(("error_own", t["trial_index"]))
        if t["error_final"] != abs(t["q_final"] - t["q_star"]):
            bad.append(("error_final", t["trial_index"]))
        if t["own_correct"] != int(is_correct(t["q_own"], t["q_star"], t["delta"])):
            bad.append(("own_correct", t["trial_index"]))
        if t["final_correct"] != int(is_correct(t["q_final"], t["q_star"], t["delta"])):
            bad.append(("final_correct", t["trial_index"]))
        cap = json.loads(p["params_json"])["max_order"]
        cap_bound = t["q_star"] > cap and t["q_ai"] == cap
        if t["ai_correct"] != int(cap_bound or is_correct(t["q_ai"], t["q_star"], t["delta"])):
            bad.append(("ai_correct", t["trial_index"]))
    r.check(not bad, g, "WoA, response class, errors and correctness all recompute",
            "recomputation mismatches: %s" % sorted(set(bad))[:10])

    classes = {t["response_class"] for t in main}
    r.check(len(classes) >= 2, g,
            "response classes present: %s" % ", ".join(sorted(classes)),
            "only one response class appears (%s) -- check the elicitation"
            % ", ".join(classes), hard=False)

    # ---------------------------------------------------------- two-stage design
    g = "two-stage"
    identical = sum(1 for t in main if t["q_own"] == t["q_final"])
    r.check(identical < n_main, g,
            "Q_own and Q_final differ on %d/%d trials" % (n_main - identical, n_main),
            "Q_own == Q_final on every trial -- stage 2 may not be recording")
    r.check(all(t["rt_stage1_ms"] > 0 for t in main), g,
            "stage-1 response times all positive",
            "stage-1 response time missing or zero on some trials")
    r.check(all(t["rt_stage2_ms"] > 0 for t in main), g,
            "stage-2 response times all positive",
            "stage-2 response time missing or zero on some trials")
    fast = [t["trial_index"] for t in main if t["rt_stage1_ms"] < 500]
    r.check(not fast, g, "no implausibly fast stage-1 responses",
            "stage-1 under 500 ms on trials %s -- possible click-through" % fast,
            hard=False)

    # ---------------------------------------------------------- replay
    # The strongest check available: rebuild the episode from the logged orders
    # and confirm the twin lands on the same numbers. If this passes, the log
    # is a faithful record of what the participant actually faced.
    g = "replay"
    # Replay against the parameters THIS participant actually ran on, not
    # whatever fitted_params.json happens to hold now. Re-running calibration
    # between a session and its verification would otherwise make the strongest
    # check in this file diverge on a perfectly good log.
    params = TwinParams(**json.loads(p["params_json"]))
    drift, risk_mismatch, live_risk = replay_main(p, main)
    r.check(not drift, g,
            "all %d main trials replay exactly from the logged orders" % n_main,
            "replay diverges at %s -- the log does not reproduce the session" % drift)
    if live_risk:
        r.check(not risk_mismatch and not drift, g,
                "risk %% shown on all %d main trials matches the live model recomputed offline" % n_main,
                "risk %% differs from the offline model (trial, logged, recomputed): %s"
                % (risk_mismatch[:5] if risk_mismatch else "replay stopped early"))

    # ---------------------------------------------------------- questionnaires
    g = "questionnaires"
    blocks = conn.execute(
        """SELECT block, instrument, COUNT(*) n FROM block_measure
           WHERE participant_id=? GROUP BY block, instrument ORDER BY block""",
        (participant_id,)).fetchall()
    got = {(b["block"], b["instrument"]): b["n"] for b in blocks}
    for blk in range(3):
        r.check(got.get((blk, "tia"), 0) >= 6, g,
                "block %d trust (TiA) recorded, %d items" % (blk, got.get((blk, "tia"), 0)),
                "block %d trust missing or incomplete" % blk, hard=False)
        r.check(got.get((blk, "tlx"), 0) >= 5, g,
                "block %d workload (TLX) recorded, %d items" % (blk, got.get((blk, "tlx"), 0)),
                "block %d workload missing or incomplete" % blk, hard=False)
    r.check(p["perceived_reliability"] is not None, g,
            "manipulation check recorded (perceived reliability = %s)"
            % p["perceived_reliability"],
            "manipulation check not recorded", hard=False)

    # ---------------------------------------------------------- ranges
    g = "ranges"
    r.check(all(0 <= t["service_level"] <= 1 for t in main), g,
            "service level within [0, 1]", "service level out of range")
    r.check(all(t["cost"] >= 0 for t in main), g,
            "costs non-negative", "negative cost logged")
    r.check(all(0 <= t["q_final"] <= params.max_order for t in main), g,
            "orders within [0, %d]" % params.max_order, "order outside the legal range")
    r.check(all(t["inventory_close"] >= 0 and t["backorders_close"] >= 0 for t in main), g,
            "inventory and backorders non-negative (no double-counted shortage)",
            "negative inventory or backorders -- the accounting bug is back")

    conn.close()
    return r.render()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # No --db: the study database -- Cloud SQL when DATABASE_URL is set, else
    # data/trials.sqlite3. --db also accepts a postgresql:// URL.
    ap.add_argument("--db", default=None,
                    help="SQLite file or postgresql:// URL (default: DATABASE_URL, "
                         "else data/trials.sqlite3)")
    ap.add_argument("--participant", required=True)
    ap.add_argument("--practice", type=int, default=5)
    ap.add_argument("--main", type=int, default=30)
    args = ap.parse_args()

    where = args.db or ("DATABASE_URL (%s)" % db.backend() if db.database_url()
                        else str(db.DEFAULT_DB))
    # Never echo a URL: it carries the database password.
    if db._is_pg_url(where):
        where = "postgres (--db URL)"
    print("verifying %s  participant=%s\n" % (where, args.participant))
    n_fail = verify(args.db, args.participant, args.practice, args.main)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
