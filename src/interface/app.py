"""Participant interface for the reliance-under-disruption experiment.

Run:
    python src/interface/app.py --port 8000
    python src/interface/app.py --dry-run          # scripted 30-trial self-test

The trial loop, per plan section 07:

    1. state         seven fields of twin status
    2. Q_own         the participant's own order, BEFORE any advice
    3. AI advice     risk %, confidence level, Q_AI revealed
    4. Q_final       accept / modify / override
    5. execute       the twin applies the order
    6. outcome       cost and stockout shown back

Step 2 is the whole reason this file exists. Without it neither Weight of
Advice nor RAIR/RSR is computable, "modify" has no measure, and there is no
within-subject unaided baseline.

Design notes that are not obvious:

* Response times are measured in the BROWSER and sent up, not measured on the
  server. Server-side timing would fold network latency into a dependent
  variable, and stage-1 time is exactly the measure most sensitive to it.
* Stage 1 is irreversible. Once Q_own is submitted the field locks, because a
  participant who revises it after seeing the advice destroys the baseline.
* The twin lives in memory but every order is logged, so `_rehydrate` can
  replay a session exactly if the process restarts mid-collection (a Cloud
  Run instance is replaced, a deploy lands mid-session). A submitted-but-not-
  yet-completed stage 1 is persisted too (`pending_stage1`), so a restart
  between the two stages cannot hand the participant a fresh stage 1 after
  they have already seen the advice.

Deployment (Cloud Run behind Firebase Hosting, Cloud SQL):

    gunicorn --chdir src interface.app:app

    DATABASE_URL     postgresql://... -> Postgres; unset -> SQLite (db.py)
    APP_SECRET_KEY   required when PRODUCTION=1 (signs the session cookie)
    PRODUCTION=1     secure cookies; refuse to boot without APP_SECRET_KEY
    DESKTOP_ONLY=0   switch off the server-side desktop gate (on by default)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, jsonify, render_template, request, session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import SupplyChainTwin, TwinParams  # noqa: E402
from policy import cost_derived_delta, is_correct, reference_order  # noqa: E402
from interface import db, numeracy  # noqa: E402
from interface.world import make_twin, session_params  # noqa: E402
from live_model import state_row  # noqa: E402
from interface.advice import (  # noqa: E402
    ADVICE_DESIGN,
    Advice,
    RELIABILITY_LEVELS,
    build_schedule,
    make_advice,
    participant_seed,
    response_class,
    weight_of_advice,
)

APP_VERSION = "1.3.0"
N_PRACTICE = 5
N_MAIN = 30
BLOCK_SIZE = 10

PRODUCTION = os.environ.get("PRODUCTION") == "1"

app = Flask(__name__)
# A fresh key per process unless one is supplied. A literal key committed to a
# repo that ships with the paper is a published key. A random one is fine for a
# local run, but NOT in production: sessions now survive a restart
# (`_rehydrate`), and they can only do so if the cookie that names the
# participant still verifies -- a per-process key would void every
# participant's cookie each time Cloud Run replaces an instance.
if PRODUCTION and not os.environ.get("APP_SECRET_KEY"):
    raise RuntimeError("PRODUCTION=1 requires APP_SECRET_KEY: without a fixed key every "
                       "participant's session cookie dies with the instance")
app.secret_key = os.environ.get("APP_SECRET_KEY") or secrets.token_hex(32)
# Firebase Hosting strips every cookie except one named `__session` before it
# forwards a request to Cloud Run. Under any other name the participant would
# arrive at every API call with no session at all.
app.config["SESSION_COOKIE_NAME"] = "__session"
# Lax: sent on same-site requests and top-level navigations (the Prolific link),
# never on cross-site sub-requests, so another site cannot drive a trial.
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
if PRODUCTION:
    # HTTPS only. Firebase Hosting always serves HTTPS; a cookie that could
    # also travel over plain HTTP could be read off the wire.
    app.config["SESSION_COOKIE_SECURE"] = True

# Participant ids are typed by hand from a recruitment page, so they are short
# and guessable by construction. Constrain the shape and cap how many distinct
# ones can ever allocate a Session: every new id opens a SQLite connection that
# lives for the process, and -- worse -- inserts a `participant` row, which
# shifts the high/low alternation below for every genuine participant after it.
_PID_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
_MAX_SESSIONS = 200

# Server-side copies of the questionnaire item names rendered in task.html.
# The client may not invent item keys: INSERT OR REPLACE would silently
# overwrite a real answer with a fabricated one.
TIA_ITEMS = tuple("tia_%d" % i for i in range(1, 7))
TLX_ITEMS = ("mental", "temporal", "performance", "effort", "frustration")
MAX_RT_MS = 600_000

# Shown on the debrief so a Prolific participant can claim payment. Set it in
# the environment when the study goes live; it is never hard-coded here.
COMPLETION_CODE = os.environ.get("PROLIFIC_COMPLETION_CODE", "")

# Desktop-only gate. The task's state panel, trend lines and two-stage entry
# were designed and piloted at desktop width with a mouse; a phone layout would
# change the stimulus, and response times from touch typing are not comparable.
# The start page checks this too, but the client can be bypassed, so the server
# decides. DESKTOP_ONLY=0 switches it off (demos on a tablet).
MIN_VIEWPORT_W = 1000
_MOBILE_UA_RE = re.compile(r"Mobi|Android|iPhone|iPad")
DESKTOP_REQUIRED_MSG = ("This study has to be done on a desktop or laptop computer "
                        "with a mouse and a browser window at least %d pixels wide. "
                        "Please open the study link on a computer." % MIN_VIEWPORT_W)

_SESSIONS: dict[str, "Session"] = {}
_LOCK = Lock()


def _token_hash(token: str) -> str:
    """What the database keeps of a resume token. Never the token itself."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class _ReplayMismatch(RuntimeError):
    """A rehydrated session does not reproduce its own log."""


class Session:
    """One participant's live state. Server-side; the browser holds only an id."""

    def __init__(self, participant_id: str, condition: str, params: TwinParams,
                 db_path=None, consented: bool = False, *, conn=None,
                 design: str | None = None, register: bool = True):
        # `register=False` (with `conn` and `design`) is the rehydration path:
        # build the same in-memory objects a fresh session would have, but
        # write nothing -- the participant row and session_start event exist.
        self.participant_id = participant_id
        self.condition = condition
        self.params = params
        self.seed = participant_seed(participant_id)
        # Serialises stage1/stage2 for THIS participant. app.run(threaded=True)
        # means a double-click, an auto-retry on a flaky connection, or a second
        # tab can deliver two concurrent requests; without this both can pass the
        # `pending is None` check, step the twin twice, and then collide on the
        # trial primary key -- after the twin has already advanced.
        self.lock = Lock()
        # Binds a browser to this session. Participant ids are handed out in
        # plain text and are guessable, so the id alone must not be enough to
        # resume, overwrite or race someone else's session.
        self.token = secrets.token_urlsafe(32)
        self.rng = np.random.default_rng(self.seed)
        # None -> the study database (DATABASE_URL, else the DEFAULT_DB file).
        self.conn = conn if conn is not None else db.connect(db_path)

        self.phase = "practice"
        self.trial_index = 0
        # Practice always uses the high-reliability schedule: practice exists to
        # teach the interface, not to start the manipulation early.
        # The error design (advice.ADVICE_DESIGN, v2 by default) is fixed per
        # session and recorded in the event log, so a log always says which
        # design produced it.
        self.design = design or ADVICE_DESIGN
        self.practice_schedule = build_schedule(participant_id + "/practice", "high", N_PRACTICE,
                                                design=self.design)
        self.main_schedule = build_schedule(participant_id, condition, N_MAIN, design=self.design)

        # Real M5 demand, identical for every participant (interface/world.py).
        self.twin = make_twin(params, self.seed, "practice")
        self.state = self.twin.reset()
        # This phase's weekly states, oldest first -- the live risk model's input.
        self.history = [state_row(self.state)]
        self.pending = None  # set by stage 1, consumed by stage 2

        if not register:
            return
        db.register_participant(
            self.conn, participant_id, condition, self.seed,
            asdict(params), APP_VERSION,
        )
        # Only the hash of the resume token is stored: it is what lets this
        # browser, and only this browser, pick the session up after a restart.
        self.conn.execute(
            "UPDATE participant SET consented=?, token_sha256=? WHERE participant_id=?",
            (int(consented), _token_hash(self.token), participant_id),
        )
        db.log_event(self.conn, participant_id, "session_start",
                     {"condition": condition, "version": APP_VERSION,
                      "design": self.design,
                      "ui": "3d" if _warehouse_3d() else "panel"})

    # ---------- phase bookkeeping ----------
    @property
    def n_trials(self) -> int:
        return N_PRACTICE if self.phase == "practice" else N_MAIN

    @property
    def schedule(self) -> list[bool]:
        return self.practice_schedule if self.phase == "practice" else self.main_schedule

    @property
    def block(self) -> int:
        return -1 if self.phase == "practice" else self.trial_index // BLOCK_SIZE

    @property
    def finished(self) -> bool:
        return self.phase == "main" and self.trial_index >= N_MAIN

    def begin_main(self) -> None:
        """Reset the twin so the main phase starts from a clean warehouse.

        Practice must not leave the participant in a hole -- a starved opening
        inventory makes every later disruption invisible against constant
        stockouts, which is the failure mode TwinParams.initial_inventory was
        chosen to avoid.
        """
        # Once the main phase has a logged trial or a submitted stage 1, a
        # second begin-main would restart it on a fresh warehouse at trial 0 and
        # the next stage 2 would collide with the logged trial key, leaving the
        # participant stuck. A repeat at the boundary (nothing yet in main) stays
        # harmless and is still allowed.
        if self.phase == "main" and (self.trial_index > 0 or self.pending is not None):
            raise ValueError("the main phase has already started")
        # A stage 1 abandoned by jumping to the main phase took one draw from
        # self.rng (make_advice) but will never produce a trial row. Rehydration
        # replays one draw per logged trial, so it would otherwise come up one
        # draw short and every later advice would differ. Record the abandoned
        # draw in the event, so the replay can take it too (and check it).
        abandoned = None
        if self.pending is not None:
            abandoned = {"phase": self.phase, "trial_index": self.trial_index,
                         "advice": self.pending["advice"].record()}
        self._enter_main()
        # A stage 1 abandoned by jumping to the main phase must not survive in
        # the database either, or a restart would try to restore it.
        self.conn.execute("DELETE FROM pending_stage1 WHERE participant_id=?",
                          (self.participant_id,))
        # This event is also how `_rehydrate` knows the main phase had begun
        # before any main trial was recorded.
        db.log_event(self.conn, self.participant_id, "main_phase_start",
                     {"abandoned_stage1": abandoned} if abandoned else None)

    def _enter_main(self) -> None:
        """The in-memory half of begin_main; rehydration replays only this."""
        self.phase = "main"
        self.trial_index = 0
        self.twin = make_twin(self.params, self.seed, "main")
        self.state = self.twin.reset()
        self.history = [state_row(self.state)]
        self.pending = None
        # NOTE: self.rng is deliberately NOT reset. It has been advancing since
        # practice trial 0, and rehydration reproduces that by replaying the
        # practice advice before the main advice.

    # ---------- the two stages ----------
    def stage1(self, q_own: int, rt_ms: int) -> dict:
        with self.lock:
            if self.finished:
                raise ValueError("this session has already finished")
            if self.pending is not None:
                raise ValueError("stage 1 already submitted for this trial")
            correct = self.schedule[self.trial_index]
            # make_advice draws from self.rng. If persisting the stage fails,
            # the draw must be undone too, or the retry gets different advice
            # and a rehydrated session (which replays one draw per trial) would
            # no longer reproduce this one.
            rng_state = copy.deepcopy(self.rng.bit_generator.state)
            adv = make_advice(self.state, self.params, self.trial_index, correct, self.rng,
                              history=self.history, design=self.design)
            try:
                # Persisted BEFORE the advice is returned: once the participant
                # can see Q_AI, the locked Q_own has to exist somewhere a
                # restart cannot erase.
                db.save_pending_stage1(self.conn, self.participant_id, self.phase,
                                       self.trial_index, q_own, rt_ms, adv.record())
            except Exception:
                self.rng.bit_generator.state = rng_state
                raise
            self.pending = {"q_own": int(q_own), "rt1": int(rt_ms), "advice": adv}
            return adv.public()

    def stage2(self, q_final: int, rt_ms: int) -> dict:
        # Same serialisation as stage1. The whole read-modify-write of the twin
        # plus the trial insert has to be one atomic unit, or the logged trial
        # index and the twin's internal period drift apart permanently.
        with self.lock:
            return self._stage2_locked(q_final, rt_ms)

    def _stage2_locked(self, q_final: int, rt_ms: int) -> dict:
        if self.pending is None:
            raise ValueError("stage 1 has not been submitted")

        q_own = self.pending["q_own"]
        adv = self.pending["advice"]
        q_final = int(np.clip(int(q_final), 0, self.params.max_order))

        state_before = self.state
        # The twin advances BEFORE the row is written, so a failed write would
        # otherwise leave the simulator a period ahead of the log with `pending`
        # still set -- and every retry would burn another period and another
        # demand draw, silently. Snapshot first, roll back if the write fails.
        snapshot = copy.deepcopy(self.twin)
        _, cost, record = self.twin.step(q_final)

        row = {
            "participant_id": self.participant_id,
            "trial_index": self.trial_index,
            "phase": self.phase,
            "block": self.block,
            "period": state_before.period,
            "demand": state_before.demand,
            "inventory": state_before.inventory,
            "on_order": state_before.on_order,
            "lead_time": state_before.lead_time,
            "backorders": state_before.backorders,
            "supplier_reliability": state_before.supplier_reliability,
            "transport_ok": int(state_before.transport_ok),
            "disruption": state_before.disruption.value,
            "q_ai": adv.q_ai,
            "ai_risk_pct": adv.risk_pct,
            "ai_level": adv.level,
            "ai_correct": int(adv.ai_correct),
            "placement": adv.placement,
            "q_star": adv.q_star,
            "delta": adv.delta,
            "q_own": q_own,
            "q_final": q_final,
            "response_class": response_class(q_own, q_final, adv.q_ai),
            "woa": weight_of_advice(q_own, q_final, adv.q_ai),
            "error_own": abs(q_own - adv.q_star),
            "error_final": abs(q_final - adv.q_star),
            "own_correct": int(is_correct(q_own, adv.q_star, adv.delta)),
            "final_correct": int(is_correct(q_final, adv.q_star, adv.delta)),
            "rt_stage1_ms": self.pending["rt1"],
            "rt_stage2_ms": int(rt_ms),
            "demand_realised": record["demand"],
            "cost": cost,
            "stockout": int(record["stockout"]),
            "service_level": record["service_level"],
            "shipped": record["shipped"],
            "arrivals": record["arrivals"],
            "inventory_close": record["inventory_close"],
            "backorders_close": record["backorders_close"],
            "submitted_utc": db.utc_now(),
        }
        try:
            # Trial row + removal of the pending stage 1, in one transaction.
            db.record_trial(self.conn, row)
        except Exception:
            # Put the twin back exactly where it was and leave `pending` set, so
            # the participant can retry the same trial rather than losing it.
            self.twin = snapshot
            raise

        self.trial_index += 1
        self.state = self.twin.observe()
        self.history.append(state_row(self.state))
        self.pending = None

        # The participant sees consequence, never ground truth: showing Q* would
        # turn the task into supervised training and contaminate later trials.
        return {
            "demand": record["demand"],
            "cost": round(cost, 2),
            "stockout": bool(record["stockout"]),
            "service_level": round(record["service_level"], 3),
            "inventory_close": record["inventory_close"],
            "backorders_close": record["backorders_close"],
            "arrivals": record["arrivals"],
            "shipped": record["shipped"],
            "total_cost": round(self.twin.total_cost(), 2),
        }

    def state_payload(self) -> dict:
        s = self.state
        return {
            "phase": self.phase,
            "trial": self.trial_index + 1,
            "n_trials": self.n_trials,
            "block": self.block,
            "period": s.period,
            "demand": s.demand,
            "inventory": s.inventory,
            "on_order": s.on_order,
            "lead_time": s.lead_time,
            "backorders": s.backorders,
            "supplier_reliability": s.supplier_reliability,
            "transport_ok": s.transport_ok,
            "disruption": s.disruption.value,
            "inventory_position": s.inventory_position,
            "max_order": self.params.max_order,
            "total_cost": round(self.twin.total_cost(), 2),
            "needs_block_measure": (
                self.phase == "main"
                and self.trial_index > 0
                and self.trial_index % BLOCK_SIZE == 0
                and self.trial_index < N_MAIN
            ),
            "finished": self.finished,
            # Non-null only when stage 1 of the current trial is already locked
            # (a reload, or a server restart, between the two stages). The page
            # must then go straight to stage 2 with this advice: a fresh stage-1
            # entry would be refused ("already submitted").
            "pending_advice": (self.pending["advice"].public()
                               if self.pending is not None else None),
            # Past weeks of THIS phase only -- values the participant has already
            # seen on earlier screens, drawn as trend lines to spare working
            # memory. Nothing about future weeks (disruptions, Q*) is ever sent.
            "history": {
                "demand": [int(r["demand"]) for r in self.history],
                "inventory": [int(r["inventory"]) for r in self.history],
                "backorders": [int(r["backorders"]) for r in self.history],
            },
            "past_stockouts": [bool(r["stockout"]) for r in self.twin.history],
        }


    # ---------- rehydration ----------
    def _replay(self, rows) -> None:
        """Re-run this phase's logged trials through the same code path.

        Each trial calls make_advice exactly as stage1 did, so self.rng takes
        the same draws in the same order and ends where the live session left
        it; then the twin steps with the logged final order. Everything the
        participant saw or the analysis uses is compared with the log on the
        way. Any difference means the session cannot be reproduced, and it is
        refused rather than continued on a silently different world.
        """
        for t in rows:
            st = self.state
            if t["trial_index"] != self.trial_index:
                raise _ReplayMismatch("gap in %s trials at %d" % (self.phase, self.trial_index))
            seen = (st.period, st.demand, st.inventory, st.on_order, st.lead_time,
                    st.backorders)
            logged = (t["period"], t["demand"], t["inventory"], t["on_order"],
                      t["lead_time"], t["backorders"])
            if seen != logged:
                raise _ReplayMismatch("state differs at %s trial %d"
                                      % (self.phase, self.trial_index))
            adv = make_advice(st, self.params, self.trial_index,
                              self.schedule[self.trial_index], self.rng,
                              history=self.history, design=self.design)
            if (adv.q_ai, adv.risk_pct, adv.level, int(adv.ai_correct), adv.placement,
                    adv.q_star, adv.delta) != (
                    t["q_ai"], t["ai_risk_pct"], t["ai_level"], t["ai_correct"],
                    t["placement"], t["q_star"], t["delta"]):
                raise _ReplayMismatch("advice differs at %s trial %d"
                                      % (self.phase, self.trial_index))
            _, cost, record = self.twin.step(int(t["q_final"]))
            if abs(cost - t["cost"]) > 1e-6 or record["demand"] != t["demand_realised"]:
                raise _ReplayMismatch("outcome differs at %s trial %d"
                                      % (self.phase, self.trial_index))
            self.trial_index += 1
            self.state = self.twin.observe()
            self.history.append(state_row(self.state))

    def _replay_abandoned(self, abandoned: dict) -> None:
        """Re-take the rng draw of a stage 1 that begin_main discarded."""
        if (abandoned["phase"], abandoned["trial_index"]) != (self.phase, self.trial_index):
            raise _ReplayMismatch("abandoned stage 1 is for %s trial %s, replay is at %s %d"
                                  % (abandoned["phase"], abandoned["trial_index"],
                                     self.phase, self.trial_index))
        adv = make_advice(self.state, self.params, self.trial_index,
                          self.schedule[self.trial_index], self.rng,
                          history=self.history, design=self.design)
        if adv != Advice(**abandoned["advice"]):
            raise _ReplayMismatch("abandoned advice differs at %s trial %d"
                                  % (self.phase, self.trial_index))

    def _restore_pending(self, pending: dict) -> None:
        """Re-arm a stage 1 that was submitted before the restart.

        The advice is regenerated, so the rng advances exactly as it did live,
        and must equal the advice stored with the locked Q_own -- the advice
        the participant has already seen.
        """
        if (pending["phase"], pending["trial_index"]) != (self.phase, self.trial_index):
            raise _ReplayMismatch("pending stage 1 is for %s trial %s, session is at %s %d"
                                  % (pending["phase"], pending["trial_index"],
                                     self.phase, self.trial_index))
        adv = make_advice(self.state, self.params, self.trial_index,
                          self.schedule[self.trial_index], self.rng,
                          history=self.history, design=self.design)
        stored = Advice(**pending["advice"])
        if adv != stored:
            raise _ReplayMismatch("pending advice differs at %s trial %d"
                                  % (self.phase, self.trial_index))
        self.pending = {"q_own": int(pending["q_own"]), "rt1": int(pending["rt1"]),
                        "advice": stored}


def _rehydrate(pid: str, token: str | None):
    """Rebuild a participant's Session from the database, or return None.

    Call with _LOCK held. Returns None -- the caller then treats the id as
    unavailable -- when the id is unknown, the participant withdrew, the
    browser's token does not hash to the stored one, or the log does not
    replay. Finished participants ARE restored: a restart between /api/finish
    and /api/debrief must not cost them their completion code.

    What is rebuilt, and from where:
      condition, seed, params  participant row (params_json is the snapshot)
      design                   the session_start event
      phase                    main trials exist, or the main_phase_start event
      schedules                recomputed: deterministic in (id, condition, design)
      twin, state, history     replay of the logged q_final orders, per phase
      rng                      replay of make_advice once per logged trial, plus
                               once for a stage 1 discarded by begin_main
      pending stage 1          the pending_stage1 row
    """
    if not token:
        return None
    conn = db.connect()
    p = conn.execute("SELECT * FROM participant WHERE participant_id=?", (pid,)).fetchone()
    if (p is None or p["withdrawn_utc"] is not None or not p["token_sha256"]
            or not hmac.compare_digest(_token_hash(token), p["token_sha256"])):
        conn.close()
        return None

    ev = conn.execute("SELECT detail FROM event_log WHERE participant_id=? "
                      "AND kind='session_start' ORDER BY id LIMIT 1", (pid,)).fetchone()
    # Same default as verify_log: a log without the field predates app 1.3.0.
    design = (json.loads(ev["detail"]).get("design") if ev and ev["detail"] else None) or "v1"
    # Every main_phase_start, oldest first. Only the first is the real
    # practice->main transition; its detail names a stage 1 that begin_main
    # discarded (its rng draw has to be replayed -- see Session.begin_main).
    starts = [json.loads(r["detail"]) if r["detail"] else {}
              for r in conn.execute("SELECT detail FROM event_log WHERE participant_id=? "
                                    "AND kind='main_phase_start' ORDER BY id",
                                    (pid,)).fetchall()]
    main_started = bool(starts)
    by_phase = {
        ph: conn.execute("SELECT * FROM trial WHERE participant_id=? AND phase=? "
                         "ORDER BY trial_index", (pid, ph)).fetchall()
        for ph in ("practice", "main")
    }
    pending = db.load_pending_stage1(conn, pid)

    params = TwinParams(**json.loads(p["params_json"]))
    sess = Session(pid, p["condition"], params, conn=conn, design=design, register=False)
    try:
        if int(p["seed"]) != sess.seed:
            raise _ReplayMismatch("stored seed does not match the participant id")
        sess._replay(by_phase["practice"])
        if starts and starts[0].get("abandoned_stage1"):
            sess._replay_abandoned(starts[0]["abandoned_stage1"])
        if any(st.get("abandoned_stage1") for st in starts[1:]):
            # begin_main called again from inside the main phase with a stage
            # 1 pending. Where that draw fell relative to the logged main
            # trials is not recoverable, so refuse rather than guess.
            raise _ReplayMismatch("stage 1 abandoned by a repeated begin-main")
        if main_started or by_phase["main"]:
            sess._enter_main()
            sess._replay(by_phase["main"])
        if pending is not None:
            sess._restore_pending(pending)
    except _ReplayMismatch as exc:
        # Loud in the server log and in the event log: this participant cannot
        # continue, and the researcher needs to know why.
        app.logger.error("rehydrate %s failed: %s", pid, exc)
        db.log_event(conn, pid, "rehydrate_failed", {"reason": str(exc)})
        conn.close()
        return None
    sess.token = token
    _SESSIONS[pid] = sess
    db.log_event(conn, pid, "session_rehydrated",
                 {"phase": sess.phase, "trial_index": sess.trial_index,
                  "pending": sess.pending is not None})
    return sess


# ------------------------------------------------------------------ routes
def _require_session() -> Session:
    pid = session.get("participant_id")
    if not pid:
        raise KeyError("no active session")
    sess = _SESSIONS.get(pid)
    if sess is None:
        # Not in memory: the process restarted (or the request reached another
        # instance). Rebuild from the database if the cookie proves it is the
        # browser that started the session.
        with _LOCK:
            sess = _SESSIONS.get(pid) or _rehydrate(pid, session.get("token"))
        if sess is None:
            raise KeyError("no active session")
    # The id alone is not proof of identity -- it is typed in by hand and is
    # guessable. Without this check, entering someone else's id on the start
    # page hands you their live session: their state, their trial record.
    if session.get("token") != sess.token:
        raise KeyError("this session is open in another browser")
    return sess


@app.errorhandler(KeyError)
def _no_session(_):
    return jsonify({"error": "session expired -- reload and re-enter your ID"}), 409


@app.errorhandler(ValueError)
def _bad_request(exc):
    return jsonify({"error": str(exc)}), 400


@app.errorhandler(TypeError)
def _bad_type(_):
    # int(None), int([]) and friends from a malformed JSON body. Without this
    # they escape as an uncaught 500 mid-trial.
    return jsonify({"error": "malformed request"}), 400


@app.route("/favicon.ico")
def favicon():
    # Browsers request this on every page; answer "nothing here" quietly.
    return "", 204


@app.errorhandler(Exception)
def _unexpected(exc):
    # Ordinary HTTP errors (404 for a mistyped URL, 405, ...) keep their own
    # status. Only genuine crashes become the 500 "not saved" message --
    # telling a participant their trial was lost when it wasn't is worse than
    # a plain "not found".
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.description}), exc.code
    # The front end parses every response as JSON. An HTML 500 surfaces to the
    # participant as an unreadable error and, worse, looks like a transient
    # glitch they should click through. Say plainly that nothing was saved.
    app.logger.exception("unhandled error: %s", exc)
    return jsonify({"error": "this trial was not saved -- tell the researcher",
                    "detail": type(exc).__name__}), 500


def _duplicate_trial(_):
    # A duplicate (participant, phase, trial_index) -- or a second pending
    # stage 1 -- means two requests raced. Report it rather than letting a 500
    # hide a desynced twin.
    return jsonify({"error": "this trial was already recorded"}), 409


# One handler per backend's duplicate-key exception (sqlite3 and psycopg).
for _exc in db.INTEGRITY_ERRORS:
    app.register_error_handler(_exc, _duplicate_trial)


@app.after_request
def _no_store_api(resp):
    # Firebase Hosting fronts Cloud Run with a CDN. An API response it cached
    # would serve one participant's state, advice or debrief to another.
    if request.path.startswith("/api/") or request.path in ("/healthz", "/health"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/healthz")
@app.get("/health")
def healthz():
    """Ready = the database answers AND the live risk model is loaded.

    Without the model the first stage 1 of every session would fail; without
    the database no trial can be saved. Either is a reason to take no traffic.
    `/health` is an alias: Cloud Run reserves some paths ending in "z" at its
    front end, so an external probe of /healthz may never reach the app.
    """
    checks = {}
    try:
        conn = db.connect()
        try:
            checks["database"] = conn.execute("SELECT 1").fetchone()[0] == 1
        finally:
            conn.close()
    except Exception as exc:  # report, never raise: this is the probe
        app.logger.error("healthz database: %s", exc)
        checks["database"] = False
    try:
        from live_model import get_model
        checks["model"] = get_model() is not None
    except Exception as exc:
        app.logger.error("healthz model: %s", exc)
        checks["model"] = False
    ok = all(checks.values())
    return jsonify({"ok": ok, **checks, "backend": db.backend(),
                    "version": APP_VERSION}), (200 if ok else 503)


@app.route("/")
def index():
    return render_template("index.html")


def _warehouse_3d() -> bool:
    # The 3D warehouse view (static/js/warehouse3d.js) pictures only numbers the
    # panel already shows. Read per request so a deployment can switch back to the
    # plain panel without a code change; each session logs which view it got.
    return os.environ.get("WAREHOUSE_3D", "1") != "0"


@app.route("/task")
def task():
    return render_template("task.html", ui3d=_warehouse_3d())


@app.route("/info")
def info():
    return render_template("info.html")


@app.get("/api/numeracy-items")
def api_numeracy_items():
    return jsonify(numeracy.public_items())


def _desktop_only() -> bool:
    # Read per request, so a deployment can flip it without a code change and
    # the tests can exercise both settings.
    return os.environ.get("DESKTOP_ONLY", "1") != "0"


def _device_info(raw) -> dict | None:
    """The start page's description of the participant's device, validated.

    None when the page sent nothing (older clients, the test suite). Types are
    checked because the values land in integer columns and drive a gate.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("malformed device information")

    def _px(key):
        v = raw.get(key)
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            raise ValueError("malformed device information")
        try:
            v = int(float(v))
        except (TypeError, ValueError):
            raise ValueError("malformed device information")
        if not 0 <= v <= 100_000:
            raise ValueError("malformed device information")
        return v

    return {
        "ua": str(raw.get("ua") or "")[:512],
        "screen_w": _px("screen_w"),
        "screen_h": _px("screen_h"),
        "viewport_w": _px("viewport_w"),
        # Strictly a boolean true; a string "false" must not read as coarse.
        "pointer_coarse": raw.get("pointer_coarse") in (True, 1),
    }


def _desktop_rejection(dev: dict) -> str | None:
    """Why this device fails the desktop-only rule, or None if it passes."""
    if dev["viewport_w"] is not None and dev["viewport_w"] < MIN_VIEWPORT_W:
        return "narrow_viewport"
    if dev["pointer_coarse"]:
        return "coarse_pointer"
    if _MOBILE_UA_RE.search(dev["ua"]):
        return "mobile_ua"
    return None


@app.post("/api/start")
def api_start():
    payload = request.get_json(force=True)
    pid = str(payload.get("participant_id", "")).strip()
    if not _PID_RE.match(pid):
        raise ValueError("participant id must be 3-32 letters, digits, - or _")
    if not payload.get("consented"):
        raise ValueError("consent is required before starting")

    # Desktop gate BEFORE screening and assignment: a phone that is turned
    # away must not leave a screening row, and must never consume an arrival
    # slot in the alternation below.
    device = _device_info(payload.get("device"))
    if device is not None and _desktop_only():
        reason = _desktop_rejection(device)
        if reason:
            conn = db.connect()
            try:
                # No identifier and no user-agent: like `screening`, this only
                # counts how many arrivals were turned away, and why.
                db.log_event(conn, None, "desktop_required", {"reason": reason})
            finally:
                conn.close()
            return jsonify({"error": DESKTOP_REQUIRED_MSG, "code": "desktop_required",
                            "reason": reason}), 400

    # Alternating assignment by arrival order keeps the two cells balanced,
    # which random assignment at n=80 does not reliably do.
    with _LOCK:
        sess = _SESSIONS.get(pid)
        resuming = sess is not None
        n_correct = None
        if resuming:
            # Resuming is legitimate (a refresh mid-session), but only from the
            # browser that started it. Anyone else typing this id is turned away
            # rather than silently handed a live participant's session.
            if session.get("token") != sess.token:
                raise ValueError("that participant id is already in use")
        else:
            conn = db.connect()
            try:
                known = conn.execute("SELECT 1 FROM participant WHERE participant_id=?",
                                     (pid,)).fetchone()
                if known:
                    # An id already in the database belongs to a finished,
                    # withdrawn or interrupted session. The browser that started
                    # it (its cookie token hashes to the stored one) gets it
                    # back, rebuilt exactly; anyone else is refused. Re-opening
                    # it fresh would stack a second session on the old row and
                    # collide on trial keys.
                    sess = _rehydrate(pid, session.get("token"))
                    if sess is None:
                        raise ValueError("that participant id has already been used")
                    resuming = True
                else:
                    # Screen BEFORE assignment: a failed screener must not
                    # consume an arrival slot, or the alternation unbalances.
                    n_correct = numeracy.score(payload.get("numeracy") or {})
                    db.record_screening(conn, n_correct, numeracy.passed(n_correct))
            finally:
                conn.close()
            if not resuming and not numeracy.passed(n_correct):
                return jsonify({"ok": False, "screened_out": True})
        if not resuming:
            if len(_SESSIONS) >= _MAX_SESSIONS:
                raise ValueError("the study is not accepting new sessions")
            conn = db.connect()
            n = conn.execute("SELECT COUNT(*) FROM participant").fetchone()[0]
            conn.close()
            condition = "high" if n % 2 == 0 else "low"
            params = session_params()
            sess = Session(pid, condition, params, consented=True)
            sess.conn.execute("UPDATE participant SET numeracy_score=? WHERE participant_id=?",
                              (n_correct, pid))
            if device is None:
                # Allowed (older start pages, scripted checks), but on record,
                # so an analysis can tell "no device info" from "desktop".
                db.log_event(sess.conn, pid, "device_missing")
            else:
                sess.conn.execute(
                    "UPDATE participant SET device_ua=?, screen_w=?, screen_h=?, "
                    "viewport_w=?, pointer_coarse=? WHERE participant_id=?",
                    (device["ua"], device["screen_w"], device["screen_h"],
                     device["viewport_w"], int(device["pointer_coarse"]), pid))
            _SESSIONS[pid] = sess
        # Write-once. Re-running start must not be able to repoint a recorded
        # Prolific id at someone else, which decides who gets paid.
        prolific = payload.get("prolific_pid")
        if prolific is not None:
            sess.conn.execute(
                "UPDATE participant SET prolific_pid=? "
                "WHERE participant_id=? AND prolific_pid IS NULL",
                (str(prolific)[:64], pid),
            )
        sess.conn.execute(
            "UPDATE participant SET consented=1 WHERE participant_id=?", (pid,))
    session["participant_id"] = pid
    session["token"] = sess.token
    return jsonify({"ok": True, "state": sess.state_payload()})


@app.get("/api/state")
def api_state():
    return jsonify(_require_session().state_payload())


def _rt(payload) -> int:
    """Response time is a dependent variable, so it gets range-checked.

    The browser supplies it (server-side timing would fold network latency into
    the measure), which means it is client-controlled and can arrive negative,
    zero, or absurd from a clock change or a tampered request.
    """
    try:
        rt = int(payload.get("rt_ms", 0))
    except (TypeError, ValueError):
        raise ValueError("response time must be a number")
    if not 0 <= rt <= MAX_RT_MS:
        raise ValueError("response time out of range")
    return rt


@app.post("/api/stage1")
def api_stage1():
    sess = _require_session()
    payload = request.get_json(force=True)
    q_own = payload.get("q_own")
    if q_own is None or int(q_own) < 0 or int(q_own) > sess.params.max_order:
        raise ValueError("order must be between 0 and %d" % sess.params.max_order)
    return jsonify(sess.stage1(int(q_own), _rt(payload)))


@app.post("/api/stage2")
def api_stage2():
    sess = _require_session()
    payload = request.get_json(force=True)
    q_final = payload.get("q_final")
    if q_final is None:
        raise ValueError("a final order is required")
    outcome = sess.stage2(int(q_final), _rt(payload))
    return jsonify({"outcome": outcome, "state": sess.state_payload()})


@app.post("/api/client-error")
def api_client_error():
    """A browser-side error on the participant page, for pre-registered exclusion 5
    ("a technical fault recorded in the event log that prevented normal display").
    The page only reports errors from its own origin (not browser extensions);
    the server caps it at 20 per session so a broken page cannot flood the log."""
    sess = _require_session()
    payload = request.get_json(force=True, silent=True) or {}
    with sess.lock:
        n = getattr(sess, "client_errors", 0)
        if n >= 20:
            return jsonify({"ok": True, "logged": False})
        sess.client_errors = n + 1
        detail = {k: str(payload.get(k, ""))[:300] for k in ("kind", "message", "source", "line")}
        detail.update(phase=sess.phase, trial_index=sess.trial_index)
        db.log_event(sess.conn, sess.participant_id, "client_error", detail)
    return jsonify({"ok": True, "logged": True})


@app.post("/api/begin-main")
def api_begin_main():
    sess = _require_session()
    sess.begin_main()
    return jsonify({"ok": True, "state": sess.state_payload()})


@app.post("/api/block-measure")
def api_block_measure():
    sess = _require_session()
    payload = request.get_json(force=True)
    instrument = payload.get("instrument")
    if instrument not in ("tia", "tlx"):
        raise ValueError("unknown instrument %r" % instrument)

    expected = TIA_ITEMS if instrument == "tia" else TLX_ITEMS
    hi = 7 if instrument == "tia" else 20
    responses = payload.get("responses") or {}
    if set(responses) != set(expected):
        # insert_block_measure uses INSERT OR REPLACE, so an invented item key
        # would persist and a wrong one could overwrite a real answer.
        raise ValueError("expected exactly the %s items" % instrument)
    try:
        values = {k: float(v) for k, v in responses.items()}
    except (TypeError, ValueError):
        raise ValueError("questionnaire responses must be numbers")
    if any(not 1 <= v <= hi for v in values.values()):
        raise ValueError("responses must be between 1 and %d" % hi)

    try:
        block = int(payload.get("block", sess.block))
    except (TypeError, ValueError):
        raise ValueError("block must be a number")
    if not 0 <= block <= 2:
        raise ValueError("block out of range")

    db.insert_block_measure(sess.conn, sess.participant_id, block,
                            instrument, values)
    return jsonify({"ok": True})


@app.post("/api/finish")
def api_finish():
    sess = _require_session()
    payload = request.get_json(force=True) or {}
    # The manipulation check lands in an INTEGER column. SQLite stored
    # whatever arrived ("abc", 1e12, 65.5); Postgres raised on it and the
    # participant got a 500 on the last screen. Validate at the boundary.
    perceived = payload.get("perceived_reliability")
    if perceived is not None:
        if isinstance(perceived, bool) or not isinstance(perceived, (int, float, str)):
            raise ValueError("enter a whole number between 0 and 100")
        try:
            as_float = float(perceived)
        except ValueError:
            raise ValueError("enter a whole number between 0 and 100")
        if not (0 <= as_float <= 100 and as_float == int(as_float)):
            raise ValueError("enter a whole number between 0 and 100")
        perceived = int(as_float)
    db.finish_participant(sess.conn, sess.participant_id, perceived)
    db.log_event(sess.conn, sess.participant_id, "session_end")
    return jsonify({"ok": True, "total_cost": round(sess.twin.total_cost(), 2)})


@app.get("/api/debrief")
def api_debrief():
    """What the debrief discloses. Only after the session is complete: during
    the task this would reveal the manipulation and void the participant."""
    sess = _require_session()
    if not sess.finished:
        raise ValueError("the debrief is available once the task is complete")
    sched = sess.main_schedule
    errors = [i for i, ok in enumerate(sched) if not ok]
    from interface.advice import placement_of
    # "Your 30 weeks" timeline: one entry per main week. The schedule and the
    # week's placement are the same values the trial rows log; the response and
    # the stockout come from the participant's own trial rows (null / false for
    # a week with no row). Deliberately no q_star, delta or risk values.
    rows = {r["trial_index"]: r for r in sess.conn.execute(
        "SELECT trial_index, ai_correct, placement, response_class, stockout FROM trial "
        "WHERE participant_id=? AND phase='main'", (sess.participant_id,)).fetchall()}
    weeks = []
    for i, ok in enumerate(sched):
        r = rows.get(i)
        weeks.append({
            "week": i + 1,
            "ai_correct": bool(r["ai_correct"]) if r is not None else bool(ok),
            "placement": r["placement"] if r is not None else placement_of(i, sess.design),
            "response": r["response_class"] if r is not None else None,
            "stockout": bool(r["stockout"]) if r is not None else False,
        })
    return jsonify({
        "condition_pct": int(round(100 * RELIABILITY_LEVELS[sess.condition])),
        "other_pct": int(round(100 * RELIABILITY_LEVELS["low" if sess.condition == "high" else "high"])),
        "n_trials": len(sched),
        "n_correct": len(sched) - len(errors),
        "n_errors": len(errors),
        "n_errors_propagating": sum(placement_of(i, sess.design) == "propagating" for i in errors),
        "n_errors_absorbed": sum(placement_of(i, sess.design) == "absorbed" for i in errors),
        # Direction as it actually happened, from this participant's own log:
        # under v2 errors are under-orders except the documented fallback when
        # the correct order was too small for any plausible under-order.
        "n_errors_under": sess.conn.execute(
            "SELECT COUNT(*) FROM trial WHERE participant_id=? AND phase='main' "
            "AND ai_correct=0 AND q_ai<q_star", (sess.participant_id,)).fetchone()[0],
        "n_errors_over": sess.conn.execute(
            "SELECT COUNT(*) FROM trial WHERE participant_id=? AND phase='main' "
            "AND ai_correct=0 AND q_ai>q_star", (sess.participant_id,)).fetchone()[0],
        "completion_code": COMPLETION_CODE,
        "weeks": weeks,
    })


@app.post("/api/withdraw")
def api_withdraw():
    """Withdraw now, during or after the task, optionally deleting all data."""
    sess = _require_session()
    payload = request.get_json(force=True) or {}
    delete = bool(payload.get("delete_data", True))
    with sess.lock:
        if not delete:
            db.log_event(sess.conn, sess.participant_id, "withdrawn_keep_data")
        db.withdraw_participant(sess.conn, sess.participant_id, delete_data=delete)
    with _LOCK:
        _SESSIONS.pop(sess.participant_id, None)
    session.clear()
    return jsonify({"ok": True, "deleted": delete})


# ------------------------------------------------------------------ dry run
def dry_run(db_path=None, participant_id="DRYRUN-01", condition="low", verbose=True):
    """Scripted full session. Open item 2: the data-integrity dry run.

    Plays a deliberately mixed strategy -- sometimes accept, sometimes override,
    sometimes modify -- so every response_class and both WoA branches appear in
    the export rather than only the ones a single strategy would exercise.
    """
    if db_path is None and db.database_url():
        # A scripted participant written into the collection database would be
        # counted in the arrival-order alternation and shift every real one.
        raise RuntimeError("refusing to dry-run against DATABASE_URL; pass --db PATH")
    params = session_params()
    sess = Session(participant_id, condition, params, db_path=db_path, consented=True)
    rng = np.random.default_rng(7)

    for phase in ("practice", "main"):
        if phase == "main":
            sess.begin_main()
        while sess.trial_index < sess.n_trials:
            st = sess.state
            # A plausible human: order-up-to with noise, occasionally lazy.
            naive = max(0, reference_order(st, params) + int(rng.normal(0, 40)))
            q_own = int(np.clip(naive, 0, params.max_order))
            adv = sess.stage1(q_own, int(rng.integers(2000, 12000)))

            roll = rng.random()
            if roll < 0.45:
                q_final = adv["q_ai"]                      # accept
            elif roll < 0.70:
                q_final = q_own                            # override
            else:
                q_final = int(round((q_own + adv["q_ai"]) / 2))  # modify
            out = sess.stage2(q_final, int(rng.integers(1000, 9000)))

            if verbose and phase == "main" and sess.trial_index <= 3:
                print("  trial %2d  Q_own=%3d  Q_AI=%3d  Q_fin=%3d  cost=%8.1f"
                      % (sess.trial_index, q_own, adv["q_ai"], q_final, out["cost"]))

        if phase == "main":
            for block in range(3):
                db.insert_block_measure(sess.conn, participant_id, block, "tia",
                                        # same item keys and scales as task.html and
                                        # /api/block-measure: TiA tia_1..6 on 1-7,
                                        # NASA-TLX five subscales on 1-20
                                        {"tia_%d" % i: float(rng.integers(1, 8))
                                         for i in range(1, 7)})
                db.insert_block_measure(sess.conn, participant_id, block, "tlx",
                                        {"mental": 11.0, "temporal": 8.0, "effort": 10.0,
                                         "performance": 13.0, "frustration": 6.0})

    db.finish_participant(sess.conn, participant_id, perceived_reliability=65)
    return sess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true",
                    help="run a scripted session and exit")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    if args.dry_run:
        sess = dry_run(db_path=args.db)
        # Never echo a postgres URL: it carries the password.
        where = "postgres (--db URL)" if db.backend(args.db) == "postgres" else (args.db or db.DEFAULT_DB)
        print("dry run complete -> %s" % where)
        return

    if args.db:
        # Serve against a named database (a demo, a pilot) instead of the
        # collection file, so rehearsals never write to data/trials.sqlite3.
        db.DEFAULT_DB = Path(args.db)
        if db.database_url():
            # An explicit --db beats DATABASE_URL: a rehearsal started from a
            # shell that happens to hold the production URL must not write to
            # the production database.
            print("--db given: ignoring DATABASE_URL for this run")
            os.environ.pop("DATABASE_URL", None)
    where = "postgres (DATABASE_URL)" if db.backend() == "postgres" else str(db.DEFAULT_DB)
    print("serving on http://%s:%d  (db: %s, design: %s)"
          % (args.host, args.port, where, ADVICE_DESIGN))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
