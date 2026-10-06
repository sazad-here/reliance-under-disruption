"""Session recovery after a server restart must be EXACT.

The server keeps each participant's live Session in memory (`_SESSIONS`). When
the process restarts -- a Cloud Run instance is replaced, a deploy lands
mid-collection -- `_rehydrate` rebuilds the Session from the database. These
tests prove that a participant who is interrupted by a restart ends up with
exactly the log they would have produced without one: same advice, same world,
same outcomes, and a stage-1 order that stays locked across the restart.

Method. A whole session is driven through the Flask test client with a fully
scripted participant: every input (Q_own, Q_final, response times,
questionnaire answers) is a deterministic function of the trial number and of
what the server showed. Two runs with the same participant id, condition and
script -- one uninterrupted, one with `_SESSIONS.clear()` at chosen points --
must produce identical trial tables and identical server responses.

A restart is simulated by clearing `_SESSIONS` only. The Flask test client keeps
its cookie (the signed `__session` with participant id + token), and the app's
secret key is unchanged -- exactly the production situation, where
APP_SECRET_KEY is fixed so cookies survive an instance replacement.

Fields EXCLUDED from the run-vs-run comparison of the `trial` table:
  * submitted_utc -- a wall-clock timestamp written by the server at stage 2;
    two runs executed at different moments cannot share it, and it carries no
    experimental information (response times are measured in the browser).
Nothing else is excluded. rt_stage1_ms / rt_stage2_ms are client-supplied, and
the script supplies the same values in both runs, so they MUST match -- in
particular rt_stage1_ms of a trial whose stage 1 was submitted before the
restart has to come back from `pending_stage1` unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = {"num_1": "500", "num_2": "10", "num_3": "0.1"}
PID = "RH-001"
TS_EXCLUDED = {"submitted_utc"}


# ------------------------------------------------------------------ harness
# Backend. By default every test runs on SQLite (a fresh file per run). With
# os.environ["PG_TEST_URL"] set, conftest.py runs each test a second time on Postgres
# (`db_backend == "postgres"`): every run name then gets its own freshly
# created schema, reached through a search_path option in DATABASE_URL, so the
# "ref" and "got" runs of one test -- which reuse the same participant id --
# stay as separate as two SQLite files, and nothing leaks between tests.
_PG_SCHEMA_PREFIX = "rh_test_"


def _pg_schema_url(base: str, schema: str) -> str:
    sep = "&" if "?" in base else "?"
    return "%s%soptions=-csearch_path%%3D%s" % (base, sep, schema)


@pytest.fixture()
def env(tmp_path, monkeypatch, db_backend):
    """Return a function that points the app at a fresh, empty study database
    and returns its target (a SQLite path, or a postgresql:// URL)."""
    from interface import app as app_module
    from interface import db

    app_module.app.config["TESTING"] = True
    made = []       # Postgres (schema, url) pairs to drop at teardown

    def _drop_pool(url):
        with db._POOL_LOCK:
            pool = db._POOLS.pop(url, None)
            db._SCHEMA_READY.discard(url)
        if pool is not None:
            pool.close(timeout=5.0)

    def fresh(name: str):
        app_module._SESSIONS.clear()
        if db_backend == "postgres":
            schema = _PG_SCHEMA_PREFIX + name
            url = _pg_schema_url(os.environ["PG_TEST_URL"], schema)
            _drop_pool(url)     # a pool from an earlier run would skip the DDL
            admin = db.connect(os.environ["PG_TEST_URL"])
            admin.execute('DROP SCHEMA IF EXISTS "%s" CASCADE' % schema)
            admin.execute('CREATE SCHEMA "%s"' % schema)
            monkeypatch.setenv("DATABASE_URL", url)
            made.append((schema, url))
            assert db.backend() == "postgres"
            return url
        path = tmp_path / ("%s.sqlite3" % name)
        monkeypatch.setattr(db, "DEFAULT_DB", path)
        return path

    yield fresh
    app_module._SESSIONS.clear()
    if made:
        admin = db.connect(os.environ["PG_TEST_URL"])
        for schema, url in made:
            _drop_pool(url)
            admin.execute('DROP SCHEMA IF EXISTS "%s" CASCADE' % schema)


def _con(target):
    """A connection to a run's database, on either backend (db.connect gives
    both the sqlite3 calling convention: `?` placeholders, rows by name)."""
    from interface import db
    return db.connect(target)


def _actions():
    """The complete, ordered list of participant actions for one session."""
    from interface.app import N_MAIN, N_PRACTICE
    acts = [("start",)]
    for i in range(N_PRACTICE):
        acts += [("s1", "practice", i), ("s2", "practice", i)]
    acts.append(("begin_main",))
    for i in range(N_MAIN):
        if i > 0 and i % 10 == 0:
            acts.append(("blocks", i // 10 - 1))
        acts += [("s1", "main", i), ("s2", "main", i)]
    acts += [("blocks", 2), ("finish",), ("debrief",)]
    return acts


ACTIONS = None


def _idx(action) -> int:
    """Position of an action in the script; restarts happen BEFORE it."""
    global ACTIONS
    if ACTIONS is None:
        ACTIONS = _actions()
    return ACTIONS.index(action)


def _q_own(state: dict, phase: str, i: int) -> int:
    # A plausible, fully deterministic human: order-up-to on what is visible,
    # plus a trial-dependent offset so Q_own and Q_AI rarely coincide.
    k = i + (0 if phase == "main" else 100)
    target = 4 * state["demand"] + 40 * ((k * 7) % 5) - 60
    q = target - state["inventory_position"]
    return int(min(max(q, 0), state["max_order"]))


def _q_final(q_own: int, q_ai: int, phase: str, i: int) -> int:
    mode = (i + (1 if phase == "practice" else 0)) % 3
    if mode == 0:
        return q_ai                              # accept
    if mode == 1:
        return q_own                             # override
    return int(round((q_own + q_ai) / 2))        # modify


def play(client, restart_before=(), prefill=0, on_restart=None):
    """Drive a whole session. Returns everything the browser was shown.

    restart_before: action indices; `_SESSIONS` is cleared right before each.
    prefill: number of other participants started first (sets the condition
             by arrival-order alternation: 0 -> high, 1 -> low).
    on_restart: optional hook(client, action) run right after each restart and
             before the action -- used to poke at the rehydrated session.
    """
    from interface import app as app_module

    for k in range(prefill):
        with app_module.app.test_client() as other:
            r = other.post("/api/start", json={"participant_id": "FILL-%02d" % k,
                                               "consented": True, "numeracy": PASS})
            assert r.status_code == 200

    shown = []
    state = None
    pending = {}
    restart_before = set(restart_before)
    for n, act in enumerate(_actions()):
        if n in restart_before:
            app_module._SESSIONS.clear()
            if on_restart is not None:
                on_restart(client, act)
        kind = act[0]
        if kind == "start":
            r = client.post("/api/start", json={"participant_id": PID, "consented": True,
                                                "numeracy": PASS})
            assert r.status_code == 200, r.get_json()
            state = r.get_json()["state"]
            shown.append(("start", state))
        elif kind == "s1":
            _, phase, i = act
            if state is None:           # after a restart the page reloads its state
                state = client.get("/api/state").get_json()
            assert (state["phase"], state["trial"]) == (phase, i + 1), (act, state)
            q_own = _q_own(state, phase, i)
            r = client.post("/api/stage1", json={"q_own": q_own, "rt_ms": 3000 + 37 * i})
            assert r.status_code == 200, (act, r.get_json())
            adv = r.get_json()
            pending = {"q_own": q_own, "adv": adv}
            shown.append((act, q_own, adv))
        elif kind == "s2":
            _, phase, i = act
            q_final = _q_final(pending["q_own"], pending["adv"]["q_ai"], phase, i)
            r = client.post("/api/stage2", json={"q_final": q_final, "rt_ms": 1500 + 23 * i})
            assert r.status_code == 200, (act, r.get_json())
            body = r.get_json()
            state = body["state"]
            shown.append((act, q_final, body))
        elif kind == "begin_main":
            r = client.post("/api/begin-main")
            assert r.status_code == 200, r.get_json()
            state = r.get_json()["state"]
            shown.append(("begin_main", state))
        elif kind == "blocks":
            blk = act[1]
            for inst, items, v in (("tia", ["tia_%d" % j for j in range(1, 7)], 1 + blk * 2),
                                   ("tlx", ["mental", "temporal", "performance",
                                            "effort", "frustration"], 5 + blk)):
                r = client.post("/api/block-measure",
                                json={"instrument": inst, "block": blk,
                                      "responses": {it: v for it in items}})
                assert r.status_code == 200, r.get_json()
        elif kind == "finish":
            r = client.post("/api/finish", json={"perceived_reliability": 70})
            assert r.status_code == 200, r.get_json()
            shown.append(("finish", r.get_json()))
        elif kind == "debrief":
            r = client.get("/api/debrief")
            assert r.status_code == 200, r.get_json()
            shown.append(("debrief", r.get_json()))
        if n + 1 in restart_before:
            # The next action follows a restart: the reloaded page must fetch
            # its state again rather than trust what it had in memory.
            state = None
    return shown


def trial_rows(db_path, pid: str = PID) -> list[dict]:
    con = _con(db_path)
    rows = con.execute("SELECT * FROM trial WHERE participant_id=? "
                       "ORDER BY phase DESC, trial_index", (pid,)).fetchall()
    con.close()
    return [{k: r[k] for k in r.keys() if k not in TS_EXCLUDED} for r in rows]


def events(db_path, pid: str = PID) -> list[str]:
    con = _con(db_path)
    kinds = [r[0] for r in con.execute(
        "SELECT kind FROM event_log WHERE participant_id=? ORDER BY id", (pid,))]
    con.close()
    return kinds


def _run(env, name, restart_before=(), prefill=0, on_restart=None):
    from interface import app as app_module
    path = env(name)
    with app_module.app.test_client() as c:
        shown = play(c, restart_before, prefill=prefill, on_restart=on_restart)
    return path, shown


def assert_same_as_uninterrupted(env, restart_before, prefill=0, on_restart=None):
    ref_db, ref_shown = _run(env, "ref", (), prefill)
    got_db, got_shown = _run(env, "got", restart_before, prefill, on_restart)

    ref, got = trial_rows(ref_db), trial_rows(got_db)
    assert len(ref) == len(got) == 35
    for a, b in zip(ref, got):
        diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
        assert not diff, "trial %s/%s differs: %s" % (a["phase"], a["trial_index"], diff)
    # Everything the participant was SHOWN (advice, outcomes, state panels,
    # totals, debrief) is identical too.
    assert ref_shown == got_shown
    # The restarts really happened and were recovered from the database.
    assert events(got_db).count("session_rehydrated") == len(set(restart_before))
    assert "rehydrate_failed" not in events(got_db)
    return got_db, ref_shown


# ------------------------------------------------------------------ 1. mid-main
@pytest.mark.parametrize("prefill", [0, 1], ids=["high", "low"])
def test_restart_mid_main_reproduces_every_logged_field(env, prefill):
    """Practice + 12 main trials, restart, finish: identical to no restart."""
    got_db, _ = assert_same_as_uninterrupted(env, [_idx(("s1", "main", 12))], prefill)
    con = _con(got_db)
    cond = con.execute("SELECT condition FROM participant WHERE participant_id=?",
                       (PID,)).fetchone()[0]
    assert cond == ("high" if prefill == 0 else "low")
    assert con.execute("SELECT COUNT(*) FROM pending_stage1").fetchone()[0] == 0


def test_restart_before_every_single_action(env):
    """Worst case: the server restarts before EVERY request of the session."""
    every = list(range(1, len(_actions())))
    assert_same_as_uninterrupted(env, every)


# ------------------------------------------------------------------ 2. between stages
@pytest.mark.parametrize("phase,i", [("main", 7), ("main", 0), ("main", 29), ("practice", 2)])
def test_restart_between_stage1_and_stage2(env, phase, i):
    """Stage 1 submitted, server restarts, stage 2 arrives at a new process."""
    got_db, _ = assert_same_as_uninterrupted(env, [_idx(("s2", phase, i))])
    assert "session_rehydrated" in events(got_db)


def test_locked_q_own_survives_restart_and_cannot_be_reasked(env):
    """After the restart the page gets the SAME advice back, stage 1 is refused,
    and the logged Q_own is the one locked before the restart."""
    from interface import app as app_module

    seen = {}

    def poke(client, act):
        # Reload after the restart: the page must be sent straight to stage 2
        # with the advice already shown, not given a fresh stage-1 entry.
        st = client.get("/api/state").get_json()
        seen["state"] = st
        assert PID in app_module._SESSIONS          # rehydrated by that request
        r = client.post("/api/stage1", json={"q_own": 1, "rt_ms": 999})
        seen["retry"] = (r.status_code, r.get_json())

    target = ("s2", "main", 7)
    got_db, ref_shown = assert_same_as_uninterrupted(env, [_idx(target)], on_restart=poke)

    advice_before = next(x for x in ref_shown if x[0] == ("s1", "main", 7))[2]
    q_own_locked = next(x for x in ref_shown if x[0] == ("s1", "main", 7))[1]
    assert seen["state"]["phase"] == "main" and seen["state"]["trial"] == 8
    assert seen["state"]["pending_advice"] == advice_before
    assert seen["retry"][0] == 400 and "already submitted" in seen["retry"][1]["error"]
    row = next(r for r in trial_rows(got_db) if r["phase"] == "main" and r["trial_index"] == 7)
    assert row["q_own"] == q_own_locked != 1
    assert row["rt_stage1_ms"] == 3000 + 37 * 7


def test_pending_state_does_not_leak_ground_truth(env):
    """The restored pending advice is the public subset only."""
    from interface import app as app_module
    got = {}

    def poke(client, act):
        got["pa"] = client.get("/api/state").get_json()["pending_advice"]

    _run(env, "got", [_idx(("s2", "main", 3))], on_restart=poke)
    assert set(got["pa"]) == {"q_ai", "risk_pct", "level"}
    app_module._SESSIONS.clear()


# ------------------------------------------------------------------ 3. practice / boundary
@pytest.mark.parametrize("where", [
    ("s1", "practice", 0),       # restart right after start, before any trial
    ("s1", "practice", 3),       # restart mid-practice
    ("begin_main",),             # all 5 practice trials done, main not yet begun
    ("s1", "main", 0),           # main begun (twin reset), no main trial yet
    ("blocks", 0),               # at the first block questionnaire
    ("debrief",),                # finished, restart before the debrief
], ids=["after-start", "mid-practice", "practice-done", "main-begun",
        "block-measure", "before-debrief"])
def test_restart_during_practice_and_at_the_boundary(env, where):
    assert_same_as_uninterrupted(env, [_idx(where)])


def test_boundary_state_after_restart(env):
    """At the practice->main boundary the rehydrated session is in the right
    phase: before begin-main it is practice with 5/5 done; after begin-main it
    is main trial 1 on a freshly reset warehouse."""
    seen = []

    def poke(client, act):
        seen.append((act, client.get("/api/state").get_json()))

    _run(env, "got", [_idx(("begin_main",)), _idx(("s1", "main", 0))], on_restart=poke)
    (a1, st1), (a2, st2) = seen
    assert st1["phase"] == "practice" and st1["trial"] == 6 and st1["pending_advice"] is None
    assert st2["phase"] == "main" and st2["trial"] == 1 and st2["period"] == 0
    assert st2["history"]["demand"] == [st2["demand"]]


# ------------------------------------------------------------------ 4. tokens
def _half_session(env, name="tok"):
    """Start PID, play 3 practice trials, then 'restart'. Returns (path, client)."""
    from interface import app as app_module
    path = env(name)
    c = app_module.app.test_client()
    c.post("/api/start", json={"participant_id": PID, "consented": True, "numeracy": PASS})
    for _ in range(3):
        adv = c.post("/api/stage1", json={"q_own": 150, "rt_ms": 4000}).get_json()
        c.post("/api/stage2", json={"q_final": adv["q_ai"], "rt_ms": 2000})
    app_module._SESSIONS.clear()
    return path, c


def test_only_a_hash_of_the_token_is_stored(env):
    from interface import app as app_module
    path, c = _half_session(env)
    with c.session_transaction() as s:
        token = s["token"]
    con = _con(path)
    stored = con.execute("SELECT token_sha256 FROM participant WHERE participant_id=?",
                         (PID,)).fetchone()[0]
    assert stored == app_module._token_hash(token) != token


def test_rehydrate_refuses_wrong_or_missing_token(env):
    from interface import app as app_module
    path, c = _half_session(env)
    with c.session_transaction() as s:
        token = s["token"]
    with app_module._LOCK:
        assert app_module._rehydrate(PID, None) is None
        assert app_module._rehydrate(PID, "") is None
        assert app_module._rehydrate(PID, "not-the-token") is None
        assert app_module._rehydrate(PID, token[:-1]) is None
        assert app_module._rehydrate("NOBODY-1", token) is None
    assert PID not in app_module._SESSIONS
    # A refused token is not a replay failure and must not be logged as one.
    assert "rehydrate_failed" not in events(path)
    assert "session_rehydrated" not in events(path)
    with app_module._LOCK:
        assert app_module._rehydrate(PID, token) is not None
    assert app_module._SESSIONS[PID].trial_index == 3


def test_another_browser_cannot_take_over_after_restart(env):
    from interface import app as app_module
    path, owner = _half_session(env)

    # 1. A stranger types the same id on the start page.
    stranger = app_module.app.test_client()
    r = stranger.post("/api/start", json={"participant_id": PID, "consented": True,
                                           "numeracy": PASS})
    assert r.status_code == 400 and "already been used" in r.get_json()["error"]
    assert PID not in app_module._SESSIONS

    # 2. A forged cookie naming the participant but carrying a wrong token.
    forger = app_module.app.test_client()
    with forger.session_transaction() as s:
        s["participant_id"] = PID
        s["token"] = "forged"
    assert forger.get("/api/state").status_code == 409
    assert forger.post("/api/stage1", json={"q_own": 5, "rt_ms": 900}).status_code == 409
    # 3. A cookie with the id but no token at all.
    bare = app_module.app.test_client()
    with bare.session_transaction() as s:
        s["participant_id"] = PID
    assert bare.get("/api/state").status_code == 409
    assert PID not in app_module._SESSIONS

    # None of that touched the log; the owner resumes exactly where they were.
    con = _con(path)
    assert con.execute("SELECT COUNT(*) FROM trial").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM pending_stage1").fetchone()[0] == 0
    con.close()
    st = owner.get("/api/state").get_json()
    assert (st["phase"], st["trial"]) == ("practice", 4)

    # And once the owner is live again, the forger is still refused.
    assert forger.get("/api/state").status_code == 409


def test_withdrawn_participant_is_not_rehydrated(env):
    from interface import app as app_module
    path, c = _half_session(env)
    assert c.post("/api/withdraw", json={"delete_data": False}).status_code == 200
    app_module._SESSIONS.clear()
    # The owner's cookie was cleared by withdraw; even a saved token fails,
    # because withdrawal also erases the stored hash.
    con = _con(path)
    assert con.execute("SELECT token_sha256 FROM participant WHERE participant_id=?",
                       (PID,)).fetchone()[0] is None
    con.close()
    assert c.get("/api/state").status_code == 409


def test_tampered_log_is_refused_not_silently_continued(env):
    """If the log no longer replays (edited, or code drift), the session must be
    refused and the reason recorded -- never continued on a different world."""
    from interface import app as app_module
    path, c = _half_session(env)
    con = _con(path)
    con.execute("UPDATE trial SET q_final = q_final + 50 WHERE participant_id=? "
                "AND phase='practice' AND trial_index=0", (PID,))
    con.close()        # both backends autocommit
    assert c.get("/api/state").status_code == 409
    assert PID not in app_module._SESSIONS
    assert "rehydrate_failed" in events(path)


# ------------------------------------------------------------------ 5. verify_log
@pytest.mark.parametrize("prefill", [0, 1], ids=["high", "low"])
def test_rehydrated_session_passes_verify_log(env, prefill):
    from interface.verify_log import verify
    restarts = [_idx(("s1", "practice", 2)), _idx(("s2", "practice", 4)),
                _idx(("begin_main",)), _idx(("s2", "main", 5)),
                _idx(("s1", "main", 12)), _idx(("blocks", 1)), _idx(("s2", "main", 29))]
    path, _ = _run(env, "verify", restarts, prefill=prefill)
    assert events(path).count("session_rehydrated") == len(restarts)
    assert verify(path, PID) == 0


# ------------------------------------------------------------------ abandoned stage 1
def _abandon_then_main(env, name, restart_after_main_trials=None):
    """Stage 1 of practice trial 1 is submitted, then the participant jumps to
    the main phase without stage 2 (begin-main is allowed at any time and
    explicitly discards the pending stage 1). Plays 4 main trials."""
    from interface import app as app_module
    path = env(name)
    c = app_module.app.test_client()
    c.post("/api/start", json={"participant_id": PID, "consented": True, "numeracy": PASS})
    adv = c.post("/api/stage1", json={"q_own": 150, "rt_ms": 4000}).get_json()
    c.post("/api/stage2", json={"q_final": adv["q_ai"], "rt_ms": 2000})
    assert c.post("/api/stage1", json={"q_own": 120, "rt_ms": 4000}).status_code == 200
    assert c.post("/api/begin-main").status_code == 200
    shown = []
    for i in range(4):
        if restart_after_main_trials == i:
            app_module._SESSIONS.clear()
            st = c.get("/api/state")
            assert st.status_code == 200, st.get_json()
        adv = c.post("/api/stage1", json={"q_own": 100 + 10 * i, "rt_ms": 3000}).get_json()
        out = c.post("/api/stage2", json={"q_final": adv["q_ai"], "rt_ms": 2000}).get_json()
        shown.append((adv, out))
    return path, shown


@pytest.mark.parametrize("at", [0, 2])
def test_restart_after_an_abandoned_practice_stage1(env, at):
    """The abandoned stage 1 drew from the advice rng but left no trial row;
    the rehydrated session must still take that draw, or every later advice
    (and the replay of the logged main trials) differs."""
    ref_db, ref_shown = _abandon_then_main(env, "ref")
    got_db, got_shown = _abandon_then_main(env, "got", restart_after_main_trials=at)
    assert "rehydrate_failed" not in events(got_db)
    assert ref_shown == got_shown
    assert trial_rows(ref_db) == trial_rows(got_db)


def test_restored_pending_then_abandoned_then_restart_again(env):
    """Restart with a practice stage 1 pending (restored from pending_stage1),
    then begin-main discards it, then a second restart: the draw must still be
    accounted for, now from the main_phase_start event."""
    from interface import app as app_module

    def run(name, restarts):
        path = env(name)
        c = app_module.app.test_client()
        c.post("/api/start", json={"participant_id": PID, "consented": True, "numeracy": PASS})
        assert c.post("/api/stage1", json={"q_own": 90, "rt_ms": 4000}).status_code == 200
        if restarts:
            app_module._SESSIONS.clear()
            assert c.get("/api/state").get_json()["pending_advice"] is not None
        assert c.post("/api/begin-main").status_code == 200
        shown = []
        for i in range(3):
            if restarts and i == 1:
                app_module._SESSIONS.clear()
            adv = c.post("/api/stage1", json={"q_own": 100, "rt_ms": 3000}).get_json()
            out = c.post("/api/stage2", json={"q_final": adv["q_ai"], "rt_ms": 2000}).get_json()
            shown.append((adv, out))
        return path, shown

    ref_db, ref_shown = run("ref", False)
    got_db, got_shown = run("got", True)
    assert events(got_db).count("session_rehydrated") == 2
    assert ref_shown == got_shown and trial_rows(ref_db) == trial_rows(got_db)


def test_double_begin_main_at_the_boundary_is_harmless(env):
    """A double click on the 'start main task' button sends begin-main twice.
    With no main trial and no pending stage 1 in between, the session and its
    recovery must be unaffected."""
    from interface import app as app_module
    target = _idx(("begin_main",))

    ref_db, ref_shown = _run(env, "ref")
    path = env("got")
    c = app_module.app.test_client()
    # Replay the script, sending begin-main twice and restarting after.
    orig = app_module.Session.begin_main
    calls = []

    def double(self):
        orig(self)
        calls.append(1)
        orig(self)
    app_module.Session.begin_main = double
    try:
        got_shown = play(c, [target + 1])
    finally:
        app_module.Session.begin_main = orig
    assert calls == [1]
    assert events(path).count("main_phase_start") == 2
    assert ref_shown == got_shown
    assert trial_rows(ref_db) == trial_rows(path)
