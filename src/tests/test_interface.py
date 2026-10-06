"""Tests for the participant interface.

The properties asserted here are the ones that, if they silently broke during
collection, would cost the whole dataset: the realised reliability, the
two-stage lock, and the fact that ground truth never reaches the browser.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import DEFAULT_CALENDAR, SupplyChainTwin, TwinParams  # noqa: E402
from policy import is_correct  # noqa: E402
from interface.advice import (  # noqa: E402
    RELIABILITY_LEVELS,
    build_schedule,
    make_advice,
    participant_seed,
    placement_of,
    response_class,
    weight_of_advice,
)


# --------------------------------------------------------------- schedule
@pytest.mark.parametrize("condition,expected_correct", [("high", 24), ("low", 18)])
def test_schedule_hits_the_nominal_rate_exactly(condition, expected_correct):
    """Exact, not in expectation. Sampling per trial would let realised
    reliability wander across the gap between the two conditions."""
    sched = build_schedule("P01", condition, 30)
    assert sum(sched) == expected_correct
    assert len(sched) == 30


def test_schedule_splits_errors_across_placements():
    sched = build_schedule("P01", "low", 30)
    errs = [i for i, ok in enumerate(sched) if not ok]
    places = {placement_of(i) for i in errs}
    assert places == {"absorbed", "propagating"}


def test_schedule_is_deterministic_per_participant():
    assert build_schedule("P07", "low", 30) == build_schedule("P07", "low", 30)
    assert build_schedule("P07", "low", 30) != build_schedule("P08", "low", 30)


def test_participant_seed_is_stable():
    assert participant_seed("P01") == participant_seed("P01")
    assert participant_seed("P01") != participant_seed("P02")


def test_v1_propagating_positions_track_the_calendar():
    for e in DEFAULT_CALENDAR:
        assert placement_of(e.period, "v1") == "propagating"
        assert placement_of(e.period - 1, "v1") == "propagating"


# --------------------------------------------------------------- design v2
def test_v2_errors_fall_only_on_selected_weeks():
    from interface.advice import V2_ABSORBED, V2_PROPAGATING
    for pid in ("A1", "B2", "C3", "D4"):
        for cond, n_err in (("high", 6), ("low", 12)):
            sched = build_schedule(pid, cond, 30, design="v2")
            errs = [i for i, ok in enumerate(sched) if not ok]
            assert len(errs) == n_err
            assert all(i in V2_PROPAGATING or i in V2_ABSORBED for i in errs)
            n_prop = sum(i in V2_PROPAGATING for i in errs)
            assert n_prop == n_err // 2


def test_v2_errors_are_under_orders_and_never_implausibly_small():
    from interface.advice import MIN_ORDER
    rng = np.random.default_rng(3)
    for period in range(0, 20):
        st, params = _state_at(period)
        for _ in range(10):
            adv = make_advice(st, params, period, False, rng, design="v2")
            assert not adv.ai_correct
            assert adv.q_ai >= MIN_ORDER
            if adv.q_star - MIN_ORDER > adv.delta:
                assert adv.q_ai < adv.q_star


def test_v2_small_q_star_falls_back_without_zero_orders():
    """When Q* is too small for any plausible under-order, the error flips to an
    over-order rather than recommending (near) zero -- and is still an error."""
    from interface.advice import MIN_ORDER
    st, params = _state_at(0)
    rng = np.random.default_rng(0)
    heavy = type(st)(**{**st.__dict__, "inventory": st.inventory + 5000})
    adv = make_advice(heavy, params, 0, False, rng, design="v2")
    assert adv.q_star <= adv.delta + MIN_ORDER
    assert adv.q_ai > adv.q_star and not adv.ai_correct


def test_v1_design_is_still_available():
    sched = build_schedule("P01", "low", 30, design="v1")
    errs = [i for i, ok in enumerate(sched) if not ok]
    assert {placement_of(i, "v1") for i in errs} == {"absorbed", "propagating"}


# --------------------------------------------------------------- advice
def _state_at(period=0, seed=3):
    twin = SupplyChainTwin(params=TwinParams.from_fitted(), seed=seed)
    st = twin.reset()
    for _ in range(period):
        st, _, _ = twin.step(140)
    return st, twin.p


def test_correct_advice_lands_inside_the_band():
    st, params = _state_at()
    rng = np.random.default_rng(0)
    for _ in range(50):
        adv = make_advice(st, params, 0, True, rng)
        assert adv.ai_correct
        assert is_correct(adv.q_ai, adv.q_star, adv.delta)


def test_error_advice_lands_outside_the_band():
    """The label must match the quantity. A clipped error that drifts back
    inside the band and is still logged as an error would corrupt the
    manipulation without any visible symptom."""
    rng = np.random.default_rng(0)
    for period in range(0, 12):
        st, params = _state_at(period)
        for _ in range(20):
            adv = make_advice(st, params, period, False, rng)
            assert not adv.ai_correct, (
                "trial labelled incorrect but Q_AI is inside the delta band"
            )


def test_advice_public_payload_hides_ground_truth():
    st, params = _state_at()
    adv = make_advice(st, params, 0, True, np.random.default_rng(0))
    public = adv.public()
    for leak in ("q_star", "delta", "ai_correct", "placement"):
        assert leak not in public


def test_risk_display_is_bounded():
    rng = np.random.default_rng(1)
    for period in range(0, 20):
        st, params = _state_at(period)
        adv = make_advice(st, params, period, True, rng)
        assert 1 <= adv.risk_pct <= 99
        assert adv.level in ("low", "moderate", "high")


# --------------------------------------------------------------- live LightGBM risk
def test_risk_shown_is_the_models_output():
    """On a correct week the displayed risk IS the LightGBM prediction."""
    from live_model import get_model, state_row
    st, params = _state_at(6)
    adv = make_advice(st, params, 6, True, np.random.default_rng(0), history=[state_row(st)])
    assert adv.risk_pct == int(np.clip(round(100 * get_model().predict([state_row(st)])), 1, 99))


def test_error_risk_agrees_with_the_wrong_order():
    """Under-order -> the AI believes there is more stock -> LOWER risk than the
    truth; over-order -> higher. A correct risk beside a wrong order leaked errors."""
    from interface.advice import _risk_display
    from live_model import state_row
    lower = higher = 0
    for period in range(3, 15):
        st, params = _state_at(period)
        hist = [state_row(st)]
        from policy import reference_order
        q = reference_order(st, params)
        true_risk, _ = _risk_display(hist, q, q)
        under, _ = _risk_display(hist, q, max(0, q - 100))
        over, _ = _risk_display(hist, q, q + 100)
        lower += under <= true_risk
        higher += over >= true_risk
    assert lower == 12 and higher == 12


def test_live_features_are_identical_for_training_and_serving():
    """One function builds features for both; the last live row must equal the
    row training computes at the same week."""
    import pandas as pd
    from live_model import RAW, features, state_row
    twin = SupplyChainTwin(params=TwinParams.from_fitted(), seed=5)
    rows = [state_row(twin.reset())]
    for _ in range(7):
        st, _, _ = twin.step(150)
        rows.append(state_row(st))
    full = features(pd.DataFrame(rows, columns=RAW))
    live = features(pd.DataFrame(rows[:5], columns=RAW)).iloc[-1]
    pd.testing.assert_series_equal(full.iloc[4], live, check_names=False)


def test_sessions_replay_the_same_real_demand_for_everyone():
    from interface.world import make_twin
    params = TwinParams.from_fitted()
    a, b = make_twin(params, 1, "main"), make_twin(params, 999, "main")
    assert a.demand_source is not None
    assert list(a.demand_source.window) == list(b.demand_source.window)
    p = make_twin(params, 1, "practice")
    assert p.demand_source.series_idx != a.demand_source.series_idx


def test_session_series_are_excluded_from_model_training():
    import json
    from live_model import META_PATH
    from interface.world import SESSION_SERIES
    meta = json.load(open(META_PATH, encoding="utf-8"))
    assert sorted(meta["excluded_series"]) == sorted(SESSION_SERIES)


# --------------------------------------------------------------- measures
def test_weight_of_advice_endpoints():
    assert weight_of_advice(100, 200, 200) == pytest.approx(1.0)   # full accept
    assert weight_of_advice(100, 100, 200) == pytest.approx(0.0)   # full override
    assert weight_of_advice(100, 150, 200) == pytest.approx(0.5)   # split
    assert weight_of_advice(100, 120, 100) is None                 # undefined


def test_response_class_covers_all_three():
    assert response_class(100, 200, 200) == "accept"
    assert response_class(100, 100, 200) == "override"
    assert response_class(100, 150, 200) == "modify"


# --------------------------------------------------------------- http layer
PASS = {"num_1": "500", "num_2": "10", "num_3": "0.1"}
FAIL = {"num_1": "600", "num_2": "100", "num_3": "0.1"}
@pytest.fixture()
def client(tmp_path, monkeypatch):
    import interface.db as db
    from interface import app as app_module

    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "t.sqlite3")
    app_module._SESSIONS.clear()
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_start_requires_consent(client):
    r = client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX01", "consented": False})
    assert r.status_code == 400


def test_stage2_requires_stage1(client):
    client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX02", "consented": True})
    r = client.post("/api/stage2", json={"q_final": 100, "rt_ms": 100})
    assert r.status_code == 400
    assert "stage 1" in r.get_json()["error"]


def test_stage1_cannot_be_resubmitted(client):
    """Revising Q_own after seeing the advice destroys the unaided baseline."""
    client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX03", "consented": True})
    client.post("/api/stage1", json={"q_own": 150, "rt_ms": 4000})
    r = client.post("/api/stage1", json={"q_own": 200, "rt_ms": 50})
    assert r.status_code == 400
    assert "already submitted" in r.get_json()["error"]


def test_orders_outside_range_are_rejected(client):
    client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX04", "consented": True})
    assert client.post("/api/stage1", json={"q_own": -5, "rt_ms": 10}).status_code == 400
    assert client.post("/api/stage1", json={"q_own": 99999, "rt_ms": 10}).status_code == 400


def test_no_session_returns_409(client):
    assert client.get("/api/state").status_code == 409


def test_begin_main_cannot_restart_a_started_main_phase(client):
    """A second begin-main after a main trial used to reset the warehouse to
    trial 0; the next stage 2 then collided with the logged trial key."""
    client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX07", "consented": True})
    assert client.post("/api/begin-main").status_code == 200
    assert client.post("/api/begin-main").status_code == 200   # repeat at the boundary: harmless
    adv = client.post("/api/stage1", json={"q_own": 150, "rt_ms": 4000}).get_json()
    client.post("/api/stage2", json={"q_final": adv["q_ai"], "rt_ms": 2000})
    r = client.post("/api/begin-main")
    assert r.status_code == 400 and "already started" in r.get_json()["error"]
    state = client.get("/api/state").get_json()
    assert state["phase"] == "main" and state["trial"] == 2   # 1-based: one main trial is done
    adv = client.post("/api/stage1", json={"q_own": 150, "rt_ms": 4000}).get_json()
    assert client.post("/api/stage2", json={"q_final": adv["q_ai"], "rt_ms": 2000}).status_code == 200


@pytest.mark.parametrize("value,ok", [(None, True), (0, True), (65, True), (100, True), ("40", True),
                                      (65.0, True), (65.5, False), (-1, False), (101, False),
                                      ("abc", False), (True, False), (10 ** 12, False)])
def test_perceived_reliability_is_validated_on_sqlite(client, value, ok):
    """Same rule as the Postgres leg (test_postgres_cli.py): null or a whole number 0-100."""
    client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX08", "consented": True})
    r = client.post("/api/finish", json={"perceived_reliability": value})
    assert (r.status_code == 200) is ok


@pytest.mark.parametrize("flag,expect_3d", [(None, True), ("1", True), ("0", False)])
def test_warehouse_3d_switch_and_logging(client, monkeypatch, flag, expect_3d):
    """The 3D view is on by default, WAREHOUSE_3D=0 restores the plain panel, and
    each session logs which view it got, so the analysis can tell them apart."""
    import json
    import interface.db as db
    if flag is None:
        monkeypatch.delenv("WAREHOUSE_3D", raising=False)
    else:
        monkeypatch.setenv("WAREHOUSE_3D", flag)
    html = client.get("/task").get_data(as_text=True)
    assert ('id="scene3d"' in html) is expect_3d
    assert ("/static/js/warehouse3d.js" in html) is expect_3d
    assert 'id="state"' in html                     # the number panel is always there
    client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX3D", "consented": True})
    con = db.connect()
    row = con.execute("SELECT detail FROM event_log WHERE kind='session_start'").fetchone()
    assert json.loads(row["detail"])["ui"] == ("3d" if expect_3d else "panel")


def test_3d_assets_are_served_locally(client):
    """three.js is vendored: the study must not depend on a third-party CDN."""
    for path in ("/static/js/warehouse3d.js", "/static/vendor/three/three.module.min.js",
                 "/static/vendor/three/OrbitControls.js", "/static/vendor/three/RoomEnvironment.js"):
        assert client.get(path).status_code == 200, path
    js = client.get("/static/js/warehouse3d.js").get_data(as_text=True)
    assert "http://" not in js and "https://" not in js


def test_motion_is_served_locally_and_referenced_from_the_pages(client):
    """Motion (UI v3.1 animations) is vendored and loaded from this app, never a CDN."""
    res = client.get("/static/vendor/motion/motion.js")
    assert res.status_code == 200
    assert b"Motion" in res.get_data()[:2000]
    local = '<script src="/static/vendor/motion/motion.js"></script>'
    task = client.get("/task").get_data(as_text=True)
    assert local in task
    assert local in client.get("/").get_data(as_text=True)
    for cdn in ("cdn.jsdelivr", "unpkg.com", "motion.dev/", "cdnjs"):
        assert cdn not in task, cdn


def test_client_errors_are_logged_capped_and_need_a_session(client):
    """Pre-registered exclusion 5 relies on browser faults reaching the event log."""
    import json
    import sqlite3
    import interface.db as db
    assert client.post("/api/client-error", json={"message": "x"}).status_code == 409
    client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX09", "consented": True})
    r = client.post("/api/client-error", json={"kind": "error", "message": "boom" * 200, "line": 7})
    assert r.get_json()["logged"] is True
    for _ in range(25):
        client.post("/api/client-error", json={"message": "again"})
    con = db.connect()          # the app's own connection: SQLite, or Postgres on that leg
    rows = con.execute("SELECT detail FROM event_log WHERE kind='client_error'").fetchall()
    con.close()
    assert len(rows) == 20                          # server-side cap
    assert len(json.loads(rows[0]["detail"])["message"]) == 300   # truncated


def test_full_trial_advances_and_logs(client):
    client.post("/api/start", json={"numeracy": PASS, "participant_id": "PX05", "consented": True})
    before = client.get("/api/state").get_json()["trial"]
    adv = client.post("/api/stage1", json={"q_own": 150, "rt_ms": 4000}).get_json()
    out = client.post("/api/stage2", json={"q_final": adv["q_ai"], "rt_ms": 2000}).get_json()
    assert out["state"]["trial"] == before + 1
    assert "cost" in out["outcome"]
    assert "q_star" not in out["outcome"]


def test_dry_run_then_verify_passes(tmp_path):
    """End-to-end: a scripted session must clear every integrity check."""
    from interface.app import dry_run
    from interface.verify_log import verify

    db_path = tmp_path / "dry.sqlite3"
    dry_run(db_path=db_path, participant_id="T-01", condition="low", verbose=False)
    assert verify(db_path, "T-01") == 0


def test_verify_catches_a_risk_the_model_did_not_produce(tmp_path):
    """Training/serving skew: a logged risk % that the live model does not
    reproduce offline (wrong model file, LightGBM version drift) must fail."""
    import sqlite3
    from interface.app import dry_run
    from interface.verify_log import verify

    db_path = tmp_path / "skew.sqlite3"
    dry_run(db_path=db_path, participant_id="T-03", condition="high", verbose=False)
    con = sqlite3.connect(db_path)
    con.execute("UPDATE trial SET ai_risk_pct = CASE WHEN ai_risk_pct > 50 THEN 1 ELSE 99 END "
                "WHERE participant_id = 'T-03' AND phase = 'main' AND trial_index = 4")
    con.commit(); con.close()
    assert verify(db_path, "T-03") != 0


# --------------------------------------------------------------- numeracy screener
def test_numeracy_scoring_accepts_formatting():
    from interface import numeracy
    assert numeracy.score({"num_1": "500", "num_2": "10", "num_3": "0.1%"}) == 3
    assert numeracy.score({"num_1": "1,000", "num_2": "ten", "num_3": ".1"}) == 1
    with pytest.raises(ValueError):
        numeracy.score({"num_1": "500"})


def test_failed_screener_creates_no_participant(client):
    """A screen-out must not consume an arrival slot, or alternation unbalances."""
    import interface.db as db
    r = client.post("/api/start", json={"participant_id": "PS01", "consented": True, "numeracy": FAIL})
    assert r.status_code == 200 and r.get_json()["screened_out"]
    conn = db.connect(db.DEFAULT_DB)
    assert conn.execute("SELECT COUNT(*) FROM participant").fetchone()[0] == 0
    row = conn.execute("SELECT numeracy_score, passed FROM screening").fetchone()
    assert (row[0], row[1]) == (1, 0)
    assert client.get("/api/state").status_code == 409   # no session was opened


def test_screen_out_does_not_shift_alternation(client):
    import interface.db as db
    client.post("/api/start", json={"participant_id": "PA01", "consented": True, "numeracy": PASS})
    client.post("/api/start", json={"participant_id": "PA02", "consented": True, "numeracy": FAIL})
    from interface import app as app_module
    with app_module.app.test_client() as c2:
        c2.post("/api/start", json={"participant_id": "PA03", "consented": True, "numeracy": PASS})
    conn = db.connect(db.DEFAULT_DB)
    conds = dict(conn.execute("SELECT participant_id, condition FROM participant"))
    assert conds == {"PA01": "high", "PA03": "low"}


def test_passed_screener_records_score(client):
    import interface.db as db
    client.post("/api/start", json={"participant_id": "PS02", "consented": True, "numeracy": PASS})
    conn = db.connect(db.DEFAULT_DB)
    assert conn.execute("SELECT numeracy_score FROM participant WHERE participant_id='PS02'").fetchone()[0] == 3


def test_information_sheet_is_served(client):
    r = client.get("/info")
    assert r.status_code == 200 and b"Participant Information Sheet" in r.data
    assert b"will not always be right" in r.data
    # the manipulation itself must not be disclosed before the task
    for leak in (b"60%", b"60 %", b"80%", b"80 %", b"scripted", b"schedule"):
        assert leak not in r.data, leak


# --------------------------------------------------------------- debrief
def test_debrief_is_locked_until_the_task_is_complete(client):
    client.post("/api/start", json={"participant_id": "PD01", "consented": True, "numeracy": PASS})
    r = client.get("/api/debrief")
    assert r.status_code == 400


def test_debrief_discloses_the_true_schedule(client):
    from interface import app as app_module
    client.post("/api/start", json={"participant_id": "PD02", "consented": True, "numeracy": PASS})
    sess = app_module._SESSIONS["PD02"]
    sess.phase, sess.trial_index = "main", app_module.N_MAIN     # jump to the end
    d = client.get("/api/debrief").get_json()
    expected_pct = 80 if sess.condition == "high" else 60
    assert d["condition_pct"] == expected_pct
    assert d["n_correct"] == sum(sess.main_schedule)
    assert d["n_errors_propagating"] + d["n_errors_absorbed"] == d["n_errors"] == 30 - d["n_correct"]


WEEK_KEYS = {"week", "ai_correct", "placement", "response", "stockout"}


def test_debrief_weeks_need_a_session_and_a_finished_task(client):
    """The per-week timeline discloses the manipulation: never before the end."""
    assert client.get("/api/debrief").status_code == 409            # no session
    client.post("/api/start", json={"participant_id": "PD03", "consented": True, "numeracy": PASS})
    client.post("/api/begin-main")
    _play_one_trial(client)
    r = client.get("/api/debrief")
    assert r.status_code == 400 and "weeks" not in r.get_json()


def test_debrief_weeks_match_the_logged_main_trials(client):
    """30 entries, fields exactly as specified, taken from this participant's own
    trial rows, and nothing about the correct order (no q_star)."""
    import interface.db as db
    from interface import app as app_module
    client.post("/api/start", json={"participant_id": "PD04", "consented": True, "numeracy": PASS})
    client.post("/api/begin-main")
    orders = [150, None, 90]                        # own, take the AI, something else
    for t in range(app_module.N_MAIN):
        adv = client.post("/api/stage1", json={"q_own": 150, "rt_ms": 3000}).get_json()
        pick = orders[t % 3]
        q_final = adv["q_ai"] if pick is None else pick
        assert client.post("/api/stage2", json={"q_final": q_final, "rt_ms": 2000}).status_code == 200
    assert client.post("/api/finish", json={"perceived_reliability": 70}).status_code == 200
    r = client.get("/api/debrief")
    assert r.status_code == 200
    assert "q_star" not in r.get_data(as_text=True)
    d = r.get_json()
    weeks = d["weeks"]
    assert [w["week"] for w in weeks] == list(range(1, 31))
    assert all(set(w) == WEEK_KEYS for w in weeks)
    conn = db.connect(db.DEFAULT_DB)
    rows = {r["trial_index"]: r for r in conn.execute(
        "SELECT * FROM trial WHERE participant_id='PD04' AND phase='main'")}
    assert len(rows) == 30
    for w in weeks:
        row = rows[w["week"] - 1]
        assert w["ai_correct"] is bool(row["ai_correct"])
        assert w["placement"] == row["placement"] in ("propagating", "absorbed", "neutral")
        assert w["response"] == row["response_class"] in ("accept", "modify", "override")
        assert w["stockout"] is bool(row["stockout"])
    assert sum(w["ai_correct"] for w in weeks) == d["n_correct"]
    assert sum(not w["ai_correct"] and w["placement"] == "propagating" for w in weeks) == d["n_errors_propagating"]
    assert {w["response"] for w in weeks} >= {"accept", "override"}


def test_debrief_weeks_without_trial_rows_fall_back_to_the_schedule(client):
    from interface import app as app_module
    client.post("/api/start", json={"participant_id": "PD05", "consented": True, "numeracy": PASS})
    sess = app_module._SESSIONS["PD05"]
    sess.phase, sess.trial_index = "main", app_module.N_MAIN
    weeks = client.get("/api/debrief").get_json()["weeks"]
    assert [w["ai_correct"] for w in weeks] == [bool(x) for x in sess.main_schedule]
    assert all(w["response"] is None and w["stockout"] is False for w in weeks)


# --------------------------------------------------------------- volunteer (SONA) wording
_FORBIDDEN = re.compile(r"prolific|\bpay\b|\bpayment|\bpaid\b|\breward|completion code|\[AMOUNT\]", re.I)


def _participant_facing(html: str) -> str:
    """The page as a participant can meet it: markup, attributes and visible text,
    minus scripts, styles and comments (script keeps a URL-parameter name for
    older links, and the stored column keeps its name)."""
    html = re.sub(r"<script\b.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style\b.*?</style>", " ", html, flags=re.S | re.I)
    return re.sub(r"<!--.*?-->", " ", html, flags=re.S)


@pytest.mark.parametrize("path", ["/", "/task", "/info"])
def test_participant_pages_have_no_payment_or_platform_text(client, path):
    page = client.get(path).get_data(as_text=True)
    hits = _FORBIDDEN.findall(_participant_facing(page))
    assert not hits, (path, hits)
    # strings the task page's own script writes into the page
    if path == "/task":
        script = " ".join(re.findall(r"<script>(.*?)</script>", page, flags=re.S))
        strings = [a or b for a, b in re.findall(r"'([^'\n]*)'|`([^`]*)`", script)]
        assert not _FORBIDDEN.findall(" ".join(strings))


def test_info_sheet_says_voluntary_unpaid_and_sona(client):
    text = client.get("/info").get_data(as_text=True)
    assert "voluntary and unpaid" in text and "SONA participant pool" in text
    for placeholder in ("[ETHICS REF]", "[INSTITUTION]", "[RETENTION PERIOD FOR RESEARCH DATA]",
                        "[DATE, e.g. when data is anonymised]", "[RESEARCHER NAME]",
                        "[DEPARTMENT, INSTITUTION]", "[SUPERVISOR NAME]", "[STUDY EMAIL]",
                        "[ETHICS COMMITTEE NAME]", "[COMMITTEE CONTACT]"):
        assert placeholder in text, placeholder


def test_start_page_prefills_the_id_from_sona_and_older_links(client):
    """?id= and ?participant= (SONA study URL) and the older parameter all pre-fill
    the ID field; the start payload keeps its stored field name."""
    page = client.get("/?id=12345").get_data(as_text=True)
    for param in ("'id'", "'participant'", "'PROLIFIC_PID'"):
        assert "params.get(%s)" % param in page, param
    assert "Use the ID shown in SONA" in page
    # the stored field is unchanged: a recruitment ID still lands in the participant row
    import interface.db as db
    client.post("/api/start", json={"participant_id": "SONA-12345", "consented": True,
                                    "numeracy": PASS, "prolific_pid": "12345"})
    row = db.connect(db.DEFAULT_DB).execute(
        "SELECT prolific_pid FROM participant WHERE participant_id='SONA-12345'").fetchone()
    assert row["prolific_pid"] == "12345"


def test_milestones_depend_only_on_the_week_number(client):
    """Encouragement is fixed text keyed by the main-phase week number; the function
    that picks it reads nothing else (no outcome, advice, condition or schedule)."""
    page = client.get("/task").get_data(as_text=True)
    fn = re.search(r"function milestoneFor\(weekState\) \{(.*?)\n\}", page, flags=re.S).group(1)
    used = set(re.findall(r"weekState\.(\w+)", fn))
    assert used == {"phase", "trial"}, used
    for banned in ("outcome", "adv", "condition", "ai_correct", "placement", "risk", "cost", "stockout"):
        assert banned not in fn, banned
    for text in ("15 of 30 weeks done", "A third of the way through", "Two thirds done", "last questions"):
        assert text in page, text


# --------------------------------------------------------------- withdrawal
def _play_one_trial(client):
    adv = client.post("/api/stage1", json={"q_own": 150, "rt_ms": 4000}).get_json()
    client.post("/api/stage2", json={"q_final": adv["q_ai"], "rt_ms": 2000})


def test_withdraw_with_deletion_removes_everything(client):
    import interface.db as db
    client.post("/api/start", json={"participant_id": "PW01", "consented": True,
                                    "numeracy": PASS, "prolific_pid": "abc123"})
    _play_one_trial(client)
    r = client.post("/api/withdraw", json={"delete_data": True})
    assert r.status_code == 200
    conn = db.connect(db.DEFAULT_DB)
    for table in ("trial", "block_measure", "event_log"):
        n = conn.execute("SELECT COUNT(*) FROM %s WHERE participant_id='PW01'" % table).fetchone()[0]
        assert n == 0, table
    p = conn.execute("SELECT * FROM participant WHERE participant_id='PW01'").fetchone()
    assert p["prolific_pid"] is None and p["numeracy_score"] is None
    assert p["data_deleted"] == 1 and p["excluded"] == 1 and p["withdrawn_utc"]
    assert p["condition"] in ("high", "low")          # row kept for alternation
    assert client.get("/api/state").status_code == 409   # session is closed


def test_withdraw_keeping_data_keeps_trials(client):
    import interface.db as db
    client.post("/api/start", json={"participant_id": "PW02", "consented": True, "numeracy": PASS})
    _play_one_trial(client)
    client.post("/api/withdraw", json={"delete_data": False})
    conn = db.connect(db.DEFAULT_DB)
    assert conn.execute("SELECT COUNT(*) FROM trial WHERE participant_id='PW02'").fetchone()[0] == 1
    p = conn.execute("SELECT excluded, data_deleted FROM participant WHERE participant_id='PW02'").fetchone()
    assert (p[0], p[1]) == (1, 0)


def test_withdrawn_id_cannot_be_reused(client):
    client.post("/api/start", json={"participant_id": "PW03", "consented": True, "numeracy": PASS})
    client.post("/api/withdraw", json={"delete_data": True})
    r = client.post("/api/start", json={"participant_id": "PW03", "consented": True, "numeracy": PASS})
    assert r.status_code == 400 and "already been used" in r.get_json()["error"]


def test_researcher_withdrawal_tool(tmp_path):
    from interface.app import dry_run
    from interface import db
    from interface.withdraw import main as withdraw_main

    db_path = tmp_path / "w.sqlite3"
    dry_run(db_path=db_path, participant_id="T-02", condition="high", verbose=False)
    conn = db.connect(db_path)
    conn.execute("UPDATE participant SET prolific_pid='PROL-9' WHERE participant_id='T-02'")
    assert withdraw_main(["--prolific", "PROL-9", "--db", str(db_path)]) == 0
    assert conn.execute("SELECT COUNT(*) FROM trial WHERE participant_id='T-02'").fetchone()[0] == 0
    assert withdraw_main(["--participant", "NOPE", "--db", str(db_path)]) == 1


# --------------------------------------------------------------- order cap
def test_reliability_stays_exact_when_participant_digs_a_hole(tmp_path):
    """Regression for the live demo: a participant who orders nothing drives Q*
    far up. Realised reliability must still be exactly 24 / 30."""
    from interface.app import Session
    from interface.world import SESSION_MAX_ORDER, session_params
    params = session_params()
    assert params.max_order == SESSION_MAX_ORDER
    sess = Session("HOLE-01", "high", params, db_path=tmp_path / "h.sqlite3", consented=True)
    sess.begin_main()
    correct = 0
    for _ in range(30):
        adv = sess.stage1(0, 3000)
        sess.stage2(0, 2000)
    rows = sess.conn.execute("SELECT ai_correct, q_star, q_ai FROM trial WHERE phase='main'").fetchall()
    assert sum(r[0] for r in rows) == 24
    assert max(r[1] for r in rows) > 600          # the old cap would have broken this


def test_cap_bound_week_counts_as_correct():
    st, params = _state_at(0)
    deep = type(st)(**{**st.__dict__, "backorders": st.backorders + 5000})
    adv = make_advice(deep, params, 0, True, np.random.default_rng(0))
    assert adv.q_star > params.max_order and adv.q_ai == params.max_order and adv.ai_correct


def test_unknown_url_is_404_not_a_false_data_loss_500(client):
    """A mistyped URL must not tell a participant their trial was lost."""
    r = client.get("/no-such-page")
    assert r.status_code == 404
    assert "not saved" not in r.get_data(as_text=True)
    assert client.get("/favicon.ico").status_code == 204
