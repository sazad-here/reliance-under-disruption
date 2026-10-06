"""The researcher CLIs and the Postgres-only failure modes, on a real Postgres.

Skipped unless PG_TEST_URL is set (see conftest.py), e.g.

    PG_TEST_URL=postgresql://postgres:test@localhost:55432/postgres

verify_log.py and withdraw.py are run exactly as a researcher would run them
in production: DATABASE_URL in the environment, no --db, in a subprocess. A
scripted session is recorded on Postgres first with dry_run(db_path=<URL>) --
the explicit URL is the intended way; dry_run refuses a bare DATABASE_URL so a
rehearsal can never land in the collection database by accident.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PG_TEST_URL = os.environ.get("PG_TEST_URL")
pytestmark = pytest.mark.skipif(not PG_TEST_URL, reason="PG_TEST_URL not set")

_TABLES = "pending_stage1, block_measure, trial, event_log, screening, participant"
PASS = {"num_1": "500", "num_2": "10", "num_3": "0.1"}


@pytest.fixture()
def pg(monkeypatch, tmp_path):
    """DATABASE_URL -> the test server, every table empty, and the SQLite
    default pointed somewhere harmless in case anything falls back to it."""
    from interface import db
    from interface import app as app_module

    monkeypatch.setenv("DATABASE_URL", PG_TEST_URL)
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "must-not-be-used.sqlite3")
    conn = db.connect(PG_TEST_URL)
    conn.execute("TRUNCATE %s RESTART IDENTITY CASCADE" % _TABLES)
    app_module._SESSIONS.clear()
    yield conn
    conn.close()
    assert not (tmp_path / "must-not-be-used.sqlite3").exists()


def _record_session(pid, condition="low"):
    from interface.app import dry_run
    return dry_run(db_path=PG_TEST_URL, participant_id=pid, condition=condition,
                   verbose=False)


def _cli(script, *args):
    env = dict(os.environ, DATABASE_URL=PG_TEST_URL, PYTHONIOENCODING="utf-8")
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, str(ROOT / "src" / "interface" / script), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=300)
    return proc, time.monotonic() - t0


def _count(conn, table, pid):
    return conn.execute("SELECT COUNT(*) FROM %s WHERE participant_id=?" % table,
                        (pid,)).fetchone()[0]


# ------------------------------------------------------------------ verify_log
def test_verify_log_cli_passes_a_postgres_recorded_session(pg):
    _record_session("PGV-01", "low")
    assert _count(pg, "trial", "PGV-01") == 35

    proc, elapsed = _cli("verify_log.py", "--participant", "PGV-01")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed, 0 warnings" in proc.stdout
    assert "replay          all 30 main trials replay exactly" in proc.stdout
    assert "verifying DATABASE_URL (postgres)" in proc.stdout
    assert PG_TEST_URL not in proc.stdout + proc.stderr     # never echo the password
    # The connection pool used to be left for the garbage collector, which
    # could not stop its threads and held every CLI run for ~20 s at exit.
    assert "couldn't stop thread" not in proc.stderr
    assert elapsed < 60


def test_verify_in_process_reads_the_same_rows_by_url_and_by_env(pg):
    from interface.verify_log import verify
    _record_session("PGV-02", "high")
    assert verify(PG_TEST_URL, "PGV-02") == 0
    assert verify(None, "PGV-02") == 0          # DATABASE_URL resolution


def test_verify_log_cli_fails_a_tampered_postgres_log(pg):
    _record_session("PGV-03", "low")
    pg.execute("UPDATE trial SET cost=cost+1 WHERE participant_id=? AND phase='main' "
               "AND trial_index=4", ("PGV-03",))
    proc, _ = _cli("verify_log.py", "--participant", "PGV-03")
    assert proc.returncode == 1
    assert "replay diverges" in proc.stdout


def test_verify_log_cli_unknown_participant(pg):
    proc, _ = _cli("verify_log.py", "--participant", "NOBODY")
    assert proc.returncode == 1
    assert "no such participant: NOBODY" in proc.stdout


# ------------------------------------------------------------------ withdraw
def test_withdraw_cli_deletes_everything_but_the_bare_row(pg):
    sess = _record_session("PGW-01", "high")
    other = _record_session("PGW-02", "low")
    before = pg.execute("SELECT condition, seed, app_version, params_json, started_utc "
                        "FROM participant WHERE participant_id='PGW-01'").fetchone()
    pg.execute("UPDATE participant SET prolific_pid='PROL-PG', numeracy_score=3, "
               "device_ua='Mozilla/5.0', screen_w=1920, screen_h=1080, viewport_w=1400, "
               "pointer_coarse=0, token_sha256='abc' WHERE participant_id='PGW-01'")
    # A half-finished trial must go too.
    pg.execute("INSERT INTO pending_stage1 (participant_id, phase, trial_index, q_own, "
               "rt_stage1_ms, advice_json, recorded_utc) VALUES (?,?,?,?,?,?,?)",
               ("PGW-01", "main", 30, 100, 1000, "{}", "2026-09-25T00:00:00.000+00:00"))
    n_other = {t: _count(pg, t, "PGW-02") for t in ("trial", "block_measure", "event_log")}
    assert all(n_other.values()) and sess.finished and other.finished

    proc, elapsed = _cli("withdraw.py", "--prolific", "PROL-PG")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PGW-01 withdrawn; all data deleted" in proc.stdout
    assert "couldn't stop thread" not in proc.stderr and elapsed < 60

    for table in ("trial", "block_measure", "event_log", "pending_stage1"):
        assert _count(pg, table, "PGW-01") == 0, table
    p = pg.execute("SELECT * FROM participant WHERE participant_id='PGW-01'").fetchone()
    assert p is not None                                   # the bare row stays
    for col in ("prolific_pid", "numeracy_score", "perceived_reliability", "device_ua",
                "screen_w", "screen_h", "viewport_w", "pointer_coarse", "token_sha256"):
        assert p[col] is None, col
    assert (p["data_deleted"], p["excluded"], p["exclusion_reason"]) == (1, 1, "withdrawn")
    assert p["withdrawn_utc"]
    assert (p["condition"], p["seed"], p["app_version"], p["params_json"],
            p["started_utc"]) == tuple(before)
    # Nobody else is touched, and alternation still sees both rows.
    assert {t: _count(pg, t, "PGW-02") for t in n_other} == n_other
    assert pg.execute("SELECT COUNT(*) FROM participant").fetchone()[0] == 2

    # A second request for the same person finds nothing to match by Prolific id.
    proc, _ = _cli("withdraw.py", "--prolific", "PROL-PG")
    assert proc.returncode == 1
    assert "no participant with that Prolific id" in proc.stdout


def test_withdraw_cli_keep_data_and_unknown_id(pg):
    _record_session("PGW-03", "low")
    n_trials = _count(pg, "trial", "PGW-03")
    proc, _ = _cli("withdraw.py", "--participant", "PGW-03", "--keep-data")
    assert proc.returncode == 0 and "PGW-03 withdrawn\n" in proc.stdout
    assert _count(pg, "trial", "PGW-03") == n_trials
    p = pg.execute("SELECT excluded, data_deleted, withdrawn_utc, perceived_reliability "
                   "FROM participant WHERE participant_id='PGW-03'").fetchone()
    assert (p[0], p[1], p[3]) == (1, 0, 65) and p[2]

    proc, _ = _cli("withdraw.py", "--participant", "NOPE")
    assert proc.returncode == 1
    # It searched Postgres; the message used to name the SQLite default file.
    assert "no participant 'NOPE' in postgres (DATABASE_URL)" in proc.stdout
    assert PG_TEST_URL not in proc.stdout


def test_withdraw_transaction_rolls_back_on_postgres(pg, monkeypatch):
    """withdraw_participant's BEGIN pins one pooled connection; a failure
    part-way must leave every row as it was."""
    from interface import db
    _record_session("PGW-04", "high")
    n = _count(pg, "trial", "PGW-04")
    conn = db.connect()
    real = conn.execute

    def failing(sql, params=()):
        if sql.lstrip().upper().startswith("DELETE FROM EVENT_LOG"):
            raise RuntimeError("boom")
        return real(sql, params)
    monkeypatch.setattr(conn, "execute", failing)
    with pytest.raises(RuntimeError):
        db.withdraw_participant(conn, "PGW-04", delete_data=True)
    assert _count(pg, "trial", "PGW-04") == n                # trial DELETE rolled back
    assert conn._pinned is None                             # connection returned
    p = pg.execute("SELECT withdrawn_utc, data_deleted FROM participant "
                   "WHERE participant_id='PGW-04'").fetchone()
    assert tuple(p) == (None, 0)


# ------------------------------------------------------------------ db layer
def test_duplicate_trial_rolls_back_and_keeps_pending_on_postgres(pg):
    from interface import db
    _record_session("PGD-01", "low")
    row = dict(pg.execute("SELECT * FROM trial WHERE participant_id='PGD-01' "
                          "AND phase='main' AND trial_index=0").fetchone())
    pg.execute("INSERT INTO pending_stage1 (participant_id, phase, trial_index, q_own, "
               "rt_stage1_ms, advice_json, recorded_utc) VALUES (?,?,?,?,?,?,?)",
               ("PGD-01", "main", 0, 1, 1, "{}", "x"))
    with pytest.raises(db.INTEGRITY_ERRORS):
        db.record_trial(pg, row)
    assert _count(pg, "pending_stage1", "PGD-01") == 1      # DELETE rolled back
    assert _count(pg, "trial", "PGD-01") == 35


def test_export_csv_on_postgres(pg, tmp_path):
    from interface import db
    _record_session("PGE-01", "low")
    out = db.export_csv(pg, tmp_path / "csv")
    with out["trial"].open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 35 and rows[0]["participant_id"] == "PGE-01"
    assert list(rows[0]) == db.table_columns(pg, "trial")
    with out["screening"].open(encoding="utf-8") as fh:        # empty table: header only
        assert next(csv.reader(fh)) == ["id", "numeracy_score", "passed", "recorded_utc"]


# ------------------------------------------------------------------ HTTP path
@pytest.fixture()
def client(pg):
    from interface import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_nul_byte_in_browser_strings_does_not_500_on_postgres(client, pg):
    """Postgres TEXT rejects 0x00; SQLite stores it. A `%00` in the Prolific
    URL parameter used to 500 /api/start after the participant row existed."""
    r = client.post("/api/start", json={
        "participant_id": "PGN-01", "consented": True, "numeracy": PASS,
        "prolific_pid": "ab\x00cd",
        "device": {"ua": "Mozilla\x00X", "screen_w": 1920, "screen_h": 1080,
                   "viewport_w": 1400, "pointer_coarse": False}})
    assert r.status_code == 200, r.get_json()
    p = pg.execute("SELECT prolific_pid, device_ua, consented FROM participant "
                   "WHERE participant_id='PGN-01'").fetchone()
    assert tuple(p) == ("ab�cd", "Mozilla�X", 1)


@pytest.mark.parametrize("value, status, stored", [
    (65, 200, 65), ("65", 200, 65), (65.0, 200, 65), (0, 200, 0), (100, 200, 100),
    (None, 200, None),
    # Each of these used to be stored as-is on SQLite and to 500 on Postgres.
    (65.5, 400, None), ("abc", 400, None), (10 ** 12, 400, None), (101, 400, None),
    (-1, 400, None), (True, 400, None), ([1], 400, None),
])
def test_perceived_reliability_is_validated(client, pg, value, status, stored):
    client.post("/api/start", json={"participant_id": "PGR-01", "consented": True,
                                    "numeracy": PASS})
    r = client.post("/api/finish", json={"perceived_reliability": value})
    assert r.status_code == status, r.get_json()
    p = pg.execute("SELECT perceived_reliability, finished_utc FROM participant "
                   "WHERE participant_id='PGR-01'").fetchone()
    assert p[0] == stored
    assert (p[1] is not None) == (status == 200)
    if status == 400:
        assert "between 0 and 100" in r.get_json()["error"]
