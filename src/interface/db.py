"""Trial logging. One row per trial, written in one transaction.

Two backends behind one `connect()`:

* SQLite (the default; tests, dry runs, local pilots). The file is the dataset,
  it must survive a crash mid-session, and it must be readable by anyone who
  opens it in five years with nothing but the sqlite3 CLI.
* PostgreSQL, when the environment variable DATABASE_URL is set (Cloud SQL in
  production). Cloud Run's filesystem is in-memory and per instance, so a
  SQLite file there would vanish with the instance -- and the dataset with it.

Deliberately plain: no ORM, no migration framework. Every call site writes
sqlite-style SQL (`?` placeholders, `row[0]` and `row["col"]` access); the
Postgres wrapper below translates, so app.py, verify_log.py and withdraw.py
never need to know which backend they are on. The SQL itself is kept to the
common dialect: `INSERT ... ON CONFLICT` instead of `INSERT OR IGNORE/REPLACE`
(SQLite >= 3.24 supports it), no AUTOINCREMENT outside the schema files.
"""

from __future__ import annotations

import atexit
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
PG_SCHEMA_PATH = Path(__file__).with_name("schema_postgres.sql")
DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "trials.sqlite3"

# Duplicate-key errors from either backend. app.py maps every one of them to
# "this trial was already recorded" (409); a tuple so callers can `except` it.
INTEGRITY_ERRORS: tuple[type[BaseException], ...] = (sqlite3.IntegrityError,)
try:  # psycopg is only needed when DATABASE_URL is set; tests run without it
    import psycopg as _psycopg
    INTEGRITY_ERRORS = INTEGRITY_ERRORS + (_psycopg.IntegrityError,)
except ImportError:  # pragma: no cover - depends on the environment
    _psycopg = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _is_pg_url(target) -> bool:
    return isinstance(target, str) and target.startswith(("postgres://", "postgresql://"))


def database_url() -> str | None:
    """The Postgres URL from the environment, read at CALL time (tests set it).

    Stripped, because a hosting dashboard's multi-line value box keeps whatever
    whitespace was pasted into it. A trailing newline ends up INSIDE the last
    query parameter and libpq rejects the entire connection -- for example
    `invalid channel_binding value` when the pasted string ended just after
    channel_binding=require. That took a Render deploy down on 6 Oct 2026:
    the image built and gunicorn started, but every connection attempt failed
    and the health check never passed, so the deploy timed out after 15m49s.
    """
    raw = os.environ.get("DATABASE_URL")
    return raw.strip() if raw and raw.strip() else None


def backend(target=None) -> str:
    """'postgres' or 'sqlite' -- which store `connect(target)` would open."""
    return "postgres" if _resolve(target)[0] == "pg" else "sqlite"


def _resolve(target):
    # Resolution rule, and why:
    #   * a postgres:// URL is always Postgres;
    #   * an explicit file path OTHER than DEFAULT_DB is always that SQLite file
    #     (a dry run or a verify of a named file must never silently land in
    #     the production database because DATABASE_URL happens to be set);
    #   * None or DEFAULT_DB means "the study database": DATABASE_URL if set,
    #     else the DEFAULT_DB file. withdraw.py passes `args.db or DEFAULT_DB`,
    #     and must reach Cloud SQL in production without being edited.
    if _is_pg_url(target):
        return "pg", target
    default = Path(DEFAULT_DB)
    if target is None or Path(target) == default:
        url = database_url()
        if url:
            return "pg", url
        return "sqlite", default
    return "sqlite", Path(target)


def connect(target: Path | str | None = None):
    """Open the study database. See `_resolve` for which one.

    Both backends return an object with sqlite3's calling convention:
    `execute(sql, params)` with `?` placeholders, rows indexable by position
    and by column name, autocommit unless an explicit BEGIN is issued.
    """
    # Resolved at CALL time. A default of `path=DEFAULT_DB` is frozen when the
    # module is imported, so redirecting DEFAULT_DB (as the tests do) silently
    # left the arrival count and screening log on the real data file.
    kind, where = _resolve(target)
    if kind == "pg":
        return PgConnection(where)
    path = Path(where)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because Flask runs threaded=True and spawns a new
    # thread per request, while a Session caches one connection across requests.
    # Without this, every trial write raises sqlite3.ProgrammingError and the
    # trial table stays empty. Safe here only because Session serialises all of
    # its own writes behind a per-session lock.
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _migrate(conn)
    return conn


def _migrate(conn) -> None:
    """Add columns introduced after a SQLite database file was first created.

    CREATE TABLE IF NOT EXISTS leaves an existing table untouched, so a DB made
    by an older build would lack the withdrawal columns and every withdrawal
    would fail mid-session. New TABLES (pending_stage1) need nothing here: the
    schema script creates them on every connect. Postgres does the same job
    with ADD COLUMN IF NOT EXISTS in schema_postgres.sql.
    """
    have = {r[1] for r in conn.execute("PRAGMA table_info(participant)")}
    for col, decl in (("withdrawn_utc", "TEXT"),
                      ("data_deleted", "INTEGER NOT NULL DEFAULT 0"),
                      # 1.4 deployment columns: resume authentication and the
                      # device the session ran on (desktop-only gate).
                      ("token_sha256", "TEXT"),
                      ("device_ua", "TEXT"),
                      ("screen_w", "INTEGER"),
                      ("screen_h", "INTEGER"),
                      ("viewport_w", "INTEGER"),
                      ("pointer_coarse", "INTEGER")):
        if col not in have:
            conn.execute("ALTER TABLE participant ADD COLUMN %s %s" % (col, decl))


# ------------------------------------------------------------------ Postgres
class Row(tuple):
    """A result row that behaves like sqlite3.Row: `r[0]`, `r["col"]`,
    `r.keys()`, `dict(r)` and tuple unpacking all work."""

    def __new__(cls, values, names):
        obj = super().__new__(cls, values)
        obj._names = names
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return tuple.__getitem__(self, self._names.index(key))
            except ValueError:
                raise IndexError("no such column: %s" % key) from None
        return tuple.__getitem__(self, key)

    def keys(self):
        return list(self._names)


class _Result:
    """Rows fetched eagerly, so the pooled connection can go straight back.

    Every query in this app returns a handful of rows (one participant's
    trials at most), so holding them in memory costs nothing.
    """

    def __init__(self, rows, rowcount):
        self._rows = rows
        self._pos = 0
        self.rowcount = rowcount

    def fetchone(self):
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self):
        rows = self._rows[self._pos:]
        self._pos = len(self._rows)
        return rows

    def __iter__(self):
        return iter(self.fetchall())


def _to_pg(sql: str) -> str:
    """sqlite `?` placeholders -> psycopg `%s`, skipping quoted literals.

    A literal `%` must be doubled once placeholders are in play, or psycopg
    reads it as the start of one.
    """
    out, quote = [], None
    for ch in sql:
        if quote:
            if ch == quote:
                quote = None
            out.append("%%" if ch == "%" else ch)
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "?":
            out.append("%s")
        elif ch == "%":
            out.append("%%")
        else:
            out.append(ch)
    return "".join(out)


def _row_factory(cursor):
    names = [d.name for d in cursor.description] if cursor.description else []

    def make(values):
        return Row(values, names)
    return make


_POOLS: dict[str, object] = {}
_SCHEMA_READY: set[str] = set()
_POOL_LOCK = Lock()


def _pool(url: str):
    """One small connection pool per process (per URL).

    Why a pool rather than one connection per participant: every Session keeps
    its connection object for the life of the process, and a Cloud SQL
    instance allows only a few dozen connections. A connection per Session
    would hit that ceiling long before 80 participants. Here a Session's
    "connection" is a lightweight handle that borrows a real connection for
    each statement (or for one explicit transaction) and gives it back.
    """
    with _POOL_LOCK:
        pool = _POOLS.get(url)
        if pool is None:
            from psycopg_pool import ConnectionPool
            pool = ConnectionPool(
                url, min_size=1, max_size=int(os.environ.get("DB_POOL_MAX", "8")),
                # prepare_threshold=None disables psycopg's automatic server-side
                # PREPARE. By default it prepares a statement after 5 executions,
                # and this app runs the same statements thousands of times. Behind a
                # TRANSACTION-mode connection pooler (Supabase Supavisor, PgBouncer)
                # a named prepared statement outlives the pooled backend it was made
                # on, so later executions fail with "prepared statement already
                # exists" -- and only once traffic is real, i.e. mid-collection.
                # PgBouncer >= 1.21 with max_prepared_statements > 0 handles it
                # (verified against 1.26 on 5 Oct 2026), but that setting is not ours
                # to control on a managed database. Disabling it costs nothing here:
                # the app serves one participant at a time, nowhere near a rate where
                # statement preparation matters.
                kwargs={"autocommit": True, "prepare_threshold": None},
                # A pooled connection can be dropped by Cloud SQL or the proxy
                # while idle; check before handing it out, not on the next
                # participant's trial write.
                check=ConnectionPool.check_connection,
                open=True, name="reliance")
            _POOLS[url] = pool
            if len(_POOLS) == 1:
                atexit.register(_close_pools)
        if url not in _SCHEMA_READY:
            with pool.connection() as conn:
                # Several Cloud Run instances can boot at once; concurrent
                # CREATE TABLE IF NOT EXISTS can still collide in Postgres's
                # catalog, so serialise schema creation across processes.
                conn.execute("SELECT pg_advisory_lock(20270924)")
                try:
                    conn.execute(PG_SCHEMA_PATH.read_text(encoding="utf-8"))
                finally:
                    conn.execute("SELECT pg_advisory_unlock(20270924)")
            _SCHEMA_READY.add(url)
        return pool


def _close_pools() -> None:
    """Close every pool at interpreter exit.

    Left to the garbage collector, psycopg_pool tries to join its worker and
    scheduler threads during shutdown, fails ("couldn't stop thread ... within
    5.0 seconds") and every CLI run against Postgres (verify_log.py,
    withdraw.py) sat for ~20 s before exiting.
    """
    with _POOL_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
        _SCHEMA_READY.clear()
    for pool in pools:
        try:
            pool.close(timeout=5.0)
        except Exception:  # pragma: no cover - best effort at shutdown
            pass


def _pg_param(value):
    """Postgres TEXT cannot hold NUL (0x00); SQLite TEXT can.

    Every string that reaches the database from a browser (Prolific id, user
    agent) could carry one -- `?PROLIFIC_PID=ab%00cd` is enough -- and on
    Postgres the write raised DataError, a 500 in the middle of /api/start
    after the participant row already existed. Replaced with U+FFFD so the
    write lands and the damage stays visible in the data.
    """
    if isinstance(value, str) and "\x00" in value:
        return value.replace("\x00", "\ufffd")
    return value


def _pg_params(params):
    return tuple(_pg_param(v) for v in params)


class PgConnection:
    """sqlite3.Connection look-alike over a psycopg pool.

    Autocommit by default, like `isolation_level=None` on the SQLite side. An
    explicit `execute("BEGIN")` pins one pooled connection to this handle until
    COMMIT or ROLLBACK, so withdraw_participant's transaction keeps its meaning.
    """

    def __init__(self, url: str):
        if _psycopg is None:
            raise RuntimeError("DATABASE_URL is set but psycopg is not installed "
                               "(python -m pip install \"psycopg[binary,pool]\")")
        self.url = url
        self._pool = _pool(url)
        self._pinned = None   # the connection holding an open transaction
        self._lock = Lock()

    def _run(self, conn, sql, params, many=False):
        cur = conn.cursor(row_factory=_row_factory)
        try:
            if many:
                cur.executemany(_to_pg(sql), [_pg_params(p) for p in params])
            elif params:
                cur.execute(_to_pg(sql), _pg_params(params))
            else:
                # No parameters: psycopg does no placeholder parsing, so the
                # text goes through untouched (literal % included).
                cur.execute(sql)
            rows = cur.fetchall() if cur.description else []
            return _Result(rows, cur.rowcount)
        finally:
            cur.close()

    def execute(self, sql: str, params=()):
        verb = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
        with self._lock:
            if verb == "BEGIN":
                if self._pinned is not None:
                    raise RuntimeError("transaction already open on this connection")
                self._pinned = self._pool.getconn()
                try:
                    self._pinned.execute("BEGIN")
                except Exception:
                    self._pool.putconn(self._pinned)
                    self._pinned = None
                    raise
                return _Result([], -1)
            if verb in ("COMMIT", "ROLLBACK") and self._pinned is not None:
                conn, self._pinned = self._pinned, None
                try:
                    conn.execute(verb)
                finally:
                    self._pool.putconn(conn)
                return _Result([], -1)
            if self._pinned is not None:
                return self._run(self._pinned, sql, params)
        with self._pool.connection() as conn:
            return self._run(conn, sql, params)

    def executemany(self, sql: str, seq):
        with self._lock:
            if self._pinned is not None:
                return self._run(self._pinned, sql, seq, many=True)
        with self._pool.connection() as conn:
            return self._run(conn, sql, seq, many=True)

    def close(self) -> None:
        # The pool owns the real connections; a handle has nothing to close
        # except a transaction someone forgot to finish.
        with self._lock:
            if self._pinned is not None:
                conn, self._pinned = self._pinned, None
                try:
                    conn.execute("ROLLBACK")
                finally:
                    self._pool.putconn(conn)


# ------------------------------------------------------------------ helpers
@contextmanager
def transaction(conn):
    """BEGIN ... COMMIT, ROLLBACK on any exception. Works on both backends."""
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def register_participant(conn, participant_id, condition, seed, params,
                         app_version, prolific_pid=None) -> None:
    conn.execute(
        """INSERT INTO participant
           (participant_id, condition, prolific_pid, seed, started_utc,
            app_version, params_json)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT (participant_id) DO NOTHING""",
        (participant_id, condition, prolific_pid, seed, utc_now(), app_version,
         json.dumps(params, sort_keys=True)),
    )


def log_event(conn, participant_id, kind, detail=None) -> None:
    conn.execute(
        "INSERT INTO event_log (participant_id, kind, detail, recorded_utc) VALUES (?,?,?,?)",
        (participant_id, kind, json.dumps(detail) if detail is not None else None, utc_now()),
    )


# Column order is fixed here and nowhere else; the app passes a dict.
_TRIAL_COLUMNS = (
    "participant_id trial_index phase block period demand inventory on_order "
    "lead_time backorders supplier_reliability transport_ok disruption "
    "q_ai ai_risk_pct ai_level ai_correct placement q_star delta "
    "q_own q_final response_class woa error_own error_final own_correct "
    "final_correct rt_stage1_ms rt_stage2_ms demand_realised cost stockout service_level "
    "shipped arrivals inventory_close backorders_close submitted_utc"
).split()


def insert_trial(conn, row: dict) -> None:
    missing = [c for c in _TRIAL_COLUMNS if c not in row]
    if missing:
        raise ValueError("trial row missing columns: %s" % ", ".join(missing))
    placeholders = ",".join("?" * len(_TRIAL_COLUMNS))
    conn.execute(
        "INSERT INTO trial (%s) VALUES (%s)" % (",".join(_TRIAL_COLUMNS), placeholders),
        tuple(row[c] for c in _TRIAL_COLUMNS),
    )


def record_trial(conn, row: dict) -> None:
    """Insert the trial row AND clear its pending stage 1, atomically.

    If the two were separate statements, a crash between them would leave a
    pending row for a trial already recorded, and a restart would offer the
    participant that trial's advice a second time.
    """
    with transaction(conn):
        insert_trial(conn, row)
        conn.execute("DELETE FROM pending_stage1 WHERE participant_id=?",
                     (row["participant_id"],))


# ---------------------------------------------------------- pending stage 1
def save_pending_stage1(conn, participant_id, phase, trial_index, q_own, rt_ms,
                        advice: dict) -> None:
    """Persist a locked Q_own and the advice it unlocked.

    Plain INSERT on purpose: the primary key is the participant, so a second
    stage 1 for the same person -- from a second instance, a second tab, a
    retry -- fails loudly instead of overwriting the first one.
    """
    conn.execute(
        """INSERT INTO pending_stage1
           (participant_id, phase, trial_index, q_own, rt_stage1_ms, advice_json,
            recorded_utc) VALUES (?,?,?,?,?,?,?)""",
        (participant_id, phase, int(trial_index), int(q_own), int(rt_ms),
         json.dumps(advice, sort_keys=True), utc_now()),
    )


def load_pending_stage1(conn, participant_id):
    row = conn.execute(
        "SELECT phase, trial_index, q_own, rt_stage1_ms, advice_json "
        "FROM pending_stage1 WHERE participant_id=?", (participant_id,)).fetchone()
    if row is None:
        return None
    return {"phase": row["phase"], "trial_index": row["trial_index"],
            "q_own": row["q_own"], "rt1": row["rt_stage1_ms"],
            "advice": json.loads(row["advice_json"])}


def insert_block_measure(conn, participant_id, block, instrument, responses: dict) -> None:
    # Upsert: re-submitting a block's questionnaire (a retry after a network
    # error) replaces the earlier answer rather than failing.
    conn.executemany(
        """INSERT INTO block_measure
           (participant_id, block, instrument, item, response, recorded_utc)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT (participant_id, block, instrument, item)
           DO UPDATE SET response=excluded.response, recorded_utc=excluded.recorded_utc""",
        [(participant_id, block, instrument, k, float(v), utc_now())
         for k, v in responses.items()],
    )


def finish_participant(conn, participant_id, perceived_reliability=None) -> None:
    conn.execute(
        "UPDATE participant SET finished_utc=?, perceived_reliability=? WHERE participant_id=?",
        (utc_now(), perceived_reliability, participant_id),
    )


def record_screening(conn, numeracy_score: int, passed: bool) -> None:
    conn.execute(
        "INSERT INTO screening (numeracy_score, passed, recorded_utc) VALUES (?,?,?)",
        (int(numeracy_score), int(passed), utc_now()),
    )


def withdraw_participant(conn, participant_id: str, delete_data: bool) -> bool:
    """Record a withdrawal and, if asked, delete everything the person produced.

    Kept: participant_id, condition, seed, app_version, params -- none of it
    describes the person, and the row must stay so arrival-order alternation
    does not shift for later participants. Removed on deletion: every trial and
    questionnaire row, every event-log row, any pending stage 1, the Prolific
    ID, the numeracy score, the perceived-reliability answer and the device
    description (a user-agent string is close enough to a fingerprint).
    Returns False if the id is unknown.
    """
    row = conn.execute("SELECT 1 FROM participant WHERE participant_id=?",
                       (participant_id,)).fetchone()
    if row is None:
        return False
    conn.execute("BEGIN")
    try:
        if delete_data:
            conn.execute("DELETE FROM trial WHERE participant_id=?", (participant_id,))
            conn.execute("DELETE FROM block_measure WHERE participant_id=?", (participant_id,))
            conn.execute("DELETE FROM event_log WHERE participant_id=?", (participant_id,))
            conn.execute(
                """UPDATE participant SET prolific_pid=NULL, numeracy_score=NULL,
                   perceived_reliability=NULL, device_ua=NULL, screen_w=NULL,
                   screen_h=NULL, viewport_w=NULL, pointer_coarse=NULL,
                   data_deleted=1
                   WHERE participant_id=?""", (participant_id,))
        # Either way the session can never be resumed: no half-finished trial
        # survives, and the resume credential is gone.
        conn.execute("DELETE FROM pending_stage1 WHERE participant_id=?", (participant_id,))
        conn.execute(
            """UPDATE participant SET withdrawn_utc=COALESCE(withdrawn_utc, ?),
               excluded=1, exclusion_reason='withdrawn', token_sha256=NULL
               WHERE participant_id=?""", (utc_now(), participant_id))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return True


def export_csv(conn, out_dir: Path | str) -> dict[str, Path]:
    """Flat CSV export, one file per table. Run this after every session."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for table in ("participant", "trial", "block_measure", "event_log", "screening"):
        rows = conn.execute("SELECT * FROM %s" % table).fetchall()
        path = out_dir / ("%s.csv" % table)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if rows:
                writer.writerow(rows[0].keys())
                writer.writerows([tuple(r) for r in rows])
            else:
                writer.writerow(table_columns(conn, table))
        written[table] = path
    return written


def table_columns(conn, table: str) -> list[str]:
    """Column names of `table` on either backend (PRAGMA is SQLite-only)."""
    if isinstance(conn, PgConnection):
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name=? AND table_schema=current_schema() "
            "ORDER BY ordinal_position", (table,)).fetchall()
        return [r[0] for r in rows]
    return [c[1] for c in conn.execute("PRAGMA table_info(%s)" % table)]
