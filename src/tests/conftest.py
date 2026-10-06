"""Database backend selection for the test suite.

Default: every test runs on SQLite only, exactly as before. With

    PG_TEST_URL=postgresql://postgres:test@localhost:55432/postgres

set, every test in test_interface.py and test_rehydration.py runs twice, once
per backend. On the
Postgres leg DATABASE_URL points at that server and every table is emptied
first, because the HTTP tests count participants and screenings from zero
(alternation, screen-outs) just as they do on a fresh SQLite file.

Tests that pass an explicit SQLite path (dry_run(db_path=...), Session(...,
db_path=...)) stay on that file on both legs -- by design, see db._resolve.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PG_TEST_URL = os.environ.get("PG_TEST_URL")

_PG_TABLES = "pending_stage1, block_measure, trial, event_log, screening, participant"


def pytest_generate_tests(metafunc):
    # Only parametrise when there is a second backend, so the default run's
    # test ids (and count) are unchanged.
    if (PG_TEST_URL and "db_backend" in metafunc.fixturenames
            and metafunc.module.__name__.endswith(("test_interface", "test_rehydration"))):
        metafunc.parametrize("db_backend", ["sqlite", "postgres"], indirect=True)


@pytest.fixture(autouse=True)
def db_backend(request, monkeypatch):
    backend = getattr(request, "param", "sqlite")
    if backend == "postgres":
        from interface import db
        monkeypatch.setenv("DATABASE_URL", PG_TEST_URL)
        conn = db.connect(PG_TEST_URL)
        conn.execute("TRUNCATE %s RESTART IDENTITY CASCADE" % _PG_TABLES)
        conn.close()
    else:
        # A DATABASE_URL left in the developer's shell must not send the
        # SQLite leg to a real server.
        monkeypatch.delenv("DATABASE_URL", raising=False)
    return backend
