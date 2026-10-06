"""Researcher tool: withdraw a participant after their session has ended.

For requests that arrive by email or Prolific message, after the browser
session is gone. Identify the person by the participant id they typed or by
their Prolific id.

    python src/interface/withdraw.py --participant P017
    python src/interface/withdraw.py --prolific 5f1c...e2 --keep-data
    python src/interface/withdraw.py --participant P017 --db data/trials.sqlite3

Default is full deletion (trials, questionnaires, event log, Prolific id,
numeracy score, reliability estimate). --keep-data only marks the person
withdrawn and excluded. The bare participant row always stays, so that
arrival-order alternation is not disturbed for anyone still to be run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interface import db  # noqa: E402


def _where(target) -> str:
    """Which database was searched, for the not-found message. Never the URL
    itself (it carries the password), and never the SQLite default path when
    DATABASE_URL actually sent the lookup to Postgres."""
    if db.backend(target) == "postgres":
        return "postgres (--db URL)" if target else "postgres (DATABASE_URL)"
    return str(target or db.DEFAULT_DB)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--participant")
    who.add_argument("--prolific")
    ap.add_argument("--keep-data", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)

    conn = db.connect(args.db or db.DEFAULT_DB)
    pid = args.participant
    if args.prolific:
        row = conn.execute("SELECT participant_id FROM participant WHERE prolific_pid=?",
                           (args.prolific,)).fetchone()
        if row is None:
            print("no participant with that Prolific id (already deleted, or never started)")
            return 1
        pid = row["participant_id"]

    if not db.withdraw_participant(conn, pid, delete_data=not args.keep_data):
        print("no participant %r in %s" % (pid, _where(args.db)))
        return 1
    print("%s withdrawn%s" % (pid, "" if args.keep_data else "; all data deleted"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
