"""SIMULATED participants for testing the analysis pipeline -- NOT study data.

Drives the REAL participant app code (interface.app.Session: the same advice
schedule, live LightGBM risk, simulator, Q* and trial logging a participant
gets) with synthetic participants whose behaviour has KNOWN ground truth, and
writes a fresh SQLite database under data/simulated/.

Behaviour of a synthetic participant i on main trial t
------------------------------------------------------
    Q_own   = Q*_t + bias_i + e,     bias_i ~ N(0, 20), e ~ N(0, sd_i), sd_i ~ U(55, 85)
    w_t     = w_i + ai_err_shift*[AI wrong at t] - drop(prev) + N(0, 0.25), clipped [-0.2, 1.2]
    Q_final = round(Q_own + w_t (Q_AI - Q_own))           (the app clips to [0, cap])

EFFECT dataset (planted ground truth):
    w_i      = 0.65 (80 %) or 0.45 (60 %), + u_i, u_i ~ N(0, 0.10) centred
               within condition (between_offsets)          -> H1: +0.20 WoA
    ai_err_shift = -0.15 (errors partly detected)
    drop     = 0.20 after a PROPAGATING error, 0.05 after an ABSORBED one
                                                           -> H3: -0.15 WoA
    trust    = 4.4 + u_i + 0.30*block (80 %) or -0.30*block (60 %)
                                                           -> H2: divergence 1.2 by block 3
    perceived reliability ~ N(75, 10) at 80 %, N(62, 10) at 60 %
NULL dataset: w_i = 0.55 in both conditions, no error shift, no drop, flat
trust, perceived ~ N(68, 10) in both.

Consequence (not planted, arises from the app): with the same weighting, worse
advice produces worse decisions, so the 60 % condition has higher decision
error in BOTH datasets (H4a). The null dataset nulls the behavioural effects
(H1, H2, H3), not that mechanical consequence.

Participants built to test exclusions (both datasets):
    idx 3   click-through: 8 of 35 stage-1 RTs < 500 ms (22.9 %)  -> excluded
    idx 4   boundary: 7 of 35 (exactly 20 %)                    -> NOT excluded
    idx 10  constant Q_own = 200 on every main trial              -> excluded
    idx 20  stops after main trial 18                             -> excluded (incomplete)
    idx 30  completes, then withdraws keeping data               -> excluded (withdrawn)
    idx 40  withdraws mid-task with deletion                      -> excluded (withdrawn)
    idx 50  a technical_fault event is logged                    -> excluded
Conditions alternate by arrival (even index 80 %, odd 60 %).

Run:  python src/analysis/simulate_participants.py   (writes both datasets)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import analysis  # noqa: F401
from interface import db
from interface.advice import placement_of
from interface.app import Session
from interface.world import session_params
from policy import reference_order

ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "data" / "simulated"
PREFIX = "SIM-"

CLICK_THROUGH, BOUNDARY_FAST, CONSTANT_OWN = 3, 4, 10
INCOMPLETE, WITHDRAW_KEEP, WITHDRAW_DELETE, TECH_FAULT = 20, 30, 40, 50

EFFECT = dict(w_high=0.65, w_low=0.45, ai_err_shift=-0.15, drop_prop=0.20,
              drop_abs=0.05, trust_slope=0.30, perceived_high=75, perceived_low=62)
NULL = dict(w_high=0.55, w_low=0.55, ai_err_shift=0.0, drop_prop=0.0,
            drop_abs=0.0, trust_slope=0.0, perceived_high=68, perceived_low=68)


def _rts(rng, n, n_fast):
    """Stage-1 response times: n_fast click-through ones (< 500 ms), rest slow."""
    rt = np.exp(rng.normal(np.log(6000), 0.4, n)).astype(int).clip(900, 60000)
    if n_fast:
        idx = rng.choice(n, size=n_fast, replace=False)
        rt[idx] = rng.integers(150, 480, n_fast)
    return rt.tolist()


def between_offsets(n: int, seed: int, sd: float = 0.10) -> np.ndarray:
    """Participant WoA offsets u_i ~ N(0, sd), centred WITHIN each condition.

    Centring makes the planted between-condition difference hold exactly in
    the sample (0.20 in the effect data, 0 in the null data), so a recovery
    test compares against the true in-sample value rather than against a
    value that one unlucky draw of 20 people can move by +-0.08.
    """
    u = np.random.default_rng([seed, 999_999]).normal(0, sd, n)
    for parity in (0, 1):
        idx = np.arange(parity, n, 2)
        if len(idx):
            u[idx] -= u[idx].mean()
    return u


def simulate_one(db_path, idx: int, truth: dict, seed: int, special: bool = True,
                 w_offset: float | None = None):
    pid = "%s%03d" % (PREFIX, idx)
    cond = "high" if idx % 2 == 0 else "low"
    params = session_params()
    rng = np.random.default_rng([seed, idx])
    sess = Session(pid, cond, params, db_path=db_path, consented=True)
    db.log_event(sess.conn, pid, "simulated_participant",
                 {"note": "SIMULATED -- not study data", "truth": truth})

    bias = rng.normal(0, 20)
    sd_own = rng.uniform(55, 85)
    if w_offset is None:
        w_offset = rng.normal(0, 0.10)
    w_i = (truth["w_high"] if cond == "high" else truth["w_low"]) + w_offset
    n_fast = {CLICK_THROUGH: 8, BOUNDARY_FAST: 7}.get(idx, 0) if special else 0
    rts = _rts(rng, 5 + 30, n_fast)

    k = 0
    stop_after = 18 if (special and idx == INCOMPLETE) else None
    for phase in ("practice", "main"):
        if phase == "main":
            sess.begin_main()
        prev_err_place = None
        while sess.trial_index < sess.n_trials:
            if phase == "main" and stop_after is not None and sess.trial_index > stop_after:
                sess.conn.close()
                return pid
            if special and idx == WITHDRAW_DELETE and phase == "main" and sess.trial_index == 12:
                db.withdraw_participant(sess.conn, pid, delete_data=True)
                sess.conn.close()
                return pid
            st = sess.state
            q_star = reference_order(st, params)
            if special and idx == CONSTANT_OWN and phase == "main":
                q_own = 200
            else:
                q_own = int(np.clip(round(q_star + bias + rng.normal(0, sd_own)), 0, params.max_order))
            adv = sess.stage1(q_own, rts[k])
            k += 1

            ai_wrong = not sess.schedule[sess.trial_index]
            w = w_i + rng.normal(0, 0.25)
            if ai_wrong:
                w += truth["ai_err_shift"]
            if prev_err_place == "propagating":
                w -= truth["drop_prop"]
            elif prev_err_place == "absorbed":
                w -= truth["drop_abs"]
            w = float(np.clip(w, -0.2, 1.2))
            q_final = int(round(q_own + w * (adv["q_ai"] - q_own)))
            sess.stage2(q_final, int(rng.integers(1500, 9000)))

            if phase == "main":
                prev_err_place = (placement_of(sess.trial_index - 1, sess.design)
                                  if ai_wrong else None)
                if sess.trial_index % 10 == 0:
                    _questionnaires(sess, rng, cond, truth, sess.trial_index // 10 - 1)

    if special and idx == TECH_FAULT:
        db.log_event(sess.conn, pid, "technical_fault",
                     {"note": "SIMULATED fault to test exclusion criterion 5"})
    mu = truth["perceived_high"] if cond == "high" else truth["perceived_low"]
    db.finish_participant(sess.conn, pid, perceived_reliability=int(np.clip(round(rng.normal(mu, 10)), 0, 100)))
    db.log_event(sess.conn, pid, "session_end")
    if special and idx == WITHDRAW_KEEP:
        db.log_event(sess.conn, pid, "withdrawn_keep_data")
        db.withdraw_participant(sess.conn, pid, delete_data=False)
    sess.conn.close()
    return pid


def _questionnaires(sess, rng, cond, truth, block):
    if not hasattr(sess, "_sim_trust_u"):
        sess._sim_trust_u = rng.normal(0, 0.6)
    sign = 1 if cond == "high" else -1
    trust = 4.4 + sess._sim_trust_u + sign * truth["trust_slope"] * block
    tia = {}
    for i in range(1, 7):
        x = trust + rng.normal(0, 0.7)
        if i <= 3:          # distrust items, stored as answered
            x = 8 - x
        tia["tia_%d" % i] = float(np.clip(round(x), 1, 7))
    tlx = {k: float(np.clip(round(rng.normal(10, 3)), 1, 20))
           for k in ("mental", "temporal", "performance", "effort", "frustration")}
    db.insert_block_measure(sess.conn, sess.participant_id, block, "tia", tia)
    db.insert_block_measure(sess.conn, sess.participant_id, block, "tlx", tlx)


def simulate(db_path, n: int = 80, effects: bool = True, seed: int = 20261019,
             special: bool = True, verbose: bool = False) -> Path:
    db_path = Path(db_path)
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    truth = EFFECT if effects else NULL
    offsets = between_offsets(n, seed)
    for i in range(n):
        simulate_one(db_path, i, truth, seed, special=special, w_offset=float(offsets[i]))
        if verbose and (i + 1) % 10 == 0:
            print("  %d / %d participants" % (i + 1, n))
    return db_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", default=str(SIM_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    write_readme(out)
    for name, eff in (("sim_effect.sqlite3", True), ("sim_null.sqlite3", False)):
        print("simulating %s (n=%d) ..." % (name, args.n))
        simulate(out / name, n=args.n, effects=eff, verbose=True)
    print("SIMULATED -- not study data -> %s" % out)


def write_readme(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(
        "# SIMULATED -- not study data\n\n"
        "Synthetic participants generated by `src/analysis/simulate_participants.py`,\n"
        "driven through the real app code (`interface.app.Session`) to test the\n"
        "analysis pipeline. No human produced any row in these files.\n\n"
        "* `sim_effect.sqlite3` -- planted effects (H1 +0.20 WoA, H2 trust divergence,\n"
        "  H3 -0.15 WoA after propagating vs absorbed errors).\n"
        "* `sim_null.sqlite3` -- no behavioural effects.\n\n"
        "Both contain participants built to trip each exclusion criterion.\n"
        "Every participant also carries a `simulated_participant` event.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
