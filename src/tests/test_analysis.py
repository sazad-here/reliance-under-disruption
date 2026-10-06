"""Analysis pipeline (src/analysis) on SIMULATED participants -- not study data.

Synthetic participants are driven through the real app code with known
ground truth (simulate_participants.py). The tests assert that the frozen
confirmatory pipeline
  * applies the OSF s8 exclusions exactly (including the 20 % boundary),
  * recovers planted H1/H2/H3 effects (right sign, 95 % CI excludes 0,
    CI covers the planted value), and
  * does not find them in a null dataset.
Kept small (40 participants per dataset) so the module runs in about a
minute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from analysis import confirmatory, exclusions, load, measures  # noqa: E402
from analysis import simulate_participants as sim  # noqa: E402

N = 40   # covers exclusion specials 3, 4, 10, 20, 30 (withdrawn, kept data)


@pytest.fixture(scope="module")
def effect(tmp_path_factory):
    path = sim.simulate(tmp_path_factory.mktemp("sim") / "effect.sqlite3", n=N, effects=True)
    return _pipeline(path)


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    path = sim.simulate(tmp_path_factory.mktemp("sim") / "null.sqlite3", n=N, effects=False)
    return _pipeline(path)


def _pipeline(path):
    data = load.load(path)
    kept, flags, report = exclusions.apply(data)
    m = measures.compute(kept)
    return {"data": data, "flags": flags, "report": report, "m": m, "conf": confirmatory.run(m)}


# ------------------------------------------------------------------ exclusions
def test_simulated_data_is_labelled(effect):
    assert effect["data"].is_simulated


def test_exclusions_hit_exactly_the_planted_participants(effect):
    f = effect["flags"].set_index("participant_id")
    assert f.loc["SIM-003", "primary_reason"] == "fast_stage1"      # 8/35 = 22.9 %
    assert not f.loc["SIM-004", "excluded"]                          # 7/35 = 20 % exactly
    assert f.loc["SIM-004", "share_fast"] == pytest.approx(0.2)
    assert f.loc["SIM-010", "primary_reason"] == "constant_own"
    assert f.loc["SIM-020", "primary_reason"] == "incomplete"
    assert f.loc["SIM-030", "primary_reason"] == "withdrawn"
    assert set(f.index[f["excluded"]]) == {"SIM-003", "SIM-010", "SIM-020", "SIM-030"}
    rep = effect["report"]
    assert rep["n_excluded"] == 4 and rep["n_analysed"] == N - 4
    assert rep["by_condition"]["low"]["primary_reason"]["fast_stage1"] == 1
    assert rep["by_condition"]["high"]["primary_reason"]["constant_own"] == 1
    assert rep["by_condition"]["high"]["primary_reason"]["incomplete"] == 1


def test_withdrawal_and_fault_exclusions(tmp_path):
    path = tmp_path / "w.sqlite3"
    truth = sim.EFFECT
    for idx in (sim.WITHDRAW_KEEP, sim.WITHDRAW_DELETE, sim.TECH_FAULT, 52):
        sim.simulate_one(path, idx, truth, seed=1)
    data = load.load(path)
    kept, flags, report = exclusions.apply(data)
    f = flags.set_index("participant_id")
    assert f.loc["SIM-030", "primary_reason"] == "withdrawn"         # kept data
    assert f.loc["SIM-040", "withdrawn"] and f.loc["SIM-040", "data_deleted"] == 1
    assert f.loc["SIM-050", "primary_reason"] == "technical_fault"
    assert not f.loc["SIM-052", "excluded"]
    assert list(kept.participants["participant_id"]) == ["SIM-052"]
    assert report["by_condition"]["high"]["withdrawn_with_deletion"] == 1


def test_client_error_excludes_only_main_phase_with_failed_replay(tmp_path):
    """Criterion 5(b): a client_error excludes only if it is in the main phase
    AND the log fails the verify_log replay check; a main-phase error on a
    log that still replays is flagged and kept."""
    from interface import db
    path = tmp_path / "ce.sqlite3"
    none_, practice, main_clean, main_tampered = "SIM-052", "SIM-054", "SIM-056", "SIM-058"
    for idx in (52, 54, 56, 58):
        sim.simulate_one(path, idx, sim.EFFECT, seed=1)
    conn = db.connect(path)
    try:
        err = {"kind": "error", "message": "TypeError: x is undefined", "source": "/static/app.js", "line": "1"}
        db.log_event(conn, practice, "client_error", {**err, "phase": "practice", "trial_index": 2})
        db.log_event(conn, practice, "client_error", {**err, "phase": "consent", "trial_index": 0})
        db.log_event(conn, main_clean, "client_error", {**err, "phase": "main", "trial_index": 11})
        db.log_event(conn, main_tampered, "client_error", {**err, "phase": "main", "trial_index": 11})
        # Tamper one logged main-phase outcome so the log no longer replays.
        conn.execute("UPDATE trial SET cost = cost + 1 WHERE participant_id=? AND phase='main' "
                     "AND trial_index=11", (main_tampered,))
    finally:
        conn.close()

    data = load.load(path)
    assert exclusions.replays_cleanly(data, main_clean)
    assert not exclusions.replays_cleanly(data, main_tampered)
    kept, flags, report = exclusions.apply(data)
    f = flags.set_index("participant_id")

    assert not f.loc[none_, "excluded"] and not f.loc[none_, "client_error_flagged"]
    assert not f.loc[practice, "excluded"] and not f.loc[practice, "client_error_main"]
    assert not f.loc[practice, "client_error_flagged"]
    assert not f.loc[main_clean, "excluded"] and f.loc[main_clean, "client_error_flagged"]
    assert f.loc[main_clean, "client_error_replay_ok"] == True  # noqa: E712
    assert f.loc[main_tampered, "excluded"]
    assert f.loc[main_tampered, "primary_reason"] == "technical_fault"
    assert not f.loc[main_tampered, "client_error_flagged"]

    assert set(kept.participants["participant_id"]) == {none_, practice, main_clean}
    assert report["n_client_error_flagged_kept"] == 1
    assert report["by_condition"]["high"]["client_error_flagged_kept"] == 1
    assert report["by_condition"]["high"]["primary_reason"]["technical_fault"] == 1


def test_woa_trial_exclusion_only_affects_woa():
    t = pd.DataFrame({"participant_id": ["a"] * 3, "trial_index": [0, 1, 2],
                      "q_own": [100, 100, 100], "q_ai": [100, 200, 40],
                      "q_final": [100, 150, 100], "q_star": [110, 200, 170],
                      "ai_correct": [1, 1, 0], "placement": ["neutral"] * 3,
                      "stockout": [0, 0, 0]})
    m = measures.trial_measures(t, 60)
    assert np.isnan(m.loc[0, "woa"])                     # Q_AI == Q_own: undefined
    assert m.loc[0, "decision_error"] == 10              # ...but kept elsewhere
    assert m.loc[1, "woa"] == pytest.approx(0.5)
    assert m.loc[2, "woa"] == pytest.approx(0.0)


# ------------------------------------------------------------------ RAIR / RSR
def test_schemmer_cases():
    # own wrong/AI right/final right -> CAIR; own right/AI wrong/final right -> CSR
    t = pd.DataFrame({"participant_id": ["a"] * 4, "trial_index": [0, 1, 2, 3],
                      "q_star": [200, 200, 200, 200],
                      "q_own": [50, 190, 50, 190],
                      "q_ai": [210, 90, 210, 90],
                      "q_final": [205, 185, 60, 100],
                      "ai_correct": [1, 0, 1, 0], "placement": ["neutral"] * 4,
                      "stockout": [0] * 4})
    m = measures.trial_measures(t, 60)
    assert list(m["case_CA"]) == [True, False, True, False]
    assert list(m["case_IA"]) == [False, True, False, True]
    assert list(m["CAIR"]) == [True, False, False, False]
    assert list(m["CSR"]) == [False, True, False, False]
    assert list(m["AR"]) == [1.0, 1.0, 0.0, 0.0]
    p = measures.participant_measures(m.assign(cost=0.0), pd.DataFrame(
        {"participant_id": ["a"], "condition": ["high"], "perceived_reliability": [70]}))
    assert p.loc[0, "rair"] == 0.5 and p.loc[0, "rsr"] == 0.5


def test_delta_is_the_policy_delta(effect):
    assert effect["m"]["delta"] == 60


# ------------------------------------------------------------------ recovery
def _ok(r, sign, truth=None):
    assert np.sign(r["estimate"]) == sign
    assert (r["ci_low"] > 0) if sign > 0 else (r["ci_high"] < 0)
    if truth is not None:
        assert r["ci_low"] <= truth <= r["ci_high"]


def test_recovers_planted_h1(effect):
    _ok(effect["conf"]["tests"]["H1"], +1, truth=0.20)


def test_recovers_planted_h2(effect):
    r = effect["conf"]["tests"]["H2"]
    _ok(r, +1, truth=4 * sim.EFFECT["trust_slope"])   # +0.3*2 - (-0.3*2)
    assert r["p"] < 0.05


def test_recovers_planted_h3(effect):
    _ok(effect["conf"]["tests"]["H3"], -1)
    # the registered model plus ai_correct recovers the planted -0.15 drop
    _ok(effect["conf"]["tests"]["H3"]["sensitivity_with_ai_correct"], -1, truth=-0.15)


def test_planted_effects_supported_after_holm(effect):
    t = effect["conf"]["tests"]
    assert t["H1"]["supported"] and t["H2"]["supported"] and t["H3"]["supported"]


def test_null_dataset_finds_nothing(null):
    t = null["conf"]["tests"]
    for k in ("H1", "H2", "H3"):
        assert not t[k]["supported"], k
        assert t[k]["ci_low"] <= 0 <= t[k]["ci_high"], k


def test_holm():
    adj = confirmatory.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_glmm_recovers_known_logit():
    rng = np.random.default_rng(5)
    g = np.repeat(np.arange(60), 20)
    x = rng.normal(size=g.size)
    u = rng.normal(0, 0.8, 60)[g]
    y = rng.random(g.size) < 1 / (1 + np.exp(-(-0.3 + 0.9 * x + u)))
    fit = confirmatory.glmm_logit_ghq(y, np.c_[np.ones_like(x), x], g)
    lo = fit["beta"] - 1.96 * fit["se"]
    hi = fit["beta"] + 1.96 * fit["se"]
    assert lo[1] <= 0.9 <= hi[1] and lo[0] <= -0.3 <= hi[0]
    assert 0.4 < fit["sigma"] < 1.3
