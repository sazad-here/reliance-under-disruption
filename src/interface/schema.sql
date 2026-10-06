-- Trial-level log for the reliance-under-disruption experiment.
-- One row per trial in `trial`; questionnaires land in `block_measure`.
--
-- Every column the analysis needs is written at the moment it is known, in one
-- transaction per trial. Nothing is reconstructed afterwards: with 80
-- participants run in a single batch there is no second attempt at the data.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS participant (
    participant_id   TEXT PRIMARY KEY,
    condition        TEXT NOT NULL CHECK (condition IN ('high', 'low')),
    prolific_pid     TEXT,
    seed             INTEGER NOT NULL,
    numeracy_score   INTEGER,
    started_utc      TEXT NOT NULL,
    finished_utc     TEXT,
    -- post-hoc subjective reliability estimate: the manipulation check
    perceived_reliability INTEGER,
    consented        INTEGER NOT NULL DEFAULT 0,
    excluded         INTEGER NOT NULL DEFAULT 0,
    exclusion_reason TEXT,
    -- withdrawal: set when the participant withdraws, in session or later on
    -- request. data_deleted=1 means every trial, questionnaire and event row
    -- was removed; the bare row stays so arrival-order alternation is stable.
    withdrawn_utc    TEXT,
    data_deleted     INTEGER NOT NULL DEFAULT 0,
    -- sha256 of the random resume token held in the participant's cookie.
    -- Only the hash is stored: a leaked database must not let anyone resume
    -- (and overwrite) a live participant's session.
    token_sha256     TEXT,
    -- the device the session started on (desktop-only gate, app.py)
    device_ua        TEXT,
    screen_w         INTEGER,
    screen_h         INTEGER,
    viewport_w       INTEGER,
    pointer_coarse   INTEGER,
    app_version      TEXT NOT NULL,
    params_json      TEXT NOT NULL      -- the exact TwinParams used, for replay
);

CREATE TABLE IF NOT EXISTS trial (
    participant_id   TEXT NOT NULL REFERENCES participant(participant_id),
    trial_index      INTEGER NOT NULL,  -- 0-based, within phase
    phase            TEXT NOT NULL CHECK (phase IN ('practice', 'main')),
    block            INTEGER NOT NULL,  -- 0,1,2 for main trials; -1 for practice

    -- ---- state vector S_t, exactly as shown to the participant ----
    period           INTEGER NOT NULL,
    demand           INTEGER NOT NULL,
    inventory        INTEGER NOT NULL,
    on_order         INTEGER NOT NULL,
    lead_time        INTEGER NOT NULL,
    backorders       INTEGER NOT NULL,
    supplier_reliability REAL NOT NULL,
    transport_ok     INTEGER NOT NULL,
    disruption       TEXT NOT NULL,

    -- ---- the AI advice ----
    q_ai             INTEGER NOT NULL,
    ai_risk_pct      INTEGER NOT NULL,
    ai_level         TEXT NOT NULL,
    ai_correct       INTEGER NOT NULL,
    placement        TEXT NOT NULL CHECK (placement IN ('absorbed', 'propagating', 'neutral')),  -- neutral: v2 weeks that are never errors

    -- ---- ground truth, never displayed ----
    q_star           INTEGER NOT NULL,
    delta            INTEGER NOT NULL,

    -- ---- the two-stage elicitation ----
    q_own            INTEGER NOT NULL,  -- stage 1, BEFORE advice. RAIR/RSR need this
    q_final          INTEGER NOT NULL,  -- stage 2, after advice
    response_class   TEXT NOT NULL CHECK (response_class IN ('accept','override','modify')),
    woa              REAL,              -- NULL when q_ai == q_own (undefined)
    error_own        INTEGER NOT NULL,  -- |q_own   - q_star|
    error_final      INTEGER NOT NULL,  -- |q_final - q_star|
    own_correct      INTEGER NOT NULL,
    final_correct    INTEGER NOT NULL,

    -- ---- timing, milliseconds, measured client-side ----
    rt_stage1_ms     INTEGER NOT NULL,
    rt_stage2_ms     INTEGER NOT NULL,

    -- ---- realised consequence ----
    -- `demand` above is D_{t-1}: what the participant SAW in the state panel.
    -- `demand_realised` is D_t, what actually arrived after they ordered. The
    -- two are different numbers and the analysis needs both; storing only the
    -- first silently loses the demand every outcome was produced by.
    demand_realised  INTEGER NOT NULL,
    cost             REAL NOT NULL,
    stockout         INTEGER NOT NULL,
    service_level    REAL NOT NULL,
    shipped          INTEGER NOT NULL,
    arrivals         INTEGER NOT NULL,
    inventory_close  INTEGER NOT NULL,
    backorders_close INTEGER NOT NULL,

    submitted_utc    TEXT NOT NULL,
    PRIMARY KEY (participant_id, phase, trial_index)
);

-- Trust (Jian et al. / Korber TiA) and NASA-TLX, administered per block of 10.
-- RQ3 asks how trust develops DURING repeated decisions, so a single post-hoc
-- administration cannot answer it. Three block-level administrations.
CREATE TABLE IF NOT EXISTS block_measure (
    participant_id   TEXT NOT NULL REFERENCES participant(participant_id),
    block            INTEGER NOT NULL,
    instrument       TEXT NOT NULL CHECK (instrument IN ('tia', 'tlx')),
    item             TEXT NOT NULL,
    response         REAL NOT NULL,
    recorded_utc     TEXT NOT NULL,
    PRIMARY KEY (participant_id, block, instrument, item)
);

-- Append-only breadcrumb of every state transition, so a session that crashes
-- mid-way can be audited rather than guessed at.
CREATE TABLE IF NOT EXISTS event_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id   TEXT,
    kind             TEXT NOT NULL,
    detail           TEXT,
    recorded_utc     TEXT NOT NULL
);

-- Numeracy screening outcomes. Deliberately carries NO identifier: people who
-- fail are never participants (no row in `participant`, no condition), and only
-- the count of screen-outs is reported. Passers' scores also land here, and on
-- their participant row as numeracy_score.
CREATE TABLE IF NOT EXISTS screening (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    numeracy_score   INTEGER NOT NULL,
    passed           INTEGER NOT NULL,
    recorded_utc     TEXT NOT NULL
);

-- A stage 1 that has been submitted but whose trial is not yet recorded. At
-- most one per participant. Persisted because Q_own is locked the moment the
-- advice is shown: if the process restarts between the two stages, the
-- participant must come back to the SAME advice with the SAME locked Q_own,
-- not get a fresh stage 1 they could answer knowing the recommendation.
-- Deleted in the same transaction that writes the trial row.
CREATE TABLE IF NOT EXISTS pending_stage1 (
    participant_id   TEXT PRIMARY KEY REFERENCES participant(participant_id),
    phase            TEXT NOT NULL CHECK (phase IN ('practice', 'main')),
    trial_index      INTEGER NOT NULL,
    q_own            INTEGER NOT NULL,
    rt_stage1_ms     INTEGER NOT NULL,
    advice_json      TEXT NOT NULL,     -- advice.Advice.record(): what was shown + ground truth
    recorded_utc     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trial_participant ON trial(participant_id, phase, trial_index);
CREATE INDEX IF NOT EXISTS idx_event_participant ON event_log(participant_id, id);
