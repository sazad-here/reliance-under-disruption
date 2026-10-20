# Reliance Under Disruption

Code, models and results for the AHFE 2027 paper *Human–AI Collaboration in Supply Chains: A
Framework for Resilient Decision Making Under Disruption* (Málaga, 26–30 July 2027).

Participants manage a simulated warehouse for 30 weeks. Each week they commit their own order,
then see an AI assistant's suggestion (a scripted recommended order plus a live, calibrated
LightGBM stockout-risk estimate), then enter a final order. The study asks whether reliance on
AI depends on **where** its errors fall (weeks where a bad order propagates vs weeks where it is
absorbed), not only on **how often** it errs.

- **Every number in the paper and where it comes from:** [`RESULTS.md`](RESULTS.md)
- **Paper source:** [`paper/build_merged_paper.js`](paper/build_merged_paper.js) (builds the Word file);
  supplement: [`paper/build_supplement.js`](paper/build_supplement.js)
  (building the paper needs AHFE's official Word template at `paper/AHFE template 2026.docx`, which
  AHFE gives to authors and which is not redistributed here)
- **Live study:** https://reliance-study.onrender.com (data collection, see the note below)

---

## For reviewers: trying the study

**Please do not start a session on the live site.** It is the data-collection instance, so every
session started there creates a study record. Run your own copy instead; it takes a minute and
writes only to a local throwaway database:

```bash
pip install -r requirements.txt
python src/interface/app.py --db demo.sqlite      # then open http://127.0.0.1:8000
```

`--db` points the app at its own SQLite file and ignores any production database URL.

**Answers for the screens you will meet:**

| Screen | What to enter |
|---|---|
| Numeracy screener (three items, 2 of 3 needed to continue; Schwartz et al., 1997) | die rolled 1,000 times, how many even: **500** · lottery with 1 % chance, 1,000 tickets: **10** · 1 in 1,000 chance, what percent: **0.1** |
| Trust questionnaire (six items, 1–7, after each 10-week block) | any rating; there are no right answers |
| Workload questionnaire (NASA-TLX, five subscales, 1–20) | any rating; there are no right answers |
| Perceived reliability (end of session) | any number from 0 to 100 |
| Weekly orders | any whole number; the AI suggestion and risk estimate appear after you lock your own order |

A scripted end-to-end session also runs without a browser:

```bash
python src/interface/app.py --dry-run --db demo.sqlite
```

---

## Status (19 October 2026)

| Area | State |
|---|---|
| Simulation | ✅ Single-echelon inventory simulation replaying real Walmart M5 demand; property-tested; benchmarked against closed-form base-stock theory (`src/benchmark_base_stock.py` → `results/twin_benchmark.json`) |
| AI component | ✅ Live LightGBM risk model (AUROC 0.897; 0.902 on unseen session demand); Platt-calibrated and frozen. Recommended orders are scripted so recommendation accuracy is exact |
| Error design | ✅ All errors are under-orders, placed on weeks chosen by measured consequence and validated on held-out simulations (`src/placement_design.py` → `results/placement_design.json`) |
| Participant app | ✅ Information sheet, numeracy screener, two-stage ordering, questionnaires, debrief, withdrawal, desktop-only gate |
| Tests | ✅ `python -m pytest src/tests` (131 passed, 23 skipped on SQLite, 19 Oct 2026) |
| Analysis scripts | ✅ `src/analysis/run_all.py` (registered models, exclusions, measures); recovers planted effects on simulated participants, none on the null set |
| Deployment | ✅ Live on Render with Neon Postgres: https://reliance-study.onrender.com |
| Pre-registration | ✅ Submitted on OSF (text in `docs/OSF-PREREGISTRATION.md`) |
| Data collection | 🟡 Opened 19 October 2026 |
| Paper | 🟡 AHFE 2026 template, 10 pages; behavioural results follow data collection |

---

## The model in one paragraph

LightGBM predicts P(stockout within 2 weeks) from the warehouse state. On the
simulator it is **top tier but not the outright winner**: AUROC 0.891, tied
with logistic regression (0.893) and XGBoost (0.893), and significantly better
than six deep-learning models including PatchTST (+0.034); neural models got
*worse* from 50 to 100 epochs (overfitting). Externally it is validated on
Walmart M5 (equivalent to **39th of 5,507 teams**), on the Ntakolia et al.
(2021) backorder benchmark (90.65 %, beating a faithful reproduction on 5 of 5
splits), and on DataCo/Olist (Li 2026). Do **not** write that it "outperformed
all models". Details: [`lightgbm-pivot/OVERVIEW.md`](lightgbm-pivot/OVERVIEW.md).

---

## Directory

```
src/                      study code
  simulator.py            warehouse simulator          policy.py      correct order Q*, δ = ±60
  live_model.py           live LightGBM risk model     demand_source.py  M5 demand replay
  placement_design.py     design v2 week selection     propagation_recovery.py  bias → 4.47 weeks
  power_analysis.py       power at 35/cell             leak_check.py  does the screen leak errors?
  interface/              participant web app (Flask)
    app.py                server, sessions, recovery, desktop check, /healthz
    advice.py             reliability schedule, errors, risk display (design v1/v2)
    world.py              shared M5 demand + session order cap
    db.py, schema.sql, schema_postgres.sql    SQLite (local) / Postgres (cloud)
    numeracy.py, withdraw.py, verify_log.py
    templates/            index (consent, screener, desktop gate), task, info sheet
  analysis/               pre-registered analysis: load, exclusions, measures, confirmatory, exploratory, run_all
  tests/                  pytest suite (111 + 23 PG-only) + e2e_desktop_gate.py (browser)
models/                   live_risk_lgbm.txt + live_risk_meta.json
lightgbm-pivot/           model benchmarks, papers/, OVERVIEW.md, report.html
demo/                     run_demo.py (live browser demo; --headed to watch)
  narrated/               make_narrated_demo.py → output/Reliance_Under_Disruption_demo.mp4 (+ transcript.md, captions.srt)
paper/                    build_paper.js, build_supplement.js → the .docx drafts; NUMBERS.md maps every number to its source
figures/paper/            paper figures (PNG + SVG) and CAPTIONS.md
deploy/                   container deploy scripts + README (production runs on Render + Neon; see render.yaml)
.github/workflows/        ci.yml (tests on every push), deploy.yml (manual, approval-gated)
docs/                     OSF pre-registration, ethics amendment, human-factors audit, run plan
data/processed/           fitted parameters, M5 demand cache (needed at runtime)
data/raw/                 M5, Olist, DataCo, Kaggle backorder (not deployed)
results/                  design checks, power analysis, earlier model results
archive/patchtst/         retired PatchTST work (kept for the record)
RESULTS.md                paper-facing record of every result
```

## Common commands

```bash
python -m pytest src/tests -q                          # all tests
python src/interface/app.py --port 8000 --db demo/output/play.sqlite3   # play locally
python demo/run_demo.py --headed                       # watch a scripted session
python demo/run_demo.py --slow                         # watch it at human speed (tour, typing, 3D animation)
python src/interface/app.py --dry-run --db /tmp/d.sqlite3 && \
  python src/interface/verify_log.py --db /tmp/d.sqlite3 --participant DRYRUN-01
python src/live_model.py                               # retrain the live model (only before G0!)
python src/placement_design.py                         # re-validate the error weeks
docker build -t reliance-study:local .                 # production image
python src/analysis/run_all.py                         # analysis on the simulated datasets
python demo/narrated/make_narrated_demo.py             # rebuild the narrated video (~12 min, overwrites in place)
NODE_PATH=<dir with docx installed> node paper/build_paper.js   # rebuild the paper (close it in Word first)
```

## Environment notes

Python 3.13 (use `python -m pip`, not bare `pip`); PyTorch CUDA build for the
archived neural models; LightGBM **4.7.0** must match the version that trained
`models/live_risk_lgbm.txt`. Flask caches templates: restart the server after
editing HTML. Production runs exactly **one** process (sessions live in memory).
