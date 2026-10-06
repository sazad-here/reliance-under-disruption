# Reliance Under Disruption

A human–AI reliance experiment for **AHFE 2027** (Málaga, 26–30 July 2027).
Participants manage a simulated warehouse for 30 weeks. Each week they enter
their own order, then see an AI assistant's suggestion (a scripted recommended
order plus a live LightGBM stockout-risk estimate), then enter a final order.
The study tests whether reliance on AI depends on **where** its errors fall
(weeks where a bad order cascades vs weeks where it is absorbed), not only
**how often** it errs.

> **Start here:** this file for orientation → [`RESULTS.md`](RESULTS.md) for
> every number and its source → [`docs/run-plan.html`](docs/run-plan.html)
> (published: https://claude.ai/artifact/FmnABqMSLjVYbhFvBSFS6M) for the
> schedule and gates.

*Last updated 30 September 2026. **Resuming? Read RESULTS.md §16 (newest) and §14 ("Start here next session") first.***

---

## Status

| Area | State |
|---|---|
| Simulator (digital twin) | ✅ Real Walmart M5 demand, shared by all participants; verified by 13 property tests and a 30-check log replay |
| AI component | ✅ Live LightGBM risk model (AUROC 0.897; 0.902 on unseen session demand). Recommended orders are scripted so reliability is exactly 80 % or 60 % |
| Error design | ✅ v2: all errors are under-orders, placed on weeks chosen by measured consequence; passes gate G0 on held-out data. v1 still available (`ADVICE_DESIGN=v1`) |
| Participant app | ✅ Information sheet, numeracy screener, two-stage ordering, questionnaires, debrief, withdrawal, desktop-only gate; UI v2 (dark mode, WCAG 2.2) |
| UI v3 (28 Sept) | 🟡 Built and tested: a 3D warehouse (three.js, vendored), a Tabler-style layout, "?" help and a practice tour. **On by default; `WAREHOUSE_3D=0` switches it off.** Awaiting the supervisor's decision (prereg Q16). The demo video still shows v2 |
| Tests | ✅ 111 passed, 23 skipped on SQLite; 218 passed with `PG_TEST_URL` set (Postgres, incl. restart recovery) |
| Analysis scripts | ✅ `src/analysis/run_all.py` (H1–H4, exclusions, measures); recovers planted effects on simulated participants, none on the null set |
| Paper | 🟡 10 pages (at the AHFE maximum), 28 refs, 3 figures, 3 tables. Model-comparison table and external-validation table in place; behavioural results reserved. **AHFE 2027 allows 6 recommended / 10 maximum** (ahfe.org/submissions.html); no margin left for additions. Supplement (10 pages, 16 figures) may be submitted alongside |
| Demo video | ✅ `demo/narrated/output/Reliance_Under_Disruption_demo.mp4` (10:54, narrated, 23 MB) |
| Deployment | 🟡 Docker image `reliance-study:local` rebuilt 26 Sept from the final code and verified (dry run 31/0/0, risk % identical to local); Firebase/Cloud Run config and CI/CD done; **nothing deployed yet** (needs your GCP project) |
| Supervisor sign-off | ⏳ Design v2, error size, UI v2, sample size (80 → 70), OSF open questions 12–15, new exclusion rule 5 |
| Pre-registration | ⏳ Draft ready (`docs/OSF-PREREGISTRATION.md`); you submit it on OSF |
| Recruitment | 🟡 Changed 28 Sept: **unpaid volunteers from the university's SONA participant pool**, not Prolific. No payment, no completion code. Recruitment target and session length to settle with the supervisor (prereg open question 17) |
| Data collection | ⏳ Planned 19 Oct – 1 Nov 2026 (pilot from 13 Oct) |

### Next steps, in order

All engineering is finished. What remains is yours (details in RESULTS.md §14):

1. Fill the placeholders in `src/interface/templates/info.html` (names, institution, ethics committee and reference, study email, two retention periods, date). Participation is unpaid: the sheet must say so, and mention course credit if your SONA pool gives it.
2. Supervisor sign-off: design v2, sample size (80 recruited → 70 analysed; may need raising for volunteer drop-out), OSF open questions 12–15 and 17, the revised exclusion rule 5 (§8). Disclose the switch to unpaid SONA recruitment to the ethics committee (`docs/ETHICS-AMENDMENT.md` §0).
3. Submit the OSF pre-registration from your own account.
4. Create the Firebase / Google Cloud project (EU region, billing), set `PROJECT_ID` in `deploy/deploy.sh` and `.firebaserc`; then `bash deploy/deploy.sh` ([`deploy/README.md`](deploy/README.md)). Claude then runs the full check against the live URL.
5. Set up the SONA study as an online (external) study pointing at the live URL, with the participant ID passed in the link. The typical SONA pattern is `https://<host>/?id=%SURVEY_CODE%`, where SONA replaces `%SURVEY_CODE%` with each participant's identification code; confirm the exact placeholder in your institution's SONA study settings and that the study's start page reads the same parameter name. There is no completion code; grant credit in SONA if your pool offers it.
6. `git init` + private GitHub repo + secrets from `.github/workflows/deploy.yml` (only when you ask).
7. Pilot 13 Oct (n ≈ 10), collect 19 Oct – 1 Nov (n = 80, `deploy/FREEZE` in place), then `python src/analysis/run_all.py` on the real data and fill the paper's results.
8. Abstract (500 words, target 23 Nov, deadline 4 Dec 2026); camera-ready by 5 Feb 2027.

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
deploy/                   deploy.sh + README (Firebase Hosting → Cloud Run → Cloud SQL)
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
