# Results and where they come from

Every number reported for this study, the file that holds it, and the script that produces it.
All model results carry the corpus fingerprint `f953645593cab9a1`; the comparison scripts refuse to
run on a different corpus.

## 1. The inventory simulation

| Result | Value | Source |
|---|---|---|
| Structure | single echelon (supplier → warehouse → customer), weekly periods, adapted from OR-Gym | `src/simulator.py` |
| Cost per week | holding *h* = 1 per unit, backorder *p* = 5 per unit | `src/simulator.py` (`TwinParams`) |
| Demand | 70 Walmart M5 store-department series × 273 weeks, rescaled to a mean of 140 (SD 36.75) | `data/processed/fitted_params.json` (`m5`), `data/processed/m5_weekly.npy` |
| Lead time | P(*L* = 1 + *k*) = 0.52, 0.29, 0.11, 0.08 for *k* = 0–3, fitted to 96,164 Olist deliveries | `fitted_params.json` (`olist`), `src/fit_params.py` |
| Disruption rate | 7.8 % of weeks, from DataCo severe delivery slips | `fitted_params.json` (`dataco`) |
| Session | 5 practice + 30 main weeks; six fixed disruptions at weeks 6, 9, 13, 17, 21, 26 (counted from 0) | `src/interface/app.py`, `src/simulator.py` (`DEFAULT_CALENDAR`) |

## 2. The reference ordering policy

| Result | Value | Source |
|---|---|---|
| Rule | order up to *S* = μ E[*L*+1] + *z* √(E[*L*+1] σ² + μ² Var(*L*)) | `src/policy.py` |
| Safety factor | *z* = 1.60 by grid search on replayed demand; the newsvendor value 0.967 costs 11.2 % more | `src/calibrate_real.py`, `data/processed/calibration_real.json` |
| Tolerance | an order within ±60 units of the reference keeps expected cost within 5 % | `src/policy.py` (`DELTA_CALIBRATED`), `calibration_real.json` |

## 3. Simulation checked against inventory theory

`src/benchmark_base_stock.py` → `results/twin_benchmark.json`. Reference policy, disruptions off,
20 seeds × 1,950 weeks (39,000 weeks).

| Case (*z* = 1.60) | Simulated | Closed form |
|---|---|---|
| Fixed lead time: weekly stockout rate | 6.5 % (95 % CI 6.2–6.8) | 6.3 % (exact, negative-binomial demand) |
| Fixed lead time: weekly cost | 93.2 | 93.6 |
| Fitted lead times: weekly stockout rate | 3.6 % | 5.5 % (normal approximation behind the order-up-to formula) |

With a fixed lead time the theory is exact and the simulation matches it; with random lead times the
textbook approximation overstates risk, which supports calibrating *z* by simulation (*z* = 1.60 was set before this benchmark was run).

## 4. The live risk model

`src/live_model.py` → `models/live_risk_lgbm.txt`, `models/live_risk_meta.json`.

| Result | Value | Source |
|---|---|---|
| Model | LightGBM, 194 trees, 31 leaves, learning rate 0.03, binary log-loss; 30 inputs (state + 4 weeks of history) | `live_risk_meta.json` |
| Target | stockout in either of the next two weeks | `src/make_corpus.py` |
| Calibration | Platt scaling, slope 0.965, intercept −0.096, fitted on ten held-out series | `live_risk_meta.json` (`platt`) |
| Training data | 48,960 label-valid weeks from 68 series: 42,720 to fit, 6,240 to calibrate; the two series participants replay are excluded | `src/live_model.py` |
| Cross-validated performance | AUROC 0.897, ECE 0.006 (five folds split by series) | `live_risk_meta.json` (`cv`) |
| First five weeks of a session | AUROC 0.833 (lag features still filling) | `live_risk_meta.json` (`cv`) |
| Participants' two demand series | AUROC 0.902 on the corpus episodes (n = 960); on 11,520 fresh weeks AUROC 0.888, ECE 0.008, calibration slope 1.00 | `live_risk_meta.json` (`session_series_check`); `lightgbm-pivot/robustness_results.json` (`C_stress_frozen_model.Baseline`) |
| Display | "Estimated chance of running out in the next 2 weeks: X %", banded low (< 25 %), moderate, high (≥ 55 %) | `src/interface/templates/task.html`, `src/interface/advice.py` |

## 5. Model comparison

`lightgbm-pivot/epoch_study.py` → `lightgbm-pivot/epoch_study.json`. Ten architectures on identical
weeks: 24,960 out-of-fold weeks, five folds split by demand series, paired series-clustered bootstrap.

| Model | AUROC | PR-AUC | Difference vs LightGBM [95 % CI] |
|---|---|---|---|
| Logistic regression | 0.893 | 0.748 | +0.002 [−0.000, +0.005] |
| XGBoost | 0.893 | 0.743 | +0.002 [+0.001, +0.003] |
| CNN (1-D) | 0.892 | 0.742 | +0.001 [−0.002, +0.004] |
| **LightGBM** | **0.891** | **0.741** | — |
| GRU, LSTM, two Transformers, MLP | 0.882–0.887 | 0.729–0.737 | −0.004 to −0.009 |
| PatchTST-style Transformer | 0.857 | 0.679 | −0.034 [−0.041, −0.028] |

- The three leading models lie within 0.002 AUROC of one another; in every fold they stay within 0.005,
  while each model's AUROC swings by 0.04 across folds (`lightgbm-pivot/chart_data.json`, `fold_auroc`).
- Net inventory position alone reaches AUROC 0.869; the positive rate is 21.5 % (`chart_data.json`).
- Accuracy 86.2 % against a 78.5 % majority-class floor (`lightgbm-pivot/simulator_results.json`).

## 6. External benchmarks

| Benchmark | Our result | Published | Source |
|---|---|---|---|
| Walmart M5 (WRMSSE, private leaderboard; a separate per-store forecaster, not the live risk model) | 0.569, a score between the 38th and 39th of 5,507 teams (a score placement, not a competition rank) | 38 of the top 50 scored better; 15 % better than the best statistical benchmark (ES_bu) | `lightgbm-pivot/m5_results.json` |
| Kaggle backorders (Ntakolia et al., 2021 protocol, balanced test set) | 90.3 % accuracy with the published settings (90.65 % is the best of five variants scored on the same splits) | 88.85 % best, 87.78 % LightGBM | `lightgbm-pivot/backorder_benchmark.json` |
| DataCo delay prioritisation (Li, 2026) | Capture@10 % 0.322 | 0.323 | `lightgbm-pivot/li2026_results.json` |

## 7. Robustness and decision value

`lightgbm-pivot/robustness_study.py` → `lightgbm-pivot/robustness_results.json`.

- **Later demand weeks:** AUROC 0.871 and calibration intercept +0.52, against +0.01 under the study's
  own conditions: discrimination holds, calibration needs refitting under shift.
- **Decision value:** an ordering rule acting on the risk estimate costs 464 less over 52 weeks than the
  same rule acting on net inventory position alone (95 % CI 331–583; 150 paired simulations).

## 8. Error placement

`src/placement_design.py` → `results/placement_design.json`.

| Result | Value |
|---|---|
| Propagating error weeks | 5, 7, 8, 9, 14, 15, 16, 19 |
| Absorbed error weeks | 2, 3, 10, 11, 12, 20, 22, 25 |
| Selection | weeks ranked on the extra cost and extra stockout weeks of one accepted error, on lead-time draws 0–99 |
| Validation (draws 100–299) | one propagating error adds 134 cost units against −62 for an absorbed one, and 3.45× the extra stockout weeks for a reference-policy follower (1.78× for a noisy follower) |
| Spread across weeks | week 16 adds 736; weeks 5 and 14 add under 10 |
| Recommendations | correct: within 30 units of the reference order; error: 90–120 units below it, never under 10 (`src/interface/advice.py`) |

## 9. Persistence of a single error

`src/propagation_recovery.py` → `results/propagation_recovery.json`. After one 150-unit short order,
later weeks whose cost differs from the clean run: 1.76 for a base-stock follower, 1.79 for a noisy
follower, 4.47 for a follower who keeps ordering 15 % below the reference.

## 10. Error detectability

`src/leak_check.py` → `results/leak_check.json`. The recommended order alone identifies scripted
errors at AUROC 0.883; adding the displayed risk raises this to 0.892; an observer given the
warehouse state as well separates them almost perfectly (AUROC 1.00).

## 11. Verification of the participant app

- 154 automated tests (`python -m pytest src/tests`): 131 run on SQLite, 23 more against Postgres.
- `src/interface/verify_log.py` replays a stored session and recomputes every displayed risk estimate.
- `python src/interface/app.py --dry-run --db demo.sqlite` runs a scripted 35-week session end to end.

## Reproducing

```bash
pip install -r requirements.txt
python src/make_corpus.py                 # simulation corpus (fingerprint f953645593cab9a1)
python src/live_model.py                  # live risk model and its metadata
python src/placement_design.py            # error-placement weeks and validation
python src/propagation_recovery.py        # persistence of a single error
python src/benchmark_base_stock.py        # simulation vs inventory theory
python src/leak_check.py                  # error detectability
python -m pytest src/tests                # test suite
```

The raw datasets (Walmart M5, Olist, DataCo) are not redistributed; `data/README.md` lists their sources.
