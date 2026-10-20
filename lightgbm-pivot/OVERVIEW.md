# Model pivot: PatchTST out, LightGBM in

*24 September 2026. Everything below was produced by `backorder_benchmark.py`
in this folder and can be regenerated.*

---

## Update, 25 September 2026: read this first

Everything below §1 was written on 24 September, after the backorder
benchmark. Since then:

| Result | Where |
|---|---|
| **Walmart M5**, official private leaderboard protocol: WRMSSE **0.5691 = equivalent to 39th of 5,507 teams** (15.2 % better than the best statistical benchmark; winner 22.4 %). Scorer reproduces two published benchmark scores exactly | `m5_lightgbm.py`, `m5_results.json` |
| **DataCo / Olist**, replicating Li (2026, arXiv:2607.18573): DataCo matches the paper (0.322 vs 0.323); Olist beats it (0.476 vs 0.415), but our XGBoost does better (0.505) | `li2026_replication.py` |
| **XGBoost on the GPU** on the backorder benchmark: 89.19 % / 89.63 %, below LightGBM (90.65 %) | `xgb_gpu_benchmark.py` |
| **Simulator**, 70 series: LightGBM AUROC 0.891; ties LR and XGBoost; EasyEnsemble does **not** help there | `simulator_training.py` |
| **Live model in the app**: LightGBM on current state + 4 weeks of history; AUROC **0.897** (cross-validation), 0.902 on the unseen session demand. Well calibrated throughout (ECE 0.006 overall, 0.004 on weeks 0–4; the 0.022 held-out figure is small-sample noise, RESULTS §18). Discrimination is genuinely lower before the lags fill: AUROC 0.833 on weeks 0–4, and **not fixable** by a history-depth feature or early recalibration | `src/live_model.py`, `models/` |
| **10 models, 50 vs 100 epochs**: LightGBM ranks 4th of 10, statistically tied with the top three and significantly better than six deep models (PatchTST −0.034). Neural models overfit with more epochs | `epoch_study.py`, `epoch_study.json` |

**Final model for the study:** plain LightGBM plus Platt calibration (the
EasyEnsemble gain is specific to the backorder benchmark). **Claim:** top tier
and externally validated, not "best of all models".

---

## 1. The decision in one paragraph

PatchTST (and its fusion with a second Transformer) is **removed** from the
study. On our simulator data it never beat logistic regression or LightGBM, and
one hand-computed feature (net inventory position) matched it on its own. The
AI component is now a **LightGBM model**, validated against a published,
peer-reviewed benchmark on a public dataset, where it beats the paper's own
reported numbers and beats a faithful re-run of the paper's model on every
seed.

---

## 2. The paper we follow

**Ntakolia, C., Kokkotis, C., Karlsson, P. & Moustakidis, S. (2021).**
*An Explainable Machine Learning Model for Material Backorder Prediction in
Inventory Management.* **Sensors 21(23), 7926.** doi:10.3390/s21237926 —
open access, `papers/01_...ANCHOR.pdf`.

**Why this one:**
- **Same problem as our study.** A backorder is a stockout, which is the risk the AI shows participants.
- **Public data.** The Kaggle *"Can you predict product backorders?"* set has 1,929,935 SKUs, of which 13,981 (0.72 %) went on backorder. It's in `data/raw/backorder/`.
- **Our model family.** The paper compares 8 models, including LightGBM and logistic regression.
- **A protocol we can rebuild.** Combine the two Kaggle files, undersample to 1:1, and split 80/20. Their confusion matrices sum to 5,588 test rows; our split gives 5,593.

**What they report (Table 3, balanced test set):**

| Model | Accuracy | AUC |
|---|---|---|
| LightGBM | 87.78 % | 0.95 |
| Balanced Bagging (their best) | 88.85 % | 0.95 |
| XGBoost | 88.53 % | 0.95 |
| Random Forest | 88.82 % | 0.95 |
| Logistic regression | 70.22 % | — |

---

## 3. Our results — 5 seeds, identical test set within each seed

| Variant | Accuracy | AUROC |
|---|---|---|
| Paper, LightGBM (reported) | 87.78 % | 0.95 |
| Paper, best model (reported) | 88.85 % | 0.95 |
| **A** LightGBM, paper's own settings (our re-run) | 90.30 ± 0.42 % | 0.9582 ± 0.0023 |
| **B** A + inventory-position features | 90.24 ± 0.48 % | 0.9577 ± 0.0027 |
| **C** B stacked with logistic regression | 90.12 ± 0.46 % | 0.9578 ± 0.0026 |
| **D** C trained as an EasyEnsemble | 90.55 ± 0.42 % | 0.9627 ± 0.0019 |
| **E — FINAL: LightGBM EasyEnsemble** | **90.65 ± 0.49 %** | **0.9635 ± 0.0019** |

**Paired gain of E over A (same seed, same test rows):**

| Seed | Δ accuracy | Δ AUROC |
|---|---|---|
| 0 | +0.03 pt | +0.0059 |
| 1 | +0.14 pt | +0.0047 |
| 2 | +0.84 pt | +0.0052 |
| 3 | +0.52 pt | +0.0047 |
| 4 | +0.19 pt | +0.0058 |
| **Mean** | **+0.34 pt** | **+0.0053** |

E beats A on **every seed, on both metrics**.

### What the final model does differently

The paper balances the data by **throwing away ~1.9 million on-time SKUs**.
Model E still tests on exactly the paper's kind of balanced test set, but trains
5 extra LightGBM models, each on the same backorder rows plus a *fresh* sample
of the discarded on-time rows, and averages their probabilities (EasyEnsemble;
Liu, Wu & Zhou 2009). Test rows never enter any bag, and the script asserts this
at runtime.

### What did **not** help — also reported

- **Inventory-position features (B):** no gain. The raw columns already carry them.
- **Stacking with logistic regression (C, D):** no gain. The ablation (D vs E) shows the entire improvement comes from the extra training data, not from combining LR with LightGBM. **So the final model is LightGBM alone.**

---

## 4. How to state the result in the paper

> "Reproducing the protocol of Ntakolia et al. (2021) on the public Kaggle
> backorder dataset, our LightGBM baseline reached 90.30 ± 0.42 % accuracy
> (AUROC 0.958) over five random splits, against 87.78 % reported. Training the
> same model as an EasyEnsemble over the majority-class records that
> undersampling discards raised this to 90.65 ± 0.49 % (AUROC 0.9635 ± 0.0019),
> improving on the reproduced baseline on all five splits."

**Do not** write "we improved on the paper by 2.9 points." Most of that gap is
reproduction noise: the paper's own LightGBM, re-run, already scores 90.3 %.
The defensible improvement is the **paired** one: **+0.34 pt accuracy,
+0.005 AUROC, on 5 of 5 seeds**. A reviewer can check the paired claim; the
headline gap will not survive a re-run.

---

## 5. Why PatchTST was dropped — the evidence

1. **It never won.** On the leak-free simulator corpus, 10 architectures landed within AUROC 0.859–0.888, and PatchTST v2 came **last** (0.859, 5 seeds × 60 epochs).
2. **One feature matches it.** Logistic regression on net inventory position alone scores AUROC **0.859**. The stockout label is decided by the current state plus future randomness, so there is no long-range pattern for PatchTST to find.
3. **The architecture fights the task.** Per-window normalisation (RevIN) removes the absolute inventory level, which is what decides a stockout, and patching blurs the most recent week.
4. **Wrong data regime.** ~15k correlated windows is small-tabular territory, where trees win (Grinsztajn et al. 2022, `papers/05`) and plain linear models rival Transformers (Zeng et al. 2023, `papers/07`).
5. **The reference paper's claim didn't reproduce.** Its PatchTST beat SVM/CNN/LSTM/Transformer; on our data those models all landed within noise of it, and three edged it out.

All PatchTST code, results and figures are in `../archive/patchtst/`. Nothing was deleted.

---

## 6. The paper we rejected, and why

**Ahmed et al. (2025), *Scientific Reports*.** SOM+ANN, 96 % accuracy on DataCo late-delivery prediction (`papers/02`).

DataCo's `Late_delivery_risk` label is simply *real shipping days > scheduled
shipping days*; that rule reproduces 97.6 % of labels. The paper uses the
real-shipping-days column as an input, a value only known **after** delivery.
Checked on our copy of DataCo:

| Input features | LightGBM accuracy | AUROC |
|---|---|---|
| All columns (paper-style) | **100.0 %** | 1.000 |
| Only what is known before dispatch | 81.6 % | 0.900 |

*Corrected 25 Sept 2026: the first version of this table said 77.2 % / 0.859 and had no script. `ahmed_leakage_check.py` reproduces it (5 random stratified splits): all columns 100 %, order-time columns 81.6 % (AUROC 0.900). Results in `ahmed_leakage_results.json`.*

Beating 96 % there is trivial and meaningless. Worth one sentence in the paper,
as a caution about the DataCo benchmark.

---

## 7. Papers folder

| File | Role in the paper |
|---|---|
| `01_Ntakolia2021_Sensors…ANCHOR` | **Benchmark we follow and beat** |
| `02_Ahmed2025_SciRep…REJECTED` | DataCo leakage example (§6) |
| `03_IslamAmin2020_JBigData` | Second backorder study on the same Kaggle data (DRF/GBM); related work |
| `04_Ke2017_NeurIPS_LightGBM` | Citation for the model |
| `05_Grinsztajn2022_NeurIPS` | Why trees beat deep learning at this data scale |
| `06_Hollmann2025_Nature_TabPFN` | Small-data foundation model; possible future comparison |
| `07_Zeng2023_AAAI` | Linear models rival Transformers on time series |
| `08_Nie2023_ICLR_PatchTST` | Original PatchTST, cited for why it was dropped |
| `09_vandenGoorbergh2022_JAMIA` | Imbalance corrections distort probabilities. **Relevant because participants see risk %: calibrate E before display** |
| `10_Chen2016_KDD_XGBoost` | Gradient-boosting background |

Not downloaded (paywalled, cite only): Liu, Wu & Zhou (2009), *Exploratory
Undersampling for Class-Imbalance Learning*, IEEE TSMC-B 39(2), 539–550, the
EasyEnsemble method used in model E.

---

## 8. Folder layout and how to re-run

```
lightgbm-pivot/
  OVERVIEW.md               <- this file
  backorder_benchmark.py    <- variants A-E, 5 seeds
  backorder_benchmark.json  <- all per-seed numbers
  benchmark_run.log         <- console output of the run above
  papers/                   <- 10 PDFs
archive/patchtst/           <- PatchTST code, results, figure (retired)
data/raw/backorder/         <- Kaggle backorder dataset
```

```bash
python lightgbm-pivot/backorder_benchmark.py      # ~10 min on CPU
```

---

## 9. Caveats to disclose

- The paper states no random seed, so its exact split can't be recreated. That's why we report 5 seeds and paired differences.
- We used the paper's *selected* LightGBM hyper-parameters (520 trees, 50 leaves), not its grid search.
- Accuracy is scored at a 0.5 threshold on a balanced test set, as in the paper. On real data, where 0.72 % are positive, accuracy means little; AUROC and calibration matter more.
- Model E's probabilities are averaged over balanced bags and are **not** calibrated to the true base rate. Calibrate before showing any risk % to participants (`papers/09`).

---

## 10. Outcome

The benchmark settled the choice of model: LightGBM is in the top tier, auditable against published
results, and fast enough to run live. It is the model participants see (`src/live_model.py`,
`models/`), and the comparison above is reported in [`docs/RESULTS.md`](../docs/RESULTS.md).
