# Review — Ntakolia, Kokkotis, Karlsson & Moustakidis (2021), *Sensors* 21(23), 7926

Reviewed from `lightgbm-pivot/papers/01_Ntakolia2021_Sensors_backorder_ANCHOR.pdf` (12 pages, all read; pp. 3, 5–9 also inspected as page images).

<scratchpad>

**Article type.** Labelled "Communication" on p. 1 (not a full Article). Received 1 Nov 2021, accepted 24 Nov 2021, published 27 Nov 2021 — 23 days from submission to publication.

**Research question (no explicit RQ is stated; the closest is the aim, p. 2 §1, verbatim):**
> "this study focused on developing an explainable machine learning pipeline for: (i) evaluating the performance of well-known machine learning models to predict backorders as a binary classification problem and (ii) interpreting the results by using a post-hoc explainability model (SHAP) on the best performing model."

**Gap claimed (p. 2 §1, verbatim):**
> "The above studies employed machine learning methods to address the backorder prediction problem, whether a product would be backordered or not. However, none of these aimed to explain and interpret the impact of features on the prediction output."

**Hypotheses.** None stated. No statistical test is performed anywhere in the paper.

**Main claims, verbatim:**
1. Abstract: "The results showed that the RF, XGB, LGBM, and BB models reached an AUC score of 0.95, while the best-performing model was the LGBM model after calibration with the Isotonic Regression method."
2. Abstract: "The explainability analysis showed that the inventory stock of a product, the volume of products that can be delivered, the imminent demand (sales), and the accurate prediction of the future demand can significantly contribute to the correct prediction of backorders."
3. p. 9 §4: "the LGBM classifier calibrated with the Isotonic Regression method presented the best overall performance, as it is asymptotically closer to the dotted line that represents the perfectly calibrated model (Figures 4 and 5)."
4. p. 9 §4: "a product with low stock and high short-term and mid-term future demand will probably be backordered".

**Every figure and table, with number and caption:**
- Table 1 (p. 3) "Dataset description."
- Table 2 (p. 4) "Hyperparameters of the selected ML models."
- Table 3 (p. 5) "Best metric scores of each ML model and the selected hyperparameters of each model."
- Figure 1 (p. 3) "Methodology steps."
- Figure 2 (p. 6) "Roc curves of the competitive ML models."
- Figure 3 (p. 6) "Calibration results for each classifier with comparative performance based on AUC results."
- Figure 4 (p. 7) "Best calibrated models for each classifier."
- Figure 5 (p. 7) "Roc curve of the LGBM + Isotonic model."
- Figure 6 (p. 8) "SHAP summary plot of LightGBM calibrated with the Isotonic Regression method."
- Figure 7 (p. 8) "Beeswarm plot of LightGBM calibrated with the Isotonic Regression method for the backordered class."
- Figure 8 (p. 8) "Product correctly classified as non-backordered."
- Figure 9 (p. 9) "Product correctly classified as backordered."

**Numbers appearing more than once, checked against each other:**
- AUC 0.95: abstract, Fig. 2 legend (RF/XGB/BB/LGBM), p. 9 §4, p. 10 §5, Fig. 5 legend. Consistent.
- Accuracy 88.85 %: Table 3 (BB), p. 9 §4 "accuracy up to 88.85%". Consistent.
- Accuracy 75.93 %: Table 3 (KNN), p. 9 §4 "lower accuracy (up to 75.93%)". Consistent.
- Recall 90.69 %, F1 89.12 %, precision 88.10 %: Table 3 (BB, BB, RF respectively), repeated p. 9 §4. Consistent.
- 87.78 % (LGBM accuracy): Table 3 only.
- "17 out of 23 features": §3.1 bullet list (17 items counted) and p. 9 §4. Consistent internally.
- Every Table 3 metric re-derived from its own confusion matrix and matched to 2 d.p. (all eight rows; each matrix sums to 5,588 = 2,774 negatives + 2,814 positives). Example: LightGBM (2400+2505)/5588 = 87.78 %; 2505/2814 = 89.02 %; 2505/2879 = 87.01 %. **Table 3 is internally consistent.**

**Inconsistencies found:**
1. **Figures 8 and 9 are contradicted by the text (pp. 8–9).** Fig. 8 caption = "Product correctly classified as non-backordered", but the text says its features "led to the prediction value of 0.35, which was transformed to 1" — i.e. backordered. Fig. 9 caption = "Product correctly classified as backordered", but the text for it ends "pushed the output prediction to the non-backordered class." The captions and the narrative are swapped relative to each other; the paper does not say which is right. Fig. 9's force plot shows f(x)=1.84 with `national_inv = 0`, consistent with the *backordered* caption, so the §4 text appears to be the error.
2. **BoostARoota's stated behaviour vs. its result.** §2.2: "BoostARoota returns an optimal subset of features by eliminating **up to 10%** of the initial set of features." §3.1/§4: it eliminated 6 of 23 = **26 %**.
3. **"Best-performing" is used in two different senses.** The abstract and §5 call LGBM+Isotonic "the best-performing model", but in Table 3 LightGBM has the **lowest** accuracy (87.78 %), recall (89.02 %), F1 (88.00 %) and precision (87.01 %) of the four AUC-0.95 models. Its selection rests only on calibration-plot proximity (Fig. 4).
4. **Split arithmetic.** §2.2 states "A 70/30% validation strategy". With the stated 9,714 backordered items, 1:1 undersampling and a 30 % test share gives 5,828 test rows; Table 3's matrices sum to 5,588. The gap is presumably the rows deleted for missing values, but the paper never reports how many rows were dropped, so the split cannot be verified.
5. **Calibration is asserted, never measured.** No Brier score, ECE, calibration slope or intercept is reported anywhere. Fig. 4's four curves overlap; "asymptotically closer to the dotted line" (§4) is not verifiable from the figure.
6. **Table 1 types.** `perf x months avg` is listed "Categorical", but Fig. 8 shows `perf_6_month_avg = 0.97`, a continuous value, and it is fed to numeric models.
7. Typos affecting readability: "2.2. Mehtodology"; "Balanced Blagging" (§2.2, Table 2 and Table 3 use "BB"); Table 2 row label "RL" for LR; Table 2 LR penalty "[11, 12]" for [l1, l2]; Table 1 "the net 3, 6, 9 months" for "next"; §4 "loval_bo_qty", "n_bank" (Table 1 has `min bank`). Ref. [38] gives a 2000 conference with a 2017 publisher; ref. [41] dates LightGBM/NeurIPS vol. 30 to December 2021 (it is 2017).

</scratchpad>

<analysis>

## 1 SNAPSHOT

- **Citation.** Ntakolia, C., Kokkotis, C., Karlsson, P., Moustakidis, S. "An Explainable Machine Learning Model for Material Backorder Prediction in Inventory Management." *Sensors* 21(23), 7926 (2021). https://doi.org/10.3390/s21237926. Open access, CC BY 4.0.
- **Type.** MDPI "Communication" (p. 1), 12 pages including 49 references. Not a full research article.
- **Problem.** Binary prediction of whether a product will go on backorder, plus post-hoc explanation of which features drive the prediction.
- **Data.** Public Kaggle "Predict Product Backorders" set: 23 features; "9714 products that were backordered and 1,038,860 that were not" (§2.1).
- **Method.** Drop rows with missing values → min–max normalise to [0,1] → majority-class undersampling to 1:1 → BoostARoota feature selection (17/23 kept) → 70/30 split with grid-search CV → 8 classifiers compared → isotonic/sigmoid calibration of the top four → SHAP on the winner.
- **Headline result.** RF, XGB, LGBM and BB all reach AUC 0.95 (Fig. 2); accuracies 87.78–88.85 % (Table 3). LGBM + isotonic regression is declared best overall on calibration (Figs. 4–5).
- **Funding / conflicts.** "This research received no external funding"; "The authors declare no conflict of interest" (pp. 10–11). Two of four authors are at a private company (AIDEAS OÜ).

## 2 ABSTRACT

- **What it promises.** A comparison of ML models for backorder prediction, model calibration, and SHAP explainability to "identify and interpret the most important features".
- **What it commits to numerically.** Exactly two things: "the RF, XGB, LGBM, and BB models reached an AUC score of 0.95" and "the best-performing model was the LGBM model after calibration with the Isotonic Regression method."
- **Accuracy of the abstract against the body.** The AUC claim is supported (Fig. 2). The "best-performing" claim is **misleading as phrased**: on every metric reported in Table 3, LightGBM is the worst of the four tied models. The body (§4) makes clear that "best" means best-calibrated, but the abstract does not.
- **Framing claim not supported by the study.** "traditional approaches in the literature are based on stochastic approximation, without incorporating information from historical data" — the paper's own §1 then cites eight prior ML backorder studies [1,2,29–31], so the framing overstates the gap. The real gap claimed is explainability, not ML use.
- [Assessment] The abstract is a fair summary of what was done, but its one comparative superlative does not survive contact with Table 3.

## 3 INTRODUCTION AND MOTIVATION

- **Motivation.** Backorders drive holding, ordering and backorder costs; "The ability of a company to address backorders impacts significantly the company's revenue, share market price, and customers' trust [2]" (p. 2).
- **Topical hook.** COVID-19 demand surges and the bullwhip/Forrester effect (p. 1). Illustrative only; no COVID data is used.
- **Positioning.** Two paragraphs of EOQ/EPQ analytical inventory work [3–28], then one paragraph of ML backorder work [1,2,29–31].
- **Stated gap (verbatim, p. 2).** "However, none of these aimed to explain and interpret the impact of features on the prediction output."
- [Assessment] The gap is narrow but real for 2021 and honestly stated. The motivation is generic; no industrial partner, cost model or decision context is given, so the paper never connects a prediction to an action. For our purposes that absence is itself the finding: this is a prediction-accuracy paper with no decision-maker in it.

## 4 LITERATURE REVIEW / RELATED WORK

- **Structure.** There is no separate related-work section; it is compressed into §1 (pp. 1–2).
- **Coverage of analytical inventory models** [3–28] is broad — EOQ/EPQ with backorders, safety stock, fuzzy demand, heuristics, carbon-constrained variants — but it is a list, not a synthesis, and none of it is used again.
- **Coverage of ML backorder work** is thin and uncritical: Hajek & Abedin's profit-maximising threshold [30], Islam & Amin's DRF/GBM [2], a deep autoencoder [29], a deep NN for imbalanced data [1], and an MSc/PhD thesis on Danish craft breweries [31]. No numbers from any of them are reported, so the paper never states what performance it is trying to beat.
- **Not cited.** de Santis et al. (2017), the standard earlier baseline on the same Kaggle set; any human-factors, trust or reliance literature; any calibration methodology beyond the two techniques it applies [42–46].
- [Assessment] Adequate for a Communication, insufficient as a benchmark: because no prior result is quoted, the paper cannot and does not claim to advance the state of the art. Anyone (including us) using it as a benchmark is supplying a comparison the authors did not make.

## 5 METHODOLOGY

**Pipeline (§2.2, Fig. 1):** preprocessing → BoostARoota feature selection → 8-model comparison → calibration → SHAP.

- **Preprocessing (verbatim).** "we deleted the rows with missing values, so that we had the maximum possible real information. In addition, we normalized the data set to [0,1]. Finally, to address the problem of imbalanced data, we reduced the samples of the majority class to reach the number of samples in the minority class."
  - The number of rows deleted is **Not stated in paper**.
  - The [0,1] normalisation is described as applied to "the data set", i.e. before the 70/30 split. That leaks test-set minima and maxima into the training transform. Minor in effect, but it is a leak.
  - Undersampling is applied **before** the split, so the *test* set is balanced 1:1 as well (Table 3 matrices: 2,774 negatives vs 2,814 positives). The deployment prevalence is ~0.93 %.
- **Feature selection.** BoostARoota, an XGBoost wrapper in the Boruta family, with "A 10-fold cross validation ... for the selection of the most important features" (§2.2). Whether that CV was run inside the training fold only is **Not stated in paper**; selection is described in §2.2 before the split is introduced, which suggests it was run on all data.
- **Models.** RF, KNN, NN(MLP), LR, SVM, XGB, LightGBM, Balanced Bagging. Hyperparameter grids in Table 2. "A 70/30% validation strategy was employed to generate the training and testing sets with an integrated cross validation strategy that employed grid search for the hyperparameter tunning" (§2.2). Whether model selection used the 30 % test set is **Not stated in paper**; the phrasing is compatible with either reading.
- **Calibration.** Platt scaling and isotonic regression [44–46], applied to RF, XGB, LGBM and BB. Whether a separate calibration set was held out, or `CalibratedClassifierCV` was used with internal CV, is **Not stated in paper** — though Fig. 5's legend reads "CalibratedClassifierCV (AUC = 0.95)", which identifies scikit-learn's cross-validated wrapper.
- **Explainability.** SHAP [47–49] on the calibrated LGBM. Which SHAP estimator was used on a calibrated (hence non-tree) pipeline is **Not stated in paper**.
- **Metrics.** "accuracy, recall, f1-score, precision, AUC" (§2.2). No metric is chosen as primary, and no uncertainty estimate, repeat run, seed or significance test appears anywhere.

- [Assessment] The pipeline is conventional and legible, and BoostARoota + calibration + SHAP is a sensible ordering. Three design choices limit what the numbers mean: (i) balancing before splitting, so all threshold metrics are reported at a prevalence the deployment never sees; (ii) selection and normalisation apparently performed on the full dataset; (iii) a single split with no repeats, so none of the 1-point differences in Table 3 can be distinguished from split noise.

## 6 RESULTS

- **Feature selection (§3.1).** 17 features retained: the four inventory/transit columns, the three forecast and four sales columns, `min bank`, `pieces past due`, the two `perf_*_avg` columns, `local bo qty`, `deck risk`, `ppap risk`. Dropped (inferred by difference, not stated): `potential issue`, `oe constraint`, `stop auto buy`, `rev stop`.
- **Classification (§3.2.1, Table 3, Fig. 2).** AUC: LR 0.79, KNN 0.82, SVC 0.83, MLP 0.92, and RF = XGB = BB = LGBM = 0.95. Accuracy: BB 88.85 %, RF 88.82 %, XGB 88.53 %, LGBM 87.78 %, MLP 85.68 %, KNN 75.93 %, SVM 72.39 %, LR 70.22 %.
- **Calibration (§3.2.2, Figs. 3–5).** Isotonic helps RF, XGB and LGBM; sigmoid helps BB. LGBM+Isotonic is declared best. Its ROC AUC is again 0.95 (Fig. 5) — i.e. calibration changed nothing measurable about discrimination, which is expected and is not commented on.
- **Explainability (§3.3, Figs. 6–9).** Global ranking: `national_inv` ≫ `in_transit_qty` > `forecast_3_month` > `sales_1_month` > `forecast_6_month`. `national_inv` has roughly four times the mean |SHAP| of the second feature (Fig. 6).
- **What the evidence actually supports.** (a) Four tree/ensemble families are indistinguishable on this task at this sample size; (b) linear and distance-based models are clearly worse; (c) the dominant signal is current inventory level; (d) isotonic calibration visually improves the reliability diagram. That is all.
- **What it does not support.** Any ranking *within* the four tied models, in either direction — no repeats, no intervals, no test. The single reported margin (88.85 % vs 87.78 % = 1.07 pt on 5,588 rows, ≈0.9 pt standard error) is inside noise.

## 7 FIGURES AND TABLES (each one, in order)

- **Table 1 (p. 3) "Dataset description."** 23 features with one-line descriptions and a type. Useful as a data dictionary. Error: `perf x months avg` is typed "Categorical" but is continuous (Fig. 8 shows 0.97). `X17–X22` is given as one row covering six flags, so the row count does not add to 23 by inspection.
- **Table 2 (p. 4) "Hyperparameters of the selected ML models."** Search grids only — not the selected values (those are in Table 3). Contains the "RL" and "[11, 12]" typos. LightGBM's grid tunes only `n_estimators` and `num_leaves`; learning rate, `min_child_samples` and regularisation are untuned, which plausibly explains why LGBM sits last of the four.
- **Table 3 (p. 5) "Best metric scores of each ML model and the selected hyperparameters of each model."** The paper's core evidence. Eight rows × (accuracy, recall, F1, precision, 2×2 confusion matrix, chosen hyperparameters). **I re-derived all four metrics from each confusion matrix; every one matches to 2 d.p.** All matrices sum to 5,588 (2,774 / 2,814). No AUC column — AUC lives only in Fig. 2. No standard deviations, no repeats.
- **Figure 1 (p. 3) "Methodology steps."** A four-box flow diagram. Adds nothing beyond §2.2.
- **Figure 2 (p. 6) "Roc curves of the competitive ML models."** Legible. Eight curves with AUCs in the legend: LR 0.79, RF 0.95, KNN 0.82, SVC 0.83, XGB 0.95, BB 0.95, MLP 0.92, LGBM 0.95. This is the only place AUC is reported per model, and only to 2 d.p., so the four tied models cannot be separated.
- **Figure 3 (p. 6) "Calibration results for each classifier..."** Four stacked reliability diagrams (RF, XGB, BB, LGBM), each with the raw model, +isotonic, +sigmoid and the diagonal. Legible but small; the three curves in each panel are near-coincident, and deviations from the diagonal are ~0.05 at most. No numeric calibration measure accompanies them.
- **Figure 4 (p. 7) "Best calibrated models for each classifier."** The four best-calibrated variants on one axis. All four track the diagonal closely; XGB+Isotonic visibly deviates around p≈0.55. Choosing LGBM+Isotonic over RF+Isotonic from this plot is a judgement call, not a measurement.
- **Figure 5 (p. 7) "Roc curve of the LGBM + Isotonic model."** Single curve, "CalibratedClassifierCV (AUC = 0.95)". Identical to the pre-calibration value, so the figure carries no new information.
- **Figure 6 (p. 8) "SHAP summary plot of LightGBM calibrated with the Isotonic Regression method."** Stacked mean |SHAP| bars over 17 features, split Class 0 / Class 1. `national_inv` ≈ 5.0 total, `in_transit_qty` ≈ 1.2, then a long tail; `ppap_risk` and `deck_risk` ≈ 0. Legible. The axis is the raw model-output scale, unlabelled as to units.
- **Figure 7 (p. 8) "Beeswarm plot ... for the backordered class."** Legible. Reads as: low `national_inv` (blue) pushes strongly positive (up to +3.0); high `national_inv` pushes negative. Low `in_transit_qty` pushes positive. Higher `forecast_3_month`/`sales_1_month` push positive. This is the direct evidence for the managerial claim in §4.
- **Figure 8 (p. 8) "Product correctly classified as non-backordered."** A SHAP force plot, f(x) = 0.35, base value −0.09351, with `min bank = 1`, `perf_6_month_avg = 0.97`, `in_transit_qty = 0`, `national_inv = 18` pushing higher. **Caption contradicts the §4 text**, which says these features "led to the prediction value of 0.35, which was transformed to 1" (i.e. backordered).
- **Figure 9 (p. 9) "Product correctly classified as backordered."** Force plot, f(x) = 1.84, with `national_inv = 0`, `in_transit_qty = 0` and `forecast_3_month = 9`, `sales_1_month = 1`. The figure supports the caption; the §4 text describing it ("pushed the output prediction to the non-backordered class") does not.

## 8 DISCUSSION

- **What §4 does.** Restates §3 in prose, then adds one managerial reading and one three-point recommendation list.
- **Managerial reading (p. 9, verbatim).** "a product with low stock and high short-term and mid-term future demand will probably be backordered, since the inventory stock will not be able to satisfy the customers' demand, and at the same time, the expected quantity of products to be delivered to the inventory is also low (Figure 7)." — Supported by Fig. 7.
- **Recommendations (p. 10).** (i) accurate demand forecasts; (ii) raise in-transit quantity and/or cut transit time; (iii) product performance/quality. Items (i) and (ii) follow from Figs. 6–7. Item (iii) does not: the paper's `perf_*_month_avg` features are *source* (supplier) performance per Table 1, not product quality, and they rank 10th–11th in Fig. 6. The causal reading ("if the product's quality satisfies the customers' requirements, the demand of this product is expected to be increased") is unsupported by anything in the paper.
- **Causal overreach in general.** SHAP attributions are associational. The discussion slides into prescriptions ("increase of the products' quantity in transit") that assume intervening on a feature moves the outcome — which this design cannot test.
- [Assessment] The discussion is the weakest section: it re-narrates the numbers, contains the Fig. 8/9 contradiction, and its one novel contribution (the managerial list) is partly not derived from the results.

## 9 LIMITATIONS

- **What the authors state (p. 10, verbatim, the entire limitations discussion):** "A limitation of this study is the use of resampling techniques to cope with imbalanced data. To this end, future work will include the use of Siamese neural networks that have proven effective in case of imbalanced datasets."
- **What that omits.** The stated limitation names undersampling but not its consequence — that the reported accuracy, precision and F1 are measured at 50 % prevalence rather than 0.93 %, and so do not transfer. Precision in particular collapses under real prevalence.
- **Limitations not acknowledged.**
  1. Single 70/30 split, no repeats, no confidence intervals, no significance test — so no claim of one model beating another is supportable.
  2. Normalisation and (apparently) feature selection performed before splitting.
  3. Calibration quality never quantified; the winner is chosen by eye from Fig. 4.
  4. The Kaggle dataset is unlabelled as to time: no temporal split, so the evaluation is random-split, not forward-looking. A backorder predictor is inherently a forecasting task.
  5. `local_bo_qty` ("Amount of stock orders overdue") is retained as a feature. Whether this is measured before or contemporaneously with the backorder label is not discussed; it ranks 7th in Fig. 6.
  6. No cost model, no decision threshold, no comparison to a no-skill or business-rule baseline.

## 10 CONCLUSION AND FUTURE WORK

- **Conclusion (§5).** Restates the pipeline and the two headline findings, and closes on the managerial point: the inventory-stock decision "should be made based on the volume of products that can be delivered, the imminent demand (sales), and the accurate prediction of the future demand."
- **Consistency with results.** Consistent with §3 and §4, and it repeats the same "best-performing" ambiguity as the abstract. It correctly attributes the LGBM selection to calibration ("presented a slightly better calibration for our data") — a weaker and more accurate phrasing than the abstract's.
- **Future work.** One item only: Siamese networks for imbalanced data. No mention of temporal validation, cost-sensitive thresholds, deployment or user studies.
- [Assessment] The conclusion claims no more than §3 shows, apart from the inherited superlative. It is more honest than the abstract.

## 11 REPRODUCIBILITY

- **Data.** Public and named: "https://www.kaggle.com/c/untadta/data" (§2.1 and Data Availability Statement, p. 11). Two different access dates are given (1 October and 23 November 2021).
- **Code.** None. No repository, no supplementary material.
- **Environment.** No library, version or hardware information. Fig. 5's "CalibratedClassifierCV" label implies scikit-learn; Table 2's parameter names imply scikit-learn, XGBoost, LightGBM and imbalanced-learn. Versions: Not stated in paper.
- **Seeds / determinism.** Not stated in paper. Undersampling, the 70/30 split and RF/BB bagging are all stochastic; none is seeded in the text.
- **Reconstructible from the paper?** Partly. The model grids (Table 2), the chosen hyperparameters (Table 3), the feature list (§3.1) and the split ratio (§2.2) are all given — enough to rebuild the pipeline. Not reconstructible: the missing-value deletion rule's effect (row count), whether feature selection/normalisation preceded the split, the calibration protocol, and the seeds. As a result the exact numbers in Table 3 cannot be reproduced, only approached.
- [Assessment] Above average for a 2021 Communication on the data side; below the current bar on everything else. A re-run should be expected to differ by ±1–3 accuracy points from the printed values purely from split and seed variation.

## 12 CRITICAL EVALUATION

**Strengths (max 3)**
1. **Table 3 is internally sound and fully auditable.** Every metric re-derives exactly from its own confusion matrix, and all eight matrices agree on the test-set size. This is rarer than it should be and is what makes the paper usable as a benchmark.
2. **The explainability contribution is real and correctly executed.** Figs. 6–7 are the first SHAP analysis on this problem, they are legible, and the global ranking (`national_inv` dominant, then in-transit and short-horizon demand) is operationally sensible and consistent between the summary and beeswarm views.
3. **Calibration is treated as a first-class concern**, not an afterthought — unusual in this literature, and the right instinct for any system whose output is shown to a person as a probability.

**Weaknesses (max 3)**
1. **No uncertainty anywhere.** One split, no repeats, no intervals, no tests. The paper's central comparative claim — that LGBM+Isotonic is "best" — rests on visual inspection of four near-coincident curves (Fig. 4), and every accuracy difference among the top four models is within split noise.
2. **Balancing before splitting.** Reported accuracy, precision and F1 are measured at 50 % prevalence against a real rate of 0.93 %. The paper names undersampling as a limitation but never states this consequence, and the abstract's numbers carry no such caveat.
3. **The Fig. 8 / Fig. 9 contradiction was not caught.** Captions and the §4 narrative disagree about which example is the backordered one, in the section that is the paper's claimed novel contribution. Combined with the 23-day submission-to-publication window, this indicates light review.

**Credibility: MEDIUM.** The dataset is public, the core result table is internally consistent and re-derivable, and the findings are unsurprising — but the single-split design with no uncertainty estimates, the balanced-test-set inflation, and an uncaught figure/text contradiction in the novel section mean the numbers should be treated as one draw, not as an established value.

## 13 RELEVANCE TO MY FOCUS

### Methodology to reuse

- **The benchmark protocol itself** — public Kaggle data, drop-missing, min–max scale, 1:1 undersample, fixed split, grid-searched trees — is cheap to rebuild and is what our `lightgbm-pivot/backorder_benchmark.py` does. Reuse it as an *external validity* check on our LightGBM, not as a state-of-the-art target.
- **Calibration as a selection criterion.** The paper's instinct (choose the model that is best calibrated, not the one with the highest accuracy) is exactly right for our study, where participants read the risk estimate as a probability. We do what they should have done: report slope and intercept numerically (`Fig04_calibration`, slopes 0.99–1.01) rather than judging by eye.
- **SHAP for conceptual soundness.** Figs. 6–7 are a usable template for our `Fig06_shap`, and their finding (inventory level dominates, then in-transit and short-horizon demand) is a useful convergent check: our live model's top features (net position, cover, inventory, recent demand) are the same operational quantities.
- **A cautionary pattern to avoid.** Do not report threshold metrics on a re-balanced test set without stating the deployment prevalence. Our `goorbergh` citation covers the principle; this paper is a concrete instance of the harm.

### Literature review

- Cite for: backorder/stockout prediction is an established ML task on tabular supply-chain data; boosted trees and bagged ensembles are the incumbent family; explainability arrived late to the problem.
- Cite for the negative point that anchors our argument: the paper evaluates purely on predictive metrics, ends at a SHAP plot and a managerial bullet list, and never involves a planner. That is the gap our study addresses.

### How our draft uses this paper — support check

| # | Where | Use | Verdict | Location in paper |
|---|---|---|---|---|
| 1 | `paper/build_paper.js:156` | "Supply chains increasingly receive AI-generated risk estimates **and order recommendations** {thaker,ntakolia,ivanov}" | **Partially** | The paper produces a backorder risk classification and SHAP explanations only (Abstract; §2.2, Fig. 1). It produces no order recommendation. Fine as one of three co-cited sources for "risk estimates"; it does not support "order recommendations". |
| 2 | `paper/build_paper.js:156` | "Such tools are evaluated almost entirely by predictive accuracy." | **Supported** | §2.2 metric list; Table 3; Figs. 2–5. No user, decision or cost outcome appears anywhere in the paper. |
| 3 | `paper/build_paper.js:166` | "predictive work covers backorder classification {ntakolia}" | **Supported** | Abstract; §1 aim (i); §3.2. |
| 4 | `paper/build_paper.js:207` (Table 1 row) | "Kaggle backorders {ntakolia} … Published 87.78 %; our re-run 90.30 %" | **Supported** | Table 3, LightGBM row (87.78 %). |
| 5 | `paper/build_paper.js:211` | "the printed 87.78 %" | **Supported** | Table 3, LightGBM row. |
| 6 | `paper/build_supplement.js:36` (Fig05 caption) | "(b) The LightGBM matrix published by Ntakolia et al. (2021, Table 3) on a balanced backorder test set." | **Supported** | Table 3, LightGBM confusion matrix [[2400, 374], [309, 2505]]; "balanced" follows from §2.2 undersampling and the 2,774/2,814 test split. The caption's "different data … not a head-to-head comparison" caveat is the right one. |
| 7 | `paper/build_supplement.js:51` (Fig09) + `lightgbm-pivot/OVERVIEW.md:51` | "using **the protocol of** Ntakolia et al. (2021)"; OVERVIEW: "Combine the two Kaggle files, undersample to 1:1, and split **80/20**" | **Partially** | §2.2 states **"A 70/30% validation strategy"**, not 80/20, and §2.1 reports **9,714 backordered and 1,038,860 not** — the training file alone, not the combined files (OVERVIEW cites 1,929,935 SKUs / 13,981 backorders). Our 80/20 reconstruction was inferred from the 5,588-row matrix total, and it fits that total better than 70/30 does — but the paper *says* 70/30. **Action: change "the protocol of" to "a reconstruction of the protocol of", and add one sentence stating that the paper reports a 70/30 split whose row counts we could not reproduce, so we matched the published test-set size instead.** A reviewer who opens §2.2 will otherwise catch this. |
| 8 | `lightgbm-pivot/make_paper_figures.py:491`, `build_supplement.js:64` (Fig12) | "Ntakolia 2021: LightGBM — floor 50.0, reported 87.78, *balanced test set*" | **Supported** | 50 % floor follows from the 1:1 undersampling (§2.2) and Table 3's 2,774/2,814 test composition. |
| 9 | `lightgbm-pivot/OVERVIEW.md:58,70` | "Balanced Bagging (**their best**)" / "Paper, best model (reported) 88.85 %" | **Partially** | Supported on accuracy (Table 3: BB 88.85 % is the highest). **Not** supported as the paper's own designation of "best": the abstract and §5 name **LGBM + Isotonic** as best-performing, on calibration. Recommend "highest published accuracy (Balanced Bagging, 88.85 %)" and a footnote that the authors' own pick was LGBM+Isotonic on calibration grounds. |
| 10 | `lightgbm-pivot/OVERVIEW.md:61` | Logistic regression AUC listed as "—" | **Partially** | The paper does report it: **LR AUC = 0.79** (Fig. 2 legend). Our table understates what is available. Cosmetic. |
| 11 | `lightgbm-pivot/OVERVIEW.md:51` | "Their confusion matrices sum to 5,588 test rows" | **Supported** | Table 3; verified for all eight rows. |
| 12 | `lightgbm-pivot/OVERVIEW.md:49` | "The Kaggle set has 1,929,935 SKUs, of which 13,981 (0.72 %) went on backorder" | **Not found in paper** | §2.1 reports 9,714 backordered / 1,038,860 not (0.93 %). Our figure is from the raw Kaggle files (train + test), which is legitimate — but it must not be presented as *the paper's* dataset. Currently it appears in the section headed "The paper we follow", which invites that reading. |
| 13 | `lightgbm-pivot/OVERVIEW.md:160` | "Benchmark we follow **and beat**" | **Partially** | True only against the printed 87.78 %. OVERVIEW §4 already makes the correct call ("Do not write 'we improved on the paper by 2.9 points'"); the table label contradicts the text two sections below it. Recommend relabelling to "Reproduction target". |
| 14 | `paper/NUMBERS.md:34` | "87.78 %; Ntakolia 2021 Table 3" | **Supported** | Table 3, LightGBM row. |

### One gap worth closing

Our main-text sentence (`build_paper.js:211`) puts "87.78 %" and "90.30 %" side by side with no mention that both are measured on a **1:1 re-balanced** test set. The supplement handles this properly (Fig05's caveat, Fig12's 50 % floor), but the main text does not, and the paper otherwise makes a point of demanding no-skill floors (`build_paper.js:226`). Adding four words — "on a balanced test set" — closes it.

### Critique we can cite against it

If a reviewer presses on why we do not simply adopt the published number: the paper reports a **single split with no repeats and no intervals**, so the 1.07-point spread across its four top models (87.78–88.85 %) is inside split noise, and our five-seed re-run at 90.30 ± 0.42 % is the better estimate of what that pipeline does. This is precisely the "paired reproductions" point in `build_paper.js:226`, and this paper is a clean example of it.

</analysis>
