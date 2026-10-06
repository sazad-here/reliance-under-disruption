# Review — Islam & Amin (2020), *Journal of Big Data* 7:65

Reviewed from `lightgbm-pivot/papers/03_IslamAmin2020_JBigData_backorder_DRF-GBM.pdf` (22 pages, all read; pp. 6, 8, 12–13, 15–19 also inspected as page images).

<scratchpad>

**Research question.** Never stated as a question. The stated aim (Abstract, verbatim):
> "We aim to use machine learning models in the area of the business decision process by predicting products' backorder while providing flexibility to the decision authority, better clarity of the process, and maintaining higher accuracy."

**Stated contributions (p. 5, verbatim, four bullets):**
> "Proposing a ranged based clustering method to reduce the dimensionality of input space and to decrease the computational cost."
> "Implementing the correlational factor among ranged clusters in the input space to increase the performance of the model."
> "Developing a tunable ranged based model for flexible inventory control by incorporating both negative and positive data types."
> "Demonstrating probable backorders scenarios using a decision-based approach where the number of ties for classification is minimized."

**Gap claimed (p. 4, verbatim):**
> "just a few researchers [10, 11] have explored the negative values in the supply chain data. Besides, very few papers have considered flexible inventory control and cost minimization techniques separately, but none of them provided the probable backorder scenarios in inventory management."

**Hypotheses.** Four, in the "Hypothesis testing" subsection and Table 2, all tested with "the Wilcoxon rank-sum test with continuity correction":
- (a) H0: n = 0, Ha: n ≥ 1, where n = quantity in stock — "Are the products out of stock resulting backorders?"
- (b) H0: Backorder_Yes = n, n = average number of products sold per month
- (c) H0: Backorder_Yes = factors_lead-time
- (d) H0: Backorder_Yes = Forecast_high
All four reported as "p-value < 2.2e−16, for α = 0.05" and all four decided "The alternative hypothesis is true."

**Main claims, verbatim:**
1. Abstract: "We have observed that the performances of the machine learning models have been improved by 20% using this ranged approach when the dataset is highly biased with random error."
2. Conclusions (p. 19): "The comparison result shows that the models' performance increases by approximately 20% when they are trained with the ranged data."
3. p. 14: "The classification accuracy of the model in our case is 0.9892".
4. p. 16: "This figure illustrates that the selected model reached AUC of almost 94% at Depth 9."
5. p. 18: "consider any specific product item 'A' with a fast lead time. Suppose that the inventory level of 'A' is observed either in negative/zero/medium levels along with the high sales level and high forecast level. In this scenario, there are 20% chances that the product item 'A' will be backorder."

**Every figure and table, with number and caption:**
- Table 1 (p. 4) "Review of the related papers"
- Table 2 (p. 7) "Hypothesis testing summary"
- Table 3 (p. 13) "Models' performances during the training phase"
- Table 4 (p. 13) "Models' performances during the testing phase"
- Table 5 (p. 14) "Confusion matrix for GBM with ranged data"
- Table 6 (p. 15) "Models' characteristics during the testing phase"
- Table 7 (p. 19) "Backorder cases where p ≥ 0.2"
- Fig. 1 (p. 6) "Distribution of samples among features"
- Fig. 2 (p. 8) "Decision tree structure"
- Fig. 3 (p. 12) "GBM model's important features with actual data"
- Fig. 4 (p. 12) "GBM model's important features with range converted data"
- Fig. 5 (p. 12) "DRF model's important features with actual data"
- Fig. 6 (p. 13) "DRF model's important features with range converted data"
- Fig. 7 (p. 15) "Area under the curve of GBM models which are trained differently"
- Fig. 8 (p. 16) "Area under the curve of DRF models which are trained differently"
- Fig. 9 (p. 16) "AUC of GBM trained with ranged data at different depth"
- Fig. 10 (p. 17) "GBM model's inventory level distribution for backorder class"
- Fig. 11 (p. 17) "GBM model's lead time distribution for backorder class"
- Fig. 12 (p. 17) "GBM model's sales distribution for backorder class"
- Fig. 13 (p. 18) "GBM model's forecast distribution for backorder class"
- Fig. 14 (p. 18) "GBM model's inventory and lead time level distribution for backorder class"
- Fig. 15 (p. 18) "GBM model's inventory and sales level distribution for backorder class"
- Fig. 16 (p. 19) "Distribution of probable backorder cases for 20% backorder probability"

**Numbers appearing more than once, checked against each other:**
- 0.9892 accuracy: text p. 14 and Table 6. Consistent.
- 0.0107 misclassification: text p. 15 and Table 6. But 1 − 0.98921 = **0.01079**, which rounds to 0.0108, not 0.0107. Off by one in the last digit.
- "20%" improvement: Abstract and Conclusions. GBM 0.7919 → 0.9892 is +19.7 **percentage points** (a +24.9 % relative change); DRF 0.8436 → 0.9835 is +14.0 points. "20%" is the GBM number only, and it is points, not percent.
- 88 % / 99 %: text p. 14–15 vs Table 6 precision 0.8869 and specificity 0.9964. Consistent.
- AUC 0.946 (GBM ranged, Table 4) vs "almost 94% at Depth 9" (p. 16, Fig. 9). Fig. 9's y-axis tops out at ~0.90.

**Inconsistencies found:**

1. **The confusion matrix does not match the stated test set.** Table 5 sums to 11,915 + 1,518 + 3,265 + 426,522 = **443,220** rows. The stated test set is **242,077** observations (p. 5). Nothing in the paper explains the difference. Positives in Table 5 are 15,180 = **3.43 %**, against a stated class balance of **0.669 %** (p. 6). The evaluation set is therefore neither the stated test file nor the stated class distribution — consistent with SMOTE/random oversampling having been applied to the evaluated data, which the paper never states.

2. **The F1 score in Table 6 is arithmetically wrong.** With precision 0.8869 and recall 0.7849, F1 = 2·P·R/(P+R) = **0.8328**. Table 6 reports **0.7845** — i.e. approximately the recall, not the harmonic mean. The same pattern holds for the other three columns: GBM actual (P 0.6896, R 0.5876 → F1 should be 0.6345; reported 0.6345 ✓ — this one is correct); DRF ranged (P 0.8231, R 0.8488 → 0.8358; reported 0.8357 ✓); DRF actual (P 0.7213, R 0.6893 → 0.7049; reported 0.7049 ✓). **Only the flagship column — GBM with ranged data — is wrong**, and it is wrong in the direction of understating the model.

3. **Tree leaf values are read as probabilities.** Fig. 10 shows H2O GBM leaf outputs `prediction: 0.19067599`, `0.16665088`, `−0.19135447`, `−0.11238663`. The text (p. 16–17) converts these directly: "the negative inventory level has produced the probability of positive 0.19 which means that the products with negative inventory may lead to the backorder in 19 cases among 100 cases"; and "the high and the medium inventory levels produce a 30% chance together" (= 0.19135 + 0.11238). GBM leaf values are additive contributions on the link scale, not probabilities, and summing sibling leaves is not a probability operation. Every number in Table 7, Fig. 16 and the worked example on p. 18 rests on this reading.

4. **Fig. 9 contradicts its own text and Table 4.** Text: "reached AUC of almost 94% at Depth 9." Fig. 9's axis runs 0.75–0.90 and the highest points are ≈0.90 at depths 8–10. Table 4 gives the same model AUC 0.946.

5. **Fig. 16's lead-time panel does not match Table 7.** Table 7 has 15 rows: leadtime_level = Fast 7, Very slow 5, Moderate 3. Fig. 16's `leadtime_level` panel has a y-axis to >20 and bars of roughly 23 / 9 / 4 (fast / moderate / very slow). The other three panels of Fig. 16 do sum to 15 and do match Table 7 (inv_level 1+2+4+4+4; sales_level 7+2+2+4; forecast_level 6+1+3+1+4).

6. **"Slight variation" describes a collapse.** p. 14: "It can be observed that the AUC and mean class errors are slightly decreased compared to the training phase. This slight variation is tolerable in the test phase." Training→test AUC for GBM-actual falls 0.994 → 0.795 and mean per-class error rises 0.029 → 0.423 (Tables 3 and 4). The same paragraph correctly calls this "overfitting", so the sentence contradicts the one before it.

7. **Statistical-test rationale contradicts the test chosen.** p. 6: "For large size samples, the distribution of the data can be ignored, and parametric tests can be applied. In this study, the Wilcoxon rank-sum test ... is used" — Wilcoxon rank-sum is non-parametric. Additionally, H0 in Table 2(a) is written "n = 0; where n = Quantity of product in the stock", which is not a hypothesis a two-sample rank-sum test can address.

8. **Table 7 contradicts the paper's own narrative.** Three of its fifteen "probable backorder" rows have `inv_level` = Medium or High with `sales_level` = High and `forecast_level` = Very low — the opposite of the low-stock/high-demand story told in Figs. 10–15. Every row has the identical value 0.2, so the table has no ordering information.

9. **Lead-time ranges have a gap.** p. 9: "0 to 10 days as fast ... 11 to 40 days as moderate, 49 to 120 days as slow, and 121 to 364 days as a very slow one." Days 41–48 are unassigned. Also, a lead time of "0 to 52 weeks" (p. 6) is 0–364 days, so the top bin is the whole tail.

10. **Equation (1) is dimensionally unusual.** `SafetyStock = Z * σ_LT * D_avg` with Z = 1.28 for a 90 % service level. The conventional demand-variability form is Z·σ_D·√LT. As written, safety stock scales with the *standard deviation of lead time* times *average demand*, which is a lead-time-variability form; the paper does not say which it intends and gives no numeric example.

11. **SMOTE is announced and then never reported.** p. 2: "a synthetic minority oversampling technique (SMOTE) [18] on the target class is selected." p. 11: "The ­H2O models are constructed for both synthetic minority oversampling techniques and random oversampling techniques." No result distinguishes the two, and it is never stated which produced Tables 3–7, nor whether oversampling was applied before or after the train/test split.

</scratchpad>

<analysis>

## 1 SNAPSHOT

- **Citation.** Islam, S., Amin, S.H. "Prediction of probable backorder scenarios in the supply chain using Distributed Random Forest and Gradient Boosting Machine learning techniques." *Journal of Big Data* 7, 65 (2020). https://doi.org/10.1186/s40537-020-00345-2. Open access, CC BY 4.0. Received 7 May 2020, accepted 14 Aug 2020.
- **Authors.** Department of Mechanical and Industrial Engineering, Ryerson University, Toronto. First author is the second author's PhD student. Funded by NSERC.
- **Problem.** Predict which products will go on backorder, and — the claimed novelty — output a human-readable list of *probable backorder scenarios* (combinations of inventory / lead-time / sales / forecast levels) that a manager can act on.
- **Data.** The same public Kaggle backorder dataset as Ntakolia et al. (2021): "23 attributes with 1,687,862 and 242,077 observations for the training and testing sets" (p. 5). Only **9** columns are used: `national_inv`, `lead_time`, four sales columns, three forecast columns.
- **Method.** Convert each of the four feature groups into 4–5 ordinal "range levels", scale each level by products of Spearman correlation coefficients with the other groups, divide the forecast level by the average percentage forecast error, then train H2O **DRF** and **GBM** on both the ranged and the raw features with SMOTE/random oversampling.
- **Headline result.** GBM on ranged data: accuracy 0.9892, AUC 0.946, recall 0.7849, specificity 0.9964 (Tables 4 and 6) — against 0.7919 / 0.795 for the same model on raw features. Claimed as a "20 %" improvement.
- **Deliverable.** Table 7: fifteen four-way level combinations with backorder probability "≥ 0.2", plus Fig. 16 summarising their distribution.

## 2 ABSTRACT

- **What it promises.** Higher accuracy *and* flexibility *and* clarity; a tunable ranged method robust to "erroneous data ... which may happen by machine or human errors"; five-level inventory/sales/forecast metrics and a four-level lead-time metric; a list of "major probable backorder scenarios".
- **The one number.** "the performances of the machine learning models have been improved by 20% using this ranged approach **when the dataset is highly biased with random error**."
  - The italicised condition is never operationalised. The paper never injects random error, never defines "highly biased", and never runs a noise-robustness experiment. The 20 % is simply GBM's accuracy gap on the ordinary dataset (0.7919 → 0.9892, +19.7 **points**).
  - It is also "20 %" only for GBM. DRF gains 14.0 points (0.8436 → 0.9835); the abstract says "the machine learning models", plural.
- **Not in the abstract.** The dataset name, the class imbalance (0.669 %), the fact that accuracy is being reported on a near-balanced set (3.4 % positives per Table 5), and any recall/precision figure. Accuracy is the only headline.
- [Assessment] The abstract's central claim is attached to a condition the paper does not test, uses "20 %" for a percentage-point difference, and leads with the one metric that is least informative under class imbalance.

## 3 INTRODUCTION AND MOTIVATION

- **Motivation.** Backorders damage "revenue, share market price, customers' trust" (p. 1–2); satisfying them late "put[s] enormous pressure on different stages of the supply chain" (p. 2).
- **The specific data problem that motivates the method (p. 2, verbatim).** "Machine learning models may misclassify many records if the dataset contains misleading or missing information ... There are very high negative and positive values in several predicting features of this dataset. Our dataset contains the number of negative records related to the inventory."
- **The argument.** Extreme values and negative inventory destabilise tree splits and create "ties"; bucketing into ordinal ranges removes both problems while producing human-readable rules.
- **Adoption framing (Abstract, p. 1).** "Prediction using machine learning algorithms is not well adapted in many parts of the business decision processes due to the lack of clarity and flexibility."
- [Assessment] The motivation is the most useful part of the paper for us, and it is stated well: it is explicitly about *whether a manager can act on the output*, not only about accuracy. It is also the part the paper's own evaluation never touches — no manager, no decision, no cost is ever measured. The gap between stated motivation and executed evaluation is exactly the gap our study is about.

## 4 LITERATURE REVIEW / RELATED WORK

- **Structure.** A dedicated "Literature review" section (pp. 3–5), ~50 references, organised roughly as: ML for supply-chain demand forecasting [19–24]; backorder-specific work [12, 25–27, 29]; forecasting-error metrics [33, 34]; then a separate half-section on decision-tree scalability [37–44].
- **Table 1 (p. 4)** compares six works (including this one) on: prediction domain, ML models, performance metrics, "flexible inventory control with ranged data", "probable decision scenarios". Only the proposed work ticks all three of the last columns. The table has no citations in-cell, only author-year, and the tick placement for the prior works is not justified in the text.
- **Closest prior work** is de Santis et al. (2017) [27], on the same Kaggle data with LOGIST/CART/ensembles. Listed in Table 1 but **no numbers from it are reported**, so the paper never states what it is improving on.
- **Not cited.** Any human-factors, trust, reliance or decision-support literature — despite an abstract built on "flexibility to the decision authority" and "clarity of the process". No calibration literature. No treatment of why accuracy is the wrong headline metric at 0.669 % prevalence.
- [Assessment] Wide but shallow. The decision-tree scalability material (pp. 4–5) is largely unused: the paper never reports runtime or memory, so the "decrease the computational cost" contribution is asserted and never measured.

## 5 METHODOLOGY

**The ranged transformation (pp. 8–10)** is the paper's method. For each of four feature groups:

- **Inventory.** Safety stock per item from Eq. (1) `SafetyStock = Z · σ_LT · D_avg`, Z = 1.28 for "the most common service level in the retail industry which is 90%" [50, 51]. Safety stock is subtracted from current stock, and the residual is binned into five levels {negative, zero, low, medium, high} coded 1–5. Then Eq. (2): `Inventory_level = Inventory_level_status · ρ_IL · ρ_IS · ρ_IF`.
- **Lead time.** Four bins {fast ≤10 d, moderate 11–40 d, slow 49–120 d, very slow 121–364 d} coded 1–4, scaled by Eq. (3).
- **Sales.** Five bins coded 1–5, scaled by Eq. (4).
- **Forecast.** Five bins, divided by ε = average percentage forecast error, Eqs. (5)–(6).
- ρ are **Spearman** coefficients, chosen because "Spearman's method can figure out both linear and non-linear relationships" (p. 7).

**Problems with the transformation as specified:**

- **The bin boundaries for low/medium/high are never given.** p. 8: "The low, medium, and high stock levels can be defined based on the prescribed ranges of supply chain managers. This range may vary based on the business types." The same holds for the five sales levels and five forecast levels. This is presented as the "tunability" contribution, but it means the transformation actually used cannot be reconstructed from the paper.
- **The correlation factors are computed on "the actual dataset"** (p. 11: "multiplied by previously captured correlational factors from the actual dataset"). Whether they were computed on the training partition only is **Not stated in paper**. If computed over all data, the ranged features embed global statistics of the evaluation rows.
- **Eqs. (2)–(4) multiply a level by a product of three correlations.** A product of three coefficients each < 1 shrinks toward zero and, with an odd number of negative ρ, flips sign — so the mapping from level to value is not order-preserving unless the ρ signs happen to cooperate. The paper does not report the ρ values, so this cannot be checked.
- **Eqs. (7)–(8) are identical.** `D_yes` and `D_no` are printed with the same right-hand side (p. 10), so the pair as printed cannot distinguish the two decisions. Neither equation is used afterwards.
- **ε requires actual sales** (Eq. 5), which are also inputs. This is not label leakage, but it does mean the forecast feature is built from realised outcomes.

**Modelling (p. 11).** H2O DRF and GBM; validation frame = 35 % of training; trees tuned 50→1000 with early stopping; learning rate 0.1; max depth 10; row sample 0.9; column sample 1.0. Four models total: {GBM, DRF} × {actual, ranged}.

**Imbalance handling.** SMOTE (p. 2) and "both synthetic minority oversampling techniques and random oversampling techniques" (p. 11). **Which was used for the reported results, and whether it was applied before the split, is Not stated in paper** — and Table 5's 3.43 % positive rate on 443,220 rows implies it was applied to the evaluated data.

**Hypothesis testing (pp. 6–7, Table 2).** Four Wilcoxon rank-sum tests, all "p < 2.2e−16". The justification paragraph argues for *parametric* tests and then uses a non-parametric one; the null hypotheses as written are not two-sample statements; and at n ≈ 1.7 M every test is significant regardless of effect size. No effect size is reported.

- [Assessment] The core idea — discretise, then scale by inter-group association — is not unreasonable and is well motivated by the negative-inventory problem. The specification is not reproducible (unstated bin edges, unreported ρ, identical Eqs. 7–8) and the evaluation set is not the one described. The hypothesis tests add nothing.

## 6 RESULTS

- **Training (Table 3).** AUC: GBM-actual 0.994, DRF-actual 0.987, DRF-ranged 0.985, GBM-ranged 0.979. Mean per-class error 0.029 / 0.030 / 0.060 / 0.089. On training data the *actual*-feature models are better.
- **Testing (Table 4).** AUC: DRF-ranged 0.959, GBM-ranged 0.946, GBM-actual 0.795, DRF-actual 0.787. Mean per-class error 0.103 / 0.070 / 0.423 / 0.430. The ordering reverses.
- **Testing, threshold metrics (Table 6).** GBM-ranged: accuracy 0.9892, precision 0.8869, recall 0.7849, specificity 0.9964, F1 0.7845 (arithmetically wrong — see §7), error 0.0107. DRF-ranged: 0.9835 / 0.8231 / 0.8488 / 0.9986 / 0.8357 / 0.0164. GBM-actual: 0.7919. DRF-actual: 0.8436.
- **Feature importance (Figs. 3–6).** With raw features, GBM's top feature is `sales_3_month` and DRF's is `forecast_3_month`. With ranged features, **both** models put `national_inv` first. The text's descriptions match the figures in all four cases.
- **Decision scenarios (Figs. 10–16, Table 7).** Fifteen level-combinations with "probability ≥ 0.2", drawn from the highest-AUC tree of the GBM-ranged model.

**What the evidence supports.** (1) On this Kaggle data, models trained on raw features with these H2O settings overfit badly (train AUC 0.99 → test AUC 0.79), and the ranged features do not. (2) Ranging changes which feature the models rank first, from a sales/forecast column to inventory. (3) The ranged models reach test AUC 0.95–0.96.

**What it does not support.**
- **That ranging "improves performance by 20 %".** The comparison is against a baseline that the paper itself diagnoses as overfitted ("the AUC is low for both models with actual data which indicates the overfitting of those two models", p. 14). A depth-10, 1000-tree GBM at lr 0.1 on raw features with no regularisation tuning is a weak baseline; the ranged transform acts partly as regularisation. The paper runs no control for this — no tuned raw-feature baseline, no simple binning without correlation scaling, no ablation of the ρ factors, no repeats, no intervals.
- **That the accuracy figure is meaningful.** 0.9892 is measured on a set that is 3.43 % positive (Table 5). On the dataset as described (0.669 % positive, p. 6), always predicting "no backorder" scores **0.9933** — higher than the paper's headline. On the evaluated set, the no-skill floor is **0.9657**. The paper reports neither floor.
- **That Table 7 gives backorder probabilities.** See §7, Figs. 10–13.

## 7 FIGURES AND TABLES (each one, in order)

- **Table 1 (p. 4) "Review of the related papers."** Six rows × five columns of ticks. Positions the paper as the only one with all of {performance metrics, flexible ranged inventory control, probable decision scenarios}. No in-cell citations; the tick assignments are not defended in the text.
- **Table 2 (p. 7) "Hypothesis testing summary."** Four hypotheses, all "p-value < 2.2e−16, for α = 0.05", all decided "The alternative hypothesis is true." Row (a) prints "H0: n = 0 ... Ha: n ≥ 1", which is not a form the stated test evaluates. No effect sizes, no sample sizes, no test statistics.
- **Table 3 (p. 13) "Models' performances during the training phase."** LogLoss / AUC / mean-per-class-error for the four models. On training data, actual-feature models win.
- **Table 4 (p. 13) "Models' performances during the testing phase."** Same three metrics. The ranking reverses. This table plus Table 3 is the paper's strongest and cleanest evidence.
- **Table 5 (p. 14) "Confusion matrix for GBM with ranged data."** TP 11,915 / FP 1,518 / FN 3,265 / TN 426,522. **Sums to 443,220, not the stated 242,077-row test set**; positives are 3.43 %, not 0.669 %. Nothing in the paper reconciles this. This is the single most consequential defect in the paper — it means the evaluation set is undefined.
- **Table 6 (p. 15) "Models' characteristics during the testing phase."** Six metrics × four models. I re-derived all six for the GBM-ranged column from Table 5: accuracy 0.98921 ✓, precision 0.88700 ✓, recall 0.78491 ✓, specificity 0.99645 ✓, error 0.01079 (reported 0.0107 — rounds to 0.0108), **F1 0.8328 (reported 0.7845) ✗**. The other three columns' F1 values are internally consistent; only the flagship column is wrong.
- **Fig. 1 (p. 6) "Distribution of samples among features."** A PCA scatter labelled "PC1: 0.36% variance", "PC2: 0.11% variance". Two components explaining 0.47 % of variance carry no information about separability, and the class-1 points are almost invisible under the class-0 cloud. The figure does not support the caption's claim about feature distribution, and the axis labels suggest the PCA was run on unscaled data.
- **Fig. 2 (p. 8) "Decision tree structure."** Schematic of the five conceptual levels (inventory → lead time → sales → forecast → decision). Illustrative; not a fitted tree.
- **Fig. 3 (p. 12) "GBM model's important features with actual data."** Legible. `sales_3_month` at 1.0, then `forecast_9_month` ≈ 0.23, `forecast_3_month` ≈ 0.21, the rest < 0.15. Matches the text.
- **Fig. 4 (p. 12) "GBM model's important features with range converted data."** Legible. `national_inv` at 1.0, `sales_9_month` ≈ 0.12, `sales_1_month` ≈ 0.11. Matches the text. Note how extreme the concentration is in both Figs. 3 and 4 — one feature at 1.0 and everything else below 0.25.
- **Fig. 5 (p. 12) "DRF model's important features with actual data."** Legible. `forecast_3_month` 1.0, `forecast_9_month` ≈ 0.12, `national_inv` ≈ 0.11. Matches the text.
- **Fig. 6 (p. 13) "DRF model's important features with range converted data."** Referenced in text as `national_inv`, `sales_9_month`, `sales_6_month`. Rendered on p. 13; consistent with the text.
- **Fig. 7 (p. 15) "Area under the curve of GBM models which are trained differently."** Two ROC panels. Left (ranged) is a normal, sharply-rising curve consistent with AUC ≈ 0.95. Right (actual) is a step-like curve that sits below the diagonal between FPR ≈ 0.05 and 0.25 before jumping — an unusual shape consistent with a badly-thresholded/overfitted model. No AUC value is printed on either panel.
- **Fig. 8 (p. 16) "Area under the curve of DRF models which are trained differently."** Same layout for DRF. Left (ranged) normal; right (actual) rises almost vertically to TPR ≈ 0.45 at FPR ≈ 0.02 then goes flat — also atypical. Again no printed AUC.
- **Fig. 9 (p. 16) "AUC of GBM trained with ranged data at different depth."** Sawtooth scatter, x = max depth 2–10, y-axis 0.75–0.90. Maxima rise to ≈0.90 at depths 8–10. **Contradicts the text's "almost 94% at Depth 9" and Table 4's 0.946.** The repeated within-depth spread (≈0.75 to ≈0.90 at every depth) is unexplained — presumably other hyperparameters varying, but the paper does not say.
- **Fig. 10 (p. 17) "GBM model's inventory level distribution for backorder class."** A tree fragment with leaves `prediction: 0.19067599` (negative inv), `0.16665088` (zero), `−0.009626` (low/NA), `−0.11238663` (medium), `−0.19135447` (high). These are **H2O GBM leaf contributions on the link scale**, not probabilities. The text reads them as "19 cases among 100" and sums two of them to "a 30% chance". This misreading propagates to Figs. 11–15, Table 7 and Fig. 16.
- **Fig. 11 (p. 17) "GBM model's lead time distribution for backorder class."** Leaves `0.2` (slow), `0.05203252` (very slow), `0.15274261` (moderate), `−0.015888` (fast/NA). Text: "slow and very slow lead time produces a 25% chance of backorder" — i.e. 0.2 + 0.052 summed. Note the leaf ordering also contradicts the narrative: *moderate* lead time has a higher positive leaf (0.153) than *very slow* (0.052), yet the text reads moderate as protective ("15% chance that the product is not a backorder product if the lead time is in the moderate range") — the sign is positive in the figure.
- **Fig. 12 (p. 17) "GBM model's sales distribution for backorder class."** Leaves `−0.0142` (very low), `−0.0102` (low), `0.0445` (moderate), `0.0504` (high). Text sums the two positives to "9% chances". Same issue.
- **Fig. 13 (p. 18) "GBM model's forecast distribution for backorder class."** Described in text (very high → 20 %, moderate → 8 %, high → 5 %). Note the non-monotonicity: *high* forecast (5 %) is below *moderate* (8 %), which the text reports without comment.
- **Fig. 14 (p. 18) "GBM model's inventory and lead time level distribution for backorder class."** Two-way version of the same construction, read as "16 to 20% chances".
- **Fig. 15 (p. 18) "GBM model's inventory and sales level distribution for backorder class."** "16 to 19% chances".
- **Table 7 (p. 19) "Backorder cases where p ≥ 0.2."** Fifteen four-way combinations, **every one with the identical value 0.2**, so the table conveys membership only, no ranking. Three rows have Medium/High inventory with High sales and Very low forecast — contradicting Figs. 10–13. All values inherit the leaf-as-probability error.
- **Fig. 16 (p. 19) "Distribution of probable backorder cases for 20% backorder probability."** Four count bar charts over Table 7's rows. `inv_level`, `sales_level` and `forecast_level` panels each sum to 15 and match Table 7. **The `leadtime_level` panel does not**: its y-axis exceeds 20 with bars of roughly 23 / 9 / 4, against Table 7's 7 / 3 / 5.

## 8 DISCUSSION

- There is **no discussion section**. The paper goes from "Tree investigation for probable backorder cases" (p. 16) straight to "Conclusions" (p. 18).
- The interpretive work happens inline on pp. 16–18 and consists almost entirely of reading tree leaves as probabilities (see §7). The worked example on p. 18 — "there are 20% chances that the product item 'A' will be backorder" — is the paper's one end-to-end demonstration of its claimed deliverable, and it rests entirely on that reading.
- The overfitting finding (p. 14) is the one genuinely interesting observation and gets three sentences, one of which contradicts the other two ("This slight variation is tolerable").
- No comparison to any prior published result, despite Table 1 naming five.
- [Assessment] The absence of a discussion section is where the paper's problems compound: nothing reconciles Table 5's row count with the stated test set, nothing questions why accuracy exceeds nothing in particular, and nothing checks the leaf values against a probability scale.

## 9 LIMITATIONS

- **There is no limitations section.** The closest is two sentences in the Conclusions (p. 20, verbatim):
  > "However, the relational factors such as local buyout quantity, past due stock, suppliers' performances, and different product risk flags have not been considered because of the lack of that information. We may focus on those factors in our future work."
  This is a scope note, not a limitation — and it is inaccurate: those columns (`local_bo_qty`, `pieces_past_due`, `perf_*_month_avg`, the risk flags) **are** in the same Kaggle file, and Ntakolia et al. (2021) use all of them.
- **Limitations the paper does not acknowledge:**
  1. The evaluation set (443,220 rows, 3.43 % positive) matches neither the stated test file (242,077) nor the stated prevalence (0.669 %).
  2. Accuracy is reported without any no-skill floor, on data whose floor is 0.9657 (evaluated set) or 0.9933 (as described).
  3. The "20 %" improvement is measured against a self-diagnosed overfitted baseline with no tuning control and no repeats.
  4. The "highly biased with random error" condition in the abstract is never tested.
  5. The ranged transform's bin edges and correlation values are not reported, so the method is not reproducible.
  6. No temporal split; backorder prediction is treated as a random-split classification problem.
  7. The computational-cost contribution is claimed four times and never measured.

## 10 CONCLUSION AND FUTURE WORK

- **Conclusion (pp. 18–20).** Restates the ranged approach, the "approximately 20%" gain, the tie-reduction argument, and the managerial value of Table 7: "The decision authority of the respective business will have a broader picture of the backorder conditions as reported in Table 7 and may take necessary actions in advance."
- **Consistency with results.** The 20 % claim carries forward the percentage-point/percent conflation and the untested "random error" condition from the abstract. The tie-reduction claim ("the use of ranged data decreases the number of ties") is asserted here for the first time with **no supporting measurement anywhere in the paper** — no tie counts, no runtime, no memory figures.
- **Future work.** Add the unused columns; model the relationship between predicted demand and predicted backorder; build an integrated model.
- [Assessment] The conclusion introduces two quantitative-sounding claims (tie reduction, time efficiency) that have no corresponding result.

## 11 REPRODUCIBILITY

- **Data.** "The dataset of this research has been published in Kaggle" (p. 5), with row counts. **No URL is given anywhere in the paper.** The Availability statement reads: "The data will be available upon request" (p. 20) — for a public dataset.
- **Code.** None. No repository.
- **Environment.** H2O named (p. 11); no version, no R/Python binding stated, no hardware.
- **Hyperparameters.** Partially given: learning rate 0.1, max depth 10, sample rate 0.9, column sample rate 1.0, trees 50–1000 with early stopping, 35 % validation frame. Missing: early-stopping metric and tolerance, the final tree count, the oversampling ratio, the classification threshold used for Tables 5–6.
- **The transformation.** Not reproducible: the low/medium/high cut-points for inventory, and all five cut-points each for sales and forecast, are explicitly left to "the prescribed ranges of supply chain managers" (p. 8). The Spearman ρ values used in Eqs. (2)–(4) are never printed.
- **Seeds.** Not stated in paper.
- **The blocker.** Even with the code, the reported numbers could not be checked, because the evaluation set that produced Table 5 (443,220 rows) is not described anywhere in the paper.
- [Assessment] Low. The data is nominally public but unlinked; the method has free parameters that are never fixed; and the evaluation set is undefined.

## 12 CRITICAL EVALUATION

**Strengths (max 3)**
1. **The problem framing is the right one.** The paper starts from "ML is not adopted in business decision processes due to lack of clarity and flexibility" and aims at an output a manager can read and tune. Very little of the backorder literature states this goal at all.
2. **The train-vs-test tables (3 and 4) are a clean, honest result.** Reporting that the actual-feature models score AUC 0.99 in training and 0.79 in test, and naming it overfitting, is a real finding that many papers would have quietly dropped. The two tables are the most trustworthy evidence in the paper.
3. **Handling negative inventory explicitly** — treating it as a distinct level rather than a numeric outlier, with a documented cause — is a genuinely practical contribution and transfers directly to any feature built on net inventory position.

**Weaknesses (max 3)**
1. **The evaluation set is undefined.** Table 5 sums to 443,220 rows at 3.43 % positive, against a stated test set of 242,077 rows at 0.669 % positive. Every number in Tables 5 and 6 — including the headline 0.9892 — is computed on a set the paper never describes, most plausibly one that oversampling has altered. Nothing in the paper acknowledges this.
2. **The central interpretive move is wrong.** GBM leaf values (Figs. 10–15) are read as probabilities and summed across siblings. The paper's claimed novel deliverable — Table 7's "probable backorder scenarios", Fig. 16, and the worked example on p. 18 — is built entirely on that reading, so the scenario probabilities are not probabilities.
3. **The headline comparison has no control.** "20 % improvement" compares the ranged models against a raw-feature baseline the authors themselves diagnose as overfitted, with no tuned baseline, no ablation separating binning from correlation scaling, no repeats and no intervals — and it is reported as "20 %" for a 19.7-percentage-point gap, under an abstract condition ("highly biased with random error") that is never tested. Supporting this, Table 6's F1 for the flagship model is arithmetically wrong (0.7845 reported vs 0.8328 derived).

**Credibility: LOW.** Peer-reviewed and open-access in a Springer journal, with a well-motivated problem and one solid finding (Tables 3–4), but the reported evaluation set cannot be reconciled with the described data, the flagship F1 does not follow from the flagship confusion matrix, the claimed deliverable rests on reading tree leaf values as probabilities, and the method's free parameters are never fixed. Individual numbers from this paper should be quoted only with their derivation shown, and none of its methods should be reused.

## 13 RELEVANCE TO MY FOCUS

### Methodology to reuse

- **Almost nothing directly.** The ranged transformation is not reproducible (unstated bin edges, unreported ρ), its evaluation is on an undefined set, and its interpretive layer is wrong. Do not import it.
- **One idea worth keeping in principle, not in implementation:** treating inventory relative to a computed safety stock as an ordinal state, with negative stock as its own level. Our simulator already carries the better version of this — inventory position relative to `Q*` — so this is convergent evidence for the feature, not a method to adopt.
- **A clean cautionary case for our evaluation section.** This paper is the tidiest available example of the three failure modes `build_paper.js:226` names: an accuracy headline with no no-skill floor, an improvement claim with no paired/repeated comparison, and a "best" model chosen against an untuned baseline. It sits alongside Ahmed et al. (2025) (leakage) and Ntakolia et al. (2021) (single split) in Fig. S12.

### Literature review

- Cite for: backorder prediction on the public Kaggle data with bagged/boosted tree ensembles is an established line; DRF/GBM under H2O is one instance; explainability in this literature stops at feature-importance bars and tree readouts, with no human in the loop.
- Cite for the stated-but-unmet goal: a 2020 *Journal of Big Data* paper whose explicit aim is "providing flexibility to the decision authority, better clarity of the process" and which nevertheless evaluates only accuracy/AUC, with no decision-maker, no decision and no cost anywhere in it. This is a strong, quotable instance of the gap our study occupies — stronger than Ntakolia, because this paper says out loud that the decision-maker is the point.

### How our draft uses this paper — support check

| # | Where | Use | Verdict | Location in paper |
|---|---|---|---|---|
| 1 | `paper/build_supplement.js:64` (Fig12 note) | "Islam & Amin (2020) report 98.9 % against a 96.6 % floor." | **Supported** | 98.92 % is Table 6, GBM-ranged column, and the text on p. 14. The 96.6 % floor is (1,518 + 426,522)/443,220 = 96.575 %, computed from Table 5. Both numbers are correctly sourced. |
| 2 | `lightgbm-pivot/make_paper_figures.py:479–480, 490` | `isl_floor = 1 − (11915+3265)/(11915+3265+1518+426522)`; row labelled "Islam & Amin 2020: GBM, ranged data … floor from their Table 5 counts" | **Supported** | The arithmetic matches Table 5 exactly, the model attribution matches Table 6's column header, and the note correctly discloses that the floor is derived from their counts rather than quoted. This is the right way to do it. |
| 3 | `lightgbm-pivot/OVERVIEW.md:162` | "Second backorder study on the same Kaggle data (DRF/GBM); related work" | **Supported, with a caveat worth stating** | Same Kaggle set (p. 5: 1,687,862 / 242,077 rows, 23 attributes); DRF and GBM on H2O (p. 11). Caveat: they use only **9 of the 23 columns** (p. 5) and evaluate on a re-sampled set, so their numbers are **not** comparable to Ntakolia's or ours even though the source file is the same. |
| 4 | `paper/build_paper.js:40` | `islam` defined in `REFS` but never passed to `cite()` in the body | **Housekeeping, not a support problem** | The build script's own `unused` check (`build_paper.js:241`) will flag it. The paper is cited only in the supplement (`build_supplement.js:90`, Fig12 note). Either drop it from the main `REFS` map or cite it in the related-work sentence at line 166 alongside `{ntakolia}`. |

### One thing to tighten before submission

Our Fig12 places Islam & Amin's 98.9 % on the same axis as Ntakolia's 87.78 %, Ahmed's 96.0 % and our own numbers, with the floor drawn from *their* Table 5. That is the most defensible construction available — but **their Table 5 sums to 443,220 rows while their stated test set is 242,077 rows at 0.669 % positive (pp. 5–6)**, so the population behind that bar is not identifiable from the paper. If the "+2.3 pt over floor" gap for this row is load-bearing in the argument, add half a sentence to the Fig12 caption or the supplement note: *"the counts in their Table 5 do not match the test set they describe, so this floor is computed from the evaluated set they report rather than from their stated data."* That converts a reviewer's objection into a demonstration of exactly the point Fig12 is making — that accuracy claims in this literature are uninterpretable without a stated population and floor.

### One claim of ours this paper does **not** support

Nothing in our draft attributes a *method* to Islam & Amin, which is correct — the ranged transformation should stay out. If anyone later reaches for their "20 % improvement" or their Table 7 scenario probabilities as evidence for anything, both are unusable: the first is a percentage-point gap against an overfitted baseline under an untested condition (Abstract; Tables 4 and 6), and the second is built from GBM leaf contributions read as probabilities (Figs. 10–13, Table 7).

</analysis>
