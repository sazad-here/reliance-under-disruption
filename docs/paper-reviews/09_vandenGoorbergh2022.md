# Review — van den Goorbergh et al. (2022), *The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression*

File: `lightgbm-pivot/papers/09_vandenGoorbergh2022_JAMIA_imbalance-correction-harm.pdf` (74 pages: pp. 1–21 main manuscript, pp. 22–74 supplementary information). This is the **author-accepted manuscript**, not the typeset JAMIA article — the title page carries "Word count main text: 3148" and there is no journal pagination, so page numbers below are the PDF's own and will not match JAMIA 29(9):1525–1534.

<scratchpad>

**Research question (stated as an aim, p.6)**
- "The aim of this study was to investigate the impact of class imbalance corrections on model performance in terms of discrimination, calibration and classification."
- The motivating question (p.4): "This raises the question how class imbalance methods affect calibration."

**Hypotheses — stated verbatim (p.4)**
- "We hypothesize (1) that imbalance correction methods distort model calibration by leading to probability estimates that are too high, and (2) that shifting the probability threshold has similar impact on sensitivity and specificity as the use of imbalance correction methods."

**Main claims verbatim**
1. "all imbalance correction methods led to poor calibration (strong overestimation of the probability to belong to the minority class), but not to better discrimination in terms of the area under the receiver operating characteristic curve." (Abstract, p.2)
2. "Imbalance correction improved classification in terms of sensitivity and specificity, but similar results were obtained by shifting the probability threshold instead." (Abstract)
3. "outcome imbalance is not a problem in itself, and … imbalance correction may even worsen model performance." (Abstract)
4. "The key finding of our work is that training logistic regression models on imbalance corrected data did not lead to better AUROC compared to models trained on uncorrected data, but did result in strong and systematic overestimation of the probability for the minority class." (Discussion, p.10)
5. "We anticipate that risk miscalibration will remain present regardless of type of model or imbalance correction technique, unless the models are re-calibrated." (p.12) — **an anticipation, not a result.**
6. "correcting class imbalance did not result in better prediction models based on standard or ridge logistic regression." (Conclusion, p.12)

**Every figure and table**
- **Table 1** (p.16): "Test set performance metrics." Definitions only (c-statistic, classification accuracy, sensitivity, specificity, calibration intercept, calibration slope, flexible calibration curve, Net Benefit). No data.
- **Figure 1** (p.17): "Flexible calibration curves on the test set for the Ridge models to diagnose ovarian cancer." Four panels: Uncorrected, RUS, ROS, SMOTE. Legible.
- **Figure 2** (p.18): decision curve / Net Benefit plot, six series (Uncorrected, RUS, ROS, SMOTE, All, None). **The caption is not printed on the page** — the figure is referred to in the text (p.8) as the Net Benefit measurement / decision curves. Figure itself legible.
- **Figure 3** (p.19): "Test set AUROC for the Ridge models in the simulation scenarios with an event fraction of 1%." 2 (N) × 4 (p) panel grid of boxplots. Legible.
- **Figure 4** (p.20): "Test set calibration intercept for the Ridge models in the simulation scenarios with an event fraction of 1%." Same grid. Legible.
- **Figure 5** (p.21): "Test set calibration slope for the Ridge models in the simulation scenarios with an event fraction of 1%."
- **Supplementary:** Figure S1 (visualization of RUS/ROS/SMOTE); Figures S2–S6 (AUROC, other event fractions and SLR); S7–S11 (calibration intercept); S12–S17 (calibration intercept after recalibration); S18–S22 (calibration slope); S23–S28 (classification accuracy); S29–S34 (sensitivity); S35–S40 (specificity). Tables S1 (data separation counts), S2 (case-study performance), S3 (AUROC), S4 (calibration intercept), S5 (calibration intercept after recalibration), S6 (calibration slope), S7 (sensitivity), S8 (specificity).

**Repeated numbers, checked against each other**
- Case study: "5914 patients were recruited … of which 3369 premenopausal patients between 18 and 59 years. The prevalence of malignancy was 20% (658/3369)". 658/3369 = 0.1953 ✓ ≈ 20%.
- "training dataset of size 2695 (518 events), and a test dataset of size 674 (140 events)". 2695+674 = 3369 ✓; 518+140 = 658 ✓; 2695/3369 = 0.800 ✓ (4:1 split).
- Threshold "0.192 (518/2695, prevalence of malignancy in the training dataset)". 518/2695 = 0.1922 ✓.
- Calibration intercepts p.8: "0.06 (95% CI −0.16 to 0.26) for SLR and 0.05 (−0.16 to 0.26) for Ridge … varied between −1.32 (−1.54 to −1.11; SMOTE followed by SLR) and −1.50 (−1.72 to −1.28; RUS followed by SLR)". Table S2 (p.28) gives exactly these. ✓
- Calibration slopes p.8: "0.99 for SLR, 1.03 for Ridge" uncorrected; "lowest … for models after RUS (0.85 for SLR, 0.93 for Ridge)". Table S2 ✓.
- Sens/spec p.8: uncorrected t=0.5 "specificity (96% for SLR and Ridge) … sensitivity (31% for SLR, 29% for Ridge)"; corrected or threshold-shifted "sensitivities between 69% and 75%, specificities between 74% and 78%". Table S2: SLR t=0.192 sens 71 spec 76; Ridge t=0.192 sens 70 spec 76; corrected sens 69–71, spec 74–78. **The "up to 75%" upper bound does not appear in Table S2**, whose maximum sensitivity is 71%. Possibly from a figure or the SLR/Ridge union — not reconcilable from Table S2 alone.
- AUROC case study: text "average AUROC of 0.79 to 0.80"; Table S2 shows 0.79–0.80 across all eight models. ✓
- Simulation design: 2 (N) × 4 (p) × 3 (event fraction) = 24 scenarios ✓ ("Twenty-four scenarios", p.6).
- Simulation: "8 different prediction models per simulation scenario" (4 datasets × 2 algorithms) ✓; plus "another 6 models" recalibrated (3 corrections × 2 algorithms) ✓.
- "A sample size of 2500 will include 25 events on average when the event fraction is 1%" — 2500 × 0.01 = 25 ✓.
- "Using RUS implied that the model with 24 predictors was trained on a dataset including only 25 events and 25 non-events on average" ✓ consistent.

**Inconsistencies between abstract, results and conclusion**
1. **p.8, factual error:** "As expected, sensitivity increased and specificity **increased** by changing the classification threshold". Table S2 shows specificity **decreasing** from 96% to 74–78%. Should read "decreased". A typo, but it reverses the meaning of the sentence.
2. **Abstract overstates the classification finding:** "Imbalance correction improved classification in terms of sensitivity and specificity". The evidence (Table S2) is that sensitivity improved (29–31% → 69–71%) while specificity **worsened** (96% → 74–78%). The accurate framing — used in the simulation results on p.9–10 — is "more balanced results for sensitivity and specificity".
3. **Abstract understates the RUS harm:** the abstract says corrections did not lead "to better discrimination". The Discussion (p.11) is stronger: "RUS resulted in lower AUROC values on the test data", and p.9 says "For RUS, the median AUROC was often lower". The abstract's "not better" is weaker than the paper's own finding of "worse".
4. **Conclusion generality vs evidence scope.** Title, abstract and conclusion all correctly say "using logistic regression" / "based on standard or ridge logistic regression". But p.12 adds an anticipation that miscalibration "will remain present regardless of type of model or imbalance correction technique" — explicitly flagged as an expectation, with the follow-up study called for on p.11: "the calibration performance of … Random Forest, Support Vector Machines and Neural Networks remains to be investigated."
5. **Correction methods studied vs mentioned.** Only RUS, ROS and SMOTE were tested. p.11–12: "other imbalance correction methods exist, such as weighting, cost-sensitive learning, or variants of RUS, ROS and SMOTE" — **these were not studied.**
6. Typo p.3: "Class imbalance **is has** been identified as a problem".
7. **Figure 2 has no printed caption.**

</scratchpad>

<analysis>

## 1 SNAPSHOT

- **Citation:** van den Goorbergh, R., van Smeden, M., Timmerman, D., Van Calster, B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *J. Am. Med. Inform. Assoc.* 29(9), 1525–1534 (2022). Affiliations: UMC Utrecht, KU Leuven, Leiden UMC.
- **Type:** Methodological study — one clinical case study plus a Monte Carlo simulation, reported under the ADEMP guideline (p.6).
- **Question:** what do random undersampling (RUS), random oversampling (ROS) and SMOTE do to discrimination, calibration and classification of logistic risk models?
- **Design:** case study on 3369 IOTA ovarian-tumour patients (20% malignancy); simulation over 24 scenarios (N ∈ {2500, 5000}; p ∈ {3, 6, 12, 24}; event fraction ∈ {0.30, 0.10, 0.01}), 2000 runs each, test set N = 100,000 per scenario.
- **Headline result:** no discrimination gain (and a loss under RUS), but median calibration intercepts of −4.5 or worse at a 1% event fraction — i.e. severe systematic overestimation of minority-class risk — which a single logistic recalibration step removes.
- **Scope:** standard and ridge logistic regression only. No tree ensembles, no neural networks, no class weighting, no focal loss.

## 2 ABSTRACT

- **Claims:** "all imbalance correction methods led to poor calibration (strong overestimation of the probability to belong to the minority class), but not to better discrimination in terms of the area under the receiver operating characteristic curve. Imbalance correction improved classification in terms of sensitivity and specificity, but similar results were obtained by shifting the probability threshold instead. Our study shows that outcome imbalance is not a problem in itself, and that imbalance correction may even worsen model performance."
- **Supported:** the calibration claim (Figs. 1, 4; Table S2; p.9), the no-discrimination-gain claim (Fig. 3; Table S3; p.9), and the threshold-shift equivalence (Table S2: SLR t=0.192 gives sens 71 / spec 76, essentially the corrected models' 69–71 / 74–78).
- **Overstated:** "improved classification in terms of sensitivity and specificity." Specificity fell from 96% to 74–78% (Table S2). The paper's own simulation section uses the correct wording — "more balanced results for sensitivity and specificity" (p.10).
- **Understated:** "not to better discrimination" — the Discussion reports RUS producing *worse* AUROC (p.11).
- [Assessment] The abstract is the weakest-worded part of an otherwise careful paper. If our draft quotes it, quote the calibration clause, not the classification clause.

## 3 INTRODUCTION AND MOTIVATION

- **The setup (p.3):** class imbalance is widely framed as a problem, with two quoted examples — He and Garcia: "classifiers tend to provide a severely imbalanced degree of accuracy, with the majority class having close to 100 percent accuracy and the minority class having accuracies of 0-10 percent"; Fernandez et al.: "classifiers … tend to have great accuracy for the majority class while obtaining poor results (closer to 0%) for the minority class."
- **The counter-argument, in two moves (pp.3–4):** (i) "the problem is specific to the classification accuracy measure", whose limitations are well known; (ii) for models that output probabilities, "an adjustment of the classification threshold probability can be used to ensure adequate classification performance … does not have to be 0.5."
- **The reframing:** "The problem then shifts from class imbalance to probability calibration: the model's probability estimates should be reliable in order to make optimal decisions."
- **The decision-theoretic grounding (p.4):** a threshold of 0.1 "means that we accept to treat up to 10 individuals in order to treat 1 individual with the event: we accept up to 9 false positives … per true positive."
- [Assessment] This is the argument our study needs, stated more sharply than we state it. The link *threshold ⇒ implied misclassification costs ⇒ calibration is the binding requirement* transfers directly to a stockout-risk display, where the participant's implied cost ratio is holding cost vs shortage cost.

## 4 LITERATURE REVIEW / RELATED WORK

- **What is targeted:** the imbalance-correction literature — Megahed et al. (Nature Methods 2021) [1], He & Garcia [2], Fernández et al. [3], SMOTE [4,5], Krawczyk [28].
- **The stated gap (p.10):** "The relation between correction for class imbalance and calibration of estimated probabilities is rarely made. For instance, it is not discussed in some key publications on class imbalance for prediction models."
- **Three near-misses the authors identify:**
  - Oommen et al. (2011) [31] "hinted at this link" but "did not systematically study typical imbalance correction methods, and the simulations were based on an unrealistic setting with only one predictor and a true AUROC around 0.99."
  - Wallace & Dahabreh (2014) [32] "quantified calibration incorrectly by using Brier score and class-specific Brier scores … This study incorrectly claimed that using RUS improved probability estimates … This, however, does not mean that the probability estimates are well calibrated, but simply means that the probabilities in the minority class are closer to 1."
  - Dal Pozzolo et al. (2015) [33] found undersampling distorts probabilities and raises variance, but "focused on classification accuracy."
- **Calibration lineage:** Van Calster et al. calibration hierarchy [21] and "Calibration: the Achilles heel of predictive analytics" [30]; Vickers & Elkin decision curve analysis [10].
- [Assessment] The related-work section does real work: it identifies a specific measurement error in a prior paper (Brier score mistaken for calibration) rather than just listing citations.

## 5 METHODOLOGY

**Corrections (p.4; details p.23).** RUS: discard majority cases at random until classes are equal. ROS: resample minority cases with replacement until classes are equal, producing duplicates. SMOTE: for each minority case, take the k = 5 nearest minority neighbours in predictor space (Euclidean), scale the differences by a U(0,1) draw, and add — producing synthetic points "on the 'line' between two original minority cases."

**Models (p.4; details p.24).** Standard maximum-likelihood logistic regression (SLR) and ridge-penalized logistic regression (Ridge), with λ tuned by 10-fold CV on deviance over a grid of 251 values from 0 to 64, log-spaced.

**Case study (p.5).** IOTA data, three recruitment waves (1999–2005, 2005–2007, 2009–2012); 3369 premenopausal patients aged 18–59; only operated patients, so histology is the reference standard; borderline tumours counted malignant. Predictors: age, maximum lesion diameter, number of papillary structures (0–4), with restricted cubic splines (3 knots) for the continuous ones. 4:1 train/test split → 2695 (518 events) / 674 (140 events). Four training sets × two algorithms = 8 models. Thresholds 0.5 and, for the uncorrected models, 0.192.

**Simulation (pp.6–7).** ADEMP-structured. Predictors drawn from independent standard normals; coefficients numerically optimized (BFGS) so that all predictors are of equal strength, the data-generating c-statistic ≈ 0.75, and the prevalence matches the scenario. 2000 runs per scenario; **one** simulated test set of N = 100,000 per scenario. Primary metric: the calibration intercept. Recalibration implemented as a logistic fit with the original linear predictor as an offset and the intercept as the only free parameter. R 3.6.2, glmnet 4.0-2, smotefamily 1.3.1, MASS 7.3-51.5, CentOS7 HPC. Data-separation failures monitored and tabulated (Table S1).

- [Assessment] Three design points worth noting. (i) The data-generating mechanism is deliberately friendly to logistic regression — linear, additive, uncorrelated normal predictors — so any miscalibration found cannot be blamed on model misspecification. That strengthens the calibration finding and weakens any extrapolation to nonlinear learners. (ii) One test set shared across all 2000 runs of a scenario means the reported medians share a common test-set draw; the IQRs reflect training variability, not total variability. (iii) At a 1% event fraction the test set carries ~1000 events, which is adequate for AUROC but leaves the extreme-probability tail thin.

## 6 RESULTS

**Case study.**
- Discrimination: "little variation in discrimination across algorithms and imbalance correction methods, with average AUROC of 0.79 to 0.80" (p.8; Table S2).
- Calibration: uncorrected intercepts 0.06 (SLR) and 0.05 (Ridge); after correction, −1.32 (SMOTE+SLR) to −1.50 (RUS+SLR). Slopes: 0.99 / 1.03 uncorrected; worst 0.85 (RUS+SLR).
- Classification: uncorrected at t=0.5 gives sens 31%/29% and spec 96%; corrected models at t=0.5, or uncorrected models at t=0.192, give sens 69–75% and spec 74–78%.
- Clinical utility: "models trained on imbalance corrected datasets had strongly reduced clinical utility, even to the extent that the Net Benefit was negative when using a probability threshold of 0.3 or higher" (p.8; Fig. 2).

**Simulation.**
- Discrimination: "did not provide evidence that imbalance correction methods systematically improved the AUROC … The median AUROC of models trained on unadjusted data was never lower than the median AUROC of models after RUS, ROS, or SMOTE. For RUS, the median AUROC was often lower, with larger differences when event fraction was lower, training set size was lower, and number of predictors was higher" (p.9).
- Calibration intercept: unadjusted medians between −0.05 and 0.03; corrected medians "−4.5 or lower for scenarios with a 1% event fraction, −2.1 or lower for … 10%, and −0.7 or lower for … 30%" (p.9).
- Recalibration: "improved median calibration intercepts to values between −0.07 and 0.03", with one non-convergent exception (SLR after RUS, 1% events, N=2500, p=24 — an average of 25 events and 25 non-events).
- Calibration slope: SMOTE and, less so, ROS give slopes below 1 (too-extreme probabilities), worse at lower event fraction, lower N, higher p. SLR after RUS gives "very low calibration slopes in some scenarios."
- Classification: at t=0.5 on unadjusted data with a 1% event fraction, "median sensitivities of 0% and median specificities of 100%"; balance is restored either by correction or by shifting the threshold.

- [Assessment] Hypothesis (1) is confirmed decisively and at a magnitude — intercepts of −4.5 at 1% prevalence correspond to risk estimates roughly two orders of magnitude too high on the odds scale. Hypothesis (2) is confirmed. The most useful practical finding for us is the least emphasized one: **a single-parameter logistic recalibration restores the intercept to near zero**, so correction is not irreparable, merely pointless unless it improves discrimination — which it does not.

## 7 FIGURES AND TABLES (each one, in order)

- **Table 1 (p.16), "Test set performance metrics."** Definitions only, no data. Two definitions are load-bearing and worth reusing verbatim: calibration intercept "< 0" means overestimation; calibration slope "< 1" means risk estimates are too extreme. Also the Net Benefit definition: "(TP − w*FP) / N, with w equal to odds(t)", and the criterion "If, for a given t, the Net Benefit of a model is not higher than the Net Benefit of the two default strategies, the model has no clinical utility."
- **Figure 1 (p.17), "Flexible calibration curves on the test set for the Ridge models to diagnose ovarian cancer."** Legible; four panels with loess curves, grey confidence bands, the 45° reference in red dashed, and rug plots of the estimated risks by outcome. The Uncorrected panel tracks the diagonal closely across [0,1]; RUS, ROS and SMOTE all sit far below it — at an estimated probability of 0.6 the observed proportion is roughly 0.25–0.3. The rug plots also show the corrected models pushing mass toward high estimated risk.
- **Figure 2 (p.18), decision curve (caption not printed on the page).** Legible. Net Benefit against probability threshold, six series. The Uncorrected curve stays above zero across the whole 0–0.95 range; RUS, ROS and SMOTE cross zero at ≈0.3 and stay negative to ≈0.85 — i.e. worse than treating no one. This is the strongest single image in the paper: miscalibration is not a cosmetic defect, it makes the model harmful at plausible thresholds. **Caption not printed in this PDF.**
- **Figure 3 (p.19), "Test set AUROC for the Ridge models in the simulation scenarios with an event fraction of 1%."** Legible 2×4 grid (N = 2500 / 5000 columns; p = 3 / 6 / 12 / 24 rows). Boxes for Uncorrected, ROS and SMOTE are essentially indistinguishable at every cell (≈0.71–0.75). RUS is visibly lower and much wider, and the gap grows with p — at p=24, N=2500 the RUS median sits around 0.65 vs ≈0.69 uncorrected, with a long lower tail below 0.55. This panel alone justifies "imbalance correction may even worsen model performance".
- **Figure 4 (p.20), "Test set calibration intercept for the Ridge models … event fraction of 1%."** Legible, same grid, y-axis 0 to −10. Uncorrected boxes sit tight on 0 in all eight cells; RUS, ROS and SMOTE sit at roughly −4.5 to −5 in all eight, with RUS showing the longest lower tails (outliers past −10 at p=12 and p=24, N=2500). The separation is total — no overlap between corrected and uncorrected in any cell.
- **Figure 5 (p.21), "Test set calibration slope for the Ridge models … event fraction of 1%."** Same grid; supports the p.9 text that SMOTE and ROS give slopes below 1.
- **Table S1 (p.~27), data separation counts** per scenario, out of 2000 runs — the paper's error-handling audit.
- **Table S2 (p.28), "Test set performance for the SLR and Ridge models to diagnose ovarian malignancy."** The numeric backing for the entire case study. AUC 0.79–0.80 everywhere; accuracy 82 (uncorrected, t=0.5) vs 73–76 (corrected); sens 29–31 vs 69–71; spec 96 vs 74–78; intercepts 0.05–0.06 vs −1.32 to −1.50; slopes 0.99–1.03 vs 0.85–0.95. "Nc" marks the non-applicable t=0.192 cells for corrected models.
- **Tables S3–S8**, medians with IQR over 2000 runs across all 24 scenarios, for AUROC (S3), calibration intercept (S4), calibration intercept after recalibration (S5), calibration slope (S6), sensitivity (S7), specificity (S8).
- **Figures S1–S40**, the full grid of the above across both algorithms and all three event fractions, plus Figure S1's visualization of RUS/ROS/SMOTE on a 100/1900 two-predictor toy dataset.
- [Assessment] The supplementary material is unusually complete — every main-text figure has its SLR counterpart and its other-prevalence counterparts, and every figure has a backing table. This is a model for how our own Supplementary Material should be organized.

## 8 DISCUSSION

- **Authors' reading (p.10):** "This strong miscalibration reduces the clinical utility of the model: models yielding probability estimates that are clearly too high may lead to overtreatment."
- **The counterintuitive-undersampling argument (p.11):** "undersampling is a very counterintuitive approach, because it deliberately decreases sample size available for model training, which may lead to artificially increased risk of overfitting. Our results are consistent with this expectation: RUS resulted in lower AUROC values on the test data."
- **Prior-work correction (pp.10–11):** the Brier-score critique of Wallace & Dahabreh, noted above, is a substantive methodological contribution in its own right.
- **Scope statement (pp.11–12):** "it is warranted to conduct follow-up studies … in particular in the context of other algorithms … Random Forest, Support Vector Machines and Neural Networks remains to be investigated. Also, other imbalance correction methods exist, such as weighting, cost-sensitive learning, or variants of RUS, ROS and SMOTE."
- **The conditional verdict (p.12):** "class imbalance correction followed by recalibration is only worth the effort if imbalance correction leads to better discrimination of the resulting models."
- [Assessment] That last sentence is the operationally useful form of the paper's finding and the one our draft should be citing: correction is not forbidden, it has to *earn* its place by improving discrimination, and here it does not.

## 9 LIMITATIONS

- **Stated by the authors (pp.11–12):** only logistic regression was studied; only RUS/ROS/SMOTE were studied; weighting and cost-sensitive learning were not; other algorithms "remain to be investigated"; the generalization beyond logistic regression is explicitly labelled an anticipation.
- **Not stated in paper, from the evidence:**
  - The data-generating mechanism is linear, additive, uncorrelated and normal, and the true c-statistic is fixed at ≈0.75. There is no scenario with interactions, nonlinearity, correlated predictors, or a strongly separable signal — the conditions under which imbalance corrections are usually argued to help.
  - One test set per scenario, shared across all 2000 runs; the reported IQRs therefore omit test-set sampling variability.
  - SMOTE was applied with k = 5 and no tuning; SMOTE variants were not examined.
  - The case study has a single 4:1 split with no repetition, so its point estimates carry split-to-split variability that is not quantified (the CIs are analytic, conditional on that split).
  - The case study's imbalance is moderate (20%); the extreme-imbalance evidence is entirely simulated.
  - Figure 2's caption is missing from the PDF.

## 10 CONCLUSION AND FUTURE WORK

- **Conclusion (p.12):** "our study shows that correcting class imbalance did not result in better prediction models based on standard or ridge logistic regression. The imbalance corrections resulted in inaccurate probability estimates without improving discrimination in terms of AUROC. We therefore warn researchers for the limitations of imbalance corrections when developing a prediction model."
- **Supported:** fully, within the stated scope (logistic regression; RUS/ROS/SMOTE).
- **Future work:** other algorithms (RF, SVM, NN) and other correction methods (weighting, cost-sensitive learning, SMOTE variants).
- [Assessment] The conclusion is scoped correctly — the title, abstract and conclusion all say "logistic regression". Any citation of this paper that drops that qualifier is stretching it.

## 11 REPRODUCIBILITY

- **Code:** yes — "The code is available via https://github.com/benvancalster/classimb_calibration" (p.7).
- **Data:** the IOTA case-study data are **not** public (secondary-use ethics approval S64709 is referenced, not a data release). The simulation data are fully synthetic and regenerable from the described mechanism plus the code.
- **Software versions:** R 3.6.2, glmnet 4.0-2, smotefamily 1.3.1, MASS 7.3-51.5, CentOS7 HPC. Unusually complete.
- **Design parameters:** all given — 24 scenarios, 2000 runs, test N = 100,000, k = 5 for SMOTE, ridge grid of 251 values from 0 to 64, 3-knot restricted cubic splines, thresholds 0.5 and 0.192.
- **Seeds:** Not stated in paper.
- **Error handling:** monitored and reported (Table S1), which is rare and good practice.
- **Reporting guideline:** ADEMP (p.6).
- [Assessment] Highest reproducibility of the three papers in this batch. The simulation is fully regenerable from the manuscript alone.

## 12 CRITICAL EVALUATION

**Strengths**
1. Hypotheses are stated in advance (p.4), the simulation is designed and reported under ADEMP with a named primary metric (the calibration intercept), and error handling is audited (Table S1) — a standard of pre-specification that the ML literature it critiques does not meet.
2. The finding is established twice, by independent routes: a real clinical dataset with a decision-curve consequence (Fig. 2: negative Net Benefit above t ≈ 0.3) and a 24-scenario, 2000-run simulation (Figs. 3–5, Tables S3–S6).
3. It supplies the fix as well as the diagnosis — a one-parameter logistic recalibration restores intercepts to −0.07…0.03 (p.9, Table S5) — and states the resulting decision rule: correction is worth it only if it improves discrimination.

**Weaknesses**
1. The abstract's classification sentence ("improved classification in terms of sensitivity and specificity") is not what the evidence shows: specificity fell from 96% to 74–78% (Table S2). The p.8 sentence "sensitivity increased and specificity increased" is a straightforward error — specificity decreased.
2. The generating mechanism is linear, additive and uncorrelated with a fixed true c-statistic of 0.75, i.e. the best case for logistic regression. The scenarios in which practitioners claim imbalance corrections help — nonlinear, interacting, high-dimensional — are untested, and the paper's extension to other algorithms is explicitly an anticipation (p.12), not a result.
3. One test set per scenario shared across 2000 runs, and a single 4:1 split in the case study, so the reported dispersion understates total variability.

**Credibility: High.** Pre-stated hypotheses, an ADEMP-reported simulation with 2000 runs per cell, public code, complete software versions, effects so large (intercepts ≈ −4.5) that no plausible design artefact explains them, and a conclusion scoped exactly to the evidence — with the one genuine defect being sloppy wording in the abstract and one reversed word on p.8.

## 13 RELEVANCE TO MY FOCUS

This is the most directly consequential of the three papers for our study, because **participants see a risk percentage**. A discrimination-only justification of our model is insufficient; the number on screen has to be calibrated or the manipulation is confounded.

**How our draft uses this paper** (grep over `paper/build_paper.js`, `paper/build_supplement.js`, `paper/NUMBERS.md`, `RESULTS.md`, `lightgbm-pivot/OVERVIEW.md`):

| Our use | Location in our draft | Supported? |
|---|---|---|
| "**Explicitly do not:** focal loss or class weighting. van den Goorbergh et al. (JAMIA 2022) show imbalance corrections give **no AUC gain** and strongly overestimate minority probabilities — and our probabilities are shown to participants." | `RESULTS.md:430–433` | **Partially.** The "no AUC gain" and "strongly overestimate" halves are fully supported (Abstract; p.9; Figs. 3–4; Tables S3–S4). But the paper tested **RUS, ROS and SMOTE only**. Focal loss is never mentioned; class weighting is named on p.11–12 as an untested method — "other imbalance correction methods exist, such as weighting, cost-sensitive learning". Extending the finding to focal loss and class weighting is the **authors' anticipation** (p.12: "We anticipate that risk miscalibration will remain present regardless of type of model or imbalance correction technique"), not their evidence. Our wording "show" should become "show … for RUS/ROS/SMOTE in logistic regression, and anticipate the same for weighting". |
| "Imbalance corrections distort probabilities. **Relevant because participants see risk %: calibrate E before display**" | `lightgbm-pivot/OVERVIEW.md:168` | **Supported.** This is exactly the paper's argument (p.4: "The problem then shifts from class imbalance to probability calibration"; p.10: overestimation "may lead to overtreatment"). |
| "Model E's probabilities are averaged over balanced bags and are **not** calibrated to the true base rate. Calibrate before showing any risk % to participants (`papers/09`)." | `lightgbm-pivot/OVERVIEW.md:201` | **Supported**, and it is the correct application: EasyEnsemble bags are balanced, which is precisely the RUS mechanism the paper shows produces intercepts of −4.5 at low prevalence (Fig. 4). The recalibration remedy is also the paper's (p.6–7, p.9, Table S5). |
| "Platt calibration turns the score into a risk"; "calibration slope 1.00"; "calibration error 0.006"; "Calibration is exact for the study conditions (+0.01) but drifts under any shift" | `paper/build_paper.js:182`, `:203`, `paper/NUMBERS.md:26`, `paper/build_supplement.js:69`, `RESULTS.md:835` | **Consistent with the paper's standard.** We report the same two quantities the paper makes primary — calibration slope (ideal 1) and calibration-in-the-large intercept (ideal 0) — with the same sign convention (Table 1, p.16). Good alignment; our supplement Fig04 mirrors their Figure 1. |
| Reference entry for van den Goorbergh et al. | `paper/build_paper.js:53` and `paper/build_supplement.js:97` | See defect 1 below. |
| "(b) Paired splits: LightGBM trained as an **EasyEnsemble** (E) improves on the paper's settings (A) on all five" — the +0.34-point accuracy gain | `paper/build_supplement.js:52`, `:54`, `lightgbm-pivot/OVERVIEW.md` | See defect 2 below. |

**Where our use is not fully supported — three things to fix**

1. **`goorbergh` is defined but never cited in the main paper, so it will not print.** `paper/build_paper.js:53` adds the reference to `REFS`, but the reference list at `:238` is built from `order`, which is populated only by `cite(key)`. Grep finds **no `{goorbergh}` marker anywhere in `build_paper.js`**. The entry is dead code: the paper's own References section will silently omit it. Meanwhile `build_supplement.js:128` prints `MORE_REFS` unconditionally, so it appears in the supplement as an **uncited** reference. Either cite it inline where the paper says "Platt calibration turns the score into a risk" (`:182`) or where the supplement discusses EasyEnsemble calibration, or drop it.

2. **Our EasyEnsemble result and this paper point in opposite directions, and the supplement does not reconcile them.** `build_supplement.js:52` reports EasyEnsemble (a balanced-bag undersampling method) *improving* on the backorder benchmark; this paper reports RUS *lowering* AUROC and wrecking calibration. The two are reconcilable — our Fig09 gain is in **accuracy at a 0.5 threshold on a balanced test set** (`OVERVIEW.md:200` says so explicitly), which is the very metric the paper argues is the wrong one (p.3: "the problem is specific to the classification accuracy measure"). A reviewer who knows this literature will see the tension. The supplement should say in one sentence that the Fig09 comparison inherits the anchor paper's balanced-accuracy protocol, that it is not a calibration claim, and that the live model uses no imbalance correction for exactly the reason in `papers/09`.

3. **Do not let "no AUC gain" travel without its scope.** The result is for **standard and ridge logistic regression** under a linear, additive, uncorrelated data-generating mechanism with true c ≈ 0.75 (p.6). Our live model is a 194-tree LightGBM on 30 engineered features at a 21.5% stockout rate. The calibration mechanism (training on a re-balanced base rate shifts the intercept) is model-agnostic arithmetic and transfers cleanly; the discrimination finding is empirical and logistic-specific. State the first, hedge the second.

**Methodology worth reusing**
- **Report intercept and slope, not "calibration error".** Our draft reports "calibration error 0.006" in one place (`OVERVIEW.md:19`, `build_paper.js:203`) and slope/intercept in others. Table 1 (p.16) gives the reason to standardize on slope + intercept: they separate *too high on average* from *too extreme*, and a single scalar error hides which one is happening. Our robustness results already have both (`RESULTS.md:835`) — use them consistently.
- **Decision curves (Fig. 2) as the consequence layer.** Our supplement's Fig S14 (the ordering rule that saves 464 units) is the same move — showing the estimate has decision value rather than only statistical quality. Vickers & Elkin is already in our `MORE_REFS`; this paper's Table 1 Net Benefit definition and its "no clinical utility" criterion are directly transferable to a cost-based framing.
- **Audit and report failures (Table S1).** Our `RESULTS.md` bug log already does this in spirit; a short table of runs that failed or did not converge would formalize it.

**For the paper's argument**
- This paper supplies the citation for the sentence our Methods needs and does not yet have: *the risk shown to participants is calibrated because an uncalibrated risk display would change the decision the participant faces, independently of the AI's error rate.* Without it, a reviewer can argue that our reliability manipulation (80% vs 60%) is confounded with systematic risk overstatement.
- It also supports the Discussion's third methodological point (`build_paper.js:226`, "model comparisons in supply-chain prediction need no-skill floors, paired reproductions and leakage checks") with a fourth: **and calibration, not just discrimination.**

</analysis>
