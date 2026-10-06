# Review — Grinsztajn, Oyallon & Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?"

File reviewed: `lightgbm-pivot/papers/05_Grinsztajn2022_NeurIPS_trees-beat-DL-tabular.pdf`
33 pages. **The PDF in our repo is the arXiv preprint** (`arXiv:2207.08815v1 [cs.LG] 18 Jul 2022`, footer on p.1: "Preprint. Under review."), not the camera-ready NeurIPS 2022 Datasets & Benchmarks version. Title on this PDF is "…on tabular data?"; the published title is "…on **typical** tabular data?" (as cited by Hollmann et al. 2025, ref. 14, pp. 507–520). This matters for §13.

All main-text pages (1–9) read; all appendix pages (13–33) read; Figures 1–6 and Figures 7, 9, 10, 13, 19, 20 inspected as page images. The appendix contains 19 further figures (Figs 7–18 benchmark replays, Figs 19, 21–25 per-dataset replays of the main experiments) and 8 hyperparameter tables; **the per-dataset replay figures are summarised in grouped form in §7** rather than one by one, because each is a 13–20 panel small-multiple of an experiment already reported in the main text.

---

<scratchpad>

**Research question (verbatim, title + §5):** "Why do tree-based models still outperform deep learning on tabular data?" and "which features of tabular data make this type of data easy to learn with tree-based methods yet more difficult with a NN?" (p.6, §5.1).

**Main claims (verbatim):**
- Abstract: "Results show that tree-based models remain state-of-the-art on medium-sized data (∼10K samples) even without accounting for their superior speed."
- Abstract: "This leads to a series of challenges which should guide researchers aiming to build tabular-specific NNs: 1. be robust to uninformative features, 2. preserve the orientation of the data, and 3. be able to easily learn irregular functions."
- §4.2: "Tree-based models are superior for every random search budget, and the performance gap stays wide even after a large number of random search iterations."
- §4.2: "Categorical variables are not the main weakness of NNs … most of this gap subsists when learning on numerical features only."
- §5.2 (Finding 1): "NNs are biased to overly smooth solutions" — "For small lengthscales, smoothing the target function on the train set decreases markedly the accuracy of tree-based models, but barely impacts that of NNs."
- §5.3 (Finding 2): "Uninformative features affect more MLP-like NNs" — "removing uninformative features (5a) reduces the performance gap between MLPs (Resnet) and the other models … while adding uninformative features widens the gap."
- §5.4 (Finding 3): "Data are non invariant by rotation, so should be learning procedures" — "random rotations reverse the performance order: NNs are now above tree-based models."
- §1/§3: contributes 45 datasets shared through OpenML plus "every point of a 20 000 compute hours hyperparameter search for each learner."

**Hypotheses:** implicit, not pre-registered. The three findings are framed as post-hoc empirical diagnoses ("we apply various transformations to tabular datasets which either narrow or widen the generalization performance gap", §5.1).

**Figures and tables, with number and caption (verbatim, abridged where long):**
- Fig. 1 (p.5): "Benchmark on medium-sized datasets, with only numerical features." Panels: Classification (15 datasets), Regression (19 datasets).
- Fig. 2 (p.5): "Benchmark on medium-sized datasets, with both numerical and categorical features." Classification (7 datasets), Regression (14 datasets).
- Fig. 3 (p.6): "Normalized test accuracy of different models for varying smoothing of the target function on the train set."
- Fig. 4 (p.8): "Test accuracy of a GBT for varying proportions of removed features."
- Fig. 5 (p.8): "Test accuracy changes when removing (a) or adding (b) uninformative features."
- Fig. 6 (p.9): "Normalized test accuracy of different models when randomly rotating our datasets." a. With all features; b. With 50% features removed.
- Fig. 7 (p.15): "Time benchmark on medium-sized datasets, with only numerical features." Classification (15), Regression (18).
- Fig. 8 (p.15): "Time benchmark on medium-sized datasets, with both numerical and categorical features." Classification (7), Regression (14).
- Fig. 9 (p.17): "Comparison of accuracies on 4 classification tasks for different train set sizes, with only numerical features."
- Fig. 10 (p.17): "Comparison of R2 scores on 3 regression tasks for different train set sizes, with only numerical features."
- Fig. 11 (p.18): "Comparison of accuracies on 2 classification tasks for different train set sizes, with both numerical and categorical features."
- Fig. 12 (p.18): "Comparison of R2 scores on 5 regression tasks for different train set sizes, with both numerical and categorical features."
- Figs. 13–16 (pp.22–24): "Unormalized benchmark results" per dataset, medium-sized setting (classification numerical; regression numerical; classification cat+num; regression cat+num).
- Figs. 17–18 (p.25): "Unormalized benchmark results for large scale datasets" (numerical only; num+cat).
- Fig. 19 (p.27): per-dataset replay of Fig. 3 (smoothing), 15 panels, unnormalized.
- Fig. 20 (p.27): "Decision boundaries of a default MLP and RandomForest for the 2 most important features of the *electricity* dataset." RandomForest (85%), MLP (80%).
- Fig. 21 (p.28): per-dataset replay of Fig. 4 (feature removal).
- Fig. 22 (p.29): per-dataset replay of Fig. 5b (adding uninformative features).
- Fig. 23 (p.29): per-dataset replay of Fig. 5a (removing uninformative features).
- Fig. 24 (p.30): per-dataset replay of Fig. 6a (random rotation, all features).
- Fig. 25 (p.30): per-dataset replay of Fig. 6b (random rotation, 50% features removed).
- Tables 1–8 (pp.19–21): hyperparameter search spaces for FT Transformer, Resnet, MLP, SAINT, XGBoost, RandomForest, GradientBoosting, HistGradientBoosting.
- §A.6 (pp.31–33): eight unnumbered hyperparameter-importance tables (rf_importance / lin_coef) for MLP, Resnet, FT Transformer, SAINT, XGBoost, GradientBoostingClassifier, RandomForest.
- §A.1 (pp.13–14): four unnumbered dataset tables (numerical classification, numerical regression, categorical classification, categorical regression).

**Inconsistencies found (abstract vs results vs conclusion vs appendix):**
1. **Dataset counts do not reconcile.** Abstract/§1/§3.1 say "45 datasets". §A.1 lists 15 + 20 + 7 + 13 = **55 dataset-setting entries** (some datasets recur across settings, e.g. `electricity`, `covertype`, `eye_movements`, `diamonds`, `nyc-taxi-green`, `house_sales`, `Brazilian_houses`, `Bike_Sharing_Demand`). The paper never states the deduplication that produces 45.
2. **Numerical regression: 19 vs 18 vs 20.** Fig. 1 caption says "Regression (19 datasets)"; Fig. 7 — the *same* benchmark plotted against time — says "Regression (18 datasets)"; §A.1.2 lists **20** numerical regression datasets. No explanation is given for any of the three differences.
3. **Categorical regression: 14 vs 13.** Figs. 2 and 8 say "Regression (14 datasets)"; §A.1.4 lists **13**.
4. **Number of random-search reorders varies without comment**: 15 shuffles (Figs. 1, 2, 3, 7–12), 30 shuffles (Figs. 4, 5, 21–23), unspecified for Fig. 6 (caption: "the distribution across random search shuffles").
5. **Ribbon definition changes between a figure and its own replay.** Fig. 4: "the ribbons correspond to the 80% interval among the different datasets." Fig. 21 (same experiment, per dataset): "the ribbons correspond to the minimum and maximum values among these 30 orders."
6. **Fig. 5 y-axis label appears copied from Fig. 4.** Both panels of Fig. 5 are labelled "Normalized **GBT** test score of best model", but the plotted series are GradientBoostingTree, RandomForest, FT Transformer and Resnet — i.e. not GBT only. (Fig. 5b's label reads "Normalized test score of best model", so the two panels of one figure are labelled differently.)
7. **Kadra et al. 2021a/2021b are used interchangeably.** §A.5 is headed "Discussion on Kadra et al. [2021b]" but its first sentence attributes the cocktail result to "Kadra et al. [2021a]", then says "the datasets considered in their paper are quite different, in particular with the presence of 'deterministic' game-inspired datasets in Kadra et al. [2021b]". 2021a = "Well-tuned Simple Nets Excel on Tabular Datasets"; 2021b = "Regularization is all you Need". §3 cites 2021a, §5.2 cites 2021b.
8. **`wine` is below the stated inclusion threshold.** §3.1 requires ≥3,000 samples; §A.1.1 lists `wine` at 2,554 with the note "we noticed a bit late that the number of samples in the transformed wine dataset was just below our threshold, and decided to keep it."
9. **Speed claim is explicitly not a controlled comparison.** Abstract: "even without accounting for their superior speed"; §A.2.1: "this should not be considered a rigorous comparison of the speed of different models, as we use different types of GPUs and CPUs" (NNs on GPU, trees on CPU).

**Numbers appearing more than once, checked:**
- "45 datasets": abstract, §1, §2, §3.1 — internally consistent with each other, inconsistent with §A.1 (see above).
- "20,000 compute hours": abstract ("20 000 compute hours"), §1 ("20,000 compute-hour random searches"), §3.3 — consistent.
- "≈400 random search iterations" (§3.3) — consistent with the x-axis of Figs. 1, 2, 9–12 (log axis running past 100 toward ~400).
- "10,000 samples" medium / "50,000 samples" large: §3.2, §A.2.2, Figs. 9–12 captions — consistent.
- "d/n ratio below 1/10" (§3.1) — consistent with §A.1 entries except `isolet` (7,797 samples, 613 features → d/n ≈ 0.079, passes) and `Mercedes_Benz…` (4,209 samples, 359 features → d/n ≈ 0.085, passes). No violation found.
- RandomForest 85% / MLP 80% on `electricity`: §A.4.1 text and Fig. 20 panel labels — consistent.
- Train/val/test split "70% … of the remaining 30%, we take 30% for the validation set and 70% for the test set" (§A.3) — arithmetic is internally consistent (≈70/9/21).

</scratchpad>

---

<analysis>

## 1 SNAPSHOT

- **Authors / venue:** Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux (Soda–Inria Saclay; ISIR/CNRS–Sorbonne). This copy is the arXiv preprint; the work was published at NeurIPS 2022, Datasets & Benchmarks Track.
- **Type:** benchmark + controlled empirical diagnosis. Not a new model.
- **Scale:** 45 datasets (see scratchpad item 1), 7 learners (RandomForest, GradientBoostingTrees/HistGradientBoostingTrees, XGBoost; MLP, Resnet, FT Transformer, SAINT), ~400 random-search iterations per learner per dataset, ≈20,000 compute hours (§3.3).
- **Headline:** "tree-based models remain state-of-the-art on medium-sized data (∼10K samples) even without accounting for their superior speed" (abstract).
- **Artefacts:** datasets as OpenML suites (ids 297–300, §A.1), code and the full random-search result table at `https://github.com/LeoGrin/tabular-benchmark` (§3.3, §A.7).
- **What it does NOT cover (explicit exclusions, §3.1–3.2):** time series and stream-like data, missing data, high-cardinality categoricals (>20 levels), imbalanced targets (classification targets are binarised and rebalanced 50/50), datasets <3,000 samples or <4 features, deterministic targets, and — in the main benchmark — training sets larger than 10,000 rows. **LightGBM is not among the benchmarked models.**

## 2 ABSTRACT

- **What it claims:** deep learning's superiority on tabular data "is not clear"; tree-based models remain SOTA at ~10K samples; three inductive-bias challenges for tabular NNs (robustness to uninformative features, preserving data orientation, learning irregular functions); a reusable benchmark plus 20,000 compute hours of raw search results.
- **What the abstract supports:** each of the three challenges maps to one §5 experiment (Figs. 3, 5, 6), so the abstract is not over-reaching relative to the body.
- **[Assessment]** The abstract's one soft spot is scope. "Tabular data" is used without the qualifier that later became the published title ("*typical* tabular data") and without flagging the i.i.d./no-missing-data/balanced-classes filtering that §3.1–3.2 impose. A reader who stops at the abstract will over-generalise the result — which is exactly the risk in our own draft (§13).

## 3 INTRODUCTION AND MOTIVATION

- **Problem framing:** "ensemble models based on decision trees like XGBoost remain the go-to tool for most practitioners [Sta] and data science competitions" while "Most corresponding publications claim to beat or match tree-based models, but their claims have been put into question" (p.1).
- **Stated cause of the confusion:** "the lack of an established benchmark for tabular data learning provides additional degrees of freedom to researchers when evaluating their method" (p.1), compounded by small datasets, "unequal hyperparameters tuning efforts [Lipton and Steinhardt, 2019]" and "failure to account for statistical uncertainty in benchmarks [Bouthillier et al., 2021]".
- **Three stated contributions (p.2):** (1) a new benchmark with an explicit dataset-selection and preprocessing methodology, shared via OpenML; (2) a comparison that accounts for the cost of hyperparameter choice, with raw search results shared; (3) an empirical investigation of *why*, via data transformations that narrow or widen the gap.
- **[Assessment]** The motivation is a methodology critique first and a model comparison second. That is the paper's real strength and the part most reusable for us: the benchmarking *protocol* (random search + bootstrap over search order) is a better citation target than the headline "trees win".

## 4 LITERATURE REVIEW / RELATED WORK

- **Four labelled sub-sections (§2):** deep learning for tabular data (encodings, hybrid tree/NN methods, tabular transformers, regularisation); comparisons between NNs and tree-based models; absence of a standard tabular benchmark; understanding the difference.
- **Positioning vs closest prior work:** "The paper closest to our work is Gorishniy et al. [2021], benchmarking novel algorithms, on 11 tabular datasets. We provide a more comprehensive benchmark, with 45 datasets, split across different settings … accounting for the hyperparameter tuning cost" (p.2).
- **Novelty claim:** "To our knowledge, this is the first empirical investigation of why tree-based models outperform NNs on tabular data" (p.3).
- **Acknowledged counter-evidence:** Kadra et al. — "searching across 13 regularization techniques for MLPs to find a dataset-specific combination gives state-of-the-art performances. This provides a partial explanation: MLPs are expressive enough for tabular data but may suffer from a lack of proper regularization" (p.3), revisited in §A.5.
- **[Assessment]** Coverage is adequate for 2022 and the honest engagement with Kadra et al. in §A.5 is unusual. It is now dated in one decisive way: the entire in-context-learning / tabular-foundation-model line (TabPFN) postdates it, and Hollmann et al. 2025 report beating trees **on this very benchmark** (their Extended Data Fig. 2a,b). See §13.

## 5 METHODOLOGY

- **Dataset selection (§3.1), seven stated criteria:** heterogeneous columns; not high-dimensional (`d/n < 1/10`); documented; **"I.I.D. data. We remove stream-like datasets or time series."**; real-world (simulated kept only if "of practical importance"); not too small (≥4 features, ≥3,000 samples); **"Not too easy"** — removed if a default Logistic/Linear Regression is within 5% relative of both a default Resnet and a default HistGradientBoosting; not deterministic (games like poker/chess removed).
- **"Removing side issues" (§3.2):** training set truncated to 10,000 rows (50,000 in the large-size appendix study); **all missing data removed**; classification targets binarised to the two most numerous classes and **rebalanced to 50/50**; categorical features with >20 levels removed; numerical features with <10 unique values removed, and 2-valued numerical features converted to categorical.
- **Tuning protocol (§3.3):** random search of ≈400 iterations per dataset, search spaces from Hyperopt-Sklearn / Gorishniy et al. 2021 / Borisov et al. 2021 (Tables 1–8); trees on CPU, NNs on GPU; "we always start the random searches with the default hyperparameters of each model." To estimate the expected score at each budget *n*, the best-on-validation configuration among the first *n* iterations is scored on test, and "We do this 15 times while shuffling the random search order at each time."
- **Aggregation (§3.4):** accuracy (classification) / R² (regression), affinely renormalised per dataset between the top model and the **10% (classification) or 50% (regression) test-error quantile** rather than the worst model, "Indeed, the worse scores are achieved by outlier models"; negative regression scores clipped to 0.
- **Preprocessing (§3.5):** `QuantileTransformer` Gaussianisation of features for NNs only; log-transform of heavy-tailed regression targets plus an optional target-Gaussianisation hyperparameter; `OneHotEncoder` for models without native categorical support.
- **Splits and folds (§A.3):** 70% train (or the cap), then 30%/70% of the remainder to validation/test, both truncated to 50,000; 1–5 evaluation folds depending on test-set size; "Every algorithm and hyperparameters combination is evaluated on the same folds." NNs: 300 epochs, early stopping (patience 40; 10 for SAINT) with checkpointing on a validation set carved out of train.
- **Diagnostic experiments (§5, §A.4):** (1) Gaussian-kernel smoothing of the **train** target with a `MinCovDet` covariance and varying squared lengthscale, on the 5 most important features per dataset; (2) feature removal/addition ordered by RandomForest importance, added features drawn i.i.d. standard Gaussian; (3) random rotations via `scipy.stats.special_ortho_group.rvs` after Gaussianisation. All three are restricted to "numerical variables and classification tasks on medium-sized datasets" (§5.1).
- **[Assessment — strengths]** The bootstrap-over-search-order estimator is the methodologically important idea: it turns "we tuned both models" into a curve with an uncertainty band, which is directly reusable. The "not too easy" filter and the quantile-based normalisation are both defensible and explicitly justified.
- **[Assessment — weaknesses]** (a) The three diagnostics are **correlational transformations of the data, not causal manipulations of the models**; Findings 1–3 are consistent with the data but alternatives (e.g. optimisation/regularisation deficits, as Kadra et al. argue) are not ruled out, and §A.5 concedes this. (b) Trees-on-CPU vs NNs-on-GPU makes every time-based claim (Figs. 7, 8) non-comparable, which the paper states but the abstract's "superior speed" does not. (c) Rebalancing classes to 50/50 and deleting all missing rows means the benchmark measures a sanitised problem; §6 lists both as open questions.

## 6 RESULTS

**Benchmark (§4.2).**
- *Authors' claim:* "Tree-based models are superior for every random search budget, and the performance gap stays wide even after a large number of random search iterations."
- *Evidence:* Fig. 1 (numerical only) — at ~400 iterations the tree models (GradientBoostingTree ≈0.94, XGBoost ≈0.90, RandomForest ≈0.89 normalized classification accuracy) sit clearly above SAINT ≈0.83, FT Transformer ≈0.80, Resnet ≈0.71, MLP ≈0.68, with non-overlapping min–max ribbons for most pairs. Regression is the same ordering (XGBoost ≈0.98 top). Fig. 2 (num+cat) — XGBoost ≈0.92 vs FT Transformer ≈0.70 and Resnet ≈0.57 in classification; the gap is visibly **wider** than in Fig. 1.
- *Second claim:* "Categorical variables are not the main weakness of NNs … most of this gap subsists when learning on numerical features only." Fig. 1 vs Fig. 2 supports this: the gap narrows but does not close.
- *Not stated in paper:* no significance test, no p-value, no per-comparison confidence interval. The ribbons are min–max over 15 search-order shuffles, i.e. variability of *tuning*, not of data sampling.

**Finding 1 — smoothing (§5.2, Fig. 3).** As the squared lengthscale goes 0 → 0.05 → 0.1 → 0.25, GradientBoostingTree drops from ≈0.86 to ≈0.64 and RandomForest from ≈0.82 to ≈0.61, while Resnet moves from ≈0.72 to ≈0.61 and FT Transformer from ≈0.76 to ≈0.65. *Interpretation given:* "the target functions in our datasets are not smooth, and … NNs struggle to fit these irregular functions compared to tree-based models", consistent with the spectral-bias result of Rahaman et al. [2019]. Fig. 20 shows the mechanism concretely on `electricity`: RandomForest (85% test) tracks vertical stripes along the date feature that MLP (80%) smooths away.

**Finding 2 — uninformative features (§5.3, Figs. 4, 5).** Fig. 4: GBT accuracy on the kept features stays ≈0.95–1.0 until ~50% of features are dropped, while a GBT trained *only on the dropped features* scores ≈0.22 at 10% removed and only reaches ≈0.85 at 90% removed — evidence that the low-importance features are genuinely uninformative rather than merely redundant. Fig. 5b: adding 0→100% uncorrelated Gaussian features costs Resnet ≈0.70→0.45 (≈0.25) versus GradientBoostingTree ≈0.92→0.81 and FT Transformer ≈0.79→0.69 (≈0.10 each). Fig. 5a: removing features shrinks the Resnet-vs-others gap.

**Finding 3 — rotation invariance (§5.4, Fig. 6).** Fig. 6a: with a random rotation, GradientBoostingTree falls ≈0.94→0.67, RandomForest ≈0.89→0.64, FT Transformer ≈0.81→0.74, while **Resnet is unchanged (≈0.75→0.75)** — "only Resnets are rotationally invariant" — and the ranking inverts. Fig. 6b (50% least-important features removed before rotating): the drops shrink (GBT ≈0.92→0.73), linking Findings 2 and 3 through Ng [2004]'s result that a rotationally invariant learner's worst-case sample complexity "grows at least linearly in the number of irrelevant features".

**Large-sized regime (§A.2.2, Figs. 9–12).** *Authors' claim:* "it seems that, in most cases, increasing the train set size reduces the gap between neural networks and tree-based models. We leave a rigorous study of this trend to future work." *Evidence:* in Fig. 9 the large-size (50,000) panel shows SAINT, FT Transformer and RandomForest converging at ≈0.93–0.96 with GradientBoostingTree now trailing, i.e. the ranking partly collapses; only 4 classification and 3 regression datasets survive the filter, so the authors' hedge is appropriate.

**Hyperparameter importance (§A.6).** "The learning rate is by far the most important parameter for neural networks and gradient-boosted trees"; for trees, "the depth of the trees is another very important parameter, and it seems that deeper trees help" (XGBoost `max_depth_10.0` lin_coef 0.35, `max_depth_2.0` 0.13; RandomForest `max_depth_None` rf_importance 0.43, lin_coef 0.54). The authors tie this back to Finding 1 — deeper trees fit more irregular functions.

## 7 FIGURES AND TABLES (in order)

**Main text**

- **Fig. 1 (p.5)** — Medium-sized, numerical only. Classification (15 datasets) and Regression (19). Legible. Shows the core result: at every budget from 1 to ~400 iterations the three tree models are above the four NNs; dotted lines mark default hyperparameters. Note the label discrepancy vs Fig. 7 and §A.1.2 (scratchpad item 2).
- **Fig. 2 (p.5)** — Medium-sized, numerical + categorical. Classification (7) and Regression (14). Legible. Same ordering, larger gap; HistGradientBoostingTree replaces GBT where categoricals are native.
- **Fig. 3 (p.6)** — Smoothing boxplots, 4 lengthscales × 4 models, 15 search reorders. Legible. Supports Finding 1 (trees lose more than NNs as the target is smoothed).
- **Fig. 4 (p.8)** — GBT accuracy vs % features removed, two curves (kept / removed-only). Legible. Supports the "tabular data contain many uninformative features" premise. Ribbon = 80% interval across datasets.
- **Fig. 5 (p.8)** — (a) removing, (b) adding uninformative features. Legible, but the y-axis of panel (a) is labelled "Normalized **GBT** test score" although four different models are plotted (scratchpad item 6).
- **Fig. 6 (p.9)** — Rotation, (a) all features, (b) 50% removed. Legible. The cleanest single figure in the paper: it is the only one where a manipulation *reverses* the ranking rather than shrinking a gap.

**Appendix — read, and summarised in groups (19 figures, 8 numbered tables, 8 unnumbered importance tables, 4 unnumbered dataset tables)**

- **§A.1, four dataset tables (pp.13–14).** Name, n_samples, n_features, original and transformed OpenML link for each of 15 numerical-classification, 20 numerical-regression, 7 categorical-classification, 13 categorical-regression entries. Largest raw sets (`Higgs` 940,160; `covertype` 566,602; `nyc-taxi-green` 581,835; `year` 515,345) are truncated to 10,000 for the main benchmark. This is the table to consult if we ever want a directly comparable dataset.
- **Figs. 7–8 (p.15), time benchmarks.** Same data as Figs. 1–2 with x = random-search seconds. Trees reach their plateau 2–3 orders of magnitude earlier (e.g. RandomForest is at ≈0.83 by 10 s where Resnet needs ≈10⁴ s to reach ≈0.80). Caveat in §A.2.1: GPU vs CPU, "should not be considered a rigorous comparison".
- **Figs. 9–12 (pp.17–18), large-size (50,000) vs medium-size (10,000).** 4 classification + 3 regression datasets (numerical), 2 + 5 (num+cat). Grouped finding: **the tree/NN gap narrows at 50,000 rows**, sometimes to nothing; the authors decline to quantify it.
- **Figs. 13–18 (pp.22–25), per-dataset unnormalised benchmark replays.** 13–20 small panels each. Grouped finding: the aggregate ordering holds on most individual datasets but not all — e.g. on `covertype`, `jannis` and `MiniBooNE` (Fig. 13) the NN curves are close to or overlapping the tree curves, and on `credit` and `house_16H` all models sit within a fraction of a percent. The aggregated figures therefore hide real per-dataset heterogeneity.
- **Fig. 19 (p.27) + Fig. 20 (p.27).** Fig. 19 = per-dataset, unnormalised replay of Fig. 3 across 15 datasets; the smoothing effect is visible on most but is near-flat on `covertype` and `Higgs`. Fig. 20 = the decision-boundary illustration (RandomForest 85% vs MLP 80% on `electricity`), the most interpretable single panel in the paper.
- **Figs. 21–25 (pp.28–30).** Per-dataset, unnormalised replays of Figs. 4, 5b, 5a, 6a and 6b respectively. Grouped finding: no replay contradicts its aggregate, but ribbons on individual datasets are wide enough that several per-dataset effects would not be individually significant.
- **Tables 1–8 (pp.19–21).** Full search spaces and defaults for FT Transformer, Resnet, MLP, SAINT, XGBoost, RandomForest, GradientBoosting, HistGradientBoosting. Directly reusable. (No LightGBM table — LightGBM is not benchmarked.)
- **§A.6 importance tables (pp.31–33).** `rf_importance` and `lin_coef` per hyperparameter per model. Grouped finding: learning rate dominates for MLP (0.22), SAINT (0.28), XGBoost (0.26) and GradientBoosting (0.40); tree depth dominates for RandomForest (`max_depth_None` 0.43).

## 8 DISCUSSION

- **Authors' synthesis (§6, Conclusion):** "This superiority is explained by specific features of tabular data: irregular patterns in the target function, uninformative features, and non rotationally-invariant data where linear combinations of features misrepresent the information."
- **Constructive turn:** the authors read Findings 2–3 as an explanation for why numerical-feature *embeddings* help (Somepalli et al. 2021; Gorishniy et al. 2022) — "this layer breaks rotation invariance … the sheer presence of an embedding which breaks the invariance is a key part of these improvements" (p.8) — and suggest cheaper ways to break rotation invariance as future work.
- **Honest hedging:** §5.2 explicitly does not claim regularisation is irrelevant ("Our findings do not contradict papers claiming benefits from regularization … as adequate regularization and careful optimization may allow NNs to learn irregular patterns"), and §A.5 concedes the result "might no longer be true when adding additional regularization techniques to our random search, such as data augmentation."
- **[Assessment]** The discussion is proportionate to the evidence, and noticeably more cautious than the abstract. The causal language ("is explained by") in the conclusion outruns what three correlational data transformations can establish; "is consistent with" would be the accurate verb.

## 9 LIMITATIONS

- **Stated by the authors (§6, "Limitation"), verbatim:** "which other inductive biases of tree-based models explain their performances on tabular data? How would our evaluation change on very small datasets? On very large datasets? What is the best way to handle specific challenges like missing data or high-cardinality categorical features, for NNs and tree-based models? With these best methods, how would the evaluation change including missing data?"
- **Stated elsewhere:** speed comparison is not rigorous (§A.2.1); the 50,000-row study is under-powered ("this excludes a lot of datasets, which makes the comparison less clear", §A.2.2); Kadra-style regularisation cocktails are untested (§A.5); `wine` breaches the size threshold (§A.1.1).
- **[Assessment] Limitations the paper does not state:**
  1. **No inferential statistics anywhere.** Every comparison is a curve with a min–max ribbon over search-order shuffles. Bouthillier et al. [2021] is cited in the introduction *for* accounting for benchmark variance, yet no confidence interval or test is reported.
  2. **Time series and non-i.i.d. data are excluded by construction** (§3.1) and the paper draws no conclusion about them — but the abstract's phrasing invites that generalisation.
  3. **Class balance is imposed** (§3.2), so nothing here speaks to rare-event prediction, which is the regime of most supply-chain risk work.
  4. **LightGBM is not benchmarked.** The tree family is RandomForest / GBT / HistGBT / XGBoost. Extending the conclusion to LightGBM is an inference, not a finding.
  5. **Medium = 10,000 training rows.** §A.2.2's own evidence points the other way above that.

## 10 CONCLUSION AND FUTURE WORK

- **Conclusion (§6):** "our systematic benchmark, going beyond the specificities of a handful of datasets and accounting for hyper-parameter choice, reveals clear trends. On such data, tree-based models more easily yield good predictions, with much less computational cost." The benchmark is offered as reusable infrastructure: "allowing researchers to use our methodology and datasets for new architectures, and to easily compare them to those we explored via the shared benchmark raw results."
- **Future work named:** other tree inductive biases; very small and very large datasets; missing data; high-cardinality categoricals; cheaper ways to break rotation invariance.
- **[Assessment]** The framing "we hope that this benchmark will stimulate tabular deep-learning research" is the right one. Read as infrastructure, the paper has aged well; read as a permanent verdict on trees vs deep learning, it has not (§13).

## 11 REPRODUCIBILITY

- **Code:** `https://github.com/LeoGrin/tabular-benchmark` (§3.3, §A.7).
- **Data:** all 45 transformed datasets published as OpenML suites (ids 297, 298, 299, 300) with per-dataset original and transformed links in §A.1; the full inclusion/exclusion spreadsheet is linked in §A.1.
- **Raw results:** "A CSV file containing the results of all our random searches. It can be used to cheaply benchmark a new method" (§A.7) — i.e. a new learner can be compared at matched tuning budget without re-running 20,000 compute hours.
- **Protocol detail:** splits, fold counts, epochs, early-stopping patience, and all search spaces are given (§A.3, Tables 1–8); random rotations name the exact SciPy call (§A.4.3); smoothing names `MinCovDet` and the kernel formula (§A.4.1); W&B sweeps are named as the driver (§A.3).
- **Gaps:** hardware is listed as a *set* of 8 GPU models and 4 CPU models "chosen based on availability" (§A.3), so timing results are not reproducible; no random seeds are reported for the search itself; the paper says search spaces were adapted — "We made some changes when combining sources, or when the original distribution was not compatible with Weight and Biases sweeps" (§A.3) — without a diff.
- **[Assessment] High for the benchmark, low for the timings.** This is one of the more reproducible ML benchmark papers of its period; the shared raw-search CSV is the single most valuable artefact.

## 12 CRITICAL EVALUATION

**Strengths (max 3)**
1. **The tuning-budget methodology.** Bootstrapping over random-search order (15 reorders, §3.3) converts an unfalsifiable "we tuned both fairly" into a budget-indexed curve with a variability band. This is the reusable contribution.
2. **Explicit, auditable dataset selection.** Seven written criteria (§3.1) plus a public spreadsheet of every dataset considered *and rejected* — including the "not too easy" filter and the admitted `wine` violation.
3. **The diagnosis is falsifiable and mechanistically specific.** Fig. 6's ranking reversal under random rotation is a strong, clean result: it identifies a property of the *learner* (rotation invariance) rather than a property of the score.

**Weaknesses (max 3)**
1. **No inferential statistics.** Given that the paper's own introduction cites Bouthillier et al. [2021] on benchmark variance, the absence of any test or CI is a real gap; the per-dataset appendix figures (Figs. 13–18, 21–25) show variability wide enough that several component effects would not survive one.
2. **Narrow, sanitised scope presented in broad language.** I.i.d. only, no missing data, balanced classes, ≤20-level categoricals, ≤10,000 training rows, LightGBM absent — yet the abstract and conclusion say "tabular data" unqualified. §A.2.2's own 50,000-row result cuts against the headline.
3. **Correlational diagnosis stated causally.** "This superiority **is explained by**…" (§6) rests on three data transformations; §A.5 concedes a regularisation-based alternative explanation that was never run.

**Credibility: HIGH.** Large pre-registered-style protocol, public datasets, public code, public raw results, and hedges that match the evidence in the body — the overclaiming is confined to abstract/conclusion phrasing, and the scope restrictions are all stated in §3.

## 13 RELEVANCE TO MY FOCUS

**Direct value for "Reliance under disruption":**
- It is the correct citation for *why our AI component is a gradient-boosted tree and not a transformer* — but only at medium-sized, i.i.d., tabular scale.
- The tuning-budget protocol (§3.3) is a defence we can borrow for the reviewer question "did you tune the deep baselines as hard as LightGBM?" — and our `epoch_study.py` / `extra_baselines` work is already a small version of it.
- §A.6 supports our LightGBM hyperparameter emphasis: learning rate first, tree depth second.
- The "not too easy" filter (§3.1) is a defensible precedent if a reviewer asks why we dropped or kept a benchmark.
- **It does not help with calibration, imbalance, or rare events** — the three things that matter most for showing a risk % to a participant. §3.2 rebalances classes away; `papers/09` (van den Goorbergh) remains the right citation there.

**Audit of our draft's use of this paper** (grepped `paper/build_paper.js`, `paper/build_supplement.js`, `paper/NUMBERS.md`, `RESULTS.md`, `lightgbm-pivot/OVERVIEW.md`):

| # | Where | How we use it | Verdict | Location in paper |
|---|---|---|---|---|
| 1 | `paper/build_paper.js:166` (Related Work) | "Gradient-boosted trees usually match or beat deep models on tabular data {grinsztajn}" | **Supported** — with a scope caveat | Abstract; §4.2 "Tree-based models are superior for every random search budget"; Figs. 1–2. "usually match or beat" is weaker than the paper's own claim, so it is safe. Caveat: the paper's evidence is medium-sized (≤10k train), i.i.d., missing-data-free, class-balanced, and does **not** include LightGBM. |
| 2 | `paper/build_paper.js:43` (reference entry) | Cited as "Why Do Tree-Based Models Still Outperform Deep Learning on **Typical** Tabular Data? … NeurIPS 35, Datasets and Benchmarks Track (2022)" | **Supported** (title/venue correct for the published version) — **but** our repo PDF is the arXiv v1 preprint, whose title omits "Typical" and which is marked "Preprint. Under review." Reference lacks page numbers; the published range is **507–520**. Add them. |
| 3 | `lightgbm-pivot/OVERVIEW.md:128` | "~15k correlated windows is small-tabular territory, where trees win (Grinsztajn et al. 2022, `papers/05`)" | **Partially supported** | §3.1 explicitly excludes exactly this data type: "**I.I.D. data. We remove stream-like datasets or time series.**" Our simulator corpus is overlapping time windows from 70 series — non-i.i.d. and outside the benchmark's population. The "trees win at ~10k" half is supported (abstract, Figs. 1–2); the "correlated windows are small-tabular territory" half is our inference, not theirs. Recommend citing `papers/07` (Zeng 2023) for the time-series half and Grinsztajn only for the tabular half. |
| 4 | `lightgbm-pivot/OVERVIEW.md:164` (papers table) | Role: "Why trees beat deep learning at this data scale" | **Supported** for the backorder benchmark (≈22k balanced i.i.d. rows, though above the paper's 10k medium regime) / **Partially** for the simulator windows (non-i.i.d., see row 3). Note §A.2.2: at 50,000 rows the paper's own gap narrows. |
| 5 | `RESULTS.md:396–398` | "tree ensembles remain SOTA at ~10k samples. Diagnosed causes — NNs are not robust to uninformative features, and are hurt by non-rotation-invariant structure. **Exactly our regime.**" | First two sentences **Supported** (abstract; §5.3 Fig. 5; §5.4 Fig. 6). "**Exactly our regime**" is **Not supported** | Three mismatches: (a) §3.1 excludes time series / stream data, which our windows are; (b) §3.2 balances classes 50/50, whereas our stockout label is rare in the raw simulator; (c) the summary omits the paper's *first* and arguably strongest finding — irregular / non-smooth target functions (§5.2, Fig. 3) — which actually **does** fit our step-like inventory threshold and would strengthen the argument. Suggest rewording to "the same medium-sized tabular regime, though their benchmark excludes time series." |
| — | `paper/NUMBERS.md` | no reference to this paper | n/a |

**Two risks a reviewer can raise, and what to do:**
1. **Hollmann et al. 2025 (`papers/06`) beats trees on *this* benchmark.** Their Extended Data Fig. 2a,b reports TabPFN above CatBoost on "the Grinsztajn medium-sized benchmark" for both the 7 categorical and 15 numerical datasets. If our related-work sentence is read as "trees beat deep learning, full stop," the very next paper in our own folder is a published counterexample. The current wording ("usually match or beat") survives this; a stronger wording would not. Consider one clause acknowledging in-context-learning models, which also justifies why `papers/06` sits in our reference list.
2. **Sample size.** Our backorder benchmark trains on roughly 22k balanced rows — above Grinsztajn's "medium-sized (∼10K)" regime and inside the range where §A.2.2 says the gap narrows. If we cite Grinsztajn as the reason not to try a deep model, the honest framing is that our *own* head-to-head (`epoch_study.json`, `extra_baselines`) is the evidence, and Grinsztajn is the prior that motivated it.

**Not stated in paper** (do not cite Grinsztajn for any of these): LightGBM specifically; calibration or probability quality; class imbalance; missing data; time-series or windowed data; datasets above 50,000 rows; any statistical test of the tree-vs-NN gap.

</analysis>
