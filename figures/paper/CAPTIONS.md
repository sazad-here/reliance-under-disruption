# Paper figures: captions and sources

All figures are made by `lightgbm-pivot/make_paper_figures.py` (run `lightgbm-pivot/chart_data.py` first).
Each comes as `.svg` (vector; insert into Word via Insert > Pictures) and `.png` (300 dpi).
Full-width figures are 6.3 in wide. Corpus fingerprint `f953645593cab9a1`.
Colours are the same model in every figure (LightGBM blue, logistic regression orange, XGBoost green, net-position baseline yellow, published values grey), and line style and marker repeat the model for greyscale print.

Suggested use: Figs 3, 7 and 12 in the paper itself (6-page limit); the rest in supplementary material.

---

**Fig. 1. Dataset profile of the simulator corpus.** (a) Ordering policy that generated each of the 520 episodes. (b) Weekly demand over 52,000 weeks (M5 store-department series rescaled to a mean of 140; axis cut at the 99.8th percentile). (c) Share of weeks followed by a stockout within two weeks, for each of the 70 series. (d) Net position (stock on hand plus on order, minus backorders, minus expected demand over the lead time) for weeks with and without a following stockout; whiskers at 1.5 IQR.
*Chart forms from Ahmed et al. 2025 (Figs 11–12) and Islam & Amin 2020 (Fig 1).*

**Fig. 2. Spearman correlations between the state variables and the stockout label.** Net position and cover ratio correlate most strongly with the label (−0.47 each).
*Form from Ahmed et al. 2025 (Fig 10).*

**Fig. 3. Discrimination on 24,960 out-of-fold weeks (5-fold series cross-validation; every M5 series tested once).** (a) ROC curves. (b) Precision–recall curves; the grey line is the 21.5 % stockout rate. LightGBM (AUROC 0.891), logistic regression (0.893) and XGBoost (0.893) overlap. A single engineered feature, net position, reaches 0.869.
*Form from Ntakolia et al. 2021 (Fig 5).*

**Fig. 4. Calibration of the four models.** (a) Observed stockout rate against mean predicted probability in each decile. (b) Calibration slope (ideal 1) and calibration-in-the-large intercept (ideal 0, plotted as intercept + 1). All slopes are between 0.99 and 1.01; the largest intercept is +0.025 (LightGBM).
*Form from van den Goorbergh et al. 2022 (Figs 1, 4, 5) and Li 2026 (Fig 1).*

**Fig. 5. Confusion matrices.** (a) Our LightGBM on the simulator at the F1-maximising threshold (0.35): recall 66.5 %, specificity 90.2 %. (b) The LightGBM confusion matrix published by Ntakolia et al. (2021, Table 3) on a balanced backorder test set. Row percentages; the two panels use different data and are shown for form, not as a head-to-head comparison.
*Form from Ahmed et al. 2025 (Fig 14); values in (b) from Ntakolia et al. 2021 (Table 3).*

**Fig. 6. What the live risk model uses (SHAP, computed with LightGBM's exact TreeSHAP).** (a) Mean absolute SHAP value for the 12 most important features, on the 960 weeks of the two demand series the participants see, which the model never trained on. (b) Beeswarm of 300 of those weeks for the top 8 features; colour shows the feature's value (blue low, red high). Low net position and low cover push the predicted risk up; a disruption pushes it up.
*Form from Ntakolia et al. 2021 (Figs 6–7) and Ahmed et al. 2025 (Figs 15–16).*

**Fig. 7. Ten architectures on identical test weeks.** (a) AUROC at each model's validation-best epoch (or tree count) with series-clustered 95 % bootstrap intervals (2,000 draws); labels give the paired comparison with LightGBM. (b) AUROC after 50 and after 100 epochs (trees for the boosted models). LightGBM is statistically tied with logistic regression and CNN, marginally below XGBoost, and significantly better than six deep architectures. Most neural models got worse with more training.
*Form from Makridakis et al. 2022 (Fig 3).*

**Fig. 8. Validation AUROC by epoch for the seven neural models.** Grey lines are the five folds; the blue line is their mean; the dot marks the mean's best epoch. Most models peak within the first 25 epochs and then decline (overfitting).
*Form from Chen & Guestrin 2016 (Fig 3) and Nie et al. 2023 (Fig 7).*

**Fig. 9. External benchmark on the Kaggle backorder data, using the protocol of Ntakolia et al. (2021).** (a) Grey bars: accuracies published in their Table 3 (single split). Dots: our five random splits; black bar: mean. (b) Paired splits: training LightGBM as an EasyEnsemble (E) improved on the paper's settings (A) on all five splits (+0.34 points on average). Our reproduction of the paper's own LightGBM already scores 90.30 %, so most of the gap to their printed 87.78 % is split-to-split variation; the defensible gain is the paired one in (b).
*Values in grey from Ntakolia et al. 2021 (Table 3).*

**Fig. 10. M5 Accuracy competition, private leaderboard.** Our LightGBM (one model per store, Tweedie loss) scores WRMSSE 0.5691, between the 38th (0.5683) and 39th (0.5698) of 5,507 teams, and 15.2 % better than the best statistical benchmark (ES_bu). Our scorer reproduces the published ES_bu (0.670981) and sNaive (0.847017) scores exactly.
*Form from Makridakis et al. 2022 (Fig 2); 38th and 39th place from the organisers' M5_Scores_and_Ranks.xlsx.*

**Fig. 11. Replication of Li (2026) on DataCo and Olist: share of delay exposure caught by reviewing the top 10 % of orders.** The vertical line is the no-model rule (review the most valuable orders first). On DataCo our pipeline matches the paper (severity × value 0.322 vs 0.323). On Olist our purchase-time features let XGBoost clear the no-model rule (+4.1 points, 95 % CI +2.2 to +6.1), which the paper's model did not; LightGBM does not clear it (CI crosses zero).
*Values in grey from Li 2026 (Table 2).*

**Fig. 12. Reported accuracy measured against each dataset's no-skill floor (the accuracy of always predicting the majority class).**
- **Thaker et al. (2026)** do not name their dataset, so no floor can be computed.
- **Ahmed et al. (2025)** predict DataCo's `Late_delivery_risk`, which is defined from the actual shipping days they also use as an input. Our LightGBM with the same columns reaches 100 %, and 81.6 % with order-time columns only (5 random splits; `ahmed_leakage_check.py`).
- **Islam & Amin (2020)** report 98.92 % against a floor of 96.6 %, which we derived from the counts in their Table 5.
- **Our simulator model** sits 7.8 points above a 78.5 % floor.

*Published values: Thaker 2026 (Results, Fig 3), Ahmed 2025 (Table 5), Islam & Amin 2020 (Tables 5–6), Ntakolia 2021 (Table 3). Our DataCo re-run: `lightgbm-pivot/ahmed_leakage_check.py`.*

**Fig. 13. Robustness of the live risk model.** (a) AUROC with 95 % intervals under different validation schemes (blue) and for the frozen model under stress (grey), on the two demand series participants see. Held-out US states give 0.891–0.899; later M5 weeks (173–272) 0.871; demand +30 % gives 0.837. (b) Calibration intercept (0 = predicted and observed risk agree on average): +0.01 with no shift, +0.52 on later weeks, +0.74 with heavier delays.
*Source: `lightgbm-pivot/robustness_study.py` → `robustness_results.json`. XGBoost comparisons ran on the GPU (CUDA).*

**Fig. 14. Decision value of the risk estimate.** Change in 52-week cost against a base-stock rule with no safety stock, over 150 paired simulations on the session demand series. Adding a top-up when the model's risk is ≥ 50 % cuts cost by 10,981; the same top-up triggered by net position < 0 (no model) cuts it by 10,517. The model rule is 464 cheaper than the no-model rule (95 % CI 331–583). Threshold (50 %) and top-up (60 units = δ) were fixed in advance, not tuned.
*Source: `robustness_results.json` (D).*

**Fig. 15. Error placement.** (a) Extra cost of one accepted AI error in each week, on selection draws 0–99; black = weeks designated propagating, hatched = absorbed, grey = never an error. (b) On held-out draws 100–299, a propagating error causes 3.45 times the extra stockout weeks of an absorbed one for a follower of the reference policy (1.78 times for a noisy follower).
*Source: `results/placement_design.json` (`src/placement_design.py`). Same content as main-text Figure 2.*

**Fig. 00 / paper Fig. 1. How the AI component is built and used.** (a) Six stages, each drawn from real project data: open data (a sparkline of the real M5 session series, the fitted Olist delay distribution, disruption ticks at the DataCo rate); the digital twin as a supplier-warehouse-customer network (8-variable state, Q* with z = 1.60); the training corpus (520 episodes, 52,000 weeks; one real episode shown with its stockout weeks in red); the 30 model inputs of one real week (standardised, blue low / red high); LightGBM as a sum of 194 leaf-wise trees (trees 1, 2 and 194 drawn from the trained model: open dots = splits, blue squares = leaves, grey triangles = deeper subtrees), then Platt calibration to a risk %; validation with the frozen model's ROC on the two held-out session series (AUROC 0.902; CV 0.897; calibration error 0.006). (b) The weekly loop: twin state, own order (locked), AI advice (scripted order + live risk %), final order; the twin runs the week and consequences carry forward.
*Visual grammar from the pipeline figures of Ntakolia et al. 2021, Ahmed et al. 2025 and Thaker et al. 2026, the usual tree-ensemble depiction of gradient boosting (Ke et al. 2017), and in-diagram sizes as in Nie et al. 2023. Source: `lightgbm-pivot/make_architecture_figure.py` (reads every number and graphic from the result files and the trained model); main text `main_text/Figure_model_pipeline` (4.8 in), supplementary `Fig00_model_pipeline` (6.3 in).*

---

## Main-text figures (`main_text/`, printed at a 122 mm text width)

| File | In the draft | Content |
|---|---|---|
| `Figure_model_pipeline.svg/.png` | Fig. 1 | Model build pipeline and weekly loop |
| `Figure1_participant_screen.png` | Fig. S16 (supplement) | Cropped from `demo/output/shots/06_main_ai_correct.png` (week 1 of the live demo) |
| `Figure2_error_placement.svg/.png` | Fig. 2 | = Fig. 15 at 4.8 in, 7–7.5 pt text |
| `Figure3_model_robustness.svg/.png` | (supplementary S13 in the current draft) | = Fig. 13 at 4.8 in |

---

## Published values entered by hand (everything else is read from result files)

| Source | Values | Checked |
|---|---|---|
| Ntakolia et al. 2021, Table 3 | accuracy, recall, F1, precision and confusion matrix for 8 models | every confusion matrix sums to 5,588; the accuracies derived from the matrices match the table |
| Islam & Amin 2020, Tables 5–6 | GBM (ranged data) accuracy 0.9892; TP 11,915, FN 3,265, FP 1,518, TN 426,522 | the accuracy derived from the counts is 0.98921 |
| Ahmed et al. 2025, Tables 4–5 | SOM+ANN test accuracy 96 %; DataCo late rate 54.8 % (our `fit_params.py`) | the label is `Late_delivery_risk` (their text) |
| Thaker et al. 2026 | 91.6 % at the 6-hour horizon | the dataset is not named in the paper |
| M5 xlsx | 38th 0.5683, 39th 0.5698 | same values as `report.html` |
