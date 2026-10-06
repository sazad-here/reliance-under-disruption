# Review — Makridakis, Spiliotis, Assimakopoulos, "The M5 Accuracy competition: Results, findings and conclusions"

Files reviewed: `lightgbm-pivot/papers/11_Makridakis2022_IJF_M5-accuracy-competition.pdf` and `lightgbm-pivot/papers/M5_Scores_and_Ranks.xlsx`
Version note: the PDF is the **ResearchGate preprint** (cover page: "Preprint · October 2020"; footer p.2: "Preprint submitted to International Journal of Forecasting, October 6, 2020"), 45 pages. Our bibliography cites the published article, *Int. J. Forecast.* 38(4), 1346–1364 (2022). Section and page references below are the preprint's; the published version's pagination differs.

<scratchpad>

**Objective (verbatim, §1)**
> "The competition's objective was to produce the most accurate point forecasts for 42,840 time series that represent the hierarchical unit sales of the largest retail company in the world, Walmart."

**Design innovations claimed (§1, verbatim, abridged)**
> "(a) significantly expanding the number of participating methods, especially those in the category of Machine Learning, (b) including exogenous/explanatory variables in addition to the time series data, (c) using grouped, correlated time series, and (d) focusing on series that display intermittency."

**Main claims (verbatim)**
- §4.1: "all top 50 submissions improve the overall forecasting accuracy of the top-performing benchmark by more than 14%, while the improvements are higher than 20% for the top five performing methods and an impressive 22.4% for the winning team."
- §4.3 Finding 1: "M5 is, therefore, the first competition where all top-performing methods were both 'pure' ML ones and significantly better than all statistical benchmarks and their combinations."
- §4.2: "most of the methods examined utilized LightGBM… LightGBM can be considered as the standard method of choice in Kaggle's recent forecasting competitions".
- §4.3 Finding 3: "all top 50 performing methods in M5 utilized 'cross-learning', exploiting all the information being offered by the dataset."
- §4.1: "the gains of the top-performing methods mainly refer to the top and middle parts of the hierarchies, and are rather limited in terms of WRMSSE at product, product-State, and product-store levels."
- §4.1 (MCB): "the winning method was ranked 13th according to the MCB test, the runner-up was ranked 47th, while the third, fourth, and fifth were ranked 6th, 17th, and 29th".
- §6: "the 50 top-performing ones achieving a superior performance of more than 14% over the most accurate statistical benchmark and the top five more than 20%."

**Key design facts for our purposes**
- 42,840 series in 12 aggregation levels (Table 1, §3.2); submission template covers only the 30,490 level-12 series, aggregated upward for coherence (§3.3).
- Data 2011-01-29 to 2016-06-19 (1,969 days). Train d1–d1913 (validation phase), validation d1914–d1941, test d1942–d1969 (§3.2).
- After the validation phase ended, "the teams were provided with the actual values of the 28 days… They were then asked to re-estimate or adjust their forecasting methods (if needed), in order to submit their final forecasts for the following 28 days" (§3.1) — i.e. the test-phase models could train on d1–d1941.
- "During this time, there was no leaderboard, meaning that no feedback was given to the teams about their actual performance after submitting their forecasts." (§3.1)
- "each team had to select a single set of forecasts (one submission). If no particular forecasts were selected, the ones of the highest accuracy during the validation phase were automatically selected by the system." (§3.1)
- Metric: RMSSE per series, weighted by "cumulative actual dollar sales that each series displayed" in the last 28 training days → WRMSSE (§3.4, eq. 1).
- **5,507 teams, 7,092 participants, 101 countries; 88,136 submissions** (§3.6).
- Of those: "4,373 entered the competition during the validation phase and 1,134 during the test phase… 1,434 teams made submissions during both the validation and the test phase, while 2,939 only during the validation phase." (§3.6)
- "2,666 (48.4%) managed to outperform the Naive benchmark, 1,972 (35.8%) outperformed the sNaive benchmark, and 415 (7.5%) beat the top-performing benchmark (ES bu)… In the latter case [best submission rather than selected], 3,510 (63.7%), 2,685 (48.8%), and 672 (12.2%)… This indicates that many teams failed to choose the best method developed, probably due to misleading validation scores." (§3.6)
- Reproducibility threshold for prizes: "any method that displayed a reproducibility rate… higher than 98%" (§3.5).
- Benchmarks: 24, described in Appendix A/B; "we decided to use ES bu as the single most accurate benchmark against which to compare the forecasting accuracy of all participating methods" (§2).

**Every figure and table (main paper)**
| # | Location | Caption / content |
|---|---|---|
| Table 1 | p.9 (§3.2) | "Number of M5 series per aggregation level." (12 levels, 42,840 total) |
| Table 2 | p.14 (§3.5) | "The six prizes of the M5 Accuracy competition." ($50,000 total) |
| Fig. 1 | p.14 (§3.6) | "Summary of the participating teams and submissions made. Top-left: daily number of submissions… Top-right: Number of participants per country… Bottom-left: distribution of the accuracy (WRMSSE) achieved by the teams that did better than the Naive benchmark… Bottom-right: The accuracy (WRMSSE) and ranks of the teams that did better than the top-performing benchmark (ES bu)." |
| Table 3 | p.17 (§4.1) | "The performance of the top 50 teams of the M5 Accuracy competition in terms of WRMSSE… both per aggregation level and overall. Overall percentage improvements are also reported in comparison to the top-performing benchmark (ES bu)." |
| Fig. 2 | p.18 (§4.1) | "Forecasting accuracy (WRMSSE) of the top 50 performing teams… per aggregation level and box-plots… The red dots indicate the performance of the top-performing benchmark (ES bu), while the green dots the performance of the winning team (YJ STU)." |
| Fig. 3 | p.19 (§4.1) | "Average ranks and 95% confidence intervals of the top 50 performing teams… plus the top-performing benchmark (ES bu) over all series: multiple comparisons with the best (RMSSE used for ranking the methods) as proposed by Koning et al. (2005)." |
| Fig. 4 | p.20 (§4.1) | "Forecasting horizon length's impact on forecasting accuracy" — WRSSE vs horizon at levels 1, 6, 12 and averaged. |
| Table 4 | p.27 (§5.1) | "Percentage improvements (according to WRMSSE) reported between indicative benchmarks… CRO, sNaive, SES, ES bu, ES td, and ESX." |
| Table 5 | p.28 (§5.1) | "Percentage improvements… between the winning submission (YJ STU) and the Croston's method." |
| Table 6 | p.32 (§5.3) | "Percentage improvements… between the winning submission, the best performing submission of each aggregation level… and the simple, equal weighted combination of these five methods." |
| Appendices A, B | supplementary | Benchmark descriptions and their accuracy overall and per level, plus significance analysis. |

**Numbers appearing more than once — cross-checks**
1. Winner WRMSSE: Table 3 = 0.520; xlsx `Accuracy-Top50 (AL)` = 0.5204381930 → 0.520 ✓. Improvement 22.4 %: (0.6710 − 0.52044)/0.6710 = **22.44 %** ✓.
2. ES_bu: not printed in the main text as a number, but xlsx `Accuracy-Benchmarks (AL)` Avg = **0.671** (0.6709809596). Every "improvement over ES bu" column in Table 3 reproduces from it. Rank 50 (Random_prediction, 0.575994): (0.671 − 0.576)/0.671 = 14.2 % ✓ matches Table 3's 14.2 and §4.1's "more than 14%".
3. §3.6 team arithmetic: 4,373 + 1,134 = 5,507 ✓; 1,434 + 2,939 = 4,373 ✓.
4. Table 5's CRO Avg = 0.957; xlsx CRO Avg = 0.9566 ✓. Improvement of the winner over CRO: (0.957 − 0.520)/0.957 = 45.7 % vs the table's 45.6 % — a rounding difference, consistent.
5. Table 4 "CRO vs ES bu" average 29.9 %: (0.671 − 0.9566)/0.671 is negative; the table's convention is (ES_bu − CRO)/CRO = (0.671 − 0.9566)/0.9566 = **−29.9 %**, i.e. the sign convention in Table 4 means "CRO is 29.9 % worse than ES_bu", not better. The text supports this reading ("we observe an average improvement of about 28%" of the ES methods over CRO) but the row label "CRO vs. ES bu" with a positive 52.7 %/29.9 % is ambiguous on its face.
6. **A genuine tension, flagged by the authors themselves**: Table 3 ranks by WRMSSE; Fig. 3 ranks by mean RMSSE rank (MCB). The two disagree sharply — the WRMSSE winner is 13th on MCB and the runner-up 47th (§4.1, §5.1). The paper's interpretation: "the winning teams developed methods that mostly focused on expensive and fast-moving products for which WRMSSE is minimized".
7. §4.1 claims all top-50 beat ES_bu "by more than 14%", but §3.6 reports only 415 teams (7.5 %) beat ES_bu at all — consistent, since 50 ⊂ 415.
8. The 42,840-series total (Table 1) vs 30,490 submitted series (§3.3): both correct; coherence assumption bridges them.

**Inconsistencies / gaps**
- The paper publishes scores for **only the top 50 teams** (Table 3; §3.6: "The various tables presented in the remainder of this paper focus on the top 50 performing teams"). Ranks 51–5,507 appear only as the unlabelled curve in Fig. 1 bottom-right (ranks 1–415).
- Method descriptions were obtained "for just 17 of them" out of the top 50 (§4.2), and §5.2 calls this "a serious problem". So Findings 2, 3, 6 and 7, which generalise about "the top 50", rest on 17 disclosed methods plus the 5 prize-winners.
- The preprint's cover page says "CITATIONS 0" and dates it October 2020, while the competition's final results were presented in May 2021 (§2) — an artefact of the ResearchGate snapshot, not of the article.

</scratchpad>

<analysis>

## 1 SNAPSHOT
- **Paper**: Makridakis, S., Spiliotis, E., Assimakopoulos, V. "The M5 Accuracy competition: Results, findings and conclusions." Preprint Oct 2020; published *Int. J. Forecast.* 38(4), 1346–1364 (2022).
- **Type**: forecasting-competition report (design, results, findings), not a method paper.
- **Task**: 28-day-ahead daily point forecasts for Walmart unit sales — 42,840 series across 12 hierarchical levels, with prices, promotions (SNAP) and event calendars as exogenous inputs; heavily intermittent at the product-store level.
- **Scale**: 5,507 teams, 7,092 participants, 88,136 submissions, 24 benchmarks, $50,000 in prizes.
- **Metric**: WRMSSE — RMSSE per series, weighted by each series' cumulative dollar sales over the last 28 training days, averaged over all 42,840 series.
- **Headline**: every one of the top 50 was a machine-learning method; most used LightGBM; the winner beat the best of 24 benchmarks by 22.4 %.

## 2 ABSTRACT
- **Claim**: M5 "extended the results of the previous four competitions" on four axes — more ML methods, exogenous variables, grouped/correlated series, and intermittency.
- **Claim**: the paper "presents the background, design, and implementation of the competition, its results, the top-performing methods, as well as its major findings and conclusions."
- **[Assessment]** The abstract makes no accuracy claim at all; it is purely descriptive. All quantitative claims live in §4. This is unusually disciplined and means there is no abstract-vs-results gap to audit.

## 3 INTRODUCTION AND MOTIVATION
- **Purpose of competitions (§1)**: "to empirically evaluate the accuracy of existing and new forecasting methods, allowing the equivalent of experimentation widely used in hard sciences."
- **Why M5 differs from M1–M4 (§1, §2)**: hosted on Kaggle (so far more teams and "unstructured" ML methods); exogenous variables supplied; grouped, hierarchically correlated series rather than "mostly unrelated series"; and "for the first time ever, the competition involved series that display intermittency, i.e., sporadic demand including many zeros".
- **Pre-registration of expectations (§1)**: "our ten predictions/hypotheses, submitted to the EiC of IJF five days before launching the competition, demonstrate our expectations of its findings", evaluated in a companion paper.
- **[Assessment]** The pre-registration of hypotheses before a competition is a methodological practice worth noting for our own paper — it is the competition-scale analogue of what we do with our pre-registration.

## 4 LITERATURE REVIEW / RELATED WORK
- §2 is organised around Hyndman (2020)'s seven desiderata for competitions (regularly updated benchmarks, clearly defined domain, objective measures, distributional assessment, large-scale multivariate data, high-frequency data, exogenous variables) and shows point by point which M5 addresses.
- Prior M competitions: M1/M2/M3 found simple statistical methods at least as accurate as sophisticated ones; M4 was "the first forecasting competition identifying two ML methods significantly more accurate than simple, statistical ones" (Finding 1), though both M4 winners were hybrids built on statistical components.
- Benchmark provenance (§2): "most of the benchmarks considered in the competition were previously tested in a different dataset, involving the daily product sales of a large Greek retail company (Spiliotis et al., 2020b)."
- Intermittent-demand literature: Croston, SBA, TSB, ADIDA, iMAPA (Syntetos & Boylan; Appendices A/B).
- **[Assessment]** The related-work framing is the field's own historiography, written by its principal authors. That is a limitation of perspective — M5's design and its "findings" are judged against a narrative the same authors constructed — but the 24 benchmarks and open data make the claims independently checkable, as our own re-run demonstrates.

## 5 METHODOLOGY (competition design)
- **Data (§3.2)**: 3,049 products × 3 categories × 7 departments × 10 stores × 3 states; 1,969 daily observations (2011-01-29 to 2016-06-19). Exogenous: special events in four classes, weekly store-level selling prices, SNAP binary flags.
- **Test-window selection (§3.2)**: "the test set was randomly chosen by the organizers from the original dataset provided by Walmart (around 6 years of data), with the only restrictions being that (i) more than 5 years of data should be available for training and (ii) at least two special events should be included in the validation and test set". The test window contains Memorial Day, part of Ramadan and the NBA finals.
- **Two phases (§3.1)**: validation (live public leaderboard, max 5 submissions/day, hidden 28 days) then test (**no leaderboard, no feedback**, one submission selected blind).
- **Submission format (§3.3)**: only the 30,490 level-12 series; higher levels obtained by summation, enforcing coherence.
- **Metric (§3.4)**: RMSSE with a Naive in-sample denominator computed "only for the periods during which the examined product(s) are actively sold"; weights from cumulative dollar sales in the last 28 training days; all 12 levels equally weighted because dollar sales sum consistently across levels.
- **Why squared, not absolute, errors (§3.4)**: "absolute errors, which are optimized for the median… would assign lower scores (better accuracy) to forecasting methods that derive forecasts close to zero. However, the objective of the competition is to accurately forecast the average sales."
- **Prize reproducibility (§3.5)**: code within 14 days, organiser re-run, "reproducibility rate… higher than 98%"; all winning methods were fully reproducible.
- **[Assessment]** The design is the strongest part of the paper: pre-specified metric, blind test phase, public data and weights, published benchmark code, and an enforced reproducibility check. The one design decision that shapes every downstream finding is the dollar-sales weighting, which the authors defend explicitly and whose consequences they then document honestly in Fig. 3 / §5.1.

## 6 RESULTS
- **Top 50 (Table 3)**: winner YJ_STU 0.520 (22.4 % better than ES_bu); 2nd Matthias 0.528 (21.3 %); 3rd mf 0.536 (20.2 %); 50th Random_prediction 0.576 (14.2 %).
- **Participation outcomes (§3.6)**: only 415 of 5,507 teams (7.5 %) beat ES_bu with the submission they selected; 672 (12.2 %) would have with their best submission. "many teams failed to choose the best method developed, probably due to misleading validation scores."
- **Where the gains are (§4.1, Fig. 2)**: "the average improvement of the methods over the benchmark is 40% at level 1, which drops to about 23% at levels 5, 6, and 7, and reaches 3% at levels 10, 11, and 12."
- **Rank vs weighted error (§4.1, Fig. 3)**: SHJ (8th), DenisKokosinskiy_STU (10th) and XueWang (11th) are significantly better on mean RMSSE rank than the rest; the WRMSSE winner is 13th and the runner-up 47th on MCB.
- **Horizon (Fig. 4)**: accuracy is roughly flat across the 28 days at high levels, deteriorates sharply at levels 10–12, and shows weekly periodicity in errors — "part of the seasonality present in the data was not appropriately captured by the forecasting methods of the competition, even the top-ranked ones."
- **Winning methods (§4.2)** —
  - 1st **YJ_STU**: equal-weighted mean of 220 LightGBM models (per store / store-category / store-department × recursive and non-recursive), 6 models per series, **Tweedie** negative log-likelihood, no early stopping, CV on the last four 28-day windows measuring both mean and SD of errors.
  - 2nd **Matthias**: 10 per-store LightGBM models with a custom asymmetric loss, adjusted by multipliers derived from N-BEATS forecasts at the top five levels; past sales not used as LightGBM features.
  - 3rd **mf**: equal-weighted combination of 43 LSTM-based networks, Tweedie regression, 100 features.
  - 4th **monsaraida**: non-recursive per-store LightGBM, **a separate model for each week of the horizon** (4 per store, 40 total), Tweedie, no early stopping, no parameter tuning.
  - 5th **Alan Lahoud**: recursive per-department LightGBM (7 models), Poisson regression with early stopping, store-department mean-matching multipliers.
  - "almost all of them adopted similar approaches to the winning submission, training recursive and non-recursive LightGBM models per store, department, or store-department."
- **Benchmarks against each other (Tables 4–5)**: seasonality matters more than exogenous variables — ESX beats ES_td by 5.7 % on average (25.5 % at level 1, negative at levels 10–12), while the exponential-smoothing family beats Croston by ~28–30 % on average. The winner beats Croston by 45.6 % overall but only 4.5 % at product-store level.
- **Combination of level-best methods (Table 6, §5.3)**: an equal-weighted combination of five submissions is 2.0 % better than the winner overall and, unlike a per-level selection, is automatically coherent.
- **[Assessment]** The two results that matter most outside the competition are (a) the gains concentrate at aggregated levels and nearly vanish at product-store level, and (b) the WRMSSE ranking and the per-series rank ranking disagree. Both are the authors' own findings and both caution against reading a WRMSSE rank as general forecasting skill.

## 7 FIGURES AND TABLES (each one, in order)
- **Table 1 (p.9)** — Series counts per level: 1, 3, 10, 3, 7, 9, 21, 30, 70, 3,049, 9,147, 30,490 = 42,840. Clean and load-bearing; every WRMSSE figure depends on it.
- **Table 2 (p.14)** — Prize schedule, $25k/$10k/$5k/$3k/$2k plus a $5k student prize. Contextual only.
- **Fig. 1 (p.14)** — Four panels, legible at 150 dpi. Top-left: submissions collapse after the validation phase ends (red dashed line, 1 June) and spike again in the final days — visible evidence for §4.1's "mainly submitted four days before the competition ended". Bottom-left: the WRMSSE density of sub-Naive teams is sharply peaked just above the ES_bu line, with a long right tail to 1.8. **Bottom-right is the only place in the paper showing scores beyond rank 50**: WRMSSE rises from ~0.52 at rank 1 through roughly 0.57 around rank 40 to 0.67 at rank 415, with a right-hand axis in percentage error decrease (0 to −20 %). It is a smooth, steep-then-flat curve — which is why small WRMSSE differences translate into large rank movements in the 30–60 range.
- **Table 3 (p.17)** — The competition's core result table: 50 teams × 12 levels + average + improvement. Values verified against `M5_Scores_and_Ranks.xlsx` (see §13). Note the near-ties: ranks 22 and 23 (CPUkiller, dont_overfit) have **identical** level-by-level scores to three decimals, i.e. two teams submitted effectively the same forecasts.
- **Fig. 2 (p.18)** — Box plots of top-50 WRMSSE per level, with ES_bu (red) and the winner (green) overlaid. Legible. The visual story is unambiguous: at levels 1–9 the whole top-50 box sits well below the red dot; at levels 10, 11 and 12 the box straddles the red dot and the green dot sits inside the box. Median top-50 WRMSSE rises from ~0.25 (level 1) to ~0.98 (level 10) and ~0.89 (level 12).
- **Fig. 3 (p.19)** — MCB intervals for 50 teams plus ES_bu, sorted by mean RMSSE rank. Legible as a list; the interval widths are readable. ES_bu sits mid-list, ahead of eight of the top-50 WRMSSE teams — a striking result that the paper states plainly.
- **Fig. 4 (p.20)** — Four panels of WRSSE vs horizon day with LOESS. Level 1 and level 6 are flat; level 12 rises with horizon; all show weekend spikes.
- **Table 4 (p.27)** — Pairwise benchmark improvements. The sign convention is not stated in the caption and reads ambiguously (scratchpad 5); the surrounding text disambiguates it.
- **Table 5 (p.28)** — Winner vs Croston per level: 77.9 % at level 1 down to 4.5 % at level 12, average 45.6 %.
- **Table 6 (p.32)** — Winner vs level-best vs equal-weighted combination. The combination is 2.0 % better than the winner overall but −0.4 % against the level-best oracle, and negative at levels 1, 4, 5, 10, 11, 12.
- **Appendices A and B** — Benchmark descriptions and per-level accuracy with significance analysis. Provided as supplementary material; not in this PDF. The corresponding numbers are in `M5_Scores_and_Ranks.xlsx`, sheet `Accuracy-Benchmarks (AL)` (24 benchmarks × 12 levels + average).

## 8 DISCUSSION
- §5.1's framing: "ML methods have entered the mainstream of forecasting applications, at least in the area of retail sales forecasting" — the qualifier is the authors' own.
- The decomposition in Tables 4–5 is the most useful analytical move: it separates seasonality handling (ES vs Croston, ~28–30 %) from exogenous variables (ESX vs ES_td, 5.7 %) from the remaining ML gain (winner vs ES family), and finds that "seasonality, observed mainly at higher aggregation levels, is the most critical factor for improving overall forecasting performance."
- §5.1 on expertise: "the winning team was a student with little forecasting knowledge and little experience in building sales forecasting models… it would not be surprising to see the value of knowledge and experience become less important in identifying and extrapolating data patterns." The approaches are characterised as "unstructured, agnostic… mostly depending on experimentation and data mining, without the need to understand the data itself".
- **[Assessment]** §5.1 is candid about the flip side: the winning methods are tuned to the metric's dollar weights, they are barely better than Croston at product-store level, and their per-series ranks are mediocre. A WRMSSE rank is therefore best read as "competent on this metric, on this dataset, in this protocol" — which is exactly the reading our own use of it should adopt.

## 9 LIMITATIONS
**Stated by the authors (§5.2):**
- Generalisability: M5 covers "retail sales data, structured across twelve cross-sectional levels", and findings "cannot be generalized beyond the specific area covered by the data, except to draw some general conclusions about the methods used or how they were selected or evaluated."
- No link to cost: "it focused on the point forecast accuracy of the submitted methods, which were not directly linked to Walmart's underlying operational costs… translating forecasting error reduction into cost savings is far from trivial." Holding and backlog cost data were not available.
- Disclosure: "only 17 of the 50 top-performing methods shared information about their forecasting approaches, even after the organizers had sent several emails requesting this information."
- §4.1 / §5.1: the winning methods' per-series ranks are poor; it "remains to be seen whether these methods can be effectively adjusted to accurately predict all the series of the competition equally".
- §5.3: hierarchical reconciliation "proved much more challenging than expected… with many teams trying to apply well-known reconciliation methods… but failing to do so."

**Not stated by the authors [Assessment]:**
- Scores for ranks 51–5,507 are never tabulated; only Fig. 1's curve covers ranks up to 415. Any comparison against "N-th of 5,507" outside the top 50 is an interpolation off a figure.
- The denominator 5,507 counts *entrants*, 2,939 of whom (53 %) submitted only during the validation phase. The set of teams that actually competed on the test window is ~2,568.
- Only one test window (28 days, one random draw). Findings 1–7 rest on a single origin; the paper cites Tashman (2000) on out-of-sample testing in §3.1 but does not apply rolling origins to the final evaluation.
- No uncertainty quantification on the team scores themselves — Fig. 3's MCB intervals are on mean *ranks*, not on WRMSSE, so no interval is given for "22.4 % better than ES_bu".
- Findings that generalise about "the top 50" rest on 17 disclosed methods.
- The competition cannot separate method quality from engineering effort and submission-selection luck; §3.6 quantifies the latter (7.5 % vs 12.2 %) but does not adjust the rankings for it.

## 10 CONCLUSION AND FUTURE WORK
- §6: "What has remained constant in all five M competitions is the finding that combining improves forecasting accuracy. What has changed is the finding of M1, M2, and M3 that simple statistical methods were more accurate than more complex, sophisticated ones… in the M5 all 50 top-performing methods were ML."
- §6: "LightGBM proved that it can be used to effectively process numerous, correlated series and exogenous/explanatory variables and reduce forecast error."
- Future work (§5.3): verify the ML advantage outside hierarchical retail; cost-benefit of ML vs statistical methods; interpretability — "Managers are typically unwilling to make decisions when they cannot understand the logic of the methods they are going to use. This is a big problem affecting all ML models"; ensemble pruning; "horses for courses" per level; better hierarchical reconciliation.
- **[Assessment]** The interpretability paragraph in §5.3 is a direct, citable motivation for our study: the competition establishes that an ML forecast is more accurate, and simultaneously states that managers will not act on what they cannot understand. That gap — accuracy established, adoption unresolved — is the space our reliance experiment occupies.

## 11 REPRODUCIBILITY
| Item | Status |
|---|---|
| Data | Fully public after the competition (§5.2), including the test actuals — explicitly contrasted with Kaggle competitions where "the test data is not made available after their competition has ended" |
| Metric code | "The code for estimating WRMSSE, as well as the exact weight of each series, can be found in the GitHub repository of the competition" (§3.4) |
| Benchmark code | "publicly available in the M5 GitHub repository" (§2, footnote 6: github.com/Mcompetitions/M5-methods) |
| Winners' code | Required within 14 days for prizes; organiser-verified at >98 % reproducibility rate; uploaded to the M5 GitHub (§3.5) |
| Method descriptions | Only 17 of the top 50 (§4.2, §5.2) |
| Scores | Top 50 teams and 24 benchmarks, per level, published — and supplied to us as `M5_Scores_and_Ranks.xlsx` |
| Full leaderboard | Not published in the paper |
| Evaluation origins | One test window of 28 days |
- **[Assessment]** For an external re-run, this is close to ideal: public train/test splits, published weights, published scorer, and published benchmark scores that a re-implementation can be validated against. Our own `m5_lightgbm.py` exploits exactly that (see §13).

## 12 CRITICAL EVALUATION

**Strengths (max 3)**
1. **The design forecloses the usual objections.** Pre-specified metric, blind test phase with no leaderboard, one selected submission, public data, public weights, public scorer, 24 published benchmarks, and an enforced >98 % reproducibility check on prize-winning code. Very few empirical comparisons in forecasting or ML meet this bar.
2. **The authors report the results that undercut their own headline.** Fig. 3 shows the WRMSSE winner is 13th and the runner-up 47th on per-series ranks; Fig. 2 and §4.1 show the ML gain falls to ~3 % at the three disaggregated levels; Table 4 attributes most of the benchmark spread to seasonality rather than to exogenous variables or ML.
3. **It is usable as infrastructure, not just as a citation.** Because the weights, scorer and benchmark scores are published, an outside model can be placed on the same scale and the placement is checkable by a third party. That is rare and is exactly what our project relies on.

**Weaknesses (max 3)**
1. **A single 28-day test origin.** Every finding rests on one randomly chosen window containing Memorial Day, Ramadan and the NBA finals. The paper itself cites Tashman (2000) on out-of-sample design and uses four-to-five rolling windows for CV in the winning methods' own tuning, but the final evaluation is one origin with no interval on any score.
2. **Generalisation beyond the top 50 is not supported by published data.** Findings 2, 3, 6 and 7 are stated about "the top 50 performing methods" but only 17 disclosed their methods; and scores below rank 50 exist only as a curve in Fig. 1. This limits both what the paper can conclude and what an outside party can verify.
3. **The metric determines the winner, and the paper says so only after the fact.** WRMSSE's dollar weighting made it optimal to concentrate effort on expensive fast-moving products; §4.1 concludes "the 'best' forecasts depend on the accuracy measure used". This is an honest and important finding, but it means the competition ranking answers "who optimised WRMSSE best", not "who forecasts Walmart best" — a distinction that citations of M5 rankings routinely lose.

**Credibility: HIGH** — pre-specified metric, blind final evaluation, public data/weights/scorer, organiser-verified reproducibility of winning code, and self-reported results that cut against the headline; the deductions are one evaluation origin, no intervals on scores, and published detail limited to the top 50.

## 13 RELEVANCE TO MY FOCUS

**Where our project uses this paper**

| Location | How it is used | Verdict |
|---|---|---|
| `paper/build_paper.js:171` — "Demand replays real Walmart M5 store-department series {makridakis} (70 series, 273 weeks, rescaled to a mean of 140 units)." | Data provenance | **Supported.** Level 9 is "Unit sales of all products, aggregated for each store and department", **70 series** (Table 1, p.9) ✓. The daily range 2011-01-29 to 2016-06-19 is 1,969 days ≈ 281 weeks, so 273 complete weeks is consistent. The rescaling to mean 140 is ours, not M5's, and the text already presents it that way. |
| `paper/build_paper.js:206` (Table 1 row) — "Walmart M5 {makridakis} \| Official private leaderboard \| WRMSSE 0.569 \| ≈ 39th of 5,507 teams" | External validation of the AI component | **Supported on the numbers, needs one qualifier** — see the detailed check below. |
| `paper/build_supplement.js:55–58` (Fig S10) — "Our LightGBM (one model per store, Tweedie loss) scores WRMSSE 0.5691, between the 38th (0.5683) and 39th (0.5698) of 5,507 teams, and 15.2 % better than the best statistical benchmark (ES_bu). Our scorer reproduces the published ES_bu and seasonal naïve scores exactly." | Supplementary figure caption | **Fully supported and exactly verifiable** — see below. |
| `lightgbm-pivot/OVERVIEW.md:15` — "WRMSSE **0.5691 = equivalent to 39th of 5,507 teams** (15.2 % better than the best statistical benchmark; winner 22.4 %)" | Headline claim | **Supported on the numbers**; "equivalent to" is the right hedge and should be kept. |
| `paper/NUMBERS.md:8, 33` | Provenance rows pointing at `m5_results.json` and the xlsx | **Correct** |
| `RESULTS.md` (many) | M5 as the demand source for the twin | **Supported**; RESULTS.md:454–455 already discloses that M5 came from a public Kaggle **mirror**, which is the right disclosure. |

**Detailed verification of the "≈ 39th of 5,507 teams" claim**

*Numbers — all check out.*

| Quantity | Our claim | Source of truth | Match |
|---|---|---|---|
| Our WRMSSE | 0.5691 (`m5_results.json`: 0.5690762943) | — | — |
| 38th place | 0.5683 | xlsx `Accuracy-Top50 (AL)`, row 38 = **jdsc_perceiving_team, 0.56830** (Table 3, p.17 prints 0.568) | ✓ exact |
| 39th place | 0.5698 | xlsx row 39 = **Abzal, 0.56980** (Table 3 prints 0.570) | ✓ exact |
| 5,507 teams | 5,507 | §3.6: "involved 7,092 participants on 5,507 teams from 101 countries" | ✓ exact |
| ES_bu | 0.670981 | xlsx `Accuracy-Benchmarks (AL)`, ES_bu Avg = **0.6709809596** | ✓ exact to 6 dp |
| sNaive | 0.847017 | xlsx sNaive Avg = **0.8470173528** | ✓ exact to 6 dp |
| 15.2 % over ES_bu | 0.1518742 | (0.6709810 − 0.5690763)/0.6709810 = **0.151874** | ✓ |
| Winner 22.4 % | 22.4 % | Table 3, p.17, rank 1 improvement column = **22.4** | ✓ |
| ES_bu is the best benchmark | "best statistical benchmark" | §2: "we decided to use ES bu as the single most accurate benchmark"; §6: "more than 14% over the most accurate statistical benchmark". Confirmed against the xlsx: ES_bu 0.671 is the lowest of all 24, ahead of ESX 0.6791, Com_tb 0.6816, ES_td 0.7204, and every ML benchmark (MLP 0.9768, RF 1.0095, gMLP 0.9614, gRF 1.0651) | ✓ — and our wording matches the paper's own §6 phrasing |

*Protocol — faithful.* `lightgbm-pivot/m5_lightgbm.py` trains on d_1…d_1941, forecasts d_1942…d_1969, and scores WRMSSE over all 42,840 series at 12 levels using the organisers' `weights_evaluation.csv` and `sales_test_evaluation.csv`, with a self-check that reproduces ES_bu and sNaive before training. Training on d1–d1941 is exactly what the test phase permitted (§3.1: after 1 June "the teams were provided with the actual values of the 28 days… They were then asked to re-estimate or adjust their forecasting methods"). The script's header states "The test window is never used for any choice", and the code bears that out: early stopping is on d_1914–d_1941, then each model is refitted through d_1941 with that round count.

*Where the claim is weaker than it reads — three qualifiers.*

1. **The denominator is entrants, not competitors.** §3.6: of the 5,507 teams, "4,373 entered the competition during the validation phase and 1,134 during the test phase… 1,434 teams made submissions during both… while 2,939 only during the validation phase." So roughly 2,939 teams (53 %) never submitted in the test phase. A tighter and more defensible reference set is the paper's own: **415 teams (7.5 %) beat ES_bu with their selected submission** (§3.6). Our 0.5691 beats ES_bu, so "inside the 415 teams that beat the best benchmark, at about position 39" is both true and harder to attack than "39th of 5,507".
2. **We did not face the two conditions that decided the leaderboard.** §3.1: the test phase had "no leaderboard, meaning that no feedback was given to the teams about their actual performance", and each team "had to select a single set of forecasts (one submission)". §3.6 quantifies what that cost: 7.5 % beat ES_bu on their *selected* submission versus 12.2 % on their *best*. We selected nothing blind, and the winning recipes had been public on the M5 GitHub since July 2020 (§3.5) — "one LightGBM per store, Tweedie loss, non-recursive" is essentially monsaraida's published 4th-place approach (§4.2). **"≈ 39th" is a score placement, not a competitive result.** The word "equivalent" in `OVERVIEW.md` carries this; the paper's Table 1 uses "≈", which is weaker. Recommend the main-paper row read "≈ 39th-place score" or "score of the 39th of 5,507 teams".
3. **The published table covers only the top 50.** §3.6: "The various tables presented in the remainder of this paper focus on the top 50 performing teams." Ranks 51+ appear only as the unlabelled curve in Fig. 1 (bottom-right, ranks 1–415). Our claim happens to land *inside* the published top 50, so it is fully verifiable — but that is luck, not design, and it is worth saying in the Supplementary Material that the placement is read off the published top-50 table.

*One qualifier the main paper is missing.* `paper/build_paper.js` Table 1 is captioned "Validation of the AI component" and its other rows all concern the **live stockout-risk classifier** (AUROC, calibration). The M5 row is a **different model on a different task**: ten LightGBM regressors, one per store, Tweedie loss, 28-day daily point forecasts. The supplement caption says so; the main paper's Protocol cell ("Official private leaderboard") does not. A reviewer reading Table 1 could take the M5 result as evidence about the deployed risk model. Recommend the Protocol cell read something like "Official private-leaderboard protocol; separate per-store LightGBM forecaster" — a few words that remove the strongest available misreading of our own table.

*A nuance worth knowing if challenged on the M5 row.* Our per-level scores (`m5_results.json`) sit differently against the top-50 distribution at different levels: Level 1 = 0.2557 (near the top-50 median in Fig. 2), but Level 10 = 1.0061, Level 11 = 0.9517 and Level 12 = 0.8978 sit in the upper (worse) tail of the top-50 boxes, close to the ES_bu red dots. That is the same profile the paper attributes to the winners (§4.1: "the winning teams developed methods that mostly focused on expensive and fast-moving products for which WRMSSE is minimized"), so it is a normal consequence of optimising WRMSSE — but if a reviewer asks whether the rank means the model forecasts individual product-store series well, the honest answer is the paper's own: at levels 10–12 the whole top 50 improves on the best benchmark by only about 3 %.

**Other material this paper supports for our paper**
- **The interpretability gap as motivation (§5.3)**: "Managers are typically unwilling to make decisions when they cannot understand the logic of the methods they are going to use. This is a big problem affecting all ML models that would eventually need to be solved." This is a citable statement, from the field's flagship competition, that accuracy alone does not produce adoption — the premise of our study's first sentence. It is currently uncited in our introduction, which cites `{thaker,ntakolia,ivanov}` instead. **Recommendation**: add `{makridakis}` to `build_paper.js:156`.
- **The accuracy-to-cost gap (§5.2)**: "it focused on the point forecast accuracy of the submitted methods, which were not directly linked to Walmart's underlying operational costs… translating forecasting error reduction into cost savings is far from trivial… we hope that the results of the competition will inspire such studies." Our decision-value check (`build_paper.js:210`: the risk-based ordering rule costs 464 less over 52 weeks) is a small instance of exactly the study M5 asks for, and can be framed that way.
- **LightGBM's standing (§4.2)**: "LightGBM can be considered as the standard method of choice in Kaggle's recent forecasting competitions." This is a defensible citation for why we chose LightGBM for the live model, alongside `{ke}` and `{grinsztajn}`.
- **Tweedie loss for intermittent, zero-inflated demand (§4.2)**: the winner used "the negative log-likelihood of the Tweedie distribution… considered an effective approach when dealing with data with a probability mass of zero and non-negative, highly right-skewed distribution." Our `m5_lightgbm.py` uses `objective="tweedie", tweedie_variance_power=1.1`; the choice is the competition's own, and citing §4.2 for it costs nothing.
- **Validation discipline (Finding 6)**: "the top four performing methods and the vast majority of the top 50 submissions considered a CV strategy where at least the last four 28-day-long windows of available data were used" and "many teams failed to select the most accurate set of forecasts… probably due to misleading validation scores." This is a strong external citation for our `RESULTS.md` §6 discipline about scoring the held-out set more than once.

**Citation hygiene**
- Our bibliography cites *Int. J. Forecast.* 38(4), 1346–1364 (2022); the PDF in `papers/` is the October 2020 preprint. The content is substantively the same, but any page-level quotation must use the published pagination, not the preprint's.
- The M5 data we used came from a Kaggle **mirror** (`vaishnavim2411/m5-forecasting-accuracy`), already disclosed in `RESULTS.md:454–455`. The paper states the official data is public (§5.2), so this is a convenience-source note, not a provenance problem — but keep the disclosure.

</analysis>
