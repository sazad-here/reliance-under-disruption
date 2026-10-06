# Review — Thaker et al. (2026), IEEE CCGE

File: `AI-Driven_Predictive_Analytics_Framework_for_Proactive_Supply_Chain_Risk_Identification_and_Mitigation.pdf` (7 pp.)
"AI-Driven Predictive Analytics Framework for Proactive Supply Chain Risk Identification and Mitigation." D.J. Thaker, J. Khengar, P. Das, P.M. Agarwal, M. Kannan, M.B. Begum.
*2026 2nd International Conference on Computing, Communication and Green Engineering (CCGE)*, Pune, India, 7–8 May 2026. IEEE. DOI 10.1109/CCGE67142.2026.11581786.

---

<scratchpad>

**Research question:** none is stated. No question, no hypothesis, no objective list.

**Main claims (verbatim):**
- Abstract: "a predictive-prescriptive architecture that is proposal-focused, uncertainty-aware, and proactive in predicting the presence of supply chain risks and mitigating them is suggested."
- Abstract: "Comparative performance against SVM, CNN, LSTM and transformer baselines shows it to perform better on a range of metrics including prediction accuracy (91.6% at 6 hours), lower reduced mean squared error (0.109-0.212), higher lead time in early risk detection (up to 14.7 hours) and less mitigation latency (less than 185 ms)."
- Abstract: "The strength and the value of each element are statistically validated and ablated."
- Abstract: "Findings on the real-life supply chain data also support the applicability and scalability of the framework."
- §I: three novelty claims — "(i) the introduction of a choice conscious learning system that links predictive modelling and mitigation outcomes, (ii) formulation of an uncertainty calibrated risk severity system … and (iii) a co-located optimization mechanism that links probabilistic risk predictions with constrained mitigation strategies."
- §IV.A: "The suggested PatchTST-based framework achieved perfect success in all the horizons compared to all base models."
- §IV.C: "The outcomes indicate that there is a statistically significant increase in the prediction accuracy and RMSE and all p-values are below 0.01."

**Figures:**
- Fig 1 (p. 3) "AI-Driven Architecture for Proactive Supply Chain Risk Management"
- Fig 2 (p. 4) "Working of PatchTST-Based Supply Chain Risk Prediction"
- Fig 3 (p. 5) "Comparison of Risk Prediction Accuracy Across Different Prediction Horizons"
- Fig 4 (p. 5) "RMSE Performance of Risk Probability Prediction Models at Varying Time Horizons"
- Fig 5 (p. 5) "Early Risk Detection Lead Time Analysis for Different Predictive Models"
- Fig 6 (p. 6) "Risk Severity Index (RSI) Estimation Error Comparison Across Prediction Horizons"
- Fig 7 (p. 6) "Average Mitigation Decision Latency Under Increasing Prediction Horizons"

**Tables:**
- TABLE I (p. 6) "ABLATION STUDY RESULTS" — the only table in the paper.

**Equations:** 32, numbered (1)–(32). All are textbook standard (min–max scaling, z-score, scaled dot-product attention, FFN, MC-dropout mean/variance, cross-entropy, Brier-style calibration term). None is derived, proved, or novel.

**Numbers appearing more than once, cross-checked:**

| Number | Where | Agrees? |
|---|---|---|
| Accuracy 91.6 % @ 6 h | Abstract, §IV.A, Table I, §V | ✅ |
| Accuracy 81.9 % @ 72 h | §IV.A, §V | ✅ |
| Transformer 87.9 / LSTM 84.7 @ 6 h | §IV.A only | — (not in any table) |
| Transformer 74.6 / LSTM 66.8 @ 72 h | §IV.A only | — |
| **RMSE @ 6 h** | Abstract "0.109-0.212"; §V "RMSE of 0.109 at 6 hours"; **Table I "0.146"**; §IV.A "RMSE at the 24-hours horizon being 0.146" | ❌ Table I's Full-Model row pairs 6-hour accuracy (91.6) with a 24-hour RMSE (0.146). Fig 4's proposed bar at 6 h reads ≈ 0.11 |
| RMSE 0.212 @ 72 h | Abstract, §IV.A, §V | ✅ |
| Lead time 14.7 h @ 72 h | Abstract ("up to 14.7"), §IV.A, Table I, §V | ⚠️ Table I lists 14.7 in the *Full Model* row alongside 6-hour accuracy and 6-hour latency; the value belongs to the 72-hour horizon |
| Lead time 6.4 h @ 12 h | §IV.A only | — |
| RSI error 7.8 % @ 6 h | §IV.A, Table I | ✅ |
| Latency 128 ms @ 6 h | §IV.A, Table I | ✅ |
| Latency "less than 185 ms" / "under 184 ms" | Abstract vs §IV.A | ⚠️ 185 vs 184 |
| "6-9 percent" accuracy drop without PatchTST | §IV.B vs Table I (91.6 → 84.2 = 7.4 points @ the tabulated horizon) | ⚠️ consistent in magnitude; "percent" used for percentage points |
| "4-6 percent" RSI-error rise without uncertainty modelling | §IV.B vs Table I (7.8 → 13.4 = +5.6) | ✅ |

**Internal contradictions:**
1. **Table I mixes horizons in one row.** Accuracy 91.6 (6 h), RMSE 0.146 (24 h per §IV.A), lead time 14.7 h (72 h per §IV.A), RSI error 7.8 (6 h), latency 128 ms (6 h). The row is presented as one operating point of one model.
2. **§V:** "it had a prediction accuracy of risk of 91.6 percent at 6-hour horizon and gracefully reducing to 81.9 percent at 72 hours, **which was considerably lower than all baselines**" — the paper's own claim is that it is *higher* than all baselines.
3. **Abstract:** "lower reduced mean squared error" — malformed; also gives an RMSE range (0.109–0.212) that Table I's single RMSE (0.146) does not sit at either end of.
4. **§IV.A:** "achieved perfect success in all the horizons" — "perfect" is not what the numbers show.
5. **Abstract:** "The strength and the value of each element are statistically validated and ablated." §IV.C reports p < 0.01 from "paired t-tests" with no test statistic, no degrees of freedom, no n, no table, and no correction for 5 metrics × 4 baselines × 5 horizons.

**Not stated anywhere in the paper (checked all 7 pages):**
- **The dataset.** No name, no source, no size, no time span, no number of series, no variables. The only reference is the abstract's "real-life supply chain data".
- Train/validation/test split, or any evaluation protocol.
- What the prediction target is, concretely (a "disruption" is never operationally defined; risk categories are named as "supply, demand, operational and logistics" in §III.E but never enumerated or counted).
- How "lead time in early risk detection" is measured.
- How "mitigation decision latency" is measured, or on what hardware.
- Any hyperparameter of PatchTST or of any baseline: patch length P, stride, number of layers/heads, d_model, lookback, epochs, optimiser, learning rate, dropout rate, MC-dropout sample count S.
- The values of λ₁, λ₂ (eq. 29) and β₁, β₂, β₃ (eq. 25), and δ (eq. 28).
- The mitigation action space (`u_r`) and how the "best mitigation action" `u*_r` is known for eq. (32).
- Code, data, or availability statement of any kind.
- Any limitations section. Any threats-to-validity section. Any future-work section.

**Reference-list observations:**
- 22 references. **PatchTST's own source (Nie et al., ICLR 2023, arXiv:2211.14730) is not cited.** The paper's PatchTST-adjacent citations are [4] MSPatch and [15] AA-PatchTST-RA.
- [16]–[22] are unrelated to supply chains: IoT consumer behaviour [16], sensor-network encryption [17], microstrip patch antenna design [18], AI for business decision-making [19], predictive analytics for innovation [20], nanoelectronic health sensors [21], ad hoc routing protocols [22]. Six of these seven share a co-author (M.B. Begum), who is the paper's sixth author. §II concedes the point: "Whereas supply chain analytics is not related to such studies…".
- [9] Belhadi et al. is given as "vol. 333, no. 2, Feb. 2021"; *Annals of Operations Research* vol. 333 is 2024.

**Language:** the text reads as machine-translated or heavily paraphrase-tooled throughout — e.g. "deep learning Mongolisms" (§I), "decision dubbed intelligence" (Abstract), "environmental unpredict abilities" (§I), "An almost empty priority threshold specified in (28)" (§III.E), "elect to learn long-range relationships" (§III). This affects readability of every substantive sentence.

</scratchpad>

---

<analysis>

## 1 SNAPSHOT

- **Paper:** Thaker, Khengar, Das, Agarwal, Kannan, Begum. "AI-Driven Predictive Analytics Framework for Proactive Supply Chain Risk Identification and Mitigation." IEEE CCGE 2026, Pune, 7–8 May 2026. DOI 10.1109/CCGE67142.2026.11581786.
- **Problem:** predict supply-chain disruption risk ahead of time and convert the prediction into a constrained mitigation decision.
- **Proposed method:** multisource ingestion → preprocessing/feature engineering → **PatchTST** severity/probability model → Monte-Carlo-dropout uncertainty → Risk Severity Index (RSI = p̂ × impact) → a composite loss `L = L_pred + λ₁L_cal + λ₂L_dec` where `L_dec` penalises wrong mitigation actions weighted by RSI.
- **Data:** **none identified.** The abstract's "real-life supply chain data" is the only reference to a dataset in 7 pages.
- **Headline:** 91.6 % accuracy at 6 h (vs Transformer 87.9, LSTM 84.7); RMSE 0.109–0.212; lead time up to 14.7 h; latency < 185 ms; ablation in Table I.
- **Code:** none.
- [Assessment] The architectural idea — a decision-aware loss term weighted by predicted severity — is legitimate and is the paper's one real contribution. Everything offered as evidence for it is unverifiable.

## 2 ABSTRACT

- **Claims:** a predictive-prescriptive, uncertainty-aware framework; better than SVM/CNN/LSTM/Transformer on five metrics; components "statistically validated and ablated"; "Findings on the real-life supply chain data also support the applicability and scalability of the framework."
- **Evidence:** five point estimates, no dataset, no baseline configuration, no uncertainty, no protocol.
- **Internal problems:** "lower reduced mean squared error (0.109-0.212)" is malformed and the range does not include Table I's 0.146. "moving above the predictive analytics to decision dubbed intelligence" is unparseable.
- [Assessment] The abstract asserts empirical superiority whose subject (the data) is never named. On its own this makes every performance claim in the paper non-evaluable — not wrong, but untestable.

## 3 INTRODUCTION AND MOTIVATION

- **Claim (§I):** conventional SCRM is reactive, threshold-based and short-horizon; LSTM/GRU "have difficulty in capturing time long range dependencies"; transformers help but their supply-chain use "has been predominantly limited to demand forecasting and risk severity and mitigation decisions have not necessarily been considered [3], [4]".
- **Gap (§I):** "the majority of predictive models are optimized in such a way that they do not consider the downstream effect of the decisions made using such predictions."
- **Novelty (§I):** three items, quoted in the scratchpad.
- [Assessment] The gap statement is the strongest passage in the paper and is genuinely the right gap — it is the same argument Li (2026) makes empirically. But the paper states no research question, no hypothesis and no objective, so there is nothing against which its own results can be judged as supporting or not. The introduction also contains the untranslated fragment "machine learning and deep learning Mongolisms", which is symptomatic of the paper's production quality.

## 4 LITERATURE REVIEW / RELATED WORK

- **Coverage (§II):** 15 topical references — AI-in-SCM surveys [1, 2], transformer forecasting [3, 4], predictive analytics for mitigation [5], multivariate predictability [6], resilience reviews [7, 9], MCDM risk assessment [8], attention ST-GCN for ergonomics [10], data-driven SC orientation [11], unsupervised pharma planning [12], edge analytics [13], Industry 5.0 [14], AA-PatchTST-RA [15].
- **Quality of engagement:** each is given one or two sentences with a stated weakness, which is the right shape. e.g. on [2]: "the research is more descriptive and bibliometric and does not exhibit precise predictive models"; on [9]: "their work is based on empirical studies carried out through surveys".
- **Gap claim:** "This clearly indicates a research gap in terms of an integrated AI-based solution integrating long-horizon predictive analytics, direct quantification of risk severity, and optimization-based reduction."
- **Problems:**
  1. **PatchTST is the paper's core component and its source paper is never cited.** Nie et al. (2023) does not appear. The nearest citations are [4] (MSPatch, a different patch-mixing architecture) and [15] (AA-PatchTST-RA, which already "incorporates adaptive fusion and risk-aware loss to make decision-aware predictions" — i.e. the closest prior work to this paper's own contribution, dismissed in one sentence with no comparison).
  2. **Seven of 22 references ([16]–[22]) are off-topic filler**, six sharing a co-author, covering antenna design, ad hoc routing and nanoelectronic sensors. §II admits they are unrelated and justifies them with "they highlight the significance of trusted data collection" — which is not a justification.
- [Assessment] The topical two-thirds of the review is competent. The citation of [15] without comparison is the substantive problem: if AA-PatchTST-RA already does risk-aware loss on PatchTST, the paper's novelty claim (i) needs a differentiation argument, and none is offered.

## 5 METHODOLOGY

**§III.A Data acquisition.** State vector `X(t) = [x₁(t),…,x_M(t)]` (eq. 1); temporal alignment operator `𝒯` (eq. 2); batch/stream fusion `X_f = αX_b + (1−α)X_r` (eq. 3). `M`, the operator `𝒯`, and `α` are never specified.

**§III.B Preprocessing.** Mean imputation (eq. 4), z-score outlier flagging with threshold τ (eq. 5–6), min–max scaling (eq. 7), supplier reliability `R_s` (eq. 8), demand volatility `V_d` (eq. 9). τ is never given.

**§III.C PatchTST.** Patching (eq. 12), linear embedding + positional encoding (13–14), standard scaled-dot-product multi-head attention (15–17), FFN (18–19), sigmoid output (20). Patch length `P`, stride, number of patches `K`, lookback `T`, layers `L`, heads `H`, `d_model` — **none specified**. Note that eqs. (15)–(19) are the vanilla Transformer; nothing here is specific to PatchTST except eq. (12), and channel-independence and RevIN — the two things that actually define PatchTST — are not mentioned.

**§III.D Uncertainty.** MC dropout: S stochastic passes (eq. 21), mean (22), variance (23). `S` and the dropout rate are not given.

**§III.E Risk classification and severity.** Argmax class (24); impact `I_r = β₁C_r + β₂S_r + β₃L_r` (25); `RSI_r = p̂_r × I_r` (26); normalisation (27); priority set `ℛ = {r | RSI_norm ≥ δ}` (28). β₁₋₃ and δ are not given, and `C_r`, `S_r`, `L_r` (cost escalation, service degradation, lead-time disruption) are never operationally defined or measured.

**§III.F Decision-aware learning.** `L = L_pred + λ₁L_cal + λ₂L_dec` (29); `L_pred` cross-entropy (30); `L_cal` squared error, i.e. a Brier term (31); `L_dec = Σ_r 𝟙(û_r ≠ u*_r)·RSI_r` (32).

**The critical gaps, in order of severity:**

1. **No dataset.** Not named, sourced, sized, dated or described. "Findings on the real-life supply chain data" (Abstract) is the whole of it.
2. **No experimental setup section.** No train/validation/test split, no cross-validation, no temporal ordering, no number of runs, no seeds.
3. **Eq. (32) is not trainable as written.** `𝟙(û_r ≠ u*_r)` is a zero-gradient indicator, and `u*_r` — the ground-truth optimal mitigation action — must come from a labelled action space that is never described. The paper does not say how the indicator is relaxed (no softmax surrogate, no hinge, no straight-through estimator), so the central novelty has no stated implementation.
4. **No hyperparameters for anything**, proposed model or baselines. "SVM, CNN, LSTM, and a typical Transformer" are named and nothing more, so four of five comparison points are undefined.
5. **"Mitigation decision latency" is a headline metric with no definition.** Whether it measures inference time, optimisation solve time, or an end-to-end loop is not stated; neither is the hardware.
6. **"Early risk detection lead time" likewise has no definition.** Fig. 5's y-axis is labelled "Risk Detection Lead Time" with no unit.
7. **No optimisation module is specified** despite "Robust Risk Mitigation Optimization" being a box in Fig. 1 and "Without Optimization" being an ablation row in Table I. No objective, no constraints, no solver.

- [Assessment] The method section is 32 standard equations with every free parameter left unspecified and the one novel equation left unimplementable. Nothing in it can be re-implemented, and the ablation row "Without Optimization" removes a component that was never described.

## 6 RESULTS

**§IV.A** Five metrics × five horizons (6, 12, 24, 48, 72 h) × five models, reported as five charts with selected values quoted in prose:
- Accuracy: 91.6 (6 h) vs Transformer 87.9, LSTM 84.7; 81.9 (72 h) vs 74.6, 66.8.
- RMSE: 0.146 (24 h) vs Transformer 0.182, LSTM 0.204; 0.212 (72 h) vs 0.246, 0.281.
- Lead time: 6.4 h (12 h window) vs 5.1, 4.2; 14.7 h (72 h) vs 11.2, 8.1.
- RSI error: 7.8 % (6 h) vs 9.6, 11.4; 14.8 / 18.5 (48/72 h) vs Transformer 17.9 / 21.2.
- Latency: 128 ms (6 h) vs 143, 151; "under 184 ms" at 72 h.

**§IV.B** Ablation (Table I): removing PatchTST costs 7.4 accuracy points; removing uncertainty modelling costs 5.6 RSI-error points; removing optimisation costs 23 ms latency and 3.0 h lead time.

**§IV.C** "statistical significance testing was conducted with paired t-tests … all p-values are below 0.01."

**What the evidence supports:**
- Nothing is independently checkable. With no dataset, no protocol and no baseline configuration, the comparison "PatchTST beats SVM/CNN/LSTM/Transformer" is an assertion about an unnamed experiment.
- The ablation is *internally* consistent in direction (every removal degrades every metric), which is itself worth noting as a mild warning sign: a genuine ablation usually produces at least one trade-off.
- **§IV.C is not a statistical analysis.** No test statistic, no df, no n, no per-comparison p-values, no multiplicity correction over ~100 implied comparisons, and the claim "a statistically significant increase in the prediction accuracy **and RMSE**" is backwards (RMSE should decrease).
- **Table I's Full-Model row is not one operating point.** Its five cells come from at least three different horizons (see §7 below). The ablation rows are therefore not comparable to it cell-by-cell either, unless the same mixing was applied consistently — which is not stated.

- [Assessment] The results section reports a large, tidy, monotone win on every metric at every horizon with no variance, no confidence intervals, no failure case and no trade-off. Taken with the missing dataset, the pattern is what one would expect from illustrative rather than measured numbers. That is a strong statement and I make it as [Assessment], not as a finding of fact: the paper provides no evidence either way, which is precisely the problem.

## 7 FIGURES AND TABLES (each one, in order)

- **Fig. 1 (p. 3) AI-Driven Architecture.** Six coloured boxes in a 2×3 grid with arrows: Multisource Data Acquisition → Data Preprocessing & Feature Engineering → PatchTST-Based Predictive Risk Modeling → Uncertainty-Aware Risk Prediction → Risk Classification & Severity Quantification → Decision-Aware Risk Learning → Robust Risk Mitigation Optimization → Closed-Loop Adaptation & Stability. Each box lists three bullet keywords. Legible. Carries no data flow, no dimensions, no interfaces — it is a contents page for §III rendered as a diagram. "Closed-Loop Adaptation & Stability" appears only here and is never described in text.
- **Fig. 2 (p. 4) Working of PatchTST-Based Supply Chain Risk Prediction.** Referenced in §III.C. The page-4 render places it in the right column; it is a schematic of the patch→embed→attention→FFN→sigmoid path already given as eqs. (11)–(20). Adds no parameter values.
- **Fig. 3 (p. 5) Risk Prediction Accuracy vs horizon.** Grouped bars, 5 models × 5 horizons, y-axis 0–100 %. Proposed is highest at every horizon; all models decline monotonically with horizon. No error bars. The 6-hour proposed bar reads ≈ 90, consistent with 91.6.
- **Fig. 4 (p. 5) RMSE vs horizon.** Grouped bars, y-axis 0–0.4. **The proposed bar at 6 h reads ≈ 0.11**, matching the conclusion's "RMSE of 0.109 at 6 hours" and **not** Table I's 0.146. At 24 h the proposed bar reads ≈ 0.145, matching §IV.A's 0.146. This confirms Table I's RMSE cell is the 24-hour value sitting in a row whose other cells are 6-hour and 72-hour values. No error bars. The series colours differ from Fig. 3's for the same five models.
- **Fig. 5 (p. 5) Early Risk Detection Lead Time.** Grouped bars, y-axis 0–20, **no unit on the axis** (hours, per the text). Lead time *increases* with prediction horizon for every model — which is coherent only if "lead time" is defined relative to the horizon, and that definition is never given.
- **Fig. 6 (p. 6) RSI Estimation Error.** Line chart, 5 series, y-axis 0–35 %. All five lines rise monotonically with horizon; proposed is lowest throughout; the ordering SVM > CNN > LSTM > Transformer > Proposed is perfectly preserved at every one of the five horizons with no crossings.
- **Fig. 7 (p. 6) Average Mitigation Decision Latency.** Line chart, y-axis 0–300 ms. Again five non-crossing monotone lines with the proposed lowest throughout. **No figure in the paper shows a crossing, a tie, or an error bar.**
- **TABLE I (p. 6) Ablation study results.** Four rows × five metrics:

| Model Variant | Accuracy (%) | RMSE | Lead Time (hrs) | RSI Error (%) | Latency (ms) |
|---|---|---|---|---|---|
| Full Model (Proposed) | 91.6 | 0.146 | 14.7 | 7.8 | 128 |
| Without PatchTST | 84.2 | 0.198 | 10.9 | 12.6 | 141 |
| Without Uncertainty Modeling | 87.3 | 0.172 | 12.8 | 13.4 | 135 |
| Without Optimization | 88.1 | 0.165 | 11.7 | 10.9 | 151 |

  **Problems:** (a) the horizon is not stated for any row, and cross-referencing §IV.A shows the Full Model row mixes 6-hour accuracy/RSI/latency with a 24-hour RMSE and a 72-hour lead time; (b) no variance, no repeats, no CI; (c) "Without Optimization" removes a component the paper never specifies; (d) removing uncertainty modelling *worsens RSI error more than removing PatchTST* (13.4 vs 12.6) while *improving* accuracy relative to it — internally plausible, but the text (§IV.B) claims "The biggest degradation is found in the removal of PatchTST", which is true for accuracy/RMSE/lead time/latency but false for RSI error.

## 8 DISCUSSION

There is no discussion section. §IV.A closes with a two-sentence summary ("the findings indicate that the suggested framework performs better on all measures considered. The technical superiority of PatchTST to recurrent and conventional transformer models is verified…"), and §IV.B/§IV.C are ablation and significance.

- **Not discussed:** why PatchTST should help on this task specifically; what the uncertainty estimates are *used for* operationally beyond entering RSI; whether the decision loss changed the decisions or only the predictions; any comparison to [15] (AA-PatchTST-RA), the paper's nearest prior work; any failure mode; any deployment consideration despite "applicability and scalability" being claimed in the abstract.
- [Assessment] The absence of a discussion means the paper never connects its numbers to its three novelty claims. In particular, novelty (i) — the decision-aware learning system — is never isolated: there is no experiment comparing `L_pred` alone against `L_pred + λ₂L_dec` on *decision* outcomes, only an accuracy ablation.

## 9 LIMITATIONS

**There is no limitations section, no threats-to-validity section, and no future-work section.** The paper contains no sentence acknowledging any constraint on its results.

- [Assessment] For a paper whose evidence rests on an unnamed dataset, this is the most consequential omission. A single paragraph naming the data and the protocol would change the paper's standing more than any additional result.

## 10 CONCLUSION AND FUTURE WORK

- **§V** restates the framework and four numbers: 91.6 % @ 6 h, 81.9 % @ 72 h, RMSE 0.109 @ 6 h and 0.212 @ 72 h, lead time 14.7 h @ 72 h.
- **Inconsistencies carried into the conclusion:** (a) "81.9 percent at 72 hours, **which was considerably lower than all baselines**" — the intended claim is *higher*; (b) RMSE 0.109 @ 6 h contradicts Table I's 0.146.
- **No future work is stated.**
- [Assessment] The conclusion introduces a number (0.109) that appears in no table, and contains a sign error in its main comparative claim.

## 11 REPRODUCIBILITY

| Item | Status |
|---|---|
| Dataset identity | ❌ **Not stated in paper** |
| Dataset size / span / variables | ❌ Not stated in paper |
| Target definition | ❌ Not stated in paper |
| Train/val/test protocol | ❌ Not stated in paper |
| Number of runs / seeds | ❌ Not stated in paper |
| Proposed-model hyperparameters | ❌ Not stated in paper (no patch length, lookback, layers, heads, d_model, lr, epochs, dropout, S) |
| Baseline configurations | ❌ Not stated in paper |
| Loss weights λ₁, λ₂; impact weights β₁₋₃; threshold δ, τ, α | ❌ Not stated in paper |
| Mitigation action space / `u*_r` source | ❌ Not stated in paper |
| Metric definitions (lead time, latency) | ❌ Not stated in paper |
| Hardware | ❌ Not stated in paper |
| Statistical test details | ❌ Only "paired t-tests" and "p < 0.01" |
| Code / data availability | ❌ No statement of any kind |
| Uncertainty on any reported number | ❌ None |

- [Assessment] Zero reproducible elements. Not "hard to reproduce" — there is no experiment described that could be attempted.

## 12 CRITICAL EVALUATION

**Strengths (max 3)**
1. **The framing gap is correctly identified.** "The majority of predictive models are optimized in such a way that they do not consider the downstream effect of the decisions made using such predictions" (§I) is the right criticism of the delay-prediction literature, and it is the same gap Li (2026) closes empirically.
2. **The decision-aware loss is a coherent architectural idea.** `L = L_pred + λ₁L_cal + λ₂L_dec` with `L_dec` weighted by predicted severity (eqs. 29, 32) is a sensible way to couple prediction to action, and coupling a *calibration* term to a *decision* term is the right instinct.
3. **The ablation is structured along the paper's three novelty claims** (PatchTST / uncertainty / optimisation), which is the correct design even though the execution is unverifiable.

**Weaknesses (max 3)**
1. **No dataset is ever named, and there is no experimental-setup section.** Every performance number in the abstract, §IV and §V is therefore an assertion about an unidentified experiment. Nothing can be checked, replicated, or bounded.
2. **The central contribution is not implementable from the paper.** Eq. (32) is a zero-gradient indicator over an action space that is never defined, with no stated relaxation; λ₁, λ₂, β₁₋₃ and δ are all unspecified; and the "optimization" component that Table I ablates is never described at all.
3. **The reported numbers do not cohere and no uncertainty is reported anywhere.** Table I's Full-Model row mixes three horizons; RMSE @ 6 h is 0.109 in the conclusion and 0.146 in Table I; §V reverses the sign of its own comparative claim; §IV.C claims p < 0.01 with no test details across ~100 implied comparisons; and no figure or table in the paper carries an error bar, a tie, or a crossing.

**Credibility: LOW.** The framing is sound and the loss formulation is a real idea, but the paper presents no identifiable data, no protocol, no hyperparameters and no uncertainty, and its headline numbers contradict each other across the abstract, Table I and the conclusion.

## 13 RELEVANCE TO MY FOCUS

**How our draft uses this paper:**

| # | Our use | Verdict | Location in Thaker et al. |
|---|---|---|---|
| 1 | `build_paper.js:26` — bibliography: authors, title, "2nd International Conference on Computing, Communication and Green Engineering (CCGE), Pune. IEEE (2026)", DOI 10.1109/CCGE67142.2026.11581786 | **Supported** — all six author surnames, the venue string, the city and the DOI match the paper's own footer exactly. ⚠️ Optionally add the dates (7–8 May 2026) | Title page; p. 1 footer |
| 2 | `build_paper.js:156` — "Supply chains increasingly receive AI-generated risk **estimates and order recommendations** {thaker,ntakolia,ivanov}" | **Partially.** "Risk estimates" is supported (eq. 20, §III.D–E). "Order recommendations" is **not** — the paper's output is a mitigation action `û_r` from an unspecified action space plus resource allocation (§III.E, Fig. 1); it never produces an order quantity or an order recommendation. If {ntakolia} or {ivanov} carries the order-recommendation half of the sentence, this is fine; if the sentence is read distributively, reword | §III.E, eqs. 24–28; Fig. 1 |
| 3 | `build_paper.js:166` — "predictive work covers … deep-learning risk frameworks {thaker,ahmed}" | **Supported** | Abstract; §III |
| 4 | `build_supplement.js:64` (Fig12 caption) — "**Thaker et al. (2026) name no dataset, so no floor can be computed.**" | **Fully supported.** Verified across all 7 pages: no dataset name, source, size, span or availability statement appears anywhere; the abstract's "real-life supply chain data" is the only reference | Abstract, p. 1; absent from §§I–V and the reference list |
| 5 | `RESULTS.md:316` — "compares its PatchTST framework against **SVM, CNN, LSTM, and a vanilla Transformer** and reports PatchTST beating all four on every metric" | **Supported** | §IV opening; Figs. 3–7 |
| 6 | `RESULTS.md:316` — "the paper … cite[d] as **PatchTST's technical reference**" | **Supported as a description of our own earlier framing, and correctly repudiated elsewhere.** The paper itself is *not* a technical reference for PatchTST: it never cites Nie et al. (2023), never mentions channel independence or RevIN, and its eqs. (15)–(19) are the vanilla Transformer. `reliance-under-disruption-PLAN.md:390` already says "Cite for PatchTST itself, not the reference paper" — that ruling is correct and should stay | §III.C; reference list (Nie et al. absent) |
| 7 | `OVERVIEW.md:~129` — "**The reference paper's claim didn't reproduce.** Its PatchTST beat SVM/CNN/LSTM/Transformer; on our data those models all landed within noise of it, and three edged it out." | **Partially — the wording overstates what we did.** The description of the paper's claim is accurate (#5). But "didn't reproduce" implies a reproduction attempt, and none is possible: the paper names no dataset, no split, no hyperparameters. What we actually showed is that *on our corpus* the claimed ordering does not hold. Suggest: "The reference paper's claimed ordering does not hold on our data — and because it names no dataset, split or hyperparameters, its own result cannot be reproduced at all." | §IV; §11 above |
| 8 | `reliance-under-disruption-PLAN.md:16, 110, 406` — "names no dataset, reports no train/test protocol, and lists no hyperparameters, so nothing in it is reproducible"; "Cite for the architectural concept only … do not cite its benchmark numbers as evidence" | **Fully supported** — this is an accurate and well-calibrated characterisation | §§III–V; §11 above |
| 9 | `reliance-under-disruption-PLAN.md:121` — GSCPI supplies "the 'external' channel Thaker et al. claim but never demonstrate" | **Supported.** §III.A claims external disruption signals ("It also allows using external streams of data to detect the disruption preconditions at a tender stage") but no external variable is ever named, sourced or shown | §III.A, eqs. 1–3 |
| 10 | `reliance-under-disruption-PLAN.md:179` — "the reference paper omits it [gradient boosting]" | **Supported** — the baseline set is SVM, CNN, LSTM, Transformer only; no tree ensemble appears | §IV opening |

**No claim in our draft attributes to this paper something it does not say.** The two items to tighten are #2 ("order recommendations") and #7 ("didn't reproduce"), both wording rather than substance.

**Methodology to reuse.**
- **The decision-aware loss idea, cited as a concept only.** `L_dec` weighted by predicted severity (eq. 32) is a reasonable anchor for our argument that prediction quality and decision quality come apart — but cite Li (2026) for the *evidence* and Elkan (2001, via Li [6]) for the *theory*. Thaker et al. supplies the framing sentence, nothing more.
- **Its ablation structure** (one row per claimed novelty) is a template worth following for our own supplement, with the corrections this paper needs: state the horizon, repeat the runs, report intervals, and allow trade-offs to appear.
- **Do not reuse** any number, any baseline configuration, or any metric definition from this paper.

**For the literature review.** Cite once, in the sentence that establishes that supply-chain AI tools now issue risk estimates to planners. It is also the cleanest available exhibit for our accuracy-floor argument (`Fig12_accuracy_vs_floor`): a peer-reviewed IEEE paper reporting 91.6 % accuracy on a dataset it never names, next to Ahmed et al.'s 96 % with no floor and Islam & Amin's 98.9 % against a 96.6 % floor. That three-way contrast is the strongest version of our point and it is fully supported by the papers themselves.

**For our study design.** The paper has no human in the loop, so it bears on our reliance experiment only as an example of the artefact a planner would be handed: a risk probability, an uncertainty estimate, a severity index and a recommended mitigation, with no stated provenance. That is a concrete argument for our design choice to show participants a calibrated risk % whose value is independently recomputable from the served model (`src/interface/verify_log.py`), rather than an opaque severity score.

</analysis>
