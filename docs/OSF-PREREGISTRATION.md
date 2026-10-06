# Pre-registration — Reliance on AI decision support under supply-chain disruption

**Status: DRAFT for the investigator to review and register. Not yet submitted
to OSF. Register before the first participant is run.**

Follows the OSF Preregistration template. Sections map onto the registry's
fields; paste or adapt.

---

## 1. Title

Error placement, not error rate: reliance on AI decision support under
supply-chain disruption.

## 2. Authors

*To complete before registration.*

## 3. Description

Participants manage a single-echelon warehouse for 30 simulated weeks, deciding
an order quantity each week while a forecasting assistant offers a
recommendation. The assistant's reliability is manipulated between subjects
(80 % vs 60 % correct recommendations). Where its errors fall is manipulated
within subjects: errors are scheduled either at **absorbed** positions (weeks
where one bad order is measurably absorbed) or **propagating** positions (weeks
where it measurably cascades, mostly in the run-up to a disruption), holding
the error rate constant. Every scripted error is an under-order.

The study asks whether reliance responds to *where* an aid errs, not only to
*how often* — a manipulation the reliability literature has not tested in a
dynamic system where a decision's consequences persist across later periods.

## 4. Hypotheses

**H1 (reliance and reliability).** Weight of Advice is higher in the 80 %
condition than in the 60 % condition.

**H2 (trust trajectory).** Trust in the assistant diverges between reliability
conditions across blocks; the Reliability × Block interaction is non-zero.

> H2 is registered as **at risk**. A recent study of dynamically reliable
> automation (n = 80) found trust and self-confidence largely insensitive to
> reliability changes. Block-level trust measurement is registered specifically
> so that a null result is informative rather than uninterpretable.

**H3 (error placement).** Errors at propagating positions reduce subsequent
reliance more than errors at absorbed positions, at equal error rate and equal
direction (all under-orders).

*Positions (design v2, fixed in `src/interface/advice.py`).* PROPAGATING weeks
5, 7, 8, 9, 14, 15, 16, 19; ABSORBED weeks 2, 3, 10, 11, 12, 20, 22, 25;
every other week is never an error. Chosen by measured consequence on the
participants' own demand path (`src/placement_design.py`): selected on
lead-time draws 0–99, validated on 100–299, where one accepted propagating
error costs +134 vs −62 and causes 3.45× the extra stockout weeks (1.78× for a
noisy follower). Each participant's errors split evenly between the two sets
(80 %: 3 + 3; 60 %: 6 + 6). Errors are 90–120 units below Q\*, never below 10
units; when Q\* is too small for that, the error is an over-order, logged.

**H4 (operational consequence).** Decision error |Q_final − Q*| and realised
cost are higher in the 60 % condition, and the gap widens following
propagating-position errors.

**Exploratory, not confirmatory:** RAIR and RSR decomposition; periods-to-recovery
after each disruption; NASA-TLX workload; response times; the relationship
between perceived and actual reliability.

## 5. Design

**Mixed 2 × 2.**

| Factor | Type | Levels |
|---|---|---|
| AI reliability | between subjects | 80 %, 60 % |
| Error placement | within subjects | absorbed, propagating |

Assignment to reliability condition alternates by arrival order, which keeps
the cells balanced more reliably than random assignment at this sample size.

**Trials.** 5 practice + 30 main. The disruption calendar is **fixed and
identical for every participant** (disruptions at weeks 6, 9, 13, 17, 21, 26).
Inventory and cost accumulate from each participant's own orders and therefore
diverge between participants; this is intended, and is why the correct order
quantity is computed live from each participant's state rather than
pre-generated.

**The error schedule is exact, not sampled.** At 30 trials, the 80 % condition
delivers exactly 6 errors and the 60 % condition exactly 12. Sampling each trial
independently would let realised reliability range roughly 65–93 %, which is
wider than the gap between the two conditions.

## 6. Two-stage elicitation

Each trial elicits the participant's order **twice**:

1. **Q_own** — their own order, before any advice is shown. The field locks on
   submission and cannot be revised.
2. **Q_final** — after the assistant's recommendation, risk estimate and
   confidence level are revealed.

Without Q_own, Weight of Advice and the RAIR/RSR family are not computable and
"modify" responses have no measure. Q_own also provides a within-subject
unaided-performance baseline, recovering a no-AI control at no recruitment cost.

## 7. Sample

| | |
|---|---|
| Recruited | 80 |
| Target analysed | 70 (35 per reliability cell) |
| Population | Volunteers from the university's SONA participant pool, numeracy screener (below) |
| Payment | None: participation is voluntary and unpaid (see open question 17 on course credit) |
| Session | 45–60 min |
| Expected observations | ~2,100 trial-level; ~630 error trials |

**Recruitment.** Participants are recruited through the university's SONA
participant pool. Taking part is voluntary and unpaid. The SONA study links to
the study URL and passes the participant's SONA identification code as the
participant ID (a participant arriving without one enters a self-chosen ID).
There is no completion code and no payment reconciliation.

**Power** (`src/power_analysis.py`; participant-level approximations of the
mixed models, so conservative). H1: 80 % power at d ≈ 0.7 (d = 0.5 gives 54 %).
H3: 80 % power for a 0.08 drop in after-error Weight of Advice at trial-level
noise SD 0.30, and ≈ 0.10–0.11 at SD 0.40. Smaller effects than these are
unlikely to be detected and would be reported as inconclusive, not as null.

**Recruitment is a single batch.** There is no sequential analysis and no
optional stopping.

**Numeracy screener.** The three-item scale of Schwartz, Woloshin, Black & Welch
(1997), *Annals of Internal Medicine* 127(11), 966–972: die rolls (500), lottery
winners (10), sweepstakes percentage (0.1 %). **Pass mark: 2 of 3 correct.** It is
administered after consent and **before condition assignment**, so a person who
fails is never assigned a condition, never creates a participant record, and does
not shift the arrival-order alternation. Screen-outs are logged anonymously
(score only, no identifier) and their count is reported.

## 8. Exclusion criteria — fixed in advance

A participant is excluded if any of the following holds:

1. the session is not completed through the final trial;
2. **more than 20 % of stage-1 responses are faster than 500 ms**, indicating
   click-through rather than a considered order;
3. Q_own is **identical on every main trial** (a constant-value strategy that
   makes Weight of Advice undefined throughout);
4. the participant withdraws (during the task, at the debrief, or later on
   request), whether or not they ask for their data to be deleted;
5. a technical fault is recorded in the event log that prevented normal display.
   A browser-side script error (`client_error` event) counts as such a fault
   **only if** it occurred during the main task **and** the participant's log
   fails the replay check (the 30 main trials do not reproduce exactly when the
   simulator is re-run from the logged orders). Script errors during consent,
   practice or the questionnaires never exclude; a main-task error whose log
   still replays exactly is **flagged but kept**, and the number of flagged
   participants is reported per condition.

Failing the numeracy screener is **not** an exclusion: it happens before
assignment, so such a person never enters the sample (§7).

Exclusions are recorded in the `excluded` and `exclusion_reason` fields and
reported with counts per condition. The 10–15 % exclusion allowance is why 80
are recruited for 70 analysed.

**Withdrawal and deletion.** Participants can withdraw at any point during the
task, again at the debrief (after the manipulation is disclosed), or later by
message quoting their participant ID. On a deletion request every
trial, questionnaire and event-log row is removed, along with any recruitment
identifier, the numeracy score and the reliability estimate. A bare participant row (ID and
condition only) is kept so that arrival-order alternation is unaffected for
participants still to be run. Withdrawals are reported per condition.

**Trial-level exclusion.** Individual trials where Q_AI equals Q_own are dropped
from Weight-of-Advice analyses only, because WoA is undefined when the
denominator is zero. These trials are retained for all other measures and their
count is reported.

## 9. Measures

**Primary.**

| Measure | Definition | When |
|---|---|---|
| Weight of Advice | (Q_final − Q_own) / (Q_AI − Q_own) | every trial |
| RAIR / RSR | Schemmer et al. appropriate-reliance decomposition | every trial |
| Decision error | \|Q_final − Q*\| | every trial |

**Secondary and exploratory.** Trust in automation (Jian et al. / Körber TiA,
administered per block of 10, three administrations); NASA-TLX workload per
block; response time for both stages, every trial; stockouts, service level,
holding + shortage cost; periods-to-recovery after each disruption; post-hoc
perceived reliability as the manipulation check.

Designating primaries is deliberate: an undifferentiated list of seventeen
dependent variables invites a multiplicity objection.

## 10. Analysis plan

Trials are nested within participants, so all trial-level models are
mixed-effects with participant random intercepts (random slopes on trial where
they converge).

```
# binary reliance outcome
logit P(AR_ij = 1) = b0 + b1*Reliability_i + b2*Block_j
                        + b3*(Reliability x Block) + u_i

# continuous advice taking
WoA_ij = b0 + b1*Reliability_i + b2*AI_correct_ij + b3*Block_j + u_i

# error placement (within subjects)
WoA_ij = b0 + b1*Placement_ij + b2*Reliability_i
              + b3*(Placement x Reliability) + u_i
```

Trial enters as **block** rather than a linear term, so reliance trajectories can
be read across three intervals.

Operational outcomes are continuous and autocorrelated within participant: use an
AR(1) residual structure, or aggregate to block level.

**Multiplicity.** Holm correction across the primary outcomes. Effect sizes
reported with confidence intervals, not p-values alone.

**Software.** R `lme4` / `glmmTMB`, or Python `statsmodels`.

## 11. Reliability level anchoring

Registered wording, to be reported verbatim in the paper:

> "Recommendations were scripted to be correct on exactly 24 of 30 trials (80 %)
> or 18 of 30 (60 %), where correct means within ±δ of the reference order. The
> 80 % level was set below the measured performance of a validated stockout-risk
> model (LightGBM; AUROC 0.891, 95 % series-clustered CI 0.874–0.906, on 70
> held-out demand series; accuracy 86.2 % against a 78.5 % majority-class
> baseline), so the high-reliability condition represents a realistic,
> conservatively calibrated deployment. The 60 % condition represents a plausibly
> degraded one."

**Why the anchor is worded this way.** The model predicts *stockout risk*, while
the manipulation concerns *order recommendations* falling within ±δ of Q\*.
These are different targets, so the paper must not claim the model "is" 80 %
accurate. The claim is only that the 80 % condition does not overstate what a
validated model achieves. **Do not anchor to raw accuracy alone:** the
majority-class floor on the stockout task is 78.5 %.

**The model participants see.** The displayed stockout-risk percentage is live
output of a LightGBM model (`src/live_model.py`) that runs on each participant's
own warehouse history every week. It is restricted to features available within
a session (current state plus at most 4 weeks of history), because the
benchmarked model's 48-week features do not exist in a 35-week session.
Cross-validated by series: AUROC 0.897, ECE 0.006. On the two demand series the
sessions replay, which are excluded from training: AUROC 0.902. Recommended
orders stay scripted, so the reliability levels are exact. On an error week the
risk is computed on the state the AI would have to believe to recommend that
order, so risk and recommendation agree.

**Demand.** Every participant replays the same real M5 demand window (series 0,
weeks 132–171; practice uses series 43), fixed by seed in
`src/interface/world.py`. Before app version 1.2.0, sessions silently used the
synthetic demand process. No participant data exists from those versions.

**External validation of the model**, reported in the paper's method section:
under the official M5 Accuracy protocol (Makridakis et al., 2022) the same
LightGBM approach scores WRMSSE 0.569, equivalent to 39th of 5,507 teams; on the
Ntakolia et al. (2021) backorder benchmark it improves on a faithful reproduction
of the published model on 5 of 5 splits.

## 12. Known departures from the original research plan

Registered here so they are disclosed before results, not after:

1. Lead-time delay distribution fitted from realised purchase-to-delivery
   dispersion, not from actual-minus-estimated timestamps (91.9 % of orders in
   the source data arrive early; the original measure captures forecast padding,
   not delivery variability).
2. The correctness tolerance δ is set by empirical grid search, not from the
   analytical cost curve, which returns a band so wide (±152) that almost any
   order scores as correct.
3. The plan's reference model, PatchTST, was replaced. Benchmarked against nine
   other architectures on the simulator, it did not outperform simpler models
   (all within bootstrap error; PatchTST AUROC 0.859 was the lowest of ten), and
   a single net-inventory-position feature matched it. The reliability anchor is
   a LightGBM gradient-boosted model instead (§11). The PatchTST work is archived,
   not deleted.
4. The simulator's inventory accounting was reimplemented rather than taken from
   OR-Gym, which pins `gym` versions that do not install on Python 3.13.
5. Demand replays **real M5 sequences** rather than a fitted negative binomial;
   the synthetic process was effectively memoryless (lag-4 autocorrelation −0.16
   vs +0.74 in real data).
6. M5 training data was obtained from a public Kaggle mirror because the
   competition download requires accepting rules. The hidden test set and
   scoring weights come from the organisers' own release, and the scorer
   reproduces the organisers' published benchmark scores exactly (ES_bu
   0.670981, seasonal naïve 0.847017), which also indicates that the mirror
   matches the official data. Re-downloading the official training file before
   submission is still advisable.

## 13. Model-evaluation caveat to report

The model is evaluated by **5-fold cross-validation over M5 series**: each of
the 70 demand series is a test series exactly once and never appears in
training or validation. Bootstrap confidence intervals are **clustered on
series**, the independent unit; clustering on episodes would treat overlapping
windows of the same demand as independent and understate uncertainty (by
roughly 1.2× on PR-AUC and 1.3× on AUROC in the earlier single-split analysis).

The registered claim about model families is one of **equivalence**: logistic
regression (AUROC 0.893), LightGBM (0.891) and XGBoost (0.893) differ by at most
0.002. LightGBM is used because it is the model validated externally (§11), not
because it won on the simulator.

## 14. Data and code availability

Trial-level data, the simulator, the participant interface and the analysis
scripts will be released. The published dataset is the de-identified version;
participant IDs (SONA identification codes or self-chosen IDs) are replaced
by arbitrary study numbers, and the schema's recruitment-ID column
(`prolific_pid`, a name kept from the earlier Prolific plan) is dropped before
release.

## 15. Timeline

AHFE 2027, University of Málaga, 26–30 July 2027. The 500-word abstract is due
**4 December 2026**; camera-ready **5 February 2027**. Notification is rolling —
abstracts are reviewed as received, roughly 10 days on average — so 4 December is
a backstop, not a target.

---

## Before registering — open questions for the investigator

1. **Author list and affiliations** are not filled in.
2. ~~**The power analysis has not been run.**~~ **Done 24 Sep** — §7.
3. **H3's directional prediction** is stated as reduced reliance after
   propagating errors. Confirm this is the intended direction — an equally
   defensible prediction is *increased* reliance, if a costly cascade is read as
   evidence the participant's own judgement was inadequate.
4. **RAIR/RSR**: confirm the exact Schemmer et al. definitions to be used, and
   check the citation. A note in the project's citation record flags that an
   earlier RAIR/RSR reference conflated two separate papers.
5. **§13 caveat**: decide whether this belongs in the pre-registration at all, or
   only in the paper. It concerns the model-evaluation phase already completed,
   not the experiment being registered.
6. ~~**H3 has a confound that must be resolved or registered before collection.**~~
   **Resolved 24 Sep (design v2):** all errors are under-orders and positions are
   chosen by measured consequence (§4, H3). Supervisor to confirm. Original note:
   In the current advice engine, error placement fully determines error
   *direction*: every propagating-position error is an under-order and almost
   every absorbed-position error an over-order (486/486 and 539/593 across 120
   simulated sessions). A placement effect could therefore be a direction
   effect. **Re-measured on the participants' real M5 demand**
   (`src/placement_check.py`): the current design's contrast is +30 extra cost;
   placement alone, with under-orders in both, is +140; extra stockout weeks
   0.19 vs 0.14 (1.4×). Either counterbalance direction within each placement (changes the
   design: supervisor decision), or register H3 as "propagating under-orders vs
   absorbed over-orders" and name the confound as a limitation.
7. ~~**The displayed risk % is computed from the correct order even on error
   trials**~~ **Resolved 24 Sep 2026 (app 1.2.0):** the risk is now live LightGBM
   output, computed on the AI's believed state; the leak it caused fell from
   +0.113 to +0.004 AUROC (`src/leak_check.py`). Original note:, so on an error trial the risk and the recommended order disagree
   (for example a high risk shown alongside a small order). A real model's risk
   and recommendation would err together. In simulation this cue lifts a simple
   observer's error detection from AUROC 0.544 to 0.657. Decide before
   registering: derive the displayed risk from the recommended order, or keep it
   and register it as an intended cue.
8. **Placeholders in the information sheet** (`src/interface/templates/info.html`,
   highlighted in yellow): ethics reference, retention periods, institution,
   contacts. They must be filled before launch. (The payment placeholder no
   longer applies now that participation is unpaid; the sheet must say so, and
   mention course credit if SONA credit is offered — see question 17.)
9. **Error size and detectability (supervisor decision).** Errors 90–120 units
   off are spotted from the gap to the participant's own order at AUROC 0.92
   (v1: 0.955) and from the order alone at 0.90 (v1: 0.81). Smaller errors
   would be subtler but have weaker consequences; the placement validation used
   100 units. Keep, or shrink and re-validate, before registering.
10. **Order cap.** Sessions allow orders up to 1,500. On a scheduled-correct
    week whose Q\* exceeds that, the AI recommends the cap and the week counts
    as correct, keeping reliability exact; such weeks are identifiable in the log.
11. **Interface v2 (supervisor to confirm).** The participant screen now shows
    trend lines of the participant's own past weeks, a timeline of past weeks
    and stockouts, and labels the AI's band as "risk level" (previously
    mislabelled "confidence"). Identical across conditions; nothing about
    future weeks is shown; keeping and taking advice take the same effort.
12. **H3 needs an `ai_correct` covariate (found 25 Sept by running the analysis on simulated participants).** Propagating weeks cluster (7-8-9, 14-15-16), so the trial after a propagating error is itself an AI error far more often than the trial after an absorbed one (57 % vs 27 % at 60 %; 19 % vs 7 % at 80 %). Without adjustment, part of the H3 contrast is "the next advice was wrong", not "the last error propagated". Recommended: register `WoA_post ~ Placement × Reliability + AI_correct + (1 | participant)` as the confirmatory H3 model. `src/analysis/confirmatory.py` already reports this as a sensitivity model.
13. **Direction of the binary appropriate-reliance (AR) model.** The share of trials where the initial order is right and the AI is wrong (the IA case) differs by condition (about 25 % at 80 %, about 55 % at 60 %), so the reliability term partly reflects this case mix. In null simulated data the estimate falls from +0.36 to +0.04 once case type is controlled. Recommended: register the model with a case-type term (initial wrong and AI right, CA, vs IA) and state that it is two-sided.
14. **Holm family.** Confirm whether H2 (trust) belongs to the corrected family. The scripts currently correct across H1, H2, H3, H4a, H4b and AR, which is the more conservative choice.
15. **Fast-response exclusion.** "More than 20 % of stage-1 responses faster than 500 ms" is currently applied to all 35 responses (practice + main). The alternative reading is main trials only. Decide and state it.
16. **Interface v3 (added 28 Sept).** The task page gains a 3D warehouse view, "?" definitions for each number, and a guided tour in practice week 1. The 3D view pictures only the numbers already on screen: stock, the on-order total, unmet orders, and supplier and transport status. It shows no delivery timing, Q* or AI information, and it is identical in both conditions. Decide whether to run the study with it (default) or with the plain panel (`WAREHOUSE_3D=0`), and state the choice in *Materials*. Each session logs its view in `session_start.ui`. If it is adopted, the pilot (13 Oct) must use it. **Update 28 Sept (v3.1):**
- The pages gain Motion transitions. Every step card, human or AI, gets the same entrance of at most 200 ms; the AI numbers are never animated.
- The 3D view gains ambient motion. It is data-independent and identical in both conditions.
- A truck now drives in or out depending on whether stock is on order.
- See RESULTS §14b.

17. **Unpaid volunteer recruitment through SONA (decided 28 Sept).** Recruitment
    moved from Prolific (paid) to the university's SONA participant pool, with
    voluntary, unpaid participation. Hypotheses and analysis are unchanged, but
    the decision has consequences to settle before registering:
    - **Drop-out.** Unpaid volunteers are more likely to abandon a 45–60 min
      task part-way, so exclusion 1 ("session not completed", §8) will fire more
      often than on a paid panel. The 80-recruited / 70-analysed allowance
      (10–15 %) was set for a paid panel and may need raising.
    - **Sample.** A student pool is younger and more homogeneous than an online
      adult panel (age, education, numeracy). Name this as an external-validity
      limitation alongside the non-expert population.
    - **Course credit.** Confirm whether SONA course credit is offered. If it
      is, "unpaid" must say so ("no payment; course credit as per the pool's
      rules") in §7, the information sheet and the paper.
    - **Recommendation:** decide with the supervisor (a) the recruitment target
      (keep 80, or recruit until 70 complete, stated in advance, which still
      counts as a single batch with no optional stopping on results) and
      (b) whether a shorter session is needed to hold completion up (any
      shortening changes §5 and must be registered).
