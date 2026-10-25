# Review — Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*

File: `lightgbm-pivot/papers/10_Chen2016_KDD_XGBoost.pdf` (13 pages; the PDF is the arXiv v3 of the KDD '16 paper — "arXiv:1603.02754v3 [cs.LG] 10 Jun 2016" in the p.1 margin. Pages 11–13 are the weighted-quantile-sketch appendix.)

<scratchpad>

**Research questions** — Not stated as questions. The stated goal (p.1 §1): "In this paper, we describe XGBoost, a scalable machine learning system for tree boosting." The design problem (p.1–2): existing systems do not address "out-of-core computation, cache-aware and sparsity-aware learning."

**Hypotheses** — Not stated in paper. This is a systems paper; contributions are listed instead (p.2 §1):
- "We design and build a highly scalable end-to-end tree boosting system."
- "We propose a theoretically justified weighted quantile sketch for efficient proposal calculation."
- "We introduce a novel sparsity-aware algorithm for parallel tree learning."
- "We propose an effective cache-aware block structure for out-of-core tree learning."

**Main claims verbatim**
1. "XGBoost scales beyond billions of examples using far fewer resources than existing systems." (Abstract)
2. "The system runs more than ten times faster than existing popular solutions on a single machine and scales to billions of examples in distributed or memory-limited settings." (p.1 §1)
3. "Among the 29 challenge winning solutions published at Kaggle's blog during 2015, 17 solutions used XGBoost … the second most popular method, deep neural nets, was used in 11 solutions." (p.1 §1)
4. "XGBoost was used by every winning team in the top-10" of KDDCup 2015. (p.1 §1)
5. "The algorithm proposed in this paper is the first unified approach to handle all kinds of sparsity patterns." (p.7 §5)
6. "To the best of our knowledge, the weighted quantile sketch proposed in this paper is the first method to solve this problem." (p.7 §5)
7. "Our final method is able to process 1.7 billion examples on a single machine." (p.9 §6.5)

**Every figure and table**
- Figure 1 (p.2): "Tree Ensemble Model. The final prediction for a given example is the sum of predictions from each tree." Schematic.
- Figure 2 (p.3): "Structure Score Calculation. We only need to sum up the gradient and second order gradient statistics on each leaf, then apply the scoring formula to get the quality score." Schematic.
- Figure 3 (p.4): "Comparison of test AUC convergence on Higgs 10M dataset. The eps parameter corresponds to the accuracy of the approximate sketch … We find that local proposals require fewer buckets, because it refine split candidates." Legible: exact greedy, local eps=0.3 and global eps=0.05 converge together to ≈0.82; global eps=0.3 plateaus at ≈0.80.
- Figure 4 (p.5): "Tree structure with default directions. An example will be classified into the default direction when the feature needed for the split is missing." Schematic, legible.
- Figure 5 (p.5): "Impact of the sparsity aware algorithm on Allstate-10K … The sparsity aware algorithm is more than 50 times faster than the naive version."
- Figure 6 (p.5): "Block structure for parallel learning. Each column in a block is sorted by the corresponding feature value."
- Figure 7 (p.6, four panels a–d): "Impact of cache-aware prefetching in exact greedy algorithm … Using cache aware prefetching improves the performance by factor of two when the dataset is large." Panels: Allstate 10M, Higgs 10M, Allstate 1M, Higgs 1M.
- Figure 8 (p.6): "Short range data dependency pattern that can cause stall due to cache miss."
- Figure 9 (p.6, two panels): "The impact of block size in the approximate algorithm … overly small blocks results in inefficient parallelization, while overly large blocks also slows down training due to cache misses."
- Figure 10 (p.8): "Comparison between XGBoost and pGBRT on Yahoo LTRC dataset."
- Figure 11 (p.9): "Comparison of out-of-core methods on different subsets of criteo data … basic algorithm can only handle 200M examples. Adding compression gives 3x speedup, and sharding into two disks gives another 2x speedup. The system runs out of file cache start from 400M examples."
- Figure 12 (p.9, panels a and b): "Comparison of different distributed systems on 32 EC2 nodes for 10 iterations on different subset of criteo data. XGBoost runs more 10x than spark per iteration and 2.2x as H2O's optimized version (However, H2O is slow in loading the data, getting worse end-to-end time)."
- Figure 13 (p.10): "Scaling of XGBoost with different number of machines on criteo full 1.7 billion dataset … causing the trend to be slightly super linear."
- Table 1 (p.7): "Comparison of major tree boosting systems." Feature matrix over XGBoost, pGBRT, Spark MLLib, H2O, scikit-learn, R GBM.
- Table 2 (p.8): "Dataset used in the Experiments." Allstate 10M/4227; Higgs Boson 10M/28; Yahoo LTRC 473K/700; Criteo 1.7B/67.
- Table 3 (p.8): "Comparison of Exact Greedy Methods with 500 trees on Higgs-1M data." XGBoost 0.6841 s/tree, AUC 0.8304; XGBoost colsample=0.5 0.6401, 0.8245; scikit-learn 28.51, 0.8302; R.gbm 1.032, 0.6224.
- Table 4 (p.8): "Comparison of Learning to Rank with 500 trees on Yahoo! LTRC Dataset." XGBoost 0.826 s/tree, NDCG@10 0.7892; colsample=0.5 0.506, 0.7913; pGBRT 2.576, 0.7915.
- Algorithms 1–3 (pp.3, 5): exact greedy split finding; approximate split finding; sparsity-aware split finding.
- Appendix A (pp.11–13): weighted quantile sketch, Definitions A.1–A.3, Lemma A.1 and following.

**Repeated numbers, checked against each other**
- "more than ten times faster than existing popular solutions on a single machine" (p.1) vs Table 3: 28.51/0.6841 = **41.7x** vs scikit-learn, but 1.032/0.6841 = **1.51x** vs R.gbm. **The 10x claim holds against scikit-learn only.** R's GBM is one of the two "existing popular solutions" in the same table and is only 1.5x slower.
- Higgs feature count: §6.2 "21 kinematic properties … seven additional derived physics quantities" = 28; Table 2 says m = 28. **Consistent.**
- Criteo: §6.2 "13 integer features and 26 ID features" → "13 integer, 26 average CTR statistics and 26 counts" = 65 … Table 2 says m = 67. **Off by 2; not reconciled in the text.**
- Yahoo LTRC: §6.2 "20K web search queries, with each query corresponding to a list of around 22 documents" → 20,000 × 22 = 440K, vs Table 2 n = 473K. Consistent with "around 22".
- Out-of-core: Fig. 11 caption "compression gives 3x speedup, and sharding into two disks gives another 2x" matches §6.5 "compression helps to speed up computation by factor of three, and sharding into two disks further gives 2x speedup". **Consistent.**
- Distributed: Fig. 12 caption "10x than spark per iteration and 2.2x as H2O's optimized version" — §6.6 text gives no numbers, only "runs faster than the baseline systems". Consistent but unquantified in prose.
- Block size: §4.2/Fig. 9 conclude 2^16 examples per block; §4.3 independently requires 2^16 for the 16-bit row-offset encoding. **Consistent and mutually reinforcing.**

**Inconsistencies between abstract, results and conclusion**
1. **"More than ten times faster than existing popular solutions"** (p.1) is contradicted within the paper's own Table 3 for R.gbm (1.5x). It holds for scikit-learn (42x) and, in the distributed setting, for Spark MLLib (Fig. 12).
2. **pGBRT is more accurate, not less.** Table 4: pGBRT NDCG@10 0.7915 > XGBoost 0.7892. The text says only "We find that XGBoost runs faster" (p.9) — correctly scoped — but the §1 framing "state-of-the-art results" invites the wrong reading of Table 4.
3. **Criteo feature count 65 (text) vs 67 (Table 2).**
4. Figure 12's caption is ungrammatical ("XGBoost runs more 10x than spark") — the intended claim is "more than 10x".
5. The conclusion (p.10 §7) claims "provides state-of-the-art results on many problems", but the paper's own accuracy evidence is three numbers (Table 3 AUC, Table 4 NDCG, Fig. 3 AUC) and the state-of-the-art claim rests on the Kaggle/KDDCup usage counts in §1, not on any experiment in §6.

</scratchpad>

<analysis>

## 1 SNAPSHOT

- **Citation:** Chen, T., Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD '16, pp. 785–794, ACM (2016). This PDF is arXiv:1603.02754v3. Both authors University of Washington.
- **Type:** Systems paper — an ML system description with algorithmic contributions and an engineering evaluation.
- **Problem:** make gradient tree boosting scale to billions of examples on limited hardware, and handle sparse inputs in a principled way.
- **Contributions:** regularized objective (§2.1, "included for completeness"); weighted quantile sketch with proven merge/prune guarantees (§3.3, Appendix A); sparsity-aware split finding with learned default directions (§3.4, Alg. 3); column-block data layout, cache-aware prefetching, block compression and sharding (§4).
- **Headline results:** 50x from sparsity-awareness (Fig. 5); ~2x from cache-awareness at 10M rows (Fig. 7); 3x from compression and a further 2x from sharding out-of-core (Fig. 11); >10x vs Spark MLLib and 2.2x vs H2O per iteration distributed (Fig. 12); processes the full 1.7B-row Criteo dataset on four machines (Fig. 13) or one machine out-of-core (§6.5).
- **What it is not:** an accuracy paper. Three accuracy measurements in total.

## 2 ABSTRACT

- **Claims:** "a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning"; "insights on cache access patterns, data compression and sharding"; "XGBoost scales beyond billions of examples using far fewer resources than existing systems."
- **Delivered:** all three. The 1.7B-example claim is measured in §6.5 (single machine, out-of-core) and §6.6 (four machines). "Far fewer resources" is supported comparatively against Spark MLLib and H2O, which could not handle the full dataset at all (§6.6).
- **Not in the abstract, and correctly so:** any accuracy claim.
- [Assessment] This is an unusually honest abstract for a paper of this influence: every claim in it is a scalability claim, and every scalability claim is measured.

## 3 INTRODUCTION AND MOTIVATION

- **Positioning (p.1):** "gradient tree boosting [10] is one technique that shines in many applications"; "it is the de-facto choice of ensemble method".
- **The empirical hook:** "Among the 29 challenge winning solutions published at Kaggle's blog during 2015, 17 solutions used XGBoost … eight solely used XGBoost … For comparison, the second most popular method, deep neural nets, was used in 11 solutions" and "XGBoost was used by every winning team in the top-10" of KDDCup 2015.
- **The claimed gap (p.2):** "While there are some existing works on parallel tree boosting [22, 23, 19], the directions such as out-of-core computation, cache-aware and sparsity-aware learning have not been explored."
- [Assessment] The Kaggle statistics are the most-cited sentence in this paper and the least verifiable part of it: the source is "Kaggle's blog" with the footnote "Solutions come from of top-3 teams of each competitions" (footnote 3), no list, no date, no URL. It is an accurate report of a 2015 snapshot, not evidence of accuracy superiority, and it is now a decade old. **Cite it for adoption, never for accuracy.**

## 4 LITERATURE REVIEW / RELATED WORK

- **Foundations (§5, p.7):** gradient boosting [10]; second-order method from Friedman et al. [12]; regularization "resembles previous work on regularized greedy forest [25], but simplifies the objective and algorithm for parallelization"; column sampling "borrowed from RandomForest [4]".
- **Parallel tree learning:** [22] pGBRT, [19] PLANET, [23] column-partitioned; "Most of these algorithms fall into the approximate framework described in this paper."
- **Sparsity:** "few works on tree learning have considered this topic in a principled way. The algorithm proposed in this paper is the first unified approach to handle all kinds of sparsity patterns."
- **Quantile sketch:** GK [14] and extensions [24] handle unweighted data; "the weighted quantile sketch proposed in this paper is the first method to solve this problem."
- **System comparison (Table 1, p.7):** six systems × six capabilities.
- [Assessment] Two "first" claims are asserted "to the best of our knowledge" and not contested here. Table 1 is a self-authored capability matrix with no citation per cell — useful orientation, weak evidence.

## 5 METHODOLOGY

**Regularized objective (§2.1–2.2, pp.2–3).** L(φ) = Σ l(ŷ,y) + Σ Ω(f), with Ω(f) = γT + ½λ‖w‖². Second-order Taylor expansion gives the simplified objective (Eq. 3); optimal leaf weight w*_j = −Σg_i / (Σh_i + λ) (Eq. 5); structure score (Eq. 6) and split gain (Eq. 7). "When the regularization parameter is set to zero, the objective falls back to the traditional gradient tree boosting."

**Overfitting controls (§2.3).** Shrinkage η [11] and column subsampling; "using column sub-sampling prevents over-fitting even more so than the traditional row sub-sampling" — attributed to "user feedback", not measured.

**Split finding (§3).** Alg. 1 exact greedy; Alg. 2 approximate, with global (per-tree) or local (per-split) proposals; Fig. 3 compares them on Higgs 10M.

**Weighted quantile sketch (§3.3, Appendix A).** Rank function r_k(z) weighted by second-order statistics h_i (Eq. 8), justified by rewriting Eq. 3 as a weighted squared loss "with labels g_i/h_i and weights h_i". The sketch supports merge (error max(ε₁,ε₂)) and prune (error ε → ε + 1/b), with proofs in Appendix A.

**Sparsity-aware split finding (§3.4, Alg. 3).** Each node gets a learned default direction; only non-missing entries I_k are enumerated, in both ascending and descending order, so complexity becomes linear in ‖x‖₀.

**System design (§4).** Column blocks in CSC format, pre-sorted once (§4.1); complexity drops from O(Kd‖x‖₀ log n) to O(Kd‖x‖₀ + ‖x‖₀ log n). Cache-aware prefetching into per-thread buffers (§4.2). Block size chosen as 2^16 (§4.2, Fig. 9). Block compression, 26–29% ratio, and block sharding across disks (§4.3).

**Evaluation protocol (§6.2).** Four datasets (Table 2). Single-machine experiments on a Dell PowerEdge R420, two eight-core Xeon E5-2470 (2.3 GHz), 64 GB RAM, all cores unless stated. Out-of-core on one AWS c3.8xlarge (32 vcores, two 320 GB SSD, 60 GB RAM). Distributed on 32 EC2 m3.2xlarge with data on S3. "In all the experiments, we boost trees with a common setting of maximum depth equals 8, shrinkage equals 0.1 and no column subsampling unless explicitly specified."

- [Assessment] The protocol is specified well enough to reproduce the *shape* of the results. Missing: number of repetitions, seeds, and any variance estimate on the timings. Every timing number in the paper is, as far as the text says, a single run. **All experiments are CPU-only** — there is no GPU experiment anywhere in this paper.

## 6 RESULTS

- **Approximate vs exact split finding (Fig. 3):** exact greedy, local ε=0.3 and global ε=0.05 all converge to ≈0.82 test AUC on Higgs 10M; global ε=0.3 plateaus ≈0.80. Conclusion: "the quantile strategy can get the same accuracy as exact greedy given reasonable approximation level."
- **Sparsity-awareness (Fig. 5):** "more than 50 times faster than the naive version" on Allstate-10K, across 1–16 threads.
- **Cache-awareness (Fig. 7):** ~2x on the 10M datasets; on the 1M datasets the two curves nearly coincide — the effect is scale-dependent, which the caption states.
- **Block size (Fig. 9):** 2^16 is the optimum; both smaller and larger are worse.
- **Classification (Table 3, Higgs-1M, 500 trees):** XGBoost 0.6841 s/tree at AUC 0.8304; scikit-learn 28.51 s/tree at 0.8302; R.gbm 1.032 s/tree at 0.6224. XGBoost and scikit-learn are accuracy-equivalent; R.gbm is far worse because it "only expands one branch of a tree".
- **Ranking (Table 4, Yahoo LTRC, 500 trees):** XGBoost 0.826 s/tree at NDCG@10 0.7892; colsample=0.5 0.506 s at 0.7913; pGBRT 2.576 s at **0.7915**. XGBoost is 3.1x faster; pGBRT is marginally more accurate.
- **Out-of-core (Fig. 11):** basic algorithm caps at 200M examples; compression 3x; +sharding 2x; the full method handles 1.7B on one machine.
- **Distributed (Fig. 12):** >10x Spark MLLib and 2.2x H2O per iteration; H2O loses further on end-to-end time due to slow loading; "spark suffers from drastic slow down when running out of memory."
- **Scaling (Fig. 13):** near-linear, "slightly super linear", 1.7B examples on four machines.
- [Assessment] The systems results are convincing and each is isolated by an ablation (with/without the technique). The accuracy results are thin by design and do **not** establish XGBoost as more accurate than anything: it ties scikit-learn (Table 3) and loses narrowly to pGBRT (Table 4).

## 7 FIGURES AND TABLES (each one, in order)

- **Figure 1 (p.2), Tree Ensemble Model.** Schematic of K additive trees summing leaf scores. Pedagogical only.
- **Figure 2 (p.3), Structure Score Calculation.** Illustrates Eq. 6 as leaf-wise sums of g and h. Pedagogical only.
- **Algorithm 1 (p.3), Exact Greedy Algorithm for Split Finding.** Complete.
- **Algorithm 2 (p.3), Approximate Algorithm for Split Finding.** Complete; distinguishes global from local proposal.
- **Figure 3 (p.4), Test AUC convergence on Higgs 10M.** Legible. The load-bearing evidence that the approximate algorithm is not a meaningful accuracy sacrifice: three of four settings are indistinguishable at ≈0.82. Only global ε=0.3 is clearly worse.
- **Figure 4 (p.5), Tree structure with default directions.** Legible worked example (missing Age / missing Gender routed to learned defaults).
- **Algorithm 3 (p.5), Sparsity-aware Split Finding.** Complete; the two-pass (missing-right then missing-left) structure is the key detail.
- **Figure 5 (p.5), Impact of the sparsity aware algorithm on Allstate-10K.** Log-scale time per tree, 1–16 threads. Gap is ~50x and roughly constant across thread counts — i.e. the gain is algorithmic, not a parallelism artifact.
- **Figure 6 (p.5), Block structure for parallel learning.** Schematic of CSC blocks with per-column sorted values and row-index pointers.
- **Figure 7 (p.6, a–d), Impact of cache-aware prefetching.** Four panels. 10M panels (a, b) show a clear ~2x gap that widens with thread count; 1M panels (c, d) show almost none. Honest reporting of a scale-dependent effect.
- **Figure 8 (p.6), Short range data dependency pattern.** Schematic explaining the stall.
- **Figure 9 (p.6, a–b), Impact of block size.** Four block sizes on two 10M datasets; 2^16 best. Supports the §4.3 encoding choice.
- **Table 1 (p.7), Comparison of major tree boosting systems.** Self-authored capability matrix; XGBoost is the only "yes" in every column. No per-cell citation.
- **Table 2 (p.8), Dataset used in the Experiments.** Allstate 10M/4227; Higgs 10M/28; Yahoo LTRC 473K/700; Criteo 1.7B/67. Note: the Criteo m=67 does not reconcile with the 13+26+26=65 described in the text.
- **Table 3 (p.8), Exact Greedy Methods, 500 trees, Higgs-1M.** The only classification-accuracy table. Shows accuracy parity with scikit-learn and a 42x speed gap; also shows the §1 "more than ten times faster" claim does not hold against R.gbm (1.5x).
- **Figure 10 (p.8), XGBoost vs pGBRT on Yahoo LTRC.** Log-scale time per tree over 1–16 threads; XGBoost below pGBRT throughout.
- **Table 4 (p.8), Learning to Rank, 500 trees, Yahoo LTRC.** XGBoost faster (3.1x), pGBRT marginally more accurate (0.7915 vs 0.7892). The paper reports this without spin.
- **Figure 11 (p.9), Out-of-core methods on criteo subsets.** Legible. Shows the file-cache transition at ~400M examples and that compression+shard degrades gracefully past it.
- **Figure 12 (p.9, a–b), Distributed systems on 32 EC2 nodes.** Legible. (a) end-to-end including loading, (b) per-iteration. H2O wins nothing in (a) because of load time; Spark's curve bends sharply upward when it exhausts memory.
- **Figure 13 (p.10), Scaling with number of machines on full 1.7B criteo.** Legible; near-linear on log-log axes, 4 → 32 machines.
- **Appendix A (pp.11–13), Weighted Quantile Sketch.** Definitions A.1–A.3, Lemma A.1 with proof, extension of function domains. Establishes the ε-approximate summary guarantee ˜r⁺(y) − ˜r⁻(y) − ˜ω(y) ≤ εω(D).

## 8 DISCUSSION

- There is no separate discussion section; §6 interleaves interpretation with each experiment.
- **Authors' reading:** each optimization is individually necessary (each has its own ablation figure), and "it is even more exciting to combine these techniques to make an end-to-end system" (p.2).
- **What the evidence shows:** exactly that, for scalability. For accuracy, the evidence shows parity with a careful CPU baseline (scikit-learn) and a narrow loss to pGBRT on ranking.
- [Assessment] The gap between what this paper measures (throughput, memory behaviour, scaling) and what it is usually cited for (predictive performance) is the single most important thing to keep straight when citing it.

## 9 LIMITATIONS

- **Stated by the authors:** none. There is no limitations section.
- **Not stated in paper, from the evidence:**
  - Single runs; no seeds, repetitions or variance on any timing or accuracy number.
  - Accuracy evidence is three measurements on two datasets; no cross-validation, no confidence intervals, no statistical test.
  - The §1 "more than ten times faster" claim is unqualified but true of only one of the two single-machine baselines in Table 3.
  - "Column sub-sampling prevents over-fitting even more so than … row sub-sampling" (§2.3) is attributed to user feedback and never tested; Table 3 shows colsample=0.5 *hurting* AUC (0.8245 vs 0.8304) while Table 4 shows it helping NDCG (0.7913 vs 0.7892).
  - The Kaggle/KDDCup adoption statistics have no reproducible source.
  - No GPU experiments; no comparison with any histogram-based system other than pGBRT's approximate algorithm (LightGBM postdates this paper).
  - All datasets are ≥473K rows; nothing here speaks to small-data behaviour.

## 10 CONCLUSION AND FUTURE WORK

- **Conclusion (p.10 §7):** "we described the lessons we learnt when building XGBoost, a scalable tree boosting system that is widely used by data scientists and provides state-of-the-art results on many problems. We proposed a novel sparsity aware algorithm … and a theoretically justified weighted quantile sketch … cache access patterns, data compression and sharding are essential elements."
- **Supported:** the systems claims, all of them, with per-technique ablations.
- **Partially supported:** "provides state-of-the-art results on many problems" — this rests on the §1 competition statistics, not on §6. The paper's own §6 accuracy evidence shows parity, not superiority.
- **Future work:** Not stated in paper. No future-work section.

## 11 REPRODUCIBILITY

- **Code:** yes — `https://github.com/dmlc/xgboost` (footnote 2, p.1) and the rabit allreduce library (footnote 6, p.7).
- **Data:** Allstate (Kaggle, registration-gated), Higgs (UCI, open), Yahoo LTRC (Yahoo Webscope, application required), Criteo terabyte logs (public download). URLs given in footnotes 8–10.
- **Hyperparameters:** partially — max depth 8, shrinkage 0.1, no column subsampling unless stated, 500 trees for Tables 3 and 4, ε values for Fig. 3, block size 2^16. Learning-rate schedules, λ, γ and other regularization values are **not stated**.
- **Preprocessing:** the Criteo ID-feature → CTR/count transformation is described in prose (§6.2) but not as code or exact windows beyond "first ten days" / "next ten days".
- **Seeds / repetitions:** Not stated in paper.
- **Hardware:** fully specified for all three settings.
- [Assessment] Reproducible in direction, not in exact numbers. As with LightGBM, the durable reproducibility asset is the released implementation rather than the experiment scripts.

## 12 CRITICAL EVALUATION

**Strengths**
1. Every systems claim has its own controlled ablation with the technique switched off (Figs. 5, 7, 9, 11), so the contributions are separable rather than bundled into one end-to-end number.
2. The weighted quantile sketch is given a real theoretical treatment (Appendix A: definitions, ε-approximation guarantee, merge/prune error bounds), not just an empirical justification.
3. Results that cut against the system are reported plainly — pGBRT's higher NDCG (Table 4), colsample hurting AUC (Table 3), cache-awareness doing nothing at 1M rows (Fig. 7c–d).

**Weaknesses**
1. "More than ten times faster than existing popular solutions" (§1) is falsified by the paper's own Table 3 for R.gbm (1.51x); it is true only of scikit-learn.
2. Zero variance estimates anywhere: every timing and every accuracy number is a single run, so none of the smaller gaps (e.g. 0.8304 vs 0.8302) is interpretable.
3. The "state-of-the-art results" claim in the abstract-adjacent framing and the conclusion is carried entirely by 2015 Kaggle adoption counts with an unciteable source, while the paper's own accuracy evidence is three numbers showing parity.

**Credibility: High.** The systems contributions are ablated, proven where provable, released as working code, and validated on a 1.7-billion-row public dataset; the overreach is confined to two rhetorical claims (the 10x sentence and "state-of-the-art results") that the paper's own tables let a careful reader correct.

## 13 RELEVANCE TO MY FOCUS

**How our draft uses this paper** (grep over `paper/build_paper.js`, `paper/build_supplement.js`, `paper/NUMBERS.md`, `RESULTS.md`, `lightgbm-pivot/OVERVIEW.md`):

| Our use | Location in our draft | Supported? |
|---|---|---|
| Reference entry: "Chen, T., Guestrin, C.: XGBoost: A Scalable Tree Boosting System. KDD '16, pp. 785–794. ACM (2016)" | `paper/build_supplement.js:86` (in `MORE_REFS`) | **Supported** as a bibliographic record. The PDF is the arXiv v3; the KDD page range 785–794 is correct per the LightGBM paper's own reference [13] (p.9) but is not printed on this PDF. |
| "Gradient-boosting background" | `lightgbm-pivot/OVERVIEW.md:169` | **Supported.** §2 is a self-contained derivation of regularized gradient tree boosting (Eqs. 1–7). |
| XGBoost as a benchmark comparator: "LR 0.893; XGBoost 0.893"; "logistic regression, LightGBM and XGBoost reached AUROC 0.893, 0.891 and 0.893" | `paper/build_paper.js:199`, `:202`, `paper/NUMBERS.md:24`, `paper/build_supplement.js:28` | **Supported as a choice of comparator.** Table 1 (p.7) and §6 establish XGBoost as the reference scalable boosting system, and Ke2017 independently states "we use XGBoost as our baseline" (Ke2017 p.3). Our own AUROC numbers make no claim about Chen2016. |
| "our XGBoost does better (0.505)" on Olist; "On Olist our purchase-time features let XGBoost clear the no-model rule (+4.1 points…)" | `lightgbm-pivot/OVERVIEW.md:16`, `paper/build_supplement.js:61` | **No claim about Chen2016**; these are our measurements. Nothing to check. |
| "XGBoost on the GPU on the backorder benchmark: 89.19 % / 89.63 %"; "(XGBoost on CUDA)" | `lightgbm-pivot/OVERVIEW.md:17`, `paper/build_supplement.js:70`, `RESULTS.md:791`, `RESULTS.md:828` | **Not found in paper.** Chen2016 contains **no GPU experiment**; all hardware is CPU (Dell R420 Xeon; AWS c3.8xlarge; m3.2xlarge — §6.2, §6.5, §6.6). The GPU tree method came later (Mitchell & Frank 2017, cited as [18] in Ke2017, not here). Our draft does not attribute CUDA to Chen2016 — keep it that way, and do not let an editing pass attach {chen} to a GPU sentence. |

**Where our use is not fully supported**

1. **Chen2016 is an uncited reference in our supplement.** `build_supplement.js:128` prints `MORE_REFS` unconditionally as an "Additional References" list, but grep finds **no inline citation marker for XGBoost anywhere in `build_supplement.js` or `build_paper.js`**. A reference that appears in the list but is never cited in the text is an editorial defect at peer review, and here it is also a missed opportunity: the supplement's Fig03/Fig05 comparisons *should* cite it at the point where XGBoost is introduced as a comparator.
2. **Do not let "XGBoost is a strong baseline" drift into "XGBoost is the accuracy state of the art."** Our supplement text ("LightGBM is statistically tied with logistic regression and CNN", Fig05 caption) is correctly worded. Chen2016 itself provides only three accuracy measurements and loses one of them to pGBRT (Table 4), so it cannot be cited to support any accuracy ranking.
3. **The Kaggle "17 of 29" statistic is tempting and should not be used.** If a reviewer asks why tree ensembles rather than deep models, cite Grinsztajn2022 (paper 05) for evidence, not Chen2016 §1 for a decade-old, unsourced competition tally.

**Methodology worth reusing**
- The one-ablation-per-claim structure (Figs. 5, 7, 9, 11): each engineering claim gets a figure in which only that technique is toggled. Our supplement's Fig09 variants A–E follows the same discipline.
- Reporting the comparator's win when it wins (Table 4, pGBRT NDCG). Our draft already does the analogue — "our re-run of their own set-up already reaches 90.30 %; the defensible gain is +0.34 points" (`build_supplement.js:54`) — which is the right register.

**Critique to borrow**
- Chen2016's single-run timings with no variance are exactly the failure mode our `RESULTS.md` §6 caught in our own pipeline (the three comparison bugs). Our series-clustered bootstrap intervals are a genuine improvement on the standard set by both boosting papers, and §4.1 of our paper should say so plainly rather than leave it to the supplement.

</analysis>
