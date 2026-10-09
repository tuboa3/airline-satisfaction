# Kaggle Playground S6E10 — Leakage-Safe GBDT Ensembling: Integrity Diagnosis, Rank Optimization, and Meta-Learning

---

## 1. Question

**Interpreted research goal.** For Kaggle Playground Series S6E10 (_Predicting Airline Satisfaction_, ROC-AUC metric, ~700k synthetic train rows, ~300k unlabeled test rows, ~130k original-data rows available as external data), identify the strongest defensible strategy for combining an existing set of CatBoost / LightGBM / XGBoost / HistGradientBoosting models, under a strict GBDT-only model-family constraint:

1. Direct **rank blending** with constrained weight optimization,
2. Direct **probability / logit blending** with constrained weight optimization,
3. A **shallow GBDT meta-learner** trained on leakage-safe out-of-fold (OOF) predictions,

while (a) diagnosing the reported OOF→leaderboard gap (OOF 0.96126 vs. public LB 0.95975, gap = 0.00151, with a reported categorical-code inversion and observed prediction-distribution differences), (b) eliminating train–test inconsistencies, and (c) minimizing ensemble-selection overfitting.

**Decision supported:** which experiment to run next, in what order, with what rejection criteria.

---

## 2. Executive Summary

Ranked takeaways:

1. **The leaderboard gap is a pipeline problem first, a blending problem second.** ROC-AUC is a pure ranking metric ([sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)), so any strictly increasing recalibration is score-neutral. A global sigmoid/calibration story _cannot_ explain a 0.00151 AUC drop. A reported **categorical-code inversion at test time**, by contrast, is a concrete mechanism that changes the ranking of test predictions while leaving OOF untouched. Audit category mappings before touching any blender.
2. **The gap magnitude is ~2.1σ of pure test-set sampling noise, so noise alone is a possible but weak explanation.** With ~300k test rows (≈165k / 135k class split), the large-sample standard error of an AUC ≈ 0.961 is ≈ 0.0007; the observed 0.00151 gap is ≈ 2.1 SE. Sampling variation is plausible, but independent S6E10 teams report CV–LB agreement within ~0.0006 ([aditbytes repo](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction)), which makes an implementation defect the higher-prior hypothesis. Both mechanisms can coexist.
3. **Blending cannot recover a 0.00151 deficit.** GBDT-family models on this dataset are highly rank-correlated (typical pairwise Spearman ρ > 0.98 between well-tuned GBDTs). Realistic blend gains over the best single model are ~0.0002–0.0008 in this regime — smaller than the gap. Expect blending to move you from ~0.960 to ~0.961, not from 0.95975 to 0.96243 (current public LB best).
4. **Do not use plain Nelder–Mead for weight optimization.** Pooled OOF AUC is a piecewise-constant function of the simplex weights (flat plateaus separated by pairwise-comparison flip hyperplanes), and SciPy's Nelder–Mead supports at most box bounds, not the simplex constraint ([SciPy docs](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)). A predeclared coarse simplex grid + Dirichlet random search + one local coordinate refinement is cheaper, reproducible, and immune to NM's initialization sensitivity on nonsmooth objectives.
5. **Equal-weight rank blending is the reference baseline; a constrained optimized blend must beat it by a predeclared margin to be retained.** Use the effective-number-of-models diagnostic N_eff = 1/Σw²ᵢ (equal weights on 4 models → 4; a 0.7-dominated vector → 1.9; a single model → 1) as a stability/robustness diagnostic, not as a hard constraint without evidence.
6. **A shallow GBDT meta-learner is justified only if the base models have complementary _conditional_ errors** (agreement/disagreement regions that carry label signal). With four same-family GBDTs on a largely monotone feature space, the prior is against it. Retain it only if a cross-fitted, independently confirmed meta-learner beats the best constrained blend by ≥ 0.0002 pooled OOF AUC with a 4/5-fold sign consistency.
7. **Recommended order of operations:** (Tier 1) category-mapping audit + submission-integrity harness + fixed-pipeline resubmission with an equal-weight blend; (Tier 2) coarse constrained rank-weight search with a held-out confirmation fold protocol; (Tier 3) shallow LightGBM meta-learner ablation; (Tier 4) only what Tiers 1–3 justify.

---

## 3. Methodology

**Search angles.** (i) Official S6E10 rules, evaluation, and data provenance (Kaggle competition + data pages); (ii) tool-authoritative documentation for ROC-AUC semantics, calibration behavior, LightGBM / CatBoost shallow-tree regularization and early stopping, and SciPy Nelder–Mead constraint handling; (iii) community/competition-specific evidence: public S6E10 repositories reporting CV–LB agreement, feature AUCs, and leaderboard structure; (iv) standard references on stacking leakage and cross-fitting.

**Source types.** Official Kaggle pages; official library documentation; a scikit-learn GitHub issue documenting isotonic calibration changing rank-based metrics; the standard large-sample AUC variance approximation; public S6E10 experiment repositories for competition-specific numbers.

**Limitations.** The reported scores (OOF 0.96126, LB 0.95975, categorical inversion, calibration shift) are treated as reported experimental facts, not independently verified — we have no access to the repo's artifacts. Base-model pairwise correlations, exact fold construction, and the specific inverted category are unknown to us; where a conclusion depends on them, this is stated and a measurement is prescribed. Public-repository claims are single-team evidence, not leaderboards.

**Verified vs. hypothesis tagging.** Statements sourced from documentation are marked [V] (verified); competition-specific or inferred statements are marked [H] (hypothesis requiring local measurement).

---

## 4. Findings

### 4.1 Part I — Audit the OOF–Leaderboard gap (do this before any new ensemble)

**The arithmetic that rules out "calibration" as the cause.** For a fixed score vector s and any strictly increasing h: AUC(y, h(s)) = AUC(y, s). sklearn computes ROC-AUC from prediction _scores_ (any monotone score works) and documents AUC/Gini as a summary of ranking ability ([sklearn roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [sklearn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)). Therefore:

- A global sigmoid recalibration of an unchanged test score vector: **AUC unchanged** [V].
- Different probability _scales_ between train and test (e.g., mean predicted probability 0.56 in OOF vs. 0.48 in test): affects calibration plots, **not** the AUC of the same model [V].
- The only ways AUC drops: (a) the _ordering_ changed (feature/pipeline defect, or genuine conditional-distribution difference), (b) ties were introduced (isotonic-style non-strict transforms), or (c) sampling variation.

**How big should the gap be under pure noise?** Using the large-sample approximation SE(AUC) ≈ sqrt(AUC(1−AUC)·(1/n₊ + 1/n₋)) with n₊ ≈ 165k, n₋ ≈ 135k (300k test rows, ~55% satisfied [H]—measure the actual train prevalence), SE ≈ 0.0007. The observed 0.00151 gap is ≈ 2.1 SE — possible under noise, but the same teams' public reports show CV–LB gaps of ~0.0006 on this competition ([aditbytes](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction)). Conclusion: investigate a defect; do not assume either pure noise or pure calibration.

#### 4.1.1 Cause A — categorical-code inversion (priority 1)

The reported inversion is the only stated mechanism that plausibly changes test ranking while leaving OOF intact. Audit every categorical feature (Gender, Customer Type, Type of Travel, Class, and any engineered categorical such as route codes [H]) across all five consumption points: fold training, full refit, test inference, submission generation, and serialization boundaries.

Failure modes to check, in order of observed frequency:

1. **Ordering assumptions in code assignment.** `pd.factorize` assigns codes by order of appearance; `pd.Categorical` by declared category order. Train and test files with the same string categories can produce _different integer codes_ if mapping is fit independently per file. Use a single, persisted mapping.
2. **Native-categorical ingestion differences.** CatBoost (`cat_features`), LightGBM (`categorical_feature`), and HistGradientBoosting (`categorical_features` with an ordinal encoder fit on training folds) all handle unseen categories and code semantics differently. An integer code that one model treats as categorical and another treats as numeric magnitude is a silent ranking killer.
3. **Serialization drift.** Pickled preprocessors/joblib pipelines with a stale `categories_` list, or feature-column order differences after a `concat`/`reindex`.
4. **Target-encoding boundaries.** If any target/label encoding was fit on full training data instead of per-fold, OOF is contaminated (optimistic) _and_ test uses full-data encodings — distorting the OOF/LB comparison in opposite directions.

**Concrete assertions (implement; fail loudly):**

```python
# A1: category bijection
assert set(train['Class'].unique()) == set(test['Class'].unique())
# A2: persisted mapping is used everywhere (never re-fit on test)
expected_map = {"Eco": 0, "Eco Plus": 1, "Business": 2}
assert saved_encoder.categories_[j] == list(expected_map)
# A3: schema and dtypes identical at inference
assert list(X_test.columns) == list(X_train.columns)
assert (X_test.dtypes == X_train.dtypes).all()
# A4: train and test predictions come from the SAME fitted pipeline object
assert inference_pipeline is fold_pipeline_registry[fold_id]  # or documented refit
# A5: no unseen categories at inference (or explicit UNKNOWN policy)
assert X_test[cat_cols].map(saved_encoder).isna().sum().sum() == 0
# A6: submission id integrity
assert (submission['id'] == test_ids).all() and len(submission) == len(test_ids)
assert submission['satisfaction'].between(0, 1).all()
assert submission['satisfaction'].isna().sum() == 0
```

**Diagnostic that discriminates "inversion changed ranking" from "distribution shift":** re-score the current submission against a locally reproduced test-prediction vector. If the two vectors' Spearman correlation < 0.999, the pipeline does not reproduce itself → implementation defect confirmed before any distributional argument [H → measure].

#### 4.1.2 Cause B — OOF vs. inference training mismatch

OOF predictions come from models trained on 4/5 of the data; test predictions from full-data refits (or fold ensembles). Legitimate differences: slightly sharper full-refit scores, different early-stopping iterations. Illegitimate differences: different feature-engineering versions between the OOF script and the submission script, different model versions, different fold-assignment handling, or OOF aligned by row _position_ rather than by `id` (alignment bug). Verify by joining, not concatenating: `oof = oof.set_index('id').loc[train_ids]` [V-procedure].

#### 4.1.3 Cause C — population/distribution differences

Measure (train-legitimate information only): class prevalence in train; predicted-score distribution of OOF vs. test (histogram + KS statistic); per-feature train-vs-test drift (public S6E10 EDA reports every feature mean differing by <2% and no material train/test drift [aditbytes](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction)); missingness (only `Arrival Delay in Minutes`, ~0.04% [same source]). Do **not** estimate test label prevalence from unlabeled features — it is not identifiable, and prevalence shift does not change a fixed model's ranking anyway (it changes calibration).

#### 4.1.4 Cause D — validation reliability

The 700k synthetic rows are generated from ~130k original rows ([Kaggle data page](https://www.kaggle.com/competitions/playground-series-s6e10/data); original dataset: [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)). This creates a specific, measurable risk: **near-duplicate rows across folds inflate OOF AUC** relative to a test fold containing genuinely distinct generated rows. Measure: exact/near-duplicate rate (hash of the feature tuple) between each validation fold and its training complement, and between train and test. If cross-fold duplication is materially higher than train↔test duplication, the OOF score is structurally optimistic and part of the gap is explained _without_ any bug. This does not require changing folds — it requires knowing the bias direction [H → measure].

#### 4.1.5 Prioritized diagnostic checklist

| Prio | Check                                                        | Method                      | Red flag                                      |
| ---- | ------------------------------------------------------------ | --------------------------- | --------------------------------------------- |
| 1    | Category mapping bijection train↔test↔saved encoder          | Assertions A1–A2            | any mismatch                                  |
| 2    | Submission reproduces local test predictions                 | Spearman ≥ 0.999 vs. re-run | < 0.999                                       |
| 3    | OOF↔`id` alignment                                           | join-based audit            | row-count/NaN after join                      |
| 4    | Feature schema/dtype parity OOF script vs. submission script | Assertions A3               | any diff                                      |
| 5    | OOF fold-model vs. full-refit provenance                     | registry check              | undocumented refit                            |
| 6    | Cross-fold near-duplicate rate                               | feature-tuple hashing       | dup rate ≫ train↔test dup rate                |
| 7    | Score-distribution shift OOF vs. test                        | KS test on scores           | KS > 0.05 _and_ rank-invariant features shift |
| 8    | Fold-level AUC variance                                      | mean ± sd across folds      | sd > 0.002                                    |

**Gate:** do not proceed to Tier 2/3 until checks 1–5 pass and the resubmitted fixed pipeline's LB score lands within ~0.0007 (1 SE) of OOF, or the residual gap is explained by check 6.

---

### 4.2 Part II — Rank preservation, calibration, and train–test consistency (Question Two)

#### 4.2.1 The invariance result, stated correctly

For a fixed score vector s and strictly increasing h: sᵢ > sⱼ ⟺ h(sᵢ) > h(sⱼ), hence AUC(y, h(s)) = AUC(y, s) [V — follows from the definition of AUC as a pairwise concordance probability; see [sklearn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)].

Implications per transform:

| Transform                   | Type                                    | Effect on AUC of a fixed vector                |
| --------------------------- | --------------------------------------- | ---------------------------------------------- |
| Sigmoid calibration (Platt) | strictly increasing                     | none                                           |
| Logit (with clipping at ε)  | strictly increasing on [ε, 1−ε]         | none (ties only at clipped extremes)           |
| Monotone rescaling          | strictly increasing                     | none                                           |
| Empirical rank map          | strictly increasing (with average ties) | none up to tie handling                        |
| **Isotonic regression**     | **nondecreasing only**                  | **can create ties → can reduce empirical AUC** |

The isotonic row is not theoretical: scikit-learn issue [#16321](https://github.com/scikit-learn/scikit-learn/issues/16321) documents isotonic calibration changing rank-based test metrics. scikit-learn's calibration docs likewise treat calibration as a probability-quality objective, not a ranking objective ([sklearn calibration](https://scikit-learn.org/stable/modules/calibration.html)). **Therefore: do not recommend calibration to repair the LB gap.** Calibration is legitimate only if (a) a downstream operation genuinely needs probabilities (blend arithmetic in probability space does — see below), or (b) it is part of a _modeling_ change that legitimately reorders.

#### 4.2.2 What "calibration shift" actually diagnoses

| Shift type                                          | Changes AUC of fixed model?                                                    | Changes calibration? |
| --------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------- |
| Probability calibration shift (score scale)         | No                                                                             | Yes                  |
| Covariate shift (feature distribution)              | Yes, if conditional P(y                                                        | X) also shifts       | possibly |
| Conditional shift P(y                               | X)                                                                             | **Yes**              | Yes      |
| Label-prevalence shift                              | No (ranking invariant), but _blends_ of recalibrated probabilities can reorder | Yes                  |
| Different training procedures (fold vs. full refit) | Score-scale differences; ranking usually preserved                             | Yes                  |
| Implementation errors                               | Any effect                                                                     | Any effect           |

Do not infer prevalence shift from mean-prediction differences alone; do not estimate test prevalence from unlabeled features (not identifiable).

#### 4.2.3 Inference-time ranking protocols compared

**Protocol A — raw probabilities (baseline).** Blend nothing; use the validated single-model pipeline. Always report it.

**Protocol B — logit-space blending.** s = Σwₘ·logit(clip(pₘ)), with clip ε = 1e-6: `p̃ = min(1−ε, max(ε, p))`. Individual monotone transforms preserve each model's own ranking, but **change the blend** because the component scales change: logit stretches the mid-range and compresses the extremes, so logit blends weight confident-disagreement regions differently than probability blends. Clipping matters: extreme scores (p < 1e-6) would otherwise produce logits of magnitude > 13.8 that dominate any weighted sum. Compare B against A on the same OOF rows; neither dominates universally.

**Protocol C — empirical rank blending.** rᵢₘ = (rank(pₘ(xᵢ)) − 1)/(n − 1), ties averaged; blend s = Σwₘ·rᵢₘ. Rank maps discard probability _spacing_ (all information about how far apart passengers are) and keep only order — this removes scale dependence between base models (useful when one model is systematically over/under-confident) but also discards genuine information when models are differently confident in informative regions. Choose the reference population deliberately:

1. **Fixed training reference (recommended default).** Fit the empirical CDF of each model's _OOF scores_ once; at inference, map test scores through that CDF (interpolated inverse). Strictly monotone, label-free, deterministic, and identical between validation and submission runs. No test data needed.
2. **Transductive reference (ranks over train+test combined).** Permitted by the competition setup — it uses only unlabeled test _scores_, never test labels, and S6E10 permits external/public data usage ([competition page](https://www.kaggle.com/competitions/playground-series-s6e10); community summary: external data allowed, 10 submissions/day, deadline 2026-10-31 [jamesidriss repo](https://github.com/jamesidriss/airline_S6E10)). Assumptions required: the unlabeled test file at validation time is the same file scored at submission time (true here), and you accept that the validation-time rank features (computed over train-only scores) are _not_ identical to inference-time ranks (computed over train+test). This train–test asymmetry must be replicated during validation or it is a silent mismatch.
3. **Fold-specific ranks** — not recommended for blending: they inject fold-boundary noise into OOF scores.

Given the current integrity concerns, adopt option 1 until the pipeline is verified; consider option 2 only as a measured A/B on the same folds.

**Protocol D — monotone post-processing (sigmoid/isotonic).** Use only if probabilities are an actual deliverable or as a component of a justified downstream operation. Expect **no** LB-AUC repair from calibration alone [V].

#### 4.2.4 Train–test consistency taxonomy for every preprocessing step

| Step                                                       | Learned from                     | Cross-fitting needed? | Test-time behavior                        |
| ---------------------------------------------------------- | -------------------------------- | --------------------- | ----------------------------------------- |
| Category encoding                                          | training folds                   | yes, per fold         | apply saved mapping                       |
| Target/label encoding                                      | training folds                   | **mandatory**         | apply fold-specific or averaged encodings |
| Rank normalization                                         | OOF score distribution           | no (labels unused)    | fixed training reference CDF              |
| Duplicate-row statistics (Flight-Distance group stats [H]) | train+test+original (label-free) | no                    | recompute identically                     |
| Early stopping                                             | fold-internal validation         | no                    | fixed iteration count from CV             |
| Original-data satisfaction rate per group [H]              | original labeled data            | no                    | apply as static feature                   |

Fail loudly on any schema/category mismatch at inference (assertions A1–A6 in §4.1.1).

#### 4.2.5 Submission-integrity checklist (automated, blocking)

1. Row count == test row count; ids exactly equal test ids, same order; no duplicates.
2. Column names/position match `sample_submission.csv` (`id,satisfaction`) ([Kaggle overview](https://www.kaggle.com/competitions/playground-series-s6e10)).
3. All predictions finite, in [0,1], no NaN/inf.
4. Reproducibility: re-running inference from saved models yields Spearman ≥ 0.9999 with the submitted vector.
5. Schema/category assertions pass (A1–A5).
6. Prediction distribution sanity: KS statistic between submission scores and OOF scores recorded in the run log (informational; not a blocker).
7. File re-read from disk before submission (catches truncation/encoding).

---

### 4.3 Part III — Constrained rank-weight optimization (Question Three)

#### 4.3.1 The objective and why it is hard

With R = (rᵢₘ) the normalized rank matrix and y the OOF labels:

maximize w ↦ AUC(y, Rw) subject to w ≥ 0, Σwₘ = 1 (optionally wₘ ≤ w_max).

For a finite sample, each pairwise comparison (i,j) flips only when w·(rᵢ − rⱼ) crosses 0 — a hyperplane in weight space. The objective is therefore **piecewise constant on cells of a hyperplane arrangement**: nonsmooth, with large flat regions when base-model rank vectors are highly correlated (small angular differences between rₘ columns ⇒ few flip hyperplanes ⇒ broad plateaus). Consequences: (i) many weight vectors are exactly tied at the optimum — weights are non-identifiable; (ii) gradient-based methods are meaningless; (iii) Nelder–Mead's convergence guarantees assume smoothness and do not apply; (iv) the optimizer's reported optimum is an arbitrary point of a plateau, highly sensitive to initialization.

#### 4.3.2 Nelder–Mead specifically

SciPy's Nelder–Mead supports **box bounds only** (via the `bounds` argument); the simplex constraint Σw = 1 must be handled by reparameterization (e.g., optimize w₁..w_{M−1} with w_M = 1 − Σ, or a softmax parameterization) ([SciPy NM docs](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)). Even then, on a piecewise-constant objective NM can stall in a plateau far from the best cell, and different restarts return different plateaus. **Verdict: NM is acceptable only as a final local polish; it must not be the primary search.** [V on capabilities; H on stall behavior → measure restart variance.]

#### 4.3.3 Recommended search (cheapest first)

1. **Predeclared coarse grid** over the simplex, step 0.05 (for M = 4: 84 points) + **Dirichlet(1) random sampling**, K = 200–1000 draws — all evaluated on pooled OOF AUC. Cost: one AUC evaluation per candidate ≈ O(n log n); trivially parallel.
2. **Coordinate refinement:** from the grid optimum, cyclic coordinate search over wₘ in ±0.05, ±0.02 steps while renormalizing. Two sweeps suffice.
3. Optional NM polish from the grid optimum, reparameterized to the simplex, with 10 restarts; keep the best; record the plateau spread.

**Stopping criterion:** stop when (a) the top-10 candidates' pooled AUC spread < 1e-4 (below the ~7e-4 noise floor, i.e., statistically indistinguishable), and (b) the selected weights' N_eff is stable across restarts (variation < 10%). If (a) triggers, the objective is flat → fall back to the **most-uniform weight vector within 1e-4 of the optimum** (see regularization below).

#### 4.3.4 Domination control: caps vs. penalties

Hard cap: wₘ ≤ w_max (e.g., 0.5 for M = 4 ⇒ N_eff ≥ 2 in the worst case; 0.4 ⇒ N_eff ≥ ~2.9). Penalties (applied to the _validation objective only_, never to the final scoring):

- Distance-from-uniform: Ω(w) = Σₘ(wₘ − 1/M)² (quadratic).
- Concentration: Ω(w) = Σₘwₘ² ⇔ targeting max N_eff.
- Baseline deviation: Ω(w) = Σₘ(wₘ − w⁰ₘ)² for a predeclared baseline w⁰ (e.g., uniform or AUC-proportional).
- Effective-count floor: penalize max(0, K_min − 1/Σwₘ²)² with predeclared K_min (e.g., 2).

Effective number of contributing models: N_eff(w) = 1/Σₘwₘ² — equals M for uniform weights (4 models → 4), 1.9 for (0.7, 0.15, 0.1, 0.05), 1 for a single model. Use it as a **diagnostic** and, only if instability is observed across restarts/fold-subsets, as the regularization target.

**Do not impose caps without evidence.** Protocol: run unconstrained; if the optimum has N_eff ≥ 2.5 and is stable, no cap. If the optimum concentrates (N_eff < 2) _and_ the concentrated solution's advantage over uniform is < 1e-4 (within noise), adopt the regularized (near-uniform) solution — it is more robust to the OOF/LB distribution difference. **λ selection:** predeclare λ ∈ {0, 10⁻⁴, 10⁻³, 10⁻²} on a normalized penalty scale; select the smallest λ whose selected weights are stable under fold-subset jackknife (leave-one-fold-out re-optimization); do not tune λ by repeatedly maximizing pooled OOF AUC — that is selection overfitting.

#### 4.3.5 Why optimal weights differ across blending spaces

Probability, logit, and rank spaces reweight regions of the score scale differently: probabilities emphasize mid-range disagreement; logits stretch the mid-range, compress extremes; ranks discard spacing entirely. Hence argmax_w AUC(y, Σwₘ·T(pₘ)) depends on T. With near-identical GBDTs, rank-space optimization is usually the most stable (scale-free) but also the flattest; logit-space often gives the largest single-candidate improvement when calibration differs across models. Run all three on identical validation rows.

#### 4.3.6 Validation-safe selection protocol (prevents ensemble-selection overfitting)

1. **Selection stage:** optimize weights on pooled folds 1–3 only.
2. **Confirmation stage:** evaluate the top-3 selected weight vectors, blind, on folds 4–5 (never used in selection). Record Δ = confirmation AUC − best single model AUC.
3. **Retention rule:** adopt optimized weights only if confirmation Δ ≥ +0.0002 AND Δ > 0 on both confirmation folds. Otherwise ship equal weights (rank space).
4. Log every candidate evaluated (count matters: with 1,000 candidates on ~560k OOF rows, the winner's-curse inflation is small — SE ≈ 7e-4 — but repeated rounds against the same labels compound it; the confirmation stage is what controls it).

---

### 4.4 Part IV — Shallow GBDT meta-learner (Question One)

#### 4.4.1 Formulation and the four combination methods

Let z(x) = (p₁(x), …, p_M(x)) be base-model scores. A GBDT meta-learner learns ŝ(x) = g_θ(z(x)) with g a permitted GBDT.

| Method               | Model         | Assumptions under which it helps                                                           |
| -------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| Probability blending | Σwₘpₘ         | base scores comparable in scale; linear-in-probability diversity is enough                 |
| Logit blending       | Σwₘ logit(pₘ) | score differences informative on the log-odds scale (mixing differently-calibrated models) |
| Rank blending        | Σwₘ rₘ        | scale-free; optimal when calibration untrustworthy but order informative                   |
| GBDT stacking        | g_θ(z)        | base models have **complementary conditional errors** (region-dependent reliability)       |

Stacking strictly generalizes the linear blends; it earns its complexity only when the conditional-error structure exists. With four same-family GBDTs trained on one dataset, the prior is that it does not [H — measure via ablation].

#### 4.4.2 Leakage-safe training protocol (the non-negotiable core)

**Meta-training features must be cross-fitted:** every meta-training row's base prediction comes from a model that never saw that row's target. Standard references on stacking are unanimous that training the meta-model on in-sample base predictions is optimistic, because in-sample scores are systematically better than any test-time score will be ([Cross Validated: stacking with cross-validation](https://stats.stackexchange.com/questions/239445/how-to-properly-do-stacking-meta-ensembling-with-cross-validation)).

Algorithm:

```
1. For m in 1..M: generate OOF predictions p_m^oof on the established K folds
   (existing registry, unchanged folds, join by id).
2. Assemble Z_oof[i, m] = p_m^oof(id_i); labels y_oof = train satisfaction.
3. Split Z_oof into meta-train / meta-valid folds (meta-level StratifiedKFold,
   e.g., 5 folds, seed fixed).
4. For each meta-fold: fit g_θ (shallow LGBM/CatBoost) on meta-train folds;
   early-stop on the meta-valid fold using AUC.
   → cross-fitted meta-OOF predictions give an unbiased estimate of the
     meta-learner's OOF AUC.
5. Final g_θ: refit on all of Z_oof with iteration count = median of
   fold-wise early-stopping iterations (no early stopping on full data).
6. Test-time: apply g_θ to the test prediction vector Z_test[i, m] = p_m^test(id_i),
   where p^test comes from the SAME base-model definitions and feature logic
   as the OOF generation.
```

**Handling the fold-model vs. full-refit mismatch.** OOF scores come from 80%-data models; test scores from full refits (sharper, better calibrated). Three options, ranked:

1. **Fold-ensemble inference (recommended):** average the K fold models' test predictions per base model. Test scores are then generated under conditions statistically matched to OOF scores (both are averages over 80%-data models). Cost: K× inference time. This also _reduces_ the OOF–test distribution difference flagged in the current campaign.
2. Accept the mismatch with a **shallow meta-learner** (depth 3–4, few splits ⇒ learns near-monotone reweighting, robust to mild scale shift).
3. Do **not** train the meta-learner on in-sample full-refit predictions to "match" — that is stacking leakage [V].

**Hyperparameter/complexity selection for the meta-learner** must itself be cross-fitted (the meta-level folds in steps 3–4); if you then want an unbiased end-to-end score, the selection and the reporting must use disjoint meta-folds — nested CV is needed **only** to support the claim "this hyperparameter choice generalizes"; it is not needed to fit a predeclared shallow configuration.

#### 4.4.3 Meta-feature blocks (ablation matrix)

| Block           | Features                                                | Prior                                                | Keep if                    |
| --------------- | ------------------------------------------------------- | ---------------------------------------------------- | -------------------------- |
| B1 raw          | zₘ = pₘ                                                 | always start                                         | —                          |
| B2 logits       | logit(clip(pₘ, 1e-6))                                   | helps with mixed calibration                         | +Δ confirmed               |
| B3 ranks        | fixed-training-reference rank of pₘ                     | helps with scale mismatch                            | +Δ confirmed               |
| B4 agreement    | mean, sd, min, max, max−min of p; pairwise diffs pₐ−p_b | only if disagreement correlates with errors          | +Δ confirmed               |
| B5 interactions | (implicit in trees: splits on pairs)                    | shallow trees find them natively; no manual products | only if B1+B4 insufficient |

Numerically stable logit: clip to [1e-6, 1−1e-6] before transforming. Rank features: use the §4.2.3 fixed training reference, never fold-specific ranks, never test labels.

**Ablation plan (cumulative):** A0 = best constrained blend (reference); A1 = B1; A2 = B1+B2; A3 = B1+B3; A4 = B1+B4; A5 = B1+B2+B4; A6 = B1+B2+B3+B4. Each candidate: pooled meta-OOF AUC + per-fold AUCs; retain a block only if it adds ≥ +0.0002 pooled with ≥ 4/5 fold-level sign consistency. Block B4 should be tested, not added reflexively — with correlated GBDTs the sd/min/max features are usually near-degenerate.

#### 4.4.4 Anti-overfitting configuration and why 700k rows do not protect you

The meta-learner does not overfit because of sample scarcity; it overfits because the **effective feature dimension is tiny (M = 4 → ≤ 20 derived features)**, the features are highly collinear, and the signal beyond the monotone average is weak — so the trees chase OOF artifacts (fold-boundary noise, per-fold calibration differences). Shallow, heavily regularized configs:

```python
# LightGBM meta-learner (start here)
lgb.LGBMClassifier(
    objective="binary", n_estimators=500, learning_rate=0.03,
    num_leaves=8, max_depth=3, min_data_in_leaf=20000,
    lambda_l2=50, feature_fraction=1.0, bagging_fraction=0.8,
    bagging_freq=1, verbosity=-1)
# early stopping on meta-valid fold with metric "auc", first_metric_only=True

# CatBoost meta-learner (only if LightGBM shows credible improvement)
cb.CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=3, l2_leaf_reg=50,
    random_seed=SEED, od_type="Iter", od_wait=100, eval_metric="AUC")
```

Rationale from the documentation: LightGBM's leaf-wise growth requires `num_leaves` well below 2^max_depth and explicit `min_data_in_leaf` to avoid overfitting ([LightGBM tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html), [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)); CatBoost offers the overfitting detector and depth/l2 tuning for the same purpose ([CatBoost overfitting detector](https://catboost.ai/docs/en/features/overfitting-detector-desc), [CatBoost parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning)).

#### 4.4.5 Objective choice

- **Binary log loss (default) as the training objective, AUC as the early-stopping/selection metric** is the pragmatic choice. [V on availability: LightGBM exposes `auc` as a metric and `binary` as objective; CatBoost exposes `eval_metric="AUC"`.]
- **Pairwise ranking objectives** (LightGBM `lambdarank`) need query groups; a single 700k-row "group" makes the pairwise formulation O(n²) — computationally infeasible here, and the documented use case (learning-to-rank with relevance grades) does not map cleanly onto AUC maximization. [V on mechanics / H on futility]
- Do not assume ranking objectives improve test AUC: most of the achievable gain lives in the base models, not the meta-objective. Decide empirically, once, via the ablation.

#### 4.4.6 Decision rule for the meta-learner

Retain the GBDT meta-learner **only if all hold**:

1. Cross-fitted meta-OOF AUC ≥ best constrained blend + 0.0002 (pooled), **and**
2. ≥ 4/5 meta-folds show positive Δ, **and**
3. The gain survives an independent confirmation stage (§4.3.6 step 2 analog at the meta level), **and**
4. The learned function is sanity-checked: PDP/ICE plots of g vs. each zₘ should be monotone-ish; wild non-monotone swings on 700k rows indicate fitting OOF artifacts.

Otherwise: reject, ship the constrained blend. This rule is deliberately conservative: a rejected meta-learner costs a few GPU-hours; an accepted-but-spurious one costs leaderboard position and weeks of confusion.

---

### 4.5 Experiment matrix

| Priority | Experiment                                                    | Method                                      | Expected information gain                        | Cost             | Success criterion                                         |
| -------- | ------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------ | ---------------- | --------------------------------------------------------- |
| 1        | Category-mapping + schema audit; submission-integrity harness | Assertions A1–A6, blocking                  | Highest: directly tests the stated inversion     | ~1 day eng.      | all assertions pass; resubmitted LB within ~0.0007 of OOF |
| 2        | Reproduce current OOF + test predictions end-to-end           | re-run from saved models                    | Confirms/denies reproducibility defect           | ~0.5 day compute | Spearman(re-run, submitted) ≥ 0.999                       |
| 3        | Equal-weight baselines: probability, logit, rank              | pooled OOF AUC per candidate on same rows   | Establishes the blend ceiling vs. best single    | ~minutes         | recorded; rank-vs-prob Δ noted                            |
| 4        | Cross-fold near-duplicate census                              | feature-tuple hashing train↔test, fold↔fold | Quantifies OOF optimism component                | ~1 day           | dup-rate table; gap explanation updated                   |
| 5        | Constrained rank-weight search (grid + Dirichlet + polish)    | §4.3.3, selection folds 1–3, confirm 4–5    | Whether optimization beats equal weights, safely | ~hours           | confirmation Δ ≥ +0.0002, N_eff stable                    |
| 6        | Probability & logit constrained search (same protocol)        | same as 5, different spaces                 | Space sensitivity of optimal weights             | ~hours           | same rule, compared on same rows                          |
| 7        | Shallow LGBM meta-learner, ablation A1–A6                     | §4.4.2–4.4.4                                | Whether conditional-error structure exists       | ~1–2 days        | decision rule §4.4.6                                      |
| 8        | CatBoost meta-learner                                         | only if 7 credible                          | Cross-family check of stacking gain              | ~1 day           | same rule                                                 |
| 9        | Transductive rank reference A/B                               | §4.2.3 option 2                             | Whether combined-population ranks help           | ~hours           | confirmed Δ ≥ +0.0002; else keep fixed reference          |
| 10       | Repeated cross-fitting / seed variance for OOF                | repeat 3 seeds                              | OOF variance estimate; noise floor               | ~days GPU        | only if Tier-1 gap persists unexplained                   |

Add no candidate that does not answer a distinct research question.

---

## 5. Source Notes

| Source                                                                                                                                                                                                    | Credibility | Last updated       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------ |
| [Kaggle Playground Series S6E10 — Predicting Airline Satisfaction (overview & evaluation)](https://www.kaggle.com/competitions/playground-series-s6e10)                                                   | 5/5         | 2026               |
| [Kaggle S6E10 — Data page (synthetic data provenance)](https://www.kaggle.com/competitions/playground-series-s6e10/data)                                                                                  | 5/5         | 2026               |
| [scikit-learn — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)                                                                                      | 5/5         | 2026 (v1.9.1 docs) |
| [scikit-learn — Metrics and scoring (AUC semantics)](https://scikit-learn.org/stable/modules/model_evaluation.html)                                                                                       | 5/5         | 2026 (v1.9.1 docs) |
| [scikit-learn — Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)                                                                                                        | 5/5         | 2026 (v1.9.1 docs) |
| [scikit-learn GitHub issue #16321 — isotonic calibration changes rank-based metrics](https://github.com/scikit-learn/scikit-learn/issues/16321)                                                           | 4/5         | -                  |
| [LightGBM — Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)                                                                                                                        | 5/5         | 2026 (v4.7 docs)   |
| [LightGBM — Parameter Tuning (leaf-wise overfitting, num_leaves, min_data_in_leaf)](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)                                                     | 5/5         | 2026 (v4.7 docs)   |
| [CatBoost — Overfitting detector](https://catboost.ai/docs/en/features/overfitting-detector-desc)                                                                                                         | 5/5         | 2026               |
| [CatBoost — Parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning)                                                                                                                      | 5/5         | 2026               |
| [SciPy — minimize(method='Nelder-Mead') (bounds support, no simplex constraint)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)                                            | 5/5         | 2026 (v1.18 docs)  |
| [Cross Validated — How to properly do stacking/meta-ensembling with cross-validation](https://stats.stackexchange.com/questions/239445/how-to-properly-do-stacking-meta-ensembling-with-cross-validation) | 3/5         | -                  |
| [aditbytes/kaggle-s6e10-airline-satisfaction (CV–LB agreement ~0.0006; feature AUCs; no train/test drift)](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction)                                | 3/5         | 2026-10-05         |
| [ChrisLegge/kaggleAirlineSatisfaction (LightGBM 0.95874→0.96071 with group statistics)](https://github.com/ChrisLegge/kaggleAirlineSatisfaction)                                                          | 3/5         | 2026               |
| [jamesidriss/airline_S6E10 (rules summary: 10 subs/day, external data allowed, deadline 2026-10-31)](https://github.com/jamesidriss/airline_S6E10)                                                        | 3/5         | 2026-10            |
| [Airline Passenger Satisfaction — original dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)                                                                           | 4/5         | -                  |

**Conflicts and caveats.** (1) The claimed "public LB best 0.96243" and the reported "OOF 0.96126 / LB 0.95975" figures come from the campaign's own reports and could not be verified against the live leaderboard here. (2) Community repositories (3/5 credibility) are single-team evidence; the ~0.0006 CV–LB agreement claim is one team's observation, not a leaderboard statistic. (3) The AUC standard error used in §2/§4.1 is a large-sample (Hanley–McNeil–style) approximation; exact variance depends on the score distribution shape. (4) Class prevalence ~55% is taken from the original dataset lineage [H] and must be measured on the actual S6E10 train file. (5) The near-duplicate mechanism (§4.1.4) is a hypothesis grounded in the 700k-from-130k synthesis; the duplication rate is measurable but was not measured here.

---

## 6. Open Questions

1. **Which category was inverted, and did the inversion change the test ranking?** Requires the repo's saved encoders and the submitted vector; the Spearman(re-run, submitted) ≥ 0.999 test (§4.1.1) settles it.
2. **What is the actual cross-fold vs. train↔test near-duplicate rate?** Determines how much of the 0.00151 gap is structural OOF optimism rather than a bug.
3. **What are the base models' pairwise Pearson/Spearman correlations and per-fold AUCs?** The blending-gain ceiling (Executive Summary point 3) is an estimate until measured; if some pair has ρ < 0.97, blending upside is larger than assumed.
4. **Were OOF predictions generated with fold-specific or full-data target encodings?** If full-data, OOF is contaminated and all pooled-OOF comparisons in this report are slightly optimistic.
5. **Is fold-ensemble (fold-averaged) inference affordable at submission time?** Determines whether the meta-learner's OOF/test mismatch mitigation (§4.4.2 option 1) is available.
6. **Does the test score-distribution shift persist after the inversion fix?** If yes with a verified-clean pipeline, a genuine conditional-distribution difference between synthetic train and test is implied — a much harder problem.

---

## 7. Recommendations / Next Steps

**Single best next experiment (Tier 1, one week):**
Fix and verify the categorical pipeline, then resubmit _the same ensemble_ unchanged.

- **Exact changes:** implement assertions A1–A6 as blocking checks in the submission path; persist one category mapping used by every fold, refit, and inference path; rebuild test predictions from saved fold models; regenerate the submission; log Spearman(old submission, new submission).
- **Evidence to retain the "inversion explains the gap" hypothesis:** new LB score within ~0.0007 of OOF 0.96126, or Spearman(old, new) materially < 1 with the new vector scoring higher.
- **Rejection condition:** new LB unchanged (±0.0002) → the inversion did not materially affect ranking; move to the duplicate census (experiment 4) and distribution diagnostics before any blending work.
- **Next experiment either way:** Tier 2 — equal-weight rank/probability/logit baselines (experiment 3), then the constrained coarse simplex search with the fold-1–3 selection / fold-4–5 confirmation protocol (experiments 5–6). Adopt optimized weights only on confirmation Δ ≥ +0.0002 with stable N_eff; otherwise ship the equal-weight rank blend.
- **Then, only if Tiers 1–2 leave budget:** the shallow LGBM meta-learner ablation (experiment 7) under the §4.4.6 decision rule; the CatBoost meta-learner (experiment 8) only if 7 is credible.

**Standing principles for this campaign:** calibration is never the LB-AUC fix; equal weights are the null hypothesis any optimizer must beat out-of-sample; the meta-learner must beat the _blend_, not just the best single model; and no weight vector or meta-config is adopted on pooled-OOF evidence alone — it must survive a confirmation fold that never participated in selection.
