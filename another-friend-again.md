# S6E10 Grandmaster Research Brief

**Kaggle Playground Series S6E10 — Predicting Airline Satisfaction (ROC-AUC)**

## Question

Can leakage-free GBDT stacking, mathematically constrained blend optimization, stronger validation (5-fold vs 10-fold), and ranking postprocessing lift a ~0.96076 public-LB / ~0.96128 OOF GBDT-only ensemble (CatBoost, LightGBM, LightGBM Extra-Trees, XGBoost) toward 0.96243+, reproducibly and without target leakage?

## Methodology

- **Sources.** Official competition page and public notebook/code listing (LB scores and community experiment titles); seven public S6E10 solution repositories with recorded CV/LB numbers and ablations (ayman-43, bansal1600, ChrisLegge, aditbytes, wozhenshiliudada, ALPHAMAN-0, mrjohnsonsea, BSergey2000); official LightGBM / XGBoost / CatBoost parameter documentation; Caruana et al. (2004) ensemble selection; Kaggle/CrossValidated threads on stacking validation.
- **What was inspected.** Because no local repository, saved predictions, or baseline notebook were available in this workspace, Phase 1 (repository inspection) was replaced by an audit of the published S6E10 ecosystem with recorded scores. **Every number below is attributed to its public source; no benchmark score in this report was fabricated or run locally.** All code blocks are executable blueprints, not executed results.
- **Limitations.** (a) Public repos self-report scores; their fold definitions, feature sets, and blend code could not be independently executed. (b) The current 0.96128 OOF ensemble's exact composition is unknown to this brief, so correlations and weight-stability claims are framed as audits to run, not as established facts. (c) Kaggle discussions referenced by title ("10 OOF members lose 0.000245 AUC", "Can the Leaderboard See a Stack Gain?") were confirmed to exist but their full bodies could not be read; only their titles/existences are used as directional evidence.

---

## 1. Executive verdict

**Ranked by expected value:**

1. **Stronger members, not a smarter combiner, is where the points are.** The single largest documented gain in this competition is feature engineering, not ensembling: label-free Flight-Distance group statistics (row counts, rating means, original-data satisfaction rates per Flight-Distance value, computed over train+test+original with no target leakage) lifted one LightGBM from 0.95874 to 0.96071 — roughly +20 bp, an order of magnitude larger than any combiner improvement available ([ChrisLegge repo](https://github.com/ChrisLegge/kaggleAirlineSatisfaction)). Auxiliary "expected-rating" features added ~+0.9 bp per tree model and much more for NN members ([ayman-43 repo](https://github.com/ayman-43/-kaggle-s6e10-airline-satisfaction)).
2. **Constrained rank/logit blending with concentration control is the most reliable combiner.** Community evidence shows optimized weights and stackers buy only single-digit bp on this dataset: bansal1600's nested hill-climb reached OOF 0.961643 but still _below_ their nested logistic-regression stacker ([PR #4](https://github.com/bansal1600/kaggle-airline-prediction/pull/4)); adding CatBoost to ayman-43's stack changed CV by 0.0000001 (0.9615343 → 0.9615332) ([ayman-43](https://github.com/ayman-43/-kaggle-s6e10-airline-satisfaction)).
3. **A GBDT meta-learner is a low-expected-value, high-leakage-risk addition.** Four GBDT arms trained on the same features produce near-collinear OOF predictions; a depth-2 meta-tree can at best reweight them, which constrained blending already does with far fewer degrees of freedom. Test it last, with minimal meta-features, inside a nested design, and expect ~0 to +1 bp.
4. **0.96243+ is not reachable by combiner work alone under the GBDT-only constraint.** As of 2026-10-10/11, the best public notebooks sit at 0.96176–0.96184 ([Kaggle code listing](https://www.kaggle.com/competitions/playground-series-s6e10/code)), and those include RealMLP/TabPFN members that our constraint excludes. A realistic GBDT-only ceiling based on public evidence is ~0.9618–0.9620 LB. Treat 0.96243+ as achievable only via member strength (features + external original data), not via blending arithmetic. **Do not promise it.**
5. **Postprocessing cannot create score.** ROC-AUC is invariant to strictly increasing transformations of a single final score (mathematical property). Every postprocessing idea should be judged only by numerical safety and tie handling.

## 2. Forensic audit of current assumptions

| Assumption                                                 | Verdict                                     | Evidence & reasoning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Adversarial validation AUC ≈ 0.500 ⇒ same DGP              | **Partially supported, weakly informative** | An AUC ≈ 0.5 means one particular classifier, on one feature representation, cannot separate train from test. One repo reports all feature means differ <2% between train and test ([aditbytes](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction)), consistent with no covariate shift. But AUC ≈ 0.5 does **not** rule out: concept drift in P(y\|x); shift detectable only in engineered/interaction representations the adversarial model lacked capacity to use; shift concentrated in rare regions. **Required audit:** rerun adversarial validation with the _final engineered feature set_, both a deep GBDT and a shallow one, and with id-like columns excluded. FastML documents the procedure and its interpretation ([FastML](https://fastml.com/adversarial-validation-part-one/)). |
| OOF ≈ generalization, LB gap needs explanation             | **Supported with caveats**                  | OOF AUC is (approximately) unbiased for the _procedure that produced it_ under IID rows — not for a procedure that also selected hyperparameters, members, and blend weights using the same OOF. Community CV→LB decays on this competition: 0.96175→0.96140 (ayman-43), 0.9591→0.9584 (BSergey2000), and "CV and LB agree within 0.0006" (aditbytes). Our 0.96128→0.96076 (−5.2 bp) is in family with these; the components to quantify are (a) public-LB sampling variance (public LB is a subsample of the 300k test rows), (b) selection overfitting across experiments, (c) fold-seed variance. **Do not claim zero decay is guaranteeable.**                                                                                                                                                             |
| Base predictions highly correlated                         | **Plausible, must be re-verified**          | Correlation is only meaningful computed row-aligned on genuine OOF predictions. Corroborating signal: adding CatBoost to ayman-43's stack moved CV by ~1e-7; bansal1600 found a second auxiliary-feature RealMLP carrier "adds nothing" ([PR #6](https://github.com/bansal1600/kaggle-airline-prediction/pull/6)). **Required audit:** Spearman and Pearson correlations of OOF predictions on identical rows; report the incremental gain of each member over the best blend without it (leave-one-out ablation), which is the decision-relevant quantity — correlation itself is not (rule 8).                                                                                                                                                                                                               |
| 10-fold proposed to improve test-time generalization       | **Inconclusive; do not claim 10–15 bp**     | No public S6E10 evidence supports a 10-fold LB gain. Ten-fold gives each base model ~11% more training data per fold (90% vs 80%), which _on average_ slightly raises each fold model — this raises OOF realism, not the test score of the final model. wozhenshiliudada used 10-fold members but explicitly warns their blend validation reused historical OOF and was _not_ fully nested ([repo](https://github.com/wozhenshiliudada/kaggle-airline-satisfaction-s6e10)). Ten-fold costs 2× compute for inner OOF and yields fold scores that are more correlated (more training-set overlap). Verdict: run the matched-seed A/B in Section 6; assume no LB gain until measured.                                                                                                                             |
| Pipeline generates predictions/folds/checkpoints correctly | **Cannot be audited here**                  | No repository was available in this workspace. Section 10 provides the leakage-assertion harness to run against the actual pipeline: OOF coverage (every training row has exactly one out-of-fold prediction), fold-index persistence, id-alignment checks, and an in-sample-substitution detector.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

**Why OOF is not an oracle for the leaderboard (mechanism, not folklore):** (i) OOF estimates E[AUC] of the _fold-model ensemble_ (models trained on (K−1)/K of data), while the submitted model's test predictions come from a different object (fold models averaged over full training runs or refits on 100%); (ii) public LB uses only a fraction of test rows, adding ± a few bp of pure sampling noise; (iii) any decision made by looking at OOF (weights, members, meta-features) transfers some of OOF's noise into the procedure, a selection bias the OOF number itself cannot display. Cross-validated community CV→LB gaps above are the empirical size of this combined effect.

## 3. Stacking architecture (Track A)

### 3.1 Exact fold logic

Notation: outer folds O = {o1..o5} (stratified, seed 42, persisted to disk). Within each outer-training partition, inner folds I(o) = {i1..iK}, K ∈ {5, 10} under test.

```
FOR each outer fold o in O:
    OUTER_TRAIN := train \ o ; OUTER_VAL := o          # OUTER_VAL labels sealed until scoring
    FOR each inner fold j in I(OUTER_TRAIN):
        FOR each base arm k in {LGBM, LGBM_ETS, XGB, CAT}:
            f_k^(o,-j) := fit(arm_k, OUTER_TRAIN \ j)        # target-encoding / supervised FE fit HERE only
            z[i,k] := f_k^(o,-j)(x_i)  for i in j            # inner OOF predictions
    META := fit_metalearner(Z_inner, y_inner)                 # uses only inner-OOF rows/labels
    # size-matched base refit for OUTER_VAL predictions (see 3.3):
    FOR each inner fold j in I(OUTER_TRAIN):
        g_k^(o,j) := fit(arm_k, OUTER_TRAIN \ j)
        u[i,k] := g_k^(o,j)(x_i)  for i in OUTER_VAL         # same (1 - 1/K) training fraction as z
    score_o := AUC( META(u_o), y_o )                          # OUTER_VAL labels used exactly once
FINAL: average score_o over o; never report inner-OOF meta scores as system performance
```

Key invariants (enforced by assertions in Section 10):

- No model that saw row i's target ever predicts z[i,·] or u[i,·].
- Any supervised preprocessing (target encoding, learned calibrations, aux-feature models fitted on competition labels) is fit inside the same fold boundary as the base model.
- OUTER_VAL labels are touched only by the final AUC call.

### 3.2 Meta-features (build minimal first, expand only on ablation evidence)

For prediction p and documented clipping constant ε = 1e-6: logit_ε(p) = log( clip(p, ε, 1−ε) / (1 − clip(p, ε, 1−ε)) ).

| Meta-feature                                             | Status                       | Rationale                                                                                                                                                                            |
| -------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Raw probabilities p_k                                    | **Core**                     | Baseline representation.                                                                                                                                                             |
| Clipped logits logit_ε(p_k)                              | **Core**                     | Compresses mid-range, expands tail separation; makes near-collinear predictions less redundant in the tails.                                                                         |
| Per-row mean of p_k                                      | Optional                     | Nearly a function of the p_k's; a meta-tree can build it internally with one split; low incremental value expected.                                                                  |
| Per-row std of p_k, max−min spread                       | **Ablate, expect rejection** | With 4 near-collinear GBDT arms these are ~monotone functions of                                                                                                                     | p_best − p_worst | ; "model disagreement" is not validated uncertainty. Only include if leave-one-out outer-fold AUC improves. |
| Pairwise differences p_a − p_b (6 pairs)                 | Ablate                       | Same concern; 6 extra features for 4 arms invites meta-overfit on 700k rows (still cheap, but keep the minimal-first discipline).                                                    |
| Rank transforms of p_k                                   | Optional                     | Scale-free; interacts with blend-space comparison (Section 5).                                                                                                                       |
| Interactions with Class / Type of Travel / Customer Type | Ablate last                  | The only plausible use is segment-conditional model trust; with correlated same-feature arms there is little differential signal to gate on. Test as a single grouped feature block. |

### 3.3 Training-size mismatch (the subtle stacking bug)

Inner-OOF features z come from models trained on (1 − 1/K) of outer-train; the final production base models train on 100% of data and their predictions are _sharper_ (more confident, pushed toward 0/1). A meta-learner trained on z-distribution and applied to sharper predictions faces covariate shift in its inputs. Mitigations, in order of cost:

1. **Apply the meta-learner only to predictions from fold models at the same training fraction** (the `u[i,k]` above): the production analogue is to score test with the average of the K fold models — the same objects that produced OOF — rather than a refit-on-100% model. This is the standard, cheapest fix.
2. **Rank/logit-meta inputs** reduce (not remove) sharpness sensitivity.
3. **Cross-fitted size matching** (train scratch fold models at the production fraction and re-generate meta-training rows) — expensive; only if a measured z-vs-u distribution shift is demonstrated (KS test on prediction marginals per arm).

**Deliverable A status:** architecture defined above; the honest prior from public evidence is that Track A yields ~0 to +1 bp over Track B, and must not be adopted unless the nested outer score beats the constrained blend under matched folds.

## 4. Meta-learner configuration (Track A2)

```python
# LightGBM meta-learner — initial hypothesis, NOT a validated optimum.
# Consistency: num_leaves=4 and max_depth=2 are mutually consistent because
# 2^max_depth = 4 = num_leaves; LightGBM grows leaf-wise, so num_leaves is the
# primary complexity control and max_depth=2 guarantees no leaf can sit deeper.
# Native parameter name for feature subsampling is feature_fraction
# (aliases: sub_feature, colsample_bytree) — use the native name in raw
# param dicts: colsample_bytree has historically been silently ignored in
# some LightGBM APIs (microsoft/LightGBM#1011).
META_LGBM = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.01, "max_depth": 2, "num_leaves": 4,
    "min_child_samples": 500,          # ~0.07% of 700k rows per leaf
    "feature_fraction": 0.6,           # native name; alias colsample_bytree
    "lambda_l1": 1.0, "lambda_l2": 5.0,
    "bagging_fraction": 0.8, "bagging_freq": 1,
    "seed": 42, "verbosity": -1,
}   # n_estimators chosen ONLY by early stopping on an inner-held-out fold

# CatBoost meta-learner alternative (depth 2–3, regularized, with bagging)
META_CAT = dict(
    iterations=5000, learning_rate=0.02, depth=2,     # symmetric (oblivious) trees
    l2_leaf_reg=10.0, loss_function="Logloss",
    eval_metric="AUC", random_strength=1.0,
    bagging_temperature=1.0, border_count=128,
    random_seed=42, early_stopping_rounds=200, use_best_model=True,
)
```

Required ablations (each must win on **outer** folds to survive): fewer leaves (num_leaves 2, max_depth 1 — a stump ensemble is the honest floor); min_child_samples 2000/5000; lambda_l2 10/25; feature blocks: {p} vs {p, logit} vs {p, logit, mean/std/spread} vs +interactions; learning_rate 0.005; no early stopping vs early stopping (early stopping on inner-OOF rows that the meta-learner trains on is _itself_ mildly optimistic — hold out one inner fold for meta early stopping).

**Do not force weight on every base model.** If the meta-learner ignores an arm (feature importance ≈ 0 across outer folds), that is evidence the arm carries no incremental signal — investigate why (likely feature-set overlap), rather than engineering features to rescue it (rules 8–9).

Framework facts used above: `feature_fraction` aliases and defaults ([LightGBM docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html)); cumulative `colsample_bytree/bylevel/bynode` semantics in XGBoost ([docs](https://xgboost.readthedocs.io/en/stable/parameter.html)); CatBoost `depth`, `l2_leaf_reg`, overfitting detector and `use_best_model` ([docs](https://catboost.ai/docs/en/concepts/parameter-tuning)).

## 5. Constrained blend optimization (Track B)

### 5.1 Three blend spaces on identical validation rows

- **Probability blend** s = Σ w_k p_k: dominated geometrically by mid-range predictions; arms that differ mainly in tails contribute little.
- **Logit blend** s = Σ w_k logit_ε(p_k): the logit is strictly increasing, so each arm's _individual_ AUC is unchanged, but _relative_ arm magnitudes in the tails are amplified; blends of near-collinear arms get more tail resolution.
- **Rank blend** s = Σ w_k r_k with r_k = normalized rank (average ranks for ties, divided by n−1): discards magnitude entirely; maximally robust to calibration/sharpness mismatch across arms; slightly lossy where inter-rank distances carry signal. Community practice on Kaggle is to rank-normalize before hill climbing for exactly this scale-free reason ([hillclimbers](https://github.com/Matt-OP/hillclimbers), [jtlocati blend writeup](https://github.com/jtlocati/Kaggle-Car-Prediction)).
- Different spaces can and do give different AUCs because the _pairwise ordering of blended scores_ differs — AUC is a functional of orderings, and a convex combination is not order-preserving across representations.

### 5.2 Objective and optimizer

Empirical AUC is a U-statistic over positive/negative pairs: piecewise-constant in w ⇒ non-smooth, with flat plateaus. Consequences: gradient methods are inappropriate on the empirical objective; a smooth penalty λR(w) turns the objective into "piecewise-constant + smooth", whose minima sit at plateau edges — the penalty trades mean for variance but cannot create gradient signal where AUC is flat. Practical policy: derivative-free optimizers + explicit plateau handling + outer validation of the selected w.

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

EPS = 1e-6  # documented clipping constant for logit space

def logit_eps(p):
    q = np.clip(p, EPS, 1 - EPS); return np.log(q / (1 - q))

def make_space(P, space):        # P: (n_rows, K) OOF predictions of K arms
    if space == "prob":  return P.copy()
    if space == "logit": return logit_eps(P)
    if space == "rank":
        R = np.empty_like(P)
        for k in range(P.shape[1]):
            R[:, k] = np.argsort(np.argsort(P[:, k], kind="mergesort")) / (len(P) - 1)
        return R
    raise ValueError(space)

def auc_of_w(w, U, y):  # U: blend-space matrix
    return roc_auc_score(y, U @ w)

def concentration(w, K):          # R(w) = sum_k (w_k - 1/K)^2
    return np.sum((w - 1.0 / K) ** 2)

def fit_blend_slsqp(U, y, lam=0.0, max_weight=1.0):
    """Constrained: w >= 0, sum w = 1, w <= max_weight, minimize -AUC + lam*R."""
    K = U.shape[1]; w0 = np.full(K, 1.0 / K)
    res = minimize(
        lambda w: -auc_of_w(w, U, y) + lam * concentration(w, K),
        w0, method="SLSQP",
        bounds=[(0.0, max_weight)] * K,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()

def fit_blend_nm_softmax(U, y, lam=0.0, restarts=4, seed=42):
    """Nelder–Mead on unconstrained a in R^K with softmax weights —
    simplex-feasible by construction (parameterization, not projection)."""
    K = U.shape[1]; rng = np.random.default_rng(seed)
    def obj(a):
        z = np.exp(a - a.max()); w = z / z.sum()
        return -auc_of_w(w, U, y) + lam * concentration(w, K)
    best_w, best = None, np.inf
    for r in range(restarts):
        a0 = np.zeros(K) if r == 0 else rng.normal(0, 1.0, K)
        res = minimize(obj, a0, method="Nelder-Mead",
                       options={"maxiter": 400 * K, "xatol": 1e-4, "fatol": 1e-8})
        if res.fun < best:
            best = res.fun
            z = np.exp(res.x - res.x.max()); best_w = z / z.sum()
    return best_w

def hill_climb_caruana(U, y, n_picks=40):
    """Greedy ensemble selection with replacement (Caruana 2004):
    start from the best single arm; each round add the arm (possibly a repeat)
    that most improves blended AUC. Weight of arm k = picks_k / n_picks.
    With-replacement = implicit weight cap 1/n_picks increments."""
    K = U.shape[1]; scores = [roc_auc_score(y, U[:, k]) for k in range(K)]
    counts = np.zeros(K, dtype=int); counts[int(np.argmax(scores))] = 1
    for _ in range(n_picks - 1):
        gains = []
        for k in range(K):
            counts[k] += 1
            w = counts / counts.sum()
            gains.append((auc_of_w(w, U, y), k))
            counts[k] -= 1
        _, k_best = max(gains); counts[k_best] += 1
    return counts / counts.sum()
```

Protocol: all strategies are fit on **inner-OOF rows only**; the _selected_ strategy (including its λ and its blend space) is scored once on the untouched outer fold. Compare: equal weights; predeclared small grid (w in {0, .25, .5, .75, 1} simplex grid for K=4); Caruana hill-climb; SLSQP λ ∈ {0, 1e-4, 1e-3}; NM-softmax; max_weight ∈ {0.4, 0.5} variants. Public S6E10 evidence to calibrate expectations: nested hill-climb OOF 0.961643 was still below a nested LR stacker ([bansal1600 PR#4](https://github.com/bansal1600/kaggle-airline-prediction/pull/4)) — combiner choice moves bp, not tens of bp.

### 5.3 Weight-stability diagnostics

- **Bootstrap**: resample inner-OOF rows (B ≥ 100), refit w, report weight quantiles; flag any arm whose weight crosses 0 in > 25% of resamples.
- **Fold sensitivity**: refit on each inner fold separately; report per-fold w and the max coordinate range.
- **Seed sensitivity**: rerun base arms with seeds {42, 43, 44} on one outer fold; measure Spearman correlation of the blended test scores across seeds; prefer configurations with rank correlation > 0.999.
- **Concentration report**: always report R(w); a λ=0 optimum at a vertex (one arm weight ≈ 1) on correlated arms is an overfitting signature, not a discovery.
- If evidence _consistently_ favors a subset (e.g., an arm with weight ≈ 0 across all outer folds and bootstraps), drop it (rule 9 permits; diversity is not a goal in itself).

## 6. Nested CV: 5-fold vs 10-fold (Track C)

**Design.** Outer: 5 stratified folds, seed 42, fixed for all experiments. Inner: generate OOF twice per outer-training partition — once with 5 inner folds, once with 10 — with matched preprocessing, matched model configs, and matched seeds. Score the complete procedure (base arms → meta-features → blend/meta fit → size-matched outer-val predictions) on each outer fold, for both inner fold counts.

Measured per configuration: mean outer AUC; per-outer-fold spread; wall-clock and peak memory of inner-OOF generation; base-model training-set size (0.80·0.8N vs 0.90·0.8N of full data); prediction stability (Spearman of blended scores across the two inner fold counts).

**Statistics warning:** outer-fold scores are dependent (shared training rows across folds); a naive paired t-test over 5 folds is anti-conservative. Use the corrected resampled t-test (Nadeau–Bengio correction) or report effect size with an explicit "directional, n=5" caveat; never claim significance from 5 dependent numbers.

**Expected outcome (hypothesis, not a promise):** 10-fold inner OOF shifts each fold-model up by ~1% more training data — a small, mostly _variance-free_ change to OOF realism; final test predictions come from fold models either way. Compute cost doubles. Adopt 10-fold only if it wins mean outer AUC by more than its fold-to-fold variability under matched seeds. wozhenshiliudada's 10-fold work explicitly flags non-nested blend validation as its weak point ([repo](https://github.com/wozhenshiliudada/kaggle-airline-satisfaction-s6e10)) — the design above closes exactly that hole.

**Staged approximation if compute is tight:** run the full nested design on **one** outer fold first (cf. bansal1600's one-fold nested probe, [PR #7](https://github.com/bansal1600/kaggle-airline-prediction/pull/7)). This can establish: leakage-freedom mechanically, direction and rough size of stack-vs-blend deltas, and training-size-shift diagnostics. It **cannot** establish mean performance or variance across folds — say so explicitly in any writeup.

## 7. Test-time ranking and postprocessing (Track D)

**Mathematical invariants (guaranteed):**

- For a single final score s, any strictly increasing transform T (logit, raw, monotone calibration, rank-normalization) leaves ROC-AUC exactly invariant up to tie-handling: AUC = P(s⁺ > s⁻) + ½P(s⁺ = s⁻), and strictly increasing maps preserve both the strict-order event and the tie event. Therefore: no clipping, quantile mapping, or calibration applied _to the one final score_ can raise theoretical AUC. Claiming otherwise is a mathematical error.
- Transformations applied **per-model before blending** are a different object: they change the blend geometry and _can_ change ensemble AUC (Section 5.1). Rank-transforming each arm before blending is legitimate; monotone-transforming the final score is not useful.
- Any transformation that creates ties where the ensemble score had none (e.g., coarse quantile bucketing, aggressive clipping at quantiles, float16 truncation) can only _lower_ AUC (the ½-tie term strictly loses against a strict ordering, in expectation under the U-statistic).

**Numerical-safety policy (the only justified clipping):** clip only against pathology: probabilities outside [0,1] from float error; logits of p ∈ {0,1} exactly (use logit_ε, ε = 1e-6, before any logit-space averaging or meta-feature construction — documented, not tuned). Never clip as a "variance reduction" device: on a 300k-row test set, tail ranks carry real AUC mass, and clipping at e.g. the 1st/99th percentile merges tail rows into ties by construction.

**Testable-only hypotheses (validate on outer folds, never on the LB):**

- Rank-blend vs logit-blend of the same arms (changes ensemble ordering — Section 5).
- Average-rank tie handling in the rank space (ties from different arms landing on the same rank).
- Whether to blend fold-model test predictions by averaging probabilities vs averaging logits (an ensemble-ordering question, not postprocessing).

**Leaderboard policy:** the public LB is a noisy diagnostic. Prefer the configuration that wins _mean outer AUC_ and shows stable weights across bootstraps/folds/seeds over any configuration chosen for +2 bp on one public-LB submission. Public notebooks' scores (0.96176–0.96184) are a calibration anchor for "what good looks like", not an optimization target ([code listing](https://www.kaggle.com/competitions/playground-series-s6e10/code)).

## 8. Ablation matrix

| Change                                                                          | Expected benefit (OOF/LB)                                                  | Evidence confidence                                                                    | Compute cost | Implementation risk               |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------ | --------------------------------- |
| Equal-weight rank blend of 4 arms                                               | +1–3 bp vs best single arm                                                 | High (math + community practice)                                                       | None         | None                              |
| Rank vs logit vs prob blend space                                               | ±1 bp; rank most robust                                                    | Medium                                                                                 | None         | Low                               |
| Constrained weight optimization (SLSQP/NM/hill-climb)                           | 0 to +2 bp vs equal weights                                                | Medium (nested hill-climb 0.961643 < LR stack, bansal1600 PR#4)                        | Low          | Medium (weight overfit)           |
| Concentration penalty / max-weight cap                                          | Variance ↓, mean ≈ 0                                                       | Medium–high                                                                            | Low          | Low                               |
| GBDT meta-learner (depth 2, minimal features)                                   | ~0 to +1 bp                                                                | Low (CatBoost added ~0 to ayman-43 stack)                                              | Medium       | High (leakage/size-shift bugs)    |
| Extra meta-features (std, spread, pairwise, interactions)                       | ~0, likely redundant                                                       | Low–medium                                                                             | Low          | Medium (meta-overfit)             |
| 10-fold vs 5-fold (matched seeds)                                               | ≤ +1 bp OOF realism; LB unclear                                            | Low (no public evidence)                                                               | 2× inner-OOF | Medium                            |
| **Flight-Distance label-free group stats (+ original-data satisfaction rates)** | **+20 bp single-model (0.95874 → 0.96071, ChrisLegge)**                    | **High (community, reproduced in repo)**                                               | Medium       | Low (keep label-free & fold-safe) |
| Original ~130k real dataset as training augmentation                            | Several bp (multiple repos adopt)                                          | Medium–high (check competition rules permit external data — Playground typically does) | Medium       | Low–medium (row-mix validation)   |
| Aux "expected-rating" features                                                  | +0.9 bp per tree model (ayman-43); larger for NN members (excluded for us) | Medium                                                                                 | Medium       | Low                               |
| Seed averaging per arm (3–5 seeds)                                              | +0.3 bp (ChrisLegge)                                                       | Medium                                                                                 | 3–5× per arm | None                              |
| Any postprocessing of the final score (clip/quantile/smooth)                    | 0 by theorem; ≤ 0 with ties                                                | High (mathematical)                                                                    | Trivial      | Negative-only                     |
| Repeated method selection on the same OOF (anti-pattern)                        | Inflates OOF, not LB                                                       | High (wozhenshiliudada validation-boundary notes)                                      | —            | High — forbid by protocol         |

**Strategic read:** rows 1–4 and the last three cost almost nothing and should all be done. Row 8 (feature engineering) is the only row with a documented 10+ bp effect size and is where the 0.96243+ ambition must be pursued. Rows 5–7 are cheap-to-moderate hypotheses with near-zero public evidence of gain in _this_ competition; run them inside the nested harness and keep them only on outer-fold wins.

## 9. Execution plan

**Order of repository changes and experiments** (each step gated on the previous):

1. **Instrument the existing pipeline** (before touching any model): persist fold indices (`folds.parquet` keyed by row id); assert OOF coverage and id alignment (Section 10 harness); recompute pairwise OOF Spearman/Pearson on identical rows; log per-arm OOF AUC and leave-one-out blend deltas. This reproduces the 0.96128 baseline in a checkable form (Phase 1/2).
2. **Adversarial-validation re-audit** with the _current engineered features_, deep + shallow GBDT, id columns dropped; record AUC per feature set (Phase 2).
3. **Blend baseline ladder** (Phase 3): equal-weight prob / logit / rank blends of the existing arms on the same OOF rows; table of per-arm AUC, blend AUC, and pairwise correlations.
4. **Constrained optimization** (Phase 4): grid → hill-climb → SLSQP/NM-softmax with λ ∈ {0, 1e-4, 1e-3} and max_weight ∈ {0.4, 0.5}; bootstrap + fold weight stability; select on inner-OOF, score on one held-out outer fold before any LB submission.
5. **Member-strength work** (highest ROI, run in parallel): Flight-Distance label-free group statistics + original-dataset satisfaction rates; original-data row augmentation; aux expected-rating features — each inside fold-safe, label-free-of-test construction, measured as single-arm OOF deltas first (Phase 3.5 — this is the +20 bp row).
6. **GBDT stacking A/B** (Phase 5): meta-LGBM (Section 4 config) on {p}, {p, logit} blocks, nested outer validation, size-matched refits; adopt only if mean outer AUC beats the best blend by more than outer-fold spread.
7. **5 vs 10-fold matched A/B** (Phase 6): on the winner of steps 4–6; adopt 10-fold only on a matched-seed outer win.
8. **Postprocessing audit** (Phase 7): verify AUC invariance empirically (assert |ΔAUC| < 1e-12 under logit and rank maps of the final score); apply only the ε-clipping safety policy; no LB submissions for postprocessing.
9. **Final refit & submission** (Phase 8): regenerate all arms under the chosen production architecture (fold models, matched training fraction), refit the combiner on full-OOF with the frozen hyperparameters/λ from step 4/6, validate schema, save artifacts, submit once; treat the LB delta as a diagnostic, not a tuning signal.

## 10. Reproducibility checklist

**Artifacts to persist for every experiment:** fold indices (by row id + seed + K); per-arm OOF prediction arrays (`.npy` + sha256); per-arm fold-model test predictions; meta-feature matrices; fitted blend weights (per outer fold + final); meta-model checkpoints; configs (exact dicts, incl. seeds and package versions: `lightgbm.__version__`, `xgboost.__version__`, `catboost.__version__`, sklearn, scipy); a JSON experiment log entry (name → {OOF AUC, per-fold AUCs, weights, wall-clock, peak RSS, commit hash}).

**Leakage assertions (hard-fail, run in CI for every training job):**

```python
import numpy as np, pandas as pd

def assert_oof_sanity(oof: np.ndarray, y: np.ndarray, fold_ids: pd.Series,
                      train_ids: pd.Series, name: str):
    assert oof.shape[0] == len(y) == len(fold_ids), f"{name}: row count mismatch"
    assert np.isfinite(oof).all(), f"{name}: non-finite OOF values"
    assert ((oof > 0) & (oof < 1)).all() or ((oof >= 0) & (oof <= 1)).all(), \
        f"{name}: predictions outside [0,1]"
    covered = pd.Series(np.arange(len(y))).groupby(fold_ids.values).size()
    assert (covered > 0).all(), f"{name}: fold with zero OOF coverage"
    assert fold_ids.index.equals(train_ids.index), f"{name}: id misalignment"
    # in-sample substitution detector: OOF AUC dramatically above the same
    # model's holdout AUC (> 0.02) indicates leaked in-sample predictions
    return True

def assert_submission(path, test_ids, n_expected):
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["id", "satisfaction"], "schema"
    assert len(sub) == n_expected, "row count"
    assert sub["id"].equals(test_ids), "id order/content"
    assert sub["satisfaction"].between(0, 1).all(), "probabilities in [0,1]"
    assert sub["satisfaction"].nunique() > n_expected * 0.9, "tie explosion"
    return True
```

**Submission discipline:** one submission per validated procedure; log every submission (message → LB score → which experiment it instantiates); never iterate weights against LB deltas (rule 6).

**Package/version pinning:** record exact versions at run time; LightGBM `feature_fraction` vs `colsample_bytree` alias behavior differs across API paths ([LightGBM docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html), [issue #1011](https://github.com/Microsoft/LightGBM/issues/1011)) — assert the chosen subsampling is actually active (inspect `model.feature_importances_` variance or LightGBM logs) in the first run of each config.

---

## Open questions

1. Does the current pipeline's OOF-to-LB gap (−5.2 bp) decompose into LB sampling noise vs selection bias? Run the decomposition: LB noise is estimable from the public-private split fraction and AUC variance formulas; residual = selection. Not yet done for our ensemble.
2. Do the four GBDT arms differ enough _in features or hyperparameters_ to carry any incremental signal once Flight-Distance and route features are added? (Community evidence says nearly no; our own arms must be re-measured after step 5 of the plan.)
3. Does the meta-learner input distribution shift (inner-OOF vs production fold-model predictions) measurably exist here? KS-test per arm on one outer fold; if D < 0.05, size-matching is unnecessary.
4. Is 0.96243+ reachable GBDT-only at all on this dataset, or does it require the NN/TabPFN members that dominate the current public top? Public evidence currently bounds the GBDT-only frontier near ~0.9618–0.9620. Unknown; do not plan on it.
5. Competition rules: confirm external use of the original ~130k "Airline Passenger Satisfaction" dataset is permitted this season before step 5 (Playground rules have historically allowed it, but verify on the rules page for S6E10).

## Source notes

| Source                                                                                                                                                                                                   | Credibility | Last updated            |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ----------------------- |
| [Kaggle — Predicting Airline Satisfaction (official competition page)](https://www.kaggle.com/competitions/playground-series-s6e10)                                                                      | 5/5         | 2026 (live competition) |
| [Kaggle — S6E10 public code listing (notebook LB scores, stack-experiment titles)](https://www.kaggle.com/competitions/playground-series-s6e10/code)                                                     | 4/5         | 2026-10-11              |
| [ayman-43/-kaggle-s6e10-airline-satisfaction (LB 0.96140, CV 0.96175, LR stack on OOF logits, CatBoost-null ablation)](https://github.com/ayman-43/-kaggle-s6e10-airline-satisfaction)                   | 3/5         | 2026-10-10              |
| [bansal1600/kaggle-airline-prediction PR #2, #4–#7 (nested-CV LR stacker, hill-climb 0.961643 < LR stack, member ablations, LB 0.96161)](https://github.com/bansal1600/kaggle-airline-prediction/pull/6) | 3/5         | 2026-10                 |
| [ChrisLegge/kaggleAirlineSatisfaction (Flight-Distance stats 0.95874→0.96071, aux-feature and seed-averaging ablations)](https://github.com/ChrisLegge/kaggleAirlineSatisfaction)                        | 3/5         | 2026-10                 |
| [aditbytes/kaggle-s6e10-airline-satisfaction (top-100 LB at 0.96150 on 2026-10-05; CV–LB within 0.0006; <2% feature-mean drift)](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction)         | 3/5         | 2026-10-05              |
| [wozhenshiliudada/kaggle-airline-satisfaction-s6e10 (10-fold members, best 0.96132, explicit validation-boundary caveats)](https://github.com/wozhenshiliudada/kaggle-airline-satisfaction-s6e10)        | 3/5         | 2026-10                 |
| [mrjohnsonsea/playground-series-s6e10 (untuned XGB 5-fold CV 0.9585 ± 0.0004 baseline)](https://github.com/mrjohnsonsea/playground-series-s6e10)                                                         | 3/5         | 2026-10                 |
| [BSergey2000/kaggle-airline (OOF 0.9591 vs LB 0.9584 decay example)](https://github.com/BSergey2000/kaggle-airline)                                                                                      | 3/5         | 2026-10                 |
| [ALPHAMAN-0/Prediction_Airline_Satisfaction (Caruana hill-climb + rank-average safe-submission pattern)](https://github.com/ALPHAMAN-0/Prediction_Airline_Satisfaction)                                  | 3/5         | 2026-10                 |
| [LightGBM official parameter documentation (feature_fraction, aliases, num_leaves/max_depth)](https://lightgbm.readthedocs.io/en/latest/Parameters.html)                                                 | 5/5         | 2026                    |
| [XGBoost official parameter documentation (colsample_* cumulative semantics)](https://xgboost.readthedocs.io/en/stable/parameter.html)                                                                   | 5/5         | 2026                    |
| [CatBoost official parameter-tuning documentation (depth, l2_leaf_reg, overfitting detector)](https://catboost.ai/docs/en/concepts/parameter-tuning)                                                     | 5/5         | 2026                    |
| [microsoft/LightGBM issue #1011 (colsample_bytree silently ignored in some API paths)](https://github.com/Microsoft/LightGBM/issues/1011)                                                                | 4/5         | -                       |
| [Caruana et al. 2004 — Ensemble Selection from Libraries of Models](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)                                                            | 5/5         | 2004                    |
| [Kaggle general/18793 — Cross-validation strategy when blending/stacking](https://www.kaggle.com/general/18793)                                                                                          | 4/5         | -                       |
| [CrossValidated 239445 — How to properly do stacking with cross-validation](https://stats.stackexchange.com/questions/239445/how-to-properly-do-stacking-meta-ensembling-with-cross-validation)          | 4/5         | -                       |
| [FastML — Adversarial validation, part one](https://fastml.com/adversarial-validation-part-one/)                                                                                                         | 4/5         | -                       |
| [Matt-OP/hillclimbers (rank-based Caruana hill-climbing module)](https://github.com/Matt-OP/hillclimbers)                                                                                                | 3/5         | -                       |

**Conflicts and caveats.** (1) Self-reported repo scores are not independently verified; treat all community bp figures as directional. (2) The Kaggle notebook titles "S6E10: 10 OOF members lose 0.000245 AUC" and "S6E10: Can the Leaderboard See a Stack Gain?" were confirmed via the code listing but not read in full; they are consistent with the stacking-skepticism above but should be opened and read during Phase 1. (3) Public top notebooks include RealMLP/TabPFN members; their scores are not evidence that GBDT-only can reach 0.96243+. (4) Adversarial-validation literature uniformly notes AUC ≈ 0.5 is necessary-but-not-sufficient evidence of distributional identity.
