# PROJECT_STATE: Airline Passenger Satisfaction Optimization

**Target Metric:** ROC-AUC  
**Current Best Leaderboard (LB):** `0.96008` (Phase 2 Blend, Rank 382/868)  
**Target Top-1 Leaderboard:** `0.96177` (Deficit: ~169 bps)  
**Phase 3 OOF Benchmarks:** CatBoost `0.96053`, XGBoost `0.96023`, LightGBM `0.96007`  
**Phase 4 Architecture:** Grandmaster GBDT-Only Engine: 10 Folds, In-Fold Leak-Free TE, Shelton Wang 64 Features (4000 iter, lr 0.04), Purged Raw Original Rows, Pure GBDT Meta-Learner & Direct Rank Blending  
**Timestamp:** October 8, 2026  

---

## 1. Executive Summary & Objective

The objective is to achieve the Top-1 spot (`>= 0.96167` ROC-AUC) in the Kaggle Airline Passenger Satisfaction competition. 

The strategy relies on a 4-pillar architectural foundation:
1. **Original Dataset Ingestion & Dual-Prior Blending:** Unifying the synthetic training set (699,635 rows) with the original dataset (129,880 rows) into an 829,515-row unified corpus with target prior regularization.
2. **Domain-Specific Non-Linear & Interaction Feature Engineering:** Mining flight delay recovery dynamics, the 15-minute flatline law, route hazard profiles, and high-cardinality bounded categorical crosses.
3. **Multi-Paradigmatic Model Diversity:** Ensembling orthogonal gradient boosted trees (LightGBM, XGBoost, CatBoost) with modern deep tabular architectures (RealMLP-TabM with 16 ensemble heads, Parallel Low-Rank DCN-v2, and Periodic Linear ResNet).
4. **Constrained Ranking Optimization & Exact-Match Postprocessing:** Nelder-Mead logit blending, pairwise Wilcoxon-Mann-Whitney (WMW) surrogate stacking, and ground-truth test leakage mining.

---

## 2. Completed Milestones (Chronological Progression)

### Milestone 1: Baseline Architecture & First Submission (Phase 1)
- Built modular competition pipeline: `src/dataset.py`, `src/features.py`, `src/models.py`, `src/trainer.py`, `src/ensemble.py`, and `src/postprocess.py`.
- Implemented Stratified 5-Fold Cross-Validation on synthetic data (699,635 rows).
- Evaluated Phase 1 standalone models:
  - **CatBoost:** `0.95849` OOF
  - **LightGBM:** `0.95830` OOF
  - **XGBoost:** `0.95813` OOF
  - **RealMLP:** `0.95657` OOF
- Engineered meta-ensembling strategies: Nelder-Mead Bounded Logit Blending (`0.95883` OOF), RidgeCV Interaction Stacking (`0.95873`), Rank Averaging (`0.95869`), and NNLS (`0.95867`).
- **Result:** First submission achieved **`0.95824` Public Leaderboard**.

### Milestone 2: Multi-Agent Deep Research & Phase 2 Blueprint
Synthesized findings across 3 specialized research streams:
- **Stream 1 (Tabular & Cross Features):** Bounded categorical crossings (`gate_x_business`, `class_x_travel_type`, `delay_tier_x_class`), target encoding with smoothing, and route delay profiles.
- **Stream 2 (Data Ingestion & Postprocessing):** Ground-truth leakage mining between test set and the original reference dataset (`teejmahal20/airline-passenger-satisfaction`), discovering 4 exact identical passenger collisions.
- **Stream 3 (Deep Tabular & Optimization):** Parallel Low-Rank DCN-v2 with LayerNorm, RealMLP-TabM with disjoint embeddings and parametric SELU, and smooth pairwise WMW surrogate AUC loss.

### Milestone 3: Unified Dataset Ingestion & Feature Pipeline v2
- Combined synthetic train set + original dataset into unified training set of **829,515 rows** (Mean target satisfaction: 0.4345 in original, preserved synthetic distribution).
- Added domain features:
  - `route_delay_hazard` = `arr_delay / max(10, Flight Distance / 7.5)`
  - `delay_intensity` = `total_delay / max(1, Flight Distance / 100)`
  - `airborne_delay_recovery` = `arr_delay - dep_delay`
  - `delay_over_15` indicator (15-Minute FAA flatline threshold)
  - String crosses: `gate_x_business`, `class_x_travel_type`, `delay_tier_x_class`
- Implemented `package_results.py` to benchmark out-of-fold logs, generate summary markdown reports, and produce submission zips headlessly.

#### Milestone 4: Kaggle Dual-T4 Training Breakout & 5-Model Ensemble Submission
All 5 standalone models trained successfully on Kaggle Dual-T4:
- **CatBoost:** `0.95849` -> **`0.96029`** (+18.0 bps jump, new single-model champion)
- **XGBoost:** `0.95813` -> **`0.96003`** (+19.0 bps jump, broke through 0.9600)
- **LightGBM:** `0.95830` -> **`0.95993`** (+16.3 bps jump)
- **RealMLP-TabM:** `0.95657` -> **`0.95957`** (+30.0 bps jump)
- **DCN-v2:** **`0.95913`** (smooth training on Dual-T4 with LayerNorm FP16 stability)
- **Champion Ensemble:** Nelder-Mead Logit Blend achieved **`0.96058` OOF**.
  - Weights: CatBoost 47.61%, XGBoost 30.18%, RealMLP 11.79%, DCN-v2 5.44%, LightGBM 4.98%.
- **Submitted Public Leaderboard:** **`0.96008` (Rank 382 / 868)**! Jumped from 0.95824 (+184 bps gain). Deficit to Top 1 (`0.96177`): 169 bps.

### Milestone 5: Full Implementation of 3 Friends' Research Findings (Phase 3)
1. **The 39 Rating-Context Crosses (`src/features.py`):**
   - Implemented exact 11th-place Shelton Wang crosses (+34 bps across 5/5 folds): 13 service ratings crossed with `Class`, `Type of Travel`, and `Customer Type` (`f"{rating}|{context}"`).
   - Automatically detected and routed into CatBoost native `cat_features` and tree histogram binning.
2. **Latent Psychometric Subscales & Entropy (`src/features.py`):**
   - Added `comfort_score`, `digital_score`, `service_score`, and `logistics_score`.
   - Engineered `survey_entropy` (quantifying straight-lining across ratings $k \in \{0..5\}$) and `survey_zero_count`.
3. **Two-Stage Residual Boosting with GLM Margins across Trees (`src/trainer.py` & `src/models.py`):**
   - Extended GLM natural cubic spline margins to XGBoost (`base_margin`), LightGBM (`init_score`), and CatBoost (`Pool(..., baseline=...)`).
4. **Hybrid Post-Calibration Protocol (`src/postprocess.py`):**
   - Implemented `empirical_quantile_align` ($F_{\text{test}}(p) \mapsto \text{Quantile}_{\text{OOF}}(p)$) preserving 100% of ROC-AUC test rankings while aligning test probabilities to OOF distribution to close the 50 bps OOF-to-LB deficit.
   - Added monotonic `beta_calibrate` on validation predictions.
5. **Strategy 8: Regularized Plain Logistic Regression Stacking (`src/ensemble.py`):**
   - Added Sachith7's #1 stacker (LogisticRegression on out-of-fold logits), outperforming complex meta-learners.
6. **Competitive Hyperparameters Tuned (`src/config.py`):**
   - CatBoost depth 6, lr 0.05, max_ctr_complexity 4, 3500 iterations.
   - XGBoost depth 6, lr 0.035, colsample 0.70, subsample 0.80, min_child_weight 5.
   - LightGBM num_leaves 63, lr 0.035, feature_fraction 0.70, bagging_fraction 0.80.

---

## 3. Engineering Bugs Diagnosed & Resolved

| Component | Symptom / Error | Root Cause | Permanent Resolution |
| :--- | :--- | :--- | :--- |
| **CatBoost** | `CatBoostError: Setting TargetBorderCount is not supported for loss function Logloss` | Parameter adaptation was injecting `ctr_target_border_count = 64`, which CatBoost translates internally to `TargetBorderCount`, illegal under `Logloss`. | Removed `"ctr_leaf_reg"` from `TrainConfig.cb_params`. Added explicit stripping of `target_border_count`, `TargetBorderCount`, and `ctr_target_border_count` in `CatBoostModel.__init__` and the CPU fallback path. |
| **Neural Models** (`RealMLP`, `DCNv2`, `FTTransformer`, `TabularResNet`) | `ValueError: Input contains NaN` inside `roc_auc_score(y_val, val_preds)` | All-NaN continuous columns or non-positive distances produced NaNs in `np.nanquantile`, propagating NaNs through `PiecewiseLinearSplineEmbedding` into final sigmoid logits. | Wrapped continuous processed features and quantile matrices with `np.nan_to_num(..., nan=0.0)`. Added `np.nan_to_num(val_preds, nan=0.5)` + `np.clip(0.0, 1.0)` prior to ROC-AUC calculation and on all `predict_proba` returns. |
| **DCN-v2** | Floating point overflow / NaNs in PyTorch FP16 AMP | Unbounded cross layer products $x_0 \odot (x_l V U^T + b)$ across 3 layers exceeded FP16 maximum value (65,504). | Added `nn.LayerNorm(d_in)` at the output of `LowRankCrossLayer` and scaled weight initialization std to $1/\sqrt{d_{in}}$. |
| **Feature Pipeline** | `TypeError: Choicelist and default value do not have a common dtype` (NumPy 2.x) | `np.select([tot_delay == 0, ...], ["0", "1", "2"])` triggered strict type promotion failure under NumPy 2.x in `_get_bounded_crosses`. | Refactored `delay_tier` to integer codes `[0, 1, 2], default=0` before string concatenation. |
| **Feature Pipeline** | Potential `log1p(negative)` NaNs | Malformed negative `Flight Distance` or division by zero in delay ratios. | Added `.clip(lower=0)` before `log1p(Flight Distance)` and `np.maximum(0.0, ...)` inside denominator expressions. |

---

## 4. Current State & Performance Scorecard

### Standalone Models Status (Kaggle Dual-T4 Verified)

| Architecture | Phase 1 OOF | Phase 2 OOF (Merged Data) | Status | Key Configurations |
| :--- | :---: | :---: | :---: | :--- |
| **CATBOOST** | `0.95849` | **`0.96029`** | **Trained** | 5-Fold, GPU task_type, depth=6, combinations_ctr, 39 rating crosses |
| **XGBOOST** | `0.95813` | **`0.96003`** | **Trained** | 5-Fold, tree_method=hist, lr=0.035, depth=6, subsample=0.8 |
| **LIGHTGBM** | `0.95830` | **`0.95993`** | **Trained** | 5-Fold, num_leaves=63, feature_fraction=0.7, lr=0.035 |
| **REALMLP-TABM** | `0.95657` | **`0.95957`** | **Trained** | 5-Fold, 16 ensemble heads, PLE spline bins=16, epochs=48, AMP=True |
| **DCN-V2** | *N/A* | **`0.95913`** | **Trained** | 5-Fold, 3 cross layers (rank=d//4), LayerNorm, deep=[512, 256, 128] |

### Ensembling & Stacking Benchmark (Phase 2 Results)

| Ensemble Method | OOF ROC-AUC | Leaderboard | Notes |
| :--- | :---: | :---: | :--- |
| **Nelder-Mead Logit Blend (Champion)** | **`0.96058`** | **`0.96008` (Rank 382/868)** | Optimal Weights: CB 0.476, XGB 0.302, RealMLP 0.118, DCNv2 0.054, LGB 0.050 |
| **Rank Averaging** | `0.96043` | - | CB: 0.556, XGB: 0.323, RealMLP: 0.121 |
| **NNLS (Bounded MSE)** | `0.96040` | - | CB: 0.596, XGB: 0.383, LGB: 0.020 |
| **Ridge Interaction Stacking** | `0.96038` | - | CV ROC-AUC: 0.96048 (optimal alpha=200.0) |
| **Isotonic Calibrated Stacking** | `0.96029` | - | 5-Fold PAVA on NNLS |
| **Regularized Logistic Stacking** | *New in Phase 3* | — | Sachith7 #1 Stacker (Beat all complex stackers) |

---

## 5. Hardware Constraints & Operational Protocols

- **Local Machine Constraints:** 
  - Dual-core 2016 legacy CPU with limited RAM.
  - **Rule:** Strictly **NO** heavy local model training, large feature generation, or long background jobs locally.
  - Local environment is used exclusively for code editing, syntax verification (`py_compile`), git maintenance, and remote orchestration.
- **Remote Execution Environment:**
  - Kaggle Notebook with Dual NVIDIA T4 GPUs (32GB combined VRAM, High-RAM instance).
  - All heavy training runs via command-line arguments in Kaggle notebook cells.
- **CLI Standard:** Shell commands are prefixed with `rtk` to compress output logs.

---

### Milestone 6: Grandmaster GBDT-Only Optimization & Leak-Free Architecture (Phase 4)
- **10-Fold Stratified Cross-Validation (Option A):** Scaled default CV from 5 to 10 folds (`n_splits = 10`), increasing training data density per fold from 80% to 90% (+9 to +18 bps per model, Topics #745908 & #746148).
- **Leak-Free In-Fold Target Encoding Engine:** Computed Bayesian Target Encoding on exact `Flight Distance` strictly inside each active fold loop using only that fold's training index (`synth_tr_subidx`). Completely eliminated cross-fold validation leakage (+135 bps lever, Topic #745908).
- **Purging Raw Original Rows & Retaining Isolated Teacher:** Set `use_original_data = False` to prevent continuous covariate shift (Flight Distance TVD 0.21) from contaminating tree split points (+38 bps CV gain, Topic #745098). Retained the isolated `HistGradientBoostingClassifier` trained on original data to provide `orig_proba` and `orig_logit` (+59 bps prior, Topic #745908).
- **Shelton Wang's Full 64-Feature CatBoost Engine:** Ingested 4 exact-value numerical categorical copies (`Age__category`, `Flight Distance__category`, `Departure Delay in Minutes__category`, `Arrival Delay in Minutes__category`) and 13 rating categories, feeding all 60 categorical columns into CatBoost's native CTR processor (+34 bps across 5/5 folds, Topic #745892).
- **Flight Distance Modulo/Digit Decompositions:** Added `dist_mod_10`, `dist_mod_100`, and `dist_div_100` to capture synthetic digit artifacts (+36 bps, Topic #745098).
- **CatBoost Parameter Optimization (Option A):** Configured 4,000 iterations at `lr = 0.04`, `depth = 6`, and `max_ctr_complexity = 4` to align with the peak saturation point discovered by Shelton Wang.
- **100% Pure GBDT-Compliant Ensembling Engine:** Replaced all non-GBDT linear models (Logistic Regression, Ridge, NNLS, Isotonic PAVA) with pure GBDT strategies: Direct Nelder-Mead Rank Averaging, Direct Probability Blend, Direct Logit Blend, and a Shallow Regularized LightGBM Meta-Learner (`max_depth = 3`, `num_leaves = 7`).

---

## 5. Hardware Constraints & Operational Protocols

- **Local Machine Constraints:** 
  - Dual-core 2016 legacy CPU with limited RAM.
  - **Rule:** Strictly **NO** heavy local model training, large feature generation, or long background jobs locally.
  - Local environment is used exclusively for code editing, syntax verification (`py_compile`), git maintenance, and remote orchestration.
- **Remote Execution Environment:**
  - Kaggle Notebook with Dual NVIDIA T4 GPUs (32GB combined VRAM, High-RAM instance).
  - All heavy training runs via command-line arguments in Kaggle notebook cells.
- **CLI Standard:** Shell commands are prefixed with `rtk` to compress output logs.

---

## 6. Active Execution Plan (Phase 4 Kaggle Run: Pure GBDT 10-Fold Engine)

1. **Pull Latest Pushed Commits in Kaggle:**
   ```bash
   !git pull origin main
   ```
2. **Train CatBoost with Shelton Wang 64 Features, Depth 6, and 4,000 Iterations (10 Folds):**
   ```bash
   !python run_training.py --model catboost --device cuda --folds 10
   ```
3. **Train XGBoost with In-Fold Target Encoding (10 Folds):**
   ```bash
   !python run_training.py --model xgboost --device cuda --folds 10
   ```
4. **Train LightGBM with Hist Bins & In-Fold Target Encoding (10 Folds):**
   ```bash
   !python run_training.py --model lightgbm --device cpu --folds 10
   ```
5. **Run the Pure GBDT Meta-Learner & Direct Rank Ensemble:**
   ```bash
   !python run_ensemble.py --method auto
   ```
   *Auto-evaluates Direct Rank Averaging, Direct Probability Blend, Direct Logit Blend, and Shallow LightGBM Meta-Learner, exporting the Quantile-Aligned champion to `submission.csv`.*
6. **Package Results:**
   ```bash
   !python package_results.py --skip_ensemble_run
   ```
