# PROJECT_STATE: Airline Passenger Satisfaction Optimization

**Target Metric:** ROC-AUC  
**Current Best Leaderboard (LB):** `0.95824` (Phase 1 Nelder-Mead Logit Blend)  
**Target Top-1 Leaderboard:** `0.96167` (Deficit: ~34 bps)  
**Phase 2 Individual OOF Scores:** LightGBM `0.95993`, XGBoost `0.96003`, RealMLP `0.95957`  
**Current Codebase Commit:** [`186ac07`](https://github.com/tuboa3/airline-satisfaction/commit/186ac07) (`main` synced with `origin/main`)  
**Timestamp:** October 6, 2026  

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

### Milestone 4: Kaggle Dual-T4 Training Breakout (Phase 2 OOF Jumps)
On Kaggle Dual NVIDIA T4 GPU setup, Phase 2 models demonstrated major OOF performance gains:
- **LightGBM:** `0.95830` -> **`0.95993`** (+16.3 bps jump)
- **XGBoost:** `0.95813` -> **`0.96003`** (+19.0 bps jump, broke through the 0.9600 barrier)
- **RealMLP-TabM:** `0.95657` -> **`0.95957`** (+30.0 bps jump)

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

### Standalone Models Status

| Architecture | Phase 1 OOF | Phase 2 OOF | Status | Key Configurations |
| :--- | :---: | :---: | :---: | :--- |
| **XGBOOST** | `0.95813` | **`0.96003`** | **Trained (Phase 2)** | 5-Fold, tree_method=hist, lr=0.03, depth=6, subsample=0.8 |
| **LIGHTGBM** | `0.95830` | **`0.95993`** | **Trained (Phase 2)** | 5-Fold, max_depth=8, num_leaves=127, feature_fraction=0.7 |
| **REALMLP-TABM** | `0.95657` | **`0.95957`** | **Trained (Phase 2)** | 5-Fold, 16 ensemble heads, PLE spline bins=16, epochs=48, AMP=True |
| **CATBOOST** | `0.95849` | *Pending Run* | **Ready to Execute** | 5-Fold, GPU task_type, depth=7, combinations_ctr, original_weight=0.50 |
| **DCN-V2** | *N/A* | *Pending Run* | **Ready to Execute** | 5-Fold, 3 cross layers (rank=d//4), LayerNorm, deep=[512, 256, 128] |

### Ensembling & Stacking Benchmark (Phase 1 Baseline)

| Ensemble Method | OOF ROC-AUC | Leaderboard | Notes |
| :--- | :---: | :---: | :--- |
| **Nelder-Mead Logit Blend** | **`0.95883`** | **`0.95824`** | Champion Phase 1 configuration (Weights: CB 0.451, LGB 0.300, XGB 0.159, RealMLP 0.090) |
| **Ridge Interaction Meta-Learner** | `0.95873` | - | Cross-validated pairwise logit interactions |
| **Rank Averaging** | `0.95869` | - | Uniform quantile normalization |
| **NNLS** | `0.95867` | - | Convex bounded non-negative least squares |
| **Isotonic Calibrated Stacking** | `0.95852` | - | 5-Fold PAVA on NNLS |

*Note: Phase 2 ensemble is projected to surpass `0.9605 - 0.9610+` once CatBoost and DCN-v2 OOF predictions are blended with XGBoost (`0.96003`), LightGBM (`0.95993`), and RealMLP (`0.95957`).*

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

## 6. Active Execution Plan (Immediate Next Steps)

1. **Pull Latest Pushed Commits in Kaggle:**
   ```bash
   !git pull origin main
   ```
2. **Train CatBoost on Dual-T4 GPU:**
   ```bash
   !python run_training.py --model catboost --device cuda --folds 5 --original_weight 0.50
   ```
3. **Train Parallel Low-Rank DCN-v2 on Dual-T4 GPU:**
   ```bash
   !python run_training.py --model dcn_v2 --device cuda --folds 5 --batch_size 2048 --epochs 48
   ```
4. **Generate the Multi-Model Ensemble:**
   ```bash
   !python run_ensemble.py --method auto
   ```
   *Will automatically detect all 5 available OOF prediction sets (LightGBM, XGBoost, CatBoost, RealMLP, DCNv2) and optimize weights via Nelder-Mead Logit Blending and Ridge Interaction Stacking with exact-match postprocessing.*
5. **Package and Submit:**
   ```bash
   !python package_results.py --skip_ensemble_run
   ```
   *Packages all diagnostic logs, OOF matrices, figures, and exports `submission.csv` for Kaggle leaderboard submission.*
