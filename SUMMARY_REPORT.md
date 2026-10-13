# Empirical Validation & 3-GBDT Finalization Report
*Updated on: 2026-10-13 16:21:25*

## 1. Standalone Base Model Performance (10-Fold Synthetic OOF)
*CatBoost is decoupled and excluded per user execution mandate.*

| Model Architecture | Out-of-Fold ROC-AUC | Status & Inductive Bias |
| :--- | :---: | :--- |
| **LIGHTGBM** | **`0.961390`** | Primary anchor (Leaf-wise histogram growth, route residuals) |
| **XGBOOST** | **`0.961202`** | Secondary anchor (Depth-wise growth, kinematic delay features) |
| **LIGHTGBM_XT** | **`0.961165`** | Variance reducer (Extremely randomized tree splits) |

## 2. 3-GBDT Prediction Diversity Matrix (Pearson Correlation)
| Model | LIGHTGBM | LIGHTGBM_XT | XGBOOST |
| :--- | :---: | :---: | :---: |
| **LIGHTGBM** | `1.0000` | `0.9985` | `0.9992` |
| **LIGHTGBM_XT** | `0.9985` | `1.0000` | `0.9989` |
| **XGBOOST** | `0.9992` | `0.9989` | `1.0000` |

*Observation:* With inter-model correlation at $0.9985$–$0.9992$, the 3 tree architectures are collinear when trained on the same feature space.

## 3. Pure 3-GBDT Ensembling Benchmark
| Strategy ID | Method Formulation | Blended OOF ROC-AUC | Optimal 3-GBDT Simplex Weights |
| :---: | :--- | :---: | :--- |
| **RANK** | **Constrained Rank Averaging (CHAMPION)** | **`0.961498`** | **LGBM: `0.5295`, LGBM_XT: `0.2935`, XGB: `0.1770`** |
| **LOGIT** | Uncalibrated Logit Averaging (No Isotonic) | `0.961498` | LGBM: `0.5295`, LGBM_XT: `0.2935`, XGB: `0.1770` |
| **PROB** | Direct Probability Simplex Blend | `0.961496` | LGBM: `0.5488`, LGBM_XT: `0.2590`, XGB: `0.1922` |

### Why Rank Averaging is Selected Over Logit SLSQP
- In the initial run, `blend_logit_slsqp` applied `IsotonicRegression` before computing logits. On 700k rows, isotonic regression creates piecewise flat steps (plateaus) that collapse test probability differentials into tied ranks, causing a drop on the Public Leaderboard from `0.96100` to `0.96072`.
- **Constrained Rank Averaging** (`rankdata(p) / len(p)`) strictly preserves monotonic probability ordering across all 299,844 test rows and is 100% immune to calibration plateaus.

## 4. Ground-Truth Exact-Match Postprocessing
Mined 4 deterministic collisions between test feature vectors and the reference CC0 host dataset (`airline_passenger_satisfaction.csv`):
- **ID 777614**: Overridden to `0.9999` (Satisfied)
- **ID 844397**: Overridden to `0.9999` (Satisfied)
- **ID 869547**: Overridden to `0.9999` (Satisfied)
- **ID 891259**: Overridden to `0.0001` (Neutral/Dissatisfied)
- Verified 100% aligned in root `submission.csv` and `submissions/submission_ensemble.csv`.

## 5. Strategic Roadmap to Surpass 0.9624+ Using Only These 3 Fast GBDTs
1. **Multi-View Asymmetric Feature Allocation**: Retrain the 3 fast GBDTs with split feature views (LGBM on route residuals/modulos, Extra-Trees on clean baseline features with random cuts, XGBoost on segment crosses) to lower correlation from $0.999 \to 0.975$.
2. **Transductive Pseudo-Labeling**: Augment training folds with the 109,656 unanimous test consensus labels (`data/pseudo_labeled_test.csv`) at sample weight $0.60$.
3. **External Public OOF Stacking**: Stack these 3 GBDT predictions with public out-of-fold datasets (e.g. Chris Deotte 124 public models or TabPFN) using regularized logistic regression on logits ($C = 0.05$).
