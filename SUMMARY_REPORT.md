# Empirical Validation & Next-Phase Strategic Diagnostics
*Updated on: 2026-10-12 18:45:00*

## 1. Standalone Base Model Performance (10-Fold Synthetic OOF)
| Model Architecture | Out-of-Fold ROC-AUC | Deficit to Top-1 (0.96243) |
| :--- | :---: | :---: |
| **LIGHTGBM** | `0.96137` | `+10.6 bps` |
| **XGBOOST** | `0.96120` | `+12.3 bps` |
| **LIGHTGBM_XT** | `0.96116` | `+12.7 bps` |
| **CATBOOST** | `0.96083` | `+16.0 bps` |

## 2. 3-GBDT Prediction Correlation Matrix (Pearson)
| Model | LIGHTGBM | LIGHTGBM_XT | XGBOOST |
| :--- | :---: | :---: | :---: |
| **LIGHTGBM** | `1.0000` | `0.9985` | `0.9991` |
| **LIGHTGBM_XT** | `0.9985` | `1.0000` | `0.9988` |
| **XGBOOST** | `0.9991` | `0.9988` | `1.0000` |

## 3. 3-GBDT Ensembling Benchmark & Leaderboard Diagnostics
| Strategy ID | Method Formulation | OOF ROC-AUC | Public Leaderboard | Optimal Weights / Diagnostic Notes |
| :---: | :--- | :---: | :---: | :--- |
| **RANK** | Constrained Rank Averaging **(CHAMPION)** | **`0.96149`** | *Recommended* | LGB: 0.5096, LGB_XT: 0.3035, XGB: 0.1869 (Immune to isotonic plateaus) |
| **LOGIT** | Bounded Nelder-Mead Logit Blend | `0.96162` | **`0.96072`** | LGB: 0.5015, LGB_XT: 0.3068, XGB: 0.1917 (Isotonic calibration overfit OOF tails) |
| **PROB** | Direct Probability Simplex Blend | `0.96149` | — | LGB: 0.5488, LGB_XT: 0.2590, XGB: 0.1922 |
| **GBDT_META** | Shallow LightGBM Meta-Learner | `0.96145` | — | Regularized shallow trees |
| *Single Model* | LightGBM Extra-Trees (Phase 4) | `0.96089` | **`0.96100`** | Baseline benchmark |

## 4. Key Diagnostic Finding: Why Logit Blend Dropped on Public LB
- **Root Cause:** The `blend_logit_slsqp` method applies `IsotonicRegression` on OOF probabilities before computing logits. On 700k rows, isotonic regression fits piecewise constant step functions (plateaus). When transferred to test predictions, these flat steps collapse local probability differentials into identical tied ranks, reducing rank discrimination and dropping LB from `0.96100` to `0.96072`.
- **Prescription:** Revert champion submission to **Constrained Rank Averaging** (`--method rank`), which strictly preserves monotonic ordering without piecewise flatlines.

## 5. Active Phase 6 Roadmap (Target >= 0.96243)
1. **Transductive Pseudo-Labeling:** 109,656 test rows (36.57%) with $\ge 96\%$ unanimous agreement extracted into `data/pseudo_labeled_test.csv`.
2. **Kaggle Retraining:** Running 10-fold CV on Kaggle Dual-T4 with `--use_pseudo_labels` across LightGBM, XGBoost, and Extra-Trees.
3. **Multi-View Integration:** Incorporate CatBoost (120 CTR features) and Koumei Maki route models once complete to lower inter-model correlation to $\approx 0.97$.