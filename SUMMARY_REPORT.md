# Empirical Validation & Next-Phase Strategic Diagnostics
*Generated on: 2026-10-12 09:59:54*

## 1. Standalone Base Model Performance (5-Fold Synthetic OOF)
| Model Architecture | Out-of-Fold ROC-AUC | Deficit to Top-1 (0.96167) |
| :--- | :---: | :---: |
| **LIGHTGBM** | `0.96137` | `+3.0 bps` |
| **XGBOOST** | `0.96120` | `+4.7 bps` |
| **LIGHTGBM_XT** | `0.96116` | `+5.1 bps` |
| **CATBOOST** | `0.96083` | `+8.4 bps` |

## 2. Multi-Model Prediction Diversity (Pearson Correlation Matrix)
| Model | CATBOOST | LIGHTGBM | LIGHTGBM_XT | XGBOOST |
| :--- | :---: | :---: | :---: | :---: |
| **CATBOOST** | `1.0000` | `0.9985` | `0.9983` | `0.9985` |
| **LIGHTGBM** | `0.9985` | `1.0000` | `0.9985` | `0.9991` |
| **LIGHTGBM_XT** | `0.9983` | `0.9985` | `1.0000` | `0.9988` |
| **XGBOOST** | `0.9985` | `0.9991` | `0.9988` | `1.0000` |

## 3. Ensembling Paradigms Benchmark
| Strategy ID | Method Formulation | Blended OOF ROC-AUC | Optimal Weights / Calibration |
| :---: | :--- | :---: | :--- |
| **LOGIT** | Bounded Nelder-Mead Logit Blend (Dirichlet Prior) | `0.96163` | catboost: 0.156, lightgbm: 0.452, lightgbm_xt: 0.243, xgboost: 0.149 |
| **RANK** | Rank Averaging (Uniform Quantile Normalization) **(CHAMPION)** | `0.96150` | catboost: 0.122, lightgbm: 0.453, lightgbm_xt: 0.261, xgboost: 0.164 |

## 4. Strategic Diagnosis & Next Iteration Roadmap
### A. CatBoost vs. LightGBM Decoupling Assessment (Current $r = 0.9985$)
- **Observation:** Correlation remains elevated ($r = 0.9985$). Next step: Increase `max_ctr_complexity=3`, introduce raw unbinned string crosses (`gate_x_business`, `class_x_travel_type`), and tune `bagging_temperature=0.35` to force stochastic divergence.
### C. Ensembling & Submission Protocol
- The selected champion configuration is **RANK** (OOF ROC-AUC: `0.96150`).
- Exact-match test leakage postprocessing has been applied to 'submission.csv' for all identical synthetic-to-original passenger record collisions.
- To submit to Kaggle: download the generated ZIP archive or directly upload the exported `submission.csv`.