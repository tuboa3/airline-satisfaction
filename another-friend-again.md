## 1. Top-1 Feature Engineering Deconstruction

### The 39 Rating–Context Crosses (Highest Measured ROI)

The most concrete, validated feature engineering result comes from Shelton Wang (11th place, Discussion 745892). The author tested crossing each of the **13 service ratings** with three background context variables: **Class**, **Type of Travel**, and **Customer Type**, producing **39 new categorical features**.

**Implementation:**
- Each service rating (e.g., `Online boarding = 2`) is combined with a context (e.g., `Type of Travel = Business travel`) into a single string: `"2|Business travel"`.
- The model uses **CatBoost's native categorical processing** — no manual target encoding.
- The control model treats all 13 service ratings as categorical values (preserving 0 as its own level) and adds exact-value categorical copies of the 4 numerical features (Age, Flight Distance, Departure Delay, Arrival Delay).

**Measured results (5-fold CV, same split):**

| Setting | Control OOF AUC | +39 Crosses OOF AUC | Δ AUC | Positive Folds |
|---|---|---|---|---|
| 3,000 iterations | 0.960486 | 0.960823 | **+0.000337** | 5/5 |
| 6,000 iterations | 0.960515 | 0.960799 | **+0.000284** | 5/5 |

The improvement persisted when the training cap was raised from 3,000 to 6,000 iterations, and **all five folds improved** in both experiments. The selected best tree counts were below 4,100 even at the 6,000 cap, so the model was not hitting the ceiling.

**Critical negative result:** The author also tested **rating–rating crosses** (e.g., Wi-Fi × Online boarding) and found only **+0.000025** pooled improvement with **3/5 positive folds** — essentially noise. Rating–context crosses are the signal; rating–rating crosses are not.

### Target-Encoding Exact Flight Distance (Largest Single Step)

Sachith7 (151st place, Discussion 745908) provides a step-by-step ablation on a single 5-fold split, starting from plain XGBoost (CV 0.95883) and ending at **0.96134 public LB**.

| Step | Δ AUC |
|---|---|
| **Target-encoding the exact Flight Distance** (inside folds) | **+0.00135** |
| Teacher model trained on the original dataset | +0.00059 |
| Route profile features | +0.00014 |
| Auxiliary "expected rating" features | +0.00014 |
| Training on 10 folds instead of 5 | +0.00009 to +0.00018 per model |

Target-encoding exact Flight Distance is the **single largest gain** measured, far exceeding any other step. This suggests that Flight Distance's relationship with satisfaction is highly non-linear and route-specific, and that categorical treatment of exact distance values captures this better than raw numeric treatment.

### Other Domain-Aware Feature Suggestions (General Competitive Patterns)

The following features are commonly used in airline satisfaction competitions but were **not** directly validated in the three discussion posts. They are provided as supplementary guidance:

- **Delay interaction features:** `delay_ratio = Arrival Delay / (Departure Delay + 1)`, `total_delay = Arrival Delay + Departure Delay`, and a zero/non-zero delay flag. Discussion 745932 notes that 91% of flights have zero delay, and the Pearson correlation between departure and arrival delay is 0.84 but Spearman is only 0.47 — the ties at zero carry no rank information.
- **Composite service scores:** `digital_score = Online boarding + Inflight wifi service + Ease of Online booking`, `comfort_score = Seat comfort + Leg room service + Cleanliness`. These capture latent satisfaction dimensions.
- **Route profile features:** Sachith7 reported +0.00014 from route profile features (likely aggregations of route-level statistics).

### High-Cardinality Categorical Combinations & Target Encoding Schemes

The 39 rating–context crosses are the primary high-cardinality categorical features validated in these discussions. For manual target encoding (beyond CatBoost's native handling), the standard competitive practice in Playground competitions is:

- **K-fold out-of-fold target encoding** with 5-fold or 10-fold splits.
- **m-estimate smoothing:** `smoothing = 10` to `20` (sometimes tuned as a hyperparameter). The m-estimate formula is `(sum_target + prior * smoothing) / (count + smoothing)`.
- **Noise addition:** Small Gaussian noise (std = 0.01–0.02) added to encoded values during training as regularization.

These parameters are general competitive patterns, not directly cited from the S6E10 discussions.

---

## 2. Synthetic vs. Original Dataset Handling

### What the Discussion Posts Cover

The three discussion posts do **not** directly address sample weighting schemes for the synthetic dataset. However, Sachith7’s **teacher model** step (+0.00059) is effectively a form of **two-stage pretrain-finetune**: a teacher model is trained on the original 129,880-row dataset, and its knowledge is distilled to guide the student model on the synthetic competition data. This is the closest validated technique for handling the original/synthetic split.

### General Competitive Patterns (Supplementary)

In Playground competitions that provide both original and synthetic data, three strategies are commonly used:

**Strategy A: Fixed sample weights.** Assign weight 1.0 to original data and 0.5 to synthetic data, or vice versa. This is the most common baseline and is simple to implement.

**Strategy B: Adversarial validation density weighting.** Train a binary classifier to distinguish original from synthetic samples. Use the predicted probability (or its inverse) as sample weights. This is more sophisticated and typically gains 5–10 bps over fixed weights. The logic: if a synthetic sample is "very synthetic" (easy to classify), downweight it; if it is "almost original-like," upweight it.

**Strategy C: Two-stage pretrain-finetune.** Pretrain on the full synthetic dataset, then finetune on the original 129,880 rows only. Sachith7’s teacher model is a variant of this approach.

**Which is most promising for S6E10?** Given that Sachith7 validated the teacher-model approach with a measured +0.00059 gain, **Strategy C (or a variant)** has direct evidence of efficacy in this specific competition. Adversarial validation density weighting remains a potential lever but is **not validated** by these three discussions.

---

## 3. Leakage & Near-Duplicate Matching

### Findings from Discussion 745932

Starkhushi (126th place) provides a statistical audit of train/test/original overlap:

- **Zero duplicate feature-tuples in train, zero in test, and zero test rows whose full tuple appears in train.** There is **no exact-match leakage** to exploit within the competition data.
- The original 129,880-row dataset overlaps train by **17 rows** and test by **4 rows**. Your 4 exact matches are confirmed — but this is the **full extent** of exact overlap.

### Near-Duplicate Fuzzy Collisions

The discussion does **not** report any near-duplicate fuzzy collisions (e.g., 20/21 matching features). Given the audit's finding of zero duplicate feature-tuples in train and test, and only 4 test rows overlapping with the original, there is **no evidence** that near-duplicate matching is a viable exploitation strategy for this competition.

**Conclusion:** Leakage exploitation is **not** a productive avenue for S6E10. Your 4 exact matches are the complete set of deterministic flips available.

---

## 4. Competitive Hyperparameter Regimes

### CatBoost Parameters (Validated in Discussion 745892)

Shelton Wang’s experiment used the following CatBoost parameters, which produced the +0.000337 AUC gain from the 39 rating–context crosses:

| Parameter | Value |
|---|---|
| Depth | 6 |
| Learning rate | 0.05 |
| L2 leaf regularization | 3 |
| Early-stopping patience | 100 |
| Max iterations | 3,000–6,000 (best tree counts ~4,100) |

### XGBoost Baseline (Discussion 745908)

Sachith7’s ablation started from plain XGBoost (CV 0.95883) and ended at 0.96134 public LB after stacking. The specific XGBoost hyperparameters are not listed in the post, but the step-by-step gains provide a roadmap for improvement.

### General GBDT Hyperparameter Ranges (Supplementary)

For context, the following ranges are commonly associated with top-tier performance in Playground tabular competitions. These are **not** directly validated in the S6E10 discussions but represent general competitive patterns:

**CatBoost:**
- `depth`: 6–8
- `learning_rate`: 0.03–0.05
- `l2_leaf_reg`: 3–10
- `max_ctr_complexity`: 4–6 (critical for categorical interactions)
- `iterations`: 3000–5000 with early stopping

**XGBoost:**
- `max_depth`: 5–7
- `learning_rate`: 0.03–0.05
- `subsample`: 0.7–0.9
- `colsample_bytree`: 0.6–0.8
- `min_child_weight`: 3–10
- `reg_alpha`: 0.1–1.0
- `reg_lambda`: 1.0–5.0

**LightGBM:**
- `num_leaves`: 31–63
- `learning_rate`: 0.03–0.05
- `feature_fraction`: 0.6–0.8
- `bagging_fraction`: 0.7–0.9
- `min_child_samples`: 10–30

### Stacking: Simpler Wins

Sachith7 explicitly states that a **plain logistic regression stacker** beat every fancier stacker tried. This aligns with general practice: when base models are already strong and correlated, a linear meta-learner is sufficient and less prone to overfitting. Your Nelder-Mead logit blend is essentially a linear combination, but a properly regularized logistic regression on OOF predictions may be better calibrated.

---

## 5. Synthesis & Actionable Recommendations

Your current ensemble (CatBoost 0.96029, XGBoost 0.96003, LightGBM 0.95993, RealMLP-TabM 0.95957, DCN-v2 0.95913) blends to 0.96058 OOF / 0.96008 LB. The gap to Top-1 (0.96177) is **169 bps**.

**Highest-ROI actions based on the validated discussion findings:**

1. **Add the 39 rating–context crosses to your CatBoost model** (+~30 bps OOF in the author's controlled experiment). Use the exact pseudocode from Discussion 745892.
2. **Target-encode exact Flight Distance inside folds** (+~135 bps in Sachith7's ablation). This is the largest single step they measured.
3. **Train a teacher model on the original dataset and distill** (+~59 bps). This is a validated two-stage pretrain-finetune protocol.
4. **Switch your stacker to plain logistic regression.** Sachith7 explicitly states it beat every fancier stacker tried.
5. **Move to 10-fold training** for a small but consistent boost across every base model.

**What not to waste time on:**
- Rating–rating crosses (noise-level gain, 3/5 folds).
- Elaborate stacking architectures (LR wins).
- Exact-match leakage exploitation (only 4 test rows overlap with the original; no train–test duplicates).
- Near-duplicate fuzzy matching (no evidence of viable collisions).

**Unresolved lever:** Adversarial validation density weighting for the synthetic dataset is **not** addressed in these three discussions. It remains a potential source of 5–10 bps, but without direct validation in S6E10, it carries more implementation risk than the teacher-model approach that Sachith7 has already measured.

**Combined potential:** Steps 1–3 alone account for approximately **+224 bps** in measured gains (0.000337 + 0.00135 + 0.00059 = 0.002277), which exceeds the 169 bps gap to Top-1. The overlap between these steps and your existing pipeline may reduce the net gain, but the direction is clear: **the 39 crosses, Flight Distance target encoding, and the teacher model are the three validated levers with the highest expected ROI.**
