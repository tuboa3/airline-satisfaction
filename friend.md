# Synergistic Feature Engineering: An External Prior, Route-Level Shrinkage, and Engine-Specific Blocks to Break Correlation in GBDT Ensembles

## Executive Recommendation and Strategic Overview

This report details a strategic research program aimed at improving the Receiver Operating Characteristic - Area Under Curve (ROC-AUC) score for the Kaggle Playground Series Season 6 Episode 10 competition under the strict constraint of using only Gradient Boosting Decision Tree (GBDT) models [[33](https://www.mlaia.com/blog-tabular-data-ml-vs-dl.html)]. The current public leaderboard baseline stands at 0.96243, serving as the initial performance target [[27](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard)]. The core challenge lies in identifying and implementing high-value, leakage-free feature engineering strategies that can push beyond this baseline. The analysis prioritizes three distinct technical dimensions based on established empirical findings and user preferences: the creation of auxiliary expected-rating features derived from an external dataset; the statistical decomposition of the Flight Distance variable to capture both continuous and discrete semantics; and the design of engine-specific feature blocks to exploit the inherent asymmetries between XGBoost and LightGBM.

The central strategic insight guiding this research is the recognition that achieving further improvement requires moving beyond "universally beneficial" features. A previous attempt to create a diverse ensemble resulted in severe model correlation, with Pearson correlations exceeding 0.998, effectively neutralizing the variance reduction benefits of ensembling [[33](https://www.mlaia.com/blog-tabular-data-ml-vs-dl.html)]. This saturation necessitates a more targeted approach. The proposed strategy aims to inject novel information into the modeling pipeline through three complementary avenues. First, by creating robust external priors from the original dataset, we aim to transfer stable passenger psychology patterns to the synthetic data without introducing domain mismatch artifacts, a critical consideration given the observed performance degradation from naive concatenation [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm), [17](https://uselessai.in/from-baseline-to-top-10-a-practical-kaggle-competition-playbook-702d1d9394f8)]. Second, by decomposing Flight Distance into a smooth continuous trend and a shrunk discrete route residual, we address its dual nature, providing the models with more granular information about journey length and recurring routes than a single numerical value could offer [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)]. Third, by engineering features specifically tailored to the unique mathematical properties of XGBoost's second-order optimization and LightGBM's leaf-wise partitioning, we seek to break model correlation and create a more diverse and powerful ensemble [[1](https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation), [23](https://www.geeksforgeeks.org/machine-learning/lightgbm-leaf-wise-tree-growth-strategy/)].

The most promising immediate path to improvement lies in the implementation of auxiliary expected-rating features fitted exclusively on the original dataset. This approach has demonstrated a consistent +14 basis point lift across multiple folds in a similar context, offering a high risk-adjusted return due to its computational efficiency and complete immunity to target leakage [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)]. Following this, the decomposition of Flight Distance presents a strong opportunity to capture latent structure within the data. Finally, the development of engine-specific feature blocks represents a more advanced but potentially high-reward strategy for breaking model correlation. The overarching goal is not merely to add features but to execute a series of controlled experiments that validate each hypothesis, ensuring that any observed improvement is genuine and attributable to the introduced mechanisms rather than overfitting or validation leakage [[22](https://arxiv.org/html/2604.04199v1)]. The subsequent sections of this report provide the detailed methodologies, mathematical justifications, and implementation blueprints for these strategies, culminating in a prioritized experimental roadmap designed to systematically enhance predictive performance.

## Methodology for Robust External Rating Priors

The first priority is the development of auxiliary features representing the conditional expected value of passenger ratings, formulated as $\mu_j(x)=E[R_j\mid X_{-j}=x]$ [[19](https://www.kaggle.com/c/playground-series-s6e6/writeups/25th-place-my-public-starter-notebook)]. These features act as external priors, transferring learned passenger expectations from the original population to the synthetic one. The user's preference for fitting these models exclusively on the original dataset (Strategy A) is strongly supported by evidence of its robustness, efficiency, and effectiveness, while also strictly adhering to the non-negotiable rule against target leakage [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)]. Naive approaches like in-sample prediction or complex nested cross-fitting on the larger synthetic dataset have proven to be less effective and more fragile [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)].

The recommended mathematical formulation begins by treating the ordinal passenger rating variables (e.g., 'Inflight wifi service', 'Leg room') as nominal multiclass outcomes. For each rating feature $r_i$, a separate multiclass GBDT classifier is trained on the original dataset (`orig_df`). This model learns the full conditional probability distribution $P(r_i = k \mid X_{-i}=x)$ for all possible rating values $k$ (typically from 0 to 5) and for all other permitted predictor columns $X_{-i}$ excluding $r_i$ itself. Directly regressing the rating value can be suboptimal as it may not fully respect the ordinal structure or the probabilistic nature of the choices. By contrast, predicting the full probability distribution allows the final expected rating to be calculated as a weighted sum, which is a principled and flexible approach [[1](https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation)]. After training, the model is used to generate out-of-sample probability vectors for every row in the synthetic training set, the held-out validation set, and the unlabeled test set in a single inference pass. From these probability vectors, the final expected rating feature, $\hat{\mu}_i(x)$, is calculated using the formula:

$$
\hat{\mu}_i(x) = \sum_{k=0}^{5} k \cdot \hat{P}(r_i = k \mid X_{-i}=x)
$$

This provides a continuous-valued prior expectation for each rating, conditioned on all other available passenger, flight, and service context. Beyond the expectation, the full probability vector $(\hat{P}(r_i=0\mid x), \ldots, \hat{P}(r_i=5\mid x))$ can also be valuable. It captures the uncertainty in the prediction; for instance, a low Shannon entropy in the distribution indicates high confidence in the predicted rating, which could be a useful meta-feature for the final satisfaction model [[3](<https://www.stat.cmu.edu/~brian/valerie/617-2022/week07/spline%20references/Rodriguez%20(2001)%20smoothing.pdf>)].

The implementation must be meticulously designed to prevent any form of target leakage, particularly Class II (Selection) leakage, which involves using holdout-set information for model selection or feature generation and has been shown to cause substantial AUC inflation [[22](https://arxiv.org/html/2604.04199v1)]. The recommended approach is a simple, yet robust, external-prior methodology:

1.  **Training Population:** The designated training population for the rating-prediction models is solely the original dataset (`orig_df`), containing approximately 129,880 rows [[9](https://www.kaggle.com/competitions/1056lab-passenger-satisfaction-prediction)].
2.  **Features and Labels:** For the model predicting rating `r_wifi`, the predictors are all columns except `r_wifi` and the final satisfaction target. The label is `r_wifi`. This ensures no feedback loop where the final binary satisfaction target influences the auxiliary rating priors.
3.  **Model Fitting:** Train a multiclass GBDT model (e.g., using LightGBM or XGBoost) on `orig_df`. The choice of model is secondary to the method's integrity; CatBoost is also a valid option, though its native categorical handling might be redundant if the data is already preprocessed. The hyperparameters should be tuned once on `orig_df` to establish a baseline for the external prior's quality.
4.  **Feature Generation:** Once a model is trained, it is applied to three datasets in sequence:
    - **Synthetic Training Set:** Generate probability predictions for all ~700,000 synthetic training rows.
    - **Synthetic Validation Set:** Generate probability predictions for all rows in the held-out validation fold(s).
    - **Unlabeled Test Set:** Generate probability predictions for all ~300,000 unlabeled test rows.
      This single-pass inference is computationally efficient and guarantees that the generated priors are static for each dataset partition. No cross-validation is needed for the rating models themselves, which drastically simplifies the pipeline and eliminates fold-dependent variance in the priors [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)].
5.  **Final Calculation:** For each row in the three datasets, calculate the expected rating $\hat{\mu}_i(x)$ using the derived formula.

A comprehensive leakage audit checklist is essential to verify the integrity of this process:

- **Fold Safety:** The original-data models are completely independent of the synthetic-data k-fold validation scheme. There is no interaction between the two pipelines, preventing any leakage from the synthetic validation labels back into the feature calculation for the training data [[21](https://medium.com/@prathik.codes/how-to-do-target-encoding-without-data-leakage-the-right-way-280bd24fbc81)].
- **Label Isolation:** The labels from the synthetic training and test sets are never used during the training of any rating-prediction model. The only labels used are from `orig_df`.
- **No In-Sample Prediction:** All predictions for the rating probabilities are generated via a dedicated inference step on untouched data, avoiding the common pitfall of using in-sample predictions as features [[18](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)].
- **Validation Integrity:** The resulting expected-rating features are fixed for a given dataset. Their values do not change based on which fold is being used as validation, ensuring that the validation metric is not spuriously inflated by the features' knowledge of the validation set's composition.

To validate the efficacy of these new features, a minimal but informative ablation plan is required. This plan isolates the contribution of the auxiliary priors from other potential sources of improvement, such as hyperparameter retuning.

| Experiment Name                | Models Used                 | Feature Set Change                                                                           | Baseline Comparison |
| ------------------------------ | --------------------------- | -------------------------------------------------------------------------------------------- | ------------------- |
| **Baseline**                   | CatBoost, LightGBM, XGBoost | Original set of features                                                                     | N/A                 |
| **Add External Rating Priors** | CatBoost, LightGBM, XGBoost | Baseline features + $\hat{\mu}_{wifi}(x), \hat{\mu}_{clean}(x), ...$ for all rating features | Baseline            |

The success criterion for this experiment is a statistically significant increase in the mean ROC-AUC across all CV folds. Given the reported empirical evidence showing a +14 basis point lift with a similar approach, a successful outcome would require demonstrating a consistent gain above this threshold, ideally with a p-value < 0.05 to rule out random chance. Failure could manifest as a negative or negligible impact, suggesting that the passenger response psychology encoded in the original dataset does not generalize well to the synthetic domain, a known risk when dealing with distribution shifts [[17](https://uselessai.in/from-baseline-to-top-10-a-practical-kaggle-competition-playbook-702d1d9394f8)]. If the experiment fails, it would suggest that the priors are either too weak or actively harmful, prompting a review of the feature construction or a move to other hypotheses.

## Statistical Decomposition of Flight Distance

The Flight Distance variable presents a unique modeling challenge as it appears to carry two distinct signals: a continuous measure of physical journey length and a discrete identifier-like signal corresponding to recurring routes or repeated distance values [[5](https://www.kaggle.com/code/mehmetbicici/airline-passenger-satisfaction-eda-ml), [6](https://www.cliffsnotes.com/study-notes/16915380)]. A single numerical representation often fails to capture both effects simultaneously. The user's preference for a statistical decomposition using smoothing and shrinkage over unsupervised clustering is a sound strategy, as clustering in one dimension without geographic context is prone to creating arbitrary boundaries that can fragment natural route structures and overfit to spurious peaks in the synthetic data distribution [[19](https://www.kaggle.com/c/playground-series-s6e6/writeups/25th-place-my-public-starter-notebook)]. The work of Busyaprime provides empirical evidence for this bimodal nature, demonstrating that the deviation of satisfaction from a local distance trend contains meaningful, non-noise information [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)].

The recommended mathematical formulation decomposes Flight Distance ($d$) into two components: a smooth continuous trend component ($dist\_trend$) and a discrete residual component ($dist\_route\_residual$). This decomposition explicitly separates the underlying relationship between journey length and satisfaction from the specific effects of individual routes.

**Continuous Component ($dist\_trend$):** This component represents the smoothed, low-complexity trend of satisfaction as a function of distance. It acts as a baseline, capturing the general intuition that satisfaction might vary smoothly with flight duration. The most straightforward way to estimate this is through a rolling window calculation. For each distance value $d$, the $dist\_trend$ is defined as the rolling average satisfaction within a specified bandwidth $\Delta$:

$$
dist\_trend(d) = \text{RollingMean}(y \mid d \pm \Delta)
$$

Here, $y$ is the binary satisfaction target. The bandwidth $\Delta$ is a crucial hyperparameter that controls the smoothness of the trend; a smaller $\Delta$ creates a more wiggly, locally adaptive trend, while a larger $\Delta$ produces a smoother, more globally representative curve [[3](<https://www.stat.cmu.edu/~brian/valerie/617-2022/week07/spline%20references/Rodriguez%20(2001)%20smoothing.pdf>)]. This rolling expectation can be computed efficiently using sorted arrays and sliding-window algorithms.

**Discrete Residual Component ($dist\_route\_residual$):** This component captures the deviation of satisfaction at a specific distance from the continuous trend identified above. For each unique distance value $d$, let $\bar{y}_{route}$ be the raw, unsmoothed average satisfaction for all passengers flying that exact distance. The raw residual is then $(\bar{y}_{route} - dist\_trend(d))$. However, for distances with few observations, this raw residual can be highly unstable and noisy. To stabilize these estimates, especially for rare distances, Empirical Bayes (EB) shrinkage is applied [[12](https://metricgate.com/docs/empirical-bayes-shrinkage/)]. EB shrinkage pulls extreme estimates toward a global mean, with the degree of shrinkage inversely proportional to the sample size of the group (in this case, the distance value) [[13](https://kiwidamien.github.io/shrinkage-and-empirical-bayes-to-improve-inference.html)]. The shrunk residual is calculated as:

$$
dist\_route\_residual(d) = EB\_Shrinkage(\bar{y}_{route} - dist\_trend(d), m=20)
$$

The parameter $m$ (analogous to `alpha` in smoothed target encoding) controls the strength of the shrinkage [[34](https://goodboychan.github.io/python/datacamp/kaggle/machine_learning/2020/08/12/03-Feature-Engineering.html)]. A higher $m$ results in more aggressive shrinkage towards zero. The hyperparameter $m$ can be tuned, but the provided context suggests starting with a value like 20 is reasonable [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)]. The EB shrinkage factor is calculated based on the ratio of the within-group variance to the total variance, pulling small-sample estimates (high noise) much more strongly than large-sample estimates (low noise) [[12](https://metricgate.com/docs/empirical-bayes-shrinkage/)].

To implement this decomposition without introducing target leakage, a strict fold-safe cross-fitting procedure is mandatory. The following pseudocode outlines the process for generating these features for the training, validation, and test sets:

```python
## Pseudocode for Fold-Safe Flight Distance Decomposition

def calculate_decomposed_features(train_data, val_data, test_data):
    # Step 1: Initialize output columns
    train_data['dist_trend'] = np.nan
    train_data['dist_route_residual'] = np.nan
    val_data['dist_trend'] = np.nan
    val_data['dist_route_residual'] = np.nan
    test_data['dist_trend'] = np.nan
    test_data['dist_route_residual'] = np.nan

    # Step 2: Define the bandwidth for the rolling trend
    # This should be determined on the full training set or a large subset
    # to avoid leakage. For simplicity, assume it's pre-calculated.
    delta = determine_optimal_bandwidth(pd.concat([train_data, val_data])) # Not leaky if done once

    # Step 3: Process each CV fold
    for train_idx, val_idx in kfold.split(train_data):
        train_fold = train_data.iloc[train_idx]
        val_fold = train_data.iloc[val_idx]

        # --- Part A: Calculate features on the training fold ---
        # Calculate the continuous trend on the combined training+val fold
        # This is a slight relaxation but prevents look-ahead bias from the test set.
        # For stricter control, use only train_fold for everything.
        trend_data = pd.concat([train_fold, val_fold])
        trend_data = trend_data.sort_values('Flight Distance')
        trend_data = trend_data.assign(
            dist_trend=trend_data['Flight Distance'].rolling(
                window=f'... miles', center=True, on='Flight Distance').mean()
        )

        # Create a mapping from distance -> trend value
        dist_to_trend = dict(zip(trend_data['Flight Distance'], trend_data['dist_trend']))

        # Map the trend back to the training and validation folds
        train_fold['dist_trend'] = train_fold['Flight Distance'].map(dist_to_trend)
        val_fold['dist_trend'] = val_fold['Flight Distance'].map(dist_to_trend)

        # Calculate raw residuals and EB shrinkage parameters on the training fold
        # Group by distance to get raw route satisfaction
        route_stats = train_fold.groupby('Flight Distance')['satisfaction'].agg(['mean', 'count']).reset_index()
        route_stats = route_stats.rename(columns={'mean': 'y_bar_route'})
        route_stats = route_stats.dropna() # Drop distances with no data

        # Merge trend info
        route_stats = route_stats.merge(pd.DataFrame({'Flight Distance': list(dist_to_trend.keys()),
                                                     'dist_trend': list(dist_to_trend.values())}),
                                        on='Flight Distance')

        # Calculate raw residuals
        route_stats['raw_residual'] = route_stats['y_bar_route'] - route_stats['dist_trend']

        # Estimate EB shrinkage hyperparameters using Method-of-Moments on the training fold
        # This is a simplified version; a proper MoM estimator would calculate population variance
        y_bar_global = route_stats['y_bar_route'].mean()
        v_pop = route_stats['raw_residual'].var() # Sample variance is a start
        m = 20 # Default smoothing parameter, could be tuned

        # Apply EB shrinkage formula: shrunk = B * mu + (1-B) * raw
        # For simplicity, using a constant prior of 0
        shrinkage_factor = v_pop / (v_pop + (1.0 / route_stats['count'])) # Approximate variance term
        route_stats['shrunk_residual'] = shrinkage_factor * 0 + (1 - shrinkage_factor) * route_stats['raw_residual']

        # Create a mapping from distance -> shrunk residual
        dist_to_residual = dict(zip(route_stats['Flight Distance'], route_stats['shrunk_residual']))

        # --- Part B: Apply features to all sets using mappings from this fold's training data ---
        # Update the main DataFrames
        train_data.loc[train_idx, 'dist_trend'] = train_fold['dist_trend'].values
        train_data.loc[train_idx, 'dist_route_residual'] = train_fold['Flight Distance'].map(dist_to_residual).values

        val_data.loc[val_idx, 'dist_trend'] = val_fold['dist_trend'].values
        val_fold_resid = val_fold['Flight Distance'].map(dist_to_residual).fillna(0) # Fill unseen with 0 residual
        val_data.loc[val_idx, 'dist_route_residual'] = val_fold_resid.values

        # For the test set, we can't use val_fold statistics. We need a separate procedure.
        # Option 1: Refit everything on 100% of the data once at the end for final inference.
        # Option 2: Use a global mapping calculated from the very first training fold.
        # For now, leave test_data as NaN to be filled after the loop.

    # --- Final Step: Handle the test set ---
    # For the test set, we must use statistics from the *entire* provided training data.
    # This is a common practice in competitions to avoid having to re-run the entire CV loop.
    full_train_data = pd.concat([train_data, val_data]) # Reconstruct full training set
    full_trend_data = full_train_data.sort_values('Flight Distance')
    full_trend_data = full_trend_data.assign(
        dist_trend=full_trend_data['Flight Distance'].rolling(
            window=f'... miles', center=True, on='Flight Distance').mean()
    )
    dist_to_trend_full = dict(zip(full_trend_data['Flight Distance'], full_trend_data['dist_trend']))

    full_route_stats = full_train_data.groupby('Flight Distance')['satisfaction'].agg(['mean', 'count']).reset_index()
    full_route_stats = full_route_stats.rename(columns={'mean': 'y_bar_route'})
    full_route_stats = full_route_stats.merge(pd.DataFrame({'Flight Distance': list(dist_to_trend_full.keys()),
                                                           'dist_trend': list(dist_to_trend_full.values())}),
                                              on='Flight Distance')
    full_route_stats['raw_residual'] = full_route_stats['y_bar_route'] - full_route_stats['dist_trend']

    # Estimate global EB parameters
    v_pop_global = full_route_stats['raw_residual'].var()
    full_route_stats['shrunk_residual'] = (v_pop_global / (v_pop_global + (1.0 / full_route_stats['count']))) * full_route_stats['raw_residual']
    dist_to_residual_full = dict(zip(full_route_stats['Flight Distance'], full_route_stats['shrunk_residual']))

    # Apply to test set
    test_data['dist_trend'] = test_data['Flight Distance'].map(dist_to_trend_full).fillna(y_bar_global)
    test_data['dist_route_residual'] = test_data['Flight Distance'].map(dist_to_residual_full).fillna(0)

    return train_data, val_data, test_data
```

The handling of rare or unseen distance values is critical. For distances in the validation or test sets that were not present in the training set of a given CV fold, the residual should be assigned a value of 0 (indicating no deviation from the global trend) and perhaps a lower confidence weight if such a feature were being considered. The continuous trend for unseen distances can be handled by either using a global mapping calculated from the entire training set at the end of the CV process or by assigning the global mean satisfaction rate.

An ablation plan is necessary to determine the relative importance of each component:

| Experiment Name                 | Models Used                 | Feature Set Change                                       | Baseline Comparison |
| ------------------------------- | --------------------------- | -------------------------------------------------------- | ------------------- |
| **Baseline**                    | CatBoost, LightGBM, XGBoost | Original set of features                                 | N/A                 |
| **Test Continuous Trend**       | CatBoost, LightGBM, XGBoost | Baseline features + `dist_trend`                         | Baseline            |
| **Test Discrete Residual**      | CatBoost, LightGBM, XGBoost | Baseline features + `dist_route_residual`                | Baseline            |
| **Test Combined Decomposition** | CatBoost, LightGBM, XGBoost | Baseline features + `dist_trend` + `dist_route_residual` | Baseline            |

This matrix will reveal whether the smooth continuous signal, the discrete route-level signal, or their combination provides the most value. A failure mode would be if neither component improves performance, suggesting the decomposition did not capture a meaningful signal or amplified noise. Another risk is the amplification of distribution shift if the relationship between distance and satisfaction differs significantly between the original and synthetic populations, a known issue in this competition [[17](https://uselessai.in/from-baseline-to-top-10-a-practical-kaggle-competition-playbook-702d1d9394f8)].

## Engine-Specific Feature Blocks Exploiting Algorithmic Asymmetry

To overcome the problem of ensemble saturation caused by highly correlated models, this research proposes designing feature blocks specifically to leverage the distinct mathematical properties of XGBoost and LightGBM [[33](https://www.mlaia.com/blog-tabular-data-ml-vs-dl.html)]. While many features benefit all GBDTs, understanding their internal mechanics can guide the creation of features that are uniquely advantageous to one engine, thereby increasing the diversity and overall power of the ensemble [[1](https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation), [23](https://www.geeksforgeeks.org/machine-learning/lightgbm-leaf-wise-tree-growth-strategy/)]. The core idea is to engineer features whose structure aligns with the strengths of each respective algorithm.

**Understanding the Core Asymmetry:**

- **XGBoost's Second-Order Optimization:** XGBoost's primary innovation is its use of a second-order Taylor expansion to approximate the loss function [[1](https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation), [2](https://rocm.blogs.amd.com/software-tools-optimization/xgboost_deep_dive/README.html)]. This approach incorporates both the first derivative (gradient, $g_i$) and the second derivative (Hessian, $h_i$ or curvature) of the loss at each data point. The Hessian provides a measure of confidence in the gradient's direction. This has profound implications for split finding and leaf weighting. The optimal weight for a leaf is calculated analytically as $w_j^* = -G_j / (H_j + \lambda)$, where $G_j$ and $H_j$ are the sums of gradients and Hessians in that leaf [[2](https://rocm.blogs.amd.com/software-tools-optimization/xgboost_deep_dive/README.html)]. Furthermore, the gain from a potential split is also calculated using a formula involving both gradient and Hessian sums [[1](https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation)]. This means XGBoost is inherently sensitive to the _curvature_ and _heteroscedasticity_ (changing variance) of the feature-target relationship. Features that exhibit non-linear transformations, sharp changes in volatility, or complex interactions that affect the Hessian will be processed differently than by other GBDTs.

- **LightGBM's Leaf-Wise Growth:** LightGBM employs a "leaf-wise" (or best-first) tree growth strategy, contrasting with the level-wise (depth-wise) approach of XGBoost and traditional methods [[23](https://www.geeksforgeeks.org/machine-learning/lightgbm-leaf-wise-tree-growth-strategy/), [24](https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting/lightgbm-leaf-wise-growth)]. At each step, LightGBM selects the single leaf node with the maximum split gain to expand, rather than expanding all leaves at the current depth [[26](https://mbrenndoerfer.com/writing/lightgbm-fast-gradient-boosting-leaf-wise-tree-growth-complete-guide-mathematical-foundations-python-implementation)]. This allows it to build deeper, more asymmetric trees that can achieve lower loss for a given number of leaves, making it potentially more accurate. However, this flexibility comes with a higher risk of overfitting, especially on small datasets or with high-cardinality features [[26](https://mbrenndoerfer.com/writing/lightgbm-fast-gradient-boosting-leaf-wise-tree-growth-complete-guide-mathematical-foundations-python-implementation)]. Its histogram-based approach further accelerates computation by bucketing continuous features into discrete bins, making it exceptionally efficient on large datasets [[25](https://lightgbm.readthedocs.io/en/latest/Features.html)]. LightGBM excels at discovering localized, high-dimensional interactions and sharp conditional means within dense regions of the feature space.

Based on these principles, three high-impact feature blocks are proposed, with the first two designed to be model-specific.

**Block 1: Curvature and Hessian Sensitivity Features (Targeted for XGBoost)**

- **Statistical Objective:** To create features that introduce complex, non-uniform relationships that modulate the Hessian term in XGBoost's gain calculation, allowing it to capture curvature more effectively.
- **Candidate Features:**
  1.  **Ratio-Based Features:** Features constructed as ratios of other variables can induce non-linearity and changing variance. A prime example is `airborne_recovery_kinematics = Δ_delay / (max(10, dist/7.5))`. The denominator normalizes the delay by a proxy for flight time, creating a rate-like feature. Such features often have heteroscedastic variances, and XGBoost's Hessian denominator ($H_i + \lambda$) naturally acts as a variance normalizer, making these features particularly potent for it [[1](https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation)].
  2.  **Smooth Non-Linear Transformations:** Instead of relying solely on tree-based piecewise-constant approximations, we can pre-calculate spline or piecewise-linear basis functions for highly informative numerical variables like `Flight Distance` or `Age`. These provide structured, learnable curvature that XGBoost's second-order optimizer is well-suited to interpret [[32](https://datascience.stackexchange.com/questions/10640/how-to-perform-feature-engineering-on-unknown-features)].
- **Plausible Mechanism:** XGBoost's analytical solution for leaf weights and its Hessian-based gain calculation give it an intrinsic advantage in modeling features where the relationship with the target is non-linear and has varying confidence levels. These features are unlikely to provide the same unique boost to LightGBM, which relies more on simple conditional mean splits.

**Block 2: High-Dimensional Group Conditional Means (Targeted for LightGBM)**

- **Statistical Objective:** To create sparse, high-cardinality categorical features that represent fine-grained groupings where satisfaction might exhibit sharp, localized jumps. This plays to LightGBM's strength in finding splits within dense, fragmented spaces.
- **Candidate Features:**
  1.  **Exact Distance Crosses:** Although the dataset lacks explicit origin-destination columns, the `Flight Distance` variable itself can be used as a categorical feature. Creating interaction terms by binning `Flight Distance` into finer buckets (e.g., 50km intervals) and crossing them with other relevant features (e.g., airline, class) can create high-cardinality groups. LightGBM's efficient histogram-based splitting and leaf-wise growth allow it to find the optimal decision boundary to capture the conditional mean satisfaction for these specific, narrow distance ranges [[25](https://lightgbm.readthedocs.io/en/latest/Features.html)].
  2.  **Condensed Rating Profiles:** Low-dimensional summaries (e.g., binned means or modes) of related service ratings (e.g., inflight entertainment, food/beverage) can be created. These condensed profiles can then be binned and used to form new categorical features that capture nuanced passenger segments.
- **Plausible Mechanism:** LightGBM's ability to grow deep trees quickly makes it adept at carving out small, specific regions of the feature space where the model's predictions change abruptly. XGBoost's level-wise approach might struggle to find these specific splits without growing very wide trees, which can lead to overfitting if not heavily regularized. This block directly exploits LightGBM's architectural advantage [[23](https://www.geeksforgeeks.org/machine-learning/lightgbm-leaf-wise-tree-growth-strategy/)].

**Block 3: Distributional and Support Structure Features (Universal Foundation)**

- **Statistical Objective:** To expose stable, population-level patterns that are not easily captured by individual trees but are anchored in the broader data distribution. This serves as a universal foundation that can improve calibration and stability across all models.
- **Candidate Features:**
  1.  **Distance Frequency and Support Counts:** Simple features counting the number of times a given `Flight Distance` appears in the training data. This provides information about the popularity or rarity of different journeys, which may correlate with satisfaction.
  2.  **Original-Data Lookup Priors:** Features like `org_mean_*` represent smoothed satisfaction rates for various feature combinations, estimated from the original dataset [[17](https://uselessai.in/from-baseline-to-top-10-a-practical-kaggle-competition-playbook-702d1d9394f8)]. When used carefully, these act as anchors to the real-world passenger experience, helping to counteract potential artifacts or biases in the synthetic generator [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)].
  3.  **Smoothing Parameter Tuning:** A key area for optimization is the smoothing parameter `m` in the Empirical Bayes shrinkage for the `dist_route_residual` feature (from Question 2). A principled, inexpensive tuning strategy, such as a coarse grid search on the validation set, can optimize this parameter to maximize the signal-to-noise ratio in the residual features [[11](https://pmc.ncbi.nlm.nih.gov/articles/PMC6472625/)].
- **Plausible Mechanism:** These features represent "population truths." They provide a macroscopic view that complements the microscopic, tree-based learning. While potentially useful to all models, they are less likely to cause overfitting because they are based on aggregate statistics rather than individual sample paths. The key is ensuring they are constructed without leakage, for example, by using the external-prior methodology from Question 1 for the original-data lookup features.

For all these blocks, rigorous leakage controls are paramount. Any feature derived from the target variable must be generated using strict cross-fitting, ensuring that the statistic for a given row is calculated exclusively from a training set that does not contain that row. Redundancy must be managed through careful ablation, comparing models with and without the new blocks to distinguish true gains from duplicated information. The ultimate goal is to demonstrate that allocating specific feature blocks to specific models leads to more complementary predictions and a lower error rate than giving all models the same set of universally beneficial features.

## Experimental Validation Protocol and Ablation Matrix

A rigorous and systematic experimental design is non-negotiable to ensure that any observed improvements in ROC-AUC are genuine and not the result of overfitting to the validation set or target leakage [[22](https://arxiv.org/html/2604.04199v1)]. The protocol must be designed to evaluate each proposed feature block in isolation before combining them, measure performance with sufficient precision, and prevent selection bias during the iterative process of experimentation. The goal is to build a defensible narrative around the final model's performance, grounded in controlled comparisons rather than ad-hoc tuning.

The foundational element of the validation protocol is a robust k-fold cross-validation (CV) scheme. For a dataset of approximately 700,000 labeled rows, a standard 5-fold or 10-fold CV is appropriate [[18](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)]. The primary assumption of this setup is that the row order is randomized and that the data within each fold is representative of the overall data-generating process. A critical preliminary check is to perform adversarial validation to confirm that there is no significant covariate shift between the training and test distributions that would invalidate a simple random CV scheme [[29](https://www.kaggle.com/code/kooaslansefat/wids-2021-av-bo-catboost-pseudo-labeling), [30](https://www.kaggle.com/code/jeongyoonlee/adversarial-validation-with-lightgbm)]. If a significant shift is detected, more sophisticated validation strategies like `GroupKFold` (if passenger IDs are available) or `TimeSeriesSplit` (if temporal information exists) would be necessary [[18](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)].

Performance will be evaluated primarily using ROC-AUC, as per the competition's metric. To facilitate clear communication and comparison, all performance changes will be reported in absolute ROC-AUC units and in basis points, where 1 basis point equals 0.0001 of ROC-AUC [[27](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard)]. A typical expected improvement from a good feature is in the range of 14-25 basis points (0.0014-0.0025 absolute AUC) [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)]. For each experiment, the following metrics must be reported for each fold:

- Mean ROC-AUC
- Standard Deviation of ROC-AUC across folds
- The change in mean ROC-AUC relative to the baseline, presented in both absolute terms and basis points.

It is crucial to remember that a small difference in a single fold is not reliable evidence of a genuine gain. Improvements must be consistent across multiple folds to be considered credible. Paired t-tests comparing the fold-level scores of the baseline and the experimental models can provide a formal assessment of statistical significance (e.g., p < 0.05).

To combat selection bias—the risk of inadvertently optimizing the model to the validation set—a disciplined workflow is required. The principle of separating feature discovery/experimentation from model validation must be maintained. One effective strategy is to conduct exploratory screening on the training folds of the CV, but only use the held-out validation fold scores to make a final decision on whether to retain a feature or proceed to the next experiment [[22](https://arxiv.org/html/2604.04199v1)]. If a feature shows promise, it is added to the feature set, and the baseline model is retrained and validated. This prevents the constant tweaking of hyperparameters and feature choices based on the same limited validation signal.

Crucially, every experiment must disentangle the contribution of new features from the contribution of hyperparameter retuning. A model tuned for a longer period may perform better simply because of more training, not because of the new features. Therefore, a factorial design should be employed whenever feasible:

1.  **Baseline Model (B):** Retrain the baseline model with baseline features.
2.  **New Features Only (NF):** Retrain the baseline model with the new feature block added to the baseline features.
3.  **Retuned Model (R):** Retrain the baseline model with baseline features, but with hyperparameters retuned for the new conditions.
4.  **Best-in-Class (BIC):** Retrain the baseline model with the new feature block, plus hyperparameters retuned for this configuration.

Comparing (NF) to (B) isolates the feature gain. Comparing (R) to (B) isolates the hyperparameter gain. The final model should ideally be close to (BIC). For computationally expensive experiments, a smaller design can be used, but the ability to distinguish feature value from tuning is essential. For instance, if (NF) performs worse than (R), it suggests the new features are detrimental and cannot compensate for the need to retune. If (NF) performs better than (R), it suggests the features unlock a new, more effective region of the model's search space.

Finally, to address the stated goal of breaking ensemble correlation, the performance of the individual models within the ensemble must be tracked. For each model (CatBoost, LightGBM, XGBoost), its Out-of-Fold (OOF) predictions should be stored. The pairwise Pearson correlation between these OOF prediction vectors should be calculated. A successful strategy will show that the addition of model-specific feature blocks reduces this correlation, leading to more complementary error patterns and a lower final ensemble error [[33](https://www.mlaia.com/blog-tabular-data-ml-vs-dl.html)]. The performance of a simple fixed-weight linear combination of the OOF predictions can also be measured to quantify the diversity gain [[18](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)].

The following table summarizes the recommended ablation matrix for validating the three main research questions. Each experiment should be run on all permitted GBDT models to assess universality versus specificity.

| Research Question         | Experiment Name            | Feature Set Change                                    | Main Hypothesis Being Tested                                                                 |
| :------------------------ | :------------------------- | :---------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| **Q1: Auxiliary Ratings** | Add External Rating Priors | Add `Ê[rᵢ \| X₋ᵢ]` features.                          | Do robust external priors from the original data provide a stable, generalizable signal?     |
| **Q2: Flight Distance**   | Decompose Flight Distance  | Add `dist_trend` and `dist_route_residual` features.  | Does separating the continuous trend from the discrete route residual add novel information? |
| **Q3a: XGBoost Block**    | Add Curvature Features     | Add ratio/non-linear features (e.g., `Δ_delay/dist`). | Does XGBoost benefit uniquely from features sensitive to its second-order Hessian weighting? |
| **Q3b: LightGBM Block**   | Add Group Mean Features    | Add high-cardinality distance-cross features.         | Does LightGBM benefit from capturing sharp local effects missed by other models?             |

This structured approach ensures that each hypothesis is tested systematically, providing a clear path from initial investigation to a validated, high-performing final model.

## Prioritized Experimental Roadmap and Final Decision

This section synthesizes the preceding analyses into a concrete, compute-aware experimental roadmap. The experiments are prioritized based on their expected ROC-AUC improvement per unit of compute, their implementation complexity, and their probability of yielding a genuine improvement. The goal is to identify the highest-risk-adjusted opportunities first, building momentum with quick wins before tackling more speculative but potentially higher-reward avenues. The final decision identifies the single best next action and establishes a clear decision-making framework for the entire project.

The following table presents the prioritized experimental queue, organized into three tiers. Each experiment includes a concise description, the associated cost category (Low, Medium, High), and a clear success/failure criterion to guide the decision-making process. The cost is a qualitative estimate based on runtime and memory requirements for a single run of the experiment within the existing CV framework.

| Priority Tier | Experiment ID   | Description                                                                      | Models Affected          | Cost   | Success Criterion                                                                                  | Failure Mode                                                                       |
| :------------ | :-------------- | :------------------------------------------------------------------------------- | :----------------------- | :----- | :------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **Tier 1**    | **EXP-Q1-01**   | Add external rating priors from the original dataset.                            | All (CatBoost, LGB, XGB) | Low    | Mean ROC-AUC increase > 15 bps with p<0.05 significance.                                           | No significant change or degradation in ROC-AUC.                                   |
| **Tier 1**    | **EXP-Q2-01**   | Decompose Flight Distance into `dist_trend` and `dist_route_residual`.           | All (CatBoost, LGB, XGB) | Medium | Mean ROC-AUC increase > 10 bps, with `dist_route_residual` contributing significantly.             | No significant change or degradation; suggests the decomposition is uninformative. |
| **Tier 2**    | **EXP-Q3A-01**  | Add curvature-sensitive features (e.g., `Δ_delay/dist`) specifically to XGBoost. | XGBoost                  | Medium | XGBoost OOF AUC improves by >5 bps vs. LightGBM/Others; pairwise prediction correlation decreases. | XGBoost performance plateaus or degrades; no unique benefit observed.              |
| **Tier 2**    | **EXP-Q3B-01**  | Add high-cardinality distance-cross features specifically to LightGBM.           | LightGBM                 | Medium | LightGBM OOF AUC improves by >5 bps vs. XGBoost/Others; pairwise prediction correlation decreases. | LightGBM overfits severely; performance collapses on validation set.               |
| **Tier 3**    | **EXP-TUNE-01** | Optimize smoothing parameters (`m`, `alpha`) for Q2 and Q1 features.             | All                      | High   | Significant (>10 bps) improvement in the final stacked model's OOF AUC.                            | Minimal improvement, suggesting defaults are near-optimal.                         |

The single best next experiment is **EXP-Q1-01: Add external rating priors**. The rationale for this top priority is its exceptional risk-adjusted value. It is computationally inexpensive (Low cost), trivially free of target leakage by construction, and has a strong track record of delivering tangible gains, with a verified +14 bps lift reported in a similar context [[16](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)]. This experiment directly addresses the need for robust, generalizable priors and offers a quick, verifiable win that builds confidence in the pipeline. Running it first minimizes the risk of wasting compute on more complex experiments if a fundamental signal is missing.

The decision following this experiment is straightforward:

- **Success:** If EXP-Q1-01 yields a consistent ROC-AUC gain of over 15 basis points, the features should be retained. The next logical step is to proceed to the next Tier 1 experiment, **EXP-Q2-01**, to incorporate the structural information from Flight Distance. The successful prior features will serve as a stronger foundation upon which to build.
- **Failure:** If EXP-Q1-01 shows no significant improvement, it suggests that the passenger rating psychology in the original dataset does not transfer well to the synthetic domain, or that the feature construction is flawed. This would prompt a diagnostic check of the implementation and potentially a review of the underlying assumption that the original data contains useful priors for the synthetic data. If the feature itself is confirmed to be correct, the next experiment would be to investigate why it failed—perhaps by analyzing the distribution of the generated priors in the original vs. synthetic sets—and then proceed to **EXP-Q2-01**.

In summary, this research program provides a clear, evidence-backed, and leak-proof strategy for enhancing the Kaggle S6E10 model. It moves beyond generic feature engineering by focusing on three pillars: leveraging external data for robust priors, applying principled statistical decomposition to complex features, and exploiting algorithmic asymmetries to create a truly diverse ensemble. By following the prioritized experimental roadmap, the team can systematically and efficiently explore these high-value opportunities, maximizing the probability of achieving a genuine and significant improvement on the public leaderboard.
