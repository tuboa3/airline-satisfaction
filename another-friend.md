# **Research Report: GBDT Architectural Diversity, Leakage-Safe Ensembling, and Categorical Split Optimization for Kaggle S6E10**

## **Executive Diagnosis**

The central research problem defining the Kaggle Playground Series S6E10 competition involves standalone out-of-fold (OOF) Gradient Boosted Decision Tree (GBDT) models that achieve highly respectable standalone Receiver Operating Characteristic Area Under the Curve (ROC-AUC) scores, yet exhibit near-perfect prediction correlation. Current baseline empirical evidence indicates standalone OOF ROC-AUC scores of 0.96092 for CatBoost, 0.96061 for LightGBM, and 0.96037 for XGBoost, with reported inter-model Pearson correlations exceeding 0.9978 and reaching 0.9989 between LightGBM and XGBoost1.  
A high Pearson correlation among models utilizing distinct algorithmic tree-growth policies is highly anomalous on real-world datasets, where differing inductive biases typically explore distinct hypothesis spaces. The diagnosis of this phenomenon rests on separating the measured empirical evidence from the theoretical hypotheses regarding the optimization surface.  
The first plausible cause of this extreme prediction similarity is the existence of a genuinely low-dimensional predictive signal resulting from the synthetic data generation process. The working dataset comprises approximately 700,000 synthetic training rows and 300,000 synthetic test rows, generated from an original dataset of 130,000 rows2. Kaggle Playground Series competitions predominantly utilize generative architectures such as Conditional Tabular Generative Adversarial Networks (CTGAN) or Tabular Variational Autoencoders (TVAE) to synthesize these datasets4. While CTGANs successfully replicate marginal distributions and low-order interactions, empirical research demonstrates that they frequently fail to faithfully reproduce high-frequency, complex combinatorial interactions present in the original real-world data6. Consequently, the resulting synthetic data occupies a smoother, lower-dimensional manifold. GBDTs, acting as universal approximators, easily and rapidly model this primary manifold. Because the synthetic generation process strips out idiosyncratic variance and complex copula dependencies, the models exhaust the available predictive signal rapidly, leaving no "hard" residual patterns for structurally diverse algorithms to capture differently. The correlation is therefore hypothesized to be a reflection of a synthetic dataset containing a fundamentally limited informational capacity, rather than a failure of hyperparameter tuning.  
The second plausible cause involves redundant feature engineering and homogenized categorical ingestion overriding algorithm-specific inductive biases. While CatBoost, LightGBM, and XGBoost handle categorical features differently in theory, practitioners frequently force identical representations, such as global label encoding or shared integer mapping, across all models to simplify the preprocessing pipeline8. If nominal variations of GBDTs are fed identical target-derived or frequency-based engineered features, the unique inductive biases of the specific tree-growth algorithms are nullified by the overwhelming strength of the engineered representation. The trees simply learn to split on the high-cardinality engineered statistics in the exact same sequence. When categorical ingestion is artificially homogenized, the learned decision boundaries inevitably homogenize, driving Pearson correlation toward 1.0.  
The third plausible cause is sub-optimal over-regularization artificially limiting effective model capacity. Standard hyperparameter optimization frameworks, such as Optuna, often favor highly regularized parameter spaces on large tabular datasets to maximize OOF ROC-AUC by minimizing variance9. This includes enforcing large minimum data requirements per leaf, aggressive L2 regularization, and restricted maximum depths. While this restricts overfitting, it artificially compresses the hypothesis space. When three distinctly designed algorithms are heavily regularized, they are mathematically constrained to only the most dominant, global splits. By prohibiting deep, localized asymmetric splits, which LightGBM inherently favors, or deep symmetric interactions, which CatBoost is designed to capture, the models are forced to produce identical, shallow representations. The extreme correlation arises because the models lack the effective capacity to express their architectural uniqueness, converging on the exact same generalized function.

## **Research and Documentation Audit**

A rigorous audit of the official documentation for the permitted GBDT frameworks, alongside credible research on ensemble diversity, reveals several mechanical pathways that can be exploited to force architectural divergence and generate complementary predictive errors.  
The official CatBoost documentation details its unique reliance on oblivious, or symmetric, decision trees and ordered target statistics. The grow\_policy parameter controls the architectural foundation of the model9. The default policy is SymmetricTree, which builds trees level by level, utilizing the identical split condition across all nodes at a given depth, enforcing a powerful global regularization that differs fundamentally from other frameworks13. Furthermore, CatBoost's processing of categorical features depends heavily on the execution hardware. The official documentation at https\://catboost.ai/docs/en/references/training-parameters/ctr explicitly outlines that on GPU, the CtrType for simple and combined categorical features supports unique representations such as FloatTargetMeanValue and FeatureFreq16. FloatTargetMeanValue computes the target mean for the category using only objects placed prior to the current object in an artificial time permutation, successfully mitigating target leakage14. Stochastic parameters such as bagging\_temperature and random\_strength are critical for inducing diversity; bagging\_temperature dictates the intensity of the Bayesian bootstrap, while random\_strength adds a normally distributed random variable to the calculated split score during the greedy search, perturbing the split selection without altering the underlying data sample9.  
The official LightGBM documentation at https\://lightgbm.readthedocs.io/en/latest/Parameters.html reveals mechanisms designed to create highly localized partitions through best-first, leaf-wise growth10. A critical parameter for enforcing diversity is extra\_trees. When set to true, this parameter fundamentally alters the split-finding algorithm. Instead of exhaustively computing the histogram to find the mathematically optimal split threshold, LightGBM evaluates only one randomly chosen threshold per feature and selects the best among them10. For native categorical handling, max\_cat\_threshold restricts the number of split points evaluated, while cat\_smooth provides Laplacian-like smoothing to the target statistics of categorical splits10. Furthermore, the path\_smooth parameter acts as a Bayesian prior on leaf outputs, shrinking the leaf prediction toward the parent node's prediction utilizing the formula: \$\\text{leaf\\\_output} \= \\frac{n}{n \+ \\text{path\\\_smooth}} \\times \\text{new\\\_leaf\\\_output} \+ \\frac{\\text{path\\\_smooth}}{n \+ \\text{path\\\_smooth}} \\times \\text{old\\\_leaf\\\_output}\$23. This introduces a powerful structural regularization independent of standard tree depth constraints.  
The official XGBoost documentation at https\://xgboost.readthedocs.io/en/stable/parameter.html details a robust depth-wise growth architecture. Native categorical support requires the explicit combination of enable\_categorical=True and tree\_method="hist"26. XGBoost utilizes one-hot encoding for low cardinality features and partition-based splits for high cardinality, governed by max\_cat\_to\_onehot and max\_cat\_threshold8. XGBoost provides granular control over feature sampling via colsample\_bytree, colsample\_bylevel, and colsample\_bynode31. The utilization of colsample\_bynode forces the model to evaluate entirely different feature subsets at every individual split, dramatically altering the structural path of the trees compared to traditional per-tree sampling.  
The Scikit-Learn HistGradientBoostingClassifier documentation provides an additional permitted architecture. This implementation offers native categorical support without requiring manual one-hot encoding by setting categorical\_features="from\_dtype"11. A unique feature for inducing diversity in this framework is interaction\_cst, which allows the practitioner to explicitly define disjoint sets of features that are permitted to interact35. By forcing specific features into separate interaction groups, the model is compelled to build purely additive structural components, guaranteeing a mathematically distinct decision boundary from models lacking interaction constraints35.  
Recent academic research on ensemble diversity in gradient-boosted trees emphasizes that prediction correlation is an insufficient metric for capturing the utility of an ensemble. Research published on arXiv establishes that knowledge uncertainty in GBDTs can be estimated by the level of spread or "disagreement" among models in an ensemble, and that injecting stochasticity via subsampling or randomized thresholds effectively broadens the explored hypothesis space37. Further research demonstrates that regularized ensembles, which dynamically drop base model predictions during training, successfully lower-bound ensemble diversity and prevent the formation of low-diversity combinations on complex tabular data38. Additionally, studies on the generalization gaps of tree ensembles confirm that while tree-based models achieve state-of-the-art performance on nonlinear tabular tasks, their performance is highly sensitive to the balance between model capacity and regularization on noisy datasets39.  
An analysis of Kaggle Playground Series S6E10 public notebooks reveals a clustering of OOF ROC-AUC scores between 0.9583 and 0.961621. The baseline models heavily rely on standard parameter configurations without exploring the deeper architectural perturbations available within the GBDT frameworks. The reported inter-model Pearson correlations exceeding 0.9978 suggest that the public baseline models are virtually identical in their functional approximation of the synthetic dataset.

## **Architectural Diversity Across Candidate Configurations**

To shatter the 0.9978 Pearson correlation without sacrificing individual ROC-AUC, the configurations must exploit fundamentally different hypothesis spaces. It is critical to differentiate a genuinely distinct learning procedure from a superficial variation in random seed. The following section defines a targeted configuration search that explores meaningful structural differences while preserving the baseline capability required to clear the 0.9610 OOF ROC-AUC threshold.

### **Ranked Configuration Strategies**

| Candidate                | Structural Distinction                                                                                                     | Initial Parameter Ranges                                                                                             | Expected Benefit                                                                                                                          | Main Risk                                                                                                                         | Priority |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- | :------- |
| **LightGBM Extra-Trees** | Randomized threshold split finding (extra\_trees=true). Replaces the standard exact greedy optimal search over histograms. | num\_leaves: \[127, 511\], extra\_trees: true, path\_smooth: \[2.0, 10.0\], feature\_fraction: \[0.5, 0.7\].         | Fundamentally alters partition boundaries; presents the highest mathematical likelihood of significantly reducing prediction correlation. | Potential loss of standalone ROC-AUC due to underfitting the 700,000-row synthetic dataset if capacity is insufficient.           | 1        |
| **CatBoost Symmetric**   | Oblivious trees with aggressive score perturbation (SymmetricTree).                                                        | depth: \[7, 9\], random\_strength: \[5, 20\], bagging\_temperature: \[1.0, 3.0\], simple\_ctr: FloatTargetMeanValue. | Enforces strict global interaction rules, maintaining a highly regularized functional form distinct from localized leaf-wise splits.      | Hardware memory constraints and high execution runtime on GPU for complex combination CTRs.                                       | 2        |
| **XGBoost Depth-wise**   | Histogram backend with highly restricted per-node feature sampling (colsample\_bynode).                                    | max\_depth: \[7, 12\], colsample\_bynode: \[0.3, 0.6\], enable\_categorical: True, tree\_method: "hist".             | Forces deep trees to utilize weak secondary features by systematically withholding dominant primary features at the node level.           | Severe overfitting if max\_depth is set too high without a correspondingly high min\_child\_weight to control leaf variance.      | 3        |
| **LightGBM Leaf-wise**   | Unrestricted, highly localized best-first growth targeting maximum loss reduction.                                         | num\_leaves: \[63, 127\], min\_data\_in\_leaf: \[100, 500\], feature\_fraction\_bynode: \[0.6, 0.9\].                | Generates highly specific, asymmetric partitions capable of mapping complex outlier sub-manifolds in the synthetic data.                  | Over-regularization converging to the same global splits as the depth-wise and symmetric configurations if leaves are restricted. | 4        |

### **Candidate A — CatBoost: Symmetric-Tree Architecture**

The primary defensible configuration for CatBoost relies on its native architectural strengths: grow\_policy='SymmetricTree', depth=8, random\_strength=10, bagging\_temperature=2.0, and simple\_ctr='FloatTargetMeanValue' executed on GPU hardware.  
Meaningful architectural diversity in CatBoost is generated by the symmetric tree constraints. Symmetric trees apply the exact same selected feature and optimal threshold across the entirety of a depth level. This structural mechanism fundamentally captures global, high-order interactions and is highly resistant to localized noise. By significantly elevating the random\_strength parameter, the model is mathematically forced to select sub-optimal splits early in the tree construction. Because the tree is symmetric, a perturbed split at depth one or two completely alters the downstream tree structure for every subsequent node, creating a wildly different final decision boundary compared to a model with zero random strength. Furthermore, utilizing Bayesian bootstrapping via bagging\_temperature ensures the training weights assigned to the synthetic rows vary aggressively between iterations.  
The smallest useful hyperparameter search involves fixing the categorical processing and varying depth \[6, 8, 10\] against random\_strength \[1, 10, 20\]. The required early-stopping criterion utilizes the validation dataset to identify the iteration with the optimal ROC-AUC, terminating training if no improvement is observed after 150 iterations. The experimental result required to justify retaining this model is a Spearman rank correlation with LightGBM dropping below 0.985 while maintaining a standalone OOF ROC-AUC of at least 0.9610.

### **Candidate B — LightGBM: Leaf-Wise Growth**

The initial defensible configuration for LightGBM focuses on maximizing the utility of its asymmetric partitioning: num\_leaves=127, min\_data\_in\_leaf=150, path\_smooth=5.0, and feature\_fraction\_bynode=0.7.  
LightGBM's leaf-wise growth policy aggressively targets the specific terminal node that yields the highest absolute loss reduction, creating deeply nested, asymmetric inference paths. To prevent this configuration from converging on the exact same splits as XGBoost, the introduction of feature\_fraction\_bynode is paramount. By evaluating a restricted, randomized subset of features at every single node—rather than once per tree—the model is physically prevented from relying exclusively on a single dominant feature to guide the primary splits. Consequently, path\_smooth is deployed to act as a Bayesian prior, preventing the highly specific, deep leaves from producing overconfident probability estimates that would miscalibrate the ROC-AUC ranking.  
The hyperparameter search should prioritize feature\_fraction\_bynode \[0.5, 0.7, 0.9\] and num\_leaves \[63, 127, 255\]. The stopping criterion is standard early stopping on the validation ROC-AUC after 100 rounds without improvement. This model represents the likely ceiling of individual performance; therefore, the retention criterion is strictly achieving a standalone OOF ROC-AUC exceeding 0.9615, establishing the foundational anchor for the final ensemble.

### **Candidate C — XGBoost: Depth-Wise Growth and Regularized Splits**

The targeted configuration for XGBoost requires activating its modern histogram capabilities: tree\_method="hist", enable\_categorical=True, max\_depth=9, colsample\_bynode=0.4, and subsample=0.85.  
While depth-wise growth is inherently more balanced and symmetric than leaf-wise growth, the true driver of structural diversity in this configuration is aggressive node-level sampling via colsample\_bynode. By severely restricting the features available at the exact moment of split evaluation, XGBoost is forced to construct parallel paths of inference utilizing secondary and tertiary features that LightGBM and CatBoost simply discard in favor of primary predictive variables. This ensures the model learns a distinct feature hierarchy.  
The essential hyperparameter search involves balancing max\_depth \[7, 9, 11\] against colsample\_bynode \[0.3, 0.5, 0.7\]. Depth must increase as node sampling decreases to allow the model sufficient capacity to combine weak features into strong predictors. The early stopping criterion relies on observing the validation ROC-AUC, halting at 100 iterations without gain. The model should be retained only if its pairwise disagreement rate on the hardest 5% of validation samples differs significantly from Candidate B, proving it has learned a distinct error topology.

### **Candidate D — LightGBM Extra-Trees**

The configuration for the Extra-Trees architecture fundamentally alters the optimization mathematics of the framework: extra\_trees=true, num\_leaves=511, min\_data\_in\_leaf=20, and feature\_fraction=0.6.  
Extremely Randomized Trees abandon the exact calculation of the maximum split gain over the entire feature histogram. By selecting random thresholds for the evaluated features, the decision boundaries become inherently stochastic. The explicit trade-off here involves a massive reduction in variance and split-search flexibility at the cost of significantly higher bias. To compensate for the reduced precision of random splits on a 700,000-row synthetic dataset, the model capacity, dictated by num\_leaves, must be substantially increased, while min\_data\_in\_leaf is relaxed. The expectation is an error profile that is highly orthogonal to exact-greedy models, producing complementary errors suitable for ensembling.  
The hyperparameter search is isolated to num\_leaves \[255, 511, 1024\] and feature\_fraction \[0.4, 0.6, 0.8\]. Early stopping is configured for 150 rounds. This model is not expected to achieve the absolute highest standalone AUC. Therefore, it is retained solely if the fixed-weight ensemble of Candidate B and Candidate D exceeds the AUC of Candidate B alone by a statistically significant margin, directly proving the utility of the randomized threshold architecture.

## **Maximizing Useful Error Diversity**

The optimization of a multi-model ensemble relies entirely on the extraction of complementary predictions without sacrificing individual model discrimination. A low Pearson correlation is not synonymous with useful diversity. A model can produce wildly uncorrelated probability estimates simply by injecting poorly calibrated noise into its predictions; this effectively lowers linear correlation but destroys ranking utility and severely damages the ensemble ROC-AUC37.

### **Defining Useful Error Diversity Correctly**

Useful error diversity exists strictly when two models make complementary ranking errors without degrading the overall statistical distribution of the true positive rates. Because ROC-AUC measures the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance, diversity is only mathematically valuable if Model B successfully ranks difficult positive-negative pairs that Model A incorrectly inverses, and vice versa.  
The diagnostic metrics required to evaluate this property include:

> 1. **Individual OOF ROC-AUC:** The foundational baseline. Models failing to approach the 0.9610 threshold are rejected unless their incremental ensemble contribution is exceptionally high.
> 2. **Pearson Correlation:** Measures the linear relationship between predicted probabilities. It is highly sensitive to the calibration of the raw probabilities. The aspiration of reaching \$\\leq 0.965\$ serves as a diagnostic warning; if correlations remain \$\> 0.995\$, the models are functionally identical.
> 3. **Spearman Rank Correlation:** The most critical inter-model evaluation metric. Because ROC-AUC is a purely rank-based metric, Spearman correlation isolates structural similarity in the models' learned decision functions, ignoring probability calibration artifacts entirely.
> 4. **Pairwise Disagreement Rate on Margin (PDR-M):** This metric isolates the OOF predictions where the true label is 1 but the predicted probability resides within the highly uncertain margin (e.g., between 0.3 and 0.7). It calculates the percentage of these specific margin instances where Model A and Model B rank the instance differently relative to their respective global thresholds.
> 5. **Ensemble ROC-AUC:** The ultimate arbiter of value. Evaluated using controlled fixed-weight combinations on out-of-sample holdouts to measure the true incremental contribution of a candidate model.

### **Investigating Sources of Diversity**

To generate the necessary error diversity, several controlled interventions must be implemented and compared.  
The most potent mechanism is manipulating the tree structure itself. Transitioning from exact greedy evaluations to randomized-threshold growth (extra\_trees=true) fundamentally alters the gradient descent path of the algorithm. This carries a high risk of degrading the standalone AUC if the data manifold is extremely simple, but it represents the minimum necessary experiment to induce structural orthogonality.  
Feature availability and node-level sampling represent the second source of diversity. Transitioning from traditional row-level bagging to strict node-level feature subsampling (colsample\_bynode) forces the utilization of surrogate splits. By withholding the dominant features at the point of node creation, the model must map the target utilizing the remaining weak features, creating a diverse functional representation.  
The third major source of diversity lies in the categorical feature representation. Shifting from generic integer-encoded ordinal representations to target-encoded permutation spaces (CatBoost's ordered CTRs) completely transforms the numerical space the tree algorithm navigates, ensuring a different split hierarchy.

### **Prioritizing Conditional Error Diversity**

To rigorously measure conditional error diversity, the OOF predictions must be analyzed across distinct categorical strata to identify subgroup-specific model strengths. The assumption that a specific subgroup is inherently difficult must be suspended until empirically measured.  
The methodology requires isolating specific combinatorial cohorts based on the actual column distributions—for example, comparing passengers where Customer Type \= "Disloyal" and Class \= "Eco" against passengers in "Business" class with high Flight Distance. By calculating cohort-specific ROC-AUC metrics for CatBoost and LightGBM independently, the analysis reveals conditional structure. If CatBoost achieves an AUC of 0.9400 on the "Disloyal Eco" cohort while LightGBM achieves 0.9300, but LightGBM significantly dominates the "Business" cohort, the differences arise from useful learned structure rather than stochastic noise. This empirical finding explicitly justifies rank-averaging or probability blending, as the models have definitively learned specialized representations for different sub-manifolds of the feature space. Validation labels must never be used to construct training-time features or to inform model-routing decisions, preserving strict leakage safety.

### **Direct Ensemble Value Testing and Leakage-Safe Selection**

The evaluation of fixed-weight combinations of the strongest models must be conducted using a leakage-safe protocol. For a small set of models, the combined prediction is defined as a generalized arithmetic blend: \$\\sum w\_i p\_i(x)\$, where \$w\_i \\geq 0\$ and \$\\sum w\_i \= 1\$.  
To prevent selection bias and avoid overfitting the blend weights to the OOF set, a strict validation procedure is required. The OOF predictions must be partitioned; for example, utilizing the predictions from folds 1 through 3 exclusively for weight optimization, and reserving the predictions from folds 4 and 5 strictly for confirmatory evaluation. A coarse, controlled grid search or a bounded optimization algorithm (such as Nelder-Mead) is applied to the optimization subset. The resulting weights are then evaluated on the reserved confirmation subset. If the ensemble ROC-AUC on the confirmation subset demonstrates a statistically significant improvement over the single strongest model, the weights are considered stable and the ensemble is validated. Utilizing linear stacking models or unrestricted optimization procedures directly on the entire OOF matrix introduces severe target leakage and is explicitly prohibited. Furthermore, the analysis must evaluate whether probability averaging or rank averaging is more robust; rank averaging transforms the raw probabilities into percentiles before blending, which frequently stabilizes ensembles composed of poorly calibrated models.

## **Categorical Ingestion and Model-Specific Representations**

The handling of categorical features represents the most significant opportunity to generate disparate decision boundaries on tabular data. Forcing all GBDT models to ingest identical one-hot or generically label-encoded variables destroys the unique inductive biases inherent in their native categorical handlers, homogenizing the resultant trees.  
An inspection of the synthetic dataset is required to delineate truly categorical columns (e.g., Gender, Customer Type, Type of Travel, Class) from discrete numerical columns or continuous features2.

### **Model-Specific Ingestion Strategy**

| Model                    | Recommended Representation                                        | Primary Parameters                                                    | Diversity Hypothesis                                                                                                                                                          | Main Leakage or Overfitting Risk                                                                                         |
| :----------------------- | :---------------------------------------------------------------- | :-------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **CatBoost**             | Native ordered categorical target statistics (CTR).               | simple\_ctr='FloatTargetMeanValue', bagging\_temperature=1.0          | Target means calculated sequentially via artificial time permutations prevent look-ahead bias and construct highly continuous numeric proxies distinct from partition splits. | Overfitting if CtrBorderCount is set too high for rare or synthetic-hallucinated categories.                             |
| **LightGBM**             | Native Fisher optimal categorical splits.                         | max\_cat\_threshold=32, cat\_smooth=20.0, cat\_l2=10.0                | Grouping high-cardinality categories into dynamic bins at each split allows for complex, non-linear categorical interaction mapping.                                          | Setting cat\_smooth too low allows the model to isolate and memorize rare synthetic identifiers generated by the CTGAN.  |
| **XGBoost**              | Native histogram-based categorical splits.                        | enable\_categorical=True, tree\_method="hist", max\_cat\_to\_onehot=4 | Distinct from LightGBM's Laplacian smoothing, XGBoost treats categoricals natively but relies exclusively on min\_child\_weight for structural regularization.                | Unseen categories in the test set causing inference path failures if synthetic data schema drifts.                       |
| **HistGradientBoosting** | Native categorical support with explicit interaction constraints. | categorical\_features="from\_dtype", interaction\_cst defined sets.   | Blocking specific categorical features from interacting with dominant numerical features forces orthogonal predictive paths.                                                  | Requires domain knowledge to define interaction groups; risk of underfitting if key predictive interactions are blocked. |

### **Prioritized Categorical Ablation Plan**

The ablation plan systematically isolates the effect of categorical representations on model diversity.

> 1. **Baseline Generation:** Train all candidate models using a single, unified integer label encoding, treating the categorical variables as continuous numericals. This configuration establishes the high-correlation baseline.
> 2. **Native Activation:** Transition CatBoost, LightGBM, and XGBoost to their respective native categorical implementations as defined in the ingestion table. Record changes in OOF ROC-AUC and the reduction in Spearman rank correlation.
> 3. **Target Statistic Perturbation (CatBoost):** Execute an ablation on CatBoost by switching the CTR configuration from the default Borders or Buckets to FloatTargetMeanValue (requiring GPU execution). This fundamentally alters the numeric mapping of the categories.
> 4. **Category Smoothing Escalation (LightGBM):** Escalate the cat\_smooth parameter from the default 10.0 to 50.0. This aggressively penalizes splits that attempt to isolate very small categorical subsets, forcing the tree to rely on broader, generalizable passenger characteristics.

### **Investigating Category Distribution Shift**

Because the S6E10 dataset is synthetic, distribution shift is a paramount concern. Generative models like CTGANs occasionally hallucinate rare category combinations or skew marginal distributions between the training and test sets4. To mitigate this risk, all target-derived categorical features must strictly utilize a fold-safe procedure, such as CatBoost's artificial time permutations, to prevent validation leakage. Furthermore, if any unsupervised preprocessing, such as global frequency encoding, is justified to induce model diversity, it must be fit on the combined synthetic training and unlabeled synthetic test populations simultaneously. This ensures the frequency statistics are globally consistent across the evaluation boundary, distinguishing unsupervised representation mapping from prohibited test-label leakage. Original-data target statistics must never be automatically transferred to the synthetic population, as the conditional joint distributions likely differ.

## **Ranked Experiment Queue**

The computational budget must be meticulously allocated to rule in or rule out the strongest hypotheses with the smallest possible experiments, adhering to a compute-aware methodology.

| Priority | Experiment ID      | Model    | Change                                                        | Main Hypothesis                                                                                                                                    | Cost   | Success Criterion                                                                                              |
| :------- | :----------------- | :------- | :------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------- | :----- | :------------------------------------------------------------------------------------------------------------- |
| 1        | EXP\_01\_LGB\_XTR  | LightGBM | Enable extra\_trees=true, increase num\_leaves to 511\.       | Randomized thresholds will significantly drop Spearman rank correlation against exact-greedy baselines by altering the mathematical gradient path. | Low    | Spearman correlation \$\< 0.985\$ with baseline LightGBM; standalone OOF ROC-AUC remains \$\\geq 0.9600\$.     |
| 2        | EXP\_02\_XGB\_NODE | XGBoost  | Implement colsample\_bynode=0.4 instead of colsample\_bytree. | Node-level subsampling forces distinctly different tree architectures by withholding dominant features during split evaluations.                   | Medium | Standalone OOF ROC-AUC \$\\geq 0.9605\$; high pairwise margin disagreement compared to LightGBM baselines.     |
| 3        | EXP\_03\_CAT\_CTR  | CatBoost | Change CTR from default to FloatTargetMeanValue (GPU).        | Altering the categorical target calculation creates a distinctly different, continuous numerical feature space mapping.                            | High   | Standalone OOF ROC-AUC \$\\geq 0.9610\$; measurable improvement in subgroup-specific ROC-AUC.                  |
| 4        | EXP\_04\_LGB\_SMTH | LightGBM | Increase path\_smooth=10.0, cat\_smooth=40.0.                 | Heavy structural regularization will physically prevent the model from converging on the exact same specific synthetic data anomalies.             | Medium | AUC remains highly stable (\$\\geq 0.9605\$) while simultaneously lowering pairwise correlation with CatBoost. |

The research plan is organized into three tiers. Tier 1 executes fast diagnostic experiments (EXP\_01\_LGB\_XTR and EXP\_02\_XGB\_NODE) utilizing a single 20% holdout fold to rapidly compute the Spearman correlation between the generated predictions. If the combination of randomized thresholds and node-level sampling fails to drop the correlation below 0.990, it conclusively demonstrates that the predictive signal is fundamentally low-dimensional, and broad hyperparameter tuning will yield diminishing returns.  
Tier 2 involves targeted optimization. Surviving candidates from Tier 1 proceed to full 5-fold cross-validation. OOF probability predictions are extracted to execute the leakage-safe ensemble-weight selection procedure.  
Tier 3 encompasses expensive experiments, specifically deploying Scikit-Learn's HistGradientBoostingClassifier with meticulously defined interaction\_cst arrays. This is executed only if Tiers 1 and 2 demonstrate a credible mathematical opportunity for ensemble improvement that justifies the high computational cost of isolating feature interactions.

## **Implementation Blueprint**

The following Python-oriented pseudocode dictates the strict, leakage-safe pipeline for fold generation, model training, OOF artifact generation, and fixed-weight ensemble evaluation.

Python  
import numpy as np  
import pandas as pd  
from sklearn.model\_selection import StratifiedKFold  
from sklearn.metrics import roc\_auc\_score  
from scipy.stats import spearmanr  
from scipy.optimize import minimize  
import lightgbm as lgb  
import catboost as cb  
import xgboost as xgb

\# 1\. Immutable Fold Generation ensuring Leakage-Safe Boundaries  
def generate\_immutable\_folds(df, target\_col, n\_splits=5, seed=42):  
skf \= StratifiedKFold(n\_splits=n\_splits, shuffle=True, random\_state=seed)  
folds \= list(skf.split(df, df\[target\_col\]))  
\# Artifact generation for reproducibility across all experiment scripts  
return folds

\# 2\. Fold-Safe Model Training and OOF Artifact Generation  
def train\_lightgbm\_extra\_trees(df, target\_col, categorical\_cols, folds):  
oof\_preds \= np.zeros(len(df))

    \# Candidate D: LightGBM Extra-Trees configuration
    params \= {
        'objective': 'binary',
        'metric': 'auc',
        'boosting\_type': 'gbdt',
        'extra\_trees': True,          \# Initiates randomized threshold split finding
        'num\_leaves': 511,            \# High capacity required to offset random splits
        'path\_smooth': 5.0,           \# Bayesian leaf output smoothing
        'feature\_fraction': 0.6,
        'cat\_smooth': 20.0,           \# Laplacian categorical smoothing
        'random\_state': 42,
        'n\_jobs': \-1
    }

    for fold, (train\_idx, val\_idx) in enumerate(folds):
        X\_train, y\_train \= df.iloc\[train\_idx\].drop(columns=\[target\_col\]), df.iloc\[train\_idx\]\[target\_col\]
        X\_val, y\_val \= df.iloc\[val\_idx\].drop(columns=\[target\_col\]), df.iloc\[val\_idx\]\[target\_col\]

        train\_set \= lgb.Dataset(X\_train, y\_train, categorical\_feature=categorical\_cols)
        val\_set \= lgb.Dataset(X\_val, y\_val, categorical\_feature=categorical\_cols, reference=train\_set)

        \# Train with stringent early stopping to prevent overfitting
        model \= lgb.train(
            params,
            train\_set,
            valid\_sets=\[train\_set, val\_set\],
            num\_boost\_round=4000,
            callbacks=\[lgb.early\_stopping(stopping\_rounds=150, verbose=False)\]
        )

        \# Construct exact OOF prediction array
        oof\_preds\[val\_idx\] \= model.predict(X\_val, num\_iteration=model.best\_iteration)

    return oof\_preds

\# 3\. Correlation Diagnostics  
def evaluate\_diversity(preds\_a, preds\_b, y\_true):  
pearson\_corr \= np.corrcoef(preds\_a, preds\_b)\[0, 1\]  
spearman\_corr, \_ \= spearmanr(preds\_a, preds\_b)  
print(f"Pearson Correlation: {pearson\_corr:.5f}")  
print(f"Spearman Rank Correlation: {spearman\_corr:.5f}")  
return spearman\_corr

\# 4\. Leakage-Safe Fixed-Weight Ensemble Optimization  
def optimize\_and\_validate\_ensemble(oof\_dict, y\_true, splits):  
\# Segregate OOF into optimization (Folds 1-3) and validation (Folds 4-5) limits selection bias  
opt\_idx \= np.concatenate(\[splits\[0\]\[1\], splits\[1\]\[1\], splits\[2\]\[1\]\])  
val\_idx \= np.concatenate(\[splits\[3\]\[1\], splits\[4\]\[1\]\])

    model\_names \= list(oof\_dict.keys())

    def loss\_function(weights):
        \# Normalize weights to sum to 1
        weights \= weights / np.sum(weights)
        blend\_opt \= np.zeros(len(opt\_idx))
        for i, name in enumerate(model\_names):
            blend\_opt \+= weights\[i\] \* oof\_dict\[name\]\[opt\_idx\]
        return \-roc\_auc\_score(y\_true.iloc\[opt\_idx\], blend\_opt)

    \# Execute bounded optimization on isolated subset
    init\_weights \= np.ones(len(model\_names)) / len(model\_names)
    bounds \= \[(0, 1\) for \_ in range(len(model\_names))\]
    result \= minimize(loss\_function, init\_weights, bounds=bounds, method='Nelder-Mead')

    optimal\_weights \= result.x / np.sum(result.x)

    \# Validate strictly on reserved hold-out folds
    blend\_val \= np.zeros(len(val\_idx))
    for i, name in enumerate(model\_names):
        blend\_val \+= optimal\_weights\[i\] \* oof\_dict\[name\]\[val\_idx\]

    val\_auc \= roc\_auc\_score(y\_true.iloc\[val\_idx\], blend\_val)
    print(f"Validated Hold-Out Ensemble AUC: {val\_auc:.5f}")

    return optimal\_weights, val\_auc

## **Final Recommendation**

Given the constraints of the Kaggle S6E10 dataset and the current baseline stagnation, the single most critical experiment to execute immediately is **EXP\_01\_LGB\_XTR** (Candidate D).  
The exact configuration must deploy LightGBM utilizing extra\_trees=true, escalating num\_leaves to 511 to counteract the high bias of randomized splits, restricting feature\_fraction to 0.6 to enforce variable diversity, and utilizing path\_smooth=5.0 to stabilize the terminal nodes.  
The rationale for prioritizing this specific configuration is rooted in the mathematical mechanics of the algorithm. The current 0.9978 Pearson correlation is a direct symptom of baseline models traversing the exact same greedy optimization path over the synthetic dataset's inherently low-dimensional manifold. By invoking the extra\_trees parameter, the split-finding algorithm randomly proposes a threshold for each feature rather than exhaustively scanning the histogram for the mathematically optimal threshold. This structural perturbation guarantees a distinct formulation of the decision boundary with minimal computational overhead.  
The success of this experiment is determined by evaluating the diagnostic metrics. Success is achieved if the Spearman rank correlation between the extra\_trees model predictions and the baseline LightGBM model predictions falls below the 0.985 threshold, while the standalone OOF ROC-AUC of the extra\_trees model remains above 0.9600.  
The definitive decision rule mandates that the model is _retained_ only if the fixed-weight combination of this model with the baseline LightGBM model improves the combined OOF ROC-AUC by a margin greater than 0.0002 when evaluated strictly on the held-out validation subset using the leakage-safe optimization protocol. The model must be _rejected_ if the standalone AUC drops below 0.9580, or if the Spearman correlation remains above 0.995, which would indicate that the data structure is so elementary that even randomized thresholds easily and unavoidably discover the identical primary splits.  
Following the outcome of EXP\_01\_LGB\_XTR, the next step is determined entirely by the result. If the model is retained, the research queue progresses immediately to EXP\_02\_XGB\_NODE to introduce a third, uncorrelated architectural perspective utilizing node-level subsampling. If the model is rejected due to persistently high correlation, it conclusively proves that structural variations cannot overcome the synthetic manifold collapse. In that scenario, the research focus must instantly pivot away from tree mechanics and toward Tier 2, heavily manipulating the categorical processing algorithms (EXP\_03\_CAT\_CTR) to alter the underlying numeric representations before the decision trees can even attempt to split on them.

#### **Works cited**

> 1. Predicting Airline Satisfaction | Kaggle, [https\://www\.kaggle.com/competitions/playground-series-s6e10/code](https://www.kaggle.com/competitions/playground-series-s6e10/code)
> 2. S6E10 | Airline Satisfaction: EDA, ML & XAI \- Kaggle, [https\://www\.kaggle.com/code/mohankrishnathalla/s6e10-airline-satisfaction-eda-ml-xai/log](https://www.kaggle.com/code/mohankrishnathalla/s6e10-airline-satisfaction-eda-ml-xai/log)
> 3. Predicting Airline Satisfaction | Kaggle, [https\://www\.kaggle.com/competitions/playground-series-s6e10](https://www.kaggle.com/competitions/playground-series-s6e10)
> 4. Synthetic Tabular Data Generator \- Medium, [https\://medium.com/@ashish28082002.ak/tabular-synthetic-data-generator-using-conditional-gans-d98fcd974148](https://medium.com/@ashish28082002.ak/tabular-synthetic-data-generator-using-conditional-gans-d98fcd974148)
> 5. Modeling Tabular Data using Conditional GAN \- arXiv, [https\://arxiv.org/pdf/1907.00503](https://arxiv.org/pdf/1907.00503)
> 6. CTGAN for Credit Analysis Synthetic Data \- Medium, [https\://rayislam.medium.com/ctgan-for-credit-analysis-synthetic-data-bbb1d04ff6f3](https://rayislam.medium.com/ctgan-for-credit-analysis-synthetic-data-bbb1d04ff6f3)
> 7. Rigorous Experimental Analysis of Tabular Data Generated using, [https\://thesai.org/Downloads/Volume15No4/Paper\_125-Rigorous\_Experimental\_Analysis\_of\_Tabular\_Data\_Generated.pdf](https://thesai.org/Downloads/Volume15No4/Paper_125-Rigorous_Experimental_Analysis_of_Tabular_Data_Generated.pdf)
> 8. 12 GBDT Tricks to Squeeze the Last 5% AUC | by Thinking Loop, [https\://medium.com/@ThinkingLoop/12-gbdt-tricks-to-squeeze-the-last-5-auc-185f4084e9ac](https://medium.com/@ThinkingLoop/12-gbdt-tricks-to-squeeze-the-last-5-auc-185f4084e9ac)
> 9. Parameter tuning \- CatBoost, [https\://catboost.ai/docs/en/concepts/parameter-tuning](https://catboost.ai/docs/en/concepts/parameter-tuning)
> 10. Parameters — LightGBM 4.7.0.99 documentation, [https\://lightgbm.readthedocs.io/en/latest/Parameters.html](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
> 11. HistGradientBoostingClassifier — scikit-learn 1.9.1 documentation, [https\://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
> 12. catboost.train, [https\://catboost.ai/docs/en/concepts/r-reference\_catboost-train](https://catboost.ai/docs/en/concepts/r-reference_catboost-train)
> 13. Fast Gradient Boosting with CatBoost \- KDnuggets, [https\://www\.kdnuggets.com/2020/10/fast-gradient-boosting-catboost.html](https://www.kdnuggets.com/2020/10/fast-gradient-boosting-catboost.html)
> 14. The Gradient Boosters V: CatBoost \- Deep & Shallow, [https\://deep-and-shallow.com/2020/02/29/the-gradient-boosters-v-catboost/](https://deep-and-shallow.com/2020/02/29/the-gradient-boosters-v-catboost/)
> 15. Common parameters \- CatBoost, [https\://catboost.ai/docs/en/references/training-parameters/common](https://catboost.ai/docs/en/references/training-parameters/common)
> 16. CTR settings \- CatBoost, [https\://catboost.ai/docs/en/references/training-parameters/ctr](https://catboost.ai/docs/en/references/training-parameters/ctr)
> 17. quantize \- CatBoost, [https\://catboost.ai/docs/en/concepts/python-reference\_pool\_quantized](https://catboost.ai/docs/en/concepts/python-reference_pool_quantized)
> 18. Categorical features parameters in CatBoost \- Medium, [https\://medium.com/data-science/categorical-features-parameters-in-catboost-4ebd1326bee5](https://medium.com/data-science/categorical-features-parameters-in-catboost-4ebd1326bee5)
> 19. CatBoost: A Solution for Building Model with Categorical Data, [https\://www\.analyticsvidhya.com/blog/2023/07/catboost-building-model-with-categorical-data/](https://www.analyticsvidhya.com/blog/2023/07/catboost-building-model-with-categorical-data/)
> 20. CatBoost hyperparameters \- Amazon SageMaker AI, [https\://docs.aws.amazon.com/sagemaker/latest/dg/catboost-hyperparameters.html](https://docs.aws.amazon.com/sagemaker/latest/dg/catboost-hyperparameters.html)
> 21. What are bagging temperature and random strength? \#373 \- GitHub, [https\://github.com/catboost/catboost/issues/373](https://github.com/catboost/catboost/issues/373)
> 22. Parameters Tuning — LightGBM 4.7.0.99 documentation, [https\://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
> 23. PDF \- LightGBM's documentation\!, [https\://lightgbm.readthedocs.io/\_/downloads/en/latest/pdf/](https://lightgbm.readthedocs.io/_/downloads/en/latest/pdf/)
> 24. 参数— LightGBM 4.6.0 文档, [https\://lightgbm.cn/en/stable/Parameters.html](https://lightgbm.cn/en/stable/Parameters.html)
> 25. Release 1.8.0 Fabio Sigrist \- Documentation of GPBoost, [https\://gpboost.readthedocs.io/\_/downloads/en/latest/pdf/](https://gpboost.readthedocs.io/_/downloads/en/latest/pdf/)
> 26. xgboost\_distribution.XGBDistribution \- xgboost-distribution, [https\://xgboost-distribution.readthedocs.io/en/latest/api/xgboost\_distribution.XGBDistribution.html](https://xgboost-distribution.readthedocs.io/en/latest/api/xgboost_distribution.XGBDistribution.html)
> 27. snowflake.ml.modeling.xgboost.XGBRegressor, [https\://docs.snowflake.com/fr/developer-guide/snowpark-ml/reference/1.0.9/api/modeling/snowflake.ml.modeling.xgboost.XGBRegressor](https://docs.snowflake.com/fr/developer-guide/snowpark-ml/reference/1.0.9/api/modeling/snowflake.ml.modeling.xgboost.XGBRegressor)
> 28. Python API Reference — xgboost 3.4.2 documentation, [https\://xgboost.readthedocs.io/en/stable/python/python\_api.html](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
> 29. xgboost/NEWS.md at master · dmlc/xgboost \- GitHub, [https\://github.com/dmlc/xgboost/blob/master/NEWS.md](https://github.com/dmlc/xgboost/blob/master/NEWS.md)
> 30. Python API Reference — xgboost 1.7.6 documentation, [https\://xgboost.readthedocs.io/en/release\_1.7.0/python/python\_api.html](https://xgboost.readthedocs.io/en/release_1.7.0/python/python_api.html)
> 31. XGBoost hyperparameters \- Amazon SageMaker AI, [https\://docs.aws.amazon.com/sagemaker/latest/dg/xgboost\_hyperparameters.html](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html)
> 32. XGBoost Parameters — xgboost 3.4.2 documentation, [https\://xgboost.readthedocs.io/en/stable/parameter.html](https://xgboost.readthedocs.io/en/stable/parameter.html)
> 33. XGBoost Parameters — xgboost 0.90 documentation, [https\://xgboost.readthedocs.io/en/release\_0.90/parameter.html](https://xgboost.readthedocs.io/en/release_0.90/parameter.html)
> 34. Post-tuning the decision threshold for cost-sensitive learning, [https\://scikit-learn.org/stable/auto\_examples/model\_selection/plot\_cost\_sensitive\_learning.html](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html)
> 35. Release Highlights for scikit-learn 1.2, [https\://scikit-learn.org/stable/auto\_examples/release\_highlights/plot\_release\_highlights\_1\_2\_0.html](https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_2_0.html)
> 36. 1.11. Ensembles: Gradient boosting, random forests, bagging, voting, [https\://scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html)
> 37. uncertainty in gradient boosting \- arXiv, [https\://arxiv.org/pdf/2006.10562](https://arxiv.org/pdf/2006.10562)
> 38. Regularized Neural Ensemblers \- arXiv, [https\://arxiv.org/pdf/2410.04520](https://arxiv.org/pdf/2410.04520)
> 39. A Bias–Variance Perspective on Tabular Data \- arXiv, [https\://arxiv.org/html/2512.05469v1](https://arxiv.org/html/2512.05469v1)
> 40. An Optimised Greedy-Weighted Ensemble Framework for Financial, [https\://arxiv.org/pdf/2603.18927](https://arxiv.org/pdf/2603.18927)
