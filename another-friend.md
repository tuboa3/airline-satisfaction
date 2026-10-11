# **Advanced GBDT Optimization and Noise-Robust Representation Engineering for Airline Passenger Satisfaction**

## **1\. Executive Verdict**

Closing the generalization gap from 0.96128 out-of-fold (OOF) ROC-AUC to the aspirational target of 0.96243+ requires shifting the gradient-boosting optimization paradigm from aggressive exploitation of local feature interactions to robust, noise-aware regularization. The comprehensive analysis of the existing ensemble baseline—comprising CatBoost, LightGBM, and XGBoost estimators—indicates that the predictive capacity under standard binary cross-entropy minimization has saturated. Pushing beyond the current performance threshold necessitates an architectural intervention targeting the underlying data generation assumptions, algorithmic diversity, and topological regularization.  
A forensic evaluation of the foundational hypotheses yields several critical determinations. First, the estimated 3.8% label ambiguity must be formally classified as stochastic target sampling (Bernoulli noise) rather than independent, symmetric label corruption1. Under a probabilistic generation mechanism, forced algorithmic corrections such as symmetric label smoothing or focal loss optimization risk exacerbating generalization error by forcing the estimators to memorize irreducible stochasticity2. Consequently, strategies that amplify the gradient contributions of "hard" examples are actively counterproductive. The most promising theoretical pathway to maximizing held-out ROC-AUC involves conservative target softening via cross-fitted student-teacher distillation, enforcing strict topological constraints (such as LightGBM's path\_smooth and extra\_trees), and manipulating categorical representation boundaries to induce orthogonal errors across the ensemble members5.  
The mathematical derivations confirm that any deployment of customized objective functions directly manipulating gradients and Hessians must be strictly governed by numerical stability bounds. Specifically, bounding gradient contributions requires the simultaneous enforcement of strictly positive Hessian floors to prevent the Newton-Raphson tree-growth approximation from failing6. The final strategic consensus dictates freezing the current feature baseline while executing a tightly controlled ablation sequence across the algorithmic arms, ensuring that variance in categorical representations yields complementary decision boundaries without degrading the standalone predictive power of the constituent models.

## **2\. Mathematical Analysis**

The mathematical foundation of the optimization strategy dictates how objective functions, label noise models, and regularization terms interact during the gradient boosting process. It is imperative to rigorously derive the proposed mechanisms to evaluate their applicability before subjecting them to empirical computation.

### **Noise Models and Label Transformations**

To interrogate the 3.8% label noise estimate, one must formally define the assumptions of symmetric label flipping versus stochastic target generation. Let $X$ represent the feature space and $Y\in \{0,1\}$ represent the binary target. Under a symmetric label-flip model with a uniform flip probability $\eta$, the clean conditional probability \$p(x) \= P(Y\_{\\text{clean}}=1 \\mid X=x)\$ is obscured, yielding the observed conditional probability \$q(x) \= P(Y\_{\\text{observed}}=1 \\mid X=x)\$. The transition is modeled as:

\$\$q(x) \= \\eta \\cdot P(Y\_{\\text{clean}}=0 \\mid X=x) \+ (1 \- \\eta) \\cdot P(Y\_{\\text{clean}}=1 \\mid X=x)\$\$

\$\$q(x) \= \\eta (1 \- p(x)) \+ (1 \- \\eta) p(x) \= \\eta \+ p(x) \- 2\\eta p(x) \\\\ q(x) \= \\eta \+ (1 \- 2\\eta)p(x)\$\$  
Assuming $\eta <0.5$, the true probability can theoretically be recovered via the inverse transformation:

$$p(x)=\frac{q(x)-\eta }{1-2\eta }$$

However, this analytical transformation poses severe identifiability limits. If the generative mechanism is not symmetric deterministic flipping but rather stochastic sampling from a true continuous probability distribution ${p}^{*}(X)$, identical or highly similar feature vectors will naturally exhibit contradictory labels. Applying the inverse transformation automatically assumes all disagreement is corruption. If the estimated $\eta =0.038$ is incorrect, the mapping will generate probabilities outside the $[0,1]$ bounds for confident regions, destroying the model's calibration. Furthermore, since ROC-AUC evaluates the ranking of predictions rather than their absolute calibration, mapping $q(x)$ to $p(x)$ via a monotonic linear transformation has mathematically zero impact on the ROC-AUC score.

### **Binary Log Loss vs. Label Smoothing**

The baseline objective is binary log loss (cross-entropy), defined for prediction ${p}_{i}$ and label ${y}_{i}$ as:

\$\$\\ell\_i \= \-y\_i \\log p\_i \- (1 \- y\_i) \\log(1 \- p\_i)\$\$  
Log loss exhibits an unbounded penalty for confident incorrect predictions; as ${p}_{i}\rightarrow 0$ for a true label of ${y}_{i}=1$, the gradient magnitude approaches infinity. This heavily penalizes the model when evaluating stochastic outliers, coercing the gradient-boosting algorithm to grow deep, specialized tree branches to minimize this extreme local loss.  
The proposed smoothing formulation, ${y}_{smooth}=y\cdot 0.924+0.038$, must be evaluated against the standard symmetric label smoothing equation:

$${y}_{smooth}=(1-\alpha )y+\frac{\alpha }{2}$$

Substituting $\alpha =0.076$ yields ${y}_{smooth}=(1-0.076)y+0.038=y\cdot 0.924+0.038$. This establishes that the proposed formula is mathematically identical to symmetric label smoothing with an $\alpha$ parameter of $0.076$9. This technique maps binary targets to $0.962$ and $0.038$, which mathematically bounds the maximum possible gradient an incorrectly labeled example can produce. However, label smoothing encodes the assumption that the true label distributions are uniformly uncertain across the entire feature space. It does not intelligently distinguish between genuinely ambiguous decision boundaries and regions of absolute certainty, making it a crude regularizer rather than a verified correction for a 3.8% noise rate.

### **Focal Loss Critique**

Binary focal loss is parameterized by a focusing parameter $\gamma$ and a balancing parameter ${\alpha }_{t}$:

\$\$\\operatorname{FL}(p\_t) \= \-\\alpha\_t(1 \- p\_t)^\\gamma \\log(p\_t)\$\$  
where ${p}_{t}$ is the predicted probability assigned to the observed class2. As ${p}_{t}\rightarrow 1$, the modulating factor $(1-{p}_{t}{)}^{\gamma }\rightarrow 0$, exponentially down-weighting the influence of easy, well-classified examples. Conversely, when ${p}_{t}\rightarrow 0$, the loss remains high, forcing the model to update heavily on difficult instances.  
In the presence of irreducible stochastic noise or corrupted labels, the "difficult" examples are predominantly the mislabeled or inherently ambiguous observations3. By emphasizing hard examples, focal loss amplifies the gradients of the suspected noise, actively worsening the generalization capability of the ensemble. Consequently, focal loss is theoretically contraindicated for this dataset, as it explicitly prevents the model from ignoring the 3.8% ambiguity.

### **Custom Objective Derivatives and Hessian Stability**

Gradient boosting frameworks approximate the objective function at iteration $t$ using a second-order Taylor expansion around the current prediction ${f}_{t-1}({x}_{i})$:

$${L}^{(t)}\approx \sum\limits_{i=1}^{n}\left[{{g}_{i}{f}_{t}({x}_{i})+\frac{1}{2}{h}_{i}{f}_{t}^{2}({x}_{i})}\right]+\Omega ({f}_{t})$$

where ${g}_{i}$ and ${h}_{i}$ are the first (gradient) and second (Hessian) derivatives of the loss with respect to the raw margin $z$6. For binary logistic regression, where $p=\sigma (z)=(1+{e}^{-z}{)}^{-1}$:

$$g=p-y$$

$$h=p(1-p)$$

During tree construction, the optimal weight ${w}_{j}$ for leaf $j$ is calculated via the Newton-Raphson step:

$${w}_{j}=-\frac{\sum\limits_{i\in {I}_{j}}^{}{g}_{i}}{\sum\limits_{i\in {I}_{j}}^{}{h}_{i}+\lambda }$$

The values of $g$ and $h$ directly dictate split gain and leaf regularization6. If a custom objective independently clips the gradient (e.g., to bound the influence of outliers) without maintaining the mathematically derived relationship with the Hessian, the Newton step is compromised. Furthermore, if $p\rightarrow 0$ or $p\rightarrow 1$, the Hessian $h\rightarrow 0$. In these regions, if $h$ reaches absolute zero or a negative value, the denominator is governed solely by the L2 regularization parameter $\lambda$, potentially causing instability, infinite leaf weights, or algorithmic divergence7. Any custom objective modification must enforce a strictly positive Hessian floor.

## **3\. Production-Ready Model Configurations**

To test the hypotheses regarding topological constraints and ensemble diversity, four distinct model arms are formalized. Each arm incorporates a baseline configuration, representing standard competitive optimization, and an alternative diversity/robustness configuration designed to induce complementary representations of the feature space5.

### **Arm 1: CatBoost GPU Oblivious Trees / CTR**

CatBoost relies on symmetric (oblivious) trees, meaning the same splitting criterion is applied across all nodes at a given depth. This architecture acts as a powerful regularizer and excels at extracting dense categorical combinations14.  
The configuration utilizes bootstrap\_type="Bayesian", which assigns random continuous weights to objects drawn from an exponential distribution. The bagging\_temperature parameter modulates the variance of this distribution; higher values exponentially increase the diversity of the trees13. On the GPU, the FloatTargetMeanValue target statistic replaces categorical values with the mean of the target computed sequentially over random permutations14.

| Parameter               | Baseline Dictionary | Diversity / Robustness Variant      |
| :---------------------- | :------------------ | :---------------------------------- |
| task\_type              | "GPU"               | "GPU"                               |
| loss\_function          | "Logloss"           | "Logloss"                           |
| eval\_metric            | "AUC"               | "AUC"                               |
| iterations              | 4500                | 6000                                |
| learning\_rate          | 0.025               | 0.015                               |
| depth                   | 7                   | 6 (Shallower for noise robustness)  |
| l2\_leaf\_reg           | 5.0                 | 12.0 (Conservative leaf values)     |
| bootstrap\_type         | "Bayesian"          | "Bayesian"                          |
| bagging\_temperature    | 1.0                 | 1.5 (Increases random weighting)    |
| max\_ctr\_complexity    | 2                   | 4 (Deeper categorical interactions) |
| simple\_ctr             | _Default_           | "FloatTargetMeanValue"              |
| combinations\_ctr       | _Default_           | "FloatTargetMeanValue"              |
| early\_stopping\_rounds | 200                 | 200                                 |

_Experiment Requirement:_ Train both variants on identically matched folds. The objective is to verify whether max\_ctr\_complexity=4 integrated with a high l2\_leaf\_reg creates a model that generalizes better against the 3.8% label noise without drastically increasing GPU VRAM exhaustion14.

### **Arm 2: LightGBM Leaf-Wise Growth**

LightGBM constructs trees by splitting the leaf with the maximum delta loss (leaf-wise growth). While this ensures rapid convergence, it is highly susceptible to fitting anomalies deep within the feature space12. Controlling this behavior requires strict regulation of num\_leaves and min\_child\_samples.  
The path\_smooth parameter acts as a topological regularizer by shrinking the output of a deep leaf toward its parent node, with stronger shrinkage applied when the leaf contains fewer samples5. This mathematically prevents deep, sample-poor leaves from adopting extreme logits in response to isolated noisy labels.

| Parameter               | Baseline Dictionary | Diversity / Robustness Variant      |
| :---------------------- | :------------------ | :---------------------------------- |
| objective               | "binary"            | "binary"                            |
| metric                  | "auc"               | "auc"                               |
| num\_leaves             | 127                 | 75 (Constrains global capacity)     |
| max\_depth              | \-1                 | 10                                  |
| learning\_rate          | 0.02                | 0.02                                |
| min\_child\_samples     | 50                  | 100 (Resists local noise isolation) |
| min\_child\_weight      | 0.001               | 0.1                                 |
| path\_smooth            | 0.0                 | 10.0 (Parent-node shrinkage)        |
| lambda\_l1 / lambda\_l2 | 0.1 / 1.0           | 1.0 / 5.0                           |
| feature\_fraction       | 0.80                | 0.65                                |
| cat\_smooth             | 10.0                | 25.0                                |

_Experiment Requirement:_ Validate whether path\_smooth=10.0 limits the asymptotic memorization of adversarial hold-out samples compared to the standard configuration5.

### **Arm 3: LightGBM Extra-Trees**

The extra\_trees=True parameter fundamentally alters the splitting algorithm. Instead of calculating optimal histogram cut-points for every feature, it selects a single, uniformly random split threshold per feature and evaluates only that threshold5. This injects massive variance into the decision boundaries.  
Because the splits are randomized, the individual tree is significantly weaker. Thus, the model requires a vastly increased capacity (num\_leaves=255) and lower feature\_fraction to ensure the ensemble converges on a strong approximation5. This randomization prevents the algorithm from finding the precise hyperplanes that isolate noise.

| Parameter           | Baseline Dictionary | Diversity / Robustness Variant       |
| :------------------ | :------------------ | :----------------------------------- |
| objective           | "binary"            | "binary"                             |
| metric              | "auc"               | "auc"                                |
| extra\_trees        | False               | True                                 |
| num\_leaves         | 127                 | 255                                  |
| learning\_rate      | 0.02                | 0.03                                 |
| min\_child\_samples | 50                  | 20                                   |
| feature\_fraction   | 0.80                | 0.50 (Orthogonal subspace selection) |
| bagging\_fraction   | 0.80                | 0.75                                 |
| path\_smooth        | 0.0                 | 5.0                                  |

_Experiment Requirement:_ Evaluate whether the randomized split thresholds sacrifice excessive standalone ROC-AUC (e.g., dropping below 0.959) or if the resulting predictions are sufficiently de-correlated to improve the meta-ensemble.

### **Arm 4: XGBoost Depth-Wise Growth / Psychometric Features**

XGBoost builds trees depth-wise (by default), ensuring all branches reach a symmetrical depth before expanding further. This naturally isolates psychometric features (the 0-5 categorical ratings) differently than leaf-wise growth. The gamma parameter (min\_split\_loss) acts as a strict penalty applied to the structure score of every node; splits are rejected if the gradient reduction does not exceed gamma6.

| Parameter           | Baseline Dictionary | Diversity / Robustness Variant        |
| :------------------ | :------------------ | :------------------------------------ |
| objective           | "binary:logistic"   | "binary:logistic"                     |
| eval\_metric        | "auc"               | "auc"                                 |
| tree\_method        | "hist"              | "hist"                                |
| enable\_categorical | True                | True                                  |
| max\_depth          | 8                   | 9 (Experimental range)                |
| learning\_rate      | 0.02                | 0.015                                 |
| min\_child\_weight  | 10                  | 20 (Hessian sum floor constraint)     |
| colsample\_bynode   | 1.0                 | 0.40 (Forces alternate feature paths) |
| subsample           | 0.80                | 0.80                                  |
| gamma               | 0.0                 | 1.5 (Strict loss reduction barrier)   |

_Experiment Requirement:_ Test whether a high gamma combined with a low colsample\_bynode creates robust sub-trees that effectively ignore random Bernoulli noise without sacrificing signal extraction6.

## **4\. Custom-Objective Implementations**

If topological regularizers (path\_smooth, gamma, extra\_trees) fail to mitigate the 3.8% ambiguity saturation, explicit robust gradients must be deployed. The implementations below interface with the frameworks via raw margins and enforce mathematically consistent Hessian behaviors.

### **LightGBM Custom Objective**

The LightGBM Python API requires the custom objective to accept preds and train\_data and return gradients and Hessians. The preds argument provides the raw, untransformed logits (the margin), mandating the use of a numerically stable sigmoid transformation prior to derivative calculation18.

Python  
import numpy as np  
import lightgbm as lgb  
from typing import Tuple

def stable\_sigmoid(z: np.ndarray) \-\> np.ndarray:  
"""Computes a numerically stable sigmoid to prevent overflow."""  
return np.where(z \>= 0,  
1.0 / (1.0 \+ np.exp(-z)),  
np.exp(z) / (1.0 \+ np.exp(z)))

def robust\_binary\_logloss\_lgbm(preds: np.ndarray, train\_data: lgb.Dataset) \-\> Tuple\[np.ndarray, np.ndarray\]:  
"""  
LightGBM Custom Objective: Robust Binary Cross-Entropy.  
Applies gradient clipping for noise resilience while maintaining  
a strictly positive Hessian floor for numerical stability.  
"""  
\# Extract ground truth labels  
labels \= train\_data.get\_label()

    \# Transform raw margins to probabilities
    p \= stable\_sigmoid(preds)

    \# Analytical derivatives of standard log loss
    grad \= p \- labels
    hess \= p \* (1.0 \- p)

    \# 1\. Gradient Clipping: Bound the extreme penalties caused by label noise
    clip\_threshold \= 0.85
    grad\_clipped \= np.clip(grad, \-clip\_threshold, clip\_threshold)

    \# 2\. Hessian Floor: Prevent zero-division in Newton-Raphson update step
    \# As p \-\> 0 or p \-\> 1, p\*(1-p) approaches 0\.
    hess\_floor \= 1e-4
    hess\_stable \= np.maximum(hess, hess\_floor)

    return grad\_clipped, hess\_stable

\# Unit/Integration Verification Blueprint  
def test\_lgbm\_objective():  
"""Validates shapes and numerical bounds."""  
y \= np.array(\[0, 1, 0, 1\])  
z \= np.array(\[-5.0, 5.0, 10.0, \-10.0\]) \# Includes confident errors  
ds \= lgb.Dataset(data=np.random.rand(4, 5), label=y)

    g, h \= robust\_binary\_logloss\_lgbm(z, ds)

    assert g.shape \== h.shape \== z.shape, "Shape mismatch."
    assert np.all(h \>= 1e-4), "Hessian floor violated."
    assert np.all(np.abs(g) \<= 0.85), "Gradient clipping failed."
    print("LightGBM custom objective passed stability validation.")

### **XGBoost Custom Objective**

The XGBoost interface similarly provides raw margins but expects interaction with xgb.DMatrix7. Note that utilizing custom objectives in XGBoost disables highly optimized SSE/AVX vectorization used in default histogram construction, increasing training times11.

Python  
import numpy as np  
import xgboost as xgb  
from typing import Tuple

def robust\_binary\_logloss\_xgb(preds: np.ndarray, dtrain: xgb.DMatrix) \-\> Tuple\[np.ndarray, np.ndarray\]:  
"""  
XGBoost Custom Objective: Robust Binary Cross-Entropy.  
Requires passing objective='binary:logitraw' to XGBClassifier/XGBRegressor  
to guarantee raw margins are returned in 'preds'.  
"""  
labels \= dtrain.get\_label()

    \# Stable exp transformation clipping bounds to \[-15, 15\]
    p \= 1.0 / (1.0 \+ np.exp(-np.clip(preds, \-15.0, 15.0)))

    grad \= p \- labels
    hess \= p \* (1.0 \- p)

    \# Apply soft attenuation to high-magnitude gradients (potential label noise)
    noise\_mask \= np.abs(grad) \> 0.90
    grad\[noise\_mask\] \= grad\[noise\_mask\] \* 0.5  \# Halve the pull of extreme errors

    \# Strict Hessian positive floor
    hess \= np.maximum(hess, 1e-4)

    return grad, hess

\# Integration Requirement  
\# clf \= xgb.XGBClassifier(objective='binary:logitraw', tree\_method='hist')  
\# clf.set\_params(objective=robust\_binary\_logloss\_xgb)

## **5\. Categorical Preprocessing Matrix**

The dataset schema consists of several distinct data types requiring specialized engine representations to maximize complementary decision boundaries. The explicit schema identified is:

> - **Numerical Continuous**: _Age, Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes_.
> - **Ordinal Categorical (0-5 integers)**: _Inflight wifi service, Ease of Online booking, Seat comfort, Food and drink, Cleanliness, etc._
> - **Nominal Categorical**: _Gender, Customer Type, Type of Travel, Class_.
> - **Binary Target**: _satisfaction_.

Artificial diversity induced by arbitrarily modifying encodings is a documented anti-pattern14. The preprocessing matrix assigns representations tailored to the algorithmic specificities of each framework.

| Feature Type                          | CatBoost Strategy                                                                                  | LightGBM (Leaf-Wise)                                                                   | LightGBM (Extra-Trees)                                                                            | XGBoost Strategy                                                                 |
| :------------------------------------ | :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------- |
| **Nominal Categorical**               | Leave as strings. Passed to cat\_features. CatBoost computes CTRs14.                               | Mapped to contiguous integers (0 to N-1). Passed to categorical\_feature5.             | One-Hot Encoded via Pandas get\_dummies. Forces randomized thresholds across explicit bins.       | Converted to Pandas category dtype. Used natively with enable\_categorical=True. |
| **Ordinal Categorical (0-5 Ratings)** | Converted to strings. Processed as high-cardinality categorical variables for deep combinations14. | Mapped to integers. Split naturally using Fisher's exact test proxy internal to LGBM5. | Left as integers. Treated strictly as continuous numerical variables for threshold randomization. | Left as integers. Split via continuous numerical rules (e.g., Rating \<= 3).     |
| **Continuous Numerical**              | Unmodified.                                                                                        | Unmodified.                                                                            | Unmodified.                                                                                       | Unmodified.                                                                      |
| **Missing Values**                    | Treated as a distinct string class "None" for CTR computation14.                                   | Retained as np.nan; algorithm routes to maximize gain natively.                        | Imputed with median or distinct out-of-bounds constant (e.g., \-999).                             | Retained as np.nan; algorithm utilizes sparsity-aware split routing17.           |

## **6\. Experimental Ablation Plan**

The experimental pathway prioritizes interventions by their theoretical expected AUC improvement and operational risk profile. Baseline conditions must be strictly preserved across folds, seeds, and hardware versions to establish paired statistical significance.

### **Phase 1: Baseline and Compatibility Audit**

> - **Objective:** Confirm OOF stability and measure raw correlation.
> - **Action:** Reproduce the existing 0.96128 AUC baseline. Log precise versions of CUDA and framework binaries.
> - **Acceptance Criteria:** Successful derivation of the pairwise Pearson and Spearman correlation matrices across the OOF predictions of all four un-tuned models.

### **Phase 2: Topological Regularization (High Expected Value, Low Risk)**

> - **Objective:** Prevent memorization of the 3.8% ambiguity via tree geometry constraints.
> - **Action:**

- Test CatBoost bagging\_temperature variations (1.5, 2.5)13.
- Test LightGBM path\_smooth variations (5.0, 10.0, 20.0)5.
- Test XGBoost gamma constraints (1.0, 2.0)17.

> - **Safeguard:** Maintain identical K-Fold validation splits. Record training times to detect algorithm deceleration.

### **Phase 3: Categorical Representation (Medium Expected Value, Medium Risk)**

> - **Objective:** Validate that the Categorical Preprocessing Matrix induces orthogonal errors without catastrophic standalone AUC degradation.
> - **Action:** Deploy the isolated preprocessing pipelines. Measure individual model OOF AUC. Measure ranking disagreement (the percentage of sample pairs ranked inversely by two models).
> - **Acceptance Criteria:** Incremental blend gain in logit space must exceed the baseline average by at least $+0.00015$ ROC-AUC.

### **Phase 4: Objective Experiments (Medium Expected Value, High Risk)**

> - **Objective:** Test targeted loss alterations if topological regularization proves insufficient.
> - **Action:**

- _Exp 4A:_ Apply label smoothing ($\alpha =0.038,0.076$) to the targets before training9.
- _Exp 4B:_ Construct cross-fitted student-teacher soft targets (${s}_{i}=0.75{y}_{i}+0.25{t}_{i}$), utilizing the baseline ensemble predictions as ${t}_{i}$.
- _Exp 4C:_ Deploy the robust\_binary\_logloss\_lgbm Python implementation on the LightGBM arm.

> - **Safeguard:** Reject any objective that severely degrades log loss calibration without providing a statistically significant boost to held-out ROC-AUC ranking.

## **7\. Ensemble-Diversity Analysis**

An ensemble whose constituent members exhibit Pearson correlations exceeding 0.998 on the predicted probability scale is approaching the asymptotic limit of naive equal-weight probability blending1. However, probabilities concentrated near 0.0 and 1.0 artificially inflate Pearson correlation coefficients while masking crucial rank disagreements in the uncertain boundary region ($p\approx 0.5$).  
To isolate and exploit genuine diversity, the following diagnostic and blending framework is mandated:

> 1. **Logit-Space Averaging:** Ensembling must be computed in the raw margin (logit) space: ${P}_{ensemble}=\sigma \left({\frac{1}{M}\sum\limits_{m=1}^{M}{\sigma }^{-1}({p}_{m})}\right)$. Logit averaging prevents highly confident, saturated predictions (e.g., $p=0.999$) from linearly overpowering less confident, but potentially more accurate, predictions22.
> 2. **Ranking Disagreement Matrix:** Compute the pairwise percentage of discordant pairs exclusively for instances where the ensemble average falls within the $[0.1,0.9]$ uncertainty band. A robust extra-trees model should demonstrate a \$\>5\\%\$ discordance rate within this band compared to the leaf-wise model, despite high global correlation.
> 3. **Shallow Meta-Learner Generation:** Train a highly constrained LightGBM meta-estimator (max\_depth=2, num\_leaves=4) on the cross-validated OOF base predictions. Set feature\_fraction=1.0 to force the estimator to construct logical AND/OR hyperplanes across the base model outputs rather than heavily weighting a single dominant arm.

## **8\. Final Execution Plan**

The sequential pipeline ensures full reproducibility and mitigates target leakage across all evaluation criteria.

> 1. **Environment Initialization:** Freeze Python environment parameters. Standardize CatBoost, LightGBM, and XGBoost library versions. Verify GPU kernel compatibility for task\_type="GPU".
> 2. **Schema Execution:** Process the raw train.csv and test.csv independently through the four distinct categorical pipelines dictated in the Categorical Preprocessing Matrix.
> 3. **Baseline Generation:** Fit the hyperparameter baselines documented in Section 3 across identical K-Fold schemas. Archive OOF arrays and diagnostic matrices.
> 4. **Topological Sweeps:** Execute Phase 2 ablations (path\_smooth, extra\_trees, gamma, bagging\_temperature). Identify the configuration per arm that maximizes validation AUC.
> 5. **Objective Intervention:** Apply student-teacher knowledge distillation (Exp 4B) exclusively to the LightGBM Leaf-Wise arm as a targeted noise regularizer. Compare against standard binary log loss.
> 6. **Meta-Optimization:** Collate the optimized OOF arrays. Optimize logit-space weights and evaluate the depth-2 LightGBM meta-learner via nested cross-validation.
> 7. **Submission Generation:** Fit the finalized estimators on the complete 700,000-row synthetic dataset. Predict the 300,000-row test instances. Apply the finalized ensemble schema. Conduct boundary verification to ensure all probabilities fall within the $(0,1)$ interval. Save the artifact record.

#### **Works cited**

> 1. train.csv
> 2. Estimating Calibrated Risks Using Focal Loss and Gradient-Boosted, [https\://www\.mdpi.com/2079-9292/14/9/1838](https://www.mdpi.com/2079-9292/14/9/1838)
> 3. Noise-Dependent Robustness of XGBoost, LightGBM, and CatBoost, [https\://www\.mdpi.com/1999-4893/19/9/730](https://www.mdpi.com/1999-4893/19/9/730)
> 4. Predicting lncRNA and disease associations with graph ... \- PMC \- NIH, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC12126542/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12126542/)
> 5. Parameters — LightGBM 4.7.0.99 documentation, [https\://lightgbm.readthedocs.io/en/latest/Parameters.html](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
> 6. XGBoost: Extreme Gradient Boosting \- Michael Brenndoerfer, [https\://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation](https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation)
> 7. Custom Objective and Evaluation Metric \- XGBoost Documentation, [https\://xgboost.readthedocs.io/en/stable/tutorials/custom\_metric\_obj.html](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)
> 8. Advanced Usage of Custom Objectives \- XGBoost Documentation, [https\://xgboost.readthedocs.io/en/latest/tutorials/advanced\_custom\_obj.html](https://xgboost.readthedocs.io/en/latest/tutorials/advanced_custom_obj.html)
> 9. Information Theory & Loss Functions: Why Cross-Entropy ... \- Medium, [https\://medium.com/@spjosyula2005/information-theory-loss-functions-why-cross-entropy-when-it-fails-and-how-to-fix-it-845ca31b47be](https://medium.com/@spjosyula2005/information-theory-loss-functions-why-cross-entropy-when-it-fails-and-how-to-fix-it-845ca31b47be)
> 10. GBDT with Nonconvex Loss for Tabular Classification in the ... \- arXiv, [https\://arxiv.org/pdf/2310.05067](https://arxiv.org/pdf/2310.05067)
> 11. XGBoost Train Model With Custom Objective Function, [https\://xgboosting.com/xgboost-train-model-with-custom-objective-function/](https://xgboosting.com/xgboost-train-model-with-custom-objective-function/)
> 12. Parameters Tuning — LightGBM 4.7.0.99 documentation, [https\://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
> 13. Large-scale modeling for housing condition prediction using ... \- PMC, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC13111609/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13111609/)
> 14. Categorical features parameters in CatBoost \- Medium, [https\://medium.com/data-science/categorical-features-parameters-in-catboost-4ebd1326bee5](https://medium.com/data-science/categorical-features-parameters-in-catboost-4ebd1326bee5)
> 15. CatBoost Parameters and Hyperparameters \- GeeksforGeeks, [https\://www\.geeksforgeeks.org/machine-learning/catboost-parameters-and-hyperparameters/](https://www.geeksforgeeks.org/machine-learning/catboost-parameters-and-hyperparameters/)
> 16. LightGBM Tuning Guide｜株式会社United Code \- note, [https\://note.com/united\_code/n/n2d809bb29262?hl=en](https://note.com/united_code/n/n2d809bb29262?hl=en)
> 17. The Notorious XGBoost \- Towards Data Science, [https\://towardsdatascience.com/the-notorious-xgboost-c7f7adc4c183/](https://towardsdatascience.com/the-notorious-xgboost-c7f7adc4c183/)
> 18. lightgbm.train — LightGBM 3.3.2 documentation, [https\://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/lightgbm.train.html](https://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/lightgbm.train.html)
> 19. lightgbm.LGBMClassifier — LightGBM 4.7.0.99 documentation, [https\://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
> 20. lightgbm.train — LightGBM 4.7.0 documentation, [https\://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html)
> 21. Custom Objective Functions and Evaluation Metrics in XGBoost, [https\://xgboost.org/concept/xgboost-custom-objective-function/](https://xgboost.org/concept/xgboost-custom-objective-function/)
> 22. Python API Reference — xgboost 0.80 documentation, [https\://xgboost.readthedocs.io/en/release\_0.80/python/python\_api.html](https://xgboost.readthedocs.io/en/release_0.80/python/python_api.html)
> 23. Prediction — xgboost 3.4.2 documentation, [https\://xgboost.readthedocs.io/en/stable/prediction.html](https://xgboost.readthedocs.io/en/stable/prediction.html)
