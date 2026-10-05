"""
Domain 2: Tail-Preserving Logit Stacking & Meta-Learning Engine for Kaggle S6E10
Implements:
1. Isotonic Recalibration to restore monotonic probability fidelity.
2. Numerical Logit Transformation: logit(p) = ln((p + eps) / (1 - p + eps)).
3. Bounded SLSQP direct ROC-AUC maximization in logit space.
4. Second-stage Ridge Meta-Learner with cross-model interaction and disagreement terms.
5. Auto-selection of superior ensemble strategy and deterministic test override hooks.
"""

import glob
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import FeatureConfig, PathConfig
from src.utils import get_logger, resolve_binary_target, timer


class EnsembleOptimizer:
    """
    Advanced Ensembling Suite:
    - Isotonic Probability Calibration
    - Bounded SLSQP Logit Blending
    - Ridge Meta-Learning with Interaction Terms
    - Uniform Rank-Averaging Baseline
    """

    def __init__(
        self,
        paths: PathConfig = PathConfig(),
        feature_cfg: FeatureConfig = FeatureConfig(),
        epsilon: float = 1e-6,
    ):
        self.paths = paths
        self.feature_cfg = feature_cfg
        self.epsilon = epsilon
        self.logger = get_logger("EnsembleOptimizer")

    def load_ground_truth(self) -> np.ndarray:
        """Loads true binary targets from training dataset."""
        train_df = pd.read_csv(self.paths.train_path)
        if self.feature_cfg.target_col in train_df.columns:
            return resolve_binary_target(train_df[self.feature_cfg.target_col])
        else:
            raise KeyError(f"Target column '{self.feature_cfg.target_col}' not found in train.csv!")

    def load_test_ids(self) -> np.ndarray:
        """Loads test passenger IDs."""
        test_df = pd.read_csv(self.paths.test_path)
        return test_df[self.feature_cfg.id_col].values

    def discover_models(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Scans outputs directory dynamically for completed model OOF and test predictions."""
        available_models = {}
        canonical_map = {
            "lgbm": "lightgbm",
            "cb": "catboost",
            "cat": "catboost",
            "xgb": "xgboost",
            "ft": "ft_transformer",
            "transformer": "ft_transformer",
            "realmlp": "realmlp",
            "tabm": "realmlp",
            "resnet": "tabular_resnet",
            "tabular_resnet": "tabular_resnet",
        }

        pattern = os.path.join(self.paths.output_dir, "oof_preds_*.npy")
        for oof_path in sorted(glob.glob(pattern)):
            fname = os.path.basename(oof_path)
            slug = fname[len("oof_preds_") : -len(".npy")]
            test_path = os.path.join(self.paths.output_dir, f"test_preds_{slug}.npy")

            if os.path.exists(test_path):
                oof = np.load(oof_path)
                test = np.load(test_path)
                if len(oof) > 0 and len(test) > 0:
                    c_name = canonical_map.get(slug.lower(), slug.lower())
                    if c_name not in available_models:
                        available_models[c_name] = (oof, test)
                        self.logger.info(
                            f"Discovered completed model outputs for: '{c_name}' (source file: '{fname}')"
                        )

        return available_models

    def rank_transform(self, preds: np.ndarray) -> np.ndarray:
        """Transforms continuous predictions to normalized [0, 1] ranks."""
        return (rankdata(preds) - 1.0) / (len(preds) - 1.0)

    def calibrate_and_logit(
        self,
        models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Applies Isotonic Regression to each model's OOF predictions,
        adjusts test predictions, and converts both to logit space.
        """
        cal_oof_logits = {}
        cal_test_logits = {}

        for name, (oof, test) in models_dict.items():
            iso = IsotonicRegression(out_of_bounds="clip")
            p_oof_cal = iso.fit_transform(oof, y_true)
            p_test_cal = iso.predict(test)

            p_oof_clip = np.clip(p_oof_cal, self.epsilon, 1.0 - self.epsilon)
            p_test_clip = np.clip(p_test_cal, self.epsilon, 1.0 - self.epsilon)

            z_oof = logit(p_oof_clip)
            z_test = logit(p_test_clip)

            cal_oof_logits[name] = z_oof
            cal_test_logits[name] = z_test

        return cal_oof_logits, cal_test_logits

    def blend_rank(
        self,
        model_names: list[str],
        oof_list: list[np.ndarray],
        test_list: list[np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
        """Classic Rank Averaging Blend."""
        oof_ranks = np.column_stack([self.rank_transform(p) for p in oof_list])
        test_ranks = np.column_stack([self.rank_transform(p) for p in test_list])

        n = len(model_names)
        if len(y_true) > 60000:
            rng = np.random.RandomState(42)
            sub_idx = rng.choice(len(y_true), size=60000, replace=False)
            oof_sub = oof_ranks[sub_idx]
            y_sub = y_true[sub_idx]
        else:
            oof_sub = oof_ranks
            y_sub = y_true

        def objective(w):
            weights = np.maximum(0.0, w)
            s = weights.sum()
            if s == 0:
                weights = np.ones_like(weights) / n
            else:
                weights = weights / s
            blend = np.dot(oof_sub, weights)
            return -roc_auc_score(y_sub, blend)

        init_w = np.ones(n) / n
        res = minimize(objective, init_w, method="Nelder-Mead", options={"maxiter": 300})

        opt_w = np.maximum(0.0, res.x)
        opt_w = opt_w / opt_w.sum()

        final_oof = np.dot(oof_ranks, opt_w)
        final_test = np.dot(test_ranks, opt_w)
        best_auc = roc_auc_score(y_true, final_oof)
        w_dict = {name: float(w) for name, w in zip(model_names, opt_w)}
        return final_oof, best_auc, final_test, w_dict

    def blend_logit_slsqp(
        self,
        model_names: list[str],
        cal_oof_logits: dict[str, np.ndarray],
        cal_test_logits: dict[str, np.ndarray],
        y_true: np.ndarray,
        dirichlet_alpha: float = 1.05,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
        """
        Direct ROC-AUC Maximization in Calibrated Logit Space via Nelder-Mead
        with Dirichlet shrinkage prior toward uniform weights to prevent single-model domination.
        Formula: z_blend = sum(w_i * z_i), p_blend = sigmoid(z_blend).
        """
        X_oof_logits = np.column_stack([cal_oof_logits[m] for m in model_names])
        X_test_logits = np.column_stack([cal_test_logits[m] for m in model_names])
        n = len(model_names)

        if len(y_true) > 60000:
            rng = np.random.RandomState(42)
            sub_idx = rng.choice(len(y_true), size=60000, replace=False)
            X_sub = X_oof_logits[sub_idx]
            y_sub = y_true[sub_idx]
        else:
            X_sub = X_oof_logits
            y_sub = y_true

        def objective(w):
            weights = np.maximum(0.0, w)
            s = weights.sum()
            if s == 0:
                weights = np.ones_like(weights) / n
            else:
                weights = weights / s
            z_blend = np.dot(X_sub, weights)
            p_blend = expit(z_blend)
            auc = roc_auc_score(y_sub, p_blend)
            dirichlet_penalty = -(dirichlet_alpha - 1.0) * np.sum(np.log(weights + 1e-12))
            return -auc + 1e-4 * dirichlet_penalty

        init_w = np.ones(n) / n
        bounds = [(0.0, 1.0) for _ in range(n)]
        res = minimize(
            objective,
            init_w,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 300},
        )

        opt_w = np.maximum(0.0, res.x)
        if opt_w.sum() > 0:
            opt_w = opt_w / np.sum(opt_w)
        else:
            opt_w = np.ones(n) / n

        z_oof_final = np.dot(X_oof_logits, opt_w)
        z_test_final = np.dot(X_test_logits, opt_w)

        final_oof = expit(z_oof_final)
        final_test = expit(z_test_final)
        best_auc = roc_auc_score(y_true, final_oof)
        w_dict = {name: float(w) for name, w in zip(model_names, opt_w)}
        return final_oof, best_auc, final_test, w_dict

    def train_nnls(self, X_meta: np.ndarray, y_meta: np.ndarray) -> np.ndarray:
        """
        Non-Negative Least Squares (NNLS) meta-learner.
        Convex minimization: min ||X w - y||_2^2 subject to w >= 0.
        Eliminates destabilizing negative weights and tail miscalibration.
        """
        X = X_meta.astype(np.float64)
        y = y_meta.astype(np.float64)
        weights, _ = nnls(X, y)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(X.shape[1]) / X.shape[1]
        return weights.astype(np.float32)

    def blend_nnls(
        self,
        model_names: list[str],
        oof_list: list[np.ndarray],
        test_list: list[np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
        """
        Non-Negative Least Squares (NNLS) Convex Probability Combination.
        """
        X_meta_oof = np.column_stack(oof_list)
        X_meta_test = np.column_stack(test_list)

        opt_w = self.train_nnls(X_meta_oof, y_true)
        final_oof = np.dot(X_meta_oof, opt_w)
        final_test = np.dot(X_meta_test, opt_w)
        auc = roc_auc_score(y_true, final_oof)
        w_dict = {name: float(w) for name, w in zip(model_names, opt_w)}
        return final_oof, auc, final_test, w_dict

    def blend_isotonic_stacking(
        self,
        model_names: list[str],
        oof_list: list[np.ndarray],
        test_list: list[np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Isotonic Stacking Meta-Learner:
        Combines base predictions via NNLS weights, then applies cross-validated
        non-parametric Isotonic Regression (PAVA) to correct tail miscalibration (p < 0.02, p > 0.98).
        """
        X_meta_oof = np.column_stack(oof_list)
        X_meta_test = np.column_stack(test_list)

        # 1. Base combination via NNLS
        weights = self.train_nnls(X_meta_oof, y_true)
        raw_oof = np.dot(X_meta_oof, weights)
        raw_test = np.dot(X_meta_test, weights)

        # 2. 5-Fold Cross-Validation Isotonic Calibration
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cal_oof = np.zeros(len(y_true), dtype=np.float32)
        cal_test = np.zeros(len(raw_test), dtype=np.float32)

        for tr_idx, va_idx in skf.split(raw_oof, y_true):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_oof[tr_idx], y_true[tr_idx])
            cal_oof[va_idx] = iso.predict(raw_oof[va_idx])
            cal_test += iso.predict(raw_test) / 5.0

        auc = roc_auc_score(y_true, cal_oof)
        return cal_oof, auc, cal_test

    def blend_empirical_wmw(
        self,
        model_names: list[str],
        oof_list: list[np.ndarray],
        test_list: list[np.ndarray],
        y_true: np.ndarray,
        beta: float = 1e-3,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
        """
        Direct Empirical Wilcoxon-Mann-Whitney (WMW) U-Statistic Optimization (Friend 1).
        Derivative-free optimization on the probability simplex with a Dirichlet barrier:
            f(w) = -U(w) - beta * sum_{m=1}^M ln(w_m + 1e-12)
        Subject to:
            sum(w) = 1, w >= 0
        where U(w) is the exact empirical Mann-Whitney U-statistic (computed in O(N log N) via ranking).
        """
        n = len(model_names)
        X_oof = np.column_stack(oof_list)
        X_test = np.column_stack(test_list)

        pos_mask = (y_true == 1)
        n1 = int(np.sum(pos_mask))
        n0 = len(y_true) - n1

        if len(y_true) > 60000:
            rng = np.random.RandomState(42)
            sub_idx = rng.choice(len(y_true), size=60000, replace=False)
            X_sub = X_oof[sub_idx]
            y_sub = y_true[sub_idx]
            pos_sub = (y_sub == 1)
            n1_sub = int(np.sum(pos_sub))
            n0_sub = len(y_sub) - n1_sub
            const_sub = n1_sub * (n1_sub + 1) / 2.0
            denom_sub = float(n0_sub * n1_sub)
        else:
            X_sub = X_oof
            pos_sub = pos_mask
            n1_sub = n1
            const_sub = n1 * (n1 + 1) / 2.0
            denom_sub = float(n0 * n1)

        def objective(w):
            weights = np.maximum(0.0, w)
            s = weights.sum()
            if s == 0:
                weights = np.ones_like(weights) / n
            else:
                weights = weights / s

            p_blend = np.dot(X_sub, weights)
            ranks = rankdata(p_blend)
            r1 = np.sum(ranks[pos_sub])
            u_stat = (r1 - const_sub) / denom_sub

            # Dirichlet barrier to prevent single-model collapse
            barrier = -beta * np.sum(np.log(weights + 1e-12))
            return -u_stat + barrier

        best_w = None
        best_val = float("inf")

        # Multi-start initializations: uniform, NNLS, and best standalone model
        nnls_w = self.train_nnls(X_oof, y_true)
        single_best_idx = int(np.argmax([roc_auc_score(y_true, oof) for oof in oof_list]))
        single_w = np.zeros(n)
        single_w[single_best_idx] = 1.0

        candidates_init = [
            np.ones(n) / n,
            nnls_w,
            single_w,
        ]

        bounds = [(0.0, 1.0) for _ in range(n)]

        for init_w in candidates_init:
            res = minimize(
                objective,
                init_w,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": 300},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_w = res.x

        opt_w = np.maximum(0.0, best_w)
        if opt_w.sum() > 0:
            opt_w = opt_w / np.sum(opt_w)
        else:
            opt_w = np.ones(n) / n

        final_oof = np.dot(X_oof, opt_w)
        final_test = np.dot(X_test, opt_w)
        best_auc = roc_auc_score(y_true, final_oof)
        w_dict = {name: float(w) for name, w in zip(model_names, opt_w)}
        return final_oof, best_auc, final_test, w_dict

    def _build_meta_features(
        self,
        model_names: list[str],
        cal_oof_logits: dict[str, np.ndarray],
        cal_test_logits: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs high-dimensional meta-features:
        1. Base calibrated logits: z_i
        2. Pairwise multiplicative interactions: z_i * z_j
        3. Pairwise model disagreement magnitude: |z_i - z_j|
        """
        meta_features_oof = []
        meta_features_test = []

        # 1. Base Logits
        for m in model_names:
            meta_features_oof.append(cal_oof_logits[m])
            meta_features_test.append(cal_test_logits[m])

        # 2. Pairwise Interactions & Disagreements
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                mi, mj = model_names[i], model_names[j]
                zi_oof, zj_oof = cal_oof_logits[mi], cal_oof_logits[mj]
                zi_test, zj_test = cal_test_logits[mi], cal_test_logits[mj]

                # Product interaction
                meta_features_oof.append(zi_oof * zj_oof)
                meta_features_test.append(zi_test * zj_test)

                # Disagreement magnitude
                meta_features_oof.append(np.abs(zi_oof - zj_oof))
                meta_features_test.append(np.abs(zi_test - zj_test))

        X_meta_oof = np.column_stack(meta_features_oof).astype(np.float32)
        X_meta_test = np.column_stack(meta_features_test).astype(np.float32)
        return X_meta_oof, X_meta_test

    def blend_smooth_wmw(
        self,
        model_names: list[str],
        cal_oof_logits: dict[str, np.ndarray],
        cal_test_logits: dict[str, np.ndarray],
        y_true: np.ndarray,
        tau: float = 0.1,
        l2_reg: float = 1e-4,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Smooth Sigmoid AUC Surrogate Optimization for High-Dimensional Meta-Features (Friend 1).
        Optimizes:
            J(w) = -U_{smooth}(w) + (lambda/2) * ||w||_2^2
        where U_{smooth}(w) = E_{i in Pos, j in Neg} [ sigma((s_i - s_j)/tau) ].
        Solves via mini-batch balanced SGD with PyTorch, cross-validated with 5-fold CV to prevent leakage.
        """
        import torch
        import torch.nn as nn

        X_meta_oof, X_meta_test = self._build_meta_features(
            model_names, cal_oof_logits, cal_test_logits
        )
        D = X_meta_oof.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_meta_preds = np.zeros(len(y_true), dtype=np.float32)
        test_meta_preds = np.zeros(len(X_meta_test), dtype=np.float32)

        batch_size = 512
        n_steps = 60

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta_oof, y_true)):
            X_tr, y_tr = X_meta_oof[tr_idx], y_true[tr_idx]
            X_va = X_meta_oof[va_idx]

            pos_tr_idx = np.where(y_tr == 1)[0]
            neg_tr_idx = np.where(y_tr == 0)[0]

            X_tr_t = torch.from_numpy(X_tr).to(device)
            pos_t = torch.from_numpy(pos_tr_idx).to(device)
            neg_t = torch.from_numpy(neg_tr_idx).to(device)

            linear = nn.Linear(D, 1, bias=True).to(device)
            with torch.no_grad():
                linear.weight.zero_()
                for m_idx in range(len(model_names)):
                    linear.weight[0, m_idx] = 1.0 / len(model_names)
                linear.bias.zero_()

            optimizer = torch.optim.AdamW(linear.parameters(), lr=0.01, weight_decay=l2_reg)

            for step in range(n_steps):
                p_samp = pos_t[torch.randint(0, len(pos_t), (batch_size,), device=device)]
                n_samp = neg_t[torch.randint(0, len(neg_t), (batch_size,), device=device)]

                x_p = X_tr_t[p_samp]
                x_n = X_tr_t[n_samp]

                s_p = linear(x_p)
                s_n = linear(x_n)

                diff = (s_p - s_n.t()) / tau
                u_smooth = torch.sigmoid(diff).mean()
                loss = -u_smooth

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            linear.eval()
            with torch.no_grad():
                va_tensor = torch.from_numpy(X_va).to(device)
                te_tensor = torch.from_numpy(X_meta_test).to(device)
                oof_meta_preds[va_idx] = linear(va_tensor).squeeze(-1).cpu().numpy()
                test_meta_preds += linear(te_tensor).squeeze(-1).cpu().numpy() / 5.0

        final_oof = expit(oof_meta_preds)
        final_test = expit(test_meta_preds)
        best_auc = roc_auc_score(y_true, final_oof)
        return final_oof, best_auc, final_test

    def blend_ridge_interactions(
        self,
        model_names: list[str],
        cal_oof_logits: dict[str, np.ndarray],
        cal_test_logits: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Second-Stage Regularized Meta-Learner (Ridge) with Interaction Terms.
        Uses _build_meta_features to generate base logits, pairwise products, and disagreement magnitudes.
        """
        X_meta_oof, X_meta_test = self._build_meta_features(
            model_names, cal_oof_logits, cal_test_logits
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_meta_preds = np.zeros(len(y_true), dtype=np.float32)
        test_meta_preds = np.zeros(len(X_meta_test), dtype=np.float32)

        alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

        best_alpha = 10.0
        best_cv_auc = -1.0
        for alpha in alphas:
            fold_aucs = []
            for tr_idx, va_idx in skf.split(X_meta_oof, y_true):
                clf = Ridge(alpha=alpha, random_state=42)
                clf.fit(X_meta_oof[tr_idx], y_true[tr_idx])
                preds = clf.predict(X_meta_oof[va_idx])
                fold_aucs.append(roc_auc_score(y_true[va_idx], preds))
            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_cv_auc:
                best_cv_auc = mean_auc
                best_alpha = alpha

        self.logger.info(
            f"Ridge Meta-Learner selected optimal alpha={best_alpha} (Mean CV ROC-AUC: {best_cv_auc:.5f})"
        )

        for tr_idx, va_idx in skf.split(X_meta_oof, y_true):
            X_tr, y_tr = X_meta_oof[tr_idx], y_true[tr_idx]
            X_va = X_meta_oof[va_idx]

            clf = Ridge(alpha=best_alpha, random_state=42)
            clf.fit(X_tr, y_tr)

            oof_meta_preds[va_idx] = clf.predict(X_va)
            test_meta_preds += clf.predict(X_meta_test) / 5.0

        best_auc = roc_auc_score(y_true, oof_meta_preds)
        return oof_meta_preds, best_auc, test_meta_preds

    def run_all(self, chosen_method: str = "auto") -> tuple[np.ndarray, float, np.ndarray]:
        """
        Executes all ensembling paradigms, compares their out-of-fold performance,
        selects the optimal submission, and logs rigorous diagnostic metrics.
        """
        models_dict = self.discover_models()
        if not models_dict:
            raise FileNotFoundError(
                f"No model predictions found in '{self.paths.output_dir}'. "
                "Train at least two models before ensembling."
            )

        model_names = list(models_dict.keys())
        y_true = self.load_ground_truth()

        oof_list = [models_dict[m][0] for m in model_names]
        test_list = [models_dict[m][1] for m in model_names]

        # 1. Standalone Performances
        self.logger.info("=" * 70)
        self.logger.info("STANDALONE MODEL PERFORMANCES (OOF ROC-AUC)")
        self.logger.info("=" * 70)
        for name, oof in zip(model_names, oof_list):
            auc = roc_auc_score(y_true, oof)
            self.logger.info(f"  * {name:<20}: {auc:.5f}")

        # 2. Prediction Pearson Correlation Matrix
        if len(model_names) > 1:
            self.logger.info("-" * 70)
            self.logger.info("PREDICTION CORRELATION MATRIX (Pearson):")
            corr_df = pd.DataFrame(
                np.corrcoef(oof_list), index=model_names, columns=model_names
            )
            for col in corr_df.columns:
                corr_str = " | ".join(
                    [f"{corr_df.loc[row, col]:.4f}" for row in corr_df.index]
                )
                self.logger.info(f"  {col:<18}: {corr_str}")
            self.logger.info("-" * 70)

        # 3. Strategy 1: Rank Averaging
        oof_rank, auc_rank, test_rank, w_rank = self.blend_rank(
            model_names, oof_list, test_list, y_true
        )
        self.logger.info(f"Strategy 1 (Rank-Averaged Blend)        : OOF ROC-AUC = {auc_rank:.5f}")

        # 4. Calibration & Logit Conversion
        cal_oof_logits, cal_test_logits = self.calibrate_and_logit(models_dict, y_true)

        # 5. Strategy 2: Bounded Logit Blending (Nelder-Mead with Dirichlet Shrinkage)
        oof_slsqp, auc_slsqp, test_slsqp, w_slsqp = self.blend_logit_slsqp(
            model_names, cal_oof_logits, cal_test_logits, y_true
        )
        self.logger.info(f"Strategy 2 (Nelder-Mead Logit Blend)    : OOF ROC-AUC = {auc_slsqp:.5f}")
        for m, w in w_slsqp.items():
            self.logger.info(f"    - Weight for {m:<18}: {w:.4f}")

        # 6. Strategy 3: Direct Empirical WMW Optimization (Dirichlet Barrier)
        oof_wmw, auc_wmw, test_wmw, w_wmw = self.blend_empirical_wmw(
            model_names, oof_list, test_list, y_true
        )
        self.logger.info(f"Strategy 3 (Direct Empirical WMW Blend) : OOF ROC-AUC = {auc_wmw:.5f}")
        for m, w in w_wmw.items():
            self.logger.info(f"    - Empirical WMW Weight for {m:<11}: {w:.4f}")

        # 7. Strategy 4: Smooth Sigmoid WMW Surrogate Stacking
        if len(model_names) > 1:
            try:
                oof_swmw, auc_swmw, test_swmw = self.blend_smooth_wmw(
                    model_names, cal_oof_logits, cal_test_logits, y_true
                )
                self.logger.info(f"Strategy 4 (Smooth Sigmoid WMW Stacking): OOF ROC-AUC = {auc_swmw:.5f}")
            except Exception as e:
                self.logger.warning(f"Smooth Sigmoid WMW Stacking failed ({e}). Falling back to empirical WMW.")
                oof_swmw, auc_swmw, test_swmw = oof_wmw, auc_wmw, test_wmw
        else:
            oof_swmw, auc_swmw, test_swmw = oof_wmw, auc_wmw, test_wmw

        # 8. Strategy 5: Ridge Meta-Learner with Interactions
        if len(model_names) > 1:
            oof_ridge, auc_ridge, test_ridge = self.blend_ridge_interactions(
                model_names, cal_oof_logits, cal_test_logits, y_true
            )
            self.logger.info(f"Strategy 5 (Ridge Interaction Stacking) : OOF ROC-AUC = {auc_ridge:.5f}")
        else:
            oof_ridge, auc_ridge, test_ridge = oof_slsqp, auc_slsqp, test_slsqp

        # 9. Strategy 6: Non-Negative Least Squares (NNLS)
        oof_nnls, auc_nnls, test_nnls, w_nnls = self.blend_nnls(
            model_names, oof_list, test_list, y_true
        )
        self.logger.info(f"Strategy 6 (Non-Negative Least Squares) : OOF ROC-AUC = {auc_nnls:.5f}")
        for m, w in w_nnls.items():
            self.logger.info(f"    - NNLS Weight for {m:<13}: {w:.4f}")

        # 10. Strategy 7: Isotonic Stacking (NNLS + Isotonic PAVA)
        oof_iso, auc_iso, test_iso = self.blend_isotonic_stacking(
            model_names, oof_list, test_list, y_true
        )
        self.logger.info(f"Strategy 7 (Isotonic Calibrated Stacking): OOF ROC-AUC = {auc_iso:.5f}")

        # 11. Selection
        candidates = {
            "rank": (oof_rank, auc_rank, test_rank),
            "logit": (oof_slsqp, auc_slsqp, test_slsqp),
            "wmw": (oof_wmw, auc_wmw, test_wmw),
            "smooth_wmw": (oof_swmw, auc_swmw, test_swmw),
            "ridge": (oof_ridge, auc_ridge, test_ridge),
            "nnls": (oof_nnls, auc_nnls, test_nnls),
            "isotonic": (oof_iso, auc_iso, test_iso),
        }

        if chosen_method in candidates:
            champion_name = chosen_method
        else:
            champion_name = max(candidates.keys(), key=lambda k: candidates[k][1])

        champ_oof, champ_auc, champ_test = candidates[champion_name]

        self.logger.info("=" * 70)
        self.logger.info(
            f"CHAMPION ENSEMBLE: '{champion_name.upper()}' with OOF ROC-AUC = {champ_auc:.5f}"
        )
        self.logger.info("=" * 70)

        # Save Final Submission
        test_ids = self.load_test_ids()
        sub_df = pd.DataFrame(
            {self.feature_cfg.id_col: test_ids, self.feature_cfg.target_col: champ_test}
        )

        os.makedirs(self.paths.submissions_dir, exist_ok=True)
        sub_path = os.path.join(self.paths.submissions_dir, "submission_ensemble.csv")
        sub_df.to_csv(sub_path, index=False)
        sub_df.to_csv("submission.csv", index=False)
        self.logger.info(
            f"Exported Champion Submissions to '{sub_path}' and 'submission.csv'"
        )

        return champ_oof, champ_auc, champ_test
