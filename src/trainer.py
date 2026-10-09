"""
Cross-Validation Training Engine: Stratified K-Fold with OOF Scoring and Submission Generation
Integrates:
- Domain 1: Original host dataset ingestion, provenance weighting, and leakage mining
- Domain 3: Un-distilled Tabular ResNet / RealMLP with PLR embeddings
- Domain 4: Rotational SVD manifolds & Multi-way Bayesian target encoding
"""

import gc
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import FeatureConfig, PathConfig, TrainConfig
from src.dataset import DatasetIngestion
from src.features import FeaturePipeline
from src.models import GLMMarginGenerator, get_model
from src.postprocess import ExactMatchPostprocessor
from src.utils import detect_hardware, resolve_binary_target, seed_everything, timer


class CrossValidationEngine:
    """
    Orchestrates full 5-fold cross-validation, OOF evaluation,
    and ensemble test predictions.
    """

    def __init__(
        self,
        paths: PathConfig = PathConfig(),
        feature_cfg: FeatureConfig = FeatureConfig(),
        train_cfg: TrainConfig = TrainConfig(),
        model_name: str = "lightgbm",
        device: str = None,
    ):
        self.paths = paths
        self.feature_cfg = feature_cfg
        self.train_cfg = train_cfg
        self.model_name = model_name

        if device is None:
            self.device, _ = detect_hardware()
        else:
            self.device = device

        seed_everything(self.train_cfg.random_state)
        self.pipeline = FeaturePipeline(self.feature_cfg)

    def run(self) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Executes complete training and inference pipeline:
        1. Ingests synthetic train, test, and auxiliary original dataset.
        2. Applies full feature pipeline (Domain A, C, 4).
        3. Runs 5-Fold Stratified CV on synthetic target distribution with augmented training.
        4. Logs OOF score, exports submission.csv, and checks exact-match overrides.
        """
        logging.info("=" * 70)
        logging.info(
            f"STARTING CROSS-VALIDATION PIPELINE: Model={self.model_name.upper()} | Device={self.device.upper()}"
        )
        logging.info("=" * 70)

        # 1. Load Data with Domain 1 Ingestion
        ingestion = DatasetIngestion(self.paths, self.feature_cfg, self.train_cfg)
        unified_train, test_synth, sample_weights, orig_df = ingestion.load_and_prepare()

        y_all = resolve_binary_target(unified_train[self.feature_cfg.target_col])
        test_ids = (
            test_synth[self.feature_cfg.id_col].values
            if self.feature_cfg.id_col in test_synth.columns
            else np.arange(len(test_synth))
        )

        # 2. Transductive Feature Engineering with Original Dataset Prior Integration
        X_train_all = self.pipeline.fit_transform(
            unified_train, test_synth, orig_df=orig_df
        )
        X_test = self.pipeline.transform(test_synth, is_train=False)

        # Route to Architecture-Specific Feature View (Breaking Pearson r > 0.998 Collinearity)
        X_train_all = self.pipeline.get_feature_view(
            X_train_all, self.model_name
        )
        X_test = self.pipeline.get_feature_view(X_test, self.model_name)

        feature_names = X_train_all.columns.tolist()
        logging.info(
            f"Engineered Feature Count for View [{self.model_name.upper()}]: {len(feature_names)}"
        )

        # Isolate synthetic indices for leak-proof CV evaluation (0: synthetic, 1: original)
        synth_mask = (unified_train["is_original"] == 0).values
        synth_indices = np.where(synth_mask)[0]
        orig_indices = np.where(~synth_mask)[0]
        y_synth = y_all[synth_mask]

        logging.info(
            f"CV Evaluation Plan: {len(synth_indices)} synthetic holdout rows across {self.train_cfg.n_splits} folds."
        )
        if len(orig_indices) > 0:
            logging.info(
                f"Auxiliary Training Support: {len(orig_indices)} original dataset rows augmenting each fold."
            )

        # 3. Stratified K-Fold Training
        skf = StratifiedKFold(
            n_splits=self.train_cfg.n_splits,
            shuffle=self.train_cfg.shuffle,
            random_state=self.train_cfg.random_state,
        )

        oof_preds = np.zeros(len(synth_indices), dtype=np.float32)
        test_preds = np.zeros(len(X_test), dtype=np.float32)
        fold_scores = []

        model_name_lower = self.model_name.lower()
        if "cat" in model_name_lower or "cb" in model_name_lower:
            params = self.train_cfg.cb_params.copy()
        elif "xgb" in model_name_lower or "xgboost" in model_name_lower:
            params = self.train_cfg.xgb_params.copy()
        elif any(k in model_name_lower for k in ["extra_trees", "xt", "lightgbm_xt", "lgb_xt"]):
            params = self.train_cfg.lgb_xt_params.copy()
        elif (
            "realmlp" in model_name_lower
            or "tabm" in model_name_lower
            or "hybrid" in model_name_lower
        ):
            params = self.train_cfg.tabm_params.copy()
        elif "dcn" in model_name_lower or "cross" in model_name_lower:
            params = self.train_cfg.dcn_params.copy()
        elif (
            "resnet" in model_name_lower
            or "tabular_resnet" in model_name_lower
            or "nn" in model_name_lower
        ):
            params = self.train_cfg.resnet_params.copy()
        elif "transformer" in model_name_lower or "ft" in model_name_lower:
            params = self.train_cfg.ft_params.copy()
        else:
            params = self.train_cfg.lgb_params.copy()

        # Check for teacher predictions if using distilled FT-Transformer
        teacher_oof = None
        teacher_test = None
        if "transformer" in model_name_lower or "ft" in model_name_lower:
            teacher_oof_files = [
                os.path.join(self.paths.output_dir, "oof_preds_lightgbm.npy"),
                os.path.join(self.paths.output_dir, "oof_preds_xgboost.npy"),
                os.path.join(self.paths.output_dir, "oof_preds_catboost.npy"),
            ]
            teacher_test_files = [
                os.path.join(self.paths.output_dir, "test_preds_lightgbm.npy"),
                os.path.join(self.paths.output_dir, "test_preds_xgboost.npy"),
                os.path.join(self.paths.output_dir, "test_preds_catboost.npy"),
            ]
            valid_oof = [np.load(f) for f in teacher_oof_files if os.path.exists(f)]
            valid_test = [np.load(f) for f in teacher_test_files if os.path.exists(f)]
            if valid_oof:
                teacher_oof = np.mean(valid_oof, axis=0)
                logging.info(f"Loaded {len(valid_oof)} GBDT teacher models for soft distillation.")
            if valid_test:
                teacher_test = np.mean(valid_test, axis=0)
                logging.info(f"Loaded {len(valid_test)} GBDT teacher models for test consistency.")

        for fold, (synth_tr_subidx, synth_va_subidx) in enumerate(skf.split(synth_indices, y_synth)):
            logging.info("-" * 50)
            logging.info(f"FOLD {fold + 1} / {self.train_cfg.n_splits}")
            logging.info("-" * 50)

            val_global_idx = synth_indices[synth_va_subidx]
            if len(orig_indices) > 0:
                train_global_idx = np.concatenate([synth_indices[synth_tr_subidx], orig_indices])
            else:
                train_global_idx = synth_indices[synth_tr_subidx]

            X_tr = X_train_all.iloc[train_global_idx].copy()
            y_tr = y_all[train_global_idx]
            sw_tr = sample_weights[train_global_idx]

            X_va = X_train_all.iloc[val_global_idx].copy()
            y_va = y_all[val_global_idx]
            X_te_fold = X_test.copy()

            extra_fit_kwargs = {}
            margin_va = None
            margin_te = None

            # Fold-Safe Continuous Flight Distance Decomposition (Friend 1 Methodology)
            if "Flight Distance" in X_tr.columns and getattr(self.feature_cfg, "enable_dist_decomposition", True):
                prior_y = float(np.mean(y_tr))
                fd_stats = (
                    pd.DataFrame({"fd": X_tr["Flight Distance"].values, "y": y_tr})
                    .groupby("fd")["y"]
                    .agg(["mean", "count"])
                )
                unique_dists = np.sort(fd_stats.index.values)

                # Continuous rolling trend across +-50 miles
                trend_dict = {}
                for d in unique_dists:
                    mask = (unique_dists >= d - 50.0) & (unique_dists <= d + 50.0)
                    near_dists = unique_dists[mask]
                    near_counts = fd_stats.loc[near_dists, "count"].values
                    near_means = fd_stats.loc[near_dists, "mean"].values
                    tot = near_counts.sum()
                    trend_dict[d] = float((near_means * near_counts).sum() / tot) if tot > 0 else prior_y

                # Map rolling trend
                X_tr["dist_trend"] = X_tr["Flight Distance"].map(trend_dict).fillna(prior_y).astype(np.float32)
                X_va["dist_trend"] = X_va["Flight Distance"].map(trend_dict).fillna(prior_y).astype(np.float32)
                X_te_fold["dist_trend"] = X_te_fold["Flight Distance"].map(trend_dict).fillna(prior_y).astype(np.float32)

                # Empirical Bayes shrunk route residual (m=20)
                m_shrink = 20.0
                res_dict = {}
                for d in unique_dists:
                    n_d = fd_stats.loc[d, "count"]
                    raw_res = fd_stats.loc[d, "mean"] - trend_dict[d]
                    res_dict[d] = float((n_d / (n_d + m_shrink)) * raw_res)

                X_tr["dist_route_residual"] = X_tr["Flight Distance"].map(res_dict).fillna(0.0).astype(np.float32)
                X_va["dist_route_residual"] = X_va["Flight Distance"].map(res_dict).fillna(0.0).astype(np.float32)
                X_te_fold["dist_route_residual"] = X_te_fold["Flight Distance"].map(res_dict).fillna(0.0).astype(np.float32)

            # Leak-Free Multi-Key In-Fold Target Encoding (Koumei Maki + Busyaprime + Goodpjw)
            # Strictly computed on the training indices of the active fold:
            if not ("cat" in model_name_lower or "cb" in model_name_lower):
                prior_y = float(np.mean(y_tr))
                smooth_m = 20.0

                te_key_defs = []
                if "Flight Distance" in X_tr.columns:
                    te_key_defs.append(
                        ("te_Flight Distance", X_tr["Flight Distance"], X_va["Flight Distance"], X_test["Flight Distance"])
                    )
                    te_key_defs.append(
                        ("te_FD_div10", (X_tr["Flight Distance"] // 10).astype(str), (X_va["Flight Distance"] // 10).astype(str), (X_test["Flight Distance"] // 10).astype(str))
                    )
                    te_key_defs.append(
                        ("te_FD_div100", (X_tr["Flight Distance"] // 100).astype(str), (X_va["Flight Distance"] // 100).astype(str), (X_test["Flight Distance"] // 100).astype(str))
                    )
                if "Age" in X_tr.columns:
                    te_key_defs.append(
                        ("te_Age", X_tr["Age"], X_va["Age"], X_test["Age"])
                    )
                if "Flight Distance" in X_tr.columns and "Class" in X_tr.columns:
                    te_key_defs.append(
                        ("te_FD_x_Class", X_tr["Flight Distance"].astype(str) + "_" + X_tr["Class"].astype(str),
                         X_va["Flight Distance"].astype(str) + "_" + X_va["Class"].astype(str),
                         X_test["Flight Distance"].astype(str) + "_" + X_test["Class"].astype(str))
                    )
                if "Flight Distance" in X_tr.columns and "Type of Travel" in X_tr.columns:
                    te_key_defs.append(
                        ("te_FD_x_Travel", X_tr["Flight Distance"].astype(str) + "_" + X_tr["Type of Travel"].astype(str),
                         X_va["Flight Distance"].astype(str) + "_" + X_va["Type of Travel"].astype(str),
                         X_test["Flight Distance"].astype(str) + "_" + X_test["Type of Travel"].astype(str))
                    )

                for te_name, tr_s, va_s, te_s in te_key_defs:
                    c_counts = tr_s.value_counts()
                    c_sums = pd.Series(y_tr, index=X_tr.index).groupby(tr_s).sum()
                    te_map = ((c_sums + smooth_m * prior_y) / (c_counts + smooth_m)).to_dict()

                    X_tr[te_name] = tr_s.map(te_map).fillna(prior_y).astype(np.float32)
                    X_va[te_name] = va_s.map(te_map).fillna(prior_y).astype(np.float32)
                    X_te_fold[te_name] = te_s.map(te_map).fillna(prior_y).astype(np.float32)

            # Native Categorical CTR for CatBoost (Shelton Wang Topic #745892)
            if "cat" in model_name_lower or "cb" in model_name_lower:
                candidate_cats = [
                    "Gender",
                    "Customer Type",
                    "Type of Travel",
                    "Class",
                    "class_x_travel_type",
                    "gate_x_business",
                    "multi_cross_1",
                    "multi_cross_2",
                    "route_class_travel",
                    "route_delay_tier",
                    "route_dissatisfaction",
                    "service_failure_class",
                    "age_class_travel",
                ]
                cat_cols = [
                    c
                    for c in X_tr.columns
                    if c in candidate_cats
                    or (
                        ("_x_" in c or "__X__" in c or "__category" in c or "__cat" in c)
                        and not c.startswith("te_")
                        and not c.startswith("log_")
                        and not c.startswith("org_")
                    )
                    or (c in self.feature_cfg.rating_cols)
                ]
                for col in cat_cols:
                    X_tr[col] = X_tr[col].astype(str)
                    X_va[col] = X_va[col].astype(str)
                    X_te_fold[col] = X_te_fold[col].astype(str)
                extra_fit_kwargs["cat_features"] = cat_cols

            teacher_tr = (
                teacher_oof[synth_tr_subidx]
                if teacher_oof is not None and len(teacher_oof) == len(synth_indices)
                else None
            )

            model = get_model(self.model_name, params=params.copy(), device=self.device)

            with timer(f"Fold {fold + 1} Training ({len(X_tr)} train rows, {len(X_va)} val rows)"):
                model.fit(
                    X_tr,
                    y_tr,
                    X_va,
                    y_va,
                    sample_weight=sw_tr,
                    teacher_train=teacher_tr,
                    X_test=X_te_fold,
                    teacher_test=teacher_test,
                    **extra_fit_kwargs,
                )

            # Build clean kwargs for predict_proba (exclude training-only keys)
            predict_kwargs = {}
            if "cat_features" in extra_fit_kwargs:
                predict_kwargs["cat_features"] = extra_fit_kwargs["cat_features"]

            if margin_va is not None:
                val_preds = model.predict_proba(
                    X_va, base_margin=margin_va, **predict_kwargs
                )
            else:
                val_preds = model.predict_proba(X_va, **predict_kwargs)
            oof_preds[synth_va_subidx] = val_preds

            fold_auc = roc_auc_score(y_va, val_preds)
            fold_scores.append(fold_auc)
            logging.info(f"--> Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

            # Accumulate test predictions
            if margin_te is not None:
                test_preds += (
                    model.predict_proba(
                        X_te_fold, base_margin=margin_te, **predict_kwargs
                    )
                    / self.train_cfg.n_splits
                )
            else:
                test_preds += (
                    model.predict_proba(X_te_fold, **predict_kwargs)
                    / self.train_cfg.n_splits
                )

            # Memory garbage collection
            del X_tr, y_tr, sw_tr, X_va, y_va, model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

        # 4. Overall Out-Of-Fold Evaluation
        overall_oof_auc = roc_auc_score(y_synth, oof_preds)
        logging.info("=" * 70)
        logging.info(f"5-FOLD CV COMPLETE: Overall Synthetic OOF ROC-AUC = {overall_oof_auc:.5f}")
        logging.info(
            f"Mean Fold AUC: {np.mean(fold_scores):.5f} | Std: {np.std(fold_scores):.5f}"
        )
        logging.info("=" * 70)

        # 5. Save Outputs & Submissions
        np.save(
            os.path.join(self.paths.output_dir, f"oof_preds_{self.model_name}.npy"),
            oof_preds,
        )
        np.save(
            os.path.join(self.paths.output_dir, f"test_preds_{self.model_name}.npy"),
            test_preds,
        )

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
        canonical_name = canonical_map.get(self.model_name.lower(), self.model_name.lower())
        if canonical_name != self.model_name.lower():
            np.save(
                os.path.join(self.paths.output_dir, f"oof_preds_{canonical_name}.npy"),
                oof_preds,
            )
            np.save(
                os.path.join(self.paths.output_dir, f"test_preds_{canonical_name}.npy"),
                test_preds,
            )

        submission_path = os.path.join(
            self.paths.submissions_dir, f"submission_{self.model_name}.csv"
        )
        sub_df = pd.DataFrame(
            {self.feature_cfg.id_col: test_ids, self.feature_cfg.target_col: test_preds}
        )
        sub_df.to_csv(submission_path, index=False)
        sub_df.to_csv("submission.csv", index=False)
        logging.info(f"Generated Submissions: '{submission_path}' & 'submission.csv'")

        # 6. Domain 1: Safe Post-Processing Exact-Match Target Leakage Mining
        if orig_df is not None:
            postprocessor = ExactMatchPostprocessor(self.paths, self.feature_cfg)
            matches = postprocessor.find_exact_matches(test_synth, orig_df)
            if len(matches) > 0:
                postprocessor.apply_overrides(submission_path, matches)
                postprocessor.apply_overrides("submission.csv", matches)

        return overall_oof_auc, oof_preds, test_preds
