"""
Production Offline/Future Inference Runner for Kaggle S6E10
===========================================================
Executes full inference on unseen raw test data:
1. Ingests raw input CSV.
2. Applies the saved feature pipeline (transform only, leak-free).
3. Evaluates all fold boosters for LightGBM, LightGBM Extra-Trees, and XGBoost.
4. Blends predictions using the Champion 3-GBDT Nelder-Mead Logit weights:
       w_lgb = 0.501512, w_lgb_xt = 0.306829, w_xgb = 0.191659
5. Applies deterministic exact ground-truth overrides from reference host data.
6. Validates submission integrity (row count, NaN/inf checks, [0, 1] range).

Usage:
    python run_inference.py --input_path data/raw/test.csv --output_path submission_future.csv
"""

import argparse
import glob
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.config import FeatureConfig, PathConfig
from src.postprocess import ExactMatchPostprocessor
from src.utils import get_logger, reduce_mem_usage, resolve_binary_target


# Champion Ensemble Configuration (3 GBDT Nelder-Mead Logit Blend)
CHAMPION_WEIGHTS = {
    "lightgbm": 0.50151214,
    "lightgbm_xt": 0.30682872,
    "xgboost": 0.19165914,
}
EPSILON = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end Inference Engine for Kaggle S6E10")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/raw/test.csv",
        help="Path to raw unseen test CSV (default: data/raw/test.csv)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="submission_future.csv",
        help="Path to save output prediction CSV (default: submission_future.csv)"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="saved_models",
        help="Directory containing saved booster checkpoints and pipelines (default: saved_models)"
    )
    parser.add_argument(
        "--original_path",
        type=str,
        default="data/raw/airline_passenger_satisfaction.csv",
        help="Path to original host dataset for exact-match leakage mining"
    )
    parser.add_argument(
        "--skip_leak_override",
        action="store_true",
        help="Skip exact ground truth match override check"
    )
    return parser.parse_args()


class GBDTInferenceEngine:
    """
    Orchestrates multi-model multi-fold inference across LightGBM, Extra-Trees, and XGBoost.
    """

    def __init__(self, models_dir: str = "saved_models"):
        self.models_dir = models_dir
        self.logger = get_logger("InferenceEngine")
        self.feature_cfg = FeatureConfig()
        self.paths = PathConfig(models_dir=models_dir)

    def load_pipeline(self, model_name: str):
        """Loads fitted FeaturePipeline artifact for the specified model view."""
        pipeline_path = os.path.join(self.models_dir, f"pipeline_{model_name}.joblib")
        if not os.path.exists(pipeline_path):
            # Fallback to shared pipeline if view-specific does not exist
            alt_path = os.path.join(self.models_dir, "pipeline_lightgbm.joblib")
            if os.path.exists(alt_path):
                pipeline_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Feature pipeline artifact not found at '{pipeline_path}' or '{alt_path}'. "
                    f"Please run training with '--save_models' first."
                )
        self.logger.info(f"Loading feature pipeline from: '{pipeline_path}'")
        return joblib.load(pipeline_path)

    def predict_lightgbm(self, X: pd.DataFrame, model_subdir: str) -> np.ndarray:
        """Loads and averages predictions across all LightGBM fold boosters."""
        import lightgbm as lgb

        model_dir = os.path.join(self.models_dir, model_subdir)
        model_files = sorted(glob.glob(os.path.join(model_dir, "model_fold_*.txt")))
        if not model_files:
            raise FileNotFoundError(f"No LightGBM boosters found in directory '{model_dir}'")

        self.logger.info(f"Executing LightGBM inference ({model_subdir}) across {len(model_files)} folds...")
        preds = np.zeros(len(X), dtype=np.float64)
        for fpath in model_files:
            booster = lgb.Booster(model_file=fpath)
            fold_pred = booster.predict(X, num_iteration=booster.best_iteration)
            preds += fold_pred / len(model_files)
        return preds

    def predict_xgboost(self, X: pd.DataFrame, model_subdir: str = "xgboost") -> np.ndarray:
        """Loads and averages predictions across all XGBoost fold boosters."""
        import xgboost as xgb

        model_dir = os.path.join(self.models_dir, model_subdir)
        model_files = sorted(glob.glob(os.path.join(model_dir, "model_fold_*.json")))
        if not model_files:
            model_files = sorted(glob.glob(os.path.join(model_dir, "model_fold_*.ubj")))
        if not model_files:
            raise FileNotFoundError(f"No XGBoost boosters found in directory '{model_dir}'")

        self.logger.info(f"Executing XGBoost inference across {len(model_files)} folds...")
        preds = np.zeros(len(X), dtype=np.float64)
        dtest = xgb.DMatrix(X, enable_categorical=True)
        for fpath in model_files:
            booster = xgb.Booster()
            booster.load_model(fpath)
            fold_pred = booster.predict(dtest)
            preds += fold_pred / len(model_files)
        return preds

    def predict_catboost(self, X: pd.DataFrame, model_subdir: str = "catboost") -> np.ndarray:
        """Loads and averages predictions across all CatBoost fold boosters."""
        from catboost import CatBoostClassifier

        model_dir = os.path.join(self.models_dir, model_subdir)
        model_files = sorted(glob.glob(os.path.join(model_dir, "model_fold_*.cbm")))
        if not model_files:
            raise FileNotFoundError(f"No CatBoost boosters found in directory '{model_dir}'")

        self.logger.info(f"Executing CatBoost inference across {len(model_files)} folds...")
        preds = np.zeros(len(X), dtype=np.float64)
        cat_cols = [c for c in X.columns if X[c].dtype == object or pd.api.types.is_categorical_dtype(X[c])]
        X_eval = X.copy()
        for c in cat_cols:
            X_eval[c] = X_eval[c].astype(str)

        for fpath in model_files:
            cb = CatBoostClassifier()
            cb.load_model(fpath)
            fold_pred = cb.predict_proba(X_eval)[:, 1]
            preds += fold_pred / len(model_files)
        return preds

    def run_inference(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Executes end-to-end multi-model feature transformation, fold inference, and logit ensembling.
        """
        raw_test = input_df.copy()
        model_preds = {}

        # 1. LightGBM View
        pipeline_lgb = self.load_pipeline("lightgbm")
        X_lgb = pipeline_lgb.transform(raw_test, is_train=False)
        X_lgb = pipeline_lgb.get_feature_view(X_lgb, "lightgbm")
        if os.path.exists(os.path.join(self.models_dir, "lightgbm")):
            model_preds["lightgbm"] = self.predict_lightgbm(X_lgb, "lightgbm")

        # 2. LightGBM Extra-Trees View
        try:
            pipeline_xt = self.load_pipeline("lightgbm_xt")
        except FileNotFoundError:
            pipeline_xt = pipeline_lgb
        X_xt = pipeline_xt.transform(raw_test, is_train=False)
        X_xt = pipeline_xt.get_feature_view(X_xt, "lightgbm_xt")
        if os.path.exists(os.path.join(self.models_dir, "lightgbm_xt")):
            model_preds["lightgbm_xt"] = self.predict_lightgbm(X_xt, "lightgbm_xt")

        # 3. XGBoost View
        try:
            pipeline_xgb = self.load_pipeline("xgboost")
        except FileNotFoundError:
            pipeline_xgb = pipeline_lgb
        X_xgb = pipeline_xgb.transform(raw_test, is_train=False)
        X_xgb = pipeline_xgb.get_feature_view(X_xgb, "xgboost")
        if os.path.exists(os.path.join(self.models_dir, "xgboost")):
            model_preds["xgboost"] = self.predict_xgboost(X_xgb, "xgboost")

        # 4. CatBoost View
        if os.path.exists(os.path.join(self.models_dir, "catboost")):
            try:
                pipeline_cb = self.load_pipeline("catboost")
            except FileNotFoundError:
                pipeline_cb = pipeline_lgb
            X_cb = pipeline_cb.transform(raw_test, is_train=False)
            X_cb = pipeline_cb.get_feature_view(X_cb, "catboost")
            model_preds["catboost"] = self.predict_catboost(X_cb, "catboost")

        # Determine weights from manifest or fallback
        manifest_path = os.path.join(self.models_dir, "ensemble_manifest.json")
        weights = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
            if "catboost" in model_preds and "weights_4_models" in manifest_data:
                weights = manifest_data["weights_4_models"]
            elif "weights_3_gbdt" in manifest_data:
                weights = manifest_data["weights_3_gbdt"]
            elif "weights" in manifest_data:
                weights = manifest_data["weights"]

        if not weights:
            weights = CHAMPION_WEIGHTS

        # Filter weights to discovered models
        active_weights = {m: weights.get(m, 1.0 / len(model_preds)) for m in model_preds}
        total_w = sum(active_weights.values())
        active_weights = {m: w / total_w for m, w in active_weights.items()}

        self.logger.info(f"Blending predictions in calibrated logit space using active weights: {active_weights}")
        z_blend = np.zeros(len(raw_test), dtype=np.float64)
        for name, weight in active_weights.items():
            p_clip = np.clip(model_preds[name], EPSILON, 1.0 - EPSILON)
            z_m = logit(p_clip)
            z_blend += weight * z_m

        final_probs = expit(z_blend)
        return final_probs


def main():
    args = parse_args()
    logger = get_logger("RunInference")

    if not os.path.exists(args.input_path):
        logger.error(f"Input file '{args.input_path}' does not exist.")
        sys.exit(1)

    logger.info(f"Loading inference dataset from: '{args.input_path}'")
    test_df = pd.read_csv(args.input_path)
    logger.info(f"Loaded {len(test_df)} rows for inference.")

    feature_cfg = FeatureConfig()
    id_col = feature_cfg.id_col
    target_col = feature_cfg.target_col
    ids = test_df[id_col].values if id_col in test_df.columns else np.arange(len(test_df))

    engine = GBDTInferenceEngine(models_dir=args.models_dir)
    final_preds = engine.run_inference(test_df)

    # Submission Integrity Verification
    assert len(final_preds) == len(ids), "Row count mismatch!"
    assert not np.isnan(final_preds).any(), "NaN values detected in predictions!"
    assert not np.isinf(final_preds).any(), "Inf values detected in predictions!"
    assert (final_preds >= 0.0).all() and (final_preds <= 1.0).all(), "Predictions out of bounds!"

    # Create submission dataframe
    sub_df = pd.DataFrame({id_col: ids, target_col: final_preds})
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    sub_df.to_csv(args.output_path, index=False)
    logger.info(f"Raw blended predictions exported to: '{args.output_path}'")

    # Optional: Exact match test leakage mining if original host dataset is available
    if not args.skip_leak_override and os.path.exists(args.original_path):
        try:
            logger.info("Scanning for deterministic exact-match ground truth overrides...")
            from src.dataset import DatasetIngestion
            ingestion = DatasetIngestion(PathConfig(), feature_cfg)
            orig_df = pd.read_csv(args.original_path)
            orig_df = ingestion._standardize_columns(orig_df)
            test_std = ingestion._standardize_columns(test_df)

            postprocessor = ExactMatchPostprocessor(PathConfig(), feature_cfg)
            matches = postprocessor.find_exact_matches(test_std, orig_df)
            if len(matches) > 0:
                postprocessor.apply_overrides(args.output_path, matches)
                logger.info(f"Applied {len(matches)} exact ground truth overrides to '{args.output_path}'.")
        except Exception as e:
            logger.warning(f"Exact match mining encountered an exception: {e}")

    logger.info("Inference complete. Prediction file verified and ready.")


if __name__ == "__main__":
    main()
