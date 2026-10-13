"""
Model Extraction & Export Script for Future Web App Inference
=============================================================
Exports all 4 GBDT models (LightGBM, LightGBM Extra-Trees, XGBoost, and CatBoost)
together with the fitted FeaturePipeline and ensemble weights into 'saved_models/'.

Usage:
    python export_trained_models.py                # Full export on synthetic train
    python export_trained_models.py --fast         # Fast refit on subsample for rapid web-app development
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import FeatureConfig, PathConfig, TrainConfig
from src.dataset import DatasetIngestion
from src.features import FeaturePipeline
from src.models import CatBoostModel, LightGBMModel, XGBoostModel
from src.utils import get_logger, resolve_binary_target


def parse_args():
    parser = argparse.ArgumentParser(description="Export all GBDT models including CatBoost for future inference")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="Directory to save exported model artifacts (default: saved_models)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Custom raw data directory (default: auto-detect)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast refit on 60,000 sample for rapid offline testing and web-app development"
    )
    return parser.parse_args()


def export_full_models():
    args = parse_args()
    logger = get_logger("ExportModels")
    os.makedirs(args.output_dir, exist_ok=True)

    paths = PathConfig(raw_dir=args.data_dir, models_dir=args.output_dir) if args.data_dir else PathConfig(models_dir=args.output_dir)
    feature_cfg = FeatureConfig()
    train_cfg = TrainConfig()

    logger.info("Ingesting training, test, and external reference data...")
    ingestion = DatasetIngestion(paths, feature_cfg, train_cfg)
    unified_train, test_synth, sample_weights, orig_df = ingestion.load_and_prepare()

    synth_mask = (unified_train["is_original"] == 0).values
    train_synth = unified_train[synth_mask].reset_index(drop=True)
    y_synth = resolve_binary_target(train_synth[feature_cfg.target_col])

    if args.fast:
        logger.info("Fast mode enabled: subsampling 60,000 rows for rapid web-app export...")
        train_synth = train_synth.iloc[:60000].reset_index(drop=True)
        y_synth = y_synth[:60000]

    # 1. Fit & Export Feature Pipeline
    logger.info("Fitting and serializing production FeaturePipeline...")
    pipeline = FeaturePipeline(feature_cfg)
    X_train_full = pipeline.fit_transform(train_synth, test_synth, orig_df=orig_df)

    pipeline_path = os.path.join(args.output_dir, "pipeline_lightgbm.joblib")
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Saved feature pipeline to: '{pipeline_path}'")

    val_n = min(1000, len(y_synth))
    X_val_sample = X_train_full.iloc[:val_n]
    y_val_sample = y_synth[:val_n]

    # 2. Fit & Export LightGBM (Route Topology View)
    logger.info("Training and exporting LightGBM Booster (Route Topology View)...")
    lgb_dir = os.path.join(args.output_dir, "lightgbm")
    os.makedirs(lgb_dir, exist_ok=True)
    X_lgb = pipeline.get_feature_view(X_train_full, "lightgbm")

    lgb_params = train_cfg.lgb_params.copy()
    lgb_params["n_estimators"] = 400 if args.fast else 600
    lgb_model = LightGBMModel(params=lgb_params, device="cpu")
    lgb_model.fit(X_lgb, y_synth, X_lgb.iloc[:val_n], y_val_sample)
    lgb_path = os.path.join(lgb_dir, "model_fold_0.txt")
    lgb_model.model.save_model(lgb_path)
    logger.info(f"Exported LightGBM model to: '{lgb_path}'")

    # 3. Fit & Export LightGBM Extra-Trees
    logger.info("Training and exporting LightGBM Extra-Trees Booster...")
    xt_dir = os.path.join(args.output_dir, "lightgbm_xt")
    os.makedirs(xt_dir, exist_ok=True)
    X_xt = pipeline.get_feature_view(X_train_full, "lightgbm_xt")

    xt_params = train_cfg.lgb_xt_params.copy()
    xt_params["n_estimators"] = 400 if args.fast else 600
    xt_model = LightGBMModel(params=xt_params, device="cpu")
    xt_model.fit(X_xt, y_synth, X_xt.iloc[:val_n], y_val_sample)
    xt_path = os.path.join(xt_dir, "model_fold_0.txt")
    xt_model.model.save_model(xt_path)
    logger.info(f"Exported LightGBM Extra-Trees model to: '{xt_path}'")

    # 4. Fit & Export XGBoost (Psychometrics & Delay Dynamics View)
    logger.info("Training and exporting XGBoost Booster...")
    xgb_dir = os.path.join(args.output_dir, "xgboost")
    os.makedirs(xgb_dir, exist_ok=True)
    X_xgb = pipeline.get_feature_view(X_train_full, "xgboost")

    xgb_params = train_cfg.xgb_params.copy()
    xgb_params["n_estimators"] = 400 if args.fast else 600
    xgb_model = XGBoostModel(params=xgb_params, device="cpu")
    xgb_model.fit(X_xgb, y_synth, X_xgb.iloc[:val_n], y_val_sample)
    xgb_path = os.path.join(xgb_dir, "model_fold_0.json")
    xgb_model.model.save_model(xgb_path)
    logger.info(f"Exported XGBoost model to: '{xgb_path}'")

    # 5. Fit & Export CatBoost (Categorical & CTR View)
    logger.info("Training and exporting CatBoost Booster (Native Categorical View)...")
    cb_dir = os.path.join(args.output_dir, "catboost")
    os.makedirs(cb_dir, exist_ok=True)
    X_cb = pipeline.get_feature_view(X_train_full, "catboost")

    cb_params = train_cfg.cb_params.copy()
    cb_params["iterations"] = 300 if args.fast else 450
    cb_params["learning_rate"] = 0.08

    cb_model = CatBoostModel(params=cb_params, device="cpu")
    cat_cols_cb = [c for c in X_cb.columns if X_cb[c].dtype == object or pd.api.types.is_categorical_dtype(X_cb[c])]
    for c in cat_cols_cb:
        X_cb[c] = X_cb[c].astype(str)
    cb_model.fit(X_cb, y_synth, X_cb.iloc[:val_n], y_val_sample, cat_features=cat_cols_cb)
    cb_path = os.path.join(cb_dir, "model_fold_0.cbm")
    cb_model.model.save_model(cb_path)
    logger.info(f"Exported CatBoost model to: '{cb_path}'")

    # 6. Export Ensemble Manifest with Both 4-Model and 3-GBDT Weights
    manifest = {
        "ensemble_method": "nelder_mead_logit",
        "weights_4_models": {
            "catboost": 0.15646206,
            "lightgbm": 0.45160618,
            "lightgbm_xt": 0.24301828,
            "xgboost": 0.14891349,
        },
        "weights_3_gbdt": {
            "lightgbm": 0.50151214,
            "lightgbm_xt": 0.30682872,
            "xgboost": 0.19165914,
        },
        "epsilon": 1e-6,
        "models_available": ["lightgbm", "lightgbm_xt", "xgboost", "catboost"],
        "target_col": feature_cfg.target_col,
        "id_col": feature_cfg.id_col,
    }
    manifest_path = os.path.join(args.output_dir, "ensemble_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Exported ensemble manifest to: '{manifest_path}'")

    logger.info("ALL 4 MODELS (LightGBM, LightGBM_XT, XGBoost, CatBoost) successfully exported to 'saved_models/'.")


if __name__ == "__main__":
    export_full_models()
