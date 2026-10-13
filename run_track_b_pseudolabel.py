"""
CLI Runner for Track B: Transductive Pseudo-Labeling Engine for the 3 Fast GBDTs
================================================================================
Automates the full Track B workflow:
1. Verifies consensus pseudo-labeled test set ('data/pseudo_labeled_test.csv').
   If missing, invokes generate_pseudo_labels.py across current 3-GBDT predictions.
2. Trains the 3 fast GBDT models (LightGBM, LightGBM Extra-Trees, XGBoost) sequentially
   with transductive pseudo-label augmentation in 10-fold CV.
3. Optimizes the pure 3-GBDT Constrained Simplex Rank Blend (excluding CatBoost).
4. Applies deterministic exact-match ground-truth test leak overrides.
5. Verifies submission integrity and updates champion submission.csv.

Usage:
    python run_track_b_pseudolabel.py                        # Full 10-fold execution
    python run_track_b_pseudolabel.py --models lightgbm      # Single model run
    python run_track_b_pseudolabel.py --skip_train           # Re-blend existing pseudo outputs
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FeatureConfig, PathConfig, TrainConfig
from src.dataset import DatasetIngestion
from src.ensemble import EnsembleOptimizer
from src.postprocess import ExactMatchPostprocessor
from src.trainer import CrossValidationEngine
from src.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Track B: Transductive Pseudo-Label Pipeline")
    parser.add_argument(
        "--models",
        type=str,
        default="lightgbm,lightgbm_xt,xgboost",
        help="Comma-separated models to train (default: 'lightgbm,lightgbm_xt,xgboost')",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)",
    )
    parser.add_argument(
        "--pseudo_path",
        type=str,
        default="data/pseudo_labeled_test.csv",
        help="Path to pseudo-labeled test dataset (default: data/pseudo_labeled_test.csv)",
    )
    parser.add_argument(
        "--pseudo_weight",
        type=float,
        default=0.60,
        help="Sample weight for pseudo-labeled rows (default: 0.60)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cuda", "cpu"],
        help="Hardware accelerator to use (default: auto-detect)",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip model training and immediately run 3-GBDT rank ensembling",
    )
    return parser.parse_args()


def main():
    logger = get_logger("TrackB_PseudoLabel")
    args = parse_args()

    paths = PathConfig()
    feature_cfg = FeatureConfig()
    train_cfg = TrainConfig(
        n_splits=args.folds,
        use_pseudo_labels=True,
        pseudo_labels_path=args.pseudo_path,
        pseudo_sample_weight=args.pseudo_weight,
        save_models=True,
    )

    logger.info("=" * 70)
    logger.info("TRACK B: TRANSDUCTIVE PSEUDO-LABELING FOR THE 3 FAST GBDTs")
    logger.info("=" * 70)

    # 1. Verify Pseudo-Labels Dataset
    pseudo_file = Path(args.pseudo_path)
    if not pseudo_file.exists():
        logger.info(f"Pseudo-label dataset not found at '{pseudo_file}'. Generating via consensus...")
        import subprocess
        subprocess.run([
            sys.executable,
            "generate_pseudo_labels.py",
            "--pos_threshold", "0.96",
            "--neg_threshold", "0.04",
            "--weight", str(args.pseudo_weight),
            "--models", "lightgbm,lightgbm_xt,xgboost",
        ], check=True)
    else:
        logger.info(f"Verified existing consensus pseudo-label dataset at: '{pseudo_file}'")

    # 2. Sequential Training of the 3 Fast GBDTs
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.skip_train:
        for model_name in model_list:
            logger.info("=" * 70)
            logger.info(f"STARTING PSEUDO-AUGMENTED 10-FOLD CV FOR: '{model_name}'")
            logger.info("=" * 70)
            engine = CrossValidationEngine(
                paths=paths,
                feature_cfg=feature_cfg,
                train_cfg=train_cfg,
                model_name=model_name,
                device=args.device,
            )
            oof_auc, _, _ = engine.run()
            logger.info(f"Model '{model_name}' completed. OOF ROC-AUC: {oof_auc:.5f}")

    # 3. Finalize 3-GBDT Rank Blend (Excluding CatBoost)
    logger.info("=" * 70)
    logger.info("FINALIZING PURE 3-GBDT CONSTRAINED RANK ENSEMBLE")
    logger.info("=" * 70)
    optimizer = EnsembleOptimizer(paths=paths, feature_cfg=feature_cfg, train_cfg=train_cfg)
    champ_oof, best_auc, champ_test = optimizer.run_all(
        chosen_method="rank",
        models=model_list,
        exclude_models=["catboost"],
    )

    # 4. Exact-Match Ground-Truth Overrides
    try:
        ingestion = DatasetIngestion(paths=paths, feature_cfg=feature_cfg)
        orig_files = ingestion.discover_original_files()
        if orig_files:
            import pandas as pd
            orig_df = pd.read_csv(orig_files[0])
            test_df = pd.read_csv(paths.test_path)
            postprocessor = ExactMatchPostprocessor(paths=paths, feature_cfg=feature_cfg)
            matches = postprocessor.find_exact_matches(test_df, orig_df)
            if len(matches) > 0:
                postprocessor.apply_overrides("submission.csv", matches)
                postprocessor.apply_overrides(
                    f"{paths.submissions_dir}/submission_ensemble.csv", matches
                )
                logger.info(f"Applied {len(matches)} exact ground-truth overrides to submissions.")
    except Exception as e:
        logger.warning(f"Exact match postprocessing check notice: {e}")

    logger.info("=" * 70)
    logger.info(f"TRACK B COMPLETE: Champion 3-GBDT Blended OOF ROC-AUC = {best_auc:.5f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
