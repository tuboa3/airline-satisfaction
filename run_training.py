"""
Top-Level Training Runner for Kaggle and Local Environments
Execution:
    python run_training.py --model lightgbm
    python run_training.py --model catboost --device cuda
    python run_training.py --model xgboost --device cuda
    python run_training.py --model resnet --device cuda
"""

import argparse
import sys
from src.config import PathConfig, FeatureConfig, TrainConfig
from src.trainer import CrossValidationEngine
from src.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle S6E10 Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=[
            "lightgbm", "lgbm",
            "lightgbm_xt", "lgb_xt", "extra_trees",
            "catboost", "cb", "cat",
            "xgboost", "xgb",
            "realmlp", "tabm", "hybrid",
            "dcn_v2", "dcn", "deep_cross",
            "resnet", "tabular_resnet", "nn",
            "transformer", "ft_transformer", "ft",
        ],
        help="Model architecture to train (default: lightgbm)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated list of models to train sequentially (e.g. 'lightgbm,lightgbm_xt,xgboost')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cuda", "cpu"],
        help="Hardware accelerator to use (default: auto-detect)"
    )
    parser.add_argument(
        "--folds", "--n_splits",
        dest="folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10 per Option A)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs for neural architectures (e.g. resnet, ft_transformer)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Mini-batch size for neural architectures (e.g. 4096)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Custom directory containing train.csv and test.csv (default: auto-detect)"
    )
    parser.add_argument(
        "--original_path",
        type=str,
        default="",
        help="Custom directory or CSV path for original host dataset"
    )
    parser.add_argument(
        "--original_weight",
        type=float,
        default=0.50,
        help="Sample weight attenuation for original dataset rows (default: 0.50)"
    )
    parser.add_argument(
        "--density_ratio",
        action="store_true",
        help="Use adversarial density ratio weighting for original data"
    )
    parser.add_argument(
        "--use_original_data",
        action="store_true",
        default=False,
        help="Concatenate raw original rows into training folds (default: False, purged due to covariate shift)"
    )
    parser.add_argument(
        "--use_pseudo_labels",
        action="store_true",
        default=False,
        help="Augment training folds with high-confidence consensus pseudo-labeled test rows"
    )
    parser.add_argument(
        "--pseudo_path",
        type=str,
        default="",
        help="Path to pseudo_labeled_test.csv (default: auto-detect)"
    )
    parser.add_argument(
        "--pseudo_weight",
        type=float,
        default=0.60,
        help="Sample weight for pseudo-labeled rows (default: 0.60)"
    )
    parser.add_argument(
        "--use_glm_margin",
        action="store_true",
        default=False,
        help="Enable Stage 1 GLM margin generation for tree models (default: False)"
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        default=True,
        help="Save trained booster models and fitted feature pipelines for future inference (default: True)"
    )
    parser.add_argument(
        "--no_save_models",
        dest="save_models",
        action="store_false",
        help="Disable saving model boosters to disk"
    )
    return parser.parse_args()


def main():
    logger = get_logger("AirlineTraining")
    args = parse_args()

    paths = PathConfig(raw_dir=args.data_dir, original_path=args.original_path) if (args.data_dir or args.original_path) else PathConfig()
    feature_cfg = FeatureConfig()
    train_cfg = TrainConfig(
        n_splits=args.folds,
        random_state=args.seed,
        use_original_data=args.use_original_data,
        original_sample_weight=args.original_weight,
        use_density_ratio_weighting=args.density_ratio,
        use_pseudo_labels=args.use_pseudo_labels,
        pseudo_labels_path=args.pseudo_path,
        pseudo_sample_weight=args.pseudo_weight,
        use_glm_margin=args.use_glm_margin,
        save_models=args.save_models,
    )

    if args.epochs is not None:
        train_cfg.tabm_params["epochs"] = args.epochs
        train_cfg.dcn_params["epochs"] = args.epochs
        train_cfg.resnet_params["epochs"] = args.epochs
        train_cfg.ft_params["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg.tabm_params["batch_size"] = args.batch_size
        train_cfg.dcn_params["batch_size"] = args.batch_size
        train_cfg.resnet_params["batch_size"] = args.batch_size
        train_cfg.ft_params["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg.tabm_params["lr"] = args.lr
        train_cfg.dcn_params["lr"] = args.lr
        train_cfg.resnet_params["lr"] = args.lr
        train_cfg.ft_params["lr"] = args.lr

    model_list = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else [args.model]
    overall_scores = {}
    for model_name in model_list:
        logger.info("=" * 70)
        logger.info(f"STARTING CROSS-VALIDATION RUN FOR: '{model_name}'")
        logger.info("=" * 70)
        engine = CrossValidationEngine(
            paths=paths,
            feature_cfg=feature_cfg,
            train_cfg=train_cfg,
            model_name=model_name,
            device=args.device,
        )
        oof_auc, _, _ = engine.run()
        overall_scores[model_name] = oof_auc
        logger.info(f"Model '{model_name}' completed. OOF ROC-AUC: {oof_auc:.5f}")

    logger.info("=" * 70)
    logger.info("ALL REQUESTED MODEL RUNS COMPLETE:")
    for m_name, score in overall_scores.items():
        logger.info(f"  * {m_name:<20}: OOF ROC-AUC = {score:.5f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
