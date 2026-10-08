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
        "--no_original",
        action="store_true",
        help="Disable original dataset ingestion and train purely on synthetic data"
    )
    parser.add_argument(
        "--no_glm_margin",
        action="store_true",
        help="Disable Stage 1 GLM margin generation for tree models"
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
        use_original_data=(not args.no_original),
        original_sample_weight=args.original_weight,
        use_density_ratio_weighting=args.density_ratio,
        use_glm_margin=(not args.no_glm_margin),
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

    engine = CrossValidationEngine(
        paths=paths,
        feature_cfg=feature_cfg,
        train_cfg=train_cfg,
        model_name=args.model,
        device=args.device,
    )

    oof_auc, _, _ = engine.run()
    logger.info(f"Execution finished successfully. Final OOF ROC-AUC: {oof_auc:.5f}")


if __name__ == "__main__":
    main()
