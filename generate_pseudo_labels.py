"""
Transductive Consensus Pseudo-Labeling Generator for Kaggle S6E10
Generates ultra-high-confidence pseudo-labels from the unanimous agreement
of all 4 distinct GBDT model families (CatBoost, LightGBM, Extra-Trees, XGBoost).
"""

import argparse
import os
import numpy as np
import pandas as pd
from src.config import PathConfig, FeatureConfig
from src.utils import get_logger, timer


def parse_args():
    parser = argparse.ArgumentParser(description="Consensus Pseudo-Label Generator")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated list of models to include for consensus (default: all available)"
    )
    parser.add_argument(
        "--exclude_models",
        type=str,
        default="",
        help="Comma-separated list of models to exclude from consensus (e.g., 'catboost')"
    )
    parser.add_argument(
        "--pos_threshold",
        type=float,
        default=0.96,
        help="Minimum unanimous probability for positive pseudo-label (default: 0.96)"
    )
    parser.add_argument(
        "--neg_threshold",
        type=float,
        default=0.04,
        help="Maximum unanimous probability for negative pseudo-label (default: 0.04)"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.60,
        help="Base sample weight for pseudo-labeled rows (default: 0.60)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/pseudo_labeled_test.csv",
        help="Output CSV path for pseudo-labeled test rows"
    )
    return parser.parse_args()


def main():
    logger = get_logger("PseudoLabelGenerator")
    args = parse_args()

    paths = PathConfig()
    feature_cfg = FeatureConfig()

    logger.info("=" * 70)
    logger.info("GENERATING TRANSDUCTIVE CONSENSUS PSEUDO-LABELS")
    logger.info(f"Thresholds: Positive >= {args.pos_threshold:.2f} | Negative <= {args.neg_threshold:.2f}")
    logger.info("=" * 70)

    # 1. Load test features
    test_df = pd.read_csv(paths.test_path)
    N_test = len(test_df)
    logger.info(f"Loaded test dataset: {N_test:,} rows from '{paths.test_path}'.")

    # 2. Discover model prediction files
    all_candidate_models = ["catboost", "lightgbm", "lightgbm_xt", "xgboost"]
    if args.models:
        candidate_models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        candidate_models = all_candidate_models

    if args.exclude_models:
        excluded = set(m.strip() for m in args.exclude_models.split(",") if m.strip())
        candidate_models = [m for m in candidate_models if m not in excluded]

    preds = {}
    for name in candidate_models:
        fpath = os.path.join(paths.submissions_dir, f"submission_{name}.csv")
        npy_path = os.path.join(paths.output_dir, f"test_preds_{name}.npy")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            preds[name] = df[feature_cfg.target_col].values
            logger.info(f"Loaded '{name}' predictions from CSV: '{fpath}'")
        elif os.path.exists(npy_path):
            preds[name] = np.load(npy_path)
            logger.info(f"Loaded '{name}' predictions from npy: '{npy_path}'")
        else:
            if args.models:
                raise FileNotFoundError(f"Missing predictions for explicitly requested model '{name}'.")
            else:
                logger.warning(f"Prediction file for '{name}' not found; skipping from consensus.")

    if len(preds) < 2:
        raise ValueError(f"Need at least 2 models for consensus pseudo-labeling, but found only {len(preds)}: {list(preds.keys())}")

    logger.info(f"Active models for consensus pseudo-labeling ({len(preds)}): {list(preds.keys())}")

    # 3. Compute Unanimous Consensus Mask
    pos_consensus = np.ones(N_test, dtype=bool)
    neg_consensus = np.ones(N_test, dtype=bool)
    all_preds_list = []

    for name, p in preds.items():
        pos_consensus &= (p >= args.pos_threshold)
        neg_consensus &= (p <= args.neg_threshold)
        all_preds_list.append(p)

    p_ensemble_mean = np.mean(all_preds_list, axis=0)

    selected_mask = pos_consensus | neg_consensus
    N_selected = selected_mask.sum()
    N_pos = pos_consensus.sum()
    N_neg = neg_consensus.sum()

    logger.info("-" * 70)
    logger.info(f"Unanimous Consensus Results across {len(preds)} Architectures:")
    logger.info(f"  * Confident Satisfied (Y=1) : {N_pos:,} rows ({N_pos / N_test * 100:.2f}%)")
    logger.info(f"  * Confident Dissatisfied (Y=0): {N_neg:,} rows ({N_neg / N_test * 100:.2f}%)")
    logger.info(f"  * Total Pseudo-Labeled Rows : {N_selected:,} rows ({N_selected / N_test * 100:.2f}%)")
    logger.info("-" * 70)

    # 4. Construct Pseudo-Labeled DataFrame
    pseudo_df = test_df[selected_mask].copy()

    # Assign hard targets
    targets = np.zeros(N_selected, dtype=np.int32)
    targets[pos_consensus[selected_mask]] = 1
    pseudo_df[feature_cfg.target_col] = targets

    # Assign confidence-scaled sample weights
    conf_scores = np.abs(2.0 * p_ensemble_mean[selected_mask] - 1.0)
    sample_weights = (args.weight * conf_scores).astype(np.float32)
    pseudo_df["sample_weight"] = sample_weights

    # Mark provenance (2 = pseudo-labeled test)
    pseudo_df["is_original"] = 2

    # Save to disk
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    pseudo_df.to_csv(args.output_path, index=False)
    logger.info(f"Successfully saved {len(pseudo_df):,} consensus pseudo-labeled rows to: '{args.output_path}'")
    logger.info(f"Sample weight distribution: Min={sample_weights.min():.4f}, Mean={sample_weights.mean():.4f}, Max={sample_weights.max():.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
