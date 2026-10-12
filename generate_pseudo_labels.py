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
    pred_files = {
        "catboost": os.path.join(paths.submissions_dir, "submission_catboost.csv"),
        "lightgbm": os.path.join(paths.submissions_dir, "submission_lightgbm.csv"),
        "lightgbm_xt": os.path.join(paths.submissions_dir, "submission_lightgbm_xt.csv"),
        "xgboost": os.path.join(paths.submissions_dir, "submission_xgboost.csv"),
    }

    preds = {}
    for name, fpath in pred_files.items():
        if not os.path.exists(fpath):
            # Check outputs/test_preds_*.npy as fallback
            npy_path = os.path.join(paths.output_dir, f"test_preds_{name}.npy")
            if os.path.exists(npy_path):
                preds[name] = np.load(npy_path)
                logger.info(f"Loaded '{name}' predictions from npy: '{npy_path}'")
            else:
                raise FileNotFoundError(f"Missing predictions for '{name}': {fpath} and {npy_path} not found.")
        else:
            df = pd.read_csv(fpath)
            preds[name] = df[feature_cfg.target_col].values
            logger.info(f"Loaded '{name}' predictions from CSV: '{fpath}'")

    # 3. Compute Unanimous Consensus Mask
    p_cb = preds["catboost"]
    p_lgb = preds["lightgbm"]
    p_xt = preds["lightgbm_xt"]
    p_xgb = preds["xgboost"]

    p_ensemble_mean = (p_cb + p_lgb + p_xt + p_xgb) / 4.0

    pos_consensus = (
        (p_cb >= args.pos_threshold)
        & (p_lgb >= args.pos_threshold)
        & (p_xt >= args.pos_threshold)
        & (p_xgb >= args.pos_threshold)
    )

    neg_consensus = (
        (p_cb <= args.neg_threshold)
        & (p_lgb <= args.neg_threshold)
        & (p_xt <= args.neg_threshold)
        & (p_xgb <= args.neg_threshold)
    )

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
