"""
Track A: Public Diverse OOF Ingestion & Regularized Logistic Stacking Engine
=============================================================================
Discovers, validates, and stacks internal 3 fast GBDTs with diverse public OOFs
(e.g., Chris Deotte 124 models, TabPFN/TFM, RealMLP) in logit space.

Usage:
    python src/stack_public.py                                # Auto-discovers data/public_oof/
    python src/stack_public.py --public_dir /kaggle/input     # Kaggle multi-dataset search
    python src/stack_public.py --C 0.05 --min_auc 0.9605      # Tuned regularization
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Add repo root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FeatureConfig, PathConfig
from src.postprocess import ExactMatchPostprocessor
from src.utils import get_logger, resolve_binary_target


def parse_args():
    parser = argparse.ArgumentParser(description="Track A: Public OOF Logistic Stacking")
    parser.add_argument(
        "--public_dir",
        type=str,
        default="data/public_oof",
        help="Directory to search for public OOFs (default: data/public_oof)",
    )
    parser.add_argument(
        "--min_auc",
        type=float,
        default=0.9600,
        help="Minimum OOF ROC-AUC threshold to include a public model (default: 0.9600)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=0.05,
        help="Inverse regularization strength for Logistic Regression (default: 0.05)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Clipping constant for logit transformation (default: 1e-6)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="submission_public_stack.csv",
        help="Output submission filename under submissions/ (default: submission_public_stack.csv)",
    )
    parser.add_argument(
        "--update_champion",
        action="store_true",
        default=True,
        help="Copy resulting submission to root submission.csv if OOF improves (default: True)",
    )
    return parser.parse_args()


def to_logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Converts probabilities to clipped numerical logits."""
    clipped = np.clip(probs, eps, 1.0 - eps)
    return logit(clipped)


def discover_public_models(
    public_dirs: list[Path], y_true: np.ndarray, n_test: int, min_auc: float, logger
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """
    Scans directories for paired (oof, test) predictions in npy, npz, or csv format.
    Validates shapes and computes OOF ROC-AUC. Deduplicates via hash of prefix.
    """
    discovered = {}
    seen_hashes = set()

    search_roots = []
    for d in public_dirs:
        if d.exists():
            search_roots.append(d)

    if not search_roots:
        logger.info(f"No existing public search roots found among: {[str(d) for d in public_dirs]}")
        return discovered

    logger.info(f"Scanning for public OOF datasets in: {[str(d) for d in search_roots]}...")

    for root in search_roots:
        # 1. npy pairs: oof_*.npy and test_*.npy (or pred_*.npy)
        for oof_file in sorted(root.rglob("oof_*.npy")):
            stem = oof_file.stem[4:]
            parent_name = oof_file.parent.name
            model_key = f"{parent_name}/{stem}" if parent_name not in ("public_oof", "data") else stem

            test_file = oof_file.with_name(f"test_{stem}.npy")
            if not test_file.exists():
                test_file = oof_file.with_name(f"pred_{stem}.npy")
            if not test_file.exists():
                continue

            try:
                oof_arr = np.load(oof_file, allow_pickle=False).astype(np.float64).ravel()
                test_arr = np.load(test_file, allow_pickle=False).astype(np.float64).ravel()
            except Exception:
                continue

            if len(oof_arr) != len(y_true) or len(test_arr) != n_test or not np.isfinite(oof_arr).all():
                continue

            # Deduplication hash of first 2,000 predictions
            fp = np.round(oof_arr[:2000], 5).tobytes()
            if fp in seen_hashes:
                continue
            seen_hashes.add(fp)

            score = roc_auc_score(y_true, oof_arr)
            if score >= min_auc:
                discovered[model_key] = (oof_arr, test_arr, score)
                logger.info(f"  + Ingested public model: {model_key:<35} | OOF AUC = {score:.5f}")

        # 2. npz archives with "oof" and "test" keys
        for npz_file in sorted(root.rglob("*.npz")):
            try:
                data = np.load(npz_file)
                if "oof" in data and ("test" in data or "test_pred" in data):
                    oof_arr = data["oof"].astype(np.float64).ravel()
                    test_key = "test" if "test" in data else "test_pred"
                    test_arr = data[test_key].astype(np.float64).ravel()
                    if len(oof_arr) == len(y_true) and len(test_arr) == n_test and np.isfinite(oof_arr).all():
                        fp = np.round(oof_arr[:2000], 5).tobytes()
                        if fp in seen_hashes:
                            continue
                        seen_hashes.add(fp)
                        score = roc_auc_score(y_true, oof_arr)
                        if score >= min_auc:
                            model_key = npz_file.stem
                            discovered[model_key] = (oof_arr, test_arr, score)
                            logger.info(f"  + Ingested public npz:   {model_key:<35} | OOF AUC = {score:.5f}")
            except Exception:
                continue

    return discovered


def main():
    logger = get_logger("TrackA_PublicStack")
    args = parse_args()

    paths = PathConfig()
    feature_cfg = FeatureConfig()

    logger.info("=" * 70)
    logger.info("TRACK A: PUBLIC DIVERSE OOF INGESTION & REGULARIZED LOGISTIC STACKING")
    logger.info("=" * 70)

    # 1. Load Ground Truth and Test IDs
    train_df = pd.read_csv(paths.train_path, usecols=[feature_cfg.target_col])
    y_true = resolve_binary_target(train_df[feature_cfg.target_col])
    test_df = pd.read_csv(paths.test_path, usecols=[feature_cfg.id_col])
    test_ids = test_df[feature_cfg.id_col].values
    n_test = len(test_ids)

    # 2. Load Internal 3 Fast GBDTs
    internal_models = ["lightgbm", "lightgbm_xt", "xgboost"]
    oof_dict = {}
    test_dict = {}

    for name in internal_models:
        oof_path = Path(paths.output_dir) / f"oof_preds_{name}.npy"
        test_path = Path(paths.output_dir) / f"test_preds_{name}.npy"
        if not oof_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing required internal 3-GBDT artifact: {oof_path} or {test_path}")
        oof_arr = np.load(oof_path).astype(np.float64).ravel()
        test_arr = np.load(test_path).astype(np.float64).ravel()
        oof_dict[name] = oof_arr
        test_dict[name] = test_arr
        auc = roc_auc_score(y_true, oof_arr)
        logger.info(f"Internal Anchor Model '{name:<15}': OOF ROC-AUC = {auc:.5f}")

    # 3. Discover Public OOFs
    search_dirs = [
        Path(args.public_dir),
        Path("data/public_oof"),
        Path("/kaggle/input"),
    ]
    public_models = discover_public_models(search_dirs, y_true, n_test, args.min_auc, logger)

    all_names = list(internal_models) + list(public_models.keys())
    for name, (o, t, _) in public_models.items():
        oof_dict[name] = o
        test_dict[name] = t

    logger.info("-" * 70)
    logger.info(
        f"Total Stacking Ensemble Candidates: {len(all_names)} models "
        f"({len(internal_models)} internal 3-GBDT + {len(public_models)} public diverse)"
    )
    logger.info("-" * 70)

    # 4. Construct Logit Feature Space
    Z_oof = np.column_stack([to_logit(oof_dict[m], args.epsilon) for m in all_names])
    Z_test = np.column_stack([to_logit(test_dict[m], args.epsilon) for m in all_names])

    # 5. Out-of-Fold 10-Fold Regularized Logistic Stacking
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y_true), dtype=np.float64)
    meta_test = np.zeros(n_test, dtype=np.float64)
    model_weights = np.zeros(len(all_names), dtype=np.float64)

    logger.info(f"Fitting 10-Fold L2-Regularized Logistic Regression (C={args.C})...")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(Z_oof, y_true)):
        clf = LogisticRegression(C=args.C, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(Z_oof[trn_idx], y_true[trn_idx])

        val_probs = clf.predict_proba(Z_oof[val_idx])[:, 1]
        meta_oof[val_idx] = val_probs

        test_probs = clf.predict_proba(Z_test)[:, 1]
        meta_test += test_probs / 10.0
        model_weights += clf.coef_[0] / 10.0

    meta_auc = roc_auc_score(y_true, meta_oof)
    logger.info("=" * 70)
    logger.info(f"TRACK A STACKING RESULT: Meta OOF ROC-AUC = {meta_auc:.6f}")
    logger.info("=" * 70)

    # Log Top Feature Weights
    sorted_idx = np.argsort(-model_weights)
    logger.info("Top Model Logistic Weights:")
    for idx in sorted_idx[:15]:
        logger.info(f"  * {all_names[idx]:<35}: coef = {model_weights[idx]:+.4f}")

    # 6. Monotonic Rank Scaling & Exact Leak Overrides
    logger.info("Applying monotonic rank scaling to preserve strict relative ordering...")
    test_ranked = rankdata(meta_test) / float(len(meta_test))

    out_dir = Path(paths.submissions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / args.output_name

    sub_df = pd.DataFrame({
        feature_cfg.id_col: test_ids,
        feature_cfg.target_col: test_ranked,
    })
    sub_df.to_csv(out_csv, index=False)

    # Apply Exact Matches
    try:
        orig_file = Path("data/raw/airline_passenger_satisfaction.csv")
        if orig_file.exists():
            orig_df = pd.read_csv(orig_file)
            test_raw = pd.read_csv(paths.test_path)
            postprocessor = ExactMatchPostprocessor(paths=paths, feature_cfg=feature_cfg)
            matches = postprocessor.find_exact_matches(test_raw, orig_df)
            if len(matches) > 0:
                postprocessor.apply_overrides(str(out_csv), matches)
                logger.info(f"Applied {len(matches)} exact ground-truth overrides to {out_csv}.")
    except Exception as e:
        logger.warning(f"Exact match postprocessing notice: {e}")

    # Submission Integrity Check
    final_sub = pd.read_csv(out_csv)
    assert len(final_sub) == n_test, f"Row count mismatch: {len(final_sub)} vs {n_test}"
    assert not final_sub[feature_cfg.target_col].isna().any(), "NaN values detected!"
    assert (final_sub[feature_cfg.target_col] >= 0.0).all() and (final_sub[feature_cfg.target_col] <= 1.0).all()
    logger.info(f"Submission Integrity PASSED: Exported to '{out_csv}'.")

    if args.update_champion and meta_auc >= 0.96149:
        sub_df = pd.read_csv(out_csv)
        sub_df.to_csv("submission.csv", index=False)
        logger.info(f"Successfully updated root 'submission.csv' with Track A champion (AUC: {meta_auc:.5f})!")


if __name__ == "__main__":
    main()
