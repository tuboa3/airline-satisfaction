"""
Production Runner for Argon Research: Asymmetric Tri-View Decoupling
and Cross-Fitted Hierarchical Residual Cascade for Kaggle S6E10
====================================================================
Implements:
1. Support-Snapped Route Lattice & Transductive Manifold (Vector 4).
2. Two-Part Hurdle Polytomous Graded Response Model (ZI-GRM) (Vector 6).
3. Asymmetric Tri-View Specialization (LightGBM Lattice, XGBoost Psychometrics,
   LightGBM Extra-Trees Haar Rotations) (Vector 2).
4. Composite Hybrid Objective (Damped BCE + Smooth WMW AUC Surrogate) (Vector 1).
5. 3-Stage Hierarchical Residual Cascade (Stage 1 LGBM -> Stage 2 XGB -> Stage 3 LGBM_XT) (Vector 5).
6. Strictly Monotonic Power-Logit Rank Combinator & Laplacian Smoothing (Vector 5).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scipy.special as sp
import lightgbm as lgb
import xgboost as xgb

from src.config import FeatureConfig, PathConfig
from src.dataset import DatasetIngestion
from src.ensemble import (
    StrictlyMonotonicPowerLogitStacker,
    apply_transductive_knn_laplacian_smoothing,
    meta_stack_parallel_families,
)
from src.features import (
    AsymmetricViewBuilder,
    HurdleGradedResponseModel,
    TransductiveManifoldAndDGPInverter,
    apply_structured_survey_dropout,
    build_original_prior_teacher_and_stats,
    engineer_s6e10_domain_features,
)
from src.models import (
    HierarchicalResidualGBDTCascade,
    GramSchmidtCascadeDecoupler,
    train_catboost_dual_representation,
    train_pytabkit_deep_tabular,
    select_certainty_masked_pseudo_labels,
)
from src.postprocess import ExactMatchPostprocessor
from src.utils import detect_hardware, get_logger, resolve_binary_target, timer


def parse_args():
    parser = argparse.ArgumentParser(description="Argon Research: Remediated Parallel 5-Family & Cascade Engine")
    parser.add_argument(
        "--architecture",
        type=str,
        default="parallel_5family",
        choices=["parallel_5family", "cascade"],
        help="Architecture: 'parallel_5family' (5-family remediated champion) or 'cascade' (3-stage residual cascade)",
    )
    parser.add_argument(
        "--enable_lgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Leaf-Wise LightGBM family",
    )
    parser.add_argument(
        "--enable_xgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Depth-Wise XGBoost family",
    )
    parser.add_argument(
        "--enable_catboost",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable All-Categorical Oblivious Tree CatBoost family",
    )
    parser.add_argument(
        "--enable_neural",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Deep Tabular Neural families (RealMLP-TD & TabM)",
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--beta_12", type=float, default=0.72, help="Stage 1 -> Stage 2 margin handoff (cascade mode)")
    parser.add_argument("--beta_23", type=float, default=0.75, help="Stage 2 -> Stage 3 init_score handoff (cascade mode)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores/threads (-1 to use all cores)")
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GPU acceleration across all available NVIDIA GPUs",
    )
    parser.add_argument(
        "--use_orig_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concatenate adversarial density-weighted original data (cascade mode)",
    )
    parser.add_argument(
        "--use_distillation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cross-fitted two-teacher logit distillation (cascade mode)",
    )
    parser.add_argument(
        "--use_pseudo_labels",
        action="store_true",
        help="Enable Bayesian certainty-masked pseudo-labeling with survey dropout",
    )
    parser.add_argument("--two_pass", action="store_true", help="Two-pass outer margin handoff (cascade mode)")
    parser.add_argument(
        "--fold_workers",
        type=int,
        default=0,
        help="Number of concurrent folds during cross-fitting (0 = auto based on CPU cores and GPUs)",
    )
    parser.add_argument("--use_laplacian", action="store_true", help="Apply Graph Laplacian test manifold smoothing")
    parser.add_argument("--fast", action="store_true", help="Fast smoke-test mode on a lightweight sample")
    return parser.parse_args()


def run_parallel_5family_pipeline(
    args,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame | None,
    orig_files: list[str],
    ingestion: DatasetIngestion,
    paths: PathConfig,
    feature_cfg: FeatureConfig,
    logger: logging.Logger,
    n_cpus: int,
    use_gpu: bool,
    num_gpus: int,
    gpu_devs: list[str],
):
    """
    Executes the Remediated Parallel Five-Family Architecture with Isolated Original-Prior Teacher
    per forensic specifications in argon-research.md (Kaggle S6E10 Rank 1 Frontier).
    """
    logger.info("=" * 80)
    logger.info("STARTING REMEDIATED PARALLEL FIVE-FAMILY ARCHITECTURE (S6E10)")
    logger.info(f"Folds: {args.folds} | Seed: {args.seed} | Fast Mode: {args.fast}")
    logger.info(f"Hardware Allocation: Active CPU Cores={n_cpus} | GPU Acceleration={use_gpu} (Total GPUs={num_gpus}: {gpu_devs})")
    logger.info("=" * 80)

    if args.fast:
        logger.info("[FAST MODE] Subsampling train to 1,200 rows and test to 600 rows for smoke test.")
        train_df = train_df.iloc[:1200].copy().reset_index(drop=True)
        test_df = test_df.iloc[:600].copy().reset_index(drop=True)
        if orig_df is not None:
            orig_df = orig_df.iloc[:600].copy().reset_index(drop=True)
        args.folds = 2

    # Tier 1: Dedicated Teacher on Original Dataset + Bayes Prior Shift (+0.45611)
    if orig_df is not None:
        logger.info("=" * 80)
        logger.info("TIER 1: ISOLATING ORIGINAL PRIOR TEACHER & MULTI-WAY BAYESIAN PRIORS")
        logger.info("=" * 80)
        train_df, test_df = build_original_prior_teacher_and_stats(
            df_syn_train=train_df,
            df_syn_test=test_df,
            df_orig=orig_df,
            target_col=feature_cfg.target_col,
            T_orig=1.15,
            seed=args.seed,
            fast_mode=args.fast,
            n_jobs=n_cpus,
        )
        logger.info("Tier 1 features generated: orig_teacher_logit_adj, orig_teacher_prob_adj, orig_bayes_prior_*")

    # Tier 2: High-Leverage Domain Features (Route ID, Psychometrics, Aux Tasks)
    logger.info("=" * 80)
    logger.info("TIER 2: ENGINEERING S6E10 DOMAIN FEATURES (ROUTE ID, PSYCHOMETRICS, AUX TASKS)")
    logger.info("=" * 80)
    train_df, test_df = engineer_s6e10_domain_features(
        df_train=train_df,
        df_test=test_df,
        target_col=feature_cfg.target_col,
        n_splits=args.folds,
        seed=args.seed,
        fast_mode=args.fast,
        n_jobs=n_cpus,
    )
    logger.info(f"Tier 2 complete. Train feature shape: {train_df.shape}, Test feature shape: {test_df.shape}")

    y_train = resolve_binary_target(train_df[feature_cfg.target_col]).astype(int)
    test_ids = test_df[feature_cfg.id_col].to_numpy()

    # Column separation
    cat_candidates = ["Gender", "Customer Type", "Type of Travel", "Class", "route_id", "route_class_id", "route_travel_id"]
    cat_cols = [c for c in cat_candidates if c in train_df.columns]
    excluded_cols = {"id", "Unnamed: 0", feature_cfg.target_col, "is_original", "sample_weight"}
    feature_cols = [c for c in train_df.columns if c not in excluded_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    base_features = [
        c for c in [
            "Gender", "Customer Type", "Age", "Type of Travel", "Class",
            "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
            "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
            "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
            "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
            "Departure Delay in Minutes", "Arrival Delay in Minutes"
        ] if c in train_df.columns
    ]
    eng_num = [c for c in num_cols if c not in base_features]

    # Convert categoricals to unified CategoricalDtype for LightGBM/XGBoost
    for c in cat_cols:
        cats = pd.concat([train_df[c].astype(str), test_df[c].astype(str)], axis=0).unique()
        cat_type = pd.CategoricalDtype(categories=cats)
        train_df[c] = train_df[c].astype(str).astype(cat_type)
        test_df[c] = test_df[c].astype(str).astype(cat_type)

    # Active model families
    active_families = []
    if args.enable_lgb:
        active_families.append("LightGBM")
    if args.enable_xgb:
        active_families.append("XGBoost")
    if args.enable_catboost:
        active_families.append("CatBoost")
    if args.enable_neural:
        active_families.append("RealMLP-TD")
        active_families.append("TabM")

    oof_logits = {fam: np.zeros(len(train_df), dtype=np.float64) for fam in active_families}
    test_logits = {fam: np.zeros(len(test_df), dtype=np.float64) for fam in active_families}

    # Device allocation
    xgb_dev = "cuda:0" if use_gpu else "cpu"
    cb_task_type = "GPU" if use_gpu else "CPU"
    neural_dev = "cuda:1" if (use_gpu and num_gpus > 1) else ("cuda:0" if use_gpu else "cpu")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    logger.info("=" * 80)
    logger.info(f"COMMENCING {args.folds}-FOLD CROSS-FITTING OF PARALLEL 5-FAMILY ENSEMBLE")
    logger.info(f"Active Families: {active_families}")
    logger.info(f"Hardware Mapping: LGBM (4 CPU cores), XGB ({xgb_dev}), CatBoost ({cb_task_type}), Neural ({neural_dev})")
    logger.info("=" * 80)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        logger.info(f"--- Processing Fold {fold + 1}/{args.folds} ---")
        X_tr_f = train_df[feature_cols].iloc[trn_idx]
        y_tr_f = y_train[trn_idx]
        X_va_f = train_df[feature_cols].iloc[val_idx]
        y_va_f = y_train[val_idx]
        X_te_f = test_df[feature_cols]

        # 1. Leaf-Wise LightGBM
        if args.enable_lgb:
            model_lgb = lgb.LGBMClassifier(
                n_estimators=3000 if not args.fast else 40,
                learning_rate=0.035 if not args.fast else 0.1,
                num_leaves=95 if not args.fast else 31,
                colsample_bytree=0.80,
                subsample=0.80,
                subsample_freq=1,
                min_child_samples=45,
                reg_alpha=1.5,
                reg_lambda=5.0,
                force_row_wise=True,
                n_jobs=n_cpus,
                random_state=args.seed + fold,
                verbose=-1,
            )
            model_lgb.fit(
                X_tr_f, y_tr_f,
                eval_set=[(X_va_f, y_va_f)],
                callbacks=[lgb.early_stopping(100 if not args.fast else 15, verbose=False)],
            )
            p_va = np.clip(model_lgb.predict_proba(X_va_f)[:, 1], 1e-6, 1.0 - 1e-6)
            p_te = np.clip(model_lgb.predict_proba(X_te_f)[:, 1], 1e-6, 1.0 - 1e-6)
            oof_logits["LightGBM"][val_idx] = sp.logit(p_va)
            test_logits["LightGBM"] += sp.logit(p_te) / args.folds
            auc_lgb = roc_auc_score(y_va_f, p_va)
            logger.info(f"Fold {fold+1} LightGBM: AUC = {auc_lgb:.5f}")

        # 2. Depth-Wise XGBoost
        if args.enable_xgb:
            model_xgb = xgb.XGBClassifier(
                n_estimators=3000 if not args.fast else 40,
                learning_rate=0.035 if not args.fast else 0.1,
                max_depth=7 if not args.fast else 4,
                tree_method="hist",
                device=xgb_dev,
                subsample=0.80,
                colsample_bytree=0.80,
                min_child_weight=25.0,
                reg_alpha=2.0,
                reg_lambda=8.0,
                eval_metric="auc",
                enable_categorical=True,
                random_state=args.seed + fold,
                n_jobs=n_cpus,
            )
            try:
                model_xgb.fit(
                    X_tr_f, y_tr_f,
                    eval_set=[(X_va_f, y_va_f)],
                    verbose=False,
                )
            except Exception as e:
                if xgb_dev != "cpu":
                    logger.warning(f"XGBoost GPU note ({e}). Falling back to multi-core CPU...")
                    model_xgb.set_params(device="cpu")
                    model_xgb.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
                else:
                    raise e
            p_va = np.clip(model_xgb.predict_proba(X_va_f)[:, 1], 1e-6, 1.0 - 1e-6)
            p_te = np.clip(model_xgb.predict_proba(X_te_f)[:, 1], 1e-6, 1.0 - 1e-6)
            oof_logits["XGBoost"][val_idx] = sp.logit(p_va)
            test_logits["XGBoost"] += sp.logit(p_te) / args.folds
            auc_xgb = roc_auc_score(y_va_f, p_va)
            logger.info(f"Fold {fold+1} XGBoost:  AUC = {auc_xgb:.5f}")

        # 3. All-Categorical Oblivious Tree CatBoost
        if args.enable_catboost:
            cb_va_z, cb_te_z = train_catboost_dual_representation(
                X_tr=X_tr_f,
                y_tr=y_tr_f,
                X_va=X_va_f,
                y_va=y_va_f,
                X_te=X_te_f,
                base_features=base_features,
                engineered_num_features=eng_num,
                seed=args.seed + fold,
                task_type=cb_task_type,
                iterations=3500 if not args.fast else 40,
                learning_rate=0.035 if not args.fast else 0.1,
                early_stopping_rounds=150 if not args.fast else 15,
            )
            oof_logits["CatBoost"][val_idx] = cb_va_z
            test_logits["CatBoost"] += cb_te_z / args.folds
            auc_cb = roc_auc_score(y_va_f, sp.expit(cb_va_z))
            logger.info(f"Fold {fold+1} CatBoost: AUC = {auc_cb:.5f}")

        # 4 & 5. Deep Tabular Neural Models (RealMLP-TD & TabM)
        if args.enable_neural:
            rmlp_va_z, rmlp_te_z, tabm_va_z, tabm_te_z = train_pytabkit_deep_tabular(
                X_tr=X_tr_f,
                y_tr=y_tr_f,
                X_va=X_va_f,
                y_va=y_va_f,
                X_te=X_te_f,
                num_cols=num_cols,
                cat_cols=cat_cols,
                seed=args.seed + fold,
                device=neural_dev,
                fast_mode=args.fast,
            )
            oof_logits["RealMLP-TD"][val_idx] = rmlp_va_z
            test_logits["RealMLP-TD"] += rmlp_te_z / args.folds
            oof_logits["TabM"][val_idx] = tabm_va_z
            test_logits["TabM"] += tabm_te_z / args.folds
            auc_rmlp = roc_auc_score(y_va_f, sp.expit(rmlp_va_z))
            auc_tabm = roc_auc_score(y_va_f, sp.expit(tabm_va_z))
            logger.info(f"Fold {fold+1} RealMLP:  AUC = {auc_rmlp:.5f} | TabM: AUC = {auc_tabm:.5f}")

    # Standalone model evaluation
    logger.info("=" * 80)
    logger.info("PARALLEL 5-FAMILY STANDALONE OOF PERFORMANCE")
    auc_per_model = {}
    for fam in active_families:
        auc_f = float(roc_auc_score(y_train, sp.expit(oof_logits[fam])))
        auc_per_model[fam] = auc_f
        logger.info(f"  {fam:<15}: OOF ROC-AUC = {auc_f:.6f}")

    # Tier 4: Temperature-Calibrated L2 Meta-Stacker
    logger.info("=" * 80)
    logger.info("TIER 4: TEMPERATURE-CALIBRATED L2 META-STACKER ON LOG-ODDS")
    logger.info("=" * 80)
    champ_oof_probs, champ_test_probs, stacked_auc, r_mean, n_eff, corr_mat = meta_stack_parallel_families(
        oof_logits_dict=oof_logits,
        test_logits_dict=test_logits,
        y=y_train,
        C_reg=0.25,
        n_splits=args.folds,
        seed=args.seed,
    )
    logger.info(f"Pairwise Off-Diagonal Mean Correlation: r_mean = {r_mean:.5f}")
    logger.info(f"Effective Ensemble Size (N_eff): {n_eff:.4f}")
    logger.info(f"CHAMPION STACKED OOF ROC-AUC: {stacked_auc:.6f}")
    logger.info(f"Target Barrier: 0.962430 | Target Gate: 0.963000")

    # Exact-Match Deterministic Ground-Truth Overrides
    try:
        if orig_files:
            orig_ref = ingestion._standardize_columns(pd.read_csv(orig_files[0]), is_original=True)
            postproc = ExactMatchPostprocessor(paths=paths, feature_cfg=feature_cfg)
            matches = postproc.find_exact_matches(test_df, orig_ref)
            if len(matches) > 0:
                logger.info(f"Applying {len(matches)} exact ground-truth test matches.")
                test_sub_df = pd.DataFrame({feature_cfg.id_col: test_ids, feature_cfg.target_col: champ_test_probs})
                for row_id, match_target in matches.items():
                    test_sub_df.loc[test_sub_df[feature_cfg.id_col] == row_id, feature_cfg.target_col] = match_target
                champ_test_probs = test_sub_df[feature_cfg.target_col].to_numpy()
    except Exception as e:
        logger.warning(f"Notice on exact-match overrides: {e}")

    # Submission Export & Integrity Verification
    os.makedirs(paths.submissions_dir, exist_ok=True)
    os.makedirs(paths.output_dir, exist_ok=True)

    sub_df = pd.DataFrame({
        feature_cfg.id_col: test_ids,
        feature_cfg.target_col: champ_test_probs,
    })
    sub_path = os.path.join(paths.submissions_dir, "submission_parallel_5family.csv")
    sub_df.to_csv(sub_path, index=False)
    sub_df.to_csv("submission.csv", index=False)

    verify_df = pd.read_csv("submission.csv")
    assert len(verify_df) == len(test_ids), "Submission integrity check: row count mismatch!"
    assert not verify_df[feature_cfg.target_col].isna().any(), "Submission integrity check: NaNs detected!"
    assert (verify_df[feature_cfg.target_col] >= 0.0).all() and (verify_df[feature_cfg.target_col] <= 1.0).all(), "Bounds violation!"

    # Save summary report
    summary = {
        "architecture": "Remediated_Parallel_Five_Family",
        "timestamp": pd.Timestamp.now().isoformat(),
        "folds": args.folds,
        "fast_mode": args.fast,
        "oof_roc_auc_per_family": auc_per_model,
        "champion_stacked_auc": float(stacked_auc),
        "collinearity_metrics": {
            "r_mean": float(r_mean),
            "effective_ensemble_size_n_eff": float(n_eff),
        },
        "target_barrier_rank1": 0.962430,
        "target_gate": 0.963000,
        "barrier_surpassed": bool(stacked_auc >= 0.962430),
    }

    logits_path = os.path.join(paths.output_dir, "argon_5family_logits.npz")
    save_dict = {f"oof_logits_{k}": v for k, v in oof_logits.items()}
    save_dict.update({f"test_logits_{k}": v for k, v in test_logits.items()})
    save_dict["y_train"] = y_train
    save_dict["champ_oof_probs"] = champ_oof_probs
    save_dict["champ_test_probs"] = champ_test_probs
    np.savez_compressed(logits_path, **save_dict)

    with open(os.path.join(paths.output_dir, "argon_evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"PARALLEL 5-FAMILY PIPELINE COMPLETE: Champion OOF ROC-AUC = {stacked_auc:.6f}")
    logger.info(f"Outputs serialized to: 'submission.csv', '{logits_path}', and '{paths.output_dir}/argon_evaluation_summary.json'")
    logger.info("=" * 80)


def main():
    logger = get_logger("ArgonCascadeEngine")
    args = parse_args()
    paths = PathConfig()
    feature_cfg = FeatureConfig()

    primary_dev, num_cpus, num_gpus, gpu_devs = detect_hardware()
    n_cpus = num_cpus if (args.n_jobs <= 0 or args.n_jobs > num_cpus) else args.n_jobs
    use_gpu = args.gpu and (num_gpus > 0)

    logger.info("=" * 80)
    logger.info(f"STARTING ARGON RESEARCH ENGINE: ARCHITECTURE = {args.architecture.upper()} (KAGGLE S6E10)")
    logger.info(f"Folds: {args.folds} | Seed: {args.seed} | Fast Mode: {args.fast}")
    logger.info(f"Hardware Allocation: Active CPU Cores={n_cpus} | GPU Acceleration={use_gpu} (Total GPUs={num_gpus}: {gpu_devs})")
    logger.info(f"Handoff beta_12: {args.beta_12} | beta_23: {args.beta_23} | Two-Pass Mode: {args.two_pass}")
    logger.info(f"Original Data Weighting: {args.use_orig_data} | Distillation: {args.use_distillation}")
    logger.info(f"Transductive Pseudo-Labels: {args.use_pseudo_labels} | Laplacian: {args.use_laplacian}")
    logger.info("=" * 80)

    # 1. Dataset Ingestion
    ingestion = DatasetIngestion(paths=paths, feature_cfg=feature_cfg)
    orig_files = ingestion.discover_original_files()
    raw_res = ingestion.load_and_prepare()
    if len(raw_res) == 4:
        train_df, test_df, sample_weights, orig_df = raw_res
    else:
        train_df, test_df, orig_df = raw_res[0], raw_res[1], raw_res[-1]

    if orig_df is None and orig_files:
        orig_df = ingestion._standardize_columns(pd.read_csv(orig_files[0]), is_original=True)
        logger.info(f"Ingested original dataset: {len(orig_df)} rows")

    if args.architecture == "parallel_5family":
        run_parallel_5family_pipeline(
            args=args,
            train_df=train_df,
            test_df=test_df,
            orig_df=orig_df,
            orig_files=orig_files,
            ingestion=ingestion,
            paths=paths,
            feature_cfg=feature_cfg,
            logger=logger,
            n_cpus=n_cpus,
            use_gpu=use_gpu,
            num_gpus=num_gpus,
            gpu_devs=gpu_devs,
        )
        return

    if args.fast:
        logger.info("[FAST MODE] Subsampling train to 1,200 rows and test to 600 rows for smoke test.")
        train_df = train_df.iloc[:1200].copy()
        test_df = test_df.iloc[:600].copy()
        if orig_df is not None:
            orig_df = orig_df.iloc[:600].copy()
        args.folds = 2
        num_boost_round = 30
        early_stopping_rounds = 10
    else:
        num_boost_round = 2200
        early_stopping_rounds = 120

    y_train = train_df[feature_cfg.target_col].to_numpy(dtype=np.float32)
    test_ids = test_df[feature_cfg.id_col].to_numpy()

    # 2. Transductive Manifold & Synthetic DGP Inversion (Vector 4)
    logger.info("Fitting Transductive Manifold & Snapping Flight Distance Route Lattice...")
    dgp_inverter = TransductiveManifoldAndDGPInverter(seed=args.seed)
    dgp_inverter.fit_full_manifold(train_df, test_df, orig_df)

    dgp_train = dgp_inverter.transform(train_df)
    dgp_test = dgp_inverter.transform(test_df)
    logger.info(f"Transductive Manifold generated {dgp_train.shape[1]} representations.")

    # 2.1 Adversarial Density Ratio Weights for Original Data Concatenation (Section 5.1)
    dgp_orig = None
    y_orig = None
    w_orig_fold = None
    if orig_df is not None and args.use_orig_data:
        logger.info("Computing Radon-Nikodym Adversarial Density Ratio weights for original data...")
        if not args.fast:
            w_orig = dgp_inverter.compute_adversarial_orig_weights(train_df, orig_df)
        else:
            w_orig = np.ones(len(orig_df), dtype=np.float32)

        w_orig_fold = 0.60 * w_orig
        dgp_orig = dgp_inverter.transform(orig_df)
        y_orig = resolve_binary_target(orig_df[feature_cfg.target_col]).astype(np.float32)
        logger.info(
            f"Prepared original dataset concatenation: {len(orig_df)} rows with mean weight {w_orig_fold.mean():.4f} (damping=0.60)."
        )

    # 2.2 Section 2.3 Cross-Fitted Temperature-Scaled Logit Distillation
    y_train_soft = y_train.copy()
    teacher_path = os.path.join(paths.output_dir, "oof_preds_lightgbm.npy")
    if args.use_distillation and os.path.exists(teacher_path):
        try:
            p_teacher_full = np.load(teacher_path)
            if len(p_teacher_full) >= len(train_df):
                p_teacher = p_teacher_full[:len(train_df)].astype(np.float64)
                z_teacher = sp.logit(np.clip(p_teacher, 1e-6, 1.0 - 1e-6))
                z_orig = dgp_train["orig_oracle_logit"].to_numpy(dtype=np.float64)
                q_target = np.clip(
                    0.78 * sp.expit(z_teacher / 1.25) + 0.22 * sp.expit(z_orig / 1.10),
                    0.003,
                    0.997,
                )
                y_train_soft = (1.0 - 0.40) * y_train + 0.40 * q_target.astype(np.float32)
                logger.info(
                    "Constructed Section 2.3 Two-Teacher Distillation targets (alpha_KD=0.40, T_syn=1.25, T_orig=1.10)."
                )
        except Exception as e:
            logger.warning(f"Notice on distillation teacher: {e}. Falling back to clean targets.")

    # 2.3 Section 4.2 & 4.3 Bayesian Certainty-Masked Soft Pseudo-Labels with Survey Dropout
    df_pl, dgp_pl, y_pl, w_pl = None, None, None, None
    if args.use_pseudo_labels:
        lgb_te_path = os.path.join(paths.output_dir, "test_preds_lightgbm.npy")
        xgb_te_path = os.path.join(paths.output_dir, "test_preds_xgboost.npy")
        xt_te_path = os.path.join(paths.output_dir, "test_preds_lightgbm_xt.npy")
        if os.path.exists(lgb_te_path) and os.path.exists(xgb_te_path) and os.path.exists(xt_te_path):
            try:
                p_lgb = np.load(lgb_te_path)[:len(test_df)]
                p_xgb = np.load(xgb_te_path)[:len(test_df)]
                p_xt = np.load(xt_te_path)[:len(test_df)]
                admitted_mask, soft_targets, pl_weights = select_certainty_masked_pseudo_labels(
                    [p_lgb, p_xgb, p_xt],
                    weights=[0.38, 0.34, 0.28],
                    conf_threshold=0.985,
                    max_std_threshold=0.0045,
                    max_pl_weight=0.35,
                )
                if admitted_mask.sum() > 0:
                    logger.info(
                        f"Bayesian Certainty Mask admitted {admitted_mask.sum()}/{len(test_df)} "
                        f"({admitted_mask.sum()/len(test_df)*100:.2f}%) pseudo-labeled instances."
                    )
                    df_pl_raw = test_df.iloc[admitted_mask].copy()
                    df_pl = apply_structured_survey_dropout(df_pl_raw, dropout_rate=0.04, seed=args.seed)
                    dgp_pl = dgp_test.iloc[admitted_mask].copy()
                    y_pl = soft_targets.astype(np.float32)
                    w_pl = pl_weights.astype(np.float32)
            except Exception as e:
                logger.warning(f"Could not load preliminary pseudo-labels: {e}")

    # Arrays to store cross-fitted OOF and Test predictions
    oof_logits_1 = np.zeros(len(train_df), dtype=np.float64)
    oof_logits_2 = np.zeros(len(train_df), dtype=np.float64)
    oof_logits_3 = np.zeros(len(train_df), dtype=np.float64)

    test_logits_1 = np.zeros(len(test_df), dtype=np.float64)
    test_logits_2 = np.zeros(len(test_df), dtype=np.float64)
    test_logits_3 = np.zeros(len(test_df), dtype=np.float64)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # 3. Cross-Validation Execution (Single-Pass or Two-Pass with High-Efficiency Concurrency)
    logger.info("=" * 80)
    logger.info(f"COMMENCING {args.folds}-FOLD LEAK-FREE RESIDUAL CASCADE (TWO_PASS={args.two_pass})")
    logger.info("=" * 80)

    ram_gb = (
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        if hasattr(os, "sysconf")
        else 16.0
    )
    cache_fold_views = (ram_gb >= 16.0) and not args.fast

    if args.fold_workers > 0:
        p1_workers = min(args.fold_workers, args.folds)
        p2_workers = min(args.fold_workers, args.folds)
    else:
        p1_workers = max(1, min(n_cpus, args.folds, 4 if n_cpus >= 4 else 2))
        if use_gpu and num_gpus > 1:
            p2_workers = min(num_gpus, n_cpus, args.folds)
        elif use_gpu and num_gpus == 1:
            p2_workers = 1
        else:
            p2_workers = max(1, min(n_cpus, args.folds, 2))

    p1_threads = max(1, n_cpus // p1_workers)
    p2_threads = max(1, n_cpus // p2_workers)
    logger.info(
        f"Concurrency Strategy: Pass 1 Workers={p1_workers} ({p1_threads} threads/fold) | "
        f"Pass 2 Workers={p2_workers} ({p2_threads} threads/fold) | View Caching={cache_fold_views}"
    )

    def _prepare_fold_views(fold: int, trn_idx: np.ndarray, val_idx: np.ndarray, fold_irt_dev: Optional[str]):
        df_tr = train_df.iloc[trn_idx].copy()
        y_tr_clean = y_train[trn_idx].copy()
        y_tr = y_train_soft[trn_idx].copy()
        dgp_tr = dgp_train.iloc[trn_idx].copy()

        df_val = train_df.iloc[val_idx].copy()
        y_val = y_train[val_idx].copy()
        dgp_val = dgp_train.iloc[val_idx].copy()

        df_tr_list = [df_tr]
        dgp_tr_list = [dgp_tr]
        y_tr_list = [y_tr]
        w_tr_list = [np.ones(len(df_tr), dtype=np.float32)]

        if orig_df is not None and args.use_orig_data and dgp_orig is not None:
            df_tr_list.append(orig_df)
            dgp_tr_list.append(dgp_orig)
            y_tr_list.append(y_orig)
            w_tr_list.append(w_orig_fold)

        if df_pl is not None:
            df_tr_list.append(df_pl)
            dgp_tr_list.append(dgp_pl)
            y_tr_list.append(y_pl)
            w_tr_list.append(w_pl)

        df_tr_aug = pd.concat(df_tr_list, axis=0, ignore_index=True)
        dgp_tr_aug = pd.concat(dgp_tr_list, axis=0, ignore_index=True)
        y_tr_aug = np.concatenate(y_tr_list)
        w_tr_aug = np.concatenate(w_tr_list)

        irt_engine = HurdleGradedResponseModel(device=fold_irt_dev)
        irt_engine.fit(df_tr, y_tr_clean)
        irt_tr_aug = irt_engine.transform(df_tr_aug)
        irt_val = irt_engine.transform(df_val)
        irt_te = irt_engine.transform(test_df)

        view_builder = AsymmetricViewBuilder(n_haar_projections=28, seed=args.seed + fold)
        view_builder.fit_fold(df_tr, y_tr_clean, dgp_tr)

        views_tr = view_builder.transform_views(df_tr_aug, dgp_tr_aug, irt_tr_aug, is_train_fold=True, y_tr=y_tr_aug)
        views_val = view_builder.transform_views(df_val, dgp_val, irt_val, is_train_fold=False)
        views_te = view_builder.transform_views(test_df, dgp_test, irt_te, is_train_fold=False)

        return views_tr, views_val, views_te, y_tr_aug, w_tr_aug, y_val

    fold_splits = list(enumerate(skf.split(train_df, y_train), start=1))
    stage1_models = {}
    fold_views_cache = {}

    if args.two_pass:
        logger.info("[TWO-PASS EXECUTION] Commencing Pass 1: Cross-fitting Stage 1 (LightGBM on View A)...")

        def _run_pass1_fold(fold_info):
            fold, trn_idx, val_idx = fold_info
            fold_gpu = gpu_devs[(fold - 1) % num_gpus] if (use_gpu and num_gpus > 0) else None
            fold_irt = gpu_devs[fold % num_gpus] if (use_gpu and num_gpus > 1) else fold_gpu

            views_tr, views_val, views_te, y_tr_aug, w_tr_aug, y_val = _prepare_fold_views(
                fold, trn_idx, val_idx, fold_irt
            )

            cascade = HierarchicalResidualGBDTCascade(
                beta_12=args.beta_12,
                beta_23=args.beta_23,
                seed=args.seed + fold,
                num_threads=p1_threads,
                gpu_device=fold_gpu,
            )
            p1_res = cascade.fit_predict_fold(
                views_tr=views_tr,
                y_tr_soft=y_tr_aug,
                w_tr=w_tr_aug,
                views_val=views_val,
                y_val_hard=y_val,
                views_te=views_te,
                stage1_only=True,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            cached_data = (views_tr, views_val, views_te, y_tr_aug, w_tr_aug, y_val) if cache_fold_views else None
            return fold, val_idx, p1_res, cached_data

        if p1_workers > 1:
            logger.info(f"Dispatching Pass 1 across {p1_workers} concurrent workers...")
            with ThreadPoolExecutor(max_workers=p1_workers) as executor:
                futures = {executor.submit(_run_pass1_fold, (f, trn, val)): f for f, (trn, val) in fold_splits}
                for fut in as_completed(futures):
                    fold, val_idx, p1_res, cached_data = fut.result()
                    oof_logits_1[val_idx] = p1_res["val_logits_1"]
                    test_logits_1 += p1_res["te_logits_1"] / args.folds
                    stage1_models[fold] = p1_res.get("model_a")
                    if cached_data is not None:
                        fold_views_cache[fold] = cached_data
                    auc_1 = roc_auc_score(y_train[val_idx], p1_res["val_logits_1"])
                    logger.info(f"Pass 1 - Fold {fold} Stage 1 AUC = {auc_1:.5f}")
        else:
            for fold, (trn_idx, val_idx) in fold_splits:
                with timer(f"Pass 1 - Fold {fold}/{args.folds} (Stage 1)"):
                    _, _, p1_res, cached_data = _run_pass1_fold((fold, trn_idx, val_idx))
                    oof_logits_1[val_idx] = p1_res["val_logits_1"]
                    test_logits_1 += p1_res["te_logits_1"] / args.folds
                    stage1_models[fold] = p1_res.get("model_a")
                    if cached_data is not None:
                        fold_views_cache[fold] = cached_data
                    auc_1 = roc_auc_score(y_train[val_idx], p1_res["val_logits_1"])
                    logger.info(f"Pass 1 - Fold {fold} Stage 1 AUC = {auc_1:.5f}")

        logger.info(f"Pass 1 Complete: Total Stage 1 OOF ROC-AUC = {roc_auc_score(y_train, oof_logits_1):.6f}")
        logger.info("[TWO-PASS EXECUTION] Commencing Pass 2: Stages 2 & 3 with true out-of-fold margin handoffs...")

    def _run_pass2_fold(fold_info):
        fold, trn_idx, val_idx = fold_info
        fold_gpu = gpu_devs[(fold - 1) % num_gpus] if (use_gpu and num_gpus > 0) else None
        fold_irt = gpu_devs[fold % num_gpus] if (use_gpu and num_gpus > 1) else fold_gpu

        if cache_fold_views and fold in fold_views_cache:
            views_tr, views_val, views_te, y_tr_aug, w_tr_aug, y_val = fold_views_cache[fold]
        else:
            views_tr, views_val, views_te, y_tr_aug, w_tr_aug, y_val = _prepare_fold_views(
                fold, trn_idx, val_idx, fold_irt
            )

        cascade = HierarchicalResidualGBDTCascade(
            beta_12=args.beta_12,
            beta_23=args.beta_23,
            seed=args.seed + fold,
            num_threads=p2_threads,
            gpu_device=fold_gpu,
        )

        stage1_oof_tr = oof_logits_1[trn_idx] if args.two_pass else None
        stage1_val_m = oof_logits_1[val_idx] if args.two_pass else None
        stage1_te_m = test_logits_1 if args.two_pass else None
        stage1_m = stage1_models.get(fold, None)

        preds_dict = cascade.fit_predict_fold(
            views_tr=views_tr,
            y_tr_soft=y_tr_aug,
            w_tr=w_tr_aug,
            views_val=views_val,
            y_val_hard=y_val,
            views_te=views_te,
            stage1_oof_tr_margin=stage1_oof_tr,
            stage1_val_margin=stage1_val_m,
            stage1_te_margin=stage1_te_m,
            stage1_model=stage1_m,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        return fold, val_idx, preds_dict

    if p2_workers > 1:
        logger.info(f"Dispatching Pass 2 across {p2_workers} concurrent workers (dual GPUs & CPU cores active)...")
        with ThreadPoolExecutor(max_workers=p2_workers) as executor:
            futures = {executor.submit(_run_pass2_fold, (f, trn, val)): f for f, (trn, val) in fold_splits}
            for fut in as_completed(futures):
                fold, val_idx, preds_dict = fut.result()
                if not args.two_pass:
                    oof_logits_1[val_idx] = preds_dict["val_logits_1"]
                    test_logits_1 += preds_dict["te_logits_1"] / args.folds

                oof_logits_2[val_idx] = preds_dict["val_logits_2"]
                oof_logits_3[val_idx] = preds_dict["val_logits_3"]

                test_logits_2 += preds_dict["te_logits_2"] / args.folds
                test_logits_3 += preds_dict["te_logits_3"] / args.folds

                auc_1 = roc_auc_score(y_train[val_idx], oof_logits_1[val_idx])
                auc_2 = roc_auc_score(y_train[val_idx], preds_dict["val_logits_2"])
                auc_3 = roc_auc_score(y_train[val_idx], preds_dict["val_logits_3"])
                logger.info(f"Fold {fold} AUCs: Stage 1 = {auc_1:.5f} | Stage 2 = {auc_2:.5f} | Stage 3 = {auc_3:.5f}")
    else:
        for fold, (trn_idx, val_idx) in fold_splits:
            with timer(f"Fold {fold}/{args.folds}"):
                _, _, preds_dict = _run_pass2_fold((fold, trn_idx, val_idx))
                if not args.two_pass:
                    oof_logits_1[val_idx] = preds_dict["val_logits_1"]
                    test_logits_1 += preds_dict["te_logits_1"] / args.folds

                oof_logits_2[val_idx] = preds_dict["val_logits_2"]
                oof_logits_3[val_idx] = preds_dict["val_logits_3"]

                test_logits_2 += preds_dict["te_logits_2"] / args.folds
                test_logits_3 += preds_dict["te_logits_3"] / args.folds

                auc_1 = roc_auc_score(y_train[val_idx], oof_logits_1[val_idx])
                auc_2 = roc_auc_score(y_train[val_idx], preds_dict["val_logits_2"])
                auc_3 = roc_auc_score(y_train[val_idx], preds_dict["val_logits_3"])
                logger.info(f"Fold {fold} AUCs: Stage 1 = {auc_1:.5f} | Stage 2 = {auc_2:.5f} | Stage 3 = {auc_3:.5f}")

    # 4. Out-of-Fold Evaluation & Inter-Model Correlation Decoupling
    auc_stage_1 = roc_auc_score(y_train, oof_logits_1)
    auc_stage_2 = roc_auc_score(y_train, oof_logits_2)
    auc_stage_3 = roc_auc_score(y_train, oof_logits_3)

    r_12, _ = pearsonr(oof_logits_1, oof_logits_2)
    r_13, _ = pearsonr(oof_logits_1, oof_logits_3)
    r_23, _ = pearsonr(oof_logits_2, oof_logits_3)
    r_mean = (r_12 + r_13 + r_23) / 3.0
    n_eff = 3.0 / (1.0 + 2.0 * r_mean)

    logger.info("=" * 80)
    logger.info("ARGON CASCADE STANDALONE & STAGE OOF PERFORMANCE")
    logger.info(f"  Stage 1 (LightGBM on View A):            OOF ROC-AUC = {auc_stage_1:.6f}")
    logger.info(f"  Stage 2 (XGBoost on View B with Margin):  OOF ROC-AUC = {auc_stage_2:.6f}")
    logger.info(f"  Stage 3 (LGBM Extra-Trees on View C):    OOF ROC-AUC = {auc_stage_3:.6f}")
    logger.info("-" * 80)
    logger.info("INTER-MODEL CORRELATION MATRIX (COLLINEARITY DECOUPLING)")
    logger.info(f"  Stage 1 vs Stage 2: r = {r_12:.5f} (Spearman = {spearmanr(oof_logits_1, oof_logits_2)[0]:.5f})")
    logger.info(f"  Stage 1 vs Stage 3: r = {r_13:.5f} (Spearman = {spearmanr(oof_logits_1, oof_logits_3)[0]:.5f})")
    logger.info(f"  Stage 2 vs Stage 3: r = {r_23:.5f} (Spearman = {spearmanr(oof_logits_2, oof_logits_3)[0]:.5f})")
    logger.info(f"  Decoupled Mean Correlation: r_mean = {r_mean:.5f}")
    logger.info(f"  Effective Ensemble Size (N_eff): {n_eff:.3f} (vs Baseline 1.002)")
    logger.info("=" * 80)

    # 5. Collinearity Decoupling Stacking (Gram-Schmidt & Power Logit)
    logger.info("Decoupling Cascade with GramSchmidtCascadeDecoupler (L2 Logistic Regression on Orthogonal Deltas)...")
    gs_decoupler = GramSchmidtCascadeDecoupler(
        beta_12=args.beta_12,
        beta_23=args.beta_23,
        gamma_1=0.55,
        gamma_2=0.45,
        C_meta=1.0,
    )
    gs_oof_probs, gs_test_probs, gs_r_mean, gs_auc = gs_decoupler.fit_predict_oof(
        z1_oof=oof_logits_1,
        z2_oof=oof_logits_2,
        z3_oof=oof_logits_3,
        y=y_train.astype(int),
        z1_test=test_logits_1,
        z2_test=test_logits_2,
        z3_test=test_logits_3,
        n_splits=args.folds,
        seed=args.seed,
    )
    logger.info(f"Gram-Schmidt Decoupled OOF ROC-AUC: {gs_auc:.6f} (r_mean = {gs_r_mean:.5f})")

    logger.info("Fitting Strictly Monotonic Power-Logit Stacker...")
    stacker = StrictlyMonotonicPowerLogitStacker(n_models=3)
    stacker.fit_oof([oof_logits_1, oof_logits_2, oof_logits_3], y_train)

    champ_oof_probs = stacker.predict_score([oof_logits_1, oof_logits_2, oof_logits_3])
    champ_test_probs = stacker.predict_score([test_logits_1, test_logits_2, test_logits_3])
    stacked_auc = roc_auc_score(y_train, champ_oof_probs)
    logger.info(f"Power-Logit Stacked OOF ROC-AUC: {stacked_auc:.6f}")

    if gs_auc > stacked_auc:
        logger.info(f"Selecting superior Gram-Schmidt Decoupler: {gs_auc:.6f} vs {stacked_auc:.6f} (+{(gs_auc - stacked_auc)*10000:.1f} bps)")
        champ_oof_probs = gs_oof_probs
        champ_test_probs = gs_test_probs
        stacked_auc = gs_auc
    else:
        logger.info(f"Selecting Power-Logit Stacker: {stacked_auc:.6f}")

    # 6. Optional Transductive kNN Laplacian Smoothing
    if args.use_laplacian:
        logger.info("Applying Transductive Graph Laplacian smoothing across joint manifold...")
        sub_cols = [c for c in ["fd_snapped", "Age_full_cnt", "delay_delta"] if c in dgp_train.columns]
        X_coords = np.vstack([dgp_train[sub_cols].to_numpy(), dgp_test[sub_cols].to_numpy()])
        oof_sm_logits, te_sm_logits = apply_transductive_knn_laplacian_smoothing(
            X_manifold_coords=X_coords,
            oof_logits=oof_logits_3,
            test_logits=test_logits_3,
            k_neighbors=25 if args.fast else 35,
        )
        champ_oof_probs = 1.0 / (1.0 + np.exp(-oof_sm_logits))
        champ_test_probs = 1.0 / (1.0 + np.exp(-te_sm_logits))
        laplacian_auc = roc_auc_score(y_train, champ_oof_probs)
        logger.info(f"POST-LAPLACIAN OOF ROC-AUC: {laplacian_auc:.6f}")

    # 7. Exact-Match Deterministic Ground-Truth Overrides
    try:
        if orig_files:
            orig_ref = ingestion._standardize_columns(pd.read_csv(orig_files[0]), is_original=True)
            postproc = ExactMatchPostprocessor(paths=paths, feature_cfg=feature_cfg)
            matches = postproc.find_exact_matches(test_df, orig_ref)
            if len(matches) > 0:
                logger.info(f"Applying {len(matches)} exact ground-truth test matches.")
                test_sub_df = pd.DataFrame({feature_cfg.id_col: test_ids, feature_cfg.target_col: champ_test_probs})
                for row_id, match_target in matches.items():
                    test_sub_df.loc[test_sub_df[feature_cfg.id_col] == row_id, feature_cfg.target_col] = match_target
                champ_test_probs = test_sub_df[feature_cfg.target_col].to_numpy()
    except Exception as e:
        logger.warning(f"Notice on exact-match overrides: {e}")

    # 8. Submission Export & Integrity Verification
    os.makedirs(paths.submissions_dir, exist_ok=True)
    os.makedirs(paths.output_dir, exist_ok=True)

    sub_df = pd.DataFrame({
        feature_cfg.id_col: test_ids,
        feature_cfg.target_col: champ_test_probs,
    })
    sub_path = os.path.join(paths.submissions_dir, "submission_argon_cascade.csv")
    sub_df.to_csv(sub_path, index=False)
    sub_df.to_csv("submission.csv", index=False)

    # Re-read verification
    verify_df = pd.read_csv("submission.csv")
    assert len(verify_df) == len(test_ids), "Submission integrity check: row count mismatch!"
    assert not verify_df[feature_cfg.target_col].isna().any(), "Submission integrity check: NaNs detected!"
    assert (verify_df[feature_cfg.target_col] >= 0.0).all() and (verify_df[feature_cfg.target_col] <= 1.0).all(), "Bounds violation!"

    # Save summary report
    summary = {
        "pipeline": "Argon_Hierarchical_Residual_Cascade",
        "timestamp": pd.Timestamp.now().isoformat(),
        "folds": args.folds,
        "fast_mode": args.fast,
        "oof_roc_auc": {
            "stage_1_lgbm": float(auc_stage_1),
            "stage_2_xgboost": float(auc_stage_2),
            "stage_3_lgbm_xt": float(auc_stage_3),
            "champion_stacked": float(stacked_auc),
        },
        "collinearity_metrics": {
            "r_stage1_stage2": float(r_12),
            "r_stage1_stage3": float(r_13),
            "r_stage2_stage3": float(r_23),
            "r_mean": float(r_mean),
            "effective_ensemble_size_n_eff": float(n_eff),
        },
        "target_gate": 0.96300,
        "barrier_surpassed": bool(stacked_auc >= 0.96300),
    }

    # Save raw logits and probabilities for offline forensic investigation
    logits_path = os.path.join(paths.output_dir, "argon_cascade_logits.npz")
    np.savez_compressed(
        logits_path,
        oof_logits_1=oof_logits_1,
        oof_logits_2=oof_logits_2,
        oof_logits_3=oof_logits_3,
        test_logits_1=test_logits_1,
        test_logits_2=test_logits_2,
        test_logits_3=test_logits_3,
        y_train=y_train,
        champ_oof_probs=champ_oof_probs,
        champ_test_probs=champ_test_probs,
    )

    with open(os.path.join(paths.output_dir, "argon_evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"ARGON CASCADE PIPELINE COMPLETE: Champion OOF ROC-AUC = {stacked_auc:.6f}")
    logger.info(f"Outputs serialized to: 'submission.csv', '{logits_path}', and '{paths.output_dir}/argon_evaluation_summary.json'")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
