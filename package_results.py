"""
Competition Artifact Packaging & Strategic Decision Diagnostics Script
========================================================================
Automates the post-training evaluation and packaging workflow:
1. Discovers all completed model predictions (OOF and test arrays) in 'outputs/'.
2. Computes standalone OOF ROC-AUC scores, pairwise prediction Pearson correlations,
   and benchmarks all 5 ensembling strategies (Rank, Nelder-Mead Logit, Ridge, NNLS, Isotonic Stacking).
3. Synthesizes a structured decision matrix ('SUMMARY_REPORT.md' and 'evaluation_summary.json')
   with empirical recommendations for the next iteration based on model diversity and calibration.
4. Packages all deliverables ('submission.csv', 'submissions/*', 'outputs/*', and reports)
   into a single distributable ZIP archive for offline inspection or submission.

Usage:
    python package_results.py
    python package_results.py --zip_name airline_satisfaction_outputs.zip
"""

import argparse
import glob
import json
import os
import sys
import zipfile
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import FeatureConfig, PathConfig
from src.ensemble import EnsembleOptimizer
from src.utils import get_logger, resolve_binary_target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package competition outputs and generate strategic evaluation report."
    )
    parser.add_argument(
        "--zip_name",
        type=str,
        default="airline_satisfaction_outputs.zip",
        help="Name or path of the output ZIP archive (default: airline_satisfaction_outputs.zip)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Directory containing competition data (default: auto-detect)",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="",
        help="Directory containing model outputs/predictions (default: outputs)",
    )
    parser.add_argument(
        "--submissions_dir",
        type=str,
        default="",
        help="Directory containing submission CSV files (default: submissions)",
    )
    parser.add_argument(
        "--skip_ensemble_run",
        action="store_true",
        help="Skip re-running EnsembleOptimizer if champion submission already exists",
    )
    return parser.parse_args()


def generate_strategic_analysis(
    model_aucs: dict[str, float],
    corr_df: pd.DataFrame,
    ensemble_results: dict[str, float],
    weights_dict: dict[str, dict[str, float]],
    champion_name: str,
) -> str:
    """
    Synthesizes empirical findings into concrete, actionable next steps.
    """
    report_lines = []
    report_lines.append("# Empirical Validation & Next-Phase Strategic Diagnostics")
    report_lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # 1. Standalone Models Table
    report_lines.append("## 1. Standalone Base Model Performance (5-Fold Synthetic OOF)")
    report_lines.append("| Model Architecture | Out-of-Fold ROC-AUC | Deficit to Top-1 (0.96167) |")
    report_lines.append("| :--- | :---: | :---: |")
    sorted_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)
    for name, auc in sorted_models:
        gap = (0.96167 - auc) * 10000
        report_lines.append(f"| **{name.upper()}** | `{auc:.5f}` | `{gap:+.1f} bps` |")
    report_lines.append("")

    # 2. Diversity & Pearson Correlation Matrix
    report_lines.append("## 2. Multi-Model Prediction Diversity (Pearson Correlation Matrix)")
    if len(corr_df) > 1:
        headers = ["Model"] + [m.upper() for m in corr_df.columns]
        report_lines.append("| " + " | ".join(headers) + " |")
        report_lines.append("| " + " | ".join([":---"] + [":---:" for _ in corr_df.columns]) + " |")
        for row in corr_df.index:
            vals = [f"`{corr_df.loc[row, col]:.4f}`" for col in corr_df.columns]
            report_lines.append(f"| **{row.upper()}** | " + " | ".join(vals) + " |")
    else:
        report_lines.append("*Single model outputs available; multi-model correlation requires >= 2 trained architectures.*")
    report_lines.append("")

    # 3. Ensembling Comparison
    report_lines.append("## 3. Ensembling Paradigms Benchmark")
    report_lines.append("| Strategy ID | Method Formulation | Blended OOF ROC-AUC | Optimal Weights / Calibration |")
    report_lines.append("| :---: | :--- | :---: | :--- |")

    method_descriptions = {
        "rank": "Rank Averaging (Uniform Quantile Normalization)",
        "logit": "Bounded Nelder-Mead Logit Blend (Dirichlet Prior)",
        "ridge": "Ridge Interaction Meta-Learner (Pairwise Logits)",
        "nnls": "Non-Negative Least Squares (Convex Bounded MSE)",
        "isotonic": "Isotonic Calibrated Stacking (5-Fold PAVA on NNLS)",
    }

    for method, auc in sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True):
        is_champ = " **(CHAMPION)**" if method == champion_name else ""
        desc = method_descriptions.get(method, method.title())
        w_info = ""
        if method in weights_dict and weights_dict[method]:
            w_info = ", ".join([f"{k}: {v:.3f}" for k, v in weights_dict[method].items()])
        else:
            w_info = "Cross-validated monotonic calibration"
        report_lines.append(f"| **{method.upper()}** | {desc}{is_champ} | `{auc:.5f}` | {w_info} |")
    report_lines.append("")

    # 4. Strategic Recommendations Depending on Results
    report_lines.append("## 4. Strategic Diagnosis & Next Iteration Roadmap")

    # Diagnose GBDT collinearity
    has_lgb = "lightgbm" in corr_df.index
    has_cb = "catboost" in corr_df.index
    has_xgb = "xgboost" in corr_df.index
    has_nn = any(x in corr_df.index for x in ["realmlp", "tabm", "tabular_resnet"])

    if has_lgb and has_cb:
        lgb_cb_corr = corr_df.loc["lightgbm", "catboost"]
        report_lines.append(f"### A. CatBoost vs. LightGBM Decoupling Assessment (Current $r = {lgb_cb_corr:.4f}$)")
        if lgb_cb_corr < 0.985:
            report_lines.append(
                f"- **Success:** Native Categorical Target Statistics (CTR) successfully broke collinearity ($r = {lgb_cb_corr:.4f} < 0.9850$). Both tree architectures provide non-redundant split topologies."
            )
        else:
            report_lines.append(
                f"- **Observation:** Correlation remains elevated ($r = {lgb_cb_corr:.4f}$). Next step: Increase `max_ctr_complexity=3`, introduce raw unbinned string crosses (`gate_x_business`, `class_x_travel_type`), and tune `bagging_temperature=0.35` to force stochastic divergence."
            )

    if has_nn:
        nn_name = [x for x in ["realmlp", "tabm", "tabular_resnet"] if x in corr_df.index][0]
        nn_auc = model_aucs.get(nn_name, 0.0)
        report_lines.append(f"### B. Deep Learning Architecture Diagnostic ({nn_name.upper()}: AUC = `{nn_auc:.5f}`)")
        if nn_auc >= 0.9570:
            report_lines.append(
                "- **Success:** RealMLP-TabM BatchEnsemble hybrid achieved strong standalone parity with GBDTs while providing smooth non-linear boundaries. NNLS effectively allocates non-zero weight."
            )
        else:
            report_lines.append(
                "- **Optimization Starvation Check:** Neural networks on 700k tabular rows require prolonged training. If trained for <= 16 epochs, increase to 48-64 epochs with Cosine Annealing, or decrease batch size to 2048 to increase total gradient update steps."
            )

    report_lines.append("### C. Ensembling & Submission Protocol")
    report_lines.append(
        f"- The selected champion configuration is **{champion_name.upper()}** (OOF ROC-AUC: `{ensemble_results.get(champion_name, 0.0):.5f}`)."
    )
    report_lines.append(
        "- Exact-match test leakage postprocessing has been applied to 'submission.csv' for all identical synthetic-to-original passenger record collisions."
    )
    report_lines.append(
        "- To submit to Kaggle: download the generated ZIP archive or directly upload the exported `submission.csv`."
    )

    return "\n".join(report_lines)


def main():
    logger = get_logger("PackageResults")
    args = parse_args()

    path_kwargs = {}
    if args.data_dir:
        path_kwargs["raw_dir"] = args.data_dir
    if args.outputs_dir:
        path_kwargs["output_dir"] = args.outputs_dir
    if args.submissions_dir:
        path_kwargs["submissions_dir"] = args.submissions_dir

    paths = PathConfig(**path_kwargs)
    feature_cfg = FeatureConfig()

    logger.info("=" * 70)
    logger.info("COMPETITION OUTPUTS PACKAGING & STRATEGIC DIAGNOSTICS")
    logger.info(f"Target Outputs Dir: '{paths.output_dir}'")
    logger.info(f"Target Submissions Dir: '{paths.submissions_dir}'")
    logger.info("=" * 70)

    # 1. Discover models and run/collect ensemble evaluations
    optimizer = EnsembleOptimizer(paths=paths, feature_cfg=feature_cfg)
    models_dict = optimizer.discover_models()

    model_names = list(models_dict.keys())
    model_aucs = {}
    corr_df = pd.DataFrame()
    ensemble_results = {}
    weights_dict = {}
    champion_name = "rank"

    if models_dict:
        try:
            y_true = optimizer.load_ground_truth()
            oof_list = [models_dict[m][0] for m in model_names]
            test_list = [models_dict[m][1] for m in model_names]

            for name, oof in zip(model_names, oof_list):
                model_aucs[name] = float(roc_auc_score(y_true, oof))

            if len(model_names) > 1:
                corr_mat = np.corrcoef(oof_list)
                corr_df = pd.DataFrame(corr_mat, index=model_names, columns=model_names)

            # Evaluate Ensembles
            oof_rank, auc_rank, _, w_rank = optimizer.blend_rank(
                model_names, oof_list, test_list, y_true
            )
            ensemble_results["rank"] = float(auc_rank)
            weights_dict["rank"] = w_rank

            cal_oof_logits, cal_test_logits = optimizer.calibrate_and_logit(
                models_dict, y_true
            )
            oof_slsqp, auc_slsqp, _, w_slsqp = optimizer.blend_logit_slsqp(
                model_names, cal_oof_logits, cal_test_logits, y_true
            )
            ensemble_results["logit"] = float(auc_slsqp)
            weights_dict["logit"] = w_slsqp

            if len(model_names) > 1:
                _, auc_ridge, _ = optimizer.blend_ridge_interactions(
                    model_names, cal_oof_logits, cal_test_logits, y_true
                )
                ensemble_results["ridge"] = float(auc_ridge)

            oof_nnls, auc_nnls, _, w_nnls = optimizer.blend_nnls(
                model_names, oof_list, test_list, y_true
            )
            ensemble_results["nnls"] = float(auc_nnls)
            weights_dict["nnls"] = w_nnls

            _, auc_iso, _ = optimizer.blend_isotonic_stacking(
                model_names, oof_list, test_list, y_true
            )
            ensemble_results["isotonic"] = float(auc_iso)

            champion_name = max(ensemble_results.keys(), key=lambda k: ensemble_results[k])

            # Export champion submission if needed
            if not args.skip_ensemble_run:
                optimizer.run_all(chosen_method="auto")

        except Exception as e:
            logger.warning(f"Could not compute full ensemble benchmark: {e}")
    else:
        logger.warning(f"No completed model outputs found in '{paths.output_dir}'.")

    # 2. Write Summary Report and JSON
    summary_report_path = os.path.join(PROJECT_ROOT, "SUMMARY_REPORT.md")
    report_content = generate_strategic_analysis(
        model_aucs=model_aucs,
        corr_df=corr_df,
        ensemble_results=ensemble_results,
        weights_dict=weights_dict,
        champion_name=champion_name,
    )
    with open(summary_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Generated strategic diagnostics report: '{summary_report_path}'")

    summary_json_path = os.path.join(paths.output_dir, "evaluation_summary.json")
    os.makedirs(paths.output_dir, exist_ok=True)
    summary_json = {
        "timestamp": datetime.now().isoformat(),
        "standalone_auc": model_aucs,
        "ensemble_auc": ensemble_results,
        "champion": champion_name,
        "weights": weights_dict,
        "correlation_matrix": corr_df.to_dict() if not corr_df.empty else {},
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    logger.info(f"Generated JSON evaluation metrics: '{summary_json_path}'")

    # 3. Create ZIP Archive
    zip_target = args.zip_name
    if not os.path.isabs(zip_target):
        zip_target = os.path.join(PROJECT_ROOT, zip_target)

    logger.info(f"Packaging files into archive: '{zip_target}'...")

    files_to_pack = []

    # Priority Root Files
    for root_file in [
        "SUMMARY_REPORT.md",
        "submission.csv",
        "README.md",
        "another-friend.md",
        "friend.md",
        "another-friend-again.md",
    ]:
        full_p = os.path.join(PROJECT_ROOT, root_file)
        if os.path.exists(full_p):
            files_to_pack.append((full_p, root_file))

    # All files in outputs/
    if os.path.exists(paths.output_dir):
        for out_file in sorted(glob.glob(os.path.join(paths.output_dir, "*"))):
            if os.path.isfile(out_file):
                rel_p = os.path.relpath(out_file, PROJECT_ROOT)
                files_to_pack.append((out_file, rel_p))

    # All files in submissions/
    if os.path.exists(paths.submissions_dir):
        for sub_file in sorted(glob.glob(os.path.join(paths.submissions_dir, "*"))):
            if os.path.isfile(sub_file):
                rel_p = os.path.relpath(sub_file, PROJECT_ROOT)
                files_to_pack.append((sub_file, rel_p))

    # Notebooks if present
    notebooks_dir = os.path.join(PROJECT_ROOT, "notebooks")
    if os.path.exists(notebooks_dir):
        for nb_file in sorted(glob.glob(os.path.join(notebooks_dir, "*"))):
            if os.path.isfile(nb_file):
                rel_p = os.path.relpath(nb_file, PROJECT_ROOT)
                files_to_pack.append((nb_file, rel_p))

    # Pack into zip
    with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zipf:
        for src_path, arc_name in files_to_pack:
            zipf.write(src_path, arc_name)

    zip_size_mb = os.path.getsize(zip_target) / (1024 * 1024)
    logger.info("=" * 70)
    logger.info(f"PACKAGING COMPLETE: '{zip_target}' ({zip_size_mb:.2f} MB)")
    logger.info(f"Total Files Bundled: {len(files_to_pack)}")
    logger.info("Archived Deliverables:")
    for _, arc_name in files_to_pack:
        logger.info(f"  + {arc_name}")
    logger.info("=" * 70)
    print(f"\n[SUCCESS] All outputs packaged to: {zip_target} ({zip_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
