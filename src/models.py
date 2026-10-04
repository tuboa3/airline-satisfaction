"""
Model Wrappers: High-Performance LightGBM, CatBoost, and XGBoost with GPU/CPU Detection
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    _ModuleBase = nn.Module
except ImportError:
    torch = None
    nn = None
    _ModuleBase = object


class BaseModel(ABC):
    """Abstract Model Interface."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, **kwargs) -> None:
        pass

    @abstractmethod
    def predict_proba(self, X, **kwargs) -> np.ndarray:
        pass


class GLMMarginGenerator:
    """
    Stage 1 Generalized Linear Model with Natural Splines & One-Hot Encoding.
    Decomposes the predictive task: GLM captures additive linear and monotonic spline signals,
    generating leak-free OOF logit margins eta = x^T beta for downstream residual boosting.
    """

    def __init__(
        self,
        continuous_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        n_knots: int = 5,
        degree: int = 3,
        C: float = 0.1,
        random_state: int = 42,
    ):
        self.continuous_cols = continuous_cols or []
        self.categorical_cols = categorical_cols or []
        self.n_knots = n_knots
        self.degree = degree
        self.C = C
        self.random_state = random_state
        self.spline = None
        self.ohe = None
        self.scaler = None
        self.glm = None
        self.actual_cont_cols: list[str] = []
        self.actual_cat_cols: list[str] = []

    def fit(
        self,
        X_train,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "GLMMarginGenerator":
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)

        self.actual_cont_cols = [c for c in self.continuous_cols if c in X_train.columns]
        self.actual_cat_cols = [c for c in self.categorical_cols if c in X_train.columns]

        parts = []
        if self.actual_cont_cols:
            X_cont = X_train[self.actual_cont_cols].fillna(0).values.astype(np.float32)
            self.scaler = StandardScaler()
            X_cont_scaled = self.scaler.fit_transform(X_cont)
            self.spline = SplineTransformer(
                n_knots=self.n_knots, degree=self.degree, include_bias=False
            )
            X_cont_splines = self.spline.fit_transform(X_cont_scaled)
            parts.append(X_cont_splines)

        if self.actual_cat_cols:
            X_cat = X_train[self.actual_cat_cols].astype(str).values
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat_encoded = self.ohe.fit_transform(X_cat)
            parts.append(X_cat_encoded)

        if not parts:
            X_mat = X_train.fillna(0).values.astype(np.float32)
            self.scaler = StandardScaler()
            X_mat_scaled = self.scaler.fit_transform(X_mat)
            parts.append(X_mat_scaled)

        X_combined = np.hstack(parts) if len(parts) > 1 else parts[0]

        self.glm = LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.glm.fit(X_combined, y_train, sample_weight=sample_weight)
        return self

    def predict_margin(self, X) -> np.ndarray:
        import pandas as pd

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        parts = []
        if self.actual_cont_cols and self.spline is not None and self.scaler is not None:
            X_cont = X[self.actual_cont_cols].fillna(0).values.astype(np.float32)
            X_cont_scaled = self.scaler.transform(X_cont)
            X_cont_splines = self.spline.transform(X_cont_scaled)
            parts.append(X_cont_splines)

        if self.actual_cat_cols and self.ohe is not None:
            X_cat = X[self.actual_cat_cols].astype(str).values
            X_cat_encoded = self.ohe.transform(X_cat)
            parts.append(X_cat_encoded)

        if not parts and self.scaler is not None:
            X_mat = X.fillna(0).values.astype(np.float32)
            X_mat_scaled = self.scaler.transform(X_mat)
            parts.append(X_mat_scaled)

        X_combined = np.hstack(parts) if len(parts) > 1 else parts[0]
        margins = self.glm.decision_function(X_combined).astype(np.float32)
        return margins


class LightGBMModel(BaseModel):
    """Production LightGBM Model with Auto-Hardware Configuration and Residual Margin Support."""

    def __init__(self, params: dict[str, Any] = None, device: str = "cpu"):
        self.params = params.copy() if params else {}
        self.device = device
        self.model = None

        # Hardware optimization
        if self.device == "cuda":
            try:
                self.params["device"] = "gpu"
            except Exception:
                self.params["device"] = "cpu"
        else:
            self.params["device"] = "cpu"
            self.params["n_jobs"] = -1

        # Sanitize foreign hyperparameters
        for invalid_key in [
            "loss_function",
            "eval_metric",
            "task_type",
            "thread_count",
            "l2_leaf_reg",
            "iterations",
            "tree_method",
            "gamma",
        ]:
            self.params.pop(invalid_key, None)

    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None, **kwargs) -> None:
        import lightgbm as lgb

        base_margin_tr = kwargs.get("base_margin_tr", None)
        base_margin_val = kwargs.get("base_margin_val", None)

        trn_data = lgb.Dataset(
            X_train, label=y_train, weight=sample_weight, init_score=base_margin_tr
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, init_score=base_margin_val, reference=trn_data
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=250),
        ]

        num_boost_round = self.params.pop("n_estimators", 2500)

        try:
            self.model = lgb.train(
                self.params,
                trn_data,
                num_boost_round=num_boost_round,
                valid_sets=[trn_data, val_data],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )
        except Exception as e:
            # Graceful GPU fallback if Kaggle LightGBM wheel lacks OpenCL
            if "gpu" in str(self.params.get("device", "")).lower():
                logging.warning(
                    f"LightGBM GPU initialization failed ({e}). Falling back to multi-core CPU..."
                )
                self.params["device"] = "cpu"
                self.params["n_jobs"] = -1
                self.model = lgb.train(
                    self.params,
                    trn_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[trn_data, val_data],
                    valid_names=["train", "valid"],
                    callbacks=callbacks,
                )
            else:
                raise e

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        from scipy.special import expit

        base_margin = kwargs.get("base_margin", None)
        if base_margin is not None:
            raw_scores = self.model.predict(
                X, num_iteration=self.model.best_iteration, raw_score=True
            )
            return expit(raw_scores + base_margin)
        else:
            return self.model.predict(X, num_iteration=self.model.best_iteration)


class CatBoostModel(BaseModel):
    """
    Production CatBoost Model with Native CTR, Ordered Target Statistics,
    and string categorical support to break collinearity with LightGBM.
    """

    def __init__(self, params: dict[str, Any] = None, device: str = "cpu"):
        from catboost import CatBoostClassifier

        self.params = params.copy() if params else {}
        if device == "cuda":
            self.params["task_type"] = "GPU"
            try:
                import torch

                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    gpu_ids = ":".join(str(i) for i in range(torch.cuda.device_count()))
                    self.params["devices"] = gpu_ids
                    logging.info(
                        f"CatBoost configured for multi-GPU training on devices: {gpu_ids}"
                    )
            except Exception:
                pass
        else:
            self.params["task_type"] = "CPU"
            self.params["thread_count"] = -1

        # Prevent duplicate/conflicting parameter errors
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)
        self.verbose = self.params.pop("verbose", 250)

        # Sanitize any accidental foreign hyperparameters
        for invalid_key in [
            "metric",
            "objective",
            "boosting_type",
            "n_estimators",
            "num_leaves",
            "colsample_bytree",
            "subsample",
            "tree_method",
            "gamma",
        ]:
            self.params.pop(invalid_key, None)

        self.model = CatBoostClassifier(**self.params)

    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None, **kwargs) -> None:
        cat_features = kwargs.get("cat_features", None)
        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            use_best_model=True,
        )

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class XGBoostModel(BaseModel):
    """
    Production XGBoost Model with Histogram GPU / Multi-threading
    and Base Margin Residual Boosting.
    """

    def __init__(self, params: dict[str, Any] = None, device: str = "cpu"):
        self.params = params.copy() if params else {}
        self.params["tree_method"] = "hist"
        if device == "cuda":
            self.params["device"] = "cuda"
        else:
            self.params["device"] = "cpu"
            self.params["n_jobs"] = -1

        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)
        self.verbose = self.params.pop("verbose", 250)
        self.num_boost_round = self.params.pop("n_estimators", 2500)

        # Sanitize foreign parameters
        for invalid_key in [
            "loss_function",
            "task_type",
            "thread_count",
            "l2_leaf_reg",
            "num_leaves",
            "boosting_type",
        ]:
            self.params.pop(invalid_key, None)

        if "objective" not in self.params:
            self.params["objective"] = "binary:logistic"
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = "auc"

        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None, **kwargs) -> None:
        import xgboost as xgb

        base_margin_tr = kwargs.get("base_margin_tr", None)
        base_margin_val = kwargs.get("base_margin_val", None)

        dtrain = xgb.DMatrix(
            X_train, label=y_train, weight=sample_weight, base_margin=base_margin_tr
        )
        dval = xgb.DMatrix(X_val, label=y_val, base_margin=base_margin_val)

        evals = [(dtrain, "train"), (dval, "valid")]
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose,
        )

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        import xgboost as xgb

        base_margin = kwargs.get("base_margin", None)
        dtest = xgb.DMatrix(X, base_margin=base_margin)
        return self.model.predict(dtest)


class FTTransformerModel(BaseModel):
    """
    Domain D: Compact Feature Tokenizer Transformer (FT-Transformer)
    for Tabular Relational Learning & Ensemble Orthogonalization.
    Includes Domain C: Teacher-Student Soft Distillation & Test Consistency Regularization.
    """

    def __init__(self, params: dict[str, Any] = None, device: str = "cpu"):
        self.params = params.copy() if params else {}
        self.device_str = device
        self.model = None
        self.cat_cols = []
        self.num_cols = []
        self.num_mean = None
        self.num_std = None

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        teacher_train=None,
        X_test=None,
        teacher_test=None,
        **kwargs,
    ) -> None:
        import torch
        from sklearn.metrics import roc_auc_score
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        # Hardware setup
        if self.device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            use_amp = True
        else:
            device = torch.device("cpu")
            use_amp = False

        logging.info(
            f"FT-Transformer initializing on device: {device} (AMP: {use_amp})"
        )

        # 1. Column Segregation: Categorical vs Numerical
        cat_candidates = [
            "Gender",
            "Customer Type",
            "Type of Travel",
            "Class",
            "class_x_travel_type",
            "gate_x_business",
            "delay_tier",
        ]
        self.cat_cols = [c for c in cat_candidates if c in X_train.columns]
        self.num_cols = [c for c in X_train.columns if c not in self.cat_cols]

        # 2. Numerical Standardization
        X_tr_num = X_train[self.num_cols].values.astype(np.float32)
        X_va_num = X_val[self.num_cols].values.astype(np.float32)

        self.num_mean = np.nanmean(X_tr_num, axis=0)
        self.num_std = np.nanstd(X_tr_num, axis=0) + 1e-6
        X_tr_num = np.nan_to_num((X_tr_num - self.num_mean) / self.num_std)
        X_va_num = np.nan_to_num((X_va_num - self.num_mean) / self.num_std)

        # 3. Categorical Index Alignment
        cat_cardinalities = []
        if self.cat_cols:
            X_tr_cat = np.clip(X_train[self.cat_cols].values.astype(np.int64), 0, None)
            X_va_cat = np.clip(X_val[self.cat_cols].values.astype(np.int64), 0, None)
            for j in range(len(self.cat_cols)):
                max_c = max(int(X_tr_cat[:, j].max()), int(X_va_cat[:, j].max())) + 1
                cat_cardinalities.append(max_c)
        else:
            X_tr_cat = np.zeros((len(X_train), 0), dtype=np.int64)
            X_va_cat = np.zeros((len(X_val), 0), dtype=np.int64)

        # 4. Hyperparameters
        embed_dim = self.params.get("embed_dim", 32)
        num_layers = self.params.get("num_layers", 3)
        num_heads = self.params.get("num_heads", 4)
        ffn_ratio = self.params.get("ffn_ratio", 2)
        dropout = self.params.get("dropout", 0.1)
        lr = self.params.get("lr", 1.5e-3)
        weight_decay = self.params.get("weight_decay", 1e-4)
        batch_size = self.params.get("batch_size", 2048)
        epochs = self.params.get("epochs", 7)
        distill_alpha = self.params.get("distillation_alpha", 0.5)
        consist_lambda = self.params.get("consistency_lambda", 0.15)

        # 5. Define Neural Network Module
        class _FTTransformer(nn.Module):
            def __init__(self, n_num, cat_cards, d, n_layers, n_heads, ffn_r, drop):
                super().__init__()
                self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
                self.num_w = (
                    nn.Parameter(torch.randn(n_num, d) * 0.02) if n_num > 0 else None
                )
                self.num_b = nn.Parameter(torch.zeros(n_num, d)) if n_num > 0 else None
                self.cat_embs = nn.ModuleList(
                    [nn.Embedding(card, d) for card in cat_cards]
                )

                self.blocks = nn.ModuleList()
                for _ in range(n_layers):
                    self.blocks.append(
                        nn.ModuleDict(
                            {
                                "norm1": nn.LayerNorm(d),
                                "mha": nn.MultiheadAttention(
                                    d, n_heads, dropout=drop, batch_first=True
                                ),
                                "drop1": nn.Dropout(drop),
                                "norm2": nn.LayerNorm(d),
                                "ffn": nn.Sequential(
                                    nn.Linear(d, d * ffn_r),
                                    nn.GELU(),
                                    nn.Dropout(drop),
                                    nn.Linear(d * ffn_r, d),
                                ),
                                "drop2": nn.Dropout(drop),
                            }
                        )
                    )

                self.head_norm = nn.LayerNorm(d)
                self.head = nn.Sequential(
                    nn.Linear(d, d // 2),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(d // 2, 1),
                )

            def forward(self, x_num, x_cat):
                amp_enabled = x_num.is_cuda
                try:
                    from torch.amp import autocast

                    ctx = autocast("cuda", enabled=amp_enabled)
                except (ImportError, TypeError):
                    from torch.cuda.amp import autocast

                    ctx = autocast(enabled=amp_enabled)

                with ctx:
                    B = x_num.size(0)
                    tokens = [self.cls_token.expand(B, -1, -1)]

                    if self.num_w is not None and x_num.size(1) > 0:
                        num_toks = x_num.unsqueeze(-1) * self.num_w + self.num_b
                        tokens.append(num_toks)

                    if len(self.cat_embs) > 0 and x_cat.size(1) > 0:
                        cat_toks = torch.stack(
                            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)],
                            dim=1,
                        )
                        tokens.append(cat_toks)

                    x = torch.cat(tokens, dim=1)
                    for blk in self.blocks:
                        # Pre-LN Self-Attention
                        nx = blk["norm1"](x)
                        attn_out, _ = blk["mha"](nx, nx, nx)
                        x = x + blk["drop1"](attn_out)
                        # Pre-LN FeedForward
                        nx = blk["norm2"](x)
                        ffn_out = blk["ffn"](nx)
                        x = x + blk["drop2"](ffn_out)

                    cls_feat = self.head_norm(x[:, 0])
                    return self.head(cls_feat).squeeze(-1)

        # Multi-GPU Detection & Scaling
        n_gpus = torch.cuda.device_count() if (device.type == "cuda") else 0
        if n_gpus > 1:
            logging.info(
                f"Multi-GPU detected ({n_gpus} GPUs)! Scaling batch size and activating torch.nn.DataParallel."
            )
            batch_size = batch_size * n_gpus

        raw_model = _FTTransformer(
            n_num=len(self.num_cols),
            cat_cards=cat_cardinalities,
            d=embed_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            ffn_r=ffn_ratio,
            drop=dropout,
        ).to(device)

        if n_gpus > 1:
            self.model = nn.DataParallel(raw_model)
        else:
            self.model = raw_model

        # 6. Tensor Datasets and Multi-threaded Loaders
        t_X_num = torch.tensor(X_tr_num, dtype=torch.float32)
        t_X_cat = torch.tensor(X_tr_cat, dtype=torch.long)
        t_y = torch.tensor(y_train, dtype=torch.float32)

        has_distill = teacher_train is not None
        has_consistency = X_test is not None and teacher_test is not None

        if has_distill:
            t_teach = torch.tensor(teacher_train, dtype=torch.float32)
            train_dataset = TensorDataset(t_X_num, t_X_cat, t_y, t_teach)
        else:
            train_dataset = TensorDataset(t_X_num, t_X_cat, t_y)

        n_workers = min(2, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            num_workers=n_workers,
        )

        # Domain C: Prepare Unlabeled Test Data Loader with Soft Teacher Targets
        if has_consistency:
            X_te_num = X_test[self.num_cols].values.astype(np.float32)
            X_te_num = np.nan_to_num((X_te_num - self.num_mean) / self.num_std)
            if self.cat_cols:
                X_te_cat = np.clip(
                    X_test[self.cat_cols].values.astype(np.int64), 0, None
                )
            else:
                X_te_cat = np.zeros((len(X_test), 0), dtype=np.int64)

            t_te_num = torch.tensor(X_te_num, dtype=torch.float32)
            t_te_cat = torch.tensor(X_te_cat, dtype=torch.long)
            t_te_teach = torch.tensor(teacher_test, dtype=torch.float32)

            test_dataset = TensorDataset(t_te_num, t_te_cat, t_te_teach)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=(device.type == "cuda"),
                num_workers=n_workers,
            )
            test_iter = iter(test_loader)
            logging.info(
                f"Initialized Domain C Test Consistency Regularizer on {len(X_test)} test rows."
            )

        v_X_num = torch.tensor(X_va_num, dtype=torch.float32)
        v_X_cat = torch.tensor(X_va_cat, dtype=torch.long)

        # 7. Optimizer, Scaler & Scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        # PyTorch modern AMP context helper
        try:
            from torch.amp import GradScaler, autocast

            scaler = GradScaler("cuda", enabled=use_amp)

            def amp_context():
                return autocast("cuda", enabled=use_amp)
        except (ImportError, TypeError):
            from torch.cuda.amp import GradScaler, autocast

            scaler = GradScaler(enabled=use_amp)

            def amp_context():
                return autocast(enabled=use_amp)

        bce_loss_fn = nn.BCEWithLogitsLoss()
        best_auc = 0.0
        best_state = None

        # 8. Training Loop
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                if has_distill:
                    b_num, b_cat, b_y, b_teach = [item.to(device) for item in batch]
                else:
                    b_num, b_cat, b_y = [item.to(device) for item in batch]
                    b_teach = None

                with amp_context():
                    logits = self.model(b_num, b_cat)
                    loss = bce_loss_fn(logits, b_y)

                    # Domain C: Soft Distillation on Training Data
                    if b_teach is not None:
                        distill_loss = bce_loss_fn(logits, b_teach)
                        loss = (
                            1.0 - distill_alpha
                        ) * loss + distill_alpha * distill_loss

                    # Domain C: Consistency Regularization on Unlabeled Test Data
                    if has_consistency:
                        try:
                            t_num, t_cat, t_teach = next(test_iter)
                        except StopIteration:
                            test_iter = iter(test_loader)
                            t_num, t_cat, t_teach = next(test_iter)

                        t_num, t_cat, t_teach = (
                            t_num.to(device),
                            t_cat.to(device),
                            t_teach.to(device),
                        )
                        test_logits = self.model(t_num, t_cat)
                        consistency_loss = bce_loss_fn(test_logits, t_teach)
                        loss = loss + consist_lambda * consistency_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            scheduler.step()

            # Validation Evaluation (Chunked to prevent CUDA OOM, accelerated across GPUs)
            self.model.eval()
            val_probs = []
            val_chunk_size = 2048 * max(1, n_gpus)
            with torch.no_grad():
                for vi in range(0, len(y_val), val_chunk_size):
                    vb_num = v_X_num[vi : vi + val_chunk_size].to(device)
                    vb_cat = v_X_cat[vi : vi + val_chunk_size].to(device)
                    vb_logits = self.model(vb_num, vb_cat)
                    val_probs.append(torch.sigmoid(vb_logits).cpu().numpy())
            val_probs = np.concatenate(val_probs, axis=0)

            val_auc = roc_auc_score(y_val, val_probs)
            logging.info(
                f"Epoch {epoch}/{epochs} | Loss: {running_loss / len(train_loader):.4f} | Val ROC-AUC: {val_auc:.5f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                m_to_save = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                best_state = {
                    k: v.cpu().clone() for k, v in m_to_save.state_dict().items()
                }

        if best_state is not None:
            m_to_save = (
                self.model.module
                if isinstance(self.model, nn.DataParallel)
                else self.model
            )
            m_to_save.load_state_dict(best_state)
            logging.info(
                f"Loaded Best FT-Transformer State (Validation ROC-AUC: {best_auc:.5f})"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    def predict_proba(self, X) -> np.ndarray:
        import torch

        device = next(self.model.parameters()).device
        self.model.eval()

        X_num = X[self.num_cols].values.astype(np.float32)
        X_num = np.nan_to_num((X_num - self.num_mean) / self.num_std)

        if self.cat_cols:
            X_cat = np.clip(X[self.cat_cols].values.astype(np.int64), 0, None)
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.int64)

        n_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
        batch_size = 4096 * max(1, n_gpus)
        probs = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                b_num = torch.tensor(X_num[i : i + batch_size], dtype=torch.float32).to(
                    device
                )
                b_cat = torch.tensor(X_cat[i : i + batch_size], dtype=torch.long).to(
                    device
                )
                logits = self.model(b_num, b_cat)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return np.concatenate(probs, axis=0)


class SurrogateAUCLoss(_ModuleBase):
    """
    Domain 3: Differentiable surrogate ranking loss maximizing ROC-AUC directly via pairwise differences.
    """

    def __init__(self, gamma: float = 15.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_logits = logits[targets == 1]
        neg_logits = logits[targets == 0]
        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        differences = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
        return torch.mean((1.0 - torch.sigmoid(self.gamma * differences)) ** 2)


class PeriodicLinearEmbeddings(_ModuleBase):
    """
    Domain 3: Piecewise Linear Representations (PLR) via Sinusoidal Periodic Embeddings.
    Allows neural networks to map irregular piecewise continuous features without GBDT mimicry.
    """

    def __init__(self, n_features: int, n_frequencies: int = 16, d_embedding: int = 32):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_features, n_frequencies) * 0.1)
        self.phases = nn.Parameter(torch.zeros(n_features, n_frequencies))
        self.proj = nn.Linear(n_frequencies * 2 + 1, d_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_unsqueezed = x.unsqueeze(-1)
        angles = 2.0 * np.pi * x_unsqueezed * self.frequencies.unsqueeze(0) + self.phases.unsqueeze(0)
        periodic = torch.cat([torch.sin(angles), torch.cos(angles), x_unsqueezed], dim=-1)
        return self.proj(periodic)


class ResNetBlock(_ModuleBase):
    """Residual Linear Block with LayerNorm, GELU, and Dropout."""

    def __init__(self, d: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d * 2, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear2(self.dropout(self.act(self.linear1(self.norm(x)))))
        return residual + out


class TabularResNetNet(_ModuleBase):
    """
    Pure Un-distilled Tabular ResNet with Periodic Linear Representations (PLR).
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        d_block: int = 256,
        n_blocks: int = 3,
        d_embedding: int = 32,
        n_frequencies: int = 16,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_num = n_num
        self.plr = (
            PeriodicLinearEmbeddings(n_num, n_frequencies=n_frequencies, d_embedding=d_embedding)
            if n_num > 0
            else None
        )
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card + 2, d_embedding) for card in cat_cardinalities]
        )
        total_dim = (n_num + len(cat_cardinalities)) * d_embedding
        self.input_proj = nn.Linear(total_dim, d_block)
        self.blocks = nn.ModuleList([ResNetBlock(d_block, dropout=dropout) for _ in range(n_blocks)])
        self.head_norm = nn.LayerNorm(d_block)
        self.head = nn.Linear(d_block, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embeds = []
        if self.plr is not None and self.n_num > 0:
            embeds.append(self.plr(x_num).flatten(1))
        if len(self.cat_embeddings) > 0 and x_cat.shape[1] > 0:
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            embeds.append(torch.cat(cat_embeds, dim=1))
        x = torch.cat(embeds, dim=1)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.head_norm(x)
        return self.head(x).squeeze(-1)


class TabularResNetModel(BaseModel):
    """
    Domain 3: Un-distilled True Neural Diversity Model.
    Employs Periodic Linear Embeddings + Residual Skip Connections.
    Strictly trained directly on binary targets (zero teacher distillation)
    to force Pearson correlation r <= 0.930 with GBDTs.
    """

    def __init__(self, params: dict[str, Any] = None, device: str = "cpu"):
        self.params = params.copy() if params else {}
        self.device_str = device
        self.model = None
        self.cat_cols = []
        self.num_cols = []
        self.num_mean = None
        self.num_std = None

    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None, **kwargs) -> None:
        import torch
        from sklearn.metrics import roc_auc_score
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        if self.device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            use_amp = True
        else:
            device = torch.device("cpu")
            use_amp = False

        logging.info(f"TabularResNet (PLR) initializing on device: {device} (AMP: {use_amp})")

        # Feature subset partitioning: isolate raw continuous and categoricals
        cat_candidates = [
            "Gender", "Customer Type", "Type of Travel", "Class",
            "class_x_travel_type", "gate_x_business", "multi_cross_1", "multi_cross_2",
        ]
        self.cat_cols = [c for c in cat_candidates if c in X_train.columns]
        self.num_cols = [c for c in X_train.columns if c not in self.cat_cols]

        # Robust scaling bounded by 2% and 98% quantiles
        X_num_tr_raw = X_train[self.num_cols].values.astype(np.float32)
        X_num_val_raw = X_val[self.num_cols].values.astype(np.float32)

        q_low = np.nanquantile(X_num_tr_raw, 0.02, axis=0)
        q_high = np.nanquantile(X_num_tr_raw, 0.98, axis=0)
        q_high = np.where(q_high == q_low, q_high + 1e-5, q_high)

        X_num_tr_clipped = np.clip(X_num_tr_raw, q_low, q_high)
        X_num_val_clipped = np.clip(X_num_val_raw, q_low, q_high)

        self.num_mean = np.nanmean(X_num_tr_clipped, axis=0)
        self.num_std = np.nanstd(X_num_tr_clipped, axis=0) + 1e-6

        X_tr_num = np.nan_to_num((X_num_tr_clipped - self.num_mean) / self.num_std)
        X_val_num = np.nan_to_num((X_num_val_clipped - self.num_mean) / self.num_std)

        cat_cards = []
        if self.cat_cols:
            X_tr_cat = np.clip(X_train[self.cat_cols].values.astype(np.int64), 0, None)
            X_val_cat = np.clip(X_val[self.cat_cols].values.astype(np.int64), 0, None)
            for i in range(len(self.cat_cols)):
                card = int(max(X_tr_cat[:, i].max(), X_val_cat[:, i].max())) + 1
                cat_cards.append(card)
        else:
            X_tr_cat = np.zeros((len(X_train), 0), dtype=np.int64)
            X_val_cat = np.zeros((len(X_val), 0), dtype=np.int64)

        d_block = self.params.get("d_block", 256)
        n_blocks = self.params.get("n_blocks", 3)
        d_embedding = self.params.get("d_embedding", 32)
        n_frequencies = self.params.get("n_frequencies", 16)
        dropout = self.params.get("dropout", 0.15)
        lr = self.params.get("lr", 1e-3)
        weight_decay = self.params.get("weight_decay", 1e-4)
        batch_size = self.params.get("batch_size", 4096)
        epochs = self.params.get("epochs", 16)
        loss_type = self.params.get("loss_type", "bce")

        self.model = TabularResNetNet(
            n_num=len(self.num_cols),
            cat_cardinalities=cat_cards,
            d_block=d_block,
            n_blocks=n_blocks,
            d_embedding=d_embedding,
            n_frequencies=n_frequencies,
            dropout=dropout,
        ).to(device)

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            logging.info(f"Distributing TabularResNet across {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        if sample_weight is not None:
            sw_tensor = torch.tensor(sample_weight, dtype=torch.float32)
            train_ds = TensorDataset(
                torch.tensor(X_tr_num, dtype=torch.float32),
                torch.tensor(X_tr_cat, dtype=torch.long),
                torch.tensor(y_train, dtype=torch.float32),
                sw_tensor,
            )
        else:
            train_ds = TensorDataset(
                torch.tensor(X_tr_num, dtype=torch.float32),
                torch.tensor(X_tr_cat, dtype=torch.long),
                torch.tensor(y_train, dtype=torch.float32),
            )

        n_workers = 2 if device.type == "cuda" else 0
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = (
            SurrogateAUCLoss(gamma=self.params.get("gamma", 15.0))
            if loss_type == "surrogate_auc"
            else nn.BCEWithLogitsLoss(reduction="none")
        )

        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

        best_auc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_acc = 0.0

            for batch in train_loader:
                if len(batch) == 4:
                    b_num, b_cat, b_y, b_w = batch
                    b_w = b_w.to(device)
                else:
                    b_num, b_cat, b_y = batch
                    b_w = None

                b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    logits = self.model(b_num, b_cat)
                    if loss_type == "surrogate_auc":
                        loss = criterion(logits, b_y)
                    else:
                        loss_raw = criterion(logits, b_y)
                        loss = torch.mean(loss_raw * b_w) if b_w is not None else torch.mean(loss_raw)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_acc += loss.item()

            scheduler.step()

            # Chunked Validation Evaluation (2048 to prevent OOM)
            self.model.eval()
            val_probs = []
            val_chunk = 2048
            with torch.no_grad():
                for idx in range(0, len(X_val_num), val_chunk):
                    v_num = torch.tensor(X_val_num[idx : idx + val_chunk], dtype=torch.float32).to(device)
                    v_cat = torch.tensor(X_val_cat[idx : idx + val_chunk], dtype=torch.long).to(device)
                    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                        logits = self.model(v_num, v_cat)
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())

            val_preds = np.concatenate(val_probs, axis=0)
            epoch_auc = roc_auc_score(y_val, val_preds)

            if epoch_auc > best_auc:
                best_auc = epoch_auc
                m_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                best_state = {k: v.cpu().clone() for k, v in m_to_save.state_dict().items()}

            logging.info(
                f"ResNet Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss_acc / len(train_loader):.4f} - Val ROC-AUC: {epoch_auc:.5f} (Best: {best_auc:.5f})"
            )

        if best_state is not None:
            m_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            m_to_save.load_state_dict(best_state)
            logging.info(f"Loaded Best TabularResNet State (Validation ROC-AUC: {best_auc:.5f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    def predict_proba(self, X) -> np.ndarray:
        import torch

        device = next(self.model.parameters()).device
        self.model.eval()

        X_num_raw = X[self.num_cols].values.astype(np.float32)
        X_num = np.nan_to_num((X_num_raw - self.num_mean) / self.num_std)

        if self.cat_cols:
            X_cat = np.clip(X[self.cat_cols].values.astype(np.int64), 0, None)
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.int64)

        n_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
        batch_size = 2048 * max(1, n_gpus)
        probs = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                b_num = torch.tensor(X_num[i : i + batch_size], dtype=torch.float32).to(device)
                b_cat = torch.tensor(X_cat[i : i + batch_size], dtype=torch.long).to(device)
                logits = self.model(b_num, b_cat)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return np.concatenate(probs, axis=0)


class DisjointSurveyEmbedding(_ModuleBase):
    """
    Explicitly separates '0' (N/A) from ordinal 1-5 responses.
    Prevents metric distortion in the continuous embedding manifold.
    """

    def __init__(self, num_survey_cols: int, emb_dim: int = 8):
        super().__init__()
        self.num_survey_cols = num_survey_cols
        self.emb_dim = emb_dim
        # Dedicated N/A embedding vectors for each column
        self.na_embeddings = nn.Parameter(torch.randn(num_survey_cols, emb_dim) * 0.02)
        # Ordinal embeddings for 1-5 (size 6 to accommodate 0-5 indexing safely)
        self.ordinal_embeddings = nn.Embedding(6, emb_dim)
        nn.init.normal_(self.ordinal_embeddings.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        out = torch.zeros(
            batch_size, self.num_survey_cols, self.emb_dim, device=x.device, dtype=torch.float32
        )
        is_na = (x == 0)
        is_ordinal = (x > 0)

        if is_ordinal.any():
            out[is_ordinal] = self.ordinal_embeddings(x[is_ordinal].clamp(0, 5))

        if is_na.any():
            na_expanded = self.na_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            out[is_na] = na_expanded[is_na]

        return out.reshape(batch_size, -1)


class PiecewiseLinearSplineEmbedding(_ModuleBase):
    """
    Robust Piecewise Linear Spline embedding for extreme non-linearities.
    Boundaries are initialized with empirical quantiles from the train set.
    """

    def __init__(self, num_features: int, num_bins: int = 16):
        super().__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        initial_b = torch.linspace(0.0, 1.0, num_bins + 1).view(1, 1, -1).repeat(1, num_features, 1)
        self.register_buffer("boundaries", initial_b)

    def set_boundaries(self, quantiles_matrix: np.ndarray) -> None:
        """quantiles_matrix shape: (num_features, num_bins + 1)"""
        tensor_b = torch.tensor(quantiles_matrix, dtype=torch.float32).unsqueeze(0)
        if self.boundaries.shape != tensor_b.shape:
            self.boundaries = tensor_b.to(self.boundaries.device)
        else:
            self.boundaries.copy_(tensor_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_unsqueezed = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        b_lower = self.boundaries[:, :, :-1]
        b_upper = self.boundaries[:, :, 1:]
        widths = b_upper - b_lower

        activations = (x_unsqueezed - b_lower) / (widths + 1e-8)
        activations = torch.clamp(activations, min=0.0, max=1.0)
        return activations.reshape(x.size(0), -1)


class NTPLinear(_ModuleBase):
    """
    Neural Tangent Parametrization Linear Layer.
    Stabilizes gradient magnitudes independent of layer width: z = (1 / sqrt(d_in)) * W * x + b
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        scale = 1.0 / math.sqrt(self.in_features)
        return F.linear(x, self.weight * scale, self.bias)


class TabM_BatchEnsembleLayer(_ModuleBase):
    """
    Parameter-Efficient Ensemble layer utilizing BatchEnsemble principles.
    LinearBE(X) = ((X * R) W) * S + B
    """

    def __init__(self, in_features: int, out_features: int, k_ensembles: int = 16):
        super().__init__()
        self.k = k_ensembles
        self.linear = NTPLinear(in_features, out_features)

        # Rank-1 adapters for each of the k ensemble members
        self.R = nn.Parameter(torch.ones(k_ensembles, in_features))
        self.S = nn.Parameter(torch.ones(k_ensembles, out_features))
        self.B = nn.Parameter(torch.zeros(k_ensembles, out_features))

        # TabM-style initialization
        nn.init.normal_(self.R, mean=1.0, std=0.05)
        nn.init.normal_(self.S, mean=1.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_adapted = x * self.R.unsqueeze(0)

        batch_size = x.size(0)
        x_flat = x_adapted.view(batch_size * self.k, -1)
        z_flat = self.linear(x_flat)
        z = z_flat.view(batch_size, self.k, -1)

        out = z * self.S.unsqueeze(0) + self.B.unsqueeze(0)
        return out


class RealMLP_TabM_Hybrid(_ModuleBase):
    """
    Complete hybrid blueprint executing RealMLP-TD inside a TabM structure.
    Integrates Disjoint Survey Embeddings, Piecewise Linear Spline Embeddings,
    Soft Feature Selection, and Parametric SELU.
    """

    def __init__(
        self,
        num_survey_cols: int,
        num_cont_cols: int,
        emb_dim: int = 8,
        num_bins: int = 16,
        hidden_dim: int = 384,
        k_ensembles: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k_ensembles
        self.dropout = dropout

        self.survey_embedder = DisjointSurveyEmbedding(num_survey_cols, emb_dim=emb_dim)
        self.ple_embedder = PiecewiseLinearSplineEmbedding(num_cont_cols, num_bins=num_bins)

        in_dim = (num_survey_cols * emb_dim) + (num_cont_cols * num_bins)

        # Soft feature selection scaling
        self.feature_scaling = nn.Parameter(torch.ones(in_dim))
        # Ensemble view expansion
        self.ensemble_expansion = nn.Parameter(torch.ones(k_ensembles, in_dim))

        # Deep TabM backbone
        self.block1 = TabM_BatchEnsembleLayer(in_dim, hidden_dim, k_ensembles)
        self.block2 = TabM_BatchEnsembleLayer(hidden_dim, hidden_dim, k_ensembles)
        self.block3 = TabM_BatchEnsembleLayer(hidden_dim, hidden_dim, k_ensembles)

        self.head = TabM_BatchEnsembleLayer(hidden_dim, 1, k_ensembles)

        # Parametric SELU: (1 - alpha) * x + alpha * F.selu(x)
        self.alpha1 = nn.Parameter(torch.ones(hidden_dim))
        self.alpha2 = nn.Parameter(torch.ones(hidden_dim))
        self.alpha3 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, survey_x: torch.Tensor, cont_x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        emb_survey = self.survey_embedder(survey_x)
        emb_cont = self.ple_embedder(cont_x)
        x = torch.cat([emb_survey, emb_cont], dim=1)

        # Apply soft feature selection
        x = x * self.feature_scaling.unsqueeze(0)

        # TabM ensemble expansion: (batch_size, k, in_dim)
        x = x.unsqueeze(1) * self.ensemble_expansion.unsqueeze(0)

        # Backbone pass with parametric SELU
        x = self.block1(x)
        x = (1.0 - self.alpha1) * x + self.alpha1 * F.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.block2(x)
        x = (1.0 - self.alpha2) * x + self.alpha2 * F.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.block3(x)
        x = (1.0 - self.alpha3) * x + self.alpha3 * F.selu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.head(x).squeeze(-1)  # (batch_size, k)

        if self.training:
            return logits
        else:
            return torch.mean(logits, dim=1)


class TabMSurrogateAUCLoss(_ModuleBase):
    """
    Differentiable margin ranking loss maximizing ROC-AUC directly across TabM ensemble heads.
    """

    def __init__(self, gamma: float = 15.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        targets = targets.unsqueeze(1).expand_as(logits)
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        k_count = logits.size(1)
        valid_heads = 0
        for k_idx in range(k_count):
            k_logits = logits[:, k_idx]
            k_targets = targets[:, k_idx]
            pos_logits = k_logits[k_targets == 1]
            neg_logits = k_logits[k_targets == 0]
            if len(pos_logits) == 0 or len(neg_logits) == 0:
                continue
            diffs = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
            member_loss = torch.mean((1.0 - torch.sigmoid(self.gamma * diffs)) ** 2)
            total_loss = total_loss + member_loss
            valid_heads += 1

        return total_loss / max(1, valid_heads)


class RealMLPTabMModel(BaseModel):
    """
    RealMLP-TD + TabM (BatchEnsemble) Hybrid Model.
    Employs Disjoint Survey Embeddings (0-isolation), Piecewise Linear Spline Embeddings,
    Parametric SELU, and Surrogate AUC loss with k=16 ensemble heads.
    Directly breaks collinearity with GBDTs.
    """

    def __init__(self, params: dict[str, Any] = None, device: str = "cpu"):
        self.params = params.copy() if params else {}
        self.device_str = device
        self.model = None
        self.survey_cols: list[str] = []
        self.cont_cols: list[str] = []
        self.quantiles: np.ndarray | None = None

    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None, **kwargs) -> None:
        import torch
        from sklearn.metrics import roc_auc_score
        from torch.utils.data import DataLoader, TensorDataset

        if self.device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            use_amp = True
        else:
            device = torch.device("cpu")
            use_amp = False

        logging.info(f"RealMLP-TabM Hybrid initializing on device: {device} (AMP: {use_amp})")

        # 1. Feature Partitioning
        standard_survey_cols = [
            "Inflight wifi service",
            "Departure/Arrival time convenient",
            "Ease of Online booking",
            "Gate location",
            "Food and drink",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Cleanliness",
        ]
        self.survey_cols = [c for c in standard_survey_cols if c in X_train.columns]
        self.cont_cols = [c for c in X_train.columns if c not in self.survey_cols]

        num_survey = len(self.survey_cols)
        num_cont = len(self.cont_cols)
        num_bins = self.params.get("num_bins", 16)
        hidden_dim = self.params.get("hidden_dim", 384)
        k_ensembles = self.params.get("k_ensembles", 16)
        dropout = self.params.get("dropout", 0.1)
        lr = self.params.get("lr", 1e-3)
        weight_decay = self.params.get("weight_decay", 1e-4)
        batch_size = self.params.get("batch_size", 4096)
        epochs = self.params.get("epochs", 32)
        gamma = self.params.get("gamma", 15.0)

        # 2. Compute empirical quantiles for continuous columns
        X_cont_tr_raw = X_train[self.cont_cols].fillna(0).values.astype(np.float32)
        q_steps = np.linspace(0.0, 1.0, num_bins + 1)
        self.quantiles = np.nanquantile(X_cont_tr_raw, q_steps, axis=0).T
        for row in range(len(self.quantiles)):
            self.quantiles[row] = np.maximum.accumulate(self.quantiles[row])

        X_cont_va_raw = X_val[self.cont_cols].fillna(0).values.astype(np.float32)

        X_survey_tr = np.clip(X_train[self.survey_cols].fillna(0).values.astype(np.int64), 0, 5)
        X_survey_va = np.clip(X_val[self.survey_cols].fillna(0).values.astype(np.int64), 0, 5)

        emb_dim = self.params.get("emb_dim", 8)
        self.model = RealMLP_TabM_Hybrid(
            num_survey_cols=num_survey,
            num_cont_cols=num_cont,
            emb_dim=emb_dim,
            num_bins=num_bins,
            hidden_dim=hidden_dim,
            k_ensembles=k_ensembles,
            dropout=dropout,
        )
        self.model.ple_embedder.set_boundaries(self.quantiles)
        self.model = self.model.to(device)

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            logging.info(f"Distributing RealMLP-TabM across {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        train_ds = TensorDataset(
            torch.tensor(X_survey_tr, dtype=torch.long),
            torch.tensor(X_cont_tr_raw, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )

        n_workers = 2 if device.type == "cuda" else 0
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = TabMSurrogateAUCLoss(gamma=gamma)
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

        best_auc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_acc = 0.0

            for b_survey, b_cont, b_y in train_loader:
                b_survey = b_survey.to(device)
                b_cont = b_cont.to(device)
                b_y = b_y.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    logits = self.model(b_survey, b_cont)
                    loss = criterion(logits, b_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss_acc += loss.item()

            scheduler.step()

            # Validation evaluation
            self.model.eval()
            val_probs = []
            val_chunk = 4096
            with torch.no_grad():
                for idx in range(0, len(X_survey_va), val_chunk):
                    v_s = torch.tensor(X_survey_va[idx : idx + val_chunk], dtype=torch.long).to(device)
                    v_c = torch.tensor(X_cont_va_raw[idx : idx + val_chunk], dtype=torch.float32).to(device)
                    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                        mean_logits = self.model(v_s, v_c)
                    val_probs.append(torch.sigmoid(mean_logits).cpu().numpy())

            val_preds = np.concatenate(val_probs, axis=0)
            epoch_auc = roc_auc_score(y_val, val_preds)

            if epoch_auc > best_auc:
                best_auc = epoch_auc
                m_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                best_state = {k: v.cpu().clone() for k, v in m_to_save.state_dict().items()}

            logging.info(
                f"RealMLP-TabM Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss_acc / len(train_loader):.4f} - Val ROC-AUC: {epoch_auc:.5f} (Best: {best_auc:.5f})"
            )

        if best_state is not None:
            m_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            m_to_save.load_state_dict(best_state)
            logging.info(f"Loaded Best RealMLP-TabM State (Validation ROC-AUC: {best_auc:.5f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        import torch

        device = next(self.model.parameters()).device
        self.model.eval()

        X_survey = np.clip(X[self.survey_cols].fillna(0).values.astype(np.int64), 0, 5)
        X_cont = X[self.cont_cols].fillna(0).values.astype(np.float32)

        n_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
        batch_size = 4096 * max(1, n_gpus)
        probs = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                b_s = torch.tensor(X_survey[i : i + batch_size], dtype=torch.long).to(device)
                b_c = torch.tensor(X_cont[i : i + batch_size], dtype=torch.float32).to(device)
                mean_logits = self.model(b_s, b_c)
                p = torch.sigmoid(mean_logits).cpu().numpy()
                probs.append(p)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return np.concatenate(probs, axis=0)


def get_model(
    model_name: str, params: dict[str, Any] = None, device: str = "cpu"
) -> BaseModel:
    """Factory function for model instantiation."""
    model_name_lower = model_name.lower()
    if "lgb" in model_name_lower or "lightgbm" in model_name_lower:
        return LightGBMModel(params=params, device=device)
    elif "cat" in model_name_lower or "catboost" in model_name_lower:
        return CatBoostModel(params=params, device=device)
    elif "xgb" in model_name_lower or "xgboost" in model_name_lower:
        return XGBoostModel(params=params, device=device)
    elif (
        "realmlp" in model_name_lower
        or "tabm" in model_name_lower
        or "hybrid" in model_name_lower
    ):
        return RealMLPTabMModel(params=params, device=device)
    elif (
        "resnet" in model_name_lower
        or "tabular_resnet" in model_name_lower
        or "nn" in model_name_lower
    ):
        return TabularResNetModel(params=params, device=device)
    elif (
        "transformer" in model_name_lower
        or "ft" in model_name_lower
    ):
        return FTTransformerModel(params=params, device=device)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Choose from 'lightgbm', 'catboost', 'xgboost', 'realmlp', 'tabm', 'resnet', 'transformer'."
        )
