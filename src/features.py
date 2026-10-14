"""
Feature Engineering Pipeline: Implements All 10 Empirically Verified Paradigms
Plus Domain A (Density & Frequency Forensics) and Domain C (Transductive Encodings)
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, TargetEncoder
import lightgbm as lgb
from scipy.special import expit, logit

# Suppress pandas fragmentation warnings when engineering 100+ columns
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.config import FeatureConfig
from src.utils import reduce_mem_usage, resolve_binary_target, timer


class FeaturePipeline:
    """
    Production-grade Feature Engineering Pipeline.
    Implements:
    - 10 Verified Empirical Paradigms (Delay dynamics, Simpson's law, Rasch IRT, Golden/Dead zones, etc.)
    - Domain A: High-Order Categorical Frequency Encodings (sampling density proxy)
    - Domain A: Geometric Centroid & Sub-Cluster Distance Features (Manifold proximity)
    - Domain A: Local Outlier / Mode Collapse Density Score
    - Domain C: Safe Transductive Feature Statistics on concat(train, test)
    - Domain 4: Rotational Manifold SVD & Multi-Way Bayesian Target Encoding
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.label_encoder_dicts: dict[str, dict[str, int]] = {}
        self.freq_maps: dict[str, dict[str, float]] = {}
        self.count_maps: dict[str, dict[str, int]] = {}

        # Centroid and density parameters
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self.global_centroid_1: np.ndarray | None = None
        self.global_centroid_0: np.ndarray | None = None
        self.mbk_1: MiniBatchKMeans | None = None
        self.mbk_0: MiniBatchKMeans | None = None
        self.mbk_anomaly: MiniBatchKMeans | None = None

        self.svd: TruncatedSVD | None = None
        self.svd_cols: list[str] = []
        self.svd_mean: np.ndarray | None = None
        self.svd_std: np.ndarray | None = None
        self.svd_rating_impute: dict[str, float] = {}
        self.target_encoding_maps: dict[str, dict[Any, float]] = {}
        self.global_target_mean: float = 0.5
        self.train_oof_te: dict[str, np.ndarray] = {}

        # Advanced Domain Extensions (Rugved Bane 0.96059 LB + Friend 1/2/3 Findings)
        self.orig_prior_model = None
        self.orig_prior_cols: list[str] = []
        self.te_flight_distance_map: dict[float, float] = {}
        self.freq_flight_distance_map: dict[float, float] = {}
        self.count_flight_distance_map: dict[float, int] = {}
        self.route_mean_arr_map: dict[float, float] = {}
        self.route_std_arr_map: dict[float, float] = {}
        self.bounded_composite_cols: list[str] = [
            "route_class_travel",
            "route_delay_tier",
            "route_dissatisfaction",
            "service_failure_class",
            "age_class_travel",
        ]

        # Route Profile & Group Conditional Residuals (Koumei Maki +124 bps)
        self.route_profile_means: dict[str, dict[Any, float]] = {}
        self.route_fd_cnt: dict[Any, int] = {}
        self.num_value_counts: dict[str, dict[Any, int]] = {}
        self.orig_lookup_maps: dict[str, dict[Any, float]] = {}
        self.orig_prior_value: float = 0.5

        # Auxiliary Expected Rating Priors (Friend 1 Methodology)
        self.expected_rating_models: dict[str, Any] = {}
        self.expected_rating_features: list[str] = []

        self.fitted: bool = False

    def _get_bounded_crosses(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Creates bounded high-cardinality composite interactions (1,000-3,000 state capacity)
        for CatBoost CTR stability and LightGBM orthogonalization.
        """
        flight_dist = (
            df["Flight Distance"]
            if "Flight Distance" in df.columns
            else pd.Series(1000, index=df.index)
        )
        dist_tier = (flight_dist // 250).clip(0, 19).astype(str)

        age = df["Age"] if "Age" in df.columns else pd.Series(40, index=df.index)
        age_tier = (age // 10).clip(0, 8).astype(str)

        cls = df["Class"].astype(str) if "Class" in df.columns else "Eco"
        travel = (
            df["Type of Travel"].astype(str)
            if "Type of Travel" in df.columns
            else "Business travel"
        )

        rating_cols = [c for c in self.config.rating_cols if c in df.columns]
        if rating_cols:
            ratings = df[rating_cols]
            dissat_count = (
                ((ratings <= 2) & (ratings > 0)).sum(axis=1).clip(0, 5).astype(str)
            )
        else:
            dissat_count = pd.Series("0", index=df.index)

        dep_d = (
            df["Departure Delay in Minutes"].fillna(0)
            if "Departure Delay in Minutes" in df.columns
            else pd.Series(0, index=df.index)
        )
        arr_d = (
            df["Arrival Delay in Minutes"].fillna(dep_d)
            if "Arrival Delay in Minutes" in df.columns
            else dep_d
        )
        tot_delay = dep_d + arr_d
        delay_tier = pd.Series(
            np.select(
                [tot_delay == 0, tot_delay < 15, tot_delay >= 15],
                [0, 1, 2],
                default=0,
            ),
            index=df.index,
        ).astype(str)

        crosses = {
            "route_class_travel": dist_tier + "_" + cls + "_" + travel,
            "route_delay_tier": dist_tier + "_" + delay_tier,
            "route_dissatisfaction": dist_tier + "_" + dissat_count,
            "service_failure_class": dissat_count + "_" + cls + "_" + travel,
            "age_class_travel": age_tier + "_" + cls + "_" + travel,
        }

        # Shelton Wang (11th place, Topic #745892): 39 Rating-Context Crosses (+34 bps across all 5 folds)
        # Each service rating crossed with Class, Type of Travel, Customer Type
        context_cols = [
            c for c in ["Class", "Type of Travel", "Customer Type"] if c in df.columns
        ]
        for rc in rating_cols:
            r_str = df[rc].astype(str)
            for ctx in context_cols:
                cross_name = f"{rc}_x_{ctx}"
                crosses[cross_name] = r_str + "|" + df[ctx].astype(str)
                crosses[f"{rc}__X__{ctx}"] = r_str + "|" + df[ctx].astype(str)

        # Shelton Wang (Topic #745892): All 13 service ratings as categories
        for rc in rating_cols:
            if rc in df.columns:
                crosses[f"{rc}__cat"] = df[rc].astype(str)

        # Shelton Wang (Topic #745892): 4 numerical features copied as exact-value categorical columns
        numeric_to_copy = [
            "Age",
            "Flight Distance",
            "Departure Delay in Minutes",
            "Arrival Delay in Minutes",
        ]
        for col in numeric_to_copy:
            if col in df.columns:
                copy_name = f"{col}__category"
                crosses[copy_name] = (
                    df[col].astype("string").fillna("__MISSING__").astype(str)
                )

        return crosses

    def _get_multi_crosses(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Generates high-cardinality multi-way topological crosses."""
        cls = df["Class"].astype(str) if "Class" in df.columns else "Unknown"
        travel = (
            df["Type of Travel"].astype(str)
            if "Type of Travel" in df.columns
            else "Unknown"
        )
        gate = df["Gate location"].astype(str) if "Gate location" in df.columns else "0"
        wifi = (
            df["Inflight wifi service"].astype(str)
            if "Inflight wifi service" in df.columns
            else "0"
        )
        booking = (
            df["Ease of Online booking"].astype(str)
            if "Ease of Online booking" in df.columns
            else "0"
        )

        mc1 = cls + "_" + travel + "_" + gate
        mc2 = wifi + "_" + booking
        return mc1, mc2

    def _get_frequency_keys(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Generates composite interaction keys for high-order frequency analysis."""
        age_tier = (df["Age"] // 10).astype(str)
        key1 = (
            df["Class"].astype(str)
            + "_"
            + df["Type of Travel"].astype(str)
            + "_"
            + df["Inflight wifi service"].astype(str)
            + "_"
            + df["Online boarding"].astype(str)
        )
        key2 = (
            df["Class"].astype(str)
            + "_"
            + df["Customer Type"].astype(str)
            + "_"
            + df["Online boarding"].astype(str)
            + "_"
            + df["Checkin service"].astype(str)
        )
        key3 = (
            df["Type of Travel"].astype(str)
            + "_"
            + df["Class"].astype(str)
            + "_"
            + df["Online boarding"].astype(str)
            + "_"
            + df["Seat comfort"].astype(str)
        )
        key4 = (
            df["Class"].astype(str)
            + "_"
            + df["Type of Travel"].astype(str)
            + "_"
            + age_tier
        )
        return {"key1": key1, "key2": key2, "key3": key3, "key4": key4}

    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        orig_df: pd.DataFrame | None = None,
    ) -> "FeaturePipeline":
        """Fits transductive frequency statistics, geometric centroids, route profiles, and label encoders."""
        with timer(
            "Fitting FeaturePipeline (Transductive Density & Geometric Forensics)"
        ):
            # 1. Prepare combined dataframe for transductive frequency and encoder fitting
            if test_df is not None:
                full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            else:
                full_df = train_df

            # 2. High-Order Categorical Frequency Maps
            if self.config.enable_high_order_freq:
                full_keys = self._get_frequency_keys(full_df)
                total_rows = len(full_df)
                for name, series in full_keys.items():
                    val_counts = series.value_counts()
                    self.count_maps[name] = val_counts.to_dict()
                    self.freq_maps[name] = (val_counts / total_rows).to_dict()

            # 3. Geometric Centroid & Density Modeling (Domain A)
            if self.config.enable_density_forensics:
                core_cols = self.config.core_centroid_cols
                # Pre-impute arrival delay for standardizer
                arr_delay_train = train_df["Arrival Delay in Minutes"].fillna(
                    train_df["Departure Delay in Minutes"]
                )
                train_core = train_df[core_cols].copy()
                train_core["Arrival Delay in Minutes"] = arr_delay_train

                self.scaler_mean = train_core.mean(axis=0).values.astype(np.float32)
                self.scaler_std = (train_core.std(axis=0) + 1e-6).values.astype(
                    np.float32
                )

                X_train_scaled = (
                    (train_core.values - self.scaler_mean) / self.scaler_std
                ).astype(np.float32)
                y_train = (
                    resolve_binary_target(train_df[self.config.target_col])
                    if self.config.target_col in train_df.columns
                    else None
                )

                if (
                    y_train is not None
                    and (y_train == 1).sum() > 0
                    and (y_train == 0).sum() > 0
                ):
                    pos_mask = y_train == 1
                    neg_mask = y_train == 0
                    # Positive and negative global centroids
                    self.global_centroid_1 = X_train_scaled[pos_mask].mean(axis=0)
                    self.global_centroid_0 = X_train_scaled[neg_mask].mean(axis=0)

                    # Sub-cluster centroids for positive & negative manifolds
                    n_c = self.config.kmeans_clusters_per_class
                    self.mbk_1 = MiniBatchKMeans(
                        n_clusters=n_c, batch_size=4096, random_state=42, n_init=3
                    ).fit(X_train_scaled[pos_mask])

                    self.mbk_0 = MiniBatchKMeans(
                        n_clusters=n_c, batch_size=4096, random_state=42, n_init=3
                    ).fit(X_train_scaled[neg_mask])
                else:
                    self.global_centroid_1 = None
                    self.global_centroid_0 = None
                    self.mbk_1 = None
                    self.mbk_0 = None

                # Local Outlier / Mode Collapse Anomaly Clustering on full data
                full_arr_delay = full_df["Arrival Delay in Minutes"].fillna(
                    full_df["Departure Delay in Minutes"]
                )
                full_core = full_df[core_cols].copy()
                full_core["Arrival Delay in Minutes"] = full_arr_delay
                X_full_scaled = (
                    (full_core.values - self.scaler_mean) / self.scaler_std
                ).astype(np.float32)

                self.mbk_anomaly = MiniBatchKMeans(
                    n_clusters=self.config.kmeans_anomaly_clusters,
                    batch_size=4096,
                    random_state=42,
                    n_init=3,
                ).fit(X_full_scaled)

            # 4. Fit LabelEncoders on full combined data
            cat_columns = [
                "Gender",
                "Customer Type",
                "Type of Travel",
                "Class",
                "class_x_travel_type",
                "gate_x_business",
                "multi_cross_1",
                "multi_cross_2",
            ]

            # Create synthetic composite categories on full_df for consistent encoder fit
            temp_is_bus = (full_df["Type of Travel"] == "Business travel").astype(
                np.int8
            )
            full_class_travel = (
                full_df["Class"].astype(str)
                + "_"
                + full_df["Type of Travel"].astype(str)
            )
            full_gate_bus = (
                full_df["Gate location"].astype(str) + "_" + temp_is_bus.astype(str)
            )
            mc1_full, mc2_full = self._get_multi_crosses(full_df)

            col_data_map = {
                "Gender": full_df["Gender"].astype(str),
                "Customer Type": full_df["Customer Type"].astype(str),
                "Type of Travel": full_df["Type of Travel"].astype(str),
                "Class": full_df["Class"].astype(str),
                "class_x_travel_type": full_class_travel,
                "gate_x_business": full_gate_bus,
                "multi_cross_1": mc1_full.astype(str),
                "multi_cross_2": mc2_full.astype(str),
            }

            # Bounded high-cardinality composite interactions for CatBoost CTR stability
            if self.config.enable_bounded_crosses:
                bounded_full = self._get_bounded_crosses(full_df)
                for b_col, b_series in bounded_full.items():
                    if b_col not in cat_columns:
                        cat_columns.append(b_col)
                    col_data_map[b_col] = b_series.astype(str)

            for col in cat_columns:
                le = LabelEncoder()
                le.fit(col_data_map[col])
                self.label_encoders[col] = le
                self.label_encoder_dicts[col] = {
                    val: idx for idx, val in enumerate(le.classes_)
                }

            # 5. Linear Rotational Variance via TruncatedSVD (Domain 4)
            if self.config.enable_svd_manifolds:
                self.svd_cols = [
                    c
                    for c in (self.config.numerical_cols + self.config.rating_cols)
                    if c in full_df.columns
                ]
                svd_data = full_df[self.svd_cols].copy()
                if (
                    "Arrival Delay in Minutes" in svd_data.columns
                    and "Departure Delay in Minutes" in svd_data.columns
                ):
                    svd_data["Arrival Delay in Minutes"] = svd_data[
                        "Arrival Delay in Minutes"
                    ].fillna(svd_data["Departure Delay in Minutes"])
                # Treat survey rating 0 as N/A (replace with answered mean so metric space is not distorted)
                for rc in self.config.rating_cols:
                    if rc in svd_data.columns:
                        pos_vals = svd_data.loc[svd_data[rc] > 0, rc]
                        mean_val = float(pos_vals.mean()) if len(pos_vals) > 0 else 3.0
                        self.svd_rating_impute[rc] = mean_val
                        svd_data[rc] = svd_data[rc].replace(0, mean_val)

                self.svd_mean = svd_data.mean(axis=0).values.astype(np.float32)
                self.svd_std = (svd_data.std(axis=0) + 1e-6).values.astype(np.float32)
                X_svd_norm = ((svd_data.values - self.svd_mean) / self.svd_std).astype(
                    np.float32
                )

                self.svd = TruncatedSVD(
                    n_components=self.config.n_svd_components, random_state=42
                )
                self.svd.fit(X_svd_norm)

            # 6. Multi-Way Bayesian Target Encoding with Leak-Free OOF (Domain 4)
            if self.config.target_col in train_df.columns:
                y_train_num = resolve_binary_target(
                    train_df[self.config.target_col]
                ).astype(np.float32)
                self.global_target_mean = float(y_train_num.mean())
                smooth_prior = 10.0

                # Build train temporary series for target encoding
                temp_is_bus_tr = (
                    train_df["Type of Travel"] == "Business travel"
                ).astype(np.int8)
                tr_cols_data = {
                    "class_x_travel_type": train_df["Class"].astype(str)
                    + "_"
                    + train_df["Type of Travel"].astype(str),
                    "gate_x_business": train_df["Gate location"].astype(str)
                    + "_"
                    + temp_is_bus_tr.astype(str),
                }
                mc1_tr, mc2_tr = self._get_multi_crosses(train_df)
                tr_cols_data["multi_cross_1"] = mc1_tr.astype(str)
                tr_cols_data["multi_cross_2"] = mc2_tr.astype(str)

                te_target_cols = [
                    "multi_cross_1",
                    "multi_cross_2",
                    "class_x_travel_type",
                    "gate_x_business",
                ]

                # Add bounded crosses to target encoding
                if self.config.enable_bounded_crosses:
                    bounded_tr = self._get_bounded_crosses(train_df)
                    for b_col in ["route_class_travel", "route_dissatisfaction"]:
                        tr_cols_data[b_col] = bounded_tr[b_col].astype(str)
                        if b_col not in te_target_cols:
                            te_target_cols.append(b_col)

                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                for col in te_target_cols:
                    s = tr_cols_data[col]
                    # Global smoothed mapping for test data
                    counts = s.value_counts()
                    sums = y_train_num
                    sum_per_cat = pd.Series(sums).groupby(s.values).sum()
                    smoothed_map = (
                        (sum_per_cat + smooth_prior * self.global_target_mean)
                        / (counts + smooth_prior)
                    ).to_dict()
                    self.target_encoding_maps[col] = smoothed_map

                    # Out-of-fold target encoding for train data
                    oof_te = np.full(
                        len(train_df), self.global_target_mean, dtype=np.float32
                    )
                    for tr_idx, val_idx in kf.split(train_df):
                        s_tr, y_tr = s.iloc[tr_idx], y_train_num[tr_idx]
                        s_va = s.iloc[val_idx]
                        c_tr = s_tr.value_counts()
                        sum_tr = pd.Series(y_tr).groupby(s_tr.values).sum()
                        map_tr = (
                            (sum_tr + smooth_prior * self.global_target_mean)
                            / (c_tr + smooth_prior)
                        ).to_dict()
                        oof_te[val_idx] = (
                            s_va.map(map_tr).fillna(self.global_target_mean).values
                        )
                    self.train_oof_te[col] = oof_te

                # Route Profiles: Flight Distance target encoding & aggregations (Rugved Bane #1 feature)
                if (
                    self.config.enable_route_profiles
                    and "Flight Distance" in train_df.columns
                ):
                    dist_s = train_df["Flight Distance"]
                    c_dist = dist_s.value_counts()
                    sum_dist = pd.Series(y_train_num).groupby(dist_s.values).sum()
                    smooth_prior_dist = 20.0
                    self.te_flight_distance_map = (
                        (sum_dist + smooth_prior_dist * self.global_target_mean)
                        / (c_dist + smooth_prior_dist)
                    ).to_dict()

                    oof_dist_te = np.full(
                        len(train_df), self.global_target_mean, dtype=np.float32
                    )
                    for tr_idx, val_idx in kf.split(train_df):
                        d_tr, y_tr = dist_s.iloc[tr_idx], y_train_num[tr_idx]
                        d_va = dist_s.iloc[val_idx]
                        cd_tr = d_tr.value_counts()
                        sd_tr = pd.Series(y_tr).groupby(d_tr.values).sum()
                        map_dist_tr = (
                            (sd_tr + smooth_prior_dist * self.global_target_mean)
                            / (cd_tr + smooth_prior_dist)
                        ).to_dict()
                        oof_dist_te[val_idx] = (
                            d_va.map(map_dist_tr).fillna(self.global_target_mean).values
                        )
                    self.train_oof_te["Flight Distance"] = oof_dist_te

                    self.freq_flight_distance_map = (
                        dist_s.value_counts() / len(train_df)
                    ).to_dict()
                    self.count_flight_distance_map = dist_s.value_counts().to_dict()
                    arr_del_s = train_df["Arrival Delay in Minutes"].fillna(
                        train_df["Departure Delay in Minutes"]
                    )
                    self.route_mean_arr_map = arr_del_s.groupby(dist_s).mean().to_dict()
                    self.route_std_arr_map = (
                        arr_del_s.groupby(dist_s).std().fillna(0.0).to_dict()
                    )

            # 7. Original Dataset Prior Model (Rugved Bane #2 and #3 features)
            if (
                self.config.enable_orig_prior
                and orig_df is not None
                and len(orig_df) > 0
                and self.config.target_col in orig_df.columns
            ):
                try:
                    from sklearn.ensemble import HistGradientBoostingClassifier

                    prior_cols = [
                        c
                        for c in (
                            self.config.categorical_cols
                            + self.config.numerical_cols
                            + self.config.rating_cols
                        )
                        if c in orig_df.columns and c in train_df.columns
                    ]
                    X_orig_prior = orig_df[prior_cols].copy()
                    for cc in self.config.categorical_cols:
                        if cc in X_orig_prior.columns:
                            m_dict = self.label_encoder_dicts.get(cc, {})
                            X_orig_prior[cc] = (
                                X_orig_prior[cc].astype(str).map(m_dict).fillna(-1).astype(np.int32)
                            )
                    if (
                        "Arrival Delay in Minutes" in X_orig_prior.columns
                        and "Departure Delay in Minutes" in X_orig_prior.columns
                    ):
                        X_orig_prior["Arrival Delay in Minutes"] = X_orig_prior[
                            "Arrival Delay in Minutes"
                        ].fillna(X_orig_prior["Departure Delay in Minutes"])
                    y_orig_prior = resolve_binary_target(
                        orig_df[self.config.target_col]
                    )

                    self.orig_prior_cols = prior_cols
                    self.orig_prior_model = HistGradientBoostingClassifier(
                        max_iter=250,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=42,
                    )
                    self.orig_prior_model.fit(X_orig_prior, y_orig_prior)
                    logging.info(
                        f"Fitted Original Dataset Prior Model (HistGradientBoosting on {len(orig_df)} rows across {len(prior_cols)} features)."
                    )
                except Exception as e:
                    logging.warning(f"Could not fit Original Dataset Prior Model: {e}")
                    self.orig_prior_model = None

            # 8. Route Profile Group Means & Residual Forensics (Koumei Maki +124 bps)
            # Transductive over train + test; Flight Distance contains zero label leakage.
            if "Flight Distance" in full_df.columns:
                base_cols_all = [
                    c
                    for c in (
                        self.config.categorical_cols
                        + self.config.numerical_cols
                        + self.config.rating_cols
                    )
                    if c in full_df.columns
                ]
                self.route_fd_cnt = (
                    full_df["Flight Distance"]
                    .value_counts(dropna=False)
                    .to_dict()
                )

                route_df = full_df[base_cols_all].copy()
                for c in self.config.categorical_cols:
                    if c in route_df.columns:
                        m_dict = self.label_encoder_dicts.get(c, {})
                        route_df[c] = (
                            route_df[c].astype(str).map(m_dict).fillna(-1).astype(np.float32)
                        )
                if (
                    "Arrival Delay in Minutes" in route_df.columns
                    and "Departure Delay in Minutes" in route_df.columns
                ):
                    route_df["Arrival Delay in Minutes"] = route_df[
                        "Arrival Delay in Minutes"
                    ].fillna(route_df["Departure Delay in Minutes"])

                grp_fd = route_df.groupby(full_df["Flight Distance"])
                for c in base_cols_all:
                    if c != "Flight Distance":
                        self.route_profile_means[c] = grp_fd[c].mean().to_dict()

            # 9. Numeric Value Counts (Koumei Maki +7 bps)
            for c in self.config.numerical_cols:
                if c in full_df.columns:
                    self.num_value_counts[c] = (
                        full_df[c].value_counts(dropna=False).to_dict()
                    )

            # 10. Smoothed Original-Data Column Lookups (Koumei Maki +48 bps)
            if (
                orig_df is not None
                and len(orig_df) > 0
                and self.config.target_col in orig_df.columns
            ):
                try:
                    y_orig_num = resolve_binary_target(
                        orig_df[self.config.target_col]
                    )
                    prior_orig = float(y_orig_num.mean())
                    self.orig_prior_value = prior_orig
                    m_smooth = 10.0
                    for c in (
                        self.config.categorical_cols
                        + self.config.numerical_cols
                        + self.config.rating_cols
                    ):
                        if c in orig_df.columns:
                            s_orig = (
                                pd.DataFrame(
                                    {"v": orig_df[c].values, "y": y_orig_num}
                                )
                                .groupby("v", dropna=False)["y"]
                                .agg(["sum", "count"])
                            )
                            self.orig_lookup_maps[c] = (
                                (s_orig["sum"] + m_smooth * prior_orig)
                                / (s_orig["count"] + m_smooth)
                            ).to_dict()
                except Exception as e:
                    logging.warning(
                        f"Could not compute original dataset column lookups: {e}"
                    )

            # 11. Auxiliary Task Expected Rating Priors (Friend 1 Methodology)
            # Fits multiclass GBDT models on orig_df predicting rating r_i in {0..5} given demographic & flight features.
            # Predicts passenger expectations E[r_i | X_{-i}] and expectations residuals with zero target leakage.
            if (
                getattr(self.config, "enable_expected_ratings", True)
                and orig_df is not None
                and len(orig_df) > 0
            ):
                try:
                    from lightgbm import LGBMClassifier

                    rating_pred_cols = [
                        c
                        for c in [
                            "Gender",
                            "Customer Type",
                            "Type of Travel",
                            "Class",
                            "Age",
                            "Flight Distance",
                            "Departure Delay in Minutes",
                            "Arrival Delay in Minutes",
                        ]
                        if c in orig_df.columns
                    ]

                    X_orig_ratings = orig_df[rating_pred_cols].copy()
                    for cat_c in self.config.categorical_cols:
                        if cat_c in X_orig_ratings.columns:
                            m_dict = self.label_encoder_dicts.get(cat_c, {})
                            X_orig_ratings[cat_c] = (
                                X_orig_ratings[cat_c]
                                .astype(str)
                                .map(m_dict)
                                .fillna(-1)
                                .astype(np.int32)
                            )
                    if (
                        "Arrival Delay in Minutes" in X_orig_ratings.columns
                        and "Departure Delay in Minutes" in X_orig_ratings.columns
                    ):
                        X_orig_ratings["Arrival Delay in Minutes"] = X_orig_ratings[
                            "Arrival Delay in Minutes"
                        ].fillna(X_orig_ratings["Departure Delay in Minutes"])
                    X_orig_ratings = X_orig_ratings.fillna(0)

                    self.expected_rating_features = rating_pred_cols
                    for r_col in self.config.rating_cols:
                        if r_col in orig_df.columns:
                            y_r = (
                                pd.to_numeric(orig_df[r_col], errors="coerce")
                                .fillna(0)
                                .astype(int)
                            )
                            if len(np.unique(y_r)) > 1:
                                clf_r = LGBMClassifier(
                                    n_estimators=35,
                                    num_leaves=15,
                                    learning_rate=0.08,
                                    min_child_samples=30,
                                    random_state=42,
                                    n_jobs=-1,
                                    verbose=-1,
                                )
                                clf_r.fit(X_orig_ratings, y_r)
                                self.expected_rating_models[r_col] = clf_r
                    logging.info(
                        f"Fitted {len(self.expected_rating_models)} Auxiliary Expected Rating Priors on {len(orig_df)} rows."
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not fit Auxiliary Expected Rating Models: {e}"
                    )
                    self.expected_rating_models = {}

            self.fitted = True
            return self

    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Applies vectorized feature transformations to training or test data."""
        if not self.fitted:
            raise RuntimeError(
                "FeaturePipeline must be fitted via fit() or fit_transform() before transform()!"
            )

        with timer(f"Feature Engineering ({'Train' if is_train else 'Test'})"):
            data = df.copy()

            # -------------------------------------------------------------
            # 1. PHYSICAL DELAY DYNAMICS & IMPUTATION
            # -------------------------------------------------------------
            data["Arrival_Delay_is_nan"] = (
                data["Arrival Delay in Minutes"].isna().astype(np.int8)
            )
            data["Arrival Delay in Minutes"] = data["Arrival Delay in Minutes"].fillna(
                data["Departure Delay in Minutes"]
            )

            dep_delay = data["Departure Delay in Minutes"]
            arr_delay = data["Arrival Delay in Minutes"]

            data["total_delay"] = dep_delay + arr_delay
            data["has_delay"] = (data["total_delay"] > 0).astype(np.int8)
            data["has_severe_delay"] = (data["total_delay"] > 30).astype(np.int8)

            # Delay Inflection Thresholds & Airborne Dynamics (Rugved Bane & Friends 1/2/3)
            data["delay_over_10"] = (data["total_delay"] > 10).astype(np.int8)
            data["arr_delay_over_10"] = (arr_delay > 10).astype(np.int8)
            data["dep_delay_over_10"] = (dep_delay > 10).astype(np.int8)
            data["delay_diff"] = (arr_delay - dep_delay).astype(np.float32)
            data["route_delay_hazard"] = (
                arr_delay
                / np.maximum(10.0, np.maximum(0.0, data["Flight Distance"]) / 7.5)
            ).astype(np.float32)

            # Airborne Delay Recovery & Difference Dynamics (Golden Features from Research)
            data["Delay_Delta"] = (dep_delay - arr_delay).astype(np.float32)
            data["arr_minus_dep"] = (arr_delay - dep_delay).astype(np.float32)
            data["Recovery_Magnitude"] = np.maximum(0.0, dep_delay - arr_delay).astype(
                np.float32
            )
            data["Compounding_Delay"] = np.maximum(0.0, arr_delay - dep_delay).astype(
                np.float32
            )
            data["delay_recovery_delta"] = (dep_delay - arr_delay).astype(np.float32)
            data["worsened_in_air"] = (arr_delay > dep_delay).astype(np.int8)

            # Delay Intensity per 100 miles
            data["delay_intensity"] = data["total_delay"] / np.maximum(
                1.0, np.maximum(0.0, data["Flight Distance"]) / 100.0
            )

            # Flight Distance digit decomposition & modulo features (karttikjangid05 Topic #745098, +36 bps)
            data["dist_mod_10"] = (data["Flight Distance"] % 10).astype(np.int8)
            data["dist_mod_100"] = (data["Flight Distance"] % 100).astype(np.int16)
            data["dist_div_100"] = (data["Flight Distance"] // 100).astype(np.int16)

            # The 15-Minute Flatline Law
            data["delay_tier"] = np.select(
                [
                    data["total_delay"] == 0,
                    data["total_delay"] < 15,
                    data["total_delay"] >= 15,
                ],
                [0, 1, 2],
                default=0,
            ).astype(np.int8)

            # Reverse-Engineered Delay Anomalies (CTGAN Zero-Dirac Mass & Correlation Artifacts)
            dep_is_0 = (dep_delay == 0)
            arr_is_0 = (arr_delay == 0)
            data["is_both_delays_zero"] = (dep_is_0 & arr_is_0).astype(np.int8)
            data["is_dep_zero_arr_nonzero"] = (dep_is_0 & ~arr_is_0).astype(np.int8)
            data["is_dep_nonzero_arr_zero"] = (~dep_is_0 & arr_is_0).astype(np.int8)
            data["delay_discrepancy"] = np.abs(arr_delay - dep_delay).astype(np.float32)
            data["delay_speedup_ratio"] = (
                (dep_delay - arr_delay) / np.maximum(1.0, dep_delay)
            ).astype(np.float32)

            # -------------------------------------------------------------
            # 2. ZERO-INFLATION & "NOT APPLICABLE" INDICATORS
            # -------------------------------------------------------------
            data["wifi_is_0"] = (data["Inflight wifi service"] == 0).astype(np.int8)
            data["booking_is_0"] = (data["Ease of Online booking"] == 0).astype(np.int8)
            data["boarding_is_0"] = (data["Online boarding"] == 0).astype(np.int8)
            data["time_convenient_is_0"] = (
                data["Departure/Arrival time convenient"] == 0
            ).astype(np.int8)
            data["total_na_ratings"] = (
                (data[self.config.rating_cols] == 0).sum(axis=1).astype(np.int8)
            )

            # The "Zero-Rating Miracle" indicator (all 3 digital services N/A: sat rate > 85%)
            data["is_all_digital_na"] = (
                (data["Inflight wifi service"] == 0)
                & (data["Online boarding"] == 0)
                & (data["Ease of Online booking"] == 0)
            ).astype(np.int8)
            data["any_digital_na"] = (
                (data["Inflight wifi service"] == 0)
                | (data["Online boarding"] == 0)
                | (data["Ease of Online booking"] == 0)
            ).astype(np.int8)

            # N/A-Adjusted Continuous Ratings (restoring monotonic satisfaction ordering for GBDTs)
            # In survey design, 0 is "Not Applicable" with high satisfaction (wifi@0=88.7%, boarding@0=62.0%, booking@0=70.8%)
            # Imputing to their empirical equivalent quantile prevents monotonic split distortions
            data["na_adj_wifi"] = np.where(
                data["Inflight wifi service"] == 0,
                4.5,
                data["Inflight wifi service"].astype(np.float32),
            ).astype(np.float32)
            data["na_adj_boarding"] = np.where(
                data["Online boarding"] == 0,
                3.2,
                data["Online boarding"].astype(np.float32),
            ).astype(np.float32)
            data["na_adj_booking"] = np.where(
                data["Ease of Online booking"] == 0,
                3.8,
                data["Ease of Online booking"].astype(np.float32),
            ).astype(np.float32)

            # -------------------------------------------------------------
            # 3. NON-LINEAR INFLECTION THRESHOLDS & TRUMP CARDS
            # -------------------------------------------------------------
            data["wifi_is_5"] = (data["Inflight wifi service"] == 5).astype(np.int8)
            data["high_online_boarding"] = (data["Online boarding"] >= 4).astype(
                np.int8
            )
            data["high_seat_comfort"] = (data["Seat comfort"] >= 4).astype(np.int8)
            data["high_entertainment"] = (data["Inflight entertainment"] >= 4).astype(
                np.int8
            )
            data["checkin_is_acceptable"] = (data["Checkin service"] >= 3).astype(
                np.int8
            )

            # Total Digital Failure Gate (<6% satisfaction)
            data["digital_failure"] = (
                (data["Online boarding"] <= 3) & (data["Inflight wifi service"] <= 3)
            ).astype(np.int8)

            # -------------------------------------------------------------
            # 4. DOMAIN PILLAR AGGREGATIONS & BOTTLENECK LAW
            # -------------------------------------------------------------
            ratings_no_zero = data[self.config.rating_cols].replace(0, np.nan)

            data["min_service_rating"] = (
                ratings_no_zero.min(axis=1).fillna(3).astype(np.float32)
            )
            data["has_service_failure"] = (data["min_service_rating"] <= 2).astype(
                np.int8
            )
            data["is_all_pass_service"] = (data["min_service_rating"] >= 3).astype(
                np.int8
            )

            data["digital_score"] = (
                ratings_no_zero[
                    [
                        "Online boarding",
                        "Inflight wifi service",
                        "Ease of Online booking",
                    ]
                ]
                .mean(axis=1)
                .fillna(3)
                .astype(np.float32)
            )

            data["cabin_score"] = (
                ratings_no_zero[
                    [
                        "Seat comfort",
                        "Leg room service",
                        "Cleanliness",
                        "Food and drink",
                    ]
                ]
                .mean(axis=1)
                .fillna(3)
                .astype(np.float32)
            )

            data["staff_score"] = (
                ratings_no_zero[
                    ["On-board service", "Baggage handling", "Checkin service"]
                ]
                .mean(axis=1)
                .fillna(3)
                .astype(np.float32)
            )

            data["comfort_score"] = data["cabin_score"]
            logistics_cols = [
                c
                for c in [
                    "Departure/Arrival time convenient",
                    "Gate location",
                    "Inflight entertainment",
                ]
                if c in ratings_no_zero.columns
            ]
            if logistics_cols:
                data["logistics_score"] = (
                    ratings_no_zero[logistics_cols]
                    .mean(axis=1)
                    .fillna(3)
                    .astype(np.float32)
                )

            data["total_service_mean"] = (
                ratings_no_zero.mean(axis=1).fillna(3).astype(np.float32)
            )
            data["service_rating_std"] = (
                ratings_no_zero.std(axis=1).fillna(0).astype(np.float32)
            )
            data["service_rating_range"] = (
                (ratings_no_zero.max(axis=1) - data["min_service_rating"])
                .fillna(0)
                .astype(np.float32)
            )

            # -------------------------------------------------------------
            # 5. PSYCHOMETRICS (RASCH DELIGHT, IRT PCM & RESPONSE STYLE FORENSICS)
            # -------------------------------------------------------------
            rasch_score = np.zeros(len(data), dtype=np.float32)
            weighted_ratings = np.zeros(len(data), dtype=np.float32)
            weighted_max = np.zeros(len(data), dtype=np.float32)

            for item, difficulty in self.config.rasch_difficulties.items():
                if item in data.columns:
                    rasch_score += difficulty * (data[item] >= 4).astype(np.float32)
                    w_item = float(1.0 / (1.0 + np.exp(difficulty)))
                    valid_mask = (data[item] > 0).astype(np.float32)
                    weighted_ratings += (
                        w_item * data[item].astype(np.float32) * valid_mask
                    )
                    weighted_max += w_item * 5.0 * valid_mask

            data["rasch_delight_score"] = rasch_score
            # Rasch Partial Credit Model (PCM) latent satisfaction trait in logit space
            p_pcm = (weighted_ratings + 0.5) / (weighted_max + 1.0)
            data["rasch_pcm_trait"] = np.log(p_pcm / (1.0 - p_pcm)).astype(np.float32)

            # Survey Response Style Forensics (Empirical Behavior Archetypes)
            rating_matrix = data[self.config.rating_cols]
            valid_counts = np.maximum(1, (rating_matrix > 0).sum(axis=1))

            # 1. Intra-passenger variance (straight-liners yield exactly 0.0)
            data["intra_passenger_var"] = (
                rating_matrix.var(axis=1).fillna(0.0).astype(np.float32)
            )
            data["straight_liner"] = (data["intra_passenger_var"] == 0.0).astype(
                np.int8
            )

            # 1b. Survey entropy across rating distribution (Friend 2)
            r_mat = rating_matrix.values
            ent_arr = np.zeros(len(data), dtype=np.float32)
            n_items = float(r_mat.shape[1])
            for k in range(6):
                pk = (r_mat == k).sum(axis=1) / n_items
                ent_arr -= np.where(pk > 0, pk * np.log(pk + 1e-9), 0.0)
            data["survey_entropy"] = ent_arr.astype(np.float32)
            data["survey_zero_count"] = (rating_matrix == 0).sum(axis=1).astype(np.int8)

            # 2. Midpoint satisficing (fraction of 3s given among answered questions)
            data["midpoint_fraction"] = (
                (rating_matrix == 3).sum(axis=1) / valid_counts
            ).astype(np.float32)
            data["midpoint_ratio"] = (
                (rating_matrix == 3).mean(axis=1).astype(np.float32)
            )

            # 3. Extremity index (fraction of 1s and 5s among answered questions)
            data["extremity_index"] = (
                ((rating_matrix == 1) | (rating_matrix == 5)).sum(axis=1) / valid_counts
            ).astype(np.float32)
            data["extremity_ratio"] = (
                ((rating_matrix == 1) | (rating_matrix == 5)).mean(axis=1)
            ).astype(np.float32)

            # 4. N/A count (absolute sum of 0s: proxy for interaction level)
            data["na_count"] = (rating_matrix == 0).sum(axis=1).astype(np.int8)

            # -------------------------------------------------------------
            # 6. SIMPSON'S INVERSION & DEMOGRAPHIC INTERACTIONS
            # -------------------------------------------------------------
            data = data.copy()
            is_business_class = (data["Class"] == "Business").astype(np.int8)
            is_business_travel = (data["Type of Travel"] == "Business travel").astype(
                np.int8
            )
            is_loyal = (data["Customer Type"] == "Loyal Customer").astype(np.int8)

            data["dist_business"] = data["Flight Distance"] * is_business_class
            data["dist_eco"] = data["Flight Distance"] * (1 - is_business_class)
            data["log_flight_distance"] = np.log1p(
                data["Flight Distance"].clip(lower=0)
            ).astype(np.float32)
            data["age_x_business"] = data["Age"] * is_business_travel
            data["gate_x_business"] = (
                data["Gate location"].astype(str) + "_" + is_business_travel.astype(str)
            )
            data["loyal_business"] = (is_loyal & is_business_class).astype(np.int8)
            data["disloyal_economy"] = (
                (1 - is_loyal) & (1 - is_business_class)
            ).astype(np.int8)

            data["premium_service_failure"] = (
                is_business_class
                & (
                    (data["Cleanliness"] <= 2)
                    | (data["On-board service"] <= 2)
                    | (data["Inflight entertainment"] <= 2)
                )
            ).astype(np.int8)

            # Simpson's Paradox Confounder Disentanglement: Departure/Arrival time convenient
            # In raw data, time_conv=1 has 50.6% satisfaction (86.2% Business Travelers),
            # while time_conv=4 has 40.2% satisfaction (56.6% Business Travelers).
            time_conv = data["Departure/Arrival time convenient"].astype(np.float32)
            data["time_conv_x_business_travel"] = (time_conv * is_business_travel).astype(
                np.float32
            )
            data["time_conv_x_business_class"] = (time_conv * is_business_class).astype(
                np.float32
            )
            data["time_conv_x_eco"] = (time_conv * (1 - is_business_class)).astype(
                np.float32
            )

            # -------------------------------------------------------------
            # 7. 3-WAY MACRO-MANIFOLDS (GOLDEN SEGMENT VS DEAD ZONE)
            # -------------------------------------------------------------
            data["is_golden_segment"] = (
                is_business_class & is_business_travel & (data["Online boarding"] >= 4)
            ).astype(np.int8)

            data["is_dead_zone"] = (
                (data["Class"] == "Eco")
                & (data["Type of Travel"] == "Personal Travel")
                & (data["Online boarding"] < 4)
            ).astype(np.int8)

            data["class_x_travel_type"] = (
                data["Class"].astype(str) + "_" + data["Type of Travel"].astype(str)
            )

            # -------------------------------------------------------------
            # 8. DOMAIN A: HIGH-ORDER CATEGORICAL FREQUENCY FORENSICS
            # -------------------------------------------------------------
            if self.config.enable_high_order_freq and self.freq_maps:
                keys = self._get_frequency_keys(data)
                for name, col_series in keys.items():
                    freq_col = f"freq_{name}"
                    log_count_col = f"log_count_{name}"
                    data[freq_col] = (
                        col_series.map(self.freq_maps[name])
                        .fillna(0.0)
                        .astype(np.float32)
                    )
                    data[log_count_col] = np.log1p(
                        col_series.map(self.count_maps[name]).fillna(0)
                    ).astype(np.float32)

            # -------------------------------------------------------------
            # 9. DOMAIN A: GEOMETRIC CENTROIDS & DENSITY ANOMALY SCORES
            # -------------------------------------------------------------
            if self.config.enable_density_forensics and self.scaler_mean is not None:
                core_cols = self.config.core_centroid_cols
                core_vals = data[core_cols].copy()
                core_vals["Arrival Delay in Minutes"] = data["Arrival Delay in Minutes"]
                X_norm = (
                    (core_vals.values - self.scaler_mean) / self.scaler_std
                ).astype(np.float32)

                if self.mbk_1 is not None and self.mbk_0 is not None:
                    # Distances to sub-cluster centroids
                    d_pos = self.mbk_1.transform(X_norm).min(axis=1)
                    d_neg = self.mbk_0.transform(X_norm).min(axis=1)

                    data["dist_to_satisfied_sub"] = d_pos.astype(np.float32)
                    data["dist_to_dissatisfied_sub"] = d_neg.astype(np.float32)
                    data["centroid_dist_ratio"] = (d_neg / (d_pos + 1e-5)).astype(
                        np.float32
                    )
                    data["centroid_dist_margin"] = (
                        (d_neg - d_pos) / (d_neg + d_pos + 1e-5)
                    ).astype(np.float32)
                    data["log_dist_satisfied"] = np.log(d_pos + 1e-5).astype(np.float32)
                    data["log_dist_dissatisfied"] = np.log(d_neg + 1e-5).astype(
                        np.float32
                    )

                    # Global class centroid distances
                    if (
                        self.global_centroid_1 is not None
                        and self.global_centroid_0 is not None
                    ):
                        g_pos = np.linalg.norm(X_norm - self.global_centroid_1, axis=1)
                        g_neg = np.linalg.norm(X_norm - self.global_centroid_0, axis=1)
                        data["global_dist_satisfied"] = g_pos.astype(np.float32)
                        data["global_dist_dissatisfied"] = g_neg.astype(np.float32)
                        data["global_dist_log_ratio"] = np.log(
                            (g_neg + 1e-5) / (g_pos + 1e-5)
                        ).astype(np.float32)

                if self.mbk_anomaly is not None:
                    # Anomaly density score: distance to nearest mode across full dataset
                    d_mode = self.mbk_anomaly.transform(X_norm).min(axis=1)
                    data["anomaly_density_score"] = np.log1p(d_mode).astype(np.float32)

            # -------------------------------------------------------------
            # 10. DOMAIN 4: MULTI-WAY TOPOLOGICAL CROSSES & SVD MANIFOLDS
            # -------------------------------------------------------------
            mc1, mc2 = self._get_multi_crosses(data)
            data["multi_cross_1"] = mc1.astype(str)
            data["multi_cross_2"] = mc2.astype(str)

            # Linear Rotational Variance via TruncatedSVD
            if self.config.enable_svd_manifolds and self.svd is not None:
                svd_data = data[self.svd_cols].copy()
                if (
                    "Arrival Delay in Minutes" in svd_data.columns
                    and "Departure Delay in Minutes" in svd_data.columns
                ):
                    svd_data["Arrival Delay in Minutes"] = svd_data[
                        "Arrival Delay in Minutes"
                    ].fillna(svd_data["Departure Delay in Minutes"])
                for rc in self.config.rating_cols:
                    if rc in svd_data.columns:
                        impute_val = self.svd_rating_impute.get(rc, 3.0)
                        svd_data[rc] = svd_data[rc].replace(0, impute_val)

                X_svd_norm = ((svd_data.values - self.svd_mean) / self.svd_std).astype(
                    np.float32
                )
                svd_comps = self.svd.transform(X_svd_norm)
                for i in range(self.config.n_svd_components):
                    data[f"svd_{i}"] = svd_comps[:, i].astype(np.float32)

            # -------------------------------------------------------------
            # 10.5 ROUTE PROFILES & BOUNDED CROSSES (Rugved Bane & Friend 1)
            # -------------------------------------------------------------
            data = data.copy()
            # Route Profiles: Flight Distance target encoding & aggregations
            if self.config.enable_route_profiles and "Flight Distance" in data.columns:
                if (
                    is_train
                    and "Flight Distance" in self.train_oof_te
                    and len(data) == len(self.train_oof_te["Flight Distance"])
                ):
                    data["te_Flight Distance"] = self.train_oof_te[
                        "Flight Distance"
                    ].astype(np.float32)
                elif self.te_flight_distance_map:
                    data["te_Flight Distance"] = (
                        data["Flight Distance"]
                        .map(self.te_flight_distance_map)
                        .fillna(self.global_target_mean)
                        .astype(np.float32)
                    )
                data["freq_flight_distance"] = (
                    data["Flight Distance"]
                    .map(self.freq_flight_distance_map)
                    .fillna(0.0)
                    .astype(np.float32)
                )
                data["log_count_flight_distance"] = np.log1p(
                    data["Flight Distance"]
                    .map(self.count_flight_distance_map)
                    .fillna(0)
                ).astype(np.float32)
                data["route_mean_arr_delay"] = (
                    data["Flight Distance"]
                    .map(self.route_mean_arr_map)
                    .fillna(arr_delay)
                    .astype(np.float32)
                )
                data["route_std_arr_delay"] = (
                    data["Flight Distance"]
                    .map(self.route_std_arr_map)
                    .fillna(0.0)
                    .astype(np.float32)
                )

            # Real-World FAA / US Trunk Air Corridor Quantization (Reverse-Engineered Discrete Modes)
            # Dominant discrete airport pairs: 2475 (JFK-LAX), 337 (LAX-SFO), 594 (ORD-ATL), 862 (DFW-ORD),
            # 404 (ORD-MSP), 447 (DFW-MSY), 236 (ORD-DTW), 192, 308, 399
            if "Flight Distance" in data.columns:
                trunk_modes = np.array(
                    [192, 236, 308, 337, 399, 404, 447, 594, 862, 2475],
                    dtype=np.float32,
                )
                fd_arr = data["Flight Distance"].values.astype(np.float32)
                diffs = np.abs(fd_arr[:, None] - trunk_modes[None, :])
                nearest_idx = np.argmin(diffs, axis=1)
                min_dist_to_mode = np.min(diffs, axis=1)
                nearest_modes = trunk_modes[nearest_idx]

                data["is_trunk_route"] = (min_dist_to_mode <= 2.0).astype(np.int8)
                data["trunk_mode_residual"] = (fd_arr - nearest_modes).astype(
                    np.float32
                )
                data["abs_trunk_mode_residual"] = min_dist_to_mode.astype(np.float32)

            # Bounded Crosses String Representation
            if self.config.enable_bounded_crosses:
                bounded_dict = self._get_bounded_crosses(data)
                new_cols = {
                    b_col: b_series.astype(str)
                    for b_col, b_series in bounded_dict.items()
                }
                data = pd.concat(
                    [data, pd.DataFrame(new_cols, index=data.index)], axis=1
                )

            # Multi-Way Bayesian Target Encoding (Domain 4 + Extensions)
            if self.target_encoding_maps:
                te_new = {}
                for col in self.target_encoding_maps.keys():
                    if col == "Flight Distance":
                        continue
                    if (
                        is_train
                        and col in self.train_oof_te
                        and len(data) == len(self.train_oof_te[col])
                    ):
                        te_new[f"te_{col}"] = self.train_oof_te[col].astype(np.float32)
                    elif col in self.target_encoding_maps and col in data.columns:
                        m = self.target_encoding_maps[col]
                        te_new[f"te_{col}"] = (
                            data[col]
                            .map(m)
                            .fillna(self.global_target_mean)
                            .astype(np.float32)
                        )
                if te_new:
                    data = pd.concat(
                        [data, pd.DataFrame(te_new, index=data.index)], axis=1
                    )

            # -------------------------------------------------------------
            # 10.8 ORIGINAL DATASET PRIOR FEATURES (Rugved Bane #2 and #3)
            # -------------------------------------------------------------
            data = data.copy()
            if self.orig_prior_model is not None and self.orig_prior_cols:
                try:
                    X_prior_eval = data[self.orig_prior_cols].copy()
                    for cat_c in self.config.categorical_cols:
                        if cat_c in X_prior_eval.columns:
                            mapping = self.label_encoder_dicts.get(cat_c, {})
                            X_prior_eval[cat_c] = (
                                X_prior_eval[cat_c].astype(str).map(mapping).fillna(-1)
                            )
                    p_prior = self.orig_prior_model.predict_proba(X_prior_eval)[
                        :, 1
                    ].astype(np.float32)
                    p_prior = np.clip(p_prior, 1e-5, 1.0 - 1e-5)
                    data["orig_proba"] = p_prior
                    data["orig_logit"] = np.log(p_prior / (1.0 - p_prior)).astype(
                        np.float32
                    )
                except Exception:
                    data["orig_proba"] = np.float32(0.5)
                    data["orig_logit"] = np.float32(0.0)
            else:
                data["orig_proba"] = np.float32(0.5)
                data["orig_logit"] = np.float32(0.0)

            # -------------------------------------------------------------
            # 10.9 ROUTE PROFILES, GROUP MEANS & RESIDUALS (Koumei Maki +124 bps)
            # -------------------------------------------------------------
            if self.route_profile_means and "Flight Distance" in data.columns:
                fd_series = data["Flight Distance"]
                data["fd_cnt"] = (
                    fd_series.map(self.route_fd_cnt).fillna(1.0).astype(np.float32)
                )
                route_adds = {}
                for c, mean_map in self.route_profile_means.items():
                    if c in data.columns:
                        val_c = data[c]
                        if c in self.config.categorical_cols:
                            mapping = self.label_encoder_dicts.get(c, {})
                            val_c = val_c.astype(str).map(mapping).fillna(-1).astype(np.float32)
                        else:
                            val_c = (
                                pd.to_numeric(val_c, errors="coerce")
                                .fillna(0.0)
                                .astype(np.float32)
                            )
                        mean_c = (
                            fd_series.map(mean_map)
                            .fillna(float(val_c.mean()))
                            .astype(np.float32)
                        )
                        route_adds[f"fdm_{c}"] = mean_c
                        route_adds[f"res_{c}"] = (val_c - mean_c).astype(np.float32)
                if route_adds:
                    data = pd.concat(
                        [data, pd.DataFrame(route_adds, index=data.index)], axis=1
                    )

            # -------------------------------------------------------------
            # 10.10 NUMERIC VALUE COUNTS (Koumei Maki +7 bps)
            # -------------------------------------------------------------
            cnt_adds = {}
            for c, cnt_map in self.num_value_counts.items():
                if c in data.columns:
                    cnt_adds[f"cnt_{c}"] = (
                        data[c].map(cnt_map).fillna(0).astype(np.float32)
                    )
            if cnt_adds:
                data = pd.concat(
                    [data, pd.DataFrame(cnt_adds, index=data.index)], axis=1
                )

            # -------------------------------------------------------------
            # 10.11 ORIGINAL DATASET SMOOTHED LOOKUPS (Koumei Maki +48 bps)
            # -------------------------------------------------------------
            org_adds = {}
            for c, lookup in self.orig_lookup_maps.items():
                if c in data.columns:
                    org_adds[f"org_mean_{c}"] = (
                        data[c]
                        .map(lookup)
                        .fillna(self.orig_prior_value)
                        .astype(np.float32)
                    )
            if org_adds:
                data = pd.concat(
                    [data, pd.DataFrame(org_adds, index=data.index)], axis=1
                )

            # -------------------------------------------------------------
            # 10.12 DIGIT & MODULO DECOMPOSITION (Busyaprime & Goodpjw)
            # -------------------------------------------------------------
            modulo_adds = {}
            if "Flight Distance" in data.columns:
                fd_v = (
                    pd.to_numeric(data["Flight Distance"], errors="coerce")
                    .fillna(0)
                    .astype(np.int64)
                )
                modulo_adds["fd_mod10"] = (fd_v % 10).astype(np.float32)
                modulo_adds["fd_mod50"] = (fd_v % 50).astype(np.float32)
                modulo_adds["fd_mod100"] = (fd_v % 100).astype(np.float32)
                modulo_adds["fd_mod250"] = (fd_v % 250).astype(np.float32)
                modulo_adds["fd_mod500"] = (fd_v % 500).astype(np.float32)
                modulo_adds["fd_div10"] = (fd_v // 10).astype(np.float32)
                modulo_adds["fd_div100"] = (fd_v // 100).astype(np.float32)
                # Harmonic smooth cyclical representations (Friend 4 Track C)
                modulo_adds["sin_fd_500"] = np.sin(2.0 * np.pi * fd_v / 500.0).astype(np.float32)
                modulo_adds["cos_fd_500"] = np.cos(2.0 * np.pi * fd_v / 500.0).astype(np.float32)
            if "Age" in data.columns:
                age_v = (
                    pd.to_numeric(data["Age"], errors="coerce")
                    .fillna(0)
                    .astype(np.int64)
                )
                modulo_adds["age_mod10"] = (age_v % 10).astype(np.float32)
                modulo_adds["age_div10"] = (age_v // 10).astype(np.float32)
            if modulo_adds:
                data = pd.concat(
                    [data, pd.DataFrame(modulo_adds, index=data.index)], axis=1
                )

            # -------------------------------------------------------------
            # 10.13 AUXILIARY EXPECTED RATINGS & RESIDUALS (Friend 1 Methodology)
            # -------------------------------------------------------------
            if self.expected_rating_models and self.expected_rating_features:
                try:
                    X_eval_ratings = data[self.expected_rating_features].copy()
                    for cat_c in self.config.categorical_cols:
                        if cat_c in X_eval_ratings.columns:
                            m_dict = self.label_encoder_dicts.get(cat_c, {})
                            X_eval_ratings[cat_c] = (
                                X_eval_ratings[cat_c]
                                .astype(str)
                                .map(m_dict)
                                .fillna(-1)
                                .astype(np.int32)
                            )
                    if (
                        "Arrival Delay in Minutes" in X_eval_ratings.columns
                        and "Departure Delay in Minutes" in X_eval_ratings.columns
                    ):
                        X_eval_ratings["Arrival Delay in Minutes"] = X_eval_ratings[
                            "Arrival Delay in Minutes"
                        ].fillna(X_eval_ratings["Departure Delay in Minutes"])
                    X_eval_ratings = X_eval_ratings.fillna(0)

                    exp_adds = {}
                    all_surprises = []
                    for r_col, clf_r in self.expected_rating_models.items():
                        probs = clf_r.predict_proba(X_eval_ratings)
                        classes = np.array(clf_r.classes_, dtype=np.float32)
                        exp_val = np.sum(probs * classes, axis=1).astype(np.float32)
                        exp_adds[f"exp_{r_col}"] = exp_val
                        
                        # Confidence (maximum predicted probability) & Conditional Entropy
                        conf_val = np.max(probs, axis=1).astype(np.float32)
                        exp_adds[f"conf_exp_{r_col}"] = conf_val
                        ent_val = -np.sum(probs * np.log(probs + 1e-9), axis=1).astype(np.float32)
                        exp_adds[f"ent_exp_{r_col}"] = ent_val

                        if r_col in data.columns:
                            act_val = (
                                pd.to_numeric(data[r_col], errors="coerce")
                                .fillna(0.0)
                                .values.astype(np.float32)
                            )
                            exp_adds[f"res_exp_{r_col}"] = (act_val - exp_val).astype(
                                np.float32
                            )
                            # Surprise score: -log P(observed rating | context)
                            class_indices = {int(c): idx for idx, c in enumerate(clf_r.classes_)}
                            act_ints = np.round(act_val).astype(int)
                            idx_arr = np.array([class_indices.get(v, 0) for v in act_ints])
                            obs_probs = probs[np.arange(len(probs)), idx_arr]
                            surp_val = -np.log(np.maximum(1e-5, obs_probs)).astype(np.float32)
                            exp_adds[f"surp_exp_{r_col}"] = surp_val
                            all_surprises.append(surp_val)

                    if all_surprises:
                        exp_adds["mean_rating_surprise"] = np.mean(all_surprises, axis=0).astype(np.float32)
                        exp_adds["max_rating_surprise"] = np.max(all_surprises, axis=0).astype(np.float32)

                    if exp_adds:
                        data = pd.concat(
                            [data, pd.DataFrame(exp_adds, index=data.index)], axis=1
                        )
                except Exception as e:
                    logging.warning(
                        f"Failed to generate Auxiliary Expected Ratings in transform: {e}"
                    )

            # -------------------------------------------------------------
            # 11. CATEGORICAL ENCODING
            # -------------------------------------------------------------
            data = data.copy()
            cat_columns = list(self.label_encoders.keys())
            for col in cat_columns:
                if col in data.columns:
                    mapping = self.label_encoder_dicts.get(col, {})
                    data[col] = (
                        data[col]
                        .astype(str)
                        .map(mapping)
                        .fillna(-1)
                        .astype(np.int16)
                    )

            # Drop identifier if present
            if self.config.id_col in data.columns:
                data = data.drop(columns=[self.config.id_col])

            # Drop target if present during feature matrix creation
            if self.config.target_col in data.columns:
                data = data.drop(columns=[self.config.target_col])

            # Memory optimization
            data = reduce_mem_usage(data, verbose=False)
            return data

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        orig_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fits transductive statistics and transforms training data."""
        self.fit(train_df, test_df, orig_df=orig_df)
        return self.transform(train_df, is_train=True)

    def get_feature_view(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """
        Routes engineered dataframe to the mathematically optimal feature view
        tailored to the specific GBDT architecture's inductive bias:
        - View 1 (CatBoost): Oblivious symmetric trees with native CTR. Focuses on categorical
          crosses, discrete ratings, numerical categories, demographic pairs, and prior logit.
          Excludes dense route residuals and continuous SVD projections.
        - View 2 (LightGBM): Leaf-wise (best-first) growth. Exploits exact route group means
          (fdm_*), residuals (res_*), route counts (fd_cnt), original-data lookups, and value counts.
          Excludes the 39 rating crosses (which hurt LightGBM by -48 bps due to leaf fragmentation).
        - View 3 (XGBoost): Depth-wise Hessian-weighted growth. Exploits airborne delay recovery dynamics,
          Rasch PCM trait logits, Shannon survey entropy, response style forensics, and digit modulo features.
        """
        name = model_name.lower()
        cols = df.columns.tolist()

        rev_eng_features = {
            "is_both_delays_zero",
            "is_dep_zero_arr_nonzero",
            "is_dep_nonzero_arr_zero",
            "delay_discrepancy",
            "delay_speedup_ratio",
            "is_all_digital_na",
            "any_digital_na",
            "na_adj_wifi",
            "na_adj_boarding",
            "na_adj_booking",
            "time_conv_x_business_travel",
            "time_conv_x_business_class",
            "time_conv_x_eco",
            "is_trunk_route",
            "trunk_mode_residual",
            "abs_trunk_mode_residual",
        }

        if any(k in name for k in ["extra_trees", "xt", "lightgbm_xt", "lgb_xt"]):
            # VIEW 4: LightGBM Extra-Trees Randomized Threshold View (Candidate D)
            selected = []
            for c in cols:
                if "__X__" in c or c.startswith("res_") or c.startswith("fdm_") or c.startswith("svd_"):
                    continue
                if "centroid" in c or "dist_to_" in c or "global_dist_" in c:
                    continue

                if (
                    c.startswith("exp_")
                    or c.startswith("res_exp_")
                    or c.startswith("conf_exp_")
                    or c.startswith("ent_exp_")
                    or c.startswith("cnt_")
                    or c.startswith("org_mean_")
                    or c.startswith("fd_mod")
                    or c.startswith("sin_fd_")
                    or c.startswith("cos_fd_")
                    or (c in rev_eng_features)
                    or (
                        c
                        in [
                            "fd_cnt",
                            "orig_logit",
                            "orig_proba",
                            "total_delay",
                            "has_delay",
                            "has_severe_delay",
                            "Delay_Delta",
                            "Recovery_Magnitude",
                            "Compounding_Delay",
                            "route_delay_hazard",
                            "rasch_delight_score",
                            "rasch_pcm_trait",
                            "survey_entropy",
                            "digital_score",
                            "cabin_score",
                            "staff_score",
                            "comfort_score",
                            "total_service_mean",
                            "service_rating_std",
                            "total_na_ratings",
                            "age_mod10",
                            "age_div10",
                        ]
                    )
                    or (c in self.config.categorical_cols)
                    or (c in self.config.rating_cols)
                    or (c in self.config.numerical_cols)
                ):
                    selected.append(c)

            if len(selected) < 15:
                return df.copy()
            return df[selected].copy()

        elif any(k in name for k in ["catboost", "cb", "cat"]):
            # VIEW 1: CatBoost Categorical & CTR Topology View
            selected = []
            for c in cols:
                # Exclude dense continuous route residuals, SVD manifolds, modulos, and centroid ratios
                if c.startswith("res_") or c.startswith("fdm_") or c.startswith("svd_") or c.startswith("fd_mod"):
                    continue
                if c.startswith("sin_fd_") or c.startswith("cos_fd_"):
                    continue
                if "centroid" in c or "dist_to_" in c or "global_dist_" in c:
                    continue
                if c in ["anomaly_density_score", "intra_passenger_var"]:
                    continue

                # Keep base features, rating crosses, categories, prior logits, and passenger coherence/surprise
                if (
                    ("__X__" in c)
                    or ("__category" in c)
                    or ("_x_" in c)
                    or ("_is_0" in c)
                    or c.startswith("exp_")
                    or c.startswith("surp_exp_")
                    or c in ["mean_rating_surprise", "max_rating_surprise"]
                    or (c in rev_eng_features)
                    or (c in self.config.categorical_cols)
                    or (c in self.config.rating_cols)
                    or (c in self.config.numerical_cols)
                    or (c in ["orig_logit", "orig_proba", "total_na_ratings"])
                    or (
                        c
                        in [
                            "route_class_travel",
                            "route_delay_tier",
                            "route_dissatisfaction",
                            "service_failure_class",
                            "age_class_travel",
                        ]
                    )
                ):
                    selected.append(c)

            if len(selected) < 15:
                return df.copy()
            return df[selected].copy()

        elif any(k in name for k in ["lightgbm", "lgb"]):
            # VIEW 2: LightGBM Route Topology, Residuals & Original Lookups View
            selected = []
            for c in cols:
                # Hard exclusion: Shelton Wang 39 rating crosses (hurts LightGBM by -48 bps)
                if "__X__" in c:
                    continue
                if c.startswith("svd_"):
                    continue
                if "centroid" in c or "dist_to_" in c or "global_dist_" in c:
                    continue

                # Include route profiles, residuals, original lookups, counts, base features, modulos, harmonics
                if (
                    c.startswith("fdm_")
                    or c.startswith("res_")
                    or c.startswith("cnt_")
                    or c.startswith("org_mean_")
                    or c.startswith("exp_")
                    or c.startswith("res_exp_")
                    or c.startswith("conf_exp_")
                    or c.startswith("ent_exp_")
                    or c.startswith("surp_exp_")
                    or c.startswith("fd_mod")
                    or c.startswith("sin_fd_")
                    or c.startswith("cos_fd_")
                    or c in ["mean_rating_surprise", "max_rating_surprise"]
                    or (c in rev_eng_features)
                    or (
                        c
                        in [
                            "fd_cnt",
                            "orig_logit",
                            "orig_proba",
                            "total_delay",
                            "has_delay",
                            "has_severe_delay",
                            "Delay_Delta",
                            "Recovery_Magnitude",
                            "Compounding_Delay",
                            "route_mean_arr_delay",
                            "route_std_arr_delay",
                            "freq_flight_distance",
                            "log_count_flight_distance",
                        ]
                    )
                    or (c in self.config.categorical_cols)
                    or (c in self.config.rating_cols)
                    or (c in self.config.numerical_cols)
                ):
                    selected.append(c)

            if len(selected) < 15:
                return df.copy()
            return df[selected].copy()

        elif any(k in name for k in ["xgboost", "xgb"]):
            # VIEW 3: XGBoost Latent Psychometrics, Airborne Dynamics & Multi-Key View
            selected = []
            for c in cols:
                if "__X__" in c or c.startswith("res_"):
                    continue
                if c.startswith("svd_"):
                    continue
                if "centroid" in c or "dist_to_" in c or "global_dist_" in c:
                    continue

                if (
                    c.startswith("exp_")
                    or c.startswith("res_exp_")
                    or c.startswith("conf_exp_")
                    or c.startswith("ent_exp_")
                    or c.startswith("fd_mod")
                    or c.startswith("sin_fd_")
                    or c.startswith("cos_fd_")
                    or (c in rev_eng_features)
                    or (
                        c
                        in [
                            "Delay_Delta",
                            "arr_minus_dep",
                            "Recovery_Magnitude",
                            "Compounding_Delay",
                            "total_delay",
                            "has_delay",
                            "has_severe_delay",
                            "route_delay_hazard",
                            "delay_intensity",
                            "delay_diff",
                            "Arrival_Delay_is_nan",
                            "rasch_delight_score",
                            "rasch_pcm_trait",
                            "intra_passenger_var",
                            "straight_liner",
                            "survey_entropy",
                            "midpoint_fraction",
                            "extremity_ratio",
                            "min_service_rating",
                            "has_service_failure",
                            "is_all_pass_service",
                            "digital_score",
                            "cabin_score",
                            "staff_score",
                            "comfort_score",
                            "total_service_mean",
                            "service_rating_std",
                            "total_na_ratings",
                            "age_mod10",
                            "age_div10",
                            "orig_logit",
                            "orig_proba",
                        ]
                    )
                    or c.startswith("cnt_")
                    or (c in self.config.categorical_cols)
                    or (c in self.config.rating_cols)
                    or (c in self.config.numerical_cols)
                ):
                    selected.append(c)

            if len(selected) < 15:
                return df.copy()
            return df[selected].copy()

        else:
            return df.copy()



# ==============================================================================
# ARGON RESEARCH VECTOR 4 & 6: ADVANCED MANIFOLD & ASYMMETRIC TOPOLOGIES
# ==============================================================================

from typing import Optional

SURVEY_COLS: list[str] = [
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

ZERO_CAPABLE_SURVEY_COLS: list[str] = [
    c for c in SURVEY_COLS if c != "Baggage handling"
]

DIGITAL_ITEMS: list[str] = [
    "Inflight wifi service",
    "Online boarding",
    "Ease of Online booking",
]
CABIN_ITEMS: list[str] = [
    "Seat comfort",
    "Leg room service",
    "Cleanliness",
    "Inflight entertainment",
    "Food and drink",
    "On-board service",
]
GROUND_ITEMS: list[str] = [
    "Gate location",
    "Checkin service",
    "Baggage handling",
    "Departure/Arrival time convenient",
]

CAT_COLS: list[str] = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS: list[str] = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


class HurdleGradedResponseModel:
    """
    Two-Part Hurdle Polytomous Graded Response Model (ZI-GRM).
    Treats 0-ratings in the zero-capable survey columns as structural
    missingness (hurdle gate) and estimates Samejima's polytomous GRM
    parameters (discrimination alpha_j and ordered thresholds beta_j,k)
    strictly on observed ratings in {1, 2, 3, 4, 5}.
    """

    def __init__(self, theta_grid_size: int = 81, device: Optional[str] = None) -> None:
        self.theta_grid: np.ndarray = np.linspace(-4.0, 4.0, theta_grid_size)
        self.alphas_: np.ndarray = np.ones(len(SURVEY_COLS), dtype=np.float64)
        self.betas_: np.ndarray = np.zeros((len(SURVEY_COLS), 4), dtype=np.float64)
        self.zero_log_odds_: np.ndarray = np.zeros(len(ZERO_CAPABLE_SURVEY_COLS))
        self.non_zero_log_odds_: np.ndarray = np.zeros(len(ZERO_CAPABLE_SURVEY_COLS))
        self.device = device

    def fit(self, df_survey: pd.DataFrame, y: np.ndarray) -> "HurdleGradedResponseModel":
        X = df_survey[SURVEY_COLS].to_numpy(dtype=np.int32)
        y_bin = y.astype(np.int32)
        pos_mask = y_bin == 1
        neg_mask = y_bin == 0

        # 1. Fit Structural Zero Hurdle Log-Odds Ratios on the zero-capable columns
        X_zero = (df_survey[ZERO_CAPABLE_SURVEY_COLS].to_numpy(dtype=np.int32) == 0)
        p_z_pos = (X_zero[pos_mask].sum(axis=0) + 10.0) / (pos_mask.sum() + 20.0)
        p_z_neg = (X_zero[neg_mask].sum(axis=0) + 10.0) / (neg_mask.sum() + 20.0)
        self.zero_log_odds_ = np.log(p_z_pos / p_z_neg)
        self.non_zero_log_odds_ = np.log((1.0 - p_z_pos) / (1.0 - p_z_neg))

        # 2. Calibrate Samejima Polytomous GRM Item Parameters on observed ratings {1..5}
        X_obs = X.astype(np.float64)
        X_obs[X == 0] = np.nan
        col_means = np.nanmean(X_obs, axis=0)
        col_stds = np.nanstd(X_obs, axis=0) + 1e-6
        Z_norm = (X_obs - col_means) / col_stds

        theta_init = np.nanmean(Z_norm, axis=1)
        theta_init = (theta_init - np.mean(theta_init)) / (np.std(theta_init) + 1e-6)

        for j in range(len(SURVEY_COLS)):
            valid = X[:, j] > 0
            r_valid = X[valid, j]
            t_valid = theta_init[valid]

            if len(r_valid) > 10:
                r_corr = np.corrcoef(r_valid, t_valid)[0, 1]
                y_corr = np.corrcoef(r_valid, y_bin[valid])[0, 1]
            else:
                r_corr, y_corr = 0.5, 0.5

            comb_corr = np.clip(0.65 * abs(r_corr) + 0.35 * abs(y_corr), 0.15, 0.92)
            self.alphas_[j] = 1.702 * comb_corr / np.sqrt(1.0 - comb_corr ** 2)

            cum_props = [np.mean(r_valid >= k) if len(r_valid) > 0 else 0.5 for k in (2, 3, 4, 5)]
            cum_props = [np.clip(p, 1e-4, 1.0 - 1e-4) for p in cum_props]
            raw_betas = [-np.log(p / (1.0 - p)) / self.alphas_[j] for p in cum_props]
            self.betas_[j, :] = np.maximum.accumulate(raw_betas)

        return self

    def _score_subset(
        self, X: np.ndarray, item_indices: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized MAP/EAP person scoring, Fisher Test Information I(theta),
        and Drasgow-Levine-Williams Z_h aberrant person-fit statistic.
        Uses GPU PyTorch if available, else exact NumPy CPU.
        """
        N = X.shape[0]
        G = len(self.theta_grid)
        log_prior = -0.5 * (self.theta_grid ** 2)

        n_sub = len(item_indices)
        log_p = np.zeros((n_sub, 6, G), dtype=np.float64)
        item_info = np.zeros((n_sub, G), dtype=np.float64)
        expected_ll = np.zeros((n_sub, G), dtype=np.float64)
        var_ll = np.zeros((n_sub, G), dtype=np.float64)

        for idx, j in enumerate(item_indices):
            a_j = self.alphas_[j]
            b_j = self.betas_[j]
            cum_p = np.zeros((6, G), dtype=np.float64)
            cum_p[0] = 1.0
            for k in range(4):
                cum_p[k + 1] = 1.0 / (1.0 + np.exp(-a_j * (self.theta_grid - b_j[k])))
            cum_p[5] = 0.0

            d_cum_p = a_j * cum_p * (1.0 - cum_p)

            for k in range(1, 6):
                p_k = np.clip(cum_p[k - 1] - cum_p[k], 1e-12, 1.0)
                dp_k = d_cum_p[k - 1] - d_cum_p[k]
                lp_k = np.log(p_k)
                log_p[idx, k] = lp_k
                item_info[idx] += (dp_k ** 2) / p_k
                expected_ll[idx] += p_k * lp_k
                var_ll[idx] += p_k * (lp_k ** 2)

            var_ll[idx] = np.maximum(var_ll[idx] - (expected_ll[idx] ** 2), 1e-8)

        # 1. GPU Acceleration via PyTorch if device is configured
        if self.device is not None and self.device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    dev = torch.device(self.device)
                    grid_t = torch.from_numpy(self.theta_grid).float().to(dev)
                    prior_t = (-0.5 * (grid_t ** 2)).to(dev)
                    X_t = torch.from_numpy(X).long().to(dev)
                    log_p_t = torch.from_numpy(log_p).float().to(dev)
                    exp_ll_t = torch.from_numpy(expected_ll).float().to(dev)
                    var_ll_t = torch.from_numpy(var_ll).float().to(dev)
                    info_t = torch.from_numpy(item_info).float().to(dev)

                    total_lp_t = prior_t.unsqueeze(0).repeat(N, 1)
                    for idx, j in enumerate(item_indices):
                        r_j_t = X_t[:, j]
                        valid_t = r_j_t > 0
                        total_lp_t[valid_t] += log_p_t[idx, r_j_t[valid_t]]

                    max_lp_t = torch.max(total_lp_t, dim=1, keepdim=True).values
                    weights_t = torch.exp(total_lp_t - max_lp_t)
                    weights_t /= torch.sum(weights_t, dim=1, keepdim=True)
                    theta_eap_t = torch.matmul(weights_t, grid_t)
                    best_g_t = torch.argmax(total_lp_t, dim=1)

                    obs_ll_t = torch.zeros(N, device=dev, dtype=torch.float32)
                    exp_sum_t = torch.zeros(N, device=dev, dtype=torch.float32)
                    var_sum_t = torch.zeros(N, device=dev, dtype=torch.float32)
                    fish_t = torch.ones(N, device=dev, dtype=torch.float32)

                    for idx, j in enumerate(item_indices):
                        r_j_t = X_t[:, j]
                        valid_t = r_j_t > 0
                        g_v_t = best_g_t[valid_t]
                        obs_ll_t[valid_t] += log_p_t[idx, r_j_t[valid_t], g_v_t]
                        exp_sum_t[valid_t] += exp_ll_t[idx, g_v_t]
                        var_sum_t[valid_t] += var_ll_t[idx, g_v_t]
                        fish_t[valid_t] += info_t[idx, g_v_t]

                    zh_t = (obs_ll_t - exp_sum_t) / torch.sqrt(torch.clamp(var_sum_t, min=1e-6))
                    return (
                        theta_eap_t.cpu().numpy().astype(np.float64),
                        fish_t.cpu().numpy().astype(np.float64),
                        zh_t.cpu().numpy().astype(np.float64),
                    )
            except Exception:
                pass  # Clean fallback to CPU NumPy

        # 2. Exact CPU NumPy Fallback
        total_log_post = np.broadcast_to(log_prior[None, :], (N, G)).copy()
        for idx, j in enumerate(item_indices):
            r_j = X[:, j]
            valid = r_j > 0
            total_log_post[valid] += log_p[idx, r_j[valid]]

        max_lp = np.max(total_log_post, axis=1, keepdims=True)
        weights = np.exp(total_log_post - max_lp)
        weights /= np.sum(weights, axis=1, keepdims=True)
        theta_eap = weights @ self.theta_grid
        best_g = np.argmax(total_log_post, axis=1)

        obs_ll = np.zeros(N, dtype=np.float64)
        exp_ll_sum = np.zeros(N, dtype=np.float64)
        var_ll_sum = np.zeros(N, dtype=np.float64)
        fisher_info = np.ones(N, dtype=np.float64)

        for idx, j in enumerate(item_indices):
            r_j = X[:, j]
            valid = r_j > 0
            g_v = best_g[valid]
            obs_ll[valid] += log_p[idx, r_j[valid], g_v]
            exp_ll_sum[valid] += expected_ll[idx, g_v]
            var_ll_sum[valid] += var_ll[idx, g_v]
            fisher_info[valid] += item_info[idx, g_v]

        zh_stat = (obs_ll - exp_ll_sum) / np.sqrt(np.maximum(var_ll_sum, 1e-6))
        return theta_eap, fisher_info, zh_stat

    def transform(self, df_survey: pd.DataFrame) -> pd.DataFrame:
        X = df_survey[SURVEY_COLS].to_numpy(dtype=np.int32)
        X_zero = (df_survey[ZERO_CAPABLE_SURVEY_COLS].to_numpy(dtype=np.int32) == 0)

        out = pd.DataFrame(index=df_survey.index)
        out["irt_zero_cnt"] = X_zero.sum(axis=1).astype(np.float32)
        out["irt_zero_llr"] = (
            X_zero @ self.zero_log_odds_ + (~X_zero) @ self.non_zero_log_odds_
        ).astype(np.float32)

        all_idx = list(range(len(SURVEY_COLS)))
        dig_idx = [SURVEY_COLS.index(c) for c in DIGITAL_ITEMS if c in SURVEY_COLS]
        cab_idx = [SURVEY_COLS.index(c) for c in CABIN_ITEMS if c in SURVEY_COLS]
        gnd_idx = [SURVEY_COLS.index(c) for c in GROUND_ITEMS if c in SURVEY_COLS]

        theta_g, info_g, zh_g = self._score_subset(X, all_idx)
        theta_dig, _, zh_dig = self._score_subset(X, dig_idx)
        theta_cab, _, zh_cab = self._score_subset(X, cab_idx)
        theta_gnd, _, _ = self._score_subset(X, gnd_idx)

        out["irt_theta_global"] = theta_g.astype(np.float32)
        out["irt_fisher_info"] = np.log1p(np.maximum(0.0, info_g)).astype(np.float32)
        out["irt_person_fit_zh"] = np.clip(zh_g, -5.0, 5.0).astype(np.float32)
        out["irt_theta_digital"] = theta_dig.astype(np.float32)
        out["irt_theta_cabin"] = theta_cab.astype(np.float32)
        out["irt_theta_ground"] = theta_gnd.astype(np.float32)
        out["irt_zh_digital"] = np.clip(zh_dig, -5.0, 5.0).astype(np.float32)
        out["irt_zh_cabin"] = np.clip(zh_cab, -5.0, 5.0).astype(np.float32)

        out["irt_diff_dig_cab"] = (theta_dig - theta_cab).astype(np.float32)
        out["irt_diff_cab_gnd"] = (theta_cab - theta_gnd).astype(np.float32)
        out["irt_diff_dig_gnd"] = (theta_dig - theta_gnd).astype(np.float32)
        return out


class TransductiveManifoldAndDGPInverter:
    """
    1. Reconstructs the exact Original-Dataset Flight Distance Lattice & snaps jittered synthetic distances.
    2. Extracts full-manifold transductive density counts & Spectral Route Graph Embeddings.
    3. Fits Variational Gaussian Mixture (VGM) mode-specific normalization inversions.
    4. Computes Adversarial Density Ratio weights w_orig(x) for Original-Dataset alignment.
    """

    def __init__(self, n_vgm_modes: int = 10, n_route_svd: int = 6, seed: int = 42) -> None:
        self.n_vgm_modes = n_vgm_modes
        self.n_route_svd = n_route_svd
        self.seed = seed
        self.orig_fd_support_: np.ndarray = np.array([])
        self.full_counts_: dict[str, pd.Series] = {}
        self.route_svd_map_: pd.DataFrame = pd.DataFrame()
        self.vgm_fd_: Optional[BayesianGaussianMixture] = None
        self.orig_means_: dict[str, pd.Series] = {}
        self.orig_global_mean_: float = 0.434
        self.orig_oracle_model_: Optional[lgb.Booster] = None
        self.ecdf_full_df_: pd.DataFrame = pd.DataFrame()
        self.n_train_full_: int = 0
        self.n_test_full_: int = 0

    def fit_full_manifold(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, df_orig: pd.DataFrame
    ) -> "TransductiveManifoldAndDGPInverter":
        raw_cols = CAT_COLS + NUM_COLS + SURVEY_COLS
        df_full = pd.concat([df_train[raw_cols], df_test[raw_cols]], ignore_index=True)

        self.orig_fd_support_ = np.sort(df_orig["Flight Distance"].dropna().unique())

        for c in NUM_COLS:
            self.full_counts_[c] = df_full[c].fillna(-1).value_counts()

        fd_snapped = self._snap_flight_distance(df_full["Flight Distance"].to_numpy())
        df_full_temp = df_full.copy()
        df_full_temp["fd_snap"] = fd_snapped
        df_full_temp["delay_delta"] = (
            df_full_temp["Arrival Delay in Minutes"].fillna(0.0)
            - df_full_temp["Departure Delay in Minutes"].fillna(0.0)
        )

        agg_spec = df_full_temp.groupby("fd_snap").agg(
            route_vol=("Age", "count"),
            mean_age=("Age", "mean"),
            mean_dep_delay=("Departure Delay in Minutes", "mean"),
            mean_delay_delta=("delay_delta", "mean"),
            std_delay_delta=("delay_delta", "std"),
            wifi_mean=("Inflight wifi service", "mean"),
            wifi_zero_rate=("Inflight wifi service", lambda x: np.mean(x == 0)),
            seat_mean=("Seat comfort", "mean"),
            baggage_mean=("Baggage handling", "mean"),
            gate_mean=("Gate location", "mean"),
            biz_share=("Class", lambda x: np.mean(x == "Business")),
            eco_share=("Class", lambda x: np.mean(x == "Eco")),
        ).fillna(0.0)

        # Section 4.1: Empirical Cumulative Distribution Function (ECDF) Route Ranks
        self.ecdf_full_df_ = pd.DataFrame(index=df_full.index)
        for c in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
            clean_name = "arr_delay" if "Arrival" in c else ("dep_delay" if "Departure" in c else "age")
            self.ecdf_full_df_[f"ecdf_{clean_name}_route"] = (
                df_full_temp.groupby("fd_snap")[c].rank(pct=True).astype(np.float32)
            )
        self.n_train_full_ = len(df_train)
        self.n_test_full_ = len(df_test)

        scaler = StandardScaler()
        M_scaled = scaler.fit_transform(agg_spec.to_numpy(dtype=np.float64))
        n_comp = min(self.n_route_svd, M_scaled.shape[1], M_scaled.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        U_route = svd.fit_transform(M_scaled)

        svd_cols = [f"route_svd_{k}" for k in range(n_comp)]
        self.route_svd_map_ = pd.DataFrame(
            U_route.astype(np.float32), index=agg_spec.index, columns=svd_cols
        )
        self.route_svd_map_["route_log_vol"] = np.log1p(agg_spec["route_vol"]).astype(np.float32)
        self.route_svd_map_["route_mean_delay_delta"] = agg_spec["mean_delay_delta"].astype(np.float32)
        self.route_svd_map_["route_biz_share"] = agg_spec["biz_share"].astype(np.float32)

        rng = np.random.RandomState(self.seed)
        sub_size = min(120000, len(df_full))
        sub_idx = rng.choice(len(df_full), size=sub_size, replace=False)
        fd_sample = df_full["Flight Distance"].iloc[sub_idx].to_numpy(dtype=np.float64).reshape(-1, 1)
        self.vgm_fd_ = BayesianGaussianMixture(
            n_components=self.n_vgm_modes,
            covariance_type="spherical",
            max_iter=150,
            random_state=self.seed,
        )
        self.vgm_fd_.fit(fd_sample)

        self.orig_global_mean_ = float(df_orig["satisfaction"].mean())
        for c in raw_cols:
            if c in df_orig.columns:
                self.orig_means_[c] = df_orig.groupby(c)["satisfaction"].mean()

        avail_raw = [c for c in raw_cols if c in df_orig.columns]
        X_orig_enc = self._encode_for_oracle(df_orig[avail_raw])
        dtrain_orig = lgb.Dataset(X_orig_enc, label=df_orig["satisfaction"].to_numpy(dtype=np.float32))
        oracle_params = {
            "objective": "binary",
            "learning_rate": 0.04,
            "num_leaves": 63,
            "min_child_samples": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "n_jobs": -1,
            "num_threads": -1,
            "random_state": self.seed,
        }
        self.orig_oracle_model_ = lgb.train(oracle_params, dtrain_orig, num_boost_round=600)
        return self

    def _snap_flight_distance(self, fd_arr: np.ndarray) -> np.ndarray:
        if len(self.orig_fd_support_) == 0:
            return fd_arr
        idx = np.searchsorted(self.orig_fd_support_, fd_arr)
        idx_left = np.clip(idx - 1, 0, len(self.orig_fd_support_) - 1)
        idx_right = np.clip(idx, 0, len(self.orig_fd_support_) - 1)
        left_val = self.orig_fd_support_[idx_left]
        right_val = self.orig_fd_support_[idx_right]
        choose_left = np.abs(fd_arr - left_val) <= np.abs(fd_arr - right_val)
        snapped = np.where(choose_left, left_val, right_val)
        jitter = np.abs(fd_arr - snapped)
        return np.where(jitter <= 2, snapped, fd_arr)

    @staticmethod
    def _encode_for_oracle(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in CAT_COLS:
            if c in out.columns:
                out[c] = out[c].astype("category")
        return out

    def compute_adversarial_orig_weights(
        self, df_syn_train: pd.DataFrame, df_orig: pd.DataFrame
    ) -> np.ndarray:
        """
        Computes Radon-Nikodym density ratio weights w_orig(x) = P(syn|x)/P(orig|x) * (N_orig/N_syn)
        to eliminate the 0.21 TV covariate shift on Flight Distance when appending Original rows.
        """
        raw_cols = [c for c in (CAT_COLS + NUM_COLS + SURVEY_COLS) if c in df_orig.columns and c in df_syn_train.columns]
        X_adv = pd.concat(
            [df_syn_train[raw_cols], df_orig[raw_cols]], ignore_index=True
        )
        for c in CAT_COLS:
            if c in X_adv.columns:
                X_adv[c] = X_adv[c].astype("category")
        y_adv = np.concatenate(
            [np.ones(len(df_syn_train), dtype=np.float32), np.zeros(len(df_orig), dtype=np.float32)]
        )
        dadv = lgb.Dataset(X_adv, label=y_adv)
        adv_model = lgb.train(
            {
                "objective": "binary",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1,
                "n_jobs": -1,
                "num_threads": -1,
                "random_state": self.seed,
            },
            dadv,
            num_boost_round=250,
        )
        p_syn_on_orig = adv_model.predict(X_adv.iloc[len(df_syn_train):])
        p_syn_on_orig = np.clip(p_syn_on_orig, 0.02, 0.98)
        prior_ratio = len(df_orig) / float(len(df_syn_train))
        w_orig = (p_syn_on_orig / (1.0 - p_syn_on_orig)) * prior_ratio
        return np.clip(w_orig, 0.05, 5.0).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_cols = CAT_COLS + NUM_COLS + SURVEY_COLS
        out = pd.DataFrame(index=df.index)

        fd_raw = df["Flight Distance"].to_numpy(dtype=np.float64)
        fd_snap = self._snap_flight_distance(fd_raw)
        out["fd_snapped"] = fd_snap.astype(np.int32)
        out["fd_ctgan_jitter"] = (fd_raw - fd_snap).astype(np.float32)

        out["fd_div10"] = (fd_raw // 10).astype(np.int32)
        out["fd_mod10"] = (fd_raw % 10).astype(np.float32)
        out["fd_mod50"] = (fd_raw % 50).astype(np.float32)
        out["fd_mod100"] = (fd_raw % 100).astype(np.float32)
        out["fd_mod250"] = (fd_raw % 250).astype(np.float32)
        out["fd_mod1000"] = (fd_raw % 1000).astype(np.int32)
        out["age_mod10"] = (df["Age"] % 10).astype(np.float32)

        dep_d = df["Departure Delay in Minutes"].fillna(0.0).to_numpy(dtype=np.float32)
        arr_d = df["Arrival Delay in Minutes"].fillna(0.0).to_numpy(dtype=np.float32)
        out["delay_delta"] = arr_d - dep_d
        out["delay_ratio"] = (arr_d + 1.0) / (dep_d + 1.0)
        out["is_dep_zero"] = (dep_d == 0).astype(np.float32)
        out["is_arr_zero"] = (arr_d == 0).astype(np.float32)
        out["is_faa_delayed_15"] = (arr_d >= 15.0).astype(np.float32)
        out["is_severe_delayed_45"] = (arr_d >= 45.0).astype(np.float32)
        out["is_airborne_recovery"] = (arr_d < dep_d).astype(np.float32)
        out["is_arrival_missing"] = df["Arrival Delay in Minutes"].isna().astype(np.float32)

        for c in NUM_COLS:
            if c in self.full_counts_:
                out[f"{c}_full_cnt"] = (
                    df[c].fillna(-1).map(self.full_counts_[c]).fillna(1).astype(np.float32)
                )

        for c in raw_cols:
            if c in self.orig_means_:
                out[f"{c}_orig_mean"] = (
                    df[c].map(self.orig_means_[c]).fillna(self.orig_global_mean_).astype(np.float32)
                )

        if self.orig_oracle_model_ is not None:
            avail_raw = [c for c in raw_cols if c in df.columns]
            out["orig_oracle_logit"] = self.orig_oracle_model_.predict(
                self._encode_for_oracle(df[avail_raw]), raw_score=True
            ).astype(np.float32)

        if len(self.route_svd_map_) > 0:
            route_feats = self.route_svd_map_.reindex(fd_snap).fillna(0.0)
            route_feats.index = df.index
            out = pd.concat([out, route_feats], axis=1)

        if self.vgm_fd_ is not None:
            vgm_probs = self.vgm_fd_.predict_proba(fd_raw.reshape(-1, 1))
            best_mode = np.argmax(vgm_probs, axis=1)
            mode_means = self.vgm_fd_.means_.ravel()[best_mode]
            mode_stds = np.sqrt(self.vgm_fd_.covariances_.ravel()[best_mode]) + 1e-6
            out["vgm_fd_mode_id"] = best_mode.astype(np.int32)
            out["vgm_fd_alpha_norm"] = ((fd_raw - mode_means) / (4.0 * mode_stds)).astype(np.float32)
            out["vgm_fd_entropy"] = (
                -np.sum(vgm_probs * np.log(np.clip(vgm_probs, 1e-9, 1.0)), axis=1)
            ).astype(np.float32)

        if len(self.ecdf_full_df_) > 0:
            if len(df) == self.n_train_full_:
                ecdf_slice = self.ecdf_full_df_.iloc[:self.n_train_full_].copy()
                ecdf_slice.index = df.index
                out = pd.concat([out, ecdf_slice], axis=1)
            elif len(df) == self.n_test_full_:
                ecdf_slice = self.ecdf_full_df_.iloc[self.n_train_full_:].copy()
                ecdf_slice.index = df.index
                out = pd.concat([out, ecdf_slice], axis=1)
            else:
                temp_df = pd.DataFrame({"fd_snap": fd_snap}, index=df.index)
                for c in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
                    clean_name = "arr_delay" if "Arrival" in c else ("dep_delay" if "Departure" in c else "age")
                    temp_df[c] = df[c].fillna(0.0)
                    out[f"ecdf_{clean_name}_route"] = (
                        temp_df.groupby("fd_snap")[c].rank(pct=True).astype(np.float32)
                    )

        return out


class AsymmetricViewBuilder:
    """
    Constructs the 3 structurally decoupled feature views (View A: LightGBM Lattice,
    View B: XGBoost Psychometric & Target Encoded Crosses, View C: LightGBM Extra-Trees
    Haar Orthogonal Rotations) strictly inside each cross-validation fold.
    """

    def __init__(self, n_haar_projections: int = 28, seed: int = 42) -> None:
        self.n_haar_projections = n_haar_projections
        self.seed = seed
        self.fwl_ridge_: Optional[Ridge] = None
        self.kin_scaler_: Optional[StandardScaler] = None
        self.haar_scaler_: Optional[StandardScaler] = None
        self.haar_matrix_: Optional[np.ndarray] = None
        self.te_encoder_: Optional[TargetEncoder] = None
        self.cat_categories_: dict[str, list] = {}

    @staticmethod
    def _build_composite_keys(df: pd.DataFrame, dgp_feats: pd.DataFrame) -> pd.DataFrame:
        fd_str = dgp_feats["fd_snapped"].astype(str)
        fd10_str = dgp_feats["fd_div10"].astype(str)
        fd_m1000_str = dgp_feats["fd_mod1000"].astype(str)
        age_str = df["Age"].astype(str)
        cls_str = df["Class"].astype(str)
        trav_str = df["Type of Travel"].astype(str)
        cust_str = df["Customer Type"].astype(str)

        return pd.DataFrame(
            {
                "k_fd": fd_str,
                "k_fd10": fd10_str,
                "k_fd_m1000": fd_m1000_str,
                "k_fd_cls": fd_str + "_" + cls_str,
                "k_age_cls": age_str + "_" + cls_str,
                "k_age_trip": age_str + "_" + trav_str + "_" + cust_str,
                "k_cls_trav_wifi": cls_str + "_" + trav_str + "_" + df["Inflight wifi service"].astype(str),
                "k_onboard_wifi": df["Online boarding"].astype(str) + "_" + df["Inflight wifi service"].astype(str),
            },
            index=df.index,
        )

    def _extract_kinematic_matrix(self, df: pd.DataFrame, dgp_feats: pd.DataFrame) -> np.ndarray:
        kin_df = pd.DataFrame(index=df.index)
        kin_df["age"] = df["Age"].astype(np.float32)
        kin_df["log_fd"] = np.log1p(df["Flight Distance"].astype(np.float32))
        kin_df["log_dep"] = np.log1p(np.maximum(0.0, df["Departure Delay in Minutes"].fillna(0.0)))
        kin_df["log_arr"] = np.log1p(np.maximum(0.0, df["Arrival Delay in Minutes"].fillna(0.0)))
        kin_df["is_biz"] = (df["Class"] == "Business").astype(np.float32)
        kin_df["is_eco"] = (df["Class"] == "Eco").astype(np.float32)
        kin_df["is_pers"] = (df["Type of Travel"] == "Personal Travel").astype(np.float32)
        kin_df["is_disloyal"] = (df["Customer Type"] == "disloyal Customer").astype(np.float32)
        if "orig_oracle_logit" in dgp_feats.columns:
            kin_df["orig_logit"] = dgp_feats["orig_oracle_logit"]
        else:
            kin_df["orig_logit"] = 0.0
        return kin_df.to_numpy(dtype=np.float32)

    def fit_fold(
        self,
        df_tr: pd.DataFrame,
        y_tr: np.ndarray,
        dgp_tr: pd.DataFrame,
    ) -> "AsymmetricViewBuilder":
        self.kin_scaler_ = StandardScaler()
        X_kin = self.kin_scaler_.fit_transform(self._extract_kinematic_matrix(df_tr, dgp_tr))
        Y_surv = df_tr[SURVEY_COLS].to_numpy(dtype=np.float32)
        self.fwl_ridge_ = Ridge(alpha=100.0, random_state=self.seed)
        self.fwl_ridge_.fit(X_kin, Y_surv)

        keys_tr = self._build_composite_keys(df_tr, dgp_tr)
        self.te_encoder_ = TargetEncoder(cv=5, smooth="auto", random_state=self.seed)
        self.te_encoder_.fit(keys_tr, y_tr)

        haar_input = np.hstack([Y_surv, X_kin])
        self.haar_scaler_ = StandardScaler()
        self.haar_scaler_.fit(haar_input)
        d_dim = haar_input.shape[1]
        n_proj = min(self.n_haar_projections, d_dim)
        rng = np.random.RandomState(self.seed + 101)
        G_mat = rng.randn(d_dim, n_proj).astype(np.float32)
        Q_mat, R_mat = np.linalg.qr(G_mat)
        diag_signs = np.sign(np.diag(R_mat))
        diag_signs[diag_signs == 0] = 1.0
        self.haar_matrix_ = (Q_mat * diag_signs[None, :]).astype(np.float32)

        for col in CAT_COLS:
            self.cat_categories_[col] = sorted(df_tr[col].dropna().unique().tolist())
        for k_col in keys_tr.columns:
            self.cat_categories_[k_col] = sorted(keys_tr[k_col].unique().tolist())

        return self

    def transform_views(
        self,
        df: pd.DataFrame,
        dgp_feats: pd.DataFrame,
        irt_feats: pd.DataFrame,
        is_train_fold: bool = False,
        y_tr: Optional[np.ndarray] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns (X_view_a_lgbm, X_view_b_xgb, X_view_c_lgbm_xt).
        """
        X_kin = self.kin_scaler_.transform(self._extract_kinematic_matrix(df, dgp_feats))
        Y_surv = df[SURVEY_COLS].to_numpy(dtype=np.float32)

        R_fwl = Y_surv - self.fwl_ridge_.predict(X_kin)
        fwl_cols = [f"fwl_res_{c}" for c in SURVEY_COLS]
        df_fwl = pd.DataFrame(R_fwl.astype(np.float32), index=df.index, columns=fwl_cols)

        keys_df = self._build_composite_keys(df, dgp_feats)
        if is_train_fold and y_tr is not None:
            te_vals = self.te_encoder_.fit_transform(keys_df, y_tr)
        else:
            te_vals = self.te_encoder_.transform(keys_df)
        te_cols = [f"te_{c}" for c in keys_df.columns]
        df_te = pd.DataFrame(te_vals.astype(np.float32), index=df.index, columns=te_cols)

        haar_in = self.haar_scaler_.transform(np.hstack([Y_surv, X_kin]))
        P_haar = haar_in @ self.haar_matrix_
        haar_cols = [f"haar_proj_{i}" for i in range(P_haar.shape[1])]
        df_haar = pd.DataFrame(P_haar.astype(np.float32), index=df.index, columns=haar_cols)

        df_cats = pd.DataFrame(index=df.index)
        for c in CAT_COLS:
            codes = pd.Categorical(df[c], categories=self.cat_categories_[c]).codes.astype(np.float32)
            codes[codes < 0] = np.nan
            df_cats[c] = codes

        df_high_card = pd.DataFrame(index=df.index)
        for k_col in ["k_fd", "k_fd10", "k_fd_cls", "k_age_cls", "k_onboard_wifi"]:
            codes = pd.Categorical(
                keys_df[k_col], categories=self.cat_categories_[k_col]
            ).codes.astype(np.float32)
            codes[codes < 0] = np.nan
            df_high_card[f"cat_{k_col}"] = codes

        view_a = pd.concat(
            [
                df_cats,
                df[NUM_COLS + SURVEY_COLS].astype(np.float32),
                dgp_feats,
                df_fwl,
                df_high_card,
                irt_feats[["irt_theta_global", "irt_zero_cnt", "irt_zero_llr", "irt_person_fit_zh"]],
            ],
            axis=1,
        )

        drop_from_dgp_b = ["fd_mod10", "fd_mod50", "fd_mod250"]
        dgp_b = dgp_feats.drop(columns=[c for c in drop_from_dgp_b if c in dgp_feats.columns])
        fwl_sub = [c for c in ["fwl_res_Online boarding", "fwl_res_Inflight wifi service", "fwl_res_Seat comfort"] if c in df_fwl.columns]
        view_b = pd.concat(
            [
                df_cats,
                df[NUM_COLS + SURVEY_COLS].astype(np.float32),
                dgp_b,
                irt_feats,
                df_te,
                df_fwl[fwl_sub],
            ],
            axis=1,
        )

        dgp_c_cols = [
            c for c in [
                "orig_oracle_logit",
                "Flight Distance_full_cnt",
                "Age_full_cnt",
                "delay_delta",
                "delay_ratio",
                "route_log_vol",
                "route_biz_share",
            ] if c in dgp_feats.columns
        ]
        irt_c_cols = [c for c in ["irt_theta_global", "irt_diff_dig_cab", "irt_diff_cab_gnd", "irt_zero_llr"] if c in irt_feats.columns]
        te_c_cols = [c for c in ["te_k_fd", "te_k_fd_cls"] if c in df_te.columns]
        view_c = pd.concat(
            [
                df_cats,
                df[NUM_COLS + SURVEY_COLS].astype(np.float32),
                df_haar,
                dgp_feats[dgp_c_cols],
                irt_feats[irt_c_cols],
                df_te[te_c_cols],
            ],
            axis=1,
        )

        return view_a, view_b, view_c


def apply_structured_survey_dropout(
    df: pd.DataFrame,
    dropout_rate: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Applies stochastic structured survey dropout augmentation (Section 4.3):
    Masks 4% of observed ratings in the zero-capable survey items to 0 on pseudo-labeled rows,
    forcing tree ensembles to learn invariant decision rules when survey items are unrated.
    """
    df_aug = df.copy()
    rng = np.random.RandomState(seed)
    for col in ZERO_CAPABLE_SURVEY_COLS:
        if col in df_aug.columns:
            vals = df_aug[col].to_numpy().copy()
            observed = vals > 0
            drop_mask = (rng.rand(len(vals)) < dropout_rate) & observed
            vals[drop_mask] = 0
            df_aug[col] = vals
    return df_aug


def build_original_prior_teacher_and_stats(
    df_syn_train: pd.DataFrame,
    df_syn_test: pd.DataFrame,
    df_orig: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    target_col: str = "satisfaction",
    T_orig: float = 1.15,
    seed: int = 42,
    fast_mode: bool = False,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Isolates the 129,880-row original dataset from synthetic training folds (Tier 1).
    Extracts:
    1. Dedicated Teacher Ensemble Logits with exact Bayes Log-Odds Prior Shift:
       z_orig_adj = (z_orig / T_orig) + delta_prior
       where delta_prior = logit(pi_syn) - logit(pi_orig) = +0.45611
    2. Multi-way m-estimate smoothed target priors from original dataset:
       - ["Flight Distance", "Class", "Online boarding"]
       - ["Type of Travel", "Class", "Inflight wifi service", "Online boarding"]
       - ["Flight Distance", "Type of Travel", "Inflight wifi service"]
       alongside binary exact match indicators (orig_group_in_orig_k).
    Row order is strictly preserved with index integrity.
    """
    from sklearn.model_selection import StratifiedKFold

    y_syn = resolve_binary_target(df_syn_train[target_col]).astype(int)
    y_orig = resolve_binary_target(df_orig[target_col]).astype(int)

    pi_syn = float(np.mean(y_syn))     # ~0.5480
    pi_orig = float(np.mean(y_orig))   # ~0.4345
    delta_prior = float(logit(pi_syn) - logit(pi_orig))  # +0.45611

    if cat_cols is None:
        cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]

    if feature_cols is None:
        excluded = {"id", "Unnamed: 0", target_col, "is_original", "sample_weight"}
        feature_cols = [
            c for c in df_orig.columns
            if c not in excluded and c in df_syn_train.columns and c in df_syn_test.columns
        ]

    # Prepare datasets for teacher model
    X_orig = df_orig[feature_cols].copy()
    X_tr_syn = df_syn_train[feature_cols].copy()
    X_te_syn = df_syn_test[feature_cols].copy()

    for c in cat_cols:
        if c in X_orig.columns:
            X_orig[c] = X_orig[c].astype(str).astype("category")
        if c in X_tr_syn.columns:
            X_tr_syn[c] = X_tr_syn[c].astype(str).astype("category")
        if c in X_te_syn.columns:
            X_te_syn[c] = X_te_syn[c].astype(str).astype("category")

    n_splits = 2 if fast_mode else 5
    n_estimators = 60 if fast_mode else 1500
    early_stop = 20 if fast_mode else 50

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_syn_logits = np.zeros(len(df_syn_train), dtype=np.float64)
    te_syn_logits = np.zeros(len(df_syn_test), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_orig, y_orig)):
        model_lgb = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.035,
            num_leaves=63,
            subsample_for_bin=50000,
            colsample_bytree=0.80,
            subsample=0.80,
            random_state=seed + fold,
            n_jobs=n_jobs,
            verbose=-1,
        )
        model_lgb.fit(
            X_orig.iloc[tr_idx],
            y_orig[tr_idx],
            eval_set=[(X_orig.iloc[va_idx], y_orig[va_idx])],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        p_tr = np.clip(model_lgb.predict_proba(X_tr_syn)[:, 1], 1e-6, 1.0 - 1e-6)
        p_te = np.clip(model_lgb.predict_proba(X_te_syn)[:, 1], 1e-6, 1.0 - 1e-6)
        tr_syn_logits += logit(p_tr) / n_splits
        te_syn_logits += logit(p_te) / n_splits

    # Apply Bayes Log-Odds Prior Shift (+0.45611) and Noise Temperature Scaling (T_orig=1.15)
    for df_target, raw_z in [(df_syn_train, tr_syn_logits), (df_syn_test, te_syn_logits)]:
        z_adj = (raw_z / T_orig) + delta_prior
        p_adj = expit(z_adj)
        df_target["orig_teacher_logit_adj"] = z_adj
        df_target["orig_teacher_prob_adj"] = p_adj
        df_target["orig_teacher_entropy"] = -(
            p_adj * np.log(p_adj + 1e-12) + (1.0 - p_adj) * np.log(1.0 - p_adj + 1e-12)
        )

    # 2. Multi-Way Conditional Bayesian Priors from Original Data
    group_sets = [
        ["Flight Distance", "Class", "Online boarding"],
        ["Type of Travel", "Class", "Inflight wifi service", "Online boarding"],
        ["Flight Distance", "Type of Travel", "Inflight wifi service"],
    ]
    m_smooth = 10.0
    for idx, grp in enumerate(group_sets):
        valid_grp = [c for c in grp if c in df_orig.columns and c in df_syn_train.columns]
        if not valid_grp:
            continue
        stats = df_orig.groupby(valid_grp, observed=False)[target_col].agg(["count", "mean"]).reset_index()
        p_smooth_orig = (stats["count"] * stats["mean"] + m_smooth * pi_orig) / (stats["count"] + m_smooth)
        stats[f"orig_bayes_prior_{idx}"] = expit(logit(np.clip(p_smooth_orig, 1e-5, 1.0 - 1e-5)) + delta_prior)
        stats[f"orig_group_in_orig_{idx}"] = 1

        keep_cols = valid_grp + [f"orig_bayes_prior_{idx}", f"orig_group_in_orig_{idx}"]
        stats_sub = stats[keep_cols].drop_duplicates(subset=valid_grp)

        df_syn_train["_tmp_idx"] = np.arange(len(df_syn_train))
        df_syn_test["_tmp_idx"] = np.arange(len(df_syn_test))

        m_tr = pd.merge(df_syn_train[valid_grp + ["_tmp_idx"]], stats_sub, on=valid_grp, how="left").sort_values("_tmp_idx")
        m_te = pd.merge(df_syn_test[valid_grp + ["_tmp_idx"]], stats_sub, on=valid_grp, how="left").sort_values("_tmp_idx")

        df_syn_train[f"orig_group_in_orig_{idx}"] = m_tr[f"orig_group_in_orig_{idx}"].fillna(0).astype(np.int8).values
        df_syn_train[f"orig_bayes_prior_{idx}"] = m_tr[f"orig_bayes_prior_{idx}"].fillna(pi_syn).astype(np.float32).values

        df_syn_test[f"orig_group_in_orig_{idx}"] = m_te[f"orig_group_in_orig_{idx}"].fillna(0).astype(np.int8).values
        df_syn_test[f"orig_bayes_prior_{idx}"] = m_te[f"orig_bayes_prior_{idx}"].fillna(pi_syn).astype(np.float32).values

        df_syn_train.drop(columns=["_tmp_idx"], inplace=True)
        df_syn_test.drop(columns=["_tmp_idx"], inplace=True)

    return df_syn_train, df_syn_test


def engineer_s6e10_domain_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "satisfaction",
    n_splits: int = 10,
    seed: int = 42,
    fast_mode: bool = False,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs the high-leverage S6E10 feature set (Tier 2):
    1. Flight Distance treated as a discrete categorical Route ID + Route aggregations.
    2. Zero-inflation indicators (WiFi==0, Boarding==0) and non-zero psychometric row stats.
    3. Delay ratios and non-linear arrival-departure diffs.
    4. Leak-free Out-of-Fold Target Encoding on Route ID & Route x Class interactions.
    5. Self-supervised Auxiliary-Task OOF predictions (Business Travel, Business Class, Boarding residual).
    """
    from sklearn.model_selection import StratifiedKFold

    n_tr = len(df_train)
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # --- 1. Route ID (Flight Distance as Category) & Route-Level Context ---
    df_all["route_id"] = df_all["Flight Distance"].astype(int).astype(str)
    df_all["route_class_id"] = df_all["route_id"] + "_" + df_all["Class"].astype(str)
    df_all["route_travel_id"] = df_all["route_id"] + "_" + df_all["Type of Travel"].astype(str)

    # Frequency encoding on combined transductive covariate distribution (train + test)
    df_all["route_freq"] = df_all["route_id"].map(df_all["route_id"].value_counts()).astype(np.int32)
    df_all["route_class_freq"] = df_all["route_class_id"].map(df_all["route_class_id"].value_counts()).astype(np.int32)
    df_all["age_freq"] = df_all["Age"].map(df_all["Age"].value_counts()).astype(np.int32)

    # Route-level service & delay aggregations
    for agg_col in ["Online boarding", "Inflight wifi service", "Departure Delay in Minutes", "Seat comfort"]:
        if agg_col in df_all.columns:
            grp_mean = df_all.groupby("route_id", observed=False)[agg_col].transform("mean").astype(np.float32)
            grp_std = df_all.groupby("route_id", observed=False)[agg_col].transform("std").fillna(0).astype(np.float32)
            df_all[f"route_mean_{agg_col}"] = grp_mean
            df_all[f"route_diff_{agg_col}"] = (df_all[agg_col] - grp_mean).astype(np.float32)
            df_all[f"route_z_{agg_col}"] = ((df_all[agg_col] - grp_mean) / (grp_std + 1e-5)).astype(np.float32)

    # --- 2. Zero-Inflation & Non-Zero Survey Psychometric Statistics ---
    valid_survey = [c for c in SURVEY_COLS if c in df_all.columns]
    ratings_mat = df_all[valid_survey].values.astype(np.float32)
    df_all["n_zero_ratings"] = (ratings_mat == 0).sum(axis=1).astype(np.int8)
    df_all["n_five_ratings"] = (ratings_mat == 5).sum(axis=1).astype(np.int8)
    df_all["n_one_ratings"] = (ratings_mat == 1).sum(axis=1).astype(np.int8)
    df_all["zero_wifi_flag"] = (df_all["Inflight wifi service"] == 0).astype(np.int8)
    df_all["zero_boarding_flag"] = (df_all["Online boarding"] == 0).astype(np.int8)

    # Mask 0 ('Not Applicable') as NaN to compute pure rater-leniency statistics
    masked_ratings = np.where(ratings_mat == 0, np.nan, ratings_mat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_r = np.nanmean(masked_ratings, axis=1)
        std_r = np.nanstd(masked_ratings, axis=1)
        min_r = np.nanmin(masked_ratings, axis=1)
        max_r = np.nanmax(masked_ratings, axis=1)

    df_all["nonzero_rating_mean"] = np.nan_to_num(mean_r, nan=2.5).astype(np.float32)
    df_all["nonzero_rating_std"] = np.nan_to_num(std_r, nan=0.0).astype(np.float32)
    df_all["nonzero_rating_min"] = np.nan_to_num(min_r, nan=0.0).astype(np.float32)
    df_all["nonzero_rating_max"] = np.nan_to_num(max_r, nan=0.0).astype(np.float32)

    # Rater-centered primary features
    df_all["boarding_minus_user_mean"] = (df_all["Online boarding"] - df_all["nonzero_rating_mean"]).astype(np.float32)
    df_all["wifi_minus_user_mean"] = (df_all["Inflight wifi service"] - df_all["nonzero_rating_mean"]).astype(np.float32)
    if "Inflight entertainment" in df_all.columns:
        df_all["entertainment_minus_user_mean"] = (df_all["Inflight entertainment"] - df_all["nonzero_rating_mean"]).astype(np.float32)
    df_all["wifi_x_boarding"] = (df_all["Inflight wifi service"] * df_all["Online boarding"]).astype(np.float32)

    # --- 3. Delay Nonlinearities ---
    dep_d = df_all["Departure Delay in Minutes"].fillna(0).astype(np.float32)
    arr_d = df_all["Arrival Delay in Minutes"].fillna(dep_d).astype(np.float32)
    df_all["delay_diff"] = (arr_d - dep_d).astype(np.float32)
    df_all["delay_total"] = (arr_d + dep_d).astype(np.float32)
    df_all["is_zero_dep_delay"] = (dep_d == 0).astype(np.int8)
    df_all["is_zero_arr_delay"] = (arr_d == 0).astype(np.int8)
    df_all["delay_per_mile"] = (df_all["delay_total"] / (df_all["Flight Distance"] + 10.0)).astype(np.float32)

    df_tr = df_all.iloc[:n_tr].copy().reset_index(drop=True)
    df_te = df_all.iloc[n_tr:].copy().reset_index(drop=True)
    y = resolve_binary_target(df_tr[target_col]).astype(int)
    global_prior = float(np.mean(y))

    # --- 4. Out-of-Fold Bayesian Target Encoding on High-Cardinality Route IDs ---
    skf_te = StratifiedKFold(n_splits=n_splits if not fast_mode else 2, shuffle=True, random_state=seed)
    te_cols = ["route_id", "route_class_id", "route_travel_id"]
    m_weight = 15.0

    for col in te_cols:
        df_tr[f"oof_te_{col}"] = np.nan
        for tr_idx, va_idx in skf_te.split(df_tr, y):
            tr_fold = df_tr.iloc[tr_idx]
            stats = tr_fold.groupby(col, observed=False)[target_col].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + m_weight * global_prior) / (stats["count"] + m_weight)
            df_tr.iloc[va_idx, df_tr.columns.get_loc(f"oof_te_{col}")] = (
                df_tr.iloc[va_idx][col].map(smoothed).fillna(global_prior).values
            )
        full_stats = df_tr.groupby(col, observed=False)[target_col].agg(["count", "mean"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + m_weight * global_prior) / (full_stats["count"] + m_weight)
        df_te[f"oof_te_{col}"] = df_te[col].map(full_smoothed).fillna(global_prior).values

    # --- 5. Self-Supervised Auxiliary-Task Features (OOF across Train+Test) ---
    aux_candidates = valid_survey + ["Flight Distance", "Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    aux_features = [c for c in aux_candidates if c in df_all.columns]
    y_aux_travel = (df_all["Type of Travel"] == "Business travel").astype(int).values
    y_aux_class = (df_all["Class"] == "Business").astype(int).values
    y_aux_boarding = df_all["Online boarding"].values.astype(np.float32)

    aux_travel_oof = np.zeros(len(df_all), dtype=np.float32)
    aux_class_oof = np.zeros(len(df_all), dtype=np.float32)
    aux_boarding_oof = np.zeros(len(df_all), dtype=np.float32)

    n_aux_splits = 2 if fast_mode else 5
    n_aux_trees = 40 if fast_mode else 200
    skf_aux = StratifiedKFold(n_splits=n_aux_splits, shuffle=True, random_state=seed)
    features_no_boarding = [c for c in aux_features if c != "Online boarding"]

    for tr_idx, va_idx in skf_aux.split(df_all, y_aux_travel):
        # Aux 1: Predict Business Travel from ratings + distance
        clf_t = lgb.LGBMClassifier(
            n_estimators=n_aux_trees,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        clf_t.fit(df_all[aux_features].iloc[tr_idx], y_aux_travel[tr_idx])
        aux_travel_oof[va_idx] = clf_t.predict_proba(df_all[aux_features].iloc[va_idx])[:, 1]

        # Aux 2: Predict Business Class from ratings + distance
        clf_c = lgb.LGBMClassifier(
            n_estimators=n_aux_trees,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        clf_c.fit(df_all[aux_features].iloc[tr_idx], y_aux_class[tr_idx])
        aux_class_oof[va_idx] = clf_c.predict_proba(df_all[aux_features].iloc[va_idx])[:, 1]

        # Aux 3: Predict Expected Online Boarding from other ratings
        reg_b = lgb.LGBMRegressor(
            n_estimators=n_aux_trees,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        reg_b.fit(df_all[features_no_boarding].iloc[tr_idx], y_aux_boarding[tr_idx])
        aux_boarding_oof[va_idx] = reg_b.predict(df_all[features_no_boarding].iloc[va_idx])

    df_tr["aux_prob_biz_travel"] = aux_travel_oof[:n_tr]
    df_tr["aux_prob_biz_class"] = aux_class_oof[:n_tr]
    df_tr["aux_expected_boarding"] = aux_boarding_oof[:n_tr]
    df_tr["aux_boarding_residual"] = (df_tr["Online boarding"] - df_tr["aux_expected_boarding"]).astype(np.float32)

    df_te["aux_prob_biz_travel"] = aux_travel_oof[n_tr:]
    df_te["aux_prob_biz_class"] = aux_class_oof[n_tr:]
    df_te["aux_expected_boarding"] = aux_boarding_oof[n_tr:]
    df_te["aux_boarding_residual"] = (df_te["Online boarding"] - df_te["aux_expected_boarding"]).astype(np.float32)

    return df_tr, df_te


