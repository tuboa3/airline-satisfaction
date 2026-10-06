"""
Feature Engineering Pipeline: Implements All 10 Empirically Verified Paradigms
Plus Domain A (Density & Frequency Forensics) and Domain C (Transductive Encodings)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

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

        # Shelton Wang (11th place): 39 Rating-Context Crosses (+34 bps across all 5 folds)
        # Each service rating crossed with Class, Type of Travel, Customer Type
        context_cols = [
            c for c in ["Class", "Type of Travel", "Customer Type"] if c in df.columns
        ]
        for rc in rating_cols:
            r_str = df[rc].astype(str)
            for ctx in context_cols:
                cross_name = f"{rc}_x_{ctx}"
                crosses[cross_name] = r_str + "|" + df[ctx].astype(str)

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
                            X_orig_prior[cc] = pd.factorize(X_orig_prior[cc])[0]
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

            # Bounded Crosses String Representation
            if self.config.enable_bounded_crosses:
                bounded_dict = self._get_bounded_crosses(data)
                for b_col, b_series in bounded_dict.items():
                    data[b_col] = b_series.astype(str)
                data = data.copy()

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
            if self.orig_prior_model is not None and self.orig_prior_cols:
                try:
                    X_prior_eval = data[self.orig_prior_cols].copy()
                    for cat_c in self.config.categorical_cols:
                        if cat_c in X_prior_eval.columns:
                            mapping = self.label_encoder_dicts.get(cat_c, None)
                            if mapping is not None:
                                X_prior_eval[cat_c] = (
                                    X_prior_eval[cat_c].map(mapping).fillna(-1)
                                )
                            else:
                                X_prior_eval[cat_c] = pd.factorize(X_prior_eval[cat_c])[
                                    0
                                ]
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
            # 11. CATEGORICAL ENCODING
            # -------------------------------------------------------------
            cat_columns = list(self.label_encoders.keys())
            for col in cat_columns:
                if col in data.columns:
                    mapping = self.label_encoder_dicts.get(col)
                    if mapping is not None:
                        data[col] = (
                            data[col]
                            .astype(str)
                            .map(mapping)
                            .fillna(-1)
                            .astype(np.int16)
                        )
                    else:
                        data[col] = pd.factorize(data[col])[0].astype(np.int16)

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
