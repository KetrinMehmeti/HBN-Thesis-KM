#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import logging
import math
import os
import warnings
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression

from checkpointing import feature_names_after_imputation

LOGGER = logging.getLogger(__name__)

TARGET_COLS = ["p_factor", "attention", "internalizing", "externalizing"]
COVARIATE_COLS = ["age", "sex"]
BASE_META_COLS = ["participant_id", "task", "n_epochs"]
NON_FEATURE_COLS = set(BASE_META_COLS + COVARIATE_COLS + TARGET_COLS)

AGGREGATE_SETS = {
    "mean_only": ("_mean",),
    "std_only": ("_std",),
    "median_only": ("_median",),
    "iqr_only": ("_iqr",),
    "mean_std": ("_mean", "_std"),
    "median_iqr": ("_median", "_iqr"),
    "central_tendency": ("_mean", "_median"),
    "variability": ("_std", "_iqr"),
    "all": ("_mean", "_std", "_median", "_iqr"),
}

FEATURE_FAMILIES = [
    "all",
    "bandpower_abs",
    "bandpower_rel",
    "bandpower",
    "theta_beta",
    "specparam",
    "hjorth",
    "entropy",
    "statistical",
    "temporal_complexity",
]

ALL_SELECTORS = ["none", "variance_threshold", "f_regression", "mutual_info", "spearman", "pca"]
ALL_MODELS = [
    "dummy",
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "bayesianridge",
    "huber",
    "pls",
    "decisiontree",
    "randomforest",
    "extratrees",
    "adaboost",
    "gradientboosting",
    "histgradientboosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "svr",
    "kernelridge",
    "knn",
]


def parse_s3_uri(path: str) -> Tuple[str, str]:
    if not str(path).startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {path}")
    no_scheme = path[5:]
    bucket, _, key = no_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {path}")
    return bucket, key


def is_s3_path(path: str) -> bool:
    return str(path).startswith("s3://")


@lru_cache(maxsize=None)
def s3_client(region_name: str = "us-east-1"):
    config = Config(
        retries={"total_max_attempts": 5, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=120,
        max_pool_connections=20,
    )
    session = boto3.Session(region_name=region_name)
    return session.client("s3", config=config)


def read_bytes(path: str, region_name: str = "us-east-1") -> bytes:
    if is_s3_path(path):
        bucket, key = parse_s3_uri(path)
        obj = s3_client(region_name).get_object(Bucket=bucket, Key=key)
        with obj["Body"] as body:
            return body.read()
    with open(path, "rb") as f:
        return f.read()


def write_bytes(path: str, data: bytes, region_name: str = "us-east-1") -> None:
    if is_s3_path(path):
        bucket, key = parse_s3_uri(path)
        s3_client(region_name).put_object(Bucket=bucket, Key=key, Body=data)
    else:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def exists_path(path: str, region_name: str = "us-east-1") -> bool:
    if is_s3_path(path):
        bucket, key = parse_s3_uri(path)
        try:
            s3_client(region_name).head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            error = exc.response.get("Error", {})
            if error.get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
    return os.path.exists(path)


def read_table(path: str, region_name: str = "us-east-1", **kwargs) -> pd.DataFrame:
    lower = str(path).lower()
    body = read_bytes(path, region_name=region_name)
    bio = io.BytesIO(body)
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(bio, **kwargs)
    if lower.endswith(".csv"):
        return pd.read_csv(bio, **kwargs)
    raise ValueError(f"Unsupported table extension for {path}. Use .csv or .parquet")


def write_table(df: pd.DataFrame, path: str, region_name: str = "us-east-1", index: bool = False) -> None:
    lower = str(path).lower()
    buf = io.BytesIO()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        df.to_parquet(buf, index=index)
    elif lower.endswith(".csv"):
        df.to_csv(buf, index=index)
    else:
        raise ValueError(f"Unsupported output extension for {path}. Use .csv or .parquet")
    write_bytes(path, buf.getvalue(), region_name=region_name)


def write_json(obj: Dict, path: str, region_name: str = "us-east-1") -> None:
    body = json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    write_bytes(path, body, region_name=region_name)


def read_json(path: str, region_name: str = "us-east-1") -> Dict:
    return json.loads(read_bytes(path, region_name=region_name).decode("utf-8"))


def join_path(base: str, *parts: str) -> str:
    base = str(base).rstrip("/")
    clean = [str(p).strip("/") for p in parts if str(p).strip("/")]
    return "/".join([base] + clean)


def list_s3_keys(bucket: str, prefix: str, suffix: Optional[str] = None, region_name: str = "us-east-1") -> List[str]:
    s3 = s3_client(region_name)
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if suffix is None or key.endswith(suffix):
                keys.append(key)
    return sorted(keys)


def list_aggregated_feature_files(
    input_bucket: str,
    input_prefix: str = "aggregated_subject_task",
    region_name: str = "us-east-1",
) -> List[Dict[str, str]]:
    keys = list_s3_keys(input_bucket, input_prefix, suffix="features_aggregated.parquet", region_name=region_name)
    rows = []
    for key in keys:
        parts = key.split("/")
        task = next((p.split("=", 1)[1] for p in parts if p.startswith("task=")), None)
        participant_id = next((p.split("=", 1)[1] for p in parts if p.startswith("participant_id=")), None)
        if task and participant_id:
            rows.append(
                {
                    "bucket": input_bucket,
                    "key": key,
                    "path": f"s3://{input_bucket}/{key}",
                    "task": task,
                    "participant_id": participant_id,
                }
            )
    return rows


def safe_corr(y_true: Sequence[float], y_pred: Sequence[float], method: str) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size < 2 or np.nanstd(yt) == 0 or np.nanstd(yp) == 0:
        return np.nan
    try:
        if method == "pearson":
            return float(stats.pearsonr(yt, yp).statistic)
        if method == "spearman":
            return float(stats.spearmanr(yt, yp).correlation)
    except Exception:
        return np.nan
    raise ValueError(method)


def concordance_correlation_coefficient(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size < 2:
        return np.nan
    var_t = float(np.var(yt))
    var_p = float(np.var(yp))
    if var_t == 0 or var_p == 0:
        return np.nan
    rho = safe_corr(yt, yp, "pearson")
    if not np.isfinite(rho):
        return np.nan
    mean_t, mean_p = float(np.mean(yt)), float(np.mean(yp))
    std_t, std_p = float(np.std(yt)), float(np.std(yp))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom == 0:
        return np.nan
    return float(2 * rho * std_t * std_p / denom)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return {"pearson": np.nan, "spearman": np.nan, "r2": np.nan, "mae": np.nan, "rmse": np.nan, "ccc": np.nan}
    out = {
        "pearson": safe_corr(yt, yp, "pearson"),
        "spearman": safe_corr(yt, yp, "spearman"),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
        "ccc": concordance_correlation_coefficient(yt, yp),
    }
    try:
        out["r2"] = float(r2_score(yt, yp)) if yt.size >= 2 else np.nan
    except Exception:
        out["r2"] = np.nan
    return out


def metric_to_score(metrics: Dict[str, float], primary_metric: str) -> float:
    if primary_metric == "neg_rmse":
        return -metrics["rmse"]
    if primary_metric == "neg_mae":
        return -metrics["mae"]
    value = metrics.get(primary_metric)
    return float(value) if value is not None else np.nan


def _family_match(col: str, feature_family: str) -> bool:
    if feature_family == "all":
        return True
    if feature_family == "bandpower_abs":
        return any(token in col for token in ["_delta_abs", "_theta_abs", "_alpha_abs", "_beta_abs", "_gamma_abs"])
    if feature_family == "bandpower_rel":
        return any(token in col for token in ["_delta_rel", "_theta_rel", "_alpha_rel", "_beta_rel", "_gamma_rel"])
    if feature_family == "bandpower":
        return _family_match(col, "bandpower_abs") or _family_match(col, "bandpower_rel")
    if feature_family == "theta_beta":
        return "theta_beta_ratio" in col
    if feature_family == "specparam":
        return "spec_exp" in col or "spec_offset" in col
    if feature_family == "hjorth":
        return any(token in col for token in ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"])
    if feature_family == "entropy":
        return "sample_entropy" in col or "perm_entropy" in col
    if feature_family == "statistical":
        return any(token in col for token in ["variance", "rms", "kurtosis", "skewness", "ptp"])
    if feature_family == "temporal_complexity":
        return _family_match(col, "hjorth") or _family_match(col, "entropy") or _family_match(col, "statistical")
    raise ValueError(f"Unknown feature_family: {feature_family}")


def select_feature_columns(
    df: pd.DataFrame,
    aggregate_set: str,
    feature_family: str,
    channel_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return numeric EEG features for one representation.

    When ``channel_ids`` is supplied, a feature must start with one of the
    exact ``<channel>_`` prefixes. This avoids selecting ``E20`` while
    requesting ``E2``.
    """
    if aggregate_set not in AGGREGATE_SETS:
        raise ValueError(f"Unknown aggregate_set: {aggregate_set}")
    suffixes = AGGREGATE_SETS[aggregate_set]
    channel_prefixes = None
    if channel_ids is not None:
        normalized_ids = [str(channel).strip() for channel in channel_ids]
        if not normalized_ids or any(not channel for channel in normalized_ids):
            raise ValueError("channel_ids must contain at least one non-empty channel ID")
        channel_prefixes = tuple(f"{channel}_" for channel in normalized_ids)

    cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not str(col).endswith(suffixes):
            continue
        if channel_prefixes is not None and not str(col).startswith(channel_prefixes):
            continue
        if _family_match(str(col), feature_family):
            cols.append(col)
    return cols


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove near-duplicate numeric columns using training data only."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("CorrelationFilter expects a two-dimensional matrix.")
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("Correlation threshold must be between 0 and 1.")
        if values.shape[1] == 0:
            self.support_mask_ = np.zeros(0, dtype=bool)
            return self

        correlations = np.corrcoef(values, rowvar=False, dtype=np.float32)
        correlations = np.nan_to_num(correlations, nan=0.0)
        upper = np.triu(np.abs(correlations), k=1)
        self.support_mask_ = ~np.any(upper >= float(self.threshold), axis=0)
        del correlations, upper
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_mask_]

    def get_support(self):
        return self.support_mask_.copy()

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features, dtype=object)
        return names[self.support_mask_]


class SafeSelectKBest(SelectKBest):
    """SelectKBest variant that tolerates upstream dimensionality reduction."""

    def fit(self, X, y):
        if self.k != "all":
            self.k = min(int(self.k), int(np.asarray(X).shape[1]))
        return super().fit(X, y)


def spearman_score_func(X, y):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    scores = np.zeros(X_arr.shape[1], dtype=float)
    for j in range(X_arr.shape[1]):
        x = X_arr[:, j]
        mask = np.isfinite(x) & np.isfinite(y_arr)
        if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y_arr[mask]) == 0:
            scores[j] = 0.0
        else:
            val = stats.spearmanr(x[mask], y_arr[mask]).correlation
            scores[j] = abs(val) if np.isfinite(val) else 0.0
    return scores, np.ones_like(scores)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def selector_step(selector_name: str, n_features: int, selector_params: Optional[Dict] = None):
    selector_params = selector_params or {}
    if selector_name == "none":
        return "passthrough"
    if selector_name == "variance_threshold":
        return VarianceThreshold(threshold=selector_params.get("threshold", 0.0))
    if selector_name == "f_regression":
        return SafeSelectKBest(score_func=f_regression, k=min(int(selector_params.get("k", 100)), n_features))
    if selector_name == "mutual_info":
        return SafeSelectKBest(score_func=mutual_info_regression, k=min(int(selector_params.get("k", 100)), n_features))
    if selector_name == "spearman":
        return SafeSelectKBest(score_func=spearman_score_func, k=min(int(selector_params.get("k", 100)), n_features))
    if selector_name == "pca":
        n_components = min(int(selector_params.get("n_components", 10)), n_features)
        return PCA(n_components=max(1, n_components), random_state=selector_params.get("random_state", 42))
    raise ValueError(f"Unknown selector: {selector_name}")


def model_needs_scaling(model_name: str) -> bool:
    return model_name in {"linear", "ridge", "lasso", "elasticnet", "bayesianridge", "huber", "pls", "svr", "kernelridge", "knn"}


def make_model(model_name: str, params: Optional[Dict] = None, random_state: int = 42):
    params = dict(params or {})
    name = model_name.lower()
    if name == "dummy":
        return DummyRegressor(strategy=params.get("strategy", "mean"))
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0), random_state=random_state)
    if name == "lasso":
        return Lasso(alpha=params.get("alpha", 0.01), max_iter=params.get("max_iter", 10000), random_state=random_state)
    if name == "elasticnet":
        return ElasticNet(
            alpha=params.get("alpha", 0.01),
            l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=params.get("max_iter", 10000),
            random_state=random_state,
        )
    if name == "bayesianridge":
        return BayesianRidge()
    if name == "huber":
        return HuberRegressor(alpha=params.get("alpha", 0.0001), epsilon=params.get("epsilon", 1.35), max_iter=1000)
    if name == "pls":
        return PLSRegression(n_components=max(1, int(params.get("n_components", 2))))
    if name == "decisiontree":
        return DecisionTreeRegressor(random_state=random_state, **params)
    if name == "randomforest":
        return RandomForestRegressor(random_state=random_state, n_jobs=params.pop("n_jobs", -1), **params)
    if name == "extratrees":
        return ExtraTreesRegressor(random_state=random_state, n_jobs=params.pop("n_jobs", -1), **params)
    if name == "adaboost":
        return AdaBoostRegressor(random_state=random_state, **params)
    if name == "gradientboosting":
        return GradientBoostingRegressor(random_state=random_state, **params)
    if name == "histgradientboosting":
        return HistGradientBoostingRegressor(random_state=random_state, **params)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception:
            warnings.warn("xgboost is not installed; skipping model.")
            return None
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", -1)
        return XGBRegressor(**params)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception:
            warnings.warn("lightgbm is not installed; skipping model.")
            return None
        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbose", -1)
        return LGBMRegressor(**params)
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception:
            warnings.warn("catboost is not installed; skipping model.")
            return None
        params.setdefault("random_state", random_state)
        params.setdefault("verbose", False)
        return CatBoostRegressor(**params)
    if name == "svr":
        return SVR(**params)
    if name == "kernelridge":
        return KernelRidge(**params)
    if name == "knn":
        return KNeighborsRegressor(**params)
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(
    eeg_feature_cols: Sequence[str],
    model_name: str,
    selector_name: str = "none",
    selector_params: Optional[Dict] = None,
    model_params: Optional[Dict] = None,
    numeric_extra_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    correlation_threshold: Optional[float] = None,
    random_state: int = 42,
) -> Pipeline:
    eeg_feature_cols = list(eeg_feature_cols)
    numeric_extra_cols = list(numeric_extra_cols or [])
    categorical_cols = list(categorical_cols or [])
    scale = model_needs_scaling(model_name)
    transformers = []
    if eeg_feature_cols:
        eeg_steps = [("imputer", SimpleImputer(strategy="median"))]
        if correlation_threshold is not None:
            eeg_steps.append(
                (
                    "correlation_filter",
                    CorrelationFilter(threshold=correlation_threshold),
                )
            )
        if scale:
            eeg_steps.append(("scaler", StandardScaler()))
        eeg_steps.append(
            (
                "selector",
                selector_step(
                    selector_name,
                    len(eeg_feature_cols),
                    selector_params,
                ),
            )
        )
        transformers.append(("eeg", Pipeline(eeg_steps), eeg_feature_cols))
    if numeric_extra_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric_extra", Pipeline(num_steps), numeric_extra_cols))
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("At least one EEG or confound column is required.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    model = make_model(model_name, model_params, random_state=random_state)
    if model is None:
        raise RuntimeError(f"Model {model_name} is unavailable in this environment.")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def suggest_selector_params(
    trial,
    selector_name: str,
    n_features: int,
    n_samples: int,
    random_state: int = 42,
    max_selected_features: Optional[int] = None,
) -> Dict:
    if selector_name in {"none", "variance_threshold"}:
        return {"threshold": 0.0} if selector_name == "variance_threshold" else {}
    if selector_name in {"f_regression", "mutual_info", "spearman"}:
        upper = min(n_features, max_selected_features or n_features)
        choices = sorted(
            {k for k in [50, 100, 200, 300, 500, 1000] if k <= upper}
            | {upper}
        )
        return {"k": trial.suggest_categorical("selector_k", choices)}
    if selector_name == "pca":
        upper = max(1, min(100, n_features, n_samples - 1))
        return {"n_components": trial.suggest_int("pca_n_components", 1, upper), "random_state": random_state}
    raise ValueError(selector_name)


def suggest_model_params(
    trial,
    model_name: str,
    n_features: int,
    n_samples: int,
    random_state: int = 42,
    svr_kernels: Optional[Sequence[str]] = None,
) -> Dict:
    name = model_name.lower()
    if name in {"dummy", "linear", "bayesianridge"}:
        return {}
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}
    if name == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e1, log=True), "max_iter": 20000}
    if name == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
            "max_iter": 20000,
        }
    if name == "huber":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e1, log=True), "epsilon": trial.suggest_float("epsilon", 1.1, 2.0)}
    if name == "pls":
        upper = max(1, min(50, n_features, n_samples - 1))
        return {"n_components": trial.suggest_int("pls_n_components", 1, upper)}
    if name == "decisiontree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if name in {"randomforest", "extratrees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(2, 21))),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
        }
    if name == "adaboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        }
    if name == "gradientboosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    if name == "histgradientboosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 500, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1e1, log=True),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500, step=50),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        }
    if name == "svr":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", list(svr_kernels or ["rbf", "linear"])
            ),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    if name == "kernelridge":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "polynomial"])
        params = {"alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True), "kernel": kernel}
        if kernel in {"rbf", "polynomial"}:
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
        return params
    if name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, min(50, max(3, n_samples - 1))),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    raise ValueError(f"Unknown model: {model_name}")


def make_optuna_objective(
    X_train: pd.DataFrame,
    y_train: Sequence[float],
    groups_train: Sequence,
    model_name: str,
    allowed_selectors: Sequence[str],
    primary_metric: str = "ccc",
    inner_folds: int = 3,
    random_state: int = 42,
    eeg_feature_cols: Optional[Sequence[str]] = None,
    numeric_extra_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    correlation_threshold: Optional[float] = None,
    selector_k_max: Optional[int] = None,
    svr_kernels: Optional[Sequence[str]] = None,
):
    y_train = np.asarray(y_train, dtype=float)
    groups_train = np.asarray(groups_train)
    eeg_feature_cols = list(eeg_feature_cols or X_train.select_dtypes(include=[np.number]).columns)
    numeric_extra_cols = list(numeric_extra_cols or [])
    categorical_cols = list(categorical_cols or [])
    n_features = max(1, len(eeg_feature_cols))

    def objective(trial):
        selector = trial.suggest_categorical("selector", list(allowed_selectors))
        selector_params = suggest_selector_params(
            trial,
            selector,
            n_features,
            len(X_train),
            random_state=random_state,
            max_selected_features=selector_k_max,
        )
        model_params = suggest_model_params(
            trial,
            model_name,
            n_features,
            len(X_train),
            random_state=random_state,
            svr_kernels=svr_kernels,
        )
        unique_groups = np.unique(groups_train)
        n_splits = min(inner_folds, len(unique_groups))
        if n_splits < 2:
            raise RuntimeError("Not enough participant groups for inner CV.")
        cv = GroupKFold(n_splits=n_splits)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train, groups_train):
            pipe = build_pipeline(
                eeg_feature_cols=eeg_feature_cols,
                model_name=model_name,
                selector_name=selector,
                selector_params=selector_params,
                model_params=model_params,
                numeric_extra_cols=numeric_extra_cols,
                categorical_cols=categorical_cols,
                correlation_threshold=correlation_threshold,
                random_state=random_state,
            )
            pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            pred = np.asarray(pipe.predict(X_train.iloc[val_idx])).ravel()
            score = metric_to_score(regression_metrics(y_train[val_idx], pred), primary_metric)
            if np.isfinite(score):
                scores.append(score)
        if not scores:
            return -1e9
        return float(np.mean(scores))

    return objective


def get_selected_feature_rows(
    fitted_pipeline: Pipeline,
    eeg_feature_cols: Sequence[str],
    run_metadata: Dict,
) -> List[Dict]:
    rows = []
    if not eeg_feature_cols:
        return rows
    try:
        pre = fitted_pipeline.named_steps["preprocessor"]
        eeg_pipe = pre.named_transformers_["eeg"]
        imputer = eeg_pipe.named_steps.get("imputer")
        transformed_feature_names = feature_names_after_imputation(
            imputer,
            eeg_feature_cols,
        )
        correlation_filter = eeg_pipe.named_steps.get("correlation_filter")
        if correlation_filter is not None and hasattr(
            correlation_filter, "get_support"
        ):
            transformed_feature_names = [
                feature_name
                for feature_name, keep in zip(
                    transformed_feature_names,
                    correlation_filter.get_support(),
                )
                if keep
            ]
        selector = eeg_pipe.named_steps.get("selector")
        selector_name = run_metadata.get("selector", "none")
        if selector_name in {"f_regression", "mutual_info", "spearman"} and hasattr(selector, "get_support"):
            mask = selector.get_support()
            scores = getattr(
                selector,
                "scores_",
                np.full(len(transformed_feature_names), np.nan),
            )
            selected = [
                (i, feature_name, scores[i] if i < len(scores) else np.nan)
                for i, feature_name in enumerate(transformed_feature_names)
                if mask[i]
            ]
            selected = sorted(selected, key=lambda t: -np.nan_to_num(t[2], nan=-np.inf))
            for rank, (_, name, score) in enumerate(selected, start=1):
                rows.append({**run_metadata, "feature_name": name, "feature_rank": rank, "feature_score": score})
        elif selector_name == "variance_threshold" and hasattr(selector, "get_support"):
            mask = selector.get_support()
            for rank, name in enumerate(
                [
                    feature_name
                    for feature_name, keep in zip(
                        transformed_feature_names,
                        mask,
                    )
                    if keep
                ],
                start=1,
            ):
                rows.append({**run_metadata, "feature_name": name, "feature_rank": rank, "feature_score": np.nan})
    except Exception as exc:
        LOGGER.warning("Could not extract selected features: %s", exc)
    return rows


def dump_joblib(obj, path: str, region_name: str = "us-east-1") -> None:
    if is_s3_path(path):
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        write_bytes(path, buf.getvalue(), region_name=region_name)
    else:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        joblib.dump(obj, path)
