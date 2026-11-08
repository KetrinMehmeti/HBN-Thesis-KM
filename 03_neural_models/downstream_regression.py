#!/usr/bin/env python3
"""Diagnostic nested-CV smoke test for frozen EEG embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


REPRESENTATIONS = [
    "cbramod",
    "biot",
    "labram_128_spatial",
    "eegpt_62_canonical",
    "eegpt_128_spatial",
]
MODELS = [
    "dummy",
    "ridge",
    "elasticnet",
    "pls",
    "extratrees",
    "histgradientboosting",
    "svr_rbf",
]
TARGETS = ["attention", "p_factor", "internalizing", "externalizing"]
SCALED_MODELS = {"ridge", "elasticnet", "pls", "svr_rbf"}

try:
    from .embedding_columns import (
        EMBEDDING_COLUMN_RULE,
        select_embedding_columns,
    )
except ImportError:
    from embedding_columns import (
        EMBEDDING_COLUMN_RULE,
        select_embedding_columns,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target", choices=TARGETS, default="attention")
    parser.add_argument("--task", default="RestingState")
    parser.add_argument(
        "--input-set",
        choices=["embedding_only", "embedding_plus_covariates"],
        default="embedding_plus_covariates",
    )
    parser.add_argument("--representations", nargs="+", default=REPRESENTATIONS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--outer-folds", type=int, default=2)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--worker-threads", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    unknown_representations = sorted(set(args.representations) - set(REPRESENTATIONS))
    unknown_models = sorted(set(args.models) - set(MODELS))
    if unknown_representations or unknown_models:
        raise ValueError(
            f"Unknown representations={unknown_representations}, models={unknown_models}"
        )
    if min(args.outer_folds, args.inner_folds, args.n_trials, args.worker_threads) < 1:
        raise ValueError("Fold, trial, and worker counts must be positive")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("Nested CV requires at least two outer and inner folds")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_s3_client(region: str, worker_threads: int):
    return boto3.Session(region_name=region).client(
        "s3",
        config=Config(
            retries={"total_max_attempts": 8, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=120,
            max_pool_connections=max(20, worker_threads * 2),
        ),
    )


def put_json(client, bucket: str, key: str, value: dict[str, Any]) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def upload_frame(
    client,
    frame: pd.DataFrame,
    bucket: str,
    key: str,
    work_dir: Path,
) -> None:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    suffix = Path(key).suffix.lower()
    local_path = work_dir / f"{digest}{suffix}"
    if suffix == ".parquet":
        frame.to_parquet(local_path, index=False, compression="zstd")
        content_type = "application/vnd.apache.parquet"
    elif suffix == ".csv":
        frame.to_csv(local_path, index=False)
        content_type = "text/csv"
    else:
        raise ValueError(f"Unsupported table output: {key}")
    client.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type, "ServerSideEncryption": "AES256"},
    )
    local_path.unlink(missing_ok=True)


def download_dataset(
    client,
    bucket: str,
    input_prefix: str,
    representation: str,
    task: str,
    work_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key = (
        f"{input_prefix.strip('/')}/representations/{representation}/"
        f"tasks/{task}/dataset.parquet"
    )
    head = client.head_object(Bucket=bucket, Key=key)
    local_path = work_dir / f"{representation}__{task}.parquet"
    client.download_file(
        bucket,
        key,
        str(local_path),
        Config=TransferConfig(max_concurrency=8, use_threads=True),
    )
    frame = pd.read_parquet(local_path)
    local_path.unlink(missing_ok=True)
    return frame, {
        "bucket": bucket,
        "key": key,
        "etag": str(head.get("ETag", "")).strip('"'),
        "bytes": int(head["ContentLength"]),
    }


def validate_dataset(
    frame: pd.DataFrame,
    representation: str,
    target: str,
    task: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    required = {"participant_id", "task", "age", "sex", target}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{representation}: missing columns {missing}")
    if frame["participant_id"].duplicated().any():
        raise RuntimeError(f"{representation}: duplicate participant IDs within {task}")
    observed_tasks = sorted(frame["task"].dropna().astype(str).unique().tolist())
    if observed_tasks != [task]:
        raise RuntimeError(
            f"{representation}: expected task {task}, observed {observed_tasks}"
        )
    embedding_cols = select_embedding_columns(frame.columns, representation)
    embedding_values = frame[embedding_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(embedding_values).all():
        raise RuntimeError(f"{representation}: non-finite embedding values")
    data = frame.dropna(subset=["participant_id", target]).copy()
    data["participant_id"] = data["participant_id"].astype(str)
    data[target] = data[target].astype(float)
    if data["participant_id"].nunique() < 100:
        raise RuntimeError(f"{representation}: fewer than 100 labeled subjects")
    audit = {
        "representation": representation,
        "task": task,
        "target": target,
        "rows_source": int(len(frame)),
        "rows_labeled": int(len(data)),
        "unique_subjects": int(data["participant_id"].nunique()),
        "embedding_dimension": len(embedding_cols),
        "embedding_column_rule": EMBEDDING_COLUMN_RULE,
        "excluded_embedding_metadata": sorted(
            str(column)
            for column in frame.columns
            if str(column).startswith("embedding_") and column not in embedding_cols
        ),
        "finite_embeddings": True,
        "target_mean": float(data[target].mean()),
        "target_std": float(data[target].std()),
    }
    return data.reset_index(drop=True), embedding_cols, audit


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model(name: str, params: dict[str, Any], random_state: int, threads: int):
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge(**params)
    if name == "elasticnet":
        return ElasticNet(max_iter=20000, random_state=random_state, **params)
    if name == "pls":
        return PLSRegression(scale=False, **params)
    if name == "extratrees":
        return ExtraTreesRegressor(
            random_state=random_state, n_jobs=threads, **params
        )
    if name == "histgradientboosting":
        return HistGradientBoostingRegressor(random_state=random_state, **params)
    if name == "svr_rbf":
        return SVR(kernel="rbf", **params)
    raise ValueError(name)


def suggest_configuration(
    trial: optuna.Trial,
    model_name: str,
    n_features: int,
    n_samples: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    reduction = trial.suggest_categorical("embedding_reduction", ["raw", "pca"])
    reduction_params: dict[str, Any] = {}
    if reduction == "pca":
        upper = max(2, min(128, n_features, n_samples - 1))
        choices = sorted({value for value in [16, 32, 64, 128] if value <= upper} | {upper})
        reduction_params["n_components"] = trial.suggest_categorical(
            "pca_n_components", choices
        )
    if model_name == "ridge":
        model_params = {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}
    elif model_name == "elasticnet":
        model_params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    elif model_name == "pls":
        model_params = {
            "n_components": trial.suggest_int(
                "pls_n_components", 1, min(32, n_features, n_samples - 1)
            )
        }
    elif model_name == "extratrees":
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.3, 0.6, 1.0]
            ),
        }
    elif model_name == "histgradientboosting":
        model_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 300, step=100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-6, 1e1, log=True
            ),
        }
    elif model_name == "svr_rbf":
        model_params = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    else:
        raise ValueError(model_name)
    return reduction, reduction_params, model_params


def build_pipeline(
    embedding_cols: list[str],
    input_set: str,
    model_name: str,
    reduction: str,
    reduction_params: dict[str, Any],
    model_params: dict[str, Any],
    random_state: int,
    threads: int,
) -> Pipeline:
    embed_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if reduction == "pca":
        embed_steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=int(reduction_params["n_components"]),
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_name in SCALED_MODELS:
        embed_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = [
        ("embedding", Pipeline(embed_steps), embedding_cols)
    ]
    if input_set == "embedding_plus_covariates":
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if model_name in SCALED_MODELS:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.extend(
            [
                ("age", Pipeline(numeric_steps), ["age"]),
                (
                    "sex",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", make_one_hot_encoder()),
                        ]
                    ),
                    ["sex"],
                ),
            ]
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", make_model(model_name, model_params, random_state, threads)),
        ]
    )


def concordance_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else math.nan


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    pearson = math.nan if np.std(y_true) == 0 or np.std(y_pred) == 0 else float(pearsonr(y_true, y_pred).statistic)
    spearman = math.nan if np.std(y_true) == 0 or np.std(y_pred) == 0 else float(spearmanr(y_true, y_pred).statistic)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "ccc": concordance_correlation(y_true, y_pred),
    }


def stable_seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def objective_factory(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    embedding_cols: list[str],
    input_set: str,
    model_name: str,
    inner_folds: int,
    random_state: int,
    threads: int,
):
    def objective(trial: optuna.Trial) -> float:
        reduction, reduction_params, model_params = suggest_configuration(
            trial, model_name, len(embedding_cols), len(X)
        )
        splitter = GroupKFold(n_splits=min(inner_folds, np.unique(groups).size))
        fold_rmse: list[float] = []
        for train_index, validation_index in splitter.split(X, y, groups):
            pipeline = build_pipeline(
                embedding_cols,
                input_set,
                model_name,
                reduction,
                reduction_params,
                model_params,
                random_state,
                threads,
            )
            pipeline.fit(X.iloc[train_index], y[train_index])
            prediction = np.asarray(pipeline.predict(X.iloc[validation_index])).ravel()
            fold_rmse.append(mean_squared_error(y[validation_index], prediction) ** 0.5)
        return -float(np.mean(fold_rmse))

    return objective


def configuration_from_best_trial(
    trial: optuna.trial.FrozenTrial,
    model_name: str,
    n_features: int,
    n_samples: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    class FrozenSuggestor:
        def __init__(self, params: dict[str, Any]):
            self.params = params

        def suggest_categorical(self, name: str, choices: list[Any]):
            return self.params[name]

        def suggest_float(self, name: str, *args, **kwargs):
            return self.params[name]

        def suggest_int(self, name: str, *args, **kwargs):
            return self.params[name]

    return suggest_configuration(FrozenSuggestor(trial.params), model_name, n_features, n_samples)


def combination_base(output_prefix: str, representation: str, args: argparse.Namespace, model: str) -> str:
    return (
        f"{output_prefix.strip('/')}/representations/{representation}/"
        f"target={args.target}/task={args.task}/input={args.input_set}/model={model}"
    )


def run_combination(
    client,
    args: argparse.Namespace,
    representation: str,
    model_name: str,
    data: pd.DataFrame,
    embedding_cols: list[str],
    source: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    X_columns = embedding_cols + (["age", "sex"] if args.input_set == "embedding_plus_covariates" else [])
    X = data[X_columns].copy()
    y = data[args.target].to_numpy(dtype=float)
    groups = data["participant_id"].to_numpy(dtype=str)
    splitter = GroupKFold(n_splits=min(args.outer_folds, np.unique(groups).size))
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    base = combination_base(args.output_prefix, representation, args, model_name)

    for outer_fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(
                f"Participant leakage in {representation}/{model_name}/fold-{outer_fold}"
            )
        seed = stable_seed(args.random_state, representation, model_name, outer_fold)
        if model_name == "dummy":
            reduction, reduction_params, model_params = "raw", {}, {}
            best_value = math.nan
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
            study.optimize(
                objective_factory(
                    X.iloc[train_index].reset_index(drop=True),
                    y[train_index],
                    groups[train_index],
                    embedding_cols,
                    args.input_set,
                    model_name,
                    args.inner_folds,
                    seed,
                    args.worker_threads,
                ),
                n_trials=args.n_trials,
                n_jobs=1,
                show_progress_bar=False,
            )
            reduction, reduction_params, model_params = configuration_from_best_trial(
                study.best_trial,
                model_name,
                len(embedding_cols),
                len(train_index),
            )
            best_value = float(study.best_value)
            for trial in study.trials:
                trial_rows.append(
                    {
                        "representation": representation,
                        "target": args.target,
                        "task": args.task,
                        "input_set": args.input_set,
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "trial": trial.number,
                        "state": trial.state.name,
                        "objective_neg_rmse": trial.value,
                        "params_json": json.dumps(trial.params, sort_keys=True),
                    }
                )
        pipeline = build_pipeline(
            embedding_cols,
            args.input_set,
            model_name,
            reduction,
            reduction_params,
            model_params,
            seed,
            args.worker_threads,
        )
        pipeline.fit(X.iloc[train_index], y[train_index])
        predictions = np.asarray(pipeline.predict(X.iloc[test_index])).ravel()
        fold_metrics = metrics(y[test_index], predictions)
        common = {
            "representation": representation,
            "target": args.target,
            "task": args.task,
            "input_set": args.input_set,
            "model": model_name,
            "outer_fold": outer_fold,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "participant_overlap": len(overlap),
            "embedding_reduction": reduction,
        }
        metrics_rows.append({**common, **fold_metrics})
        best_rows.append(
            {
                **common,
                "inner_objective_neg_rmse": best_value,
                "reduction_params_json": json.dumps(reduction_params, sort_keys=True),
                "model_params_json": json.dumps(model_params, sort_keys=True),
            }
        )
        for row_index, prediction in zip(test_index, predictions):
            prediction_rows.append(
                {
                    "representation": representation,
                    "target": args.target,
                    "task": args.task,
                    "input_set": args.input_set,
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "participant_id": groups[row_index],
                    "y_true": y[row_index],
                    "y_pred": float(prediction),
                }
            )

    metrics_frame = pd.DataFrame(metrics_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    best_frame = pd.DataFrame(best_rows)
    trials_frame = pd.DataFrame(trial_rows)
    aggregate = {
        metric: float(metrics_frame[metric].mean())
        for metric in ["pearson", "spearman", "r2", "mae", "rmse", "ccc"]
    }
    upload_frame(client, metrics_frame, args.output_bucket, f"{base}/metrics_by_fold.csv", args.work_dir)
    upload_frame(client, predictions_frame, args.output_bucket, f"{base}/fold_predictions.parquet", args.work_dir)
    upload_frame(client, best_frame, args.output_bucket, f"{base}/best_trials.csv", args.work_dir)
    if not trials_frame.empty:
        upload_frame(client, trials_frame, args.output_bucket, f"{base}/optuna_trials.csv", args.work_dir)
    result = {
        "status": "succeeded",
        "created_at_utc": utc_now(),
        "representation": representation,
        "target": args.target,
        "task": args.task,
        "input_set": args.input_set,
        "model": model_name,
        "source_dataset": source,
        "n_subjects": int(data["participant_id"].nunique()),
        "embedding_dimension": len(embedding_cols),

        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "n_trials": 0 if model_name == "dummy" else args.n_trials,
        "participant_overlap_all_folds": int(metrics_frame["participant_overlap"].sum()),
        "metrics_mean": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    put_json(client, args.output_bucket, f"{base}/run_config.json", result)
    put_json(client, args.output_bucket, f"{base}/done.json", result)
    return result


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    client = build_s3_client(args.region, args.worker_threads)
    runtime_base = f"{args.output_prefix.strip('/')}/runtime"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    try:
        run_manifest = {
            "phase": "E2A-smoke",
            "status": "running",
            "created_at_utc": utc_now(),
            "target": args.target,
            "task": args.task,
            "input_set": args.input_set,
            "representations": args.representations,
            "models": args.models,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "n_trials": args.n_trials,
            "primary_optimization_metric": "negative_rmse",
            "reported_metrics": ["pearson", "spearman", "r2", "mae", "rmse", "ccc"],
            "split_unit": "participant_id",
            "embedding_reductions": ["raw", "pca"],
            "python": sys.version,
            "platform": platform.platform(),
            "library_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "optuna": optuna.__version__,
            },
        }
        put_json(client, args.output_bucket, f"{args.output_prefix.strip('/')}/run_manifest.json", run_manifest)
        total = len(args.representations) * len(args.models)
        completed = 0
        for representation in args.representations:
            frame, source = download_dataset(
                client,
                args.input_bucket,
                args.input_prefix,
                representation,
                args.task,
                args.work_dir,
            )
            data, embedding_cols, audit = validate_dataset(
                frame, representation, args.target, args.task
            )
            audit["source_dataset"] = source
            audits.append(audit)
            for model_name in args.models:
                result = run_combination(
                    client,
                    args,
                    representation,
                    model_name,
                    data,
                    embedding_cols,
                    source,
                )
                results.append(result)
                completed += 1
                progress = {
                    "status": "running",
                    "completed_combinations": completed,
                    "total_combinations": total,
                    "last_representation": representation,
                    "last_model": model_name,
                    "errors": 0,
                    "elapsed_seconds": time.time() - started,
                }
                put_json(client, args.output_bucket, f"{runtime_base}/progress.json", progress)
                print(json.dumps(progress), flush=True)

        benchmark_log = pd.DataFrame(
            [
                {
                    "representation": result["representation"],
                    "target": result["target"],
                    "task": result["task"],
                    "input_set": result["input_set"],
                    "model": result["model"],
                    "n_subjects": result["n_subjects"],
                    "embedding_dimension": result["embedding_dimension"],
                    **result["metrics_mean"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "status": result["status"],
                }
                for result in results
            ]
        )
        summary_base = f"{args.output_prefix.strip('/')}/summary"
        upload_frame(client, benchmark_log, args.output_bucket, f"{summary_base}/benchmark_log.csv", args.work_dir)
        upload_frame(client, pd.DataFrame(audits), args.output_bucket, f"{summary_base}/input_audit.csv", args.work_dir)
        summary = {
            "status": "succeeded",
            "created_at_utc": utc_now(),
            "target": args.target,
            "task": args.task,
            "input_set": args.input_set,
            "representations_completed": len(args.representations),
            "models_completed": len(args.models),
            "combinations_completed": len(results),
            "expected_combinations": total,
            "folds_completed": len(results) * args.outer_folds,
            "participant_leakage_detected": False,
            "errors": [],
            "elapsed_seconds": time.time() - started,
            "benchmark_log_key": f"{summary_base}/benchmark_log.csv",
        }
        put_json(client, args.output_bucket, f"{summary_base}/summary.json", summary)
        put_json(client, args.output_bucket, f"{args.output_prefix.strip('/')}/done.json", summary)
        print(json.dumps(summary), flush=True)
        return 0
    except (ClientError, OSError, RuntimeError, ValueError, KeyError) as exc:
        error = {
            "status": "failed",
            "created_at_utc": utc_now(),
            "type": type(exc).__name__,
            "message": str(exc),
            "completed_combinations": len(results),
            "elapsed_seconds": time.time() - started,
        }
        try:
            put_json(client, args.output_bucket, f"{runtime_base}/error.json", error)
        except Exception:
            pass
        print(json.dumps(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
