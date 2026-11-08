#!/usr/bin/env python3
"""Checkpointed E2A input ablation for one psychopathology target.

This phase keeps the completed E2A v2 results immutable and computes only the
two missing input sets:

* frozen EEG embedding without age/sex;
* age/sex without any EEG embedding.

The covariate baseline is evaluated once per task/model, rather than repeated
for every embedding representation.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import sys
import time
from typing import Any

from botocore.exceptions import ClientError
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import downstream_regression_entrypoint  # Applies the PCA-aware PLS correction.
import downstream_regression as pipeline


TASKS = [
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
    "RestingState",
    "ThePresent",
    "contrastChangeDetection_1",
    "contrastChangeDetection_2",
    "contrastChangeDetection_3",
    "seqLearning6target",
    "seqLearning8target",
    "surroundSupp_1",
    "surroundSupp_2",
    "symbolSearch",
]
EMBEDDING_MODELS = [model for model in pipeline.MODELS if model != "dummy"]
COVARIATE_MODELS = list(pipeline.MODELS)
COVARIATE_REPRESENTATION = "covariates_baseline"
COVARIATE_SOURCE_REPRESENTATION = "cbramod"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--combined-reference-prefix", required=True)
    parser.add_argument("--target", choices=pipeline.TARGETS, required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--representations", nargs="+", default=pipeline.REPRESENTATIONS
    )
    parser.add_argument(
        "--embedding-models", nargs="+", default=EMBEDDING_MODELS
    )
    parser.add_argument(
        "--covariate-models", nargs="+", default=COVARIATE_MODELS
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--worker-threads", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    unknown_tasks = sorted(set(args.tasks) - set(TASKS))
    unknown_representations = sorted(
        set(args.representations) - set(pipeline.REPRESENTATIONS)
    )
    unknown_embedding_models = sorted(
        set(args.embedding_models) - set(EMBEDDING_MODELS)
    )
    unknown_covariate_models = sorted(
        set(args.covariate_models) - set(COVARIATE_MODELS)
    )
    if (
        unknown_tasks
        or unknown_representations
        or unknown_embedding_models
        or unknown_covariate_models
    ):
        raise ValueError(
            "Unknown values: "
            f"tasks={unknown_tasks}, representations={unknown_representations}, "
            f"embedding_models={unknown_embedding_models}, "
            f"covariate_models={unknown_covariate_models}"
        )
    if args.outer_folds < 2 or args.inner_folds < 2 or args.n_trials < 1:
        raise ValueError("Nested CV requires at least two folds and one trial")
    return args


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def read_json(client, bucket: str, key: str) -> dict[str, Any]:
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        return json.loads(body.read().decode("utf-8"))


def combination_base(
    output_prefix: str,
    representation: str,
    args: argparse.Namespace,
    model: str,
) -> str:
    root = output_prefix.strip("/")
    if args.input_set == "covariates_only":
        return (
            f"{root}/input_sets/covariates_only/target={args.target}/"
            f"task={args.task}/model={model}"
        )
    return (
        f"{root}/input_sets/embedding_only/representations/{representation}/"
        f"target={args.target}/task={args.task}/model={model}"
    )


def build_covariate_pipeline(
    model_name: str,
    model_params: dict[str, Any],
    random_state: int,
    threads: int,
) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if model_name in pipeline.SCALED_MODELS:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            ("age", Pipeline(numeric_steps), ["age"]),
            (
                "sex",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", pipeline.make_one_hot_encoder()),
                    ]
                ),
                ["sex"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                pipeline.make_model(
                    model_name, model_params, random_state, threads
                ),
            ),
        ]
    )


def covariate_objective_factory(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    inner_folds: int,
    random_state: int,
    threads: int,
):
    def objective(trial: optuna.Trial) -> float:
        _, _, model_params = pipeline.suggest_configuration(
            trial, model_name, 2, len(X)
        )
        splitter = pipeline.GroupKFold(
            n_splits=min(inner_folds, np.unique(groups).size)
        )
        fold_rmse: list[float] = []
        for train_index, validation_index in splitter.split(X, y, groups):
            estimator = build_covariate_pipeline(
                model_name, model_params, random_state, threads
            )
            estimator.fit(X.iloc[train_index], y[train_index])
            prediction = np.asarray(
                estimator.predict(X.iloc[validation_index])
            ).ravel()
            fold_rmse.append(
                pipeline.mean_squared_error(
                    y[validation_index], prediction
                )
                ** 0.5
            )
        return -float(np.mean(fold_rmse))

    return objective


def run_covariate_combination(
    client,
    args: argparse.Namespace,
    model_name: str,
    data: pd.DataFrame,
    source: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    X = data[["age", "sex"]].copy()
    y = data[args.target].to_numpy(dtype=float)
    groups = data["participant_id"].to_numpy(dtype=str)
    splitter = pipeline.GroupKFold(
        n_splits=min(args.outer_folds, np.unique(groups).size)
    )
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    base = combination_base(
        args.output_prefix, COVARIATE_REPRESENTATION, args, model_name
    )

    for outer_fold, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(
                f"Participant leakage in covariates/{model_name}/fold-{outer_fold}"
            )
        seed = pipeline.stable_seed(
            args.random_state, COVARIATE_REPRESENTATION, model_name, outer_fold
        )
        if model_name == "dummy":
            model_params: dict[str, Any] = {}
            best_value = math.nan
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
            study.optimize(
                covariate_objective_factory(
                    X.iloc[train_index].reset_index(drop=True),
                    y[train_index],
                    groups[train_index],
                    model_name,
                    args.inner_folds,
                    seed,
                    args.worker_threads,
                ),
                n_trials=args.n_trials,
                n_jobs=1,
                show_progress_bar=False,
            )
            _, _, model_params = pipeline.configuration_from_best_trial(
                study.best_trial, model_name, 2, len(train_index)
            )
            best_value = float(study.best_value)
            for trial in study.trials:
                trial_rows.append(
                    {
                        "representation": COVARIATE_REPRESENTATION,
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

        estimator = build_covariate_pipeline(
            model_name, model_params, seed, args.worker_threads
        )
        estimator.fit(X.iloc[train_index], y[train_index])
        predictions = np.asarray(estimator.predict(X.iloc[test_index])).ravel()
        fold_metrics = pipeline.metrics(y[test_index], predictions)
        common = {
            "representation": COVARIATE_REPRESENTATION,
            "target": args.target,
            "task": args.task,
            "input_set": args.input_set,
            "model": model_name,
            "outer_fold": outer_fold,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "participant_overlap": len(overlap),
            "embedding_reduction": "not_applicable",
        }
        metrics_rows.append({**common, **fold_metrics})
        best_rows.append(
            {
                **common,
                "inner_objective_neg_rmse": best_value,
                "reduction_params_json": "{}",
                "model_params_json": json.dumps(model_params, sort_keys=True),
            }
        )
        for row_index, prediction in zip(test_index, predictions):
            prediction_rows.append(
                {
                    "representation": COVARIATE_REPRESENTATION,
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
    pipeline.upload_frame(
        client,
        metrics_frame,
        args.output_bucket,
        f"{base}/metrics_by_fold.csv",
        args.work_dir,
    )
    pipeline.upload_frame(
        client,
        predictions_frame,
        args.output_bucket,
        f"{base}/fold_predictions.parquet",
        args.work_dir,
    )
    pipeline.upload_frame(
        client,
        best_frame,
        args.output_bucket,
        f"{base}/best_trials.csv",
        args.work_dir,
    )
    if not trials_frame.empty:
        pipeline.upload_frame(
            client,
            trials_frame,
            args.output_bucket,
            f"{base}/optuna_trials.csv",
            args.work_dir,
        )
    result = {
        "status": "succeeded",
        "created_at_utc": pipeline.utc_now(),
        "representation": COVARIATE_REPRESENTATION,
        "target": args.target,
        "task": args.task,
        "input_set": args.input_set,
        "model": model_name,
        "source_dataset": source,
        "n_subjects": int(data["participant_id"].nunique()),
        "embedding_dimension": 0,
        "covariates": ["age", "sex"],
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "n_trials": 0 if model_name == "dummy" else args.n_trials,
        "participant_overlap_all_folds": int(
            metrics_frame["participant_overlap"].sum()
        ),
        "metrics_mean": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    pipeline.put_json(
        client, args.output_bucket, f"{base}/run_config.json", result
    )
    pipeline.put_json(client, args.output_bucket, f"{base}/done.json", result)
    return result


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
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


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    client = pipeline.build_s3_client(args.region, args.worker_threads)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pipeline.combination_base = combination_base
    embedding_expected = (
        len(args.tasks)
        * len(args.representations)
        * len(args.embedding_models)
    )
    covariate_expected = len(args.tasks) * len(args.covariate_models)
    expected_combinations = embedding_expected + covariate_expected
    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    resumed = 0
    runtime_base = f"{args.output_prefix.strip('/')}/runtime/{args.target}"

    try:
        manifest = {
            "phase": "E2A-input-ablation",
            "version": "v1",
            "status": "running",
            "created_at_utc": pipeline.utc_now(),
            "target": args.target,
            "tasks": args.tasks,
            "input_sets_computed": ["embedding_only", "covariates_only"],
            "combined_reference_prefix": args.combined_reference_prefix,
            "representations": args.representations,
            "embedding_models": args.embedding_models,
            "covariate_models": args.covariate_models,
            "covariates": ["age", "sex"],
            "covariate_source_representation": COVARIATE_SOURCE_REPRESENTATION,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "n_trials": args.n_trials,
            "primary_optimization_metric": "negative_rmse",
            "reported_metrics": [
                "pearson",
                "spearman",
                "r2",
                "mae",
                "rmse",
                "ccc",
            ],
            "split_unit": "participant_id",
            "expected_combinations": expected_combinations,
            "expected_outer_folds": expected_combinations * args.outer_folds,
            "resume_from_done_markers": True,
            "source_data_mutation": False,
            "python": sys.version,
            "platform": platform.platform(),
            "library_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "optuna": optuna.__version__,
            },
        }
        pipeline.put_json(
            client,
            args.output_bucket,
            f"{args.output_prefix.strip('/')}/targets/{args.target}/run_manifest.json",
            manifest,
        )

        completed = 0
        for task in args.tasks:
            args.task = task
            args.input_set = "embedding_only"
            for representation in args.representations:
                frame, source = pipeline.download_dataset(
                    client,
                    args.input_bucket,
                    args.input_prefix,
                    representation,
                    task,
                    args.work_dir,
                )
                data, embedding_cols, audit = pipeline.validate_dataset(
                    frame, representation, args.target, task
                )
                audit.update(
                    {"input_set": args.input_set, "source_dataset": source}
                )
                audits.append(audit)
                for model_name in args.embedding_models:
                    base = combination_base(
                        args.output_prefix, representation, args, model_name
                    )
                    done_key = f"{base}/done.json"
                    if object_exists(client, args.output_bucket, done_key):
                        result = read_json(
                            client, args.output_bucket, done_key
                        )
                        if result.get("status") != "succeeded":
                            raise RuntimeError(
                                f"Non-successful checkpoint at s3://{args.output_bucket}/{done_key}"
                            )
                        resumed += 1
                    else:
                        result = pipeline.run_combination(
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
                    put_progress(
                        client,
                        args,
                        runtime_base,
                        started,
                        completed,
                        expected_combinations,
                        resumed,
                        task,
                        representation,
                        model_name,
                    )

            args.input_set = "covariates_only"
            frame, source = pipeline.download_dataset(
                client,
                args.input_bucket,
                args.input_prefix,
                COVARIATE_SOURCE_REPRESENTATION,
                task,
                args.work_dir,
            )
            data, _, audit = pipeline.validate_dataset(
                frame,
                COVARIATE_SOURCE_REPRESENTATION,
                args.target,
                task,
            )
            audit.update(
                {
                    "representation": COVARIATE_REPRESENTATION,
                    "input_set": args.input_set,
                    "embedding_dimension": 0,
                    "finite_embeddings": "not_applicable",
                    "source_dataset": source,
                }
            )
            audits.append(audit)
            for model_name in args.covariate_models:
                base = combination_base(
                    args.output_prefix,
                    COVARIATE_REPRESENTATION,
                    args,
                    model_name,
                )
                done_key = f"{base}/done.json"
                if object_exists(client, args.output_bucket, done_key):
                    result = read_json(client, args.output_bucket, done_key)
                    if result.get("status") != "succeeded":
                        raise RuntimeError(
                            f"Non-successful checkpoint at s3://{args.output_bucket}/{done_key}"
                        )
                    resumed += 1
                else:
                    result = run_covariate_combination(
                        client, args, model_name, data, source
                    )
                results.append(result)
                completed += 1
                put_progress(
                    client,
                    args,
                    runtime_base,
                    started,
                    completed,
                    expected_combinations,
                    resumed,
                    task,
                    COVARIATE_REPRESENTATION,
                    model_name,
                )

        if len(results) != expected_combinations:
            raise RuntimeError(
                f"Expected {expected_combinations} results, obtained {len(results)}"
            )

        summary_base = (
            f"{args.output_prefix.strip('/')}/targets/{args.target}/summary"
        )
        pipeline.upload_frame(
            client,
            pd.DataFrame([result_row(result) for result in results]),
            args.output_bucket,
            f"{summary_base}/benchmark_log.csv",
            args.work_dir,
        )
        pipeline.upload_frame(
            client,
            pd.DataFrame(audits),
            args.output_bucket,
            f"{summary_base}/input_audit.csv",
            args.work_dir,
        )
        summary = {
            "status": "succeeded",
            "created_at_utc": pipeline.utc_now(),
            "target": args.target,
            "tasks_completed": len(args.tasks),
            "representations_completed": len(args.representations),
            "input_sets_completed": ["embedding_only", "covariates_only"],
            "combinations_completed": len(results),
            "expected_combinations": expected_combinations,
            "folds_completed": len(results) * args.outer_folds,
            "expected_folds": expected_combinations * args.outer_folds,
            "resumed_combinations": resumed,
            "participant_leakage_detected": False,
            "errors": [],
            "elapsed_seconds": time.time() - started,
            "benchmark_log_key": f"{summary_base}/benchmark_log.csv",
            "combined_reference_prefix": args.combined_reference_prefix,
        }
        pipeline.put_json(
            client,
            args.output_bucket,
            f"{summary_base}/summary.json",
            summary,
        )
        pipeline.put_json(
            client,
            args.output_bucket,
            f"{args.output_prefix.strip('/')}/targets/{args.target}/done.json",
            summary,
        )
        print(json.dumps(summary), flush=True)
        return 0
    except (ClientError, OSError, RuntimeError, ValueError, KeyError) as exc:
        error = {
            "status": "failed",
            "created_at_utc": pipeline.utc_now(),
            "target": args.target,
            "type": type(exc).__name__,
            "message": str(exc),
            "completed_combinations": len(results),
            "expected_combinations": expected_combinations,
            "elapsed_seconds": time.time() - started,
        }
        try:
            pipeline.put_json(
                client,
                args.output_bucket,
                f"{runtime_base}/error.json",
                error,
            )
        except Exception:
            pass
        print(json.dumps(error), file=sys.stderr, flush=True)
        return 1


def put_progress(
    client,
    args: argparse.Namespace,
    runtime_base: str,
    started: float,
    completed: int,
    expected: int,
    resumed: int,
    task: str,
    representation: str,
    model_name: str,
) -> None:
    progress = {
        "status": "running",
        "target": args.target,
        "completed_combinations": completed,
        "expected_combinations": expected,
        "resumed_combinations": resumed,
        "last_input_set": args.input_set,
        "last_task": task,
        "last_representation": representation,
        "last_model": model_name,
        "completed_outer_folds": completed * args.outer_folds,
        "expected_outer_folds": expected * args.outer_folds,
        "errors": 0,
        "elapsed_seconds": time.time() - started,
    }
    pipeline.put_json(
        client,
        args.output_bucket,
        f"{runtime_base}/progress.json",
        progress,
    )
    print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    sys.exit(main())
