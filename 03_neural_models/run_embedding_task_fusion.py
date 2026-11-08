#!/usr/bin/env python3
"""Run one checkpointed E2B target shard with subject-level task fusion."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

from botocore.exceptions import ClientError
import numpy as np
import optuna
import pandas as pd

import downstream_regression_entrypoint  # Applies the PCA-aware PLS search-space correction.
import downstream_regression as pipeline

try:
    from .task_fusion import (
        fuse_mean_available,
        fuse_task_concat,
        fusion_audit,
        task_counts,
        validate_subject_metadata,
    )
except ImportError:
    from task_fusion import (
        fuse_mean_available,
        fuse_task_concat,
        fusion_audit,
        task_counts,
        validate_subject_metadata,
    )


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
FUSION_STRATEGIES = ["mean_available", "task_concat"]
INPUT_SETS = ["embedding_only", "embedding_plus_covariates"]
MODELS = [
    "ridge",
    "elasticnet",
    "pls",
    "extratrees",
    "histgradientboosting",
    "svr_rbf",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target", choices=pipeline.TARGETS, required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--representations", nargs="+", default=pipeline.REPRESENTATIONS
    )
    parser.add_argument(
        "--fusion-strategies", nargs="+", default=FUSION_STRATEGIES
    )
    parser.add_argument("--input-sets", nargs="+", default=INPUT_SETS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--min-tasks", type=int, default=2)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--worker-threads", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    unknown = {
        "tasks": sorted(set(args.tasks) - set(TASKS)),
        "representations": sorted(
            set(args.representations) - set(pipeline.REPRESENTATIONS)
        ),
        "fusion_strategies": sorted(
            set(args.fusion_strategies) - set(FUSION_STRATEGIES)
        ),
        "input_sets": sorted(set(args.input_sets) - set(INPUT_SETS)),
        "models": sorted(set(args.models) - set(MODELS)),
    }
    invalid = {name: values for name, values in unknown.items() if values}
    if invalid:
        raise ValueError(f"Unknown values: {invalid}")
    if not 2 <= args.min_tasks <= len(args.tasks):
        raise ValueError("min-tasks must be between 2 and the number of tasks")
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
    return (
        f"{output_prefix.strip('/')}/representations/{representation}/"
        f"target={args.target}/fusion={args.fusion_strategy}/"
        f"input={args.input_set}/model={model}"
    )


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "representation": result["representation"],
        "target": result["target"],
        "fusion_strategy": result["fusion_strategy"],
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
    expected = (
        len(args.representations)
        * len(args.fusion_strategies)
        * len(args.input_sets)
        * len(args.models)
    )
    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    runtime_base = f"{args.output_prefix.strip('/')}/runtime/{args.target}"

    try:
        manifest = {
            "phase": "E2B-task-fusion",
            "version": "v1",
            "status": "running",
            "created_at_utc": pipeline.utc_now(),
            "target": args.target,
            "tasks": args.tasks,
            "fusion_strategies": args.fusion_strategies,
            "representations": args.representations,
            "input_sets": args.input_sets,
            "models": args.models,
            "covariates": ["age", "sex"],
            "min_tasks": args.min_tasks,
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
            "expected_combinations": expected,
            "expected_outer_folds": expected * args.outer_folds,
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
        target_base = f"{args.output_prefix.strip('/')}/targets/{args.target}"
        pipeline.put_json(
            client,
            args.output_bucket,
            f"{target_base}/run_manifest.json",
            manifest,
        )

        completed = 0
        resumed = 0
        for representation in args.representations:
            frames: dict[str, pd.DataFrame] = {}
            sources: dict[str, dict[str, Any]] = {}
            embedding_columns: list[str] | None = None
            for task in args.tasks:
                frame, source = pipeline.download_dataset(
                    client,
                    args.input_bucket,
                    args.input_prefix,
                    representation,
                    task,
                    args.work_dir,
                )
                data, columns, _ = pipeline.validate_dataset(
                    frame, representation, args.target, task
                )
                if embedding_columns is None:
                    embedding_columns = columns
                elif columns != embedding_columns:
                    raise RuntimeError(
                        f"Inconsistent embedding columns for {representation}/{task}"
                    )
                frames[task] = data
                sources[task] = source
            if embedding_columns is None:
                raise RuntimeError(f"No task data loaded for {representation}")

            metadata = validate_subject_metadata(frames)
            counts = task_counts(frames)
            fused_by_strategy: dict[str, tuple[pd.DataFrame, list[str]]] = {}
            for strategy in args.fusion_strategies:
                if strategy == "mean_available":
                    fused = fuse_mean_available(
                        frames,
                        embedding_columns,
                        metadata,
                        counts,
                        args.min_tasks,
                    )
                else:
                    fused = fuse_task_concat(
                        frames,
                        embedding_columns,
                        metadata,
                        counts,
                        args.min_tasks,
                    )
                audit = fusion_audit(
                    fused[0],
                    fused[1],
                    representation,
                    args.target,
                    strategy,
                    args.min_tasks,
                    len(args.tasks),
                )
                audit["source_datasets"] = sources
                audits.append(audit)
                fused_by_strategy[strategy] = fused

            for strategy, (fused_data, feature_columns) in fused_by_strategy.items():
                data = fused_data.dropna(subset=[args.target]).reset_index(drop=True)
                args.fusion_strategy = strategy
                args.task = f"all_tasks__{strategy}"
                for input_set in args.input_sets:
                    args.input_set = input_set
                    for model_name in args.models:
                        base = combination_base(
                            args.output_prefix, representation, args, model_name
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
                            result = pipeline.run_combination(
                                client,
                                args,
                                representation,
                                model_name,
                                data,
                                feature_columns,
                                {
                                    "fusion_strategy": strategy,
                                    "task_sources": sources,
                                },
                            )
                            result.update(
                                {
                                    "fusion_strategy": strategy,
                                    "tasks_fused": args.tasks,
                                    "min_tasks": args.min_tasks,
                                }
                            )
                            pipeline.put_json(
                                client,
                                args.output_bucket,
                                f"{base}/run_config.json",
                                result,
                            )
                            pipeline.put_json(
                                client,
                                args.output_bucket,
                                done_key,
                                result,
                            )
                        results.append(result)
                        completed += 1
                        progress = {
                            "status": "running",
                            "target": args.target,
                            "completed_combinations": completed,
                            "expected_combinations": expected,
                            "resumed_combinations": resumed,
                            "last_representation": representation,
                            "last_fusion_strategy": strategy,
                            "last_input_set": input_set,
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

        if len(results) != expected:
            raise RuntimeError(
                f"Expected {expected} results, obtained {len(results)}"
            )
        summary_base = f"{target_base}/summary"
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
            "tasks_fused": len(args.tasks),
            "fusion_strategies_completed": len(args.fusion_strategies),
            "representations_completed": len(args.representations),
            "input_sets_completed": args.input_sets,
            "models_completed": len(args.models),
            "combinations_completed": len(results),
            "expected_combinations": expected,
            "folds_completed": len(results) * args.outer_folds,
            "expected_folds": expected * args.outer_folds,
            "resumed_combinations": resumed,
            "participant_leakage_detected": False,
            "errors": [],
            "elapsed_seconds": time.time() - started,
            "benchmark_log_key": f"{summary_base}/benchmark_log.csv",
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
            f"{target_base}/done.json",
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
            "expected_combinations": expected,
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


if __name__ == "__main__":
    sys.exit(main())


