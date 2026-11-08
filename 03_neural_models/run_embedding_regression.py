#!/usr/bin/env python3
"""Run one checkpointed E2A target shard across all tasks and embeddings."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target", choices=pipeline.TARGETS, required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--input-set",
        choices=["embedding_only", "embedding_plus_covariates"],
        default="embedding_plus_covariates",
    )
    parser.add_argument(
        "--representations", nargs="+", default=pipeline.REPRESENTATIONS
    )
    parser.add_argument("--models", nargs="+", default=pipeline.MODELS)
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
    unknown_models = sorted(set(args.models) - set(pipeline.MODELS))
    if unknown_tasks or unknown_representations or unknown_models:
        raise ValueError(
            "Unknown values: "
            f"tasks={unknown_tasks}, representations={unknown_representations}, "
            f"models={unknown_models}"
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


def progress_key(args: argparse.Namespace) -> str:
    return f"{args.output_prefix.strip('/')}/runtime/{args.target}/progress.json"


def error_key(args: argparse.Namespace) -> str:
    return f"{args.output_prefix.strip('/')}/runtime/{args.target}/error.json"


def target_base(args: argparse.Namespace) -> str:
    return f"{args.output_prefix.strip('/')}/targets/{args.target}"


def benchmark_log(results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
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


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    client = pipeline.build_s3_client(args.region, args.worker_threads)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    expected_combinations = (
        len(args.tasks) * len(args.representations) * len(args.models)
    )
    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    try:
        manifest = {
            "phase": "E2A",
            "version": "full_v1",
            "status": "running",
            "created_at_utc": pipeline.utc_now(),
            "target": args.target,
            "tasks": args.tasks,
            "input_set": args.input_set,
            "representations": args.representations,
            "models": args.models,
            "embedding_reductions": ["raw", "pca"],
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
            f"{target_base(args)}/run_manifest.json",
            manifest,
        )

        completed = 0
        resumed = 0
        for task in args.tasks:
            args.task = task
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
                audit["source_dataset"] = source
                audits.append(audit)
                for model_name in args.models:
                    base = pipeline.combination_base(
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
                            embedding_cols,
                            source,
                        )
                    results.append(result)
                    completed += 1
                    progress = {
                        "status": "running",
                        "target": args.target,
                        "completed_combinations": completed,
                        "expected_combinations": expected_combinations,
                        "resumed_combinations": resumed,
                        "last_task": task,
                        "last_representation": representation,
                        "last_model": model_name,
                        "completed_outer_folds": completed * args.outer_folds,
                        "expected_outer_folds": expected_combinations
                        * args.outer_folds,
                        "errors": 0,
                        "elapsed_seconds": time.time() - started,
                    }
                    pipeline.put_json(
                        client,
                        args.output_bucket,
                        progress_key(args),
                        progress,
                    )
                    print(json.dumps(progress), flush=True)

        if len(results) != expected_combinations:
            raise RuntimeError(
                f"Expected {expected_combinations} results, obtained {len(results)}"
            )
        summary_base = f"{target_base(args)}/summary"
        pipeline.upload_frame(
            client,
            benchmark_log(results),
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
            "models_completed": len(args.models),
            "combinations_completed": len(results),
            "expected_combinations": expected_combinations,
            "folds_completed": len(results) * args.outer_folds,
            "expected_folds": expected_combinations * args.outer_folds,
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
            f"{target_base(args)}/done.json",
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
                client, args.output_bucket, error_key(args), error
            )
        except Exception:
            pass
        print(json.dumps(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
