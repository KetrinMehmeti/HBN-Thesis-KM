#!/usr/bin/env python3
"""Audit and consolidate one frozen EEG representation into task-level Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
import numpy as np
import pandas as pd


TARGET_COLUMNS = ["attention", "p_factor", "internalizing", "externalizing"]
COVARIATE_COLUMNS = ["age", "sex"]
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
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--targets-uri", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--expected-files", type=int, default=14066)
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI, got {uri}")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def build_client(region: str, threads: int):
    session = boto3.Session(region_name=region)
    return session.client(
        "s3",
        config=Config(
            retries={"total_max_attempts": 8, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=120,
            max_pool_connections=max(20, threads * 2),
        ),
    )


def list_objects(client, bucket: str, prefix: str) -> list[dict[str, Any]]:
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            if item["Key"].endswith(".npz"):
                objects.append(
                    {
                        "key": item["Key"],
                        "etag": str(item.get("ETag", "")).strip('"'),
                        "size": int(item["Size"]),
                    }
                )
    return sorted(objects, key=lambda item: item["key"])


def parse_embedding_key(source_prefix: str, key: str) -> tuple[str, str]:
    base = source_prefix.rstrip("/") + "/"
    if not key.startswith(base) or not key.endswith(".npz"):
        raise ValueError(f"Unexpected embedding key: {key}")
    relative = key[len(base) :]
    parts = relative.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Expected <task>/<subject>.npz below {base}, got {key}")
    return parts[0], parts[1][:-4]


def read_json(client, bucket: str, key: str) -> dict[str, Any]:
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        return json.loads(body.read().decode("utf-8"))


def put_json(client, bucket: str, key: str, value: dict[str, Any]) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def download_targets(client, targets_uri: str, destination: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    bucket, key = parse_s3_uri(targets_uri)
    head = client.head_object(Bucket=bucket, Key=key)
    transfer = TransferConfig(max_concurrency=8, use_threads=True)
    client.download_file(bucket, key, str(destination), Config=transfer)
    columns = ["participant_id", *COVARIATE_COLUMNS, *TARGET_COLUMNS]
    frame = pd.read_parquet(destination, columns=columns)
    frame["participant_id"] = frame["participant_id"].astype(str)

    conflicts: dict[str, int] = {}
    for column in [*COVARIATE_COLUMNS, *TARGET_COLUMNS]:
        conflicts[column] = int(
            frame.groupby("participant_id", sort=False)[column]
            .nunique(dropna=True)
            .gt(1)
            .sum()
        )
    if any(conflicts.values()):
        raise RuntimeError(f"Target table has conflicting subject-level values: {conflicts}")

    subjects = frame.drop_duplicates("participant_id", keep="first").copy()
    subjects = subjects.sort_values("participant_id").reset_index(drop=True)
    manifest = {
        "source_uri": targets_uri,
        "source_etag": str(head.get("ETag", "")).strip('"'),
        "source_bytes": int(head["ContentLength"]),
        "rows_in_source": int(len(frame)),
        "unique_subjects": int(subjects["participant_id"].nunique()),
        "columns": columns,
        "conflicting_subject_values": conflicts,
    }
    return subjects, manifest


def load_subject_mean(client, bucket: str, item: dict[str, Any], source_prefix: str) -> dict[str, Any]:
    task, participant_id = parse_embedding_key(source_prefix, item["key"])
    buffer = io.BytesIO()
    client.download_fileobj(bucket, item["key"], buffer)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        if "subject_mean" not in archive or "epoch_index" not in archive:
            raise RuntimeError(f"Missing subject_mean or epoch_index in {item['key']}")
        vector = np.asarray(archive["subject_mean"], dtype=np.float32)
        n_epochs = int(np.asarray(archive["epoch_index"]).shape[0])
    if vector.ndim != 1 or vector.size == 0:
        raise RuntimeError(f"Invalid subject_mean shape {vector.shape} in {item['key']}")
    if not np.isfinite(vector).all():
        raise RuntimeError(f"Non-finite subject_mean in {item['key']}")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"Empty subject_mean in {item['key']}")
    return {
        "participant_id": participant_id,
        "task": task,
        "n_epochs": n_epochs,
        "source_embedding_key": item["key"],
        "source_embedding_etag": item["etag"],
        "source_embedding_bytes": item["size"],
        "embedding_norm": norm,
        "vector": vector,
    }


def dataframe_from_rows(rows: list[dict[str, Any]], targets: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if not rows:
        raise RuntimeError("Cannot build a task dataset without embedding rows")
    dimensions = {int(row["vector"].shape[0]) for row in rows}
    if len(dimensions) != 1:
        raise RuntimeError(f"Mixed embedding dimensions: {sorted(dimensions)}")
    dimension = dimensions.pop()
    rows = sorted(rows, key=lambda row: row["participant_id"])
    vectors = np.stack([row.pop("vector") for row in rows]).astype(np.float32, copy=False)
    metadata = pd.DataFrame(rows)
    if metadata["participant_id"].duplicated().any():
        duplicates = metadata.loc[metadata["participant_id"].duplicated(), "participant_id"].tolist()
        raise RuntimeError(f"Duplicate subjects inside one task: {duplicates[:10]}")
    dataset = metadata.merge(targets, on="participant_id", how="left", validate="one_to_one")
    feature_names = [f"embedding_{index:04d}" for index in range(dimension)]
    feature_frame = pd.DataFrame(vectors, columns=feature_names)
    dataset = pd.concat([dataset.reset_index(drop=True), feature_frame], axis=1)
    return dataset, vectors


def task_audit(dataset: pd.DataFrame, vectors: np.ndarray, representation: str, task: str) -> dict[str, Any]:
    variances = np.var(vectors.astype(np.float64), axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "status": "succeeded",
        "created_at_utc": utc_now(),
        "representation": representation,
        "task": task,
        "rows": int(len(dataset)),
        "unique_subjects": int(dataset["participant_id"].nunique()),
        "embedding_dimension": int(vectors.shape[1]),
        "total_epochs": int(dataset["n_epochs"].sum()),
        "finite_embedding_values": bool(np.isfinite(vectors).all()),
        "zero_variance_dimensions": int(np.count_nonzero(variances == 0)),
        "near_zero_variance_dimensions": int(np.count_nonzero(variances < 1e-12)),
        "mean_dimension_variance": float(np.mean(variances)),
        "embedding_norm_min": float(np.min(norms)),
        "embedding_norm_mean": float(np.mean(norms)),
        "embedding_norm_max": float(np.max(norms)),
        "missing_targets": {column: int(dataset[column].isna().sum()) for column in TARGET_COLUMNS},
        "available_targets": {column: int(dataset[column].notna().sum()) for column in TARGET_COLUMNS},
    }


def upload_parquet(client, frame: pd.DataFrame, bucket: str, key: str, work_dir: Path) -> None:
    destination = work_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.parquet"
    frame.to_parquet(destination, index=False, compression="zstd")
    client.upload_file(
        str(destination),
        bucket,
        key,
        ExtraArgs={
            "ContentType": "application/vnd.apache.parquet",
            "ServerSideEncryption": "AES256",
        },
    )
    destination.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    started = time.time()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(args.region, args.threads)
    output_base = args.output_prefix.strip("/")
    representation_base = f"{output_base}/representations/{args.representation}"
    runtime_base = f"{output_base}/runtime/{args.representation}"

    try:
        objects = list_objects(client, args.input_bucket, args.source_prefix)
        if len(objects) != args.expected_files:
            raise RuntimeError(
                f"Expected {args.expected_files} embeddings for {args.representation}, found {len(objects)}"
            )
        by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
        unknown_tasks: set[str] = set()
        for item in objects:
            task, _ = parse_embedding_key(args.source_prefix, item["key"])
            if task not in by_task:
                unknown_tasks.add(task)
            else:
                by_task[task].append(item)
        if unknown_tasks:
            raise RuntimeError(f"Unknown tasks in source prefix: {sorted(unknown_tasks)}")
        missing_tasks = [task for task, task_objects in by_task.items() if not task_objects]
        if missing_tasks:
            raise RuntimeError(f"No embeddings found for tasks: {missing_tasks}")

        target_path = args.work_dir / "targets.parquet"
        targets, targets_manifest = download_targets(client, args.targets_uri, target_path)
        target_path.unlink(missing_ok=True)

        source_manifest = {
            "status": "running",
            "created_at_utc": utc_now(),
            "representation": args.representation,
            "input_bucket": args.input_bucket,
            "source_prefix": args.source_prefix,
            "source_files": len(objects),
            "source_bytes": int(sum(item["size"] for item in objects)),
            "expected_tasks": TASKS,
            "aggregation": "mean_over_native_10_second_epoch_embeddings",
            "targets": targets_manifest,
            "output_schema": {
                "metadata_columns": [
                    "participant_id", "task", "n_epochs", "source_embedding_key",
                    "source_embedding_etag", "source_embedding_bytes", "embedding_norm",
                    *COVARIATE_COLUMNS, *TARGET_COLUMNS,
                ],
                "embedding_columns": "embedding_0000 through embedding_NNNN",
            },
        }
        put_json(client, args.output_bucket, f"{representation_base}/representation_manifest.json", source_manifest)

        completed_tasks: list[str] = []
        task_summaries: list[dict[str, Any]] = []
        total_rows = 0
        for task in TASKS:
            done_key = f"{representation_base}/tasks/{task}/done.json"
            audit_key = f"{representation_base}/tasks/{task}/audit.json"
            dataset_key = f"{representation_base}/tasks/{task}/dataset.parquet"
            if object_exists(client, args.output_bucket, done_key) and object_exists(
                client, args.output_bucket, dataset_key
            ):
                audit = read_json(client, args.output_bucket, audit_key)
                completed_tasks.append(task)
                task_summaries.append(audit)
                total_rows += int(audit["rows"])
                continue

            task_started = time.time()
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                rows = list(
                    executor.map(
                        lambda item: load_subject_mean(
                            client, args.input_bucket, item, args.source_prefix
                        ),
                        by_task[task],
                    )
                )
            dataset, vectors = dataframe_from_rows(rows, targets)
            audit = task_audit(dataset, vectors, args.representation, task)
            audit["source_files"] = len(by_task[task])
            audit["source_bytes"] = int(sum(item["size"] for item in by_task[task]))
            audit["dataset_key"] = dataset_key
            audit["elapsed_seconds"] = time.time() - task_started
            upload_parquet(client, dataset, args.output_bucket, dataset_key, args.work_dir)
            put_json(client, args.output_bucket, audit_key, audit)
            put_json(
                client,
                args.output_bucket,
                done_key,
                {
                    "status": "succeeded",
                    "representation": args.representation,
                    "task": task,
                    "dataset_key": dataset_key,
                    "audit_key": audit_key,
                    "rows": int(len(dataset)),
                    "created_at_utc": utc_now(),
                },
            )
            completed_tasks.append(task)
            task_summaries.append(audit)
            total_rows += int(len(dataset))
            progress = {
                "status": "running",
                "representation": args.representation,
                "completed_tasks": completed_tasks,
                "tasks_completed": len(completed_tasks),
                "tasks_total": len(TASKS),
                "rows_consolidated": total_rows,
                "errors": 0,
                "elapsed_seconds": time.time() - started,
            }
            put_json(client, args.output_bucket, f"{runtime_base}/progress.json", progress)
            print(json.dumps(progress), flush=True)

        dimensions = sorted({int(item["embedding_dimension"]) for item in task_summaries})
        if len(dimensions) != 1:
            raise RuntimeError(f"Inconsistent task dimensions: {dimensions}")
        summary = {
            "status": "succeeded",
            "created_at_utc": utc_now(),
            "representation": args.representation,
            "source_files": len(objects),
            "tasks_completed": len(completed_tasks),
            "tasks": completed_tasks,
            "rows_consolidated": total_rows,
            "embedding_dimension": dimensions[0],
            "errors": [],
            "task_counts": {item["task"]: item["rows"] for item in task_summaries},
            "target_availability": {
                target: int(sum(item["available_targets"][target] for item in task_summaries))
                for target in TARGET_COLUMNS
            },
            "elapsed_seconds": time.time() - started,
        }
        source_manifest["status"] = "succeeded"
        source_manifest["completed_at_utc"] = utc_now()
        source_manifest["embedding_dimension"] = dimensions[0]
        source_manifest["task_counts"] = summary["task_counts"]
        put_json(client, args.output_bucket, f"{representation_base}/representation_manifest.json", source_manifest)
        put_json(client, args.output_bucket, f"{runtime_base}/summary.json", summary)
        print(json.dumps(summary), flush=True)
        return 0
    except (ClientError, OSError, RuntimeError, ValueError, KeyError) as exc:
        error = {
            "status": "failed",
            "created_at_utc": utc_now(),
            "representation": args.representation,
            "type": type(exc).__name__,
            "message": str(exc),
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
