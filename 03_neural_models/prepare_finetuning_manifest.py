#!/usr/bin/env python3
"""Create deterministic paired-task cohorts and outer folds for E3 v3."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from sklearn.model_selection import KFold


BYTES_PER_EPOCH = 128 * 2000 * 4
NPY_HEADER_BYTES = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--targets-uri", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--manifest-prefix", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--active-task", required=True)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI, got {uri}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def s3_client(region: str):
    session = boto3.Session(region_name=region)
    return session.client(
        "s3",
        config=Config(
            retries={"total_max_attempts": 10, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=300,
            max_pool_connections=20,
            s3={"us_east_1_regional_endpoint": "regional"},
            user_agent_appid="eeg-thesis-e3-v3-manifest/1.0",
        ),
    )


def estimated_epoch_count(size: int) -> int:
    payload = int(size) - NPY_HEADER_BYTES
    if payload <= 0 or payload % BYTES_PER_EPOCH:
        raise ValueError(f"Unexpected uncompressed EEG NPY object size: {size}")
    return payload // BYTES_PER_EPOCH


def list_task_inventory(
    client, bucket: str, prefix: str, task: str, work_dir: Path,
) -> pd.DataFrame:
    cache = work_dir / f"inventory-{task}.csv"
    if cache.is_file():
        frame = pd.read_csv(cache)
        frame["participant_id"] = frame["participant_id"].astype(str)
        return frame

    rows: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix.strip('/')}/{task}/")
    for page in pages:
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if not key.endswith(".npy"):
                continue
            rows.append({
                "participant_id": Path(key).stem,
                "source_key": key,
                "source_size": int(item["Size"]),
                "epoch_count": estimated_epoch_count(int(item["Size"])),
            })
    frame = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    if frame.empty or frame.participant_id.duplicated().any():
        raise RuntimeError(f"Invalid or empty signal inventory for {task}")
    frame.to_csv(cache, index=False, lineterminator="\n")
    return frame


def load_targets(client, uri: str, target: str, work_dir: Path) -> tuple[pd.DataFrame, str]:
    bucket, key = parse_s3_uri(uri)
    metadata = client.head_object(Bucket=bucket, Key=key)
    cache = work_dir / "df_ml.parquet"
    if not cache.is_file() or cache.stat().st_size != int(metadata["ContentLength"]):
        client.download_file(bucket, key, str(cache))
    frame = pd.read_parquet(cache, columns=["participant_id", target])
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target]).drop_duplicates("participant_id")
    return frame, str(metadata.get("ETag", "")).strip('"')


def active_task_for_target(target: str) -> str:
    return {
        "attention": "seqLearning8target",
        "p_factor": "contrastChangeDetection_3",
        "internalizing": "contrastChangeDetection_2",
        "externalizing": "contrastChangeDetection_2",
    }[target]


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def existing_manifest(client, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey:
        return None
    with response["Body"] as body:
        return json.loads(body.read())


def put_object(client, bucket: str, key: str, body: bytes, content_type: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        ServerSideEncryption="AES256",
    )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.active_task != active_task_for_target(args.target):
        raise ValueError(f"Unexpected active task for {args.target}: {args.active_task}")
    if args.minimum_epochs < 8 or args.outer_folds < 5:
        raise ValueError("Full manifests require at least eight epochs and five folds")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    client = s3_client(args.region)
    base = f"{args.manifest_prefix.strip('/')}/{args.target}"
    manifest_key = f"{base}/manifest.json"
    previous = existing_manifest(client, args.output_bucket, manifest_key)
    expected = {
        "target": args.target,
        "active_task": args.active_task,
        "minimum_epochs": args.minimum_epochs,
        "outer_folds": args.outer_folds,
        "seed": args.seed,
    }
    if previous is not None:
        if any(previous.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Existing manifest configuration mismatch at {manifest_key}")
        client.head_object(Bucket=args.output_bucket, Key=f"{base}/cohort.csv")
        client.head_object(Bucket=args.output_bucket, Key=f"{base}/split_manifest.csv")
        print(json.dumps(previous, sort_keys=True))
        return previous

    targets, targets_etag = load_targets(
        client, args.targets_uri, args.target, args.work_dir,
    )
    resting = list_task_inventory(
        client, args.input_bucket, args.input_prefix, "RestingState", args.work_dir,
    )
    active = list_task_inventory(
        client, args.input_bucket, args.input_prefix, args.active_task, args.work_dir,
    )
    target_ids = set(targets.participant_id)
    resting_ids = set(
        resting.loc[resting.epoch_count >= args.minimum_epochs, "participant_id"]
    )
    active_ids = set(
        active.loc[active.epoch_count >= args.minimum_epochs, "participant_id"]
    )
    participant_ids = sorted(target_ids & resting_ids & active_ids)
    if len(participant_ids) < args.outer_folds * 8:
        raise RuntimeError(
            f"Only {len(participant_ids)} paired eligible participants for {args.target}"
        )

    cohort = pd.DataFrame({"participant_id": participant_ids})
    split_rows: list[dict[str, Any]] = []
    splitter = KFold(
        n_splits=args.outer_folds, shuffle=True, random_state=args.seed,
    )
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(participant_ids), start=1,
    ):
        for split, indices in (
            ("outer_train", train_indices), ("outer_test", test_indices),
        ):
            split_rows.extend({
                "fold": fold,
                "split": split,
                "participant_id": participant_ids[int(index)],
            } for index in indices)
    splits = pd.DataFrame(split_rows)
    cohort_body = csv_bytes(cohort)
    split_body = csv_bytes(splits)
    manifest = {
        **expected,
        "resting_task": "RestingState",
        "participant_count": len(participant_ids),
        "targets_etag": targets_etag,
        "target_available_count": len(target_ids),
        "resting_eligible_count": len(resting_ids),
        "active_eligible_count": len(active_ids),
        "cohort_sha256": hashlib.sha256(cohort_body).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split_body).hexdigest(),
        "statistical_unit": "participant",
        "paired_task_comparison": True,
    }
    put_object(
        client, args.output_bucket, f"{base}/cohort.csv", cohort_body, "text/csv",
    )
    put_object(
        client, args.output_bucket, f"{base}/split_manifest.csv", split_body, "text/csv",
    )
    put_object(
        client, args.output_bucket, manifest_key,
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> int:
    args = parse_args()
    try:
        build_manifest(args)
        return 0
    except (ClientError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Manifest preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

