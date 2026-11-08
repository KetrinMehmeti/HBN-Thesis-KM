#!/usr/bin/env python3
"""Checkpointable full extraction of LaBraM embeddings from all 128 EGI channels."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botocore.exceptions import ClientError

from labram_128_adapter import (
    LABRAM_SHA256,
    MODEL_NAME,
    adapt_positions_to_egi,
    anti_collapse,
    extract_labram,
    load_labram,
    put_json,
    s3_client,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", default="ica_qamad_ready")
    parser.add_argument("--channel-info-key", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--labram-weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/e1b-labram128/work"))
    return parser.parse_args()


def list_inputs(client, bucket: str, prefix: str, tasks: list[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    paginator = client.get_paginator("list_objects_v2")
    for task in tasks:
        task_prefix = f"{prefix.strip('/')}/{task}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=task_prefix):
            for item in page.get("Contents", []):
                if item["Key"].endswith(".npy"):
                    items.append(
                        {"task": task, "key": item["Key"], "etag": item["ETag"].strip('"')}
                    )
    return sorted(items, key=lambda item: (item["task"], item["key"]))


def list_completed(client, bucket: str, base: str, tasks: list[str]) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    paginator = client.get_paginator("list_objects_v2")
    model_root = f"{base}/models/{MODEL_NAME}/done"
    for task in tasks:
        prefix = f"{model_root}/{task}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                if item["Key"].endswith(".json"):
                    completed.add((task, Path(item["Key"]).stem))
    return completed


def load_channel_info(client, bucket: str, key: str) -> list[dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        path = Path(temp.name)
    try:
        client.download_file(bucket, key, str(path))
        with path.open("rb") as stream:
            return pickle.load(stream)
    finally:
        path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one task is required")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if sha256(args.labram_weights) != LABRAM_SHA256:
        raise RuntimeError("LaBraM checkpoint SHA-256 mismatch")

    client = s3_client(args.region)
    base = args.output_prefix.strip("/")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(32, os.cpu_count() or 1)))

    channel_info = load_channel_info(client, args.input_bucket, args.channel_info_key)
    vendor = Path(__file__).resolve().parent / "vendor"
    model = load_labram(vendor, args.labram_weights, device)
    input_chans, spatial_report = adapt_positions_to_egi(model, channel_info)
    model.eval().requires_grad_(False)

    config = {
        "phase": "E1B-LaBraM-128-spatial",
        "model_name": MODEL_NAME,
        "checkpoint_sha256": LABRAM_SHA256,
        "input_channels": 128,
        "sampling_rate_hz": 200,
        "native_epoch_seconds": 10,
        "input_scaling": "volts_to_microvolts_x100",
        "embedding_dtype": "float32",
        "spatial_adaptation": spatial_report,
    }
    put_json(client, args.output_bucket, f"{base}/models/{MODEL_NAME}/model_config.json", config)

    items = list_inputs(client, args.input_bucket, args.input_prefix, tasks)
    completed = list_completed(client, args.output_bucket, base, tasks)
    pending = [item for item in items if (item["task"], Path(item["key"]).stem) not in completed]
    run_started = time.time()
    errors: list[dict[str, str]] = []
    sampled_means: list[np.ndarray] = []
    completed_now = 0
    epochs_now = 0

    manifest = {
        "status": "running",
        "phase": "E1B-LaBraM-128-spatial",
        "shard_id": args.shard_id,
        "tasks": tasks,
        "input_files": len(items),
        "completed_before_run": len(items) - len(pending),
        "pending_at_start": len(pending),
        "device": str(device),
        "batch_size": args.batch_size,
    }
    put_json(client, args.output_bucket, f"{base}/runtime/{args.shard_id}/manifest.json", manifest)

    for index, item in enumerate(pending, start=1):
        subject = Path(item["key"]).stem
        local_input = args.work_dir / f"{subject}.npy"
        local_output = args.work_dir / f"{subject}.{MODEL_NAME}.npz"
        try:
            client.download_file(args.input_bucket, item["key"], str(local_input))
            data = np.load(local_input, mmap_mode="r", allow_pickle=False)
            if data.dtype != np.float32 or data.ndim != 3 or data.shape[1:] != (128, 2000):
                raise ValueError(f"Unexpected input dtype/shape: {data.dtype} {data.shape}")
            if not np.isfinite(data).all():
                raise ValueError("Input contains non-finite samples")

            started = time.time()
            embeddings = extract_labram(model, data, input_chans, device, args.batch_size)
            if not np.isfinite(embeddings).all() or np.count_nonzero(embeddings) != embeddings.size:
                raise RuntimeError("LaBraM-128 produced invalid embeddings")
            mean = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
            if len(sampled_means) < 64:
                sampled_means.append(mean)
            np.savez_compressed(
                local_output,
                embeddings=embeddings,
                subject_mean=mean,
                epoch_index=np.arange(len(data), dtype=np.int32),
            )

            model_root = f"{base}/models/{MODEL_NAME}"
            embedding_key = f"{model_root}/embeddings/{item['task']}/{subject}.npz"
            metadata_key = f"{model_root}/metadata/{item['task']}/{subject}.json"
            done_key = f"{model_root}/done/{item['task']}/{subject}.json"
            client.upload_file(
                str(local_output), args.output_bucket, embedding_key,
                ExtraArgs={"ContentType": "application/octet-stream", "ServerSideEncryption": "AES256"},
            )
            record = {
                "model": MODEL_NAME,
                "task": item["task"],
                "subject": subject,
                "source_key": item["key"],
                "source_etag": item["etag"],
                "epochs": int(len(data)),
                "input_channels": 128,
                "embedding_key": embedding_key,
                "embedding_shape": list(embeddings.shape),
                "elapsed_seconds": time.time() - started,
            }
            put_json(client, args.output_bucket, metadata_key, record)
            put_json(client, args.output_bucket, done_key, record)
            completed_now += 1
            epochs_now += int(len(data))
            print(json.dumps({"progress": f"{index}/{len(pending)}", **record}), flush=True)
        except (ClientError, OSError, RuntimeError, ValueError) as exc:
            error = {
                "task": item["task"], "subject": subject, "key": item["key"],
                "type": type(exc).__name__, "message": str(exc),
            }
            errors.append(error)
            print(json.dumps({"error": error}), file=sys.stderr, flush=True)
        finally:
            local_input.unlink(missing_ok=True)
            local_output.unlink(missing_ok=True)

        if index == 1 or index % 25 == 0 or index == len(pending):
            progress = {
                "status": "running",
                "shard_id": args.shard_id,
                "processed": index,
                "pending_at_start": len(pending),
                "completed_now": completed_now,
                "epochs_now": epochs_now,
                "errors": len(errors),
                "elapsed_seconds": time.time() - run_started,
            }
            put_json(client, args.output_bucket, f"{base}/runtime/{args.shard_id}/progress.json", progress)

    quality = anti_collapse(sampled_means) if len(sampled_means) >= 8 else {"subjects": len(sampled_means), "passed": None}
    summary = {
        "status": "succeeded" if not errors else "completed_with_errors",
        "phase": "E1B-LaBraM-128-spatial",
        "shard_id": args.shard_id,
        "tasks": tasks,
        "input_files": len(items),
        "completed_before_run": len(items) - len(pending),
        "completed_now": completed_now,
        "epochs_now": epochs_now,
        "errors": errors,
        "sample_quality": quality,
        "elapsed_seconds": time.time() - run_started,
    }
    put_json(client, args.output_bucket, f"{base}/runtime/{args.shard_id}/summary.json", summary)
    if errors:
        raise RuntimeError(f"Shard completed with {len(errors)} subject errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
