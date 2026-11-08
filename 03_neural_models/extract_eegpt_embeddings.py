#!/usr/bin/env python3
"""Checkpointable extraction of canonical-62 and spatial-128 EEGPT embeddings."""

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

from eegpt_spatial_adapter import (
    MODEL_62,
    MODEL_128,
    adapt_eegpt_to_egi128,
    compare_representations,
)
from foundation_model_common import (
    EEGPT_SHA256,
    anti_collapse,
    build_montage,
    extract_eegpt,
    load_eegpt,
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
    parser.add_argument("--eegpt-weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/e1c-eegpt-dual/work"))
    return parser.parse_args()


def list_inputs(client, bucket: str, prefix: str, tasks: list[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    paginator = client.get_paginator("list_objects_v2")
    for task in tasks:
        pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix.strip('/')}/{task}/")
        for page in pages:
            for item in page.get("Contents", []):
                if item["Key"].endswith(".npy"):
                    items.append(
                        {"task": task, "key": item["Key"], "etag": item["ETag"].strip('"')}
                    )
    return sorted(items, key=lambda item: (item["task"], item["key"]))


def list_done(client, bucket: str, base: str, model_name: str, tasks: list[str]) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    paginator = client.get_paginator("list_objects_v2")
    for task in tasks:
        prefix = f"{base}/models/{model_name}/done/{task}/"
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
    if sha256(args.eegpt_weights) != EEGPT_SHA256:
        raise RuntimeError("EEGPT checkpoint SHA-256 mismatch")

    client = s3_client(args.region)
    base = args.output_prefix.strip("/")
    channel_info = load_channel_info(client, args.input_bucket, args.channel_info_key)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(32, os.cpu_count() or 1)))

    vendor = Path(__file__).resolve().parent / "vendor"
    model_62, channel_ids_62 = load_eegpt(vendor, args.eegpt_weights, device)
    indices_62, mapping_62 = build_montage(channel_info)
    model_128, channel_ids_128, mapping_128 = adapt_eegpt_to_egi128(
        model_62, channel_info, device
    )
    indices_128 = np.arange(128, dtype=np.int64)
    model_config = {
        MODEL_62: {
            "phase": "E1C-EEGPT-dual",
            "model": MODEL_62,
            "checkpoint_sha256": EEGPT_SHA256,
            "input_channels": 62,
            "channel_strategy": "unique minimum-cost mapping from 62 canonical positions to EGI-128",
            "temporal_protocol": "10-second epoch; 0-4, 3-7, 6-10 second windows; 200-to-256 Hz; mean pooling",
            "mapping": mapping_62,
        },
        MODEL_128: {
            "phase": "E1C-EEGPT-dual",
            "model": MODEL_128,
            "checkpoint_sha256": EEGPT_SHA256,
            "input_channels": 128,
            "channel_strategy": "spatial interpolation of the learned 62-position EEGPT channel table",
            "temporal_protocol": "10-second epoch; 0-4, 3-7, 6-10 second windows; 200-to-256 Hz; mean pooling",
            "mapping": mapping_128,
        },
    }
    for name, config in model_config.items():
        put_json(client, args.output_bucket, f"{base}/models/{name}/model_config.json", config)

    items = list_inputs(client, args.input_bucket, args.input_prefix, tasks)
    completed_62 = list_done(client, args.output_bucket, base, MODEL_62, tasks)
    completed_128 = list_done(client, args.output_bucket, base, MODEL_128, tasks)
    completed_both = completed_62 & completed_128
    pending = [
        item for item in items
        if (item["task"], Path(item["key"]).stem) not in completed_both
    ]
    run_started = time.time()
    errors: list[dict[str, str]] = []
    sample_means: dict[str, list[np.ndarray]] = {MODEL_62: [], MODEL_128: []}
    completed_now = 0
    epochs_now = 0
    manifest = {
        "status": "running", "phase": "E1C-EEGPT-dual", "shard_id": args.shard_id,
        "tasks": tasks, "input_files": len(items),
        "completed_before_run": len(items) - len(pending), "pending_at_start": len(pending),
        "device": str(device), "batch_size": args.batch_size,
        "models": [MODEL_62, MODEL_128],
    }
    put_json(client, args.output_bucket, f"{base}/runtime/{args.shard_id}/manifest.json", manifest)

    for index, item in enumerate(pending, start=1):
        subject = Path(item["key"]).stem
        local_input = args.work_dir / f"{subject}.npy"
        local_outputs = {
            MODEL_62: args.work_dir / f"{subject}.{MODEL_62}.npz",
            MODEL_128: args.work_dir / f"{subject}.{MODEL_128}.npz",
        }
        try:
            client.download_file(args.input_bucket, item["key"], str(local_input))
            data = np.load(local_input, mmap_mode="r", allow_pickle=False)
            if data.dtype != np.float32 or data.ndim != 3 or data.shape[1:] != (128, 2000):
                raise ValueError(f"Unexpected input dtype/shape: {data.dtype} {data.shape}")
            if not np.isfinite(data).all():
                raise ValueError("Input contains non-finite samples")
            started = time.time()
            outputs = {
                MODEL_62: extract_eegpt(
                    model_62, data, indices_62, channel_ids_62, device, args.batch_size
                ),
                MODEL_128: extract_eegpt(
                    model_128, data, indices_128, channel_ids_128, device, args.batch_size
                ),
            }
            for name, embeddings in outputs.items():
                if embeddings.shape != (len(data), 512):
                    raise RuntimeError(f"Unexpected {name} shape {embeddings.shape}")
                if not np.isfinite(embeddings).all() or np.count_nonzero(embeddings) != embeddings.size:
                    raise RuntimeError(f"{name} produced invalid embeddings")
                mean = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
                if len(sample_means[name]) < 64:
                    sample_means[name].append(mean)
                np.savez_compressed(
                    local_outputs[name], embeddings=embeddings, subject_mean=mean,
                    epoch_index=np.arange(len(data), dtype=np.int32),
                )
                model_root = f"{base}/models/{name}"
                embedding_key = f"{model_root}/embeddings/{item['task']}/{subject}.npz"
                client.upload_file(
                    str(local_outputs[name]), args.output_bucket, embedding_key,
                    ExtraArgs={"ContentType": "application/octet-stream", "ServerSideEncryption": "AES256"},
                )
                metadata = {
                    "model": name, "task": item["task"], "subject": subject,
                    "source_key": item["key"], "source_etag": item["etag"],
                    "epochs": int(len(data)), "embedding_shape": list(embeddings.shape),
                    "embedding_key": embedding_key,
                }
                put_json(client, args.output_bucket, f"{model_root}/metadata/{item['task']}/{subject}.json", metadata)
                put_json(client, args.output_bucket, f"{model_root}/done/{item['task']}/{subject}.json", metadata)
            completed_now += 1
            epochs_now += int(len(data))
            print(json.dumps({
                "progress": f"{index}/{len(pending)}", "task": item["task"],
                "subject": subject, "epochs": int(len(data)),
                "elapsed_seconds": time.time() - started,
            }), flush=True)
        except (ClientError, OSError, RuntimeError, ValueError) as exc:
            error = {
                "task": item["task"], "subject": subject, "key": item["key"],
                "type": type(exc).__name__, "message": str(exc),
            }
            errors.append(error)
            print(json.dumps({"error": error}), file=sys.stderr, flush=True)
        finally:
            local_input.unlink(missing_ok=True)
            for path in local_outputs.values():
                path.unlink(missing_ok=True)

        if index == 1 or index % 25 == 0 or index == len(pending):
            progress = {
                "status": "running", "shard_id": args.shard_id,
                "processed": index, "pending_at_start": len(pending),
                "completed_now": completed_now, "epochs_now": epochs_now,
                "errors": len(errors), "elapsed_seconds": time.time() - run_started,
            }
            put_json(client, args.output_bucket, f"{base}/runtime/{args.shard_id}/progress.json", progress)

    quality = {
        name: anti_collapse(vectors) if len(vectors) >= 8 else {"subjects": len(vectors), "passed": None}
        for name, vectors in sample_means.items()
    }
    comparison = (
        compare_representations(sample_means[MODEL_62], sample_means[MODEL_128])
        if len(sample_means[MODEL_62]) >= 2 else {"subjects": len(sample_means[MODEL_62])}
    )
    summary = {
        "status": "succeeded" if not errors else "completed_with_errors",
        "phase": "E1C-EEGPT-dual", "shard_id": args.shard_id, "tasks": tasks,
        "input_files": len(items), "completed_before_run": len(items) - len(pending),
        "completed_now": completed_now, "epochs_now": epochs_now,
        "errors": errors, "sample_quality": quality,
        "cross_representation_comparison": comparison,
        "elapsed_seconds": time.time() - run_started,
    }
    put_json(client, args.output_bucket, f"{base}/runtime/{args.shard_id}/summary.json", summary)
    if errors:
        raise RuntimeError(f"Shard completed with {len(errors)} subject errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
