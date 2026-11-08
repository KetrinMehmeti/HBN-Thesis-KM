#!/usr/bin/env python3
"""Extract frozen CBraMod and BIOT embeddings from native 10-second HBN epochs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import boto3
import mne
import numpy as np
import torch
from botocore.config import Config
from botocore.exceptions import ClientError


BIOT_PAIRS = [
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", default="ica_qamad_ready")
    parser.add_argument("--channel-info-key", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated task names")
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--cbramod-weights", type=Path, required=True)
    parser.add_argument("--biot-weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--anti-collapse-check", action="store_true")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/e1a/work"))
    return parser.parse_args()


def make_s3_client(region: str):
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"total_max_attempts": 8, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=180,
            max_pool_connections=20,
            s3={"us_east_1_regional_endpoint": "regional"},
            user_agent_appid="eeg-thesis-e1a/1.0",
        ),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_position(position: np.ndarray) -> np.ndarray:
    vector = np.asarray(position, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid electrode position")
    return vector / norm


def build_biot_montage(channel_info: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    egi = {
        item["ch_name"]: normalized_position(np.asarray(item["loc"][:3], dtype=float))
        for item in channel_info
    }
    standard = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    targets = sorted({name for pair in BIOT_PAIRS for name in pair})
    names = list(egi)
    mapping: dict[str, int] = {}
    report: list[dict[str, Any]] = []
    for target in targets:
        target_position = normalized_position(np.asarray(standard[target], dtype=float))
        similarities = [float(np.dot(target_position, egi[name])) for name in names]
        best = int(np.argmax(similarities))
        egi_name = names[best]
        mapping[target] = int(egi_name.removeprefix("E")) - 1
        report.append(
            {
                "standard_electrode": target,
                "egi_electrode": egi_name,
                "angular_distance_deg": float(
                    np.degrees(np.arccos(np.clip(similarities[best], -1.0, 1.0)))
                ),
            }
        )
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("Nearest-electrode mapping contains duplicate EGI electrodes")
    return mapping, report


def precision_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_models(
    vendor: Path,
    cbramod_weights: Path,
    biot_weights: Path,
    device: torch.device,
):
    sys.path.insert(0, str(vendor / "CBraMod"))
    sys.path.insert(0, str(vendor / "BIOT"))
    from models.cbramod import CBraMod
    from model.biot import BIOTEncoder

    cbramod = CBraMod().to(device)
    cbramod.load_state_dict(
        torch.load(cbramod_weights, map_location="cpu", weights_only=True), strict=True
    )
    cbramod.eval().requires_grad_(False)

    biot = BIOTEncoder(
        emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100
    ).to(device)
    biot.load_state_dict(
        torch.load(biot_weights, map_location="cpu", weights_only=True), strict=True
    )
    biot.eval().requires_grad_(False)
    return cbramod, biot


def list_inputs(client, bucket: str, input_prefix: str, tasks: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for task in tasks:
        prefix = f"{input_prefix.strip('/')}/{task}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                if item["Key"].endswith(".npy"):
                    results.append(
                        {
                            "task": task,
                            "key": item["Key"],
                            "size": int(item["Size"]),
                            "etag": item["ETag"].strip('"'),
                        }
                    )
    return sorted(results, key=lambda item: item["key"])


def list_done(client, bucket: str, output_prefix: str, tasks: list[str]) -> set[str]:
    completed: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for task in tasks:
        prefix = f"{output_prefix.strip('/')}/done/{task}/"
        for key in paginator.paginate(Bucket=bucket, Prefix=prefix).search("Contents[].Key"):
            if key:
                completed.add(f"{task}/{Path(key).stem}")
    return completed


def upload_json(client, payload: dict[str, Any], bucket: str, key: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, allow_nan=False).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def extract_embeddings(
    data: np.ndarray,
    batch_size: int,
    mapping: dict[str, int],
    cbramod,
    biot,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray([mapping[a] for a, _ in BIOT_PAIRS], dtype=int)
    right = np.asarray([mapping[b] for _, b in BIOT_PAIRS], dtype=int)
    cbramod_parts: list[np.ndarray] = []
    biot_parts: list[np.ndarray] = []

    for start in range(0, data.shape[0], batch_size):
        stored = np.asarray(data[start : start + batch_size], dtype=np.float32)
        if not np.isfinite(stored).all():
            raise ValueError(f"Non-finite samples in epoch batch beginning at {start}")
        cb_input = torch.from_numpy((stored * 100.0).reshape(-1, 128, 10, 200)).to(device)
        bipolar = (stored[:, left, :] - stored[:, right, :]).astype(np.float32, copy=False)
        scale = np.quantile(np.abs(bipolar), 0.95, axis=-1, keepdims=True)
        bipolar = (bipolar / (scale + 1e-8)).astype(np.float32, copy=False)
        biot_input = torch.from_numpy(bipolar).to(device)
        with torch.inference_mode(), precision_context(device):
            cb = cbramod(cb_input).float().mean(dim=(1, 2))
            bt = biot(biot_input).float()
        cbramod_parts.append(cb.cpu().numpy())
        biot_parts.append(bt.cpu().numpy())
    return (
        np.concatenate(cbramod_parts).astype(np.float32, copy=False),
        np.concatenate(biot_parts).astype(np.float32, copy=False),
    )


def process_one(
    client,
    item: dict[str, Any],
    args: argparse.Namespace,
    mapping: dict[str, int],
    cbramod,
    biot,
    device: torch.device,
) -> dict[str, Any]:
    task = item["task"]
    subject = Path(item["key"]).stem
    local_input = args.work_dir / f"{subject}.npy"
    local_cbramod = args.work_dir / f"{subject}.cbramod.npz"
    local_biot = args.work_dir / f"{subject}.biot.npz"
    started = time.perf_counter()
    try:
        client.download_file(args.input_bucket, item["key"], str(local_input))
        data = np.load(local_input, mmap_mode="r", allow_pickle=False)
        if data.dtype != np.float32 or data.ndim != 3 or data.shape[1:] != (128, 2000):
            raise ValueError(f"Expected float32 [epochs,128,2000], got {data.dtype} {data.shape}")
        if data.shape[0] == 0:
            raise ValueError("Input contains zero epochs")
        cb, bt = extract_embeddings(
            data, args.batch_size, mapping, cbramod, biot, device
        )
        if not np.isfinite(cb).all() or not np.isfinite(bt).all():
            raise RuntimeError("Encoder produced non-finite embeddings")
        cb_mean = cb.mean(axis=0, dtype=np.float64).astype(np.float32)
        bt_mean = bt.mean(axis=0, dtype=np.float64).astype(np.float32)
        epoch_index = np.arange(data.shape[0], dtype=np.int32)
        np.savez_compressed(local_cbramod, embeddings=cb, subject_mean=cb_mean, epoch_index=epoch_index)
        np.savez_compressed(local_biot, embeddings=bt, subject_mean=bt_mean, epoch_index=epoch_index)
        base = args.output_prefix.strip("/")
        cbramod_key = f"{base}/models/cbramod/embeddings/{task}/{subject}.npz"
        biot_key = f"{base}/models/biot/embeddings/{task}/{subject}.npz"
        for local_path, embedding_key in ((local_cbramod, cbramod_key), (local_biot, biot_key)):
            client.upload_file(
                str(local_path), args.output_bucket, embedding_key,
                ExtraArgs={"ContentType": "application/octet-stream", "ServerSideEncryption": "AES256"},
            )
        metadata = {
            "status": "succeeded",
            "task": task,
            "subject": subject,
            "source_bucket": args.input_bucket,
            "source_key": item["key"],
            "source_size": item["size"],
            "source_etag": item["etag"],
            "epochs": int(data.shape[0]),
            "native_epoch_seconds": 10,
            "sampling_rate_hz": 200,
            "cbramod_shape": list(cb.shape),
            "biot_shape": list(bt.shape),
            "embedding_keys": {"cbramod": cbramod_key, "biot": biot_key},
            "preprocessing": {"cbramod": "stored_x100_1s_patches", "biot": "16_bipolar_then_epoch_channel_abs_q95"},
            "subject_means": {"cbramod": cb_mean.tolist(), "biot": bt_mean.tolist()},
            "elapsed_seconds": time.perf_counter() - started,
        }
        upload_json(
            client,
            metadata,
            args.output_bucket,
            f"{base}/metadata/{task}/{subject}.json",
        )
        upload_json(
            client,
            metadata,
            args.output_bucket,
            f"{base}/done/{task}/{subject}.json",
        )
        return metadata
    finally:
        local_input.unlink(missing_ok=True)
        local_cbramod.unlink(missing_ok=True)
        local_biot.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one task is required")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    client = make_s3_client(args.region)
    base = args.output_prefix.strip("/")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        channel_path = Path(temp.name)
    try:
        client.download_file(args.input_bucket, args.channel_info_key, str(channel_path))
        with channel_path.open("rb") as stream:
            channel_info = pickle.load(stream)
    finally:
        channel_path.unlink(missing_ok=True)
    if len(channel_info) != 128:
        raise ValueError(f"Expected 128 channel metadata records, got {len(channel_info)}")
    mapping, mapping_report = build_biot_montage(channel_info)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(32, os.cpu_count() or 1)))
    vendor = Path(__file__).resolve().parent / "vendor"
    cbramod, biot = load_models(
        vendor, args.cbramod_weights, args.biot_weights, device
    )

    items = list_inputs(client, args.input_bucket, args.input_prefix, tasks)
    if args.max_files > 0:
        by_task = {task: [item for item in items if item["task"] == task] for task in tasks}
        items = []
        while len(items) < args.max_files and any(by_task.values()):
            for task in tasks:
                if by_task[task] and len(items) < args.max_files:
                    items.append(by_task[task].pop(0))
    completed_subjects = list_done(client, args.output_bucket, base, tasks)
    pending = [
        item
        for item in items
        if f"{item['task']}/{Path(item['key']).stem}" not in completed_subjects
    ]
    run_started = time.time()
    manifest = {
        "phase": "E1A",
        "shard_id": args.shard_id,
        "tasks": tasks,
        "input_files": len(items),
        "already_completed": len(items) - len(pending),
        "pending_at_start": len(pending),
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoint_sha256": {
            "cbramod": sha256(args.cbramod_weights),
            "biot": sha256(args.biot_weights),
        },
        "biot_electrode_mapping": mapping_report,
    }
    upload_json(client, manifest, args.output_bucket, f"{base}/runtime/{args.shard_id}/manifest.json")

    completed_now = 0
    errors: list[dict[str, Any]] = []
    subject_means: dict[str, list[np.ndarray]] = {"cbramod": [], "biot": []}
    for index, item in enumerate(pending, start=1):
        try:
            result = process_one(client, item, args, mapping, cbramod, biot, device)
            completed_now += 1
            for model in subject_means:
                subject_means[model].append(np.asarray(result["subject_means"][model], dtype=np.float32))
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(pending)}",
                        "task": result["task"],
                        "subject": result["subject"],
                        "epochs": result["epochs"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        except (ClientError, OSError, RuntimeError, ValueError) as exc:
            error = {
                "task": item["task"],
                "key": item["key"],
                "type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error)
            print(json.dumps({"error": error}), file=sys.stderr, flush=True)
        if index == 1 or index % 25 == 0 or index == len(pending):
            progress = {
                "status": "running",
                "shard_id": args.shard_id,
                "processed": index,
                "pending_at_start": len(pending),
                "completed_now": completed_now,
                "errors": len(errors),
                "elapsed_seconds": time.time() - run_started,
            }
            upload_json(client, progress, args.output_bucket, f"{base}/runtime/{args.shard_id}/progress.json")

    anti_collapse: dict[str, Any] = {}
    for model, vectors in subject_means.items():
        matrix = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
        normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12) if len(matrix) else matrix
        similarities = normalized @ normalized.T if len(matrix) else np.empty((0, 0))
        upper = similarities[np.triu_indices(len(matrix), 1)] if len(matrix) > 1 else np.asarray([])
        variance = float(np.var(matrix, axis=0, dtype=np.float64).mean()) if len(matrix) else 0.0
        max_cosine = float(np.max(upper)) if upper.size else 1.0
        passed = len(matrix) >= 4 and variance > 1e-8 and max_cosine < 0.99999
        anti_collapse[model] = {
            "subjects": len(matrix), "mean_dimension_variance": variance,
            "pairwise_cosine_min": float(np.min(upper)) if upper.size else None,
            "pairwise_cosine_mean": float(np.mean(upper)) if upper.size else None,
            "pairwise_cosine_max": max_cosine if upper.size else None, "passed": passed,
        }
    if args.anti_collapse_check and not all(report["passed"] for report in anti_collapse.values()):
        errors.append({"type": "EmbeddingCollapseError", "message": "Anti-collapse thresholds failed", "report": anti_collapse})

    summary = {
        "status": "succeeded" if not errors else "completed_with_errors",
        "shard_id": args.shard_id,
        "tasks": tasks,
        "input_files": len(items),
        "completed_before_run": len(items) - len(pending),
        "completed_now": completed_now,
        "errors": errors,
        "anti_collapse": anti_collapse,
        "elapsed_seconds": time.time() - run_started,
    }
    upload_json(client, summary, args.output_bucket, f"{base}/runtime/{args.shard_id}/summary.json")
    if errors:
        raise RuntimeError(f"Shard completed with {len(errors)} subject errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
