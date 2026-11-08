#!/usr/bin/env python3
"""Smoke test for LaBraM embeddings adapted to all 128 HBN EGI channels."""

from __future__ import annotations

import argparse
from collections import OrderedDict
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


LABRAM_SHA256 = "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"
MODEL_NAME = "labram_128_spatial"

# Rows 1..128 of LaBraM's learned positional table correspond to the first
# 128 entries of the official standard_1020 vocabulary. The final eight are
# bipolar labels and have no point coordinate, so they are excluded as
# interpolation sources while their table capacity is reused for EGI targets.
LABRAM_POSITION_ORDER = [
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2", "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9H", "TTP7H", "TPP9H", "FTT10H", "TPP8H", "TPP10H",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", default="ica_qamad_ready")
    parser.add_argument("--channel-info-key", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--labram-weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/e0-labram128/work"))
    return parser.parse_args()


def s3_client(region: str):
    return boto3.client(
        "s3", region_name=region,
        config=Config(
            retries={"total_max_attempts": 8, "mode": "adaptive"},
            connect_timeout=10, read_timeout=180, max_pool_connections=20,
            s3={"us_east_1_regional_endpoint": "regional"},
            user_agent_appid="eeg-thesis-e0-labram128/1.0",
        ),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit(position: Any) -> np.ndarray:
    vector = np.asarray(position, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid channel position")
    return vector / norm


def load_labram(vendor: Path, weights: Path, device: torch.device):
    sys.path.insert(0, str(vendor / "LaBraM"))
    import modeling_finetune

    model = modeling_finetune.labram_base_patch200_200(
        EEG_size=2000, num_classes=0, use_mean_pooling=True,
        use_abs_pos_emb=True, use_rel_pos_bias=False, init_values=0.1, qkv_bias=False,
    )
    safe_types = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType, argparse.Namespace,
    ]
    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint.get("module", checkpoint))
    filtered = OrderedDict((key[8:], value) for key, value in state.items() if key.startswith("student."))
    if not filtered:
        filtered = OrderedDict(state)
    for suffix in ("weight", "bias"):
        source = f"norm.{suffix}"
        if source in filtered:
            filtered[f"fc_norm.{suffix}"] = filtered.pop(source)
    for key in list(filtered):
        if (
            "relative_position_index" in key
            or key == "mask_token"
            or key.startswith("lm_head.")
            or key.startswith("head.")
        ):
            filtered.pop(key)
    incompatible = model.load_state_dict(filtered, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"LaBraM checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if tuple(model.pos_embed.shape) != (1, 129, 200):
        raise RuntimeError(f"Unexpected LaBraM positional table {tuple(model.pos_embed.shape)}")
    return model.to(device)


def adapt_positions_to_egi(model, channel_info: list[dict[str, Any]], neighbours: int = 4):
    if len(channel_info) != 128:
        raise ValueError(f"Expected 128 EGI channels, got {len(channel_info)}")
    egi_names = [item["ch_name"] for item in channel_info]
    if len(set(egi_names)) != 128:
        raise ValueError("EGI channel names are not unique")
    egi_positions = np.stack([unit(item["loc"][:3]) for item in channel_info])

    standard_raw = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    standard = {name.upper(): unit(position) for name, position in standard_raw.items()}
    source_rows, source_names, source_positions = [], [], []
    for row, name in enumerate(LABRAM_POSITION_ORDER, start=1):
        if name in standard:
            source_rows.append(row)
            source_names.append(name)
            source_positions.append(standard[name])
    if len(source_rows) < 80:
        raise RuntimeError(f"Only {len(source_rows)} LaBraM positions have valid coordinates")

    source_positions_array = np.stack(source_positions)
    angles = np.arccos(np.clip(egi_positions @ source_positions_array.T, -1.0, 1.0))
    nearest = np.argsort(angles, axis=1)[:, :neighbours]
    nearest_angles = np.take_along_axis(angles, nearest, axis=1)
    weights = 1.0 / np.maximum(nearest_angles, 1e-4) ** 2
    exact = nearest_angles[:, 0] < 1e-6
    weights /= weights.sum(axis=1, keepdims=True)
    weights[exact] = 0.0
    weights[exact, 0] = 1.0

    source_embedding = model.pos_embed.detach().cpu().numpy()[0, source_rows, :]
    target_embedding = np.sum(source_embedding[nearest] * weights[:, :, None], axis=1).astype(np.float32)
    with torch.no_grad():
        model.pos_embed[:, 1:129, :].copy_(torch.from_numpy(target_embedding).to(model.pos_embed))

    mapping = []
    for egi_index, egi_name in enumerate(egi_names):
        contributors = [
            {
                "labram_position": source_names[int(index)],
                "labram_row": int(source_rows[int(index)]),
                "weight": float(weight),
                "angular_distance_deg": float(np.degrees(angle)),
            }
            for index, weight, angle in zip(nearest[egi_index], weights[egi_index], nearest_angles[egi_index])
        ]
        mapping.append({"egi_electrode": egi_name, "egi_index": egi_index, "contributors": contributors})
    nearest_degrees = np.degrees(nearest_angles[:, 0])
    summary = {
        "method": "inverse_squared_angular_distance_interpolation",
        "neighbours": neighbours,
        "source_positions": len(source_rows),
        "target_egi_positions": 128,
        "nearest_distance_deg": {
            "min": float(nearest_degrees.min()),
            "mean": float(nearest_degrees.mean()),
            "p95": float(np.quantile(nearest_degrees, 0.95)),
            "max": float(nearest_degrees.max()),
        },
        "mapping": mapping,
    }
    return list(range(129)), summary


def list_inputs(client, bucket: str, prefix: str, tasks: list[str], limit: int):
    by_task: dict[str, list[dict[str, str]]] = {}
    paginator = client.get_paginator("list_objects_v2")
    for task in tasks:
        entries = []
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix.strip('/')}/{task}/"):
            for item in page.get("Contents", []):
                if item["Key"].endswith(".npy"):
                    entries.append({"task": task, "key": item["Key"], "etag": item["ETag"].strip('"')})
        by_task[task] = sorted(entries, key=lambda item: item["key"])
    selected = []
    while len(selected) < limit and any(by_task.values()):
        for task in tasks:
            if by_task[task] and len(selected) < limit:
                selected.append(by_task[task].pop(0))
    return selected


def extract_labram(model, data, input_chans, device, batch_size):
    outputs = []
    for start in range(0, len(data), batch_size):
        stored = np.asarray(data[start:start + batch_size], dtype=np.float32)
        x = torch.from_numpy((stored * 100.0).reshape(-1, 128, 10, 200)).to(device)
        with torch.inference_mode():
            outputs.append(model(x, input_chans=input_chans).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def put_json(client, bucket: str, key: str, payload: dict[str, Any]):
    client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(payload, indent=2, allow_nan=False).encode(),
        ContentType="application/json", ServerSideEncryption="AES256",
    )


def anti_collapse(vectors: list[np.ndarray]):
    matrix = np.stack(vectors)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    upper = (normalized @ normalized.T)[np.triu_indices(len(matrix), 1)]
    report = {
        "subjects": len(matrix),
        "mean_dimension_variance": float(np.var(matrix, axis=0, dtype=np.float64).mean()),
        "pairwise_cosine_min": float(np.min(upper)),
        "pairwise_cosine_mean": float(np.mean(upper)),
        "pairwise_cosine_p95": float(np.quantile(upper, 0.95)),
        "pairwise_cosine_p999": float(np.quantile(upper, 0.999)),
        "near_identical_pair_fraction": float(np.mean(upper > 0.99999)),
    }
    report["passed"] = (
        len(matrix) >= 8 and report["mean_dimension_variance"] > 1e-8
        and report["pairwise_cosine_mean"] < 0.999 and report["pairwise_cosine_p95"] < 0.9999
    )
    return report


def main() -> int:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if sha256(args.labram_weights) != LABRAM_SHA256:
        raise RuntimeError("LaBraM checkpoint SHA-256 mismatch")
    client = s3_client(args.region)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        channel_path = Path(temp.name)
    try:
        client.download_file(args.input_bucket, args.channel_info_key, str(channel_path))
        with channel_path.open("rb") as stream:
            channel_info = pickle.load(stream)
    finally:
        channel_path.unlink(missing_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(32, os.cpu_count() or 1)))
    vendor = Path(__file__).resolve().parent / "vendor"
    model = load_labram(vendor, args.labram_weights, device)
    input_chans, spatial_report = adapt_positions_to_egi(model, channel_info)
    model.eval().requires_grad_(False)

    items = list_inputs(client, args.input_bucket, args.input_prefix, tasks, args.max_files)
    base = args.output_prefix.strip("/")
    means, records = [], []
    started = time.time()
    for number, item in enumerate(items, start=1):
        subject = Path(item["key"]).stem
        local_input = args.work_dir / f"{subject}.npy"
        local_output = args.work_dir / f"{subject}.{MODEL_NAME}.npz"
        try:
            client.download_file(args.input_bucket, item["key"], str(local_input))
            data = np.load(local_input, mmap_mode="r", allow_pickle=False)
            if data.dtype != np.float32 or data.ndim != 3 or data.shape[1:] != (128, 2000):
                raise ValueError(f"Unexpected input {data.dtype} {data.shape}")
            if not np.isfinite(data).all():
                raise ValueError("Input contains non-finite samples")
            embeddings = extract_labram(model, data, input_chans, device, args.batch_size)
            if not np.isfinite(embeddings).all() or np.count_nonzero(embeddings) != embeddings.size:
                raise RuntimeError("LaBraM-128 produced invalid embeddings")
            mean = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
            means.append(mean)
            np.savez_compressed(
                local_output, embeddings=embeddings, subject_mean=mean,
                epoch_index=np.arange(len(data), dtype=np.int32),
            )
            embedding_key = f"{base}/models/{MODEL_NAME}/embeddings/{item['task']}/{subject}.npz"
            client.upload_file(
                str(local_output), args.output_bucket, embedding_key,
                ExtraArgs={"ContentType": "application/octet-stream", "ServerSideEncryption": "AES256"},
            )
            record = {
                "task": item["task"], "subject": subject, "source_key": item["key"],
                "source_etag": item["etag"], "epochs": int(len(data)),
                "input_channels": 128, "embedding_key": embedding_key,
                "embedding_shape": list(embeddings.shape),
            }
            records.append(record)
            put_json(client, args.output_bucket, f"{base}/metadata/{item['task']}/{subject}.json", record)
            put_json(client, args.output_bucket, f"{base}/done/{item['task']}/{subject}.json", record)
            print(json.dumps({"progress": f"{number}/{len(items)}", **record}), flush=True)
        finally:
            local_input.unlink(missing_ok=True)
            local_output.unlink(missing_ok=True)

    quality = anti_collapse(means)
    report = {
        "status": "succeeded" if quality["passed"] else "failed_anti_collapse",
        "phase": "E0-LaBraM-128-spatial", "device": str(device),
        "tasks": tasks, "files": len(records), "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": LABRAM_SHA256,
        "model_output": MODEL_NAME,
        "preprocessing": "all 128 EGI channels, native 200 Hz, microvolts, ten 1-second patches per 10-second epoch",
        "spatial_adaptation": spatial_report,
        "anti_collapse": quality,
        "records": records,
    }
    put_json(client, args.output_bucket, f"{base}/smoke_report.json", report)
    if report["status"] != "succeeded":
        raise RuntimeError("Anti-collapse validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
