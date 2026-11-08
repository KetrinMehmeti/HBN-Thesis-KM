#!/usr/bin/env python3
"""Multi-subject smoke test for frozen LaBraM and EEGPT embeddings."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
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
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly


CHANNELS = [
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
]
EEGPT_WINDOW_STARTS = (0, 600, 1200)
EEGPT_SHA256 = "9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0"
LABRAM_SHA256 = "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"
LABRAM_CHANNEL_ORDER = [
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10", "O1", "OZ", "O2",
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
    parser.add_argument("--eegpt-weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/e0-le/work"))
    return parser.parse_args()


def s3_client(region: str):
    return boto3.client(
        "s3", region_name=region,
        config=Config(
            retries={"total_max_attempts": 8, "mode": "adaptive"},
            connect_timeout=10, read_timeout=180, max_pool_connections=20,
            s3={"us_east_1_regional_endpoint": "regional"},
            user_agent_appid="eeg-thesis-e0-labram-eegpt/1.0",
        ),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit(position) -> np.ndarray:
    vector = np.asarray(position, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid channel position")
    return vector / norm


def build_montage(channel_info: list[dict[str, Any]]):
    egi_names = [item["ch_name"] for item in channel_info]
    egi_positions = np.stack([unit(item["loc"][:3]) for item in channel_info])
    standard_raw = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    standard = {name.upper(): position for name, position in standard_raw.items()}
    targets = np.stack([unit(standard[name]) for name in CHANNELS])
    similarities = targets @ egi_positions.T
    rows, columns = linear_sum_assignment(-similarities)
    if not np.array_equal(rows, np.arange(len(CHANNELS))):
        raise RuntimeError("Unexpected montage assignment")
    report = [
        {
            "standard_electrode": CHANNELS[row], "egi_electrode": egi_names[column],
            "egi_index": int(column),
            "angular_distance_deg": float(np.degrees(np.arccos(np.clip(similarities[row, column], -1, 1)))),
        }
        for row, column in zip(rows, columns)
    ]
    return columns.astype(np.int64), report


def list_inputs(client, bucket: str, prefix: str, tasks: list[str], limit: int):
    by_task = {}
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


def load_labram(vendor: Path, weights: Path, device: torch.device):
    sys.path.insert(0, str(vendor / "LaBraM"))
    import modeling_finetune

    model = modeling_finetune.labram_base_patch200_200(
        EEG_size=2000, num_classes=0, use_mean_pooling=True,
        use_abs_pos_emb=True, use_rel_pos_bias=False, init_values=0.1, qkv_bias=False,
    )
    # The official LaBraM checkpoint stores NumPy scalar metadata. Keep the
    # restricted weights-only loader and allowlist only that required type.
    with torch.serialization.safe_globals([(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType, argparse.Namespace]):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint.get("module", checkpoint))
    filtered = OrderedDict((key[8:], value) for key, value in state.items() if key.startswith("student."))
    if not filtered:
        filtered = OrderedDict(state)
    # Transfer the pretrained token normalisation into the mean-pooling
    # normalisation used by the official downstream model.
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
    model.eval().requires_grad_(False).to(device)
    return model, [0] + [LABRAM_CHANNEL_ORDER.index(name) + 1 for name in CHANNELS]


def load_eegpt(vendor: Path, weights: Path, device: torch.device):
    sys.path.insert(0, str(vendor / "EEGPT" / "downstream"))
    from Modules.models.EEGPT_mcae import EEGTransformer

    model = EEGTransformer(
        img_size=[len(CHANNELS), 1024], patch_size=64, embed_num=4, embed_dim=512,
        depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0, init_std=0.02, qkv_bias=True,
        norm_layer=lambda size: torch.nn.LayerNorm(size, eps=1e-6),
    )
    # The official LaBraM checkpoint stores NumPy scalar metadata. Keep the
    # restricted weights-only loader and allowlist only that required type.
    # This legacy Lightning checkpoint uses a pickle opcode unsupported by
    # weights_only=True. main() verifies its exact official SHA-256 first;
    # retain only state_dict immediately after loading.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("target_encoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("target_encoder.")
    }
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, model.prepare_chan_ids(CHANNELS).to(device)


def extract_labram(model, data, indices, input_chans, device, batch_size):
    outputs = []
    for start in range(0, len(data), batch_size):
        stored = np.asarray(data[start:start + batch_size, indices, :], dtype=np.float32)
        x = torch.from_numpy((stored * 100.0).reshape(-1, len(indices), 10, 200)).to(device)
        with torch.inference_mode():
            outputs.append(model(x, input_chans=input_chans).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def extract_eegpt(model, data, indices, channel_ids, device, batch_size):
    outputs = []
    for start in range(0, len(data), batch_size):
        stored = np.asarray(data[start:start + batch_size, indices, :], dtype=np.float32)
        millivolts = stored / 10.0
        windows = []
        for offset in EEGPT_WINDOW_STARTS:
            window = resample_poly(millivolts[:, :, offset:offset + 800], 32, 25, axis=-1)
            with torch.inference_mode():
                tokens = model(torch.from_numpy(window.astype(np.float32)).to(device), channel_ids).float()
            windows.append(tokens.mean(dim=(1, 2)).cpu().numpy())
        outputs.append(np.mean(windows, axis=0))
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
    if sha256(args.eegpt_weights) != EEGPT_SHA256:
        raise RuntimeError("EEGPT checkpoint SHA-256 mismatch")
    client = s3_client(args.region)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        channel_path = Path(temp.name)
    try:
        client.download_file(args.input_bucket, args.channel_info_key, str(channel_path))
        with channel_path.open("rb") as stream:
            channel_info = pickle.load(stream)
    finally:
        channel_path.unlink(missing_ok=True)
    indices, montage_report = build_montage(channel_info)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(32, os.cpu_count() or 1)))
    vendor = Path(__file__).resolve().parent / "vendor"
    labram, labram_channels = load_labram(vendor, args.labram_weights, device)
    eegpt, eegpt_channels = load_eegpt(vendor, args.eegpt_weights, device)
    items = list_inputs(client, args.input_bucket, args.input_prefix, tasks, args.max_files)
    base = args.output_prefix.strip("/")
    means = {"labram": [], "eegpt": []}
    records = []
    started = time.time()
    for number, item in enumerate(items, start=1):
        subject = Path(item["key"]).stem
        local_input = args.work_dir / f"{subject}.npy"
        try:
            client.download_file(args.input_bucket, item["key"], str(local_input))
            data = np.load(local_input, mmap_mode="r", allow_pickle=False)
            if data.dtype != np.float32 or data.ndim != 3 or data.shape[1:] != (128, 2000):
                raise ValueError(f"Unexpected input {data.dtype} {data.shape}")
            if not np.isfinite(data).all():
                raise ValueError("Input contains non-finite samples")
            embeddings = {
                "labram": extract_labram(labram, data, indices, labram_channels, device, args.batch_size),
                "eegpt": extract_eegpt(eegpt, data, indices, eegpt_channels, device, args.batch_size),
            }
            keys = {}
            for model_name, values in embeddings.items():
                if not np.isfinite(values).all() or np.count_nonzero(values) != values.size:
                    raise RuntimeError(f"{model_name} produced invalid embeddings")
                mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
                means[model_name].append(mean)
                local_output = args.work_dir / f"{subject}.{model_name}.npz"
                np.savez_compressed(local_output, embeddings=values, subject_mean=mean,
                                    epoch_index=np.arange(len(data), dtype=np.int32))
                key = f"{base}/models/{model_name}/embeddings/{item['task']}/{subject}.npz"
                client.upload_file(str(local_output), args.output_bucket, key,
                                   ExtraArgs={"ContentType": "application/octet-stream", "ServerSideEncryption": "AES256"})
                local_output.unlink(missing_ok=True)
                keys[model_name] = key
            record = {
                "task": item["task"], "subject": subject, "source_key": item["key"],
                "source_etag": item["etag"], "epochs": int(len(data)),
                "embedding_keys": keys, "shapes": {name: list(value.shape) for name, value in embeddings.items()},
            }
            records.append(record)
            put_json(client, args.output_bucket, f"{base}/metadata/{item['task']}/{subject}.json", record)
            put_json(client, args.output_bucket, f"{base}/done/{item['task']}/{subject}.json", record)
            print(json.dumps({"progress": f"{number}/{len(items)}", **record}), flush=True)
        finally:
            local_input.unlink(missing_ok=True)
    report = {
        "status": "succeeded", "phase": "E0-L/E", "device": str(device),
        "tasks": tasks, "files": len(records), "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": {"labram": LABRAM_SHA256, "eegpt": EEGPT_SHA256},
        "preprocessing": {
            "common_montage": "58 EEGPT canonical electrodes assigned uniquely to nearest EGI-128 electrodes",
            "labram": "native 200 Hz, microvolts, ten 1-second patches per 10-second epoch",
            "eegpt": "millivolts, 200-to-256-Hz resampling, windows 0-4, 3-7, 6-10 seconds, mean pooling",
        },
        "montage": montage_report,
        "anti_collapse": {name: anti_collapse(vectors) for name, vectors in means.items()},
        "records": records,
    }
    if not all(item["passed"] for item in report["anti_collapse"].values()):
        report["status"] = "failed_anti_collapse"
    put_json(client, args.output_bucket, f"{base}/smoke_report.json", report)
    if report["status"] != "succeeded":
        raise RuntimeError("Anti-collapse validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


