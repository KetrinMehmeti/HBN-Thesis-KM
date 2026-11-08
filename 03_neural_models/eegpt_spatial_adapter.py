#!/usr/bin/env python3
"""Smoke-test EEGPT with canonical 62-channel and adapted EGI-128 representations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import mne
import numpy as np
import torch

from foundation_model_common import (
    CHANNELS,
    EEGPT_SHA256,
    anti_collapse,
    build_montage,
    extract_eegpt,
    list_inputs,
    load_eegpt,
    put_json,
    s3_client,
    sha256,
)


MODEL_62 = "eegpt_62_canonical"
MODEL_128 = "eegpt_128_spatial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", default="ica_qamad_ready")
    parser.add_argument("--channel-info-key", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--eegpt-weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/e0-eegpt-dual/work"))
    return parser.parse_args()


def unit(position: Any) -> np.ndarray:
    vector = np.asarray(position, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid electrode position")
    return vector / norm


def adapt_eegpt_to_egi128(
    model: torch.nn.Module,
    channel_info: list[dict[str, Any]],
    device: torch.device,
    neighbours: int = 4,
) -> tuple[torch.nn.Module, torch.Tensor, dict[str, Any]]:
    if len(channel_info) != 128:
        raise ValueError(f"Expected 128 EGI channels, got {len(channel_info)}")
    egi_names = [item["ch_name"] for item in channel_info]
    if len(set(egi_names)) != 128:
        raise ValueError("EGI channel names are not unique")
    egi_positions = np.stack([unit(item["loc"][:3]) for item in channel_info])

    standard_raw = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    standard = {name.upper(): unit(position) for name, position in standard_raw.items()}
    source_positions = np.stack([standard[name] for name in CHANNELS])
    angles = np.arccos(np.clip(egi_positions @ source_positions.T, -1.0, 1.0))
    nearest = np.argsort(angles, axis=1)[:, :neighbours]
    nearest_angles = np.take_along_axis(angles, nearest, axis=1)
    weights = 1.0 / np.maximum(nearest_angles, 1e-4) ** 2
    exact = nearest_angles[:, 0] < 1e-6
    weights /= weights.sum(axis=1, keepdims=True)
    weights[exact] = 0.0
    weights[exact, 0] = 1.0

    source_embeddings = model.chan_embed.weight.detach().cpu().numpy()
    if source_embeddings.shape[0] != len(CHANNELS):
        raise RuntimeError(f"Unexpected EEGPT channel table shape {source_embeddings.shape}")
    target_embeddings = np.sum(
        source_embeddings[nearest] * weights[:, :, None], axis=1
    ).astype(np.float32)

    adapted = copy.deepcopy(model)
    adapted.chan_embed = torch.nn.Embedding.from_pretrained(
        torch.from_numpy(target_embeddings).to(device), freeze=True
    )
    adapted.patch_embed.img_size = [128, 1024]
    adapted.patch_embed.num_patches = (128, 16)
    adapted.num_patches = adapted.patch_embed.num_patches
    adapted.eval().requires_grad_(False)
    channel_ids = torch.arange(128, device=device).unsqueeze(0).long()

    mapping = []
    for target_index, target_name in enumerate(egi_names):
        contributors = [
            {
                "canonical_electrode": CHANNELS[int(source_index)],
                "weight": float(weight),
                "angular_distance_deg": float(np.degrees(angle)),
            }
            for source_index, weight, angle in zip(
                nearest[target_index], weights[target_index], nearest_angles[target_index]
            )
        ]
        mapping.append(
            {"egi_electrode": target_name, "egi_index": target_index, "contributors": contributors}
        )
    nearest_degrees = np.degrees(nearest_angles[:, 0])
    report = {
        "method": "inverse_squared_angular_distance_interpolation",
        "source_canonical_positions": len(CHANNELS),
        "target_egi_positions": 128,
        "neighbours": neighbours,
        "nearest_distance_deg": {
            "min": float(nearest_degrees.min()),
            "mean": float(nearest_degrees.mean()),
            "p95": float(np.quantile(nearest_degrees, 0.95)),
            "max": float(nearest_degrees.max()),
        },
        "mapping": mapping,
    }
    return adapted, channel_ids, report


def load_channel_info(client, bucket: str, key: str) -> list[dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        path = Path(temp.name)
    try:
        client.download_file(bucket, key, str(path))
        with path.open("rb") as stream:
            return pickle.load(stream)
    finally:
        path.unlink(missing_ok=True)


def compare_representations(
    means_62: list[np.ndarray], means_128: list[np.ndarray]
) -> dict[str, Any]:
    matrix_62 = np.stack(means_62).astype(np.float64)
    matrix_128 = np.stack(means_128).astype(np.float64)
    normalized_62 = matrix_62 / np.maximum(np.linalg.norm(matrix_62, axis=1, keepdims=True), 1e-12)
    normalized_128 = matrix_128 / np.maximum(np.linalg.norm(matrix_128, axis=1, keepdims=True), 1e-12)
    cosine = np.sum(normalized_62 * normalized_128, axis=1)
    return {
        "subjects": len(cosine),
        "matched_subject_cosine_min": float(cosine.min()),
        "matched_subject_cosine_mean": float(cosine.mean()),
        "matched_subject_cosine_max": float(cosine.max()),
        "mean_absolute_difference": float(np.mean(np.abs(matrix_62 - matrix_128))),
        "representations_are_numerically_distinct": bool(not np.allclose(matrix_62, matrix_128)),
    }


def main() -> int:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if len(tasks) < 2:
        raise ValueError("Use at least two tasks for the dual EEGPT smoke test")
    if args.max_files < 8:
        raise ValueError("Use at least eight files for anti-collapse validation")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if sha256(args.eegpt_weights) != EEGPT_SHA256:
        raise RuntimeError("EEGPT checkpoint SHA-256 mismatch")

    client = s3_client(args.region)
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

    items = list_inputs(client, args.input_bucket, args.input_prefix, tasks, args.max_files)
    if len(items) != args.max_files:
        raise RuntimeError(f"Requested {args.max_files} files, found {len(items)}")
    base = args.output_prefix.strip("/")
    means: dict[str, list[np.ndarray]] = {MODEL_62: [], MODEL_128: []}
    records: list[dict[str, Any]] = []
    started = time.time()

    model_config = {
        MODEL_62: {
            "model": MODEL_62,
            "input_channels": 62,
            "channel_strategy": "unique minimum-cost mapping from 62 canonical positions to EGI-128",
            "mapping": mapping_62,
        },
        MODEL_128: {
            "model": MODEL_128,
            "input_channels": 128,
            "channel_strategy": "spatial interpolation of the learned 62-position EEGPT channel table",
            "mapping": mapping_128,
        },
    }
    for name, config in model_config.items():
        put_json(client, args.output_bucket, f"{base}/models/{name}/model_config.json", config)

    for number, item in enumerate(items, start=1):
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

            outputs = {
                MODEL_62: extract_eegpt(
                    model_62, data, indices_62, channel_ids_62, device, args.batch_size
                ),
                MODEL_128: extract_eegpt(
                    model_128, data, indices_128, channel_ids_128, device, args.batch_size
                ),
            }
            subject_record: dict[str, Any] = {
                "task": item["task"], "subject": subject, "source_key": item["key"],
                "source_etag": item["etag"], "epochs": int(len(data)), "models": {},
            }
            for name, embeddings in outputs.items():
                if embeddings.shape != (len(data), 512):
                    raise RuntimeError(f"Unexpected {name} shape {embeddings.shape}")
                if not np.isfinite(embeddings).all() or np.count_nonzero(embeddings) != embeddings.size:
                    raise RuntimeError(f"{name} produced invalid embeddings")
                mean = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
                means[name].append(mean)
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
                put_json(
                    client, args.output_bucket,
                    f"{model_root}/metadata/{item['task']}/{subject}.json", metadata,
                )
                put_json(
                    client, args.output_bucket,
                    f"{model_root}/done/{item['task']}/{subject}.json", metadata,
                )
                subject_record["models"][name] = metadata
            records.append(subject_record)
            print(json.dumps({"progress": f"{number}/{len(items)}", **subject_record}), flush=True)
        finally:
            local_input.unlink(missing_ok=True)
            for path in local_outputs.values():
                path.unlink(missing_ok=True)

    quality = {name: anti_collapse(vectors) for name, vectors in means.items()}
    comparison = compare_representations(means[MODEL_62], means[MODEL_128])
    passed = all(report["passed"] for report in quality.values()) and comparison[
        "representations_are_numerically_distinct"
    ]
    report = {
        "status": "succeeded" if passed else "failed_quality_check",
        "phase": "E0-EEGPT-dual-representation",
        "device": str(device), "tasks": tasks, "files": len(records),
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": EEGPT_SHA256,
        "temporal_protocol": "10-second native epoch; 0-4, 3-7, 6-10 second windows; 200-to-256 Hz resampling; mean pooling",
        "models": model_config, "anti_collapse": quality,
        "cross_representation_comparison": comparison, "records": records,
    }
    put_json(client, args.output_bucket, f"{base}/smoke_report.json", report)
    if not passed:
        raise RuntimeError("EEGPT dual-representation quality validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
