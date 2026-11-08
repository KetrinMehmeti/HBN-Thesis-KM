#!/usr/bin/env python3
"""Leakage-free subject-level fine-tuning for the E3 EEG benchmark.

The same entry point is used for architecture smoke tests and full runs.  A
subject, rather than a ten-second epoch, is the statistical unit.  Several
epochs are sampled for every subject and their encoder outputs are pooled
before regression.  Outer-test subjects never influence target or covariate
normalisation, model selection, or early stopping.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import io
import json
import math
import os
import pickle
import random
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import boto3
import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from scipy.signal import resample_poly
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

from eegpt_spatial_adapter import adapt_eegpt_to_egi128
from labram_128_adapter import adapt_positions_to_egi, load_labram
from foundation_model_common import build_montage, load_eegpt
from extract_cbramod_biot_embeddings import BIOT_PAIRS, build_biot_montage


BACKBONES = (
    "cbramod",
    "biot",
    "labram_128_spatial",
    "eegpt_62_canonical",
    "eegpt_128_spatial",
)
TARGETS = ("attention", "p_factor", "internalizing", "externalizing")
PIPELINE_PHASE = "E3-v3"
PIPELINE_REVISION = "v10-memory-safe-quality-audit"
TASK_CONDITIONS = {
    "RestingState": "resting_state",
    "DespicableMe": "passive_movie_watching",
    "DiaryOfAWimpyKid": "passive_movie_watching",
    "FunwithFractals": "passive_movie_watching",
    "ThePresent": "passive_movie_watching",
    "contrastChangeDetection_1": "active_task",
    "contrastChangeDetection_2": "active_task",
    "contrastChangeDetection_3": "active_task",
    "seqLearning6target": "active_task",
    "seqLearning8target": "active_task",
    "surroundSupp_1": "active_task",
    "surroundSupp_2": "active_task",
    "symbolSearch": "active_task",
}
EXPECTED_CHECKPOINT_SHA256 = {
    "cbramod": "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
    "biot": "40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70",
    "labram_128_spatial": "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c",
    "eegpt_62_canonical": "9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0",
    "eegpt_128_spatial": "9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0",
}

EEGPT_WINDOW_STARTS = (0, 600, 1200)
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=16 * 1024 * 1024,
    multipart_chunksize=16 * 1024 * 1024,
    max_concurrency=8,
    use_threads=True,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", default="ica_qamad_ready")
    parser.add_argument("--channel-info-key", required=True)
    parser.add_argument("--targets-uri", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--task", default="RestingState")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cbramod-weights", type=Path)
    parser.add_argument("--biot-weights", type=Path)
    parser.add_argument("--labram-weights", type=Path)
    parser.add_argument("--eegpt-weights", type=Path)
    parser.add_argument("--max-subjects", type=int, default=16)
    parser.add_argument("--outer-folds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--full-unfreeze-epoch", type=int, default=2)
    parser.add_argument("--partial-block-count", type=int, default=4)
    parser.add_argument("--train-epochs-per-subject", type=int, default=2)
    parser.add_argument("--train-epoch-chunk-size", type=int, default=1)
    parser.add_argument("--eval-epochs-per-subject", type=int, default=4)
    parser.add_argument("--eval-epoch-chunk-size", type=int, default=8)
    parser.add_argument("--subject-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-5)
    parser.add_argument("--full-encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--minimum-training-coverage", type=float, default=0.1)
    parser.add_argument("--minimum-prediction-std-ratio", type=float, default=0.05)
    parser.add_argument("--biot-normalization", choices=("official_volts", "q95_ablation"), default="official_volts")
    parser.add_argument("--cohort-uri")
    parser.add_argument("--split-manifest-uri")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_subjects != 0 and args.max_subjects < args.outer_folds * 4:
        raise ValueError("max-subjects must be zero or at least four per outer fold")
    if args.outer_folds < 2:
        raise ValueError("outer-folds must be at least two")
    if not 0 <= args.warmup_epochs < args.full_unfreeze_epoch < args.max_epochs:
        raise ValueError("Expected warmup-epochs < full-unfreeze-epoch < max-epochs")
    if args.partial_block_count < 1:
        raise ValueError("partial-block-count must be positive")
    if args.train_epochs_per_subject < 2:
        raise ValueError("At least two training epochs per subject are required")
    if args.eval_epochs_per_subject < 0 or args.eval_epochs_per_subject == 1:
        raise ValueError("eval-epochs-per-subject must be zero for all epochs or at least two")
    if (
        args.train_epoch_chunk_size < 1
        or args.eval_epoch_chunk_size < 1
        or args.subject_batch_size < 1
    ):
        raise ValueError("Batch and evaluation chunk sizes must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient-accumulation-steps must be positive")
    if not 0 < args.minimum_prediction_std_ratio < 1:
        raise ValueError("minimum-prediction-std-ratio must be between zero and one")
    if not 0 <= args.minimum_training_coverage <= 1:
        raise ValueError("minimum-training-coverage must be between zero and one")
    if args.mode == "full":
        if args.max_subjects != 0 or args.outer_folds < 5:
            raise ValueError("Full runs require all subjects and at least five outer folds")
        if args.max_epochs < 12 or args.train_epochs_per_subject < 8:
            raise ValueError("Full runs require at least 12 optimizer epochs and eight EEG epochs per subject")
        if args.eval_epochs_per_subject != 0:
            raise ValueError("Full runs must evaluate every available EEG epoch")
        if args.minimum_training_coverage < 0.8:
            raise ValueError("Full runs require minimum-training-coverage of at least 0.8")
    required = {
        "cbramod": args.cbramod_weights,
        "biot": args.biot_weights,
        "labram_128_spatial": args.labram_weights,
        "eegpt_62_canonical": args.eegpt_weights,
        "eegpt_128_spatial": args.eegpt_weights,
    }[args.backbone]
    if required is None or not required.is_file():
        raise ValueError(f"Missing checkpoint for {args.backbone}: {required}")
    actual_checkpoint_sha256 = sha256_file(required)
    if actual_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256[args.backbone]:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {args.backbone}: {actual_checkpoint_sha256}"
        )
    if args.backbone != "biot" and args.biot_normalization != "official_volts":
        raise ValueError("biot-normalization is only valid for the BIOT backbone")


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
            user_agent_appid="eeg-thesis-e3-v3/1.0",
        ),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI, got {uri}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(args: argparse.Namespace) -> Path:
    return {
        "cbramod": args.cbramod_weights,
        "biot": args.biot_weights,
        "labram_128_spatial": args.labram_weights,
        "eegpt_62_canonical": args.eegpt_weights,
        "eegpt_128_spatial": args.eegpt_weights,
    }[args.backbone]


def task_condition(task: str) -> str:
    return TASK_CONDITIONS.get(task, "unknown")

def validate_json_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Non-finite JSON value at {path}: {numeric}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            validate_json_numbers(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_json_numbers(item, f"{path}[{index}]")


def put_json(client, bucket: str, key: str, payload: dict[str, Any]) -> None:
    validate_json_numbers(payload)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, allow_nan=False, default=str).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def upload_frame(client, frame: pd.DataFrame, bucket: str, key: str, work_dir: Path) -> None:
    path = work_dir / (hashlib.sha256(key.encode()).hexdigest() + ".csv")
    frame.to_csv(path, index=False)
    client.upload_file(
        str(path), bucket, key,
        ExtraArgs={"ContentType": "text/csv", "ServerSideEncryption": "AES256"},
        Config=TRANSFER_CONFIG,
    )
    path.unlink(missing_ok=True)


def download_frame(client, bucket: str, key: str, work_dir: Path) -> pd.DataFrame:
    path = work_dir / (hashlib.sha256(key.encode()).hexdigest() + ".download.csv")
    client.download_file(bucket, key, str(path), Config=TRANSFER_CONFIG)
    try:
        return pd.read_csv(path)
    finally:
        path.unlink(missing_ok=True)


def upload_checkpoint(client, payload: dict[str, Any], bucket: str, key: str, work_dir: Path) -> None:
    path = work_dir / (hashlib.sha256(key.encode()).hexdigest() + ".pt")
    torch.save(payload, path)
    client.upload_file(
        str(path), bucket, key,
        ExtraArgs={"ContentType": "application/octet-stream", "ServerSideEncryption": "AES256"},
        Config=TRANSFER_CONFIG,
    )
    path.unlink(missing_ok=True)


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def get_json_or_none(client, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        with response["Body"] as body:
            return json.loads(body.read())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON at s3://{bucket}/{key}") from exc


def load_channel_info(client, bucket: str, key: str, work_dir: Path) -> list[dict[str, Any]]:
    path = work_dir / "channel_info.pkl"
    client.download_file(bucket, key, str(path), Config=TRANSFER_CONFIG)
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    finally:
        path.unlink(missing_ok=True)
    if not isinstance(value, list) or len(value) != 128:
        raise ValueError("Expected metadata for 128 EGI channels")
    return value


def download_targets(client, uri: str, target: str, work_dir: Path) -> pd.DataFrame:
    bucket, key = parse_s3_uri(uri)
    path = work_dir / "targets.parquet"
    client.download_file(bucket, key, str(path), Config=TRANSFER_CONFIG)
    try:
        frame = pd.read_parquet(path, columns=["participant_id", target])
    finally:
        path.unlink(missing_ok=True)
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target]).drop_duplicates("participant_id").reset_index(drop=True)
    if frame[target].nunique() < 8:
        raise RuntimeError(f"Target {target} has fewer than eight distinct values")
    return frame


def download_table(
    client, uri: str, work_dir: Path, label: str,
) -> tuple[pd.DataFrame, str]:
    bucket, key = parse_s3_uri(uri)
    suffix = Path(key).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError(f"{label} must be CSV or Parquet: {uri}")
    path = work_dir / f"{label}{suffix}"
    client.download_file(bucket, key, str(path), Config=TRANSFER_CONFIG)
    try:
        digest = sha256_file(path)
        frame = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
    finally:
        path.unlink(missing_ok=True)
    if "participant_id" not in frame.columns:
        raise ValueError(f"{label} is missing participant_id")
    frame["participant_id"] = frame["participant_id"].astype(str)
    return frame, digest


def npy_shape_from_s3(client, bucket: str, key: str) -> tuple[int, ...]:
    """Read only the NumPy header from S3 and return its declared shape."""
    initial_bytes = 4096
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{initial_bytes - 1}")
    with response["Body"] as body:
        payload = body.read()
    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        raise ValueError(f"Invalid NPY header at s3://{bucket}/{key}")
    major = int(payload[6])
    if major == 1:
        length_size = 2
        encoding = "latin1"
    elif major in (2, 3):
        length_size = 4
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ValueError(f"Unsupported NPY version {major} at s3://{bucket}/{key}")
    header_start = 8 + length_size
    if len(payload) < header_start:
        raise ValueError(f"Truncated NPY header at s3://{bucket}/{key}")
    header_length = int.from_bytes(payload[8:header_start], byteorder="little", signed=False)
    header_end = header_start + header_length
    if header_end > len(payload):
        response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{header_end - 1}")
        with response["Body"] as body:
            payload = body.read()
    if len(payload) < header_end:
        raise ValueError(f"Truncated NPY metadata at s3://{bucket}/{key}")
    metadata = ast.literal_eval(payload[header_start:header_end].decode(encoding).strip())
    shape = metadata.get("shape") if isinstance(metadata, dict) else None
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError(f"Unexpected NPY shape metadata at s3://{bucket}/{key}: {shape}")
    return tuple(int(value) for value in shape)


def list_signal_keys(client, bucket: str, prefix: str, task: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix.strip('/')}/{task}/")
    for page in pages:
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(".npy"):
                continue
            subject = Path(key).stem
            if subject in result:
                raise RuntimeError(f"Duplicate signal file for subject {subject}")
            result[subject] = {
                "source_key": key,
                "source_size": int(item["Size"]),
                "source_etag": str(item.get("ETag", "")).strip('"'),
            }
    for metadata in result.values():
        shape = npy_shape_from_s3(client, bucket, metadata["source_key"])
        if shape[1:] != (128, 2000):
            raise ValueError(f"Unexpected signal shape {shape} for {metadata['source_key']}")
        metadata["epoch_count"] = int(shape[0])
    return result


def select_subjects(frame: pd.DataFrame, target: str, max_subjects: int, seed: int) -> pd.DataFrame:
    frame = frame.sort_values([target, "participant_id"]).reset_index(drop=True)
    if max_subjects and len(frame) > max_subjects:
        # Evenly cover the target distribution instead of taking the first IDs.
        positions = np.linspace(0, len(frame) - 1, max_subjects).round().astype(int)
        frame = frame.iloc[np.unique(positions)].copy()
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@dataclass(frozen=True)
class SubjectRecord:
    participant_id: str
    target: float
    source_key: str
    source_size: int
    source_etag: str
    epoch_count: int


class SignalStore:
    def __init__(self, client, bucket: str, cache_dir: Path):
        self.client = client
        self.bucket = bucket
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, record: SubjectRecord) -> Path:
        return self.cache_dir / (hashlib.sha256(record.source_key.encode()).hexdigest() + ".npy")

    def load(self, record: SubjectRecord) -> np.ndarray:
        path = self.path_for(record)
        if not path.exists() or path.stat().st_size != record.source_size:
            temporary = path.with_suffix(".part")
            temporary.unlink(missing_ok=True)
            self.client.download_file(self.bucket, record.source_key, str(temporary), Config=TRANSFER_CONFIG)
            os.replace(temporary, path)
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.dtype != np.float32 or data.ndim != 3 or data.shape[1:] != (128, 2000):
            raise ValueError(f"Unexpected input for {record.participant_id}: {data.dtype} {data.shape}")
        if len(data) < 2:
            raise ValueError(f"Subject {record.participant_id} has fewer than two epochs")
        return data

    def sample(
        self,
        record: SubjectRecord,
        count: int,
        rng: np.random.Generator,
        deterministic: bool,
        rotation_step: int | None = None,
        rotation_seed: int = 0,
    ) -> tuple[np.ndarray, list[int]]:
        data = self.load(record)
        actual = len(data) if count == 0 else min(count, len(data))
        if deterministic:
            indices = (
                np.arange(len(data), dtype=int)
                if actual == len(data)
                else np.linspace(0, len(data) - 1, actual).round().astype(int)
            )
        elif rotation_step is not None:
            digest = hashlib.sha256(
                f"{record.participant_id}|{rotation_seed}".encode("utf-8")
            ).digest()
            stable_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
            permutation = np.random.default_rng(stable_seed).permutation(len(data))
            start = (rotation_step * actual) % len(data)
            positions = (np.arange(start, start + actual) % len(data)).astype(int)
            indices = np.sort(permutation[positions])
        else:
            indices = np.sort(rng.choice(len(data), size=actual, replace=False))
        values = np.asarray(data[indices], dtype=np.float32)
        return values, [int(index) for index in indices]


class EncoderAdapter(torch.nn.Module):
    def __init__(self, backbone: str, encoder: torch.nn.Module, feature_dim: int, **state: Any):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.state = state

    def preprocess(self, subject_epochs: list[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
        counts = [len(value) for value in subject_epochs]
        if len(set(counts)) != 1:
            raise ValueError("All subjects in a batch must contribute the same number of epochs")
        stored = np.concatenate(subject_epochs, axis=0).astype(np.float32, copy=False)
        if self.backbone in ("cbramod", "labram_128_spatial"):
            array = (stored * 100.0).reshape(-1, 128, 10, 200)
        elif self.backbone == "biot":
            left = self.state["biot_left"]
            right = self.state["biot_right"]
            array = (stored[:, left, :] - stored[:, right, :]).astype(np.float32, copy=False)
            if self.state["biot_normalization"] == "q95_ablation":
                scale = np.quantile(np.abs(array), 0.95, axis=-1, keepdims=True)
                array = (array / (scale + 1e-8)).astype(np.float32, copy=False)
        else:
            indices = self.state["eegpt_indices"]
            selected = stored[:, indices, :] / 10.0
            windows = [
                resample_poly(selected[:, :, offset:offset + 800], 32, 25, axis=-1).astype(np.float32)
                for offset in EEGPT_WINDOW_STARTS
            ]
            array = np.stack(windows, axis=1).reshape(-1, len(indices), 1024)
        return torch.from_numpy(np.ascontiguousarray(array)), counts

    def encode_epochs(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.backbone == "cbramod":
            patch_embeddings = self.encoder.patch_embedding(tensor)
            epoch_features = self.encoder.encoder(patch_embeddings).float().mean(dim=(1, 2))
        elif self.backbone == "biot":
            epoch_features = self.encoder(tensor).float()
        elif self.backbone == "labram_128_spatial":
            epoch_features = self.encoder(tensor, input_chans=self.state["labram_input_chans"]).float()
        else:
            tokens = self.encoder(tensor, self.state["eegpt_channel_ids"]).float()
            window_features = tokens.mean(dim=(1, 2))
            if len(window_features) % 3:
                raise RuntimeError("EEGPT produced a non-divisible number of temporal windows")
            epoch_features = window_features.reshape(-1, 3, window_features.shape[-1]).mean(dim=1)
        if epoch_features.shape[-1] != self.feature_dim:
            raise RuntimeError(f"Unexpected {self.backbone} feature shape {tuple(epoch_features.shape)}")
        return epoch_features

    def forward(self, tensor: torch.Tensor, subject_count: int, epochs_per_subject: int) -> torch.Tensor:
        if self.backbone.startswith("eegpt_") and self.training:
            return self._forward_eegpt_memory_safe(
                tensor, subject_count, epochs_per_subject,
            )
        if self.backbone == "labram_128_spatial" and self.training:
            return self._forward_labram_memory_safe(
                tensor, subject_count, epochs_per_subject,
            )
        epoch_features = self.encode_epochs(tensor)
        if len(epoch_features) != subject_count * epochs_per_subject:
            raise RuntimeError("Encoder epoch count does not match the subject batch")
        return epoch_features.reshape(subject_count, epochs_per_subject, -1).mean(dim=1)

    def _forward_eegpt_memory_safe(
        self,
        tensor: torch.Tensor,
        subject_count: int,
        epochs_per_subject: int,
    ) -> torch.Tensor:
        """Pool EEGPT epochs while checkpointing encoder activations.

        EEGPT expands every ten-second epoch into three four-second windows. A
        single subject with eight epochs therefore enters the transformer as 24
        windows. On 16 GiB T4 GPUs the full encoder plus AdamW state can fit,
        but retaining all window activations until backward is borderline. We
        preserve the exact subject-level mean while recomputing small chunks in
        backward, trading compute for bounded activation memory.
        """
        rows_per_epoch = len(EEGPT_WINDOW_STARTS)
        expected_rows = subject_count * epochs_per_subject * rows_per_epoch
        if len(tensor) != expected_rows:
            raise RuntimeError(
                "EEGPT input rows do not match subjects, epochs, and temporal windows"
            )
        chunk_epochs = min(
            int(self.state.get("train_epoch_chunk_size", 1)),
            epochs_per_subject,
        )
        subject_rows = epochs_per_subject * rows_per_epoch
        checkpoint_encoder = any(
            parameter.requires_grad for parameter in self.encoder.parameters()
        )
        pooled_subjects: list[torch.Tensor] = []
        for subject in range(subject_count):
            subject_tensor = tensor[
                subject * subject_rows:(subject + 1) * subject_rows
            ]
            chunks: list[torch.Tensor] = []
            for start_epoch in range(0, epochs_per_subject, chunk_epochs):
                stop_epoch = min(start_epoch + chunk_epochs, epochs_per_subject)
                chunk = subject_tensor[
                    start_epoch * rows_per_epoch:stop_epoch * rows_per_epoch
                ]
                if checkpoint_encoder:
                    features = checkpoint(
                        self.encode_epochs, chunk, use_reentrant=False,
                    )
                else:
                    features = self.encode_epochs(chunk)
                chunks.append(features)
            epoch_features = torch.cat(chunks, dim=0)
            if len(epoch_features) != epochs_per_subject:
                raise RuntimeError("EEGPT chunking changed the epoch count")
            pooled_subjects.append(epoch_features.mean(dim=0))
        return torch.stack(pooled_subjects, dim=0)


    def _forward_labram_memory_safe(
        self,
        tensor: torch.Tensor,
        subject_count: int,
        epochs_per_subject: int,
    ) -> torch.Tensor:
        """Pool LabRAM epochs with activation checkpointing during training.

        LabRAM receives all 128 channels and, when the full encoder is
        unfrozen, retaining activations for eight ten-second epochs can exceed
        the 16 GiB available on a T4. Processing the epochs in small chunks
        preserves the exact subject-level mean while checkpointing the encoder
        activations, so backward recomputes each chunk instead of retaining all
        of them concurrently.
        """
        expected_rows = subject_count * epochs_per_subject
        if len(tensor) != expected_rows:
            raise RuntimeError(
                "LabRAM input rows do not match subjects and EEG epochs"
            )
        chunk_epochs = min(
            int(self.state.get("train_epoch_chunk_size", 1)),
            epochs_per_subject,
        )
        checkpoint_encoder = any(
            parameter.requires_grad for parameter in self.encoder.parameters()
        )
        pooled_subjects: list[torch.Tensor] = []
        for subject in range(subject_count):
            start = subject * epochs_per_subject
            subject_tensor = tensor[start:start + epochs_per_subject]
            chunks: list[torch.Tensor] = []
            for start_epoch in range(0, epochs_per_subject, chunk_epochs):
                chunk = subject_tensor[start_epoch:start_epoch + chunk_epochs]
                if checkpoint_encoder:
                    features = checkpoint(
                        self.encode_epochs, chunk, use_reentrant=False,
                    )
                else:
                    features = self.encode_epochs(chunk)
                chunks.append(features)
            epoch_features = torch.cat(chunks, dim=0)
            if len(epoch_features) != epochs_per_subject:
                raise RuntimeError("LabRAM chunking changed the epoch count")
            pooled_subjects.append(epoch_features.mean(dim=0))
        return torch.stack(pooled_subjects, dim=0)

class SubjectRegressor(torch.nn.Module):
    def __init__(self, adapter: EncoderAdapter):
        super().__init__()
        self.adapter = adapter
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(adapter.feature_dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(adapter.feature_dim, 1),
        )

    def forward(self, tensor: torch.Tensor, subject_count: int, epochs_per_subject: int) -> torch.Tensor:
        features = self.adapter(tensor, subject_count, epochs_per_subject)
        return self.head(features).reshape(-1)


def load_adapter(args: argparse.Namespace, channel_info: list[dict[str, Any]], device: torch.device) -> EncoderAdapter:
    vendor = Path(__file__).resolve().parent / "vendor"
    if args.backbone == "cbramod":
        sys.path.insert(0, str(vendor / "CBraMod"))
        from models.cbramod import CBraMod
        encoder = CBraMod()
        encoder.load_state_dict(torch.load(args.cbramod_weights, map_location="cpu", weights_only=True), strict=True)
        return EncoderAdapter(args.backbone, encoder.to(device), 200, representation="pre_projection_latent")
    if args.backbone == "biot":
        # Import the encoder module directly. The upstream model package eagerly
        # imports unrelated architectures that are absent from the compact asset.
        sys.path.insert(0, str(vendor / "BIOT" / "model"))
        from biot import BIOTEncoder
        encoder = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100)
        encoder.load_state_dict(torch.load(args.biot_weights, map_location="cpu", weights_only=True), strict=True)
        mapping, _ = build_biot_montage(channel_info)
        left = np.asarray([mapping[a] for a, _ in BIOT_PAIRS], dtype=int)
        right = np.asarray([mapping[b] for _, b in BIOT_PAIRS], dtype=int)
        return EncoderAdapter(
            args.backbone, encoder.to(device), 256, biot_left=left, biot_right=right,
            biot_normalization=args.biot_normalization, representation="official_encoder_output",
        )
    if args.backbone == "labram_128_spatial":
        encoder = load_labram(vendor, args.labram_weights, device)
        input_chans, _ = adapt_positions_to_egi(encoder, channel_info)
        return EncoderAdapter(args.backbone, encoder, 200, labram_input_chans=input_chans)
    base, channel_ids = load_eegpt(vendor, args.eegpt_weights, device)
    if args.backbone == "eegpt_62_canonical":
        indices, _ = build_montage(channel_info)
        encoder = base
    else:
        encoder, channel_ids, _ = adapt_eegpt_to_egi128(base, channel_info, device)
        indices = np.arange(128, dtype=np.int64)
    return EncoderAdapter(
        args.backbone, encoder, 512,
        eegpt_indices=np.asarray(indices, dtype=np.int64), eegpt_channel_ids=channel_ids,
        train_epoch_chunk_size=args.train_epoch_chunk_size,
    )




def stage_for_epoch(epoch: int, args: argparse.Namespace) -> str:
    if epoch < args.warmup_epochs:
        return "head_warmup"
    if epoch < args.full_unfreeze_epoch:
        return "partial_encoder"
    return "full_encoder"


def _is_trainable_encoder_parameter(backbone: str, name: str, parameter: torch.nn.Parameter) -> bool:
    if not (parameter.is_floating_point() or parameter.is_complex()):
        return False
    if backbone == "cbramod" and (name.startswith("proj_out.") or name.endswith("mask_encoding")):
        return False
    if backbone == "biot" and name == "index":
        return False
    return True


def set_trainable_stage(
    model: SubjectRegressor,
    stage: str,
    partial_block_count: int,
) -> list[str]:
    for parameter in model.adapter.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    if stage == "head_warmup":
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    encoder = model.adapter.encoder
    selected_modules: list[torch.nn.Module] = []
    if stage == "partial_encoder":
        if model.adapter.backbone == "cbramod":
            selected_modules.extend(list(encoder.encoder.layers)[-partial_block_count:])
            if getattr(encoder.encoder, "norm", None) is not None:
                selected_modules.append(encoder.encoder.norm)
        elif model.adapter.backbone == "biot":
            selected_modules.append(encoder.transformer)
        elif hasattr(encoder, "blocks"):
            selected_modules.extend(list(encoder.blocks)[-partial_block_count:])
            for name in ("norm", "fc_norm"):
                module = getattr(encoder, name, None)
                if isinstance(module, torch.nn.Module):
                    selected_modules.append(module)
        else:
            raise RuntimeError(f"No partial-unfreeze rule for {model.adapter.backbone}")
        for module in selected_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif stage == "full_encoder":
        for name, parameter in encoder.named_parameters():
            parameter.requires_grad = _is_trainable_encoder_parameter(
                model.adapter.backbone, name, parameter,
            )
    else:
        raise ValueError(f"Unknown training stage {stage}")

    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def make_optimizer(
    model: SubjectRegressor,
    args: argparse.Namespace,
    stage: str,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [{
        "params": [parameter for parameter in model.head.parameters() if parameter.requires_grad],
        "lr": args.head_learning_rate,
    }]
    encoder_parameters = [
        parameter for parameter in model.adapter.encoder.parameters() if parameter.requires_grad
    ]
    if encoder_parameters:
        encoder_lr = (
            args.full_encoder_learning_rate
            if stage == "full_encoder"
            else args.encoder_learning_rate
        )
        groups.append({"params": encoder_parameters, "lr": encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def encoder_reference_state(model: SubjectRegressor) -> dict[str, torch.Tensor]:
    return {
        f"adapter.encoder.{name}": parameter.detach().float().cpu().clone()
        for name, parameter in model.adapter.encoder.named_parameters()
        if _is_trainable_encoder_parameter(model.adapter.backbone, name, parameter)
    }


def parameter_counts(model: SubjectRegressor) -> dict[str, int]:
    return {
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_parameters": sum(parameter.numel() for parameter in model.adapter.encoder.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def prediction_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    target_std = float(np.std(np.asarray(y_true, dtype=np.float64)))
    prediction_std = float(np.std(np.asarray(y_pred, dtype=np.float64)))
    ratio = prediction_std / target_std if target_std > 0 else 0.0
    return {
        "target_std": target_std,
        "prediction_std": prediction_std,
        "prediction_std_ratio": ratio,
        "unique_predictions": int(np.unique(np.round(y_pred, decimals=8)).size),
    }


def prediction_collapse_detected(
    diagnostics: dict[str, float | int],
    sample_count: int,
    minimum_prediction_std_ratio: float,
) -> bool:
    minimum_unique = min(sample_count, max(3, math.ceil(sample_count * 0.05)))
    return bool(
        not math.isfinite(float(diagnostics["prediction_std_ratio"]))
        or float(diagnostics["prediction_std_ratio"]) < minimum_prediction_std_ratio
        or int(diagnostics["unique_predictions"]) < minimum_unique
    )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics received non-finite targets or predictions")
    error = y_true - y_pred
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    pearson_value = float(pearsonr(y_true, y_pred).statistic) if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0
    spearman_value = float(spearmanr(y_true, y_pred).statistic) if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0
    values = {
        "r2": float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else 0.0,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson": pearson_value if math.isfinite(pearson_value) else 0.0,
        "spearman": spearman_value if math.isfinite(spearman_value) else 0.0,
    }
    validate_json_numbers(values, "$.metrics")
    return values


def iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def prepare_batch(
    adapter: EncoderAdapter,
    records: list[SubjectRecord],
    batch_indices: np.ndarray,
    store: SignalStore,
    epochs_per_subject: int,
    rng: np.random.Generator,
    deterministic: bool,
    device: torch.device,
    rotation_step: int | None = None,
    rotation_seed: int = 0,
) -> tuple[torch.Tensor, list[list[int]]]:
    samples: list[np.ndarray] = []
    selected: list[list[int]] = []
    for index in batch_indices:
        value, epoch_indices = store.sample(
            records[int(index)], epochs_per_subject, rng, deterministic,
            rotation_step=rotation_step, rotation_seed=rotation_seed,
        )
        samples.append(value)
        selected.append(epoch_indices)
    tensor, counts = adapter.preprocess(samples)
    if len(set(counts)) != 1:
        raise RuntimeError("Inconsistent sampled epoch counts")
    return tensor.to(device, non_blocking=True), selected


def evaluate(
    model: SubjectRegressor,
    records: list[SubjectRecord],
    indices: np.ndarray,
    store: SignalStore,
    epochs_per_subject: int,
    target_mean: float,
    target_std: float,
    seed: int,
    device: torch.device,
    epoch_chunk_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model.eval()
    # Subject-level batches are small, so FP32 is affordable and prevents
    # mixed-precision overflows from obscuring encoder diagnostics.
    mixed_precision_enabled = False
    predictions: list[float] = []
    audit: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for raw_index in indices:
            index = int(raw_index)
            sample, epoch_indices = store.sample(
                records[index], epochs_per_subject, rng, deterministic=True,
            )
            epoch_features: list[torch.Tensor] = []
            for start in range(0, len(sample), epoch_chunk_size):
                chunk = sample[start:start + epoch_chunk_size]
                tensor, counts = model.adapter.preprocess([chunk])
                if counts != [len(chunk)]:
                    raise RuntimeError("Evaluation epoch chunk has inconsistent size")
                tensor = tensor.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=mixed_precision_enabled,
                ):
                    epoch_features.append(model.adapter.encode_epochs(tensor).float())
            pooled = torch.cat(epoch_features, dim=0).mean(dim=0, keepdim=True)
            standardized = model.head(pooled).reshape(-1)
            value = float(standardized.float().cpu().item() * target_std + target_mean)
            predictions.append(value)
            audit.append({
                "participant_id": records[index].participant_id,
                "epoch_indices": epoch_indices,
                "available_epoch_count": records[index].epoch_count,
                "evaluation_epoch_coverage": len(epoch_indices) / records[index].epoch_count,
            })
    return np.asarray(predictions, dtype=np.float64), audit

def parameter_fingerprint(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    values = [parameter.detach().float().reshape(-1)[:1024].cpu() for parameter in parameters if parameter.requires_grad]
    return torch.cat(values) if values else torch.empty(0)
def update_gradient_diagnostics(
    maximum: float, steps: int, nonfinite_steps: int,
    total_gradient: float, encoder_gradient: float,
) -> tuple[float, int, int]:
    steps += 1
    if not (math.isfinite(total_gradient) and math.isfinite(encoder_gradient)):
        nonfinite_steps += 1
    elif encoder_gradient > 0:
        maximum = max(maximum, encoder_gradient)
    return maximum, steps, nonfinite_steps




def run_fold(
    args: argparse.Namespace,
    client,
    records: list[SubjectRecord],
    train_outer: np.ndarray,
    test_indices: np.ndarray,
    fold: int,
    channel_info: list[dict[str, Any]],
    store: SignalStore,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rng = np.random.default_rng(args.seed + fold * 1009)
    shuffled = rng.permutation(train_outer)
    validation_count = max(2, int(round(len(shuffled) * args.validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 2)
    validation_indices = np.sort(shuffled[:validation_count])
    train_indices = np.sort(shuffled[validation_count:])
    overlap = (
        set(train_indices) & set(validation_indices)
        | set(train_indices) & set(test_indices)
        | set(validation_indices) & set(test_indices)
    )
    if overlap:
        raise RuntimeError(f"Participant leakage detected in fold {fold}")

    y = np.asarray([record.target for record in records], dtype=np.float32)
    target_mean = float(y[train_indices].mean())
    target_std = float(y[train_indices].std())
    if target_std < 1e-8:
        raise RuntimeError(f"Near-constant training target in fold {fold}")

    adapter = load_adapter(args, channel_info, device)
    model = SubjectRegressor(adapter).to(device)
    # Use FP32 for every backbone during end-to-end fine-tuning.
    mixed_precision_enabled = False
    base_encoder_reference = encoder_reference_state(model)
    current_stage = ""
    optimizer: torch.optim.Optimizer | None = None
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision_enabled)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience_used = 0
    history: list[dict[str, Any]] = []
    all_epoch_audit: list[dict[str, Any]] = []
    maximum_encoder_gradient = 0.0
    encoder_gradient_steps = 0
    nonfinite_gradient_steps = 0
    full_stage_reached = False
    stage_parameter_counts: dict[str, dict[str, int]] = {}
    trainable_names_by_stage: dict[str, list[str]] = {}
    training_epoch_sets = {
        records[int(index)].participant_id: set() for index in train_indices
    }
    largest_training_record = max(records[int(index)].epoch_count for index in train_indices)
    coverage_driven_count = math.ceil(
        largest_training_record * args.minimum_training_coverage / args.max_epochs
    )
    effective_train_epochs = max(args.train_epochs_per_subject, coverage_driven_count)

    for epoch in range(args.max_epochs):
        stage = stage_for_epoch(epoch, args)
        if stage != current_stage:
            trainable_names = set_trainable_stage(model, stage, args.partial_block_count)
            trainable_names_by_stage[stage] = trainable_names
            stage_parameter_counts[stage] = parameter_counts(model)
            optimizer = make_optimizer(model, args, stage)
            optimizer.zero_grad(set_to_none=True)
            current_stage = stage
            if stage == "full_encoder":
                full_stage_reached = True
                best_loss = math.inf
                best_state = None
                patience_used = 0
        if optimizer is None:
            raise RuntimeError("Optimizer was not initialized")

        model.train()
        order = rng.permutation(train_indices)
        batches = list(iter_batches(order, args.subject_batch_size))
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_number, batch in enumerate(batches, start=1):
            tensor, selected = prepare_batch(
                model.adapter, records, batch, store, effective_train_epochs,
                rng, False, device, rotation_step=epoch,
                rotation_seed=args.seed + fold * 1009,
            )
            target_values = torch.tensor(
                [(records[int(index)].target - target_mean) / target_std for index in batch],
                dtype=torch.float32, device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision_enabled,
            ):
                predictions = model(tensor, len(batch), len(selected[0]))
                raw_loss = torch.nn.functional.smooth_l1_loss(predictions.float(), target_values)
                loss = raw_loss / args.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in fold {fold}")
            scaler.scale(loss).backward()
            should_step = (
                batch_number % args.gradient_accumulation_steps == 0
                or batch_number == len(batches)
            )
            if should_step:
                scaler.unscale_(optimizer)
                encoder_gradient = 0.0
                if stage != "head_warmup":
                    encoder_norms = [
                        parameter.grad.detach().float().norm(2)
                        for parameter in model.adapter.encoder.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    if encoder_norms:
                        encoder_gradient = float(
                            torch.linalg.vector_norm(torch.stack(encoder_norms)).detach().cpu()
                        )
                gradient = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu()
                )
                if stage != "head_warmup":
                    maximum_encoder_gradient, encoder_gradient_steps, nonfinite_gradient_steps = (
                        update_gradient_diagnostics(
                            maximum_encoder_gradient, encoder_gradient_steps,
                            nonfinite_gradient_steps, gradient, encoder_gradient,
                        )
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.detach().cpu()))
            for index, epochs in zip(batch, selected):
                participant_id = records[int(index)].participant_id
                training_epoch_sets[participant_id].update(epochs)
                all_epoch_audit.append({
                    "fold": fold,
                    "split": "train",
                    "training_epoch": epoch + 1,
                    "stage": stage,
                    "participant_id": participant_id,
                    "available_epoch_count": records[int(index)].epoch_count,
                    "epoch_indices": json.dumps(epochs),
                })

        validation_prediction, validation_audit = evaluate(
            model, records, validation_indices, store, args.eval_epochs_per_subject,
            target_mean, target_std, args.seed + fold, device, args.eval_epoch_chunk_size,
        )
        validation_true = y[validation_indices].astype(np.float64)
        if not np.isfinite(validation_prediction).all():
            raise RuntimeError(f"Non-finite validation predictions in fold {fold}, epoch {epoch + 1}")
        validation_loss = float(np.mean((validation_true - validation_prediction) ** 2))
        if not math.isfinite(validation_loss):
            raise RuntimeError(f"Non-finite validation loss in fold {fold}, epoch {epoch + 1}")
        current_coverages = [
            len(training_epoch_sets[records[int(index)].participant_id])
            / records[int(index)].epoch_count
            for index in train_indices
        ]
        history.append({
            "fold": fold,
            "epoch": epoch + 1,
            "stage": stage,
            "mean_train_loss": float(np.mean(losses)),
            "validation_mse": validation_loss,
            "training_epoch_coverage_mean": float(np.mean(current_coverages)),
            "training_epoch_coverage_min": float(np.min(current_coverages)),
            **stage_parameter_counts[stage],
        })
        for row in validation_audit:
            all_epoch_audit.append({
                "fold": fold,
                "split": "validation",
                "training_epoch": epoch + 1,
                "stage": stage,
                "participant_id": row["participant_id"],
                "available_epoch_count": row["available_epoch_count"],
                "epoch_indices": json.dumps(row["epoch_indices"]),
            })
        if stage == "full_encoder":
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                }
                patience_used = 0
            else:
                patience_used += 1
            coverage_ready = min(current_coverages) >= args.minimum_training_coverage
            if patience_used >= args.patience and coverage_ready:
                break

    if not full_stage_reached or best_state is None:
        raise RuntimeError(f"No full-encoder checkpoint produced for fold {fold}")
    model.load_state_dict(best_state, strict=True)
    if encoder_gradient_steps == 0 or nonfinite_gradient_steps >= encoder_gradient_steps:
        raise RuntimeError(f"No finite encoder-gradient step completed in fold {fold}")
    final_parameters = dict(model.named_parameters())
    encoder_delta_state = {
        name: (final_parameters[name].detach().float().cpu() - reference)
        for name, reference in base_encoder_reference.items()
    }
    for name, delta in encoder_delta_state.items():
        if not torch.isfinite(delta).all():
            raise RuntimeError(f"Non-finite encoder checkpoint delta for {name} in fold {fold}")
    encoder_delta = max(
        (float(delta.abs().max()) for delta in encoder_delta_state.values()),
        default=0.0,
    )
    if not math.isfinite(encoder_delta):
        raise RuntimeError(f"Non-finite encoder delta in fold {fold}")
    head_state = {
        name: value.detach().float().cpu()
        for name, value in model.state_dict().items()
        if name.startswith("head.")
    }
    checkpoint_payload = {
        "format": "base-plus-full-encoder-delta-v2",
        "pipeline_phase": PIPELINE_PHASE,
        "pipeline_revision": PIPELINE_REVISION,
        "fold": fold,
        "target": args.target,
        "backbone": args.backbone,
        "task": args.task,
        "task_condition": task_condition(args.task),
        "requires_base_checkpoint": True,
        "base_checkpoint_sha256": args.base_checkpoint_sha256,
        "trainable_parameter_names_by_stage": trainable_names_by_stage,
        "encoder_delta_state": encoder_delta_state,
        "head_state": head_state,
    }

    test_prediction, test_audit = evaluate(
        model, records, test_indices, store, args.eval_epochs_per_subject,
        target_mean, target_std, args.seed + fold * 17, device, args.eval_epoch_chunk_size,
    )
    test_true = y[test_indices].astype(np.float64)
    if not np.isfinite(test_prediction).all():
        raise RuntimeError(f"Non-finite test predictions in fold {fold}")
    metrics = metric_dict(test_true, test_prediction)
    diagnostics = prediction_diagnostics(test_true, test_prediction)
    diagnostics["prediction_collapse_detected"] = prediction_collapse_detected(
        diagnostics, len(test_prediction), args.minimum_prediction_std_ratio,
    )
    if encoder_delta <= 0 or maximum_encoder_gradient <= 0 or not math.isfinite(maximum_encoder_gradient):
        raise RuntimeError(f"Encoder did not update during full fine-tuning in fold {fold}")

    training_coverages = [
        len(training_epoch_sets[records[int(index)].participant_id])
        / records[int(index)].epoch_count
        for index in train_indices
    ]
    evaluation_coverages = [float(row["evaluation_epoch_coverage"]) for row in test_audit]
    prediction_rows = []
    for index, prediction, audit in zip(test_indices, test_prediction, test_audit):
        prediction_rows.append({
            "fold": fold,
            "participant_id": records[int(index)].participant_id,
            "y_true": float(records[int(index)].target),
            "y_pred": float(prediction),
            "source_key": records[int(index)].source_key,
            "available_epoch_count": records[int(index)].epoch_count,
            "evaluation_epoch_count": len(audit["epoch_indices"]),
            "evaluation_epoch_coverage": audit["evaluation_epoch_coverage"],
            "evaluation_epoch_indices": json.dumps(audit["epoch_indices"]),
        })
    fold_result = {
        "fold": fold,
        "target": args.target,
        "backbone": args.backbone,
        "task": args.task,
        "task_condition": task_condition(args.task),
        "n_train": len(train_indices),
        "n_validation": len(validation_indices),
        "n_test": len(test_indices),
        "target_train_mean": target_mean,
        "target_train_std": target_std,
        "best_validation_mse": best_loss,
        "epochs_completed": len(history),
        "effective_train_epochs_per_subject_step": effective_train_epochs,
        "training_epoch_coverage_mean": float(np.mean(training_coverages)),
        "training_epoch_coverage_min": float(np.min(training_coverages)),
        "evaluation_epoch_coverage_mean": float(np.mean(evaluation_coverages)),
        "evaluation_epoch_coverage_min": float(np.min(evaluation_coverages)),
        "encoder_parameter_max_delta": encoder_delta,
        "maximum_trainable_gradient_norm": maximum_encoder_gradient,
        "participant_overlap": 0,
        "trainable_parameter_names_by_stage": json.dumps(trainable_names_by_stage),
        "stage_parameter_counts": json.dumps(stage_parameter_counts),
        "full_stage_reached": full_stage_reached,
        "encoder_gradient_steps": encoder_gradient_steps,
        "nonfinite_gradient_steps": nonfinite_gradient_steps,
        "nonfinite_gradient_fraction": nonfinite_gradient_steps / encoder_gradient_steps,
        "checkpoint_format": checkpoint_payload["format"],
        **diagnostics,
        **metrics,
    }
    return (
        fold_result,
        pd.DataFrame(prediction_rows),
        pd.DataFrame(history + all_epoch_audit),
        checkpoint_payload,
    )

def main_work(args: argparse.Namespace) -> int:
    validate_args(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.base_checkpoint_sha256 = sha256_file(checkpoint_path(args))
    client = s3_client(args.region)
    base = args.output_prefix.strip("/")
    runtime = f"{base}/runtime"
    started = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("E3 fine-tuning requires a CUDA GPU")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    condition = task_condition(args.task)
    if condition == "unknown":
        raise ValueError(f"Task {args.task} is absent from the task taxonomy")
    put_json(client, args.output_bucket, f"{runtime}/status.json", {
        "status": "running",
        "phase": PIPELINE_PHASE,
        "pipeline_revision": PIPELINE_REVISION,
        "mode": args.mode,
        "backbone": args.backbone,
        "target": args.target,
        "task": args.task,
        "task_condition": condition,
        "started_at_utc": utc_now(),
        "device": str(device),
    })
    channel_info = load_channel_info(
        client, args.input_bucket, args.channel_info_key, args.work_dir,
    )
    target_frame = download_targets(client, args.targets_uri, args.target, args.work_dir)
    cohort_sha256 = None
    if args.cohort_uri:
        cohort_frame, cohort_sha256 = download_table(
            client, args.cohort_uri, args.work_dir, "cohort_manifest",
        )
        if cohort_frame.participant_id.duplicated().any():
            raise ValueError("Cohort manifest contains duplicate participant_id values")
        target_frame = target_frame[
            target_frame.participant_id.isin(set(cohort_frame.participant_id))
        ].copy()

    signals = list_signal_keys(client, args.input_bucket, args.input_prefix, args.task)
    minimum_required_epochs = max(2, args.train_epochs_per_subject, args.eval_epochs_per_subject)
    target_subjects = set(target_frame.participant_id)
    eligibility_rows = [
        {
            "participant_id": participant_id,
            "source_key": metadata["source_key"],
            "epoch_count": metadata["epoch_count"],
            "minimum_required_epochs": minimum_required_epochs,
            "target_available": participant_id in target_subjects,
            "cohort_required": bool(args.cohort_uri),
            "eligible": (
                participant_id in target_subjects
                and metadata["epoch_count"] >= minimum_required_epochs
            ),
        }
        for participant_id, metadata in sorted(signals.items())
    ]
    upload_frame(
        client, pd.DataFrame(eligibility_rows), args.output_bucket,
        f"{base}/eligibility_audit.csv", args.work_dir,
    )
    eligible_signals = {
        participant_id: {
            "source_key": metadata["source_key"],
            "source_size": metadata["source_size"],
            "source_etag": metadata["source_etag"],
            "epoch_count": metadata["epoch_count"],
        }
        for participant_id, metadata in signals.items()
        if metadata["epoch_count"] >= minimum_required_epochs
    }
    target_frame = target_frame[
        target_frame.participant_id.isin(eligible_signals)
    ].copy()
    target_frame = select_subjects(target_frame, args.target, args.max_subjects, args.seed)
    records = [
        SubjectRecord(
            participant_id=str(row.participant_id),
            target=float(getattr(row, args.target)),
            **eligible_signals[str(row.participant_id)],
        )
        for row in target_frame.itertuples(index=False)
    ]
    if len(records) < args.outer_folds * 4:
        raise RuntimeError(f"Only {len(records)} eligible subjects for {args.target}")

    record_index = {record.participant_id: index for index, record in enumerate(records)}
    split_manifest_sha256 = None
    split_rows: list[dict[str, Any]] = []
    if args.split_manifest_uri:
        supplied_splits, split_manifest_sha256 = download_table(
            client, args.split_manifest_uri, args.work_dir, "split_manifest_supplied",
        )
        required_columns = {"fold", "split", "participant_id"}
        if not required_columns.issubset(supplied_splits.columns):
            raise ValueError("Split manifest requires fold, split and participant_id columns")
        supplied_splits = supplied_splits[list(required_columns)].copy()
        supplied_splits["fold"] = pd.to_numeric(
            supplied_splits["fold"], errors="raise",
        ).astype(int)
        if supplied_splits.duplicated(["fold", "split", "participant_id"]).any():
            raise ValueError("Split manifest contains duplicate rows")
        expected_ids = set(record_index)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(1, args.outer_folds + 1):
            fold_frame = supplied_splits[supplied_splits.fold == fold]
            train_ids = set(
                fold_frame.loc[fold_frame.split == "outer_train", "participant_id"]
            )
            test_ids = set(
                fold_frame.loc[fold_frame.split == "outer_test", "participant_id"]
            )
            if train_ids & test_ids or train_ids | test_ids != expected_ids:
                raise ValueError(f"Invalid participant partition in supplied fold {fold}")
            train = np.asarray(sorted(record_index[value] for value in train_ids), dtype=int)
            test = np.asarray(sorted(record_index[value] for value in test_ids), dtype=int)
            splits.append((train, test))
            for split_name, ids in (("outer_train", train_ids), ("outer_test", test_ids)):
                split_rows.extend(
                    {"fold": fold, "split": split_name, "participant_id": value}
                    for value in sorted(ids)
                )
        unexpected_folds = set(supplied_splits.fold) - set(range(1, args.outer_folds + 1))
        unexpected_splits = set(supplied_splits.split) - {"outer_train", "outer_test"}
        if unexpected_folds or unexpected_splits:
            raise ValueError("Split manifest contains unexpected fold or split values")
    else:
        outer = KFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        splits = list(outer.split(np.arange(len(records))))
        for fold, (train, test) in enumerate(splits, start=1):
            for split_name, indices in (("outer_train", train), ("outer_test", test)):
                split_rows.extend(
                    {
                        "fold": fold,
                        "split": split_name,
                        "participant_id": records[int(index)].participant_id,
                    }
                    for index in indices
                )
    upload_frame(
        client, pd.DataFrame(split_rows), args.output_bucket,
        f"{base}/split_manifest.csv", args.work_dir,
    )

    representation = {
        "cbramod": "pre_projection_latent",
        "biot": "official_encoder_output",
        "labram_128_spatial": "official_encoder_output_128_spatial",
        "eegpt_62_canonical": "official_encoder_output_62_canonical",
        "eegpt_128_spatial": "adapted_encoder_output_128_spatial",
    }[args.backbone]
    preprocessing = {
        "cbramod": "stored_volts_to_uV_then_10x1s_patches",
        "biot": f"16_bipolar_channels_{args.biot_normalization}",
        "labram_128_spatial": "stored_volts_to_uV_then_128_channel_spatial_mapping",
        "eegpt_62_canonical": "62_canonical_channels_0.1_scale_three_4s_windows_256Hz",
        "eegpt_128_spatial": "128_spatial_adapter_0.1_scale_three_4s_windows_256Hz",
    }[args.backbone]
    put_json(client, args.output_bucket, f"{base}/architecture_audit.json", {
        "phase": PIPELINE_PHASE,
        "pipeline_revision": PIPELINE_REVISION,
        "backbone": args.backbone,
        "adapter_class": "EncoderAdapter",
        "representation": representation,
        "preprocessing": preprocessing,
        "numeric_precision": (
            "full_fp32" if args.backbone == "labram_128_spatial"
            else "automatic_mixed_precision_fp16"
        ),
        "activation_memory_strategy": (
            "per_epoch_nonreentrant_checkpointing"
            if args.backbone.startswith("eegpt_") else "standard_forward"
        ),
        "train_epoch_chunk_size": args.train_epoch_chunk_size,
        "cuda_allocator_configuration": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "biot_normalization": args.biot_normalization,
        "base_checkpoint_name": checkpoint_path(args).name,
        "base_checkpoint_sha256": args.base_checkpoint_sha256,
        "progressive_unfreezing": {
            "head_warmup_epochs": args.warmup_epochs,
            "full_unfreeze_epoch_zero_based": args.full_unfreeze_epoch,
            "partial_block_count": args.partial_block_count,
        },
    })
    put_json(client, args.output_bucket, f"{base}/run_config.json", {
        "phase": PIPELINE_PHASE,
        "mode": args.mode,
        "backbone": args.backbone,
        "pipeline_revision": PIPELINE_REVISION,
        "target_requested": args.target,
        "target_loaded": args.target,
        "task": args.task,
        "task_condition": condition,
        "subjects": len(records),
        "outer_folds": args.outer_folds,
        "max_epochs": args.max_epochs,
        "warmup_epochs": args.warmup_epochs,
        "full_unfreeze_epoch": args.full_unfreeze_epoch,
        "partial_block_count": args.partial_block_count,
        "minimum_required_epochs": minimum_required_epochs,
        "signal_subjects_total": len(signals),
        "signal_subjects_with_target": sum(subject in target_subjects for subject in signals),
        "subjects_excluded_insufficient_epochs": sum(
            subject in target_subjects and metadata["epoch_count"] < minimum_required_epochs
            for subject, metadata in signals.items()
        ),
        "train_epochs_per_subject_minimum": args.train_epochs_per_subject,
        "eval_epochs_per_subject": args.eval_epochs_per_subject,
        "eval_epoch_chunk_size": args.eval_epoch_chunk_size,
        "subject_batch_size": args.subject_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_epoch_chunk_size": args.train_epoch_chunk_size,
        "head_learning_rate": args.head_learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "full_encoder_learning_rate": args.full_encoder_learning_rate,
        "minimum_training_coverage": args.minimum_training_coverage,
        "minimum_prediction_std_ratio": args.minimum_prediction_std_ratio,
        "biot_normalization": args.biot_normalization,
        "base_checkpoint_sha256": args.base_checkpoint_sha256,
        "cohort_uri": args.cohort_uri,
        "cohort_sha256": cohort_sha256,
        "split_manifest_uri": args.split_manifest_uri,
        "split_manifest_sha256": split_manifest_sha256,
        "seed": args.seed,
        "statistical_unit": "participant",
        "input_set": "eeg_only",
    })

    store = SignalStore(client, args.input_bucket, args.work_dir / "signal-cache")
    fold_results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    for fold, (train, test) in enumerate(splits, start=1):
        done_key = f"{base}/done/fold-{fold:02d}.json"
        checkpoint_key = f"{base}/folds/fold-{fold:02d}/checkpoint_delta.pt"
        existing = get_json_or_none(client, args.output_bucket, done_key)
        if existing is not None and object_exists(
            client, args.output_bucket, checkpoint_key,
        ):
            fold_results.append(existing)
            continue
        result, predictions, history, checkpoint = run_fold(
            args, client, records, train, test, fold, channel_info, store, device,
        )
        fold_results.append(result)
        prediction_frames.append(predictions)
        history_frames.append(history)
        upload_frame(
            client, predictions, args.output_bucket,
            f"{base}/folds/fold-{fold:02d}/predictions.csv", args.work_dir,
        )
        upload_frame(
            client, history, args.output_bucket,
            f"{base}/folds/fold-{fold:02d}/training_history_and_epoch_audit.csv",
            args.work_dir,
        )
        upload_checkpoint(client, checkpoint, args.output_bucket, checkpoint_key, args.work_dir)
        put_json(client, args.output_bucket, f"{base}/folds/fold-{fold:02d}/metrics.json", result)
        put_json(client, args.output_bucket, done_key, result)
        put_json(client, args.output_bucket, f"{runtime}/progress.json", {
            "status": "running",
            "completed_folds": fold,
            "expected_folds": args.outer_folds,
            "elapsed_seconds": time.time() - started,
        })

    metrics_frame = pd.DataFrame(fold_results).sort_values("fold")
    all_prediction_frames = [
        download_frame(
            client, args.output_bucket,
            f"{base}/folds/fold-{fold:02d}/predictions.csv", args.work_dir,
        )
        for fold in range(1, args.outer_folds + 1)
    ]
    all_predictions = pd.concat(all_prediction_frames, ignore_index=True)
    if len(all_predictions) != len(records) or all_predictions.participant_id.duplicated().any():
        raise RuntimeError("Out-of-fold predictions do not contain exactly one row per participant")
    upload_frame(
        client, all_predictions, args.output_bucket,
        f"{base}/predictions_all_folds.csv", args.work_dir,
    )
    all_history_frames = [
        download_frame(
            client, args.output_bucket,
            f"{base}/folds/fold-{fold:02d}/training_history_and_epoch_audit.csv",
            args.work_dir,
        )
        for fold in range(1, args.outer_folds + 1)
    ]
    upload_frame(
        client, pd.concat(all_history_frames, ignore_index=True),
        args.output_bucket, f"{base}/training_history_and_epoch_audit_all_folds.csv",
        args.work_dir,
    )
    upload_frame(
        client, metrics_frame, args.output_bucket,
        f"{base}/metrics_by_fold.csv", args.work_dir,
    )
    if prediction_frames:
        upload_frame(
            client, pd.concat(prediction_frames, ignore_index=True), args.output_bucket,
            f"{base}/predictions_new_folds.csv", args.work_dir,
        )
    if history_frames:
        upload_frame(
            client, pd.concat(history_frames, ignore_index=True), args.output_bucket,
            f"{base}/training_history_and_epoch_audit_new_folds.csv", args.work_dir,
        )

    metric_names = ["r2", "mae", "rmse", "pearson", "spearman"]
    global_diagnostics = prediction_diagnostics(
        all_predictions.y_true.to_numpy(dtype=float),
        all_predictions.y_pred.to_numpy(dtype=float),
    )
    global_diagnostics["prediction_collapse_detected"] = prediction_collapse_detected(
        global_diagnostics, len(all_predictions), args.minimum_prediction_std_ratio,
    )
    expected_full_evaluation = args.eval_epochs_per_subject == 0
    integrity_gate_passed = bool(
        len(metrics_frame) == args.outer_folds
        and (metrics_frame.participant_overlap == 0).all()
        and (metrics_frame.encoder_parameter_max_delta > 0).all()
        and (metrics_frame.maximum_trainable_gradient_norm > 0).all()
        and metrics_frame.full_stage_reached.all()
        and (metrics_frame.training_epoch_coverage_min >= args.minimum_training_coverage).all()
        and (
            not expected_full_evaluation
            or (metrics_frame.evaluation_epoch_coverage_min >= 1.0 - 1e-12).all()
        )
        and np.isfinite(metrics_frame[metric_names].to_numpy(dtype=float)).all()
        and np.isfinite(
            metrics_frame.maximum_trainable_gradient_norm.to_numpy(dtype=float)
        ).all()
        and (metrics_frame.nonfinite_gradient_fraction <= 0.05).all()
        and (
            metrics_frame.checkpoint_format == "base-plus-full-encoder-delta-v2"
        ).all()
        and condition != "unknown"
    )
    if args.mode == "smoke":
        # Small smoke folds can have low variance by chance. Assess collapse
        # on all out-of-fold predictions while retaining fold diagnostics.
        prediction_quality_gate_passed = bool(
            not global_diagnostics["prediction_collapse_detected"]
        )
    else:
        prediction_quality_gate_passed = bool(
            not metrics_frame.prediction_collapse_detected.any()
            and not global_diagnostics["prediction_collapse_detected"]
        )
    smoke_passed = integrity_gate_passed and prediction_quality_gate_passed
    if not integrity_gate_passed:
        summary_status = "failed_integrity_gate"
    elif prediction_quality_gate_passed:
        summary_status = "succeeded"
    elif args.mode == "full":
        # A weak or constant predictor is a scientifically relevant negative
        # result in the full benchmark. Preserve all metrics and let the batch
        # continue, while making the warning impossible to overlook.
        summary_status = "succeeded_with_prediction_quality_warning"
    else:
        summary_status = "failed_quality_gate"
    summary = {
        "status": summary_status,
        "phase": PIPELINE_PHASE,
        "mode": args.mode,
        "backbone": args.backbone,
        "target": args.target,
        "task": args.task,
        "task_condition": condition,
        "subjects": len(records),
        "completed_folds": len(metrics_frame),
        "expected_folds": args.outer_folds,
        "pipeline_revision": PIPELINE_REVISION,
        "out_of_fold_prediction_rows": len(all_predictions),
        "checkpoint_count": args.outer_folds,
        "base_checkpoint_sha256": args.base_checkpoint_sha256,
        "nonfinite_gradient_steps_total": int(metrics_frame.nonfinite_gradient_steps.sum()),
        "metrics_mean": {
            name: float(metrics_frame[name].mean()) for name in metric_names
        },
        "metrics_std": {
            name: float(metrics_frame[name].std(ddof=1)) for name in metric_names
        },
        "out_of_fold_prediction_diagnostics": global_diagnostics,
        "training_epoch_coverage_min": float(
            metrics_frame.training_epoch_coverage_min.min()
        ),
        "evaluation_epoch_coverage_min": float(
            metrics_frame.evaluation_epoch_coverage_min.min()
        ),
        "participant_leakage_detected": False,
        "multiple_epochs_used": True,
        "full_encoder_stage_reached": bool(metrics_frame.full_stage_reached.all()),
        "encoder_updated": bool((metrics_frame.encoder_parameter_max_delta > 0).all()),
        "scientific_integrity_gate_passed": integrity_gate_passed,
        "prediction_quality_gate_passed": prediction_quality_gate_passed,
        "smoke_quality_gate_passed": smoke_passed,
        "elapsed_seconds": time.time() - started,
        "finished_at_utc": utc_now(),
    }
    put_json(client, args.output_bucket, f"{base}/summary.json", summary)
    put_json(client, args.output_bucket, f"{runtime}/status.json", summary)
    if not integrity_gate_passed or (args.mode == "smoke" and not smoke_passed):
        raise RuntimeError("E3 quality gate failed")
    return 0

def main() -> int:
    args = parse_args()
    started = time.time()
    try:
        return main_work(args)
    except (ClientError, OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        error = {
            "status": "failed", "phase": PIPELINE_PHASE, "backbone": getattr(args, "backbone", None),
            "target": getattr(args, "target", None), "type": type(exc).__name__,
            "message": str(exc), "elapsed_seconds": time.time() - started,
            "traceback": traceback.format_exc(limit=20),
            "finished_at_utc": utc_now(),
        }
        try:
            client = s3_client(args.region)
            base = args.output_prefix.strip("/")
            put_json(client, args.output_bucket, f"{base}/runtime/error.json", error)
            put_json(client, args.output_bucket, f"{base}/runtime/status.json", error)
        except (ClientError, OSError, ValueError):
            pass
        print(json.dumps(error, allow_nan=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
