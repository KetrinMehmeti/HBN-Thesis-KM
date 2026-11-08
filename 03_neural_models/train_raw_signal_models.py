#!/usr/bin/env python3
"""Subject-level masked multi-task deep-learning benchmark on raw EEG.

Each architecture is trained from random initialization, therefore this is an
end-to-end raw-signal benchmark, not a fine-tuning run. The statistical unit is
always the participant: epochs are pooled within task and tasks are fused with
an attention mask so missing tasks are never encoded as numerical zero values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

import finetune_foundation_models as single
import finetune_multitask as multitask


PHASE = "E4-raw-multitask"
REVISION = "v4-raw-eegconformer-masked-subject-fusion"
TARGETS = single.TARGETS
TASKS = tuple(single.TASK_CONDITIONS)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-bucket", required=True)
    p.add_argument("--input-prefix", default="ica_qamad_ready")
    p.add_argument("--targets-uri", required=True)
    p.add_argument("--output-bucket", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--architecture", choices=("eegnet", "deep4", "eegconformer"), default="eegnet")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--max-subjects", type=int, default=32)
    p.add_argument("--outer-folds", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--train-epochs-per-task", type=int, default=2)
    p.add_argument("--eval-epochs-per-task", type=int, default=2)
    p.add_argument("--minimum-tasks-per-subject", type=int, default=2)
    p.add_argument("--minimum-epochs-per-task", type=int, default=2)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--minimum-prediction-std-ratio", type=float, default=0.03)
    p.add_argument("--frozen-cohort", type=Path, default=None, help="CSV with participant-task rows and fixed outer_fold values")
    p.add_argument("--assigned-fold", type=int, default=0, help="Run one 1-based outer fold; 0 runs every fold")
    p.add_argument("--finalize", action="store_true", help="After this worker's fold, validate and aggregate every fold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--work-dir", type=Path, required=True)
    return p.parse_args()


def validate(a: argparse.Namespace) -> tuple[str, ...]:
    tasks = tuple(value.strip() for value in a.tasks.split(",") if value.strip())
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError("tasks must be a non-empty subset of the HBN task taxonomy")
    if a.outer_folds < 2 or (a.max_subjects != 0 and a.max_subjects < a.outer_folds * 6):
        raise ValueError("Need at least six subjects per outer fold")
    if a.train_epochs_per_task < 2 or a.eval_epochs_per_task < 1:
        raise ValueError("Use at least two training epochs and one evaluation epoch per task")
    if a.minimum_tasks_per_subject < 1:
        raise ValueError("Require at least one available task per subject")
    if a.mode == "smoke" and a.minimum_tasks_per_subject < 2:
        raise ValueError("The smoke test must exercise task masking with at least two tasks per subject")
    if a.mode == "full" and a.frozen_cohort is None:
        raise ValueError("Full runs require a frozen cohort manifest")
    if a.assigned_fold and not 1 <= a.assigned_fold <= a.outer_folds:
        raise ValueError("assigned-fold must be between 1 and outer-folds")
    if a.feature_dim < 8:
        raise ValueError("feature-dim must be at least 8")
    return tasks


class EEGNetEncoder(torch.nn.Module):
    """Compact EEGNet feature extractor, adapted to 128 channels at 200 Hz."""
    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        f1, depth_multiplier = 8, 2
        f2 = feature_dim
        self.temporal = torch.nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
        self.norm1 = torch.nn.BatchNorm2d(f1)
        self.spatial = torch.nn.Conv2d(f1, f1 * depth_multiplier, (128, 1), groups=f1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(f1 * depth_multiplier)
        self.pool1 = torch.nn.AvgPool2d((1, 4))
        self.separable_depthwise = torch.nn.Conv2d(f1 * depth_multiplier, f1 * depth_multiplier, (1, 16), padding=(0, 8), groups=f1 * depth_multiplier, bias=False)
        self.separable_pointwise = torch.nn.Conv2d(f1 * depth_multiplier, f2, 1, bias=False)
        self.norm3 = torch.nn.BatchNorm2d(f2)
        self.pool2 = torch.nn.AvgPool2d((1, 8))
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ELU()
        self.output = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x); x = self.norm1(x)
        x = self.spatial(x); x = self.norm2(x); x = self.activation(x); x = self.pool1(x); x = self.dropout(x)
        x = self.separable_depthwise(x); x = self.separable_pointwise(x); x = self.norm3(x); x = self.activation(x); x = self.pool2(x); x = self.dropout(x)
        return self.output(x).flatten(1)


class EEGNetAdapter(torch.nn.Module):
    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = "eegnet_raw"
        self.feature_dim = feature_dim
        self.encoder = EEGNetEncoder(feature_dim, dropout)

    def preprocess(self, subject_epochs: list[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
        counts = [len(value) for value in subject_epochs]
        stored = np.concatenate(subject_epochs, axis=0).astype(np.float32, copy=False)
        if stored.ndim != 3 or stored.shape[1:] != (128, 2000) or not np.isfinite(stored).all():
            raise ValueError("EEGNet expected finite float32 arrays shaped (epochs, 128, 2000)")
        # Per-epoch and per-channel standardisation is fitted only to the epoch
        # itself, so it has no access to other subjects or held-out labels.
        mean = stored.mean(axis=-1, keepdims=True)
        std = stored.std(axis=-1, keepdims=True)
        array = np.clip((stored - mean) / np.maximum(std, 1e-6), -8.0, 8.0)
        return torch.from_numpy(np.ascontiguousarray(array[:, None, :, :])), counts

    def encode_epochs(self, tensor: torch.Tensor) -> torch.Tensor:
        result = self.encoder(tensor.float())
        if result.ndim != 2 or result.shape[1] != self.feature_dim:
            raise RuntimeError("Unexpected EEGNet feature shape")
        return result

    def forward(self, tensor: torch.Tensor, subject_count: int, epochs_per_subject: int) -> torch.Tensor:
        values = self.encode_epochs(tensor)
        if len(values) != subject_count * epochs_per_subject:
            raise RuntimeError("EEGNet epoch count does not match subject batch")
        return values.reshape(subject_count, epochs_per_subject, -1).mean(dim=1)


class Deep4Encoder(torch.nn.Module):
    """Four-block Deep4/DeepConvNet encoder for 128-channel, 200 Hz EEG."""
    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        def block(in_channels: int, out_channels: int, kernel: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, (1, kernel), bias=False),
                torch.nn.BatchNorm2d(out_channels), torch.nn.ELU(),
                torch.nn.MaxPool2d((1, 3), stride=(1, 3)), torch.nn.Dropout(dropout),
            )
        self.temporal = torch.nn.Conv2d(1, 25, (1, 25), bias=False)
        self.spatial = torch.nn.Conv2d(25, 25, (128, 1), bias=False)
        self.norm = torch.nn.BatchNorm2d(25)
        self.activation = torch.nn.ELU()
        self.pool = torch.nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout = torch.nn.Dropout(dropout)
        self.block2 = block(25, 50, 10)
        self.block3 = block(50, 100, 10)
        self.block4 = block(100, feature_dim, 10)
        self.output = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x); x = self.spatial(x); x = self.norm(x)
        x = self.activation(x); x = self.pool(x); x = self.dropout(x)
        x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.output(x).flatten(1)


class Deep4Adapter(EEGNetAdapter):
    def __init__(self, feature_dim: int, dropout: float) -> None:
        torch.nn.Module.__init__(self)
        self.backbone = "deep4_raw"
        self.feature_dim = feature_dim
        self.encoder = Deep4Encoder(feature_dim, dropout)


class ConformerBlock(torch.nn.Module):
    """Pre-norm self-attention and feed-forward block for EEG tokens."""

    def __init__(self, embedding_dim: int, heads: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(embedding_dim)
        self.attention = torch.nn.MultiheadAttention(
            embedding_dim, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(embedding_dim)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, expansion * embedding_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(expansion * embedding_dim, embedding_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + self.attention_dropout(attended)
        return x + self.feedforward(self.feedforward_norm(x))


class EEGConformerEncoder(torch.nn.Module):
    """EEG Conformer adapted to 128-channel, 200 Hz, ten-second epochs."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        embedding_dim, heads, depth = 40, 10, 6
        self.tokenizer = torch.nn.Sequential(
            torch.nn.Conv2d(1, embedding_dim, (1, 25), bias=False),
            torch.nn.Conv2d(embedding_dim, embedding_dim, (128, 1), bias=False),
            torch.nn.BatchNorm2d(embedding_dim),
            torch.nn.ELU(),
            torch.nn.AvgPool2d((1, 75), stride=(1, 15)),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(embedding_dim, embedding_dim, (1, 1), bias=False),
        )
        self.transformer = torch.nn.Sequential(
            *(ConformerBlock(embedding_dim, heads, dropout) for _ in range(depth))
        )
        self.output_norm = torch.nn.LayerNorm(embedding_dim)
        self.output_projection = torch.nn.Linear(embedding_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        if x.shape[2] != 1:
            raise RuntimeError("EEG Conformer spatial tokenizer did not collapse all 128 channels")
        x = x.squeeze(2).transpose(1, 2)
        x = self.transformer(x)
        x = self.output_norm(x).mean(dim=1)
        return self.output_projection(x)


class EEGConformerAdapter(EEGNetAdapter):
    def __init__(self, feature_dim: int, dropout: float) -> None:
        torch.nn.Module.__init__(self)
        self.backbone = "eegconformer_raw"
        self.feature_dim = feature_dim
        self.encoder = EEGConformerEncoder(feature_dim, dropout)

def subject_inputs(model, record, store, epochs_per_task, rng, deterministic, device, rotation_step=None):
    return multitask.subject_inputs(model, record, store, epochs_per_task, rng, deterministic, device, rotation_step)


def build_records_without_label_selection(a, client, target_frame, tasks, store):
    """Build smoke cohorts from stable IDs and signal availability only."""
    task_keys = {task: multitask.list_task_keys_fast(client, a.input_bucket, a.input_prefix, task) for task in tasks}
    availability = pd.DataFrame({"participant_id": target_frame.participant_id, **{task: target_frame.participant_id.map(lambda pid: pid in task_keys[task]) for task in tasks}})
    availability["listed_task_count"] = availability[list(tasks)].sum(axis=1)
    candidates = target_frame.merge(availability[["participant_id", "listed_task_count"]], on="participant_id")
    candidates = candidates[candidates.listed_task_count >= a.minimum_tasks_per_subject].copy()
    candidates["_id_rank"] = candidates.participant_id.map(lambda pid: hashlib.sha256(f"{a.seed}|{pid}".encode("utf-8")).hexdigest())
    candidates = candidates.sort_values("_id_rank").drop(columns="_id_rank")
    rows, records = [], []
    for row in candidates.itertuples(index=False):
        subject_tasks = {}
        for task in tasks:
            metadata = task_keys[task].get(str(row.participant_id))
            if metadata is None: continue
            candidate = single.SubjectRecord(str(row.participant_id), 0.0, epoch_count=0, **metadata)
            try: count = len(store.load(candidate))
            except Exception as exc:
                rows.append({"participant_id":row.participant_id,"task":task,"eligible":False,"reason":type(exc).__name__}); continue
            eligible = count >= a.minimum_epochs_per_task
            rows.append({"participant_id":row.participant_id,"task":task,"eligible":eligible,"epoch_count":count,"reason":"" if eligible else "insufficient_epochs"})
            if eligible: subject_tasks[task] = single.SubjectRecord(str(row.participant_id), 0.0, epoch_count=count, **metadata)
        if len(subject_tasks) >= a.minimum_tasks_per_subject:
            records.append(multitask.MultiSubjectRecord(participant_id=str(row.participant_id), targets=np.asarray([getattr(row,target) for target in TARGETS], dtype=np.float32), tasks=subject_tasks))
        if a.max_subjects > 0 and len(records) >= a.max_subjects: break
    if len(records) < a.outer_folds * 6: raise RuntimeError(f"Only {len(records)} subjects have {a.minimum_tasks_per_subject}+ valid tasks")
    return records, pd.DataFrame(rows)
def build_records_from_frozen_cohort(a: argparse.Namespace) -> tuple[list[multitask.MultiSubjectRecord], pd.DataFrame, list[int]]:
    """Reconstruct a pre-audited cohort without querying labels or S3 listings."""
    frame = pd.read_csv(a.frozen_cohort)
    required = {"participant_id", "task", "source_key", "source_size", "epoch_count", "outer_fold", *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Frozen cohort is missing columns: {sorted(missing)}")
    frame["participant_id"] = frame.participant_id.astype(str)
    frame["task"] = frame.task.astype(str)
    frame = frame[frame.task.isin(a.task_order)].copy()
    if (frame.epoch_count < a.minimum_epochs_per_task).any():
        raise ValueError("Frozen cohort contains a task below the epoch eligibility threshold")
    allowed_folds = set(range(1, a.outer_folds + 1))
    if not set(frame.outer_fold.astype(int).unique()).issubset(allowed_folds):
        raise ValueError("Frozen cohort has invalid outer-fold labels")
    records, audit, folds = [], [], []
    for participant_id, group in frame.sort_values(["participant_id", "task"]).groupby("participant_id", sort=True):
        if group.outer_fold.nunique() != 1:
            raise ValueError(f"Participant {participant_id} appears in multiple folds")
        target_values = group.iloc[0][list(TARGETS)].to_numpy(dtype=np.float32)
        if not np.isfinite(target_values).all():
            raise ValueError(f"Non-finite target for {participant_id}")
        subject_tasks = {}
        for row in group.itertuples(index=False):
            subject_tasks[row.task] = single.SubjectRecord(str(participant_id), 0.0, str(row.source_key), int(row.source_size), str(getattr(row, "source_etag", "")), int(row.epoch_count))
            audit.append({"participant_id": str(participant_id), "task": row.task, "eligible": True, "epoch_count": int(row.epoch_count), "reason": "frozen_prevalidated_manifest"})
        if len(subject_tasks) < a.minimum_tasks_per_subject:
            continue
        records.append(multitask.MultiSubjectRecord(participant_id=str(participant_id), targets=target_values, tasks=subject_tasks))
        folds.append(int(group.outer_fold.iloc[0]))
    if len(records) < a.outer_folds * 6:
        raise RuntimeError("Frozen cohort is too small for the requested outer CV")
    return records, pd.DataFrame(audit), folds

def _resume_checkpoint(client, bucket: str, key: str, work_dir: Path) -> dict[str, Any] | None:
    """Load a train-only progress checkpoint, if a Spot worker was interrupted."""
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    path = work_dir / (hashlib.sha256(key.encode()).hexdigest() + ".resume.pt")
    client.download_file(bucket, key, str(path), Config=single.TRANSFER_CONFIG)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        path.unlink(missing_ok=True)


def run_fold(a, records, train, test, fold, store, device, client, base):
    if set(train) & set(test):
        raise RuntimeError("Participant leakage detected")
    rng = np.random.default_rng(a.seed + fold * 997)
    y = np.stack([record.targets for record in records])
    mean, std = y[train].mean(axis=0), y[train].std(axis=0)
    if np.any(std < 1e-8):
        raise RuntimeError("Near-constant training target")
    adapter_types = {"eegnet": EEGNetAdapter, "deep4": Deep4Adapter, "eegconformer": EEGConformerAdapter}
    adapter_type = adapter_types[a.architecture]
    adapter = adapter_type(a.feature_dim, a.dropout).to(device)
    model = multitask.MaskedTaskFusionRegressor(adapter, tuple(a.task_order), a.feature_dim).to(device)
    reference = {name: value.detach().float().cpu().clone() for name, value in model.adapter.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    history, max_grad, nonfinite, grad_steps = [], 0.0, 0, 0
    resume_key = f"{base}/folds/fold-{fold:02d}/training_resume.pt"
    resume = _resume_checkpoint(client, a.output_bucket, resume_key, a.work_dir)
    start_epoch, active_order, active_position, active_losses = 0, None, 0, []
    if resume is not None:
        if resume.get("fold") != fold or resume.get("train_indices") != [int(i) for i in train]:
            raise RuntimeError("Training checkpoint does not match the frozen outer-train split")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        history = list(resume.get("history", []))
        max_grad = float(resume.get("max_grad", 0.0)); nonfinite = int(resume.get("nonfinite", 0)); grad_steps = int(resume.get("grad_steps", 0))
        start_epoch = int(resume["epoch"])
        active_order = resume.get("active_order")
        active_position = int(resume.get("active_position", 0)); active_losses = list(resume.get("active_losses", []))
        rng.bit_generator.state = resume["rng_state"]
    model.train()
    for epoch in range(start_epoch, a.max_epochs):
        if active_order is None:
            active_order = [int(i) for i in rng.permutation(train)]
            active_position, active_losses = 0, []
        for position in range(active_position, len(active_order)):
            index = active_order[position]
            optimizer.zero_grad(set_to_none=True)
            inputs, _ = subject_inputs(model, records[index], store, a.train_epochs_per_task, rng, False, device, rotation_step=epoch)
            prediction, _ = model.forward_subject(inputs)
            target = torch.tensor((y[index] - mean) / std, dtype=torch.float32, device=device)
            loss = torch.nn.functional.smooth_l1_loss(prediction.float(), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {a.architecture} loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
            grad_steps += 1
            if not math.isfinite(norm): nonfinite += 1
            else: max_grad = max(max_grad, norm)
            optimizer.step(); active_losses.append(float(loss.detach().cpu()))
            active_position = position + 1
            if active_position % 25 == 0 or active_position == len(active_order):
                payload = {"format": f"{a.architecture}-multitask-train-resume-v1", "fold": fold,
                    "train_indices": [int(i) for i in train], "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                    "active_order": active_order, "active_position": active_position, "active_losses": active_losses,
                    "history": history, "max_grad": max_grad, "nonfinite": nonfinite, "grad_steps": grad_steps,
                    "rng_state": rng.bit_generator.state, "checkpoint_scope": "outer_train_only"}
                single.upload_checkpoint(client, payload, a.output_bucket, resume_key, a.work_dir)
                single.put_json(client, a.output_bucket, f"{base}/runtime/fold-{fold:02d}/training_progress.json",
                    {"fold": fold, "epoch": epoch + 1, "train_subjects_completed_in_epoch": active_position,
                     "train_subjects_total_in_epoch": len(active_order), "checkpoint_scope": "outer_train_only"})
        history.append({"fold": fold, "epoch": epoch + 1, "stage": "end_to_end_from_random_initialization", "mean_train_loss": float(np.mean(active_losses)), "task_fusion": "masked_attention"})
        active_order, active_position, active_losses = None, 0, []
    model.eval(); rows = []
    with torch.no_grad():
        for raw_index in test:
            index = int(raw_index)
            inputs, selected = subject_inputs(model, records[index], store, a.eval_epochs_per_task, rng, True, device)
            standardized, mask = model.forward_subject(inputs)
            prediction = standardized.float().cpu().numpy() * std + mean
            rows.append({"fold": fold, "participant_id": records[index].participant_id, **{f"y_true_{t}": float(y[index, n]) for n, t in enumerate(TARGETS)}, **{f"y_pred_{t}": float(prediction[n]) for n, t in enumerate(TARGETS)}, "available_tasks": json.dumps([task for task in a.task_order if bool(mask[model.task_index[task]])]), "task_mask": json.dumps(mask.cpu().tolist()), "selected_epoch_indices": json.dumps(selected)})
    predictions = pd.DataFrame(rows)
    true = predictions[[f"y_true_{t}" for t in TARGETS]].to_numpy(float)
    pred = predictions[[f"y_pred_{t}" for t in TARGETS]].to_numpy(float)
    deltas = [float(torch.max(torch.abs(model.adapter.encoder.state_dict()[name].detach().float().cpu() - initial))) for name, initial in reference.items()]
    result = {"fold": fold, "participant_overlap": 0, "encoder_parameter_max_delta": max(deltas, default=0.0), "maximum_trainable_gradient_norm": max_grad, "nonfinite_gradient_steps": nonfinite, "nonfinite_gradient_fraction": nonfinite / max(grad_steps, 1), "full_stage_reached": True, "tasks_in_fusion": len(a.task_order), "masked_task_fusion": True, **multitask.metrics_for_targets(true, pred)}
    checkpoint = {"format": f"{a.architecture}-multitask-state-v1", "fold": fold, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "task_order": list(a.task_order), "targets": list(TARGETS)}
    return result, predictions, pd.DataFrame(history), checkpoint


def main_work(a: argparse.Namespace) -> int:
    tasks = validate(a); a.task_order = tasks; a.work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError(f"{a.architecture} raw benchmark requires CUDA")
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    client = single.s3_client(a.region); base = a.output_prefix.strip("/"); runtime = f"{base}/runtime"; started = time.time()
    single.put_json(client, a.output_bucket, f"{runtime}/status.json", {"status":"running", "phase":PHASE, "pipeline_revision":REVISION, "mode":a.mode, "backbone":f"{a.architecture}_raw", "started_at_utc":single.utc_now()})
    store = single.SignalStore(client, a.input_bucket, a.work_dir / "signal-cache")
    if a.frozen_cohort is not None:
        records, audit, fold_labels = build_records_from_frozen_cohort(a)
        splits = [(np.flatnonzero(np.asarray(fold_labels) != fold), np.flatnonzero(np.asarray(fold_labels) == fold)) for fold in range(1, a.outer_folds + 1)]
    else:
        targets = multitask.download_all_targets(client, a.targets_uri, a.work_dir)
        records, audit = build_records_without_label_selection(a, client, targets, tasks, store)
        splits = list(KFold(n_splits=a.outer_folds, shuffle=True, random_state=a.seed).split(np.arange(len(records))))
    single.upload_frame(client, audit, a.output_bucket, f"{base}/input_audit.csv", a.work_dir)
    split_rows = [{"fold":fold, "split":name, "participant_id":records[int(i)].participant_id} for fold,(tr,te) in enumerate(splits,1) for name,indices in (("outer_train",tr),("outer_test",te)) for i in indices]
    single.upload_frame(client, pd.DataFrame(split_rows), a.output_bucket, f"{base}/split_manifest.csv", a.work_dir)
    config = {"phase":PHASE,"pipeline_revision":REVISION,"mode":a.mode,"backbone":{"eegnet":"EEGNet","deep4":"Deep4/DeepConvNet","eegconformer":"EEG Conformer"}[a.architecture],"model_initialization":"random","statistical_unit":"participant","input":"all_available_preprocessed_tasks_raw_eeg","array_shape":"(epochs,128,2000)","sampling_hz":200,"channel_order":"E1_to_E128_GSN-HydroCel-128","epoch_normalization":"per_epoch_per_channel_zscore_clipped_8","task_order":list(tasks),"task_fusion":"masked_attention","missing_task_policy":"attention_mask_not_numerical_zero","cohort_selection":"stable_participant_id_hash_and_signal_availability_only","targets":list(TARGETS),"subjects":len(records),"outer_folds":a.outer_folds,"max_epochs":a.max_epochs,"train_epochs_per_task":a.train_epochs_per_task,"eval_epochs_per_task":a.eval_epochs_per_task,"minimum_tasks_per_subject":a.minimum_tasks_per_subject,"frozen_cohort":str(a.frozen_cohort) if a.frozen_cohort else None,"target_specific_outputs":True,"metrics":["r2","mae","rmse","pearson","spearman"]}
    single.put_json(client, a.output_bucket, f"{base}/run_config.json", config)
    results=[]
    fold_plan = [(fold, train, test) for fold, (train, test) in enumerate(splits, 1) if not a.assigned_fold or fold == a.assigned_fold]
    for fold,train,test in fold_plan:
        done_key = f"{base}/done/fold-{fold:02d}.json"
        try:
            existing = json.loads(client.get_object(Bucket=a.output_bucket, Key=done_key)["Body"].read())
            results.append(existing)
            continue
        except client.exceptions.NoSuchKey:
            pass
        result,predictions,history,checkpoint=run_fold(a,records,train,test,fold,store,device,client,base); results.append(result)
        single.upload_frame(client,predictions,a.output_bucket,f"{base}/folds/fold-{fold:02d}/predictions.csv",a.work_dir)
        single.upload_frame(client,history,a.output_bucket,f"{base}/folds/fold-{fold:02d}/training_history.csv",a.work_dir)
        single.upload_checkpoint(client,checkpoint,a.output_bucket,f"{base}/folds/fold-{fold:02d}/checkpoint.pt",a.work_dir)
        single.put_json(client,a.output_bucket,done_key,result)
        single.put_json(client,a.output_bucket,f"{runtime}/progress.json",{"status":"running","completed_folds":len(results),"expected_folds":a.outer_folds,"active_fold":fold,"elapsed_seconds":time.time()-started})
    if a.assigned_fold and not a.finalize:
        worker_summary={"status":"succeeded_fold","phase":PHASE,"pipeline_revision":REVISION,"assigned_fold":a.assigned_fold,"completed_folds":len(results),"expected_folds":a.outer_folds,"elapsed_seconds":time.time()-started,"finished_at_utc":single.utc_now()}
        single.put_json(client,a.output_bucket,f"{base}/folds/fold-{a.assigned_fold:02d}/worker_summary.json",worker_summary)
        single.put_json(client,a.output_bucket,f"{runtime}/status.json",worker_summary)
        return 0
    if a.mode == "full":
        results=[]
        for fold in range(1, a.outer_folds + 1):
            try:
                results.append(json.loads(client.get_object(Bucket=a.output_bucket, Key=f"{base}/done/fold-{fold:02d}.json")["Body"].read()))
            except client.exceptions.NoSuchKey as exc:
                raise RuntimeError(f"Cannot finalize: fold {fold} has no completion marker") from exc
    metrics=pd.DataFrame(results).sort_values("fold"); predictions=pd.concat([single.download_frame(client,a.output_bucket,f"{base}/folds/fold-{fold:02d}/predictions.csv",a.work_dir) for fold in range(1,a.outer_folds+1)],ignore_index=True)
    if predictions.participant_id.duplicated().any() or len(predictions)!=len(records): raise RuntimeError("Out-of-fold predictions must contain each participant once")
    single.upload_frame(client,metrics,a.output_bucket,f"{base}/metrics_by_fold.csv",a.work_dir); single.upload_frame(client,predictions,a.output_bucket,f"{base}/predictions_all_folds.csv",a.work_dir)
    # Keep wide joint-model outputs and also save each target independently.
    for target in TARGETS:
        target_predictions = predictions[["fold", "participant_id", f"y_true_{target}", f"y_pred_{target}", "available_tasks", "task_mask", "selected_epoch_indices"]].rename(columns={f"y_true_{target}": "y_true", f"y_pred_{target}": "y_pred"})
        target_metrics = metrics[["fold", "participant_overlap", "encoder_parameter_max_delta", "maximum_trainable_gradient_norm", "nonfinite_gradient_steps", "nonfinite_gradient_fraction", f"{target}_r2", f"{target}_mae", f"{target}_rmse", f"{target}_pearson", f"{target}_spearman"]].rename(columns={f"{target}_r2": "r2", f"{target}_mae": "mae", f"{target}_rmse": "rmse", f"{target}_pearson": "pearson", f"{target}_spearman": "spearman"})
        single.upload_frame(client,target_predictions,a.output_bucket,f"{base}/targets/{target}/predictions_all_folds.csv",a.work_dir)
        single.upload_frame(client,target_metrics,a.output_bucket,f"{base}/targets/{target}/metrics_by_fold.csv",a.work_dir)
    quality={}; mean_by_target={}
    for target in TARGETS:
        diagnostic=single.prediction_diagnostics(predictions[f"y_true_{target}"],predictions[f"y_pred_{target}"]); quality[target]=not single.prediction_collapse_detected(diagnostic,len(predictions),a.minimum_prediction_std_ratio)
        mean_by_target[target]={metric:float(metrics[f"{target}_{metric}"].mean()) for metric in ("r2","mae","rmse","pearson","spearman")}; mean_by_target[target]["prediction_std_ratio"]=float(diagnostic["prediction_std_ratio"])
    integrity=bool(len(metrics)==a.outer_folds and (metrics.participant_overlap==0).all() and (metrics.encoder_parameter_max_delta>0).all() and (metrics.maximum_trainable_gradient_norm>0).all() and (metrics.nonfinite_gradient_steps==0).all() and np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all())
    summary={"status":"succeeded" if integrity and all(quality.values()) else "failed_quality_gate","phase":PHASE,"pipeline_revision":REVISION,"mode":a.mode,"backbone":{"eegnet":"EEGNet","deep4":"Deep4/DeepConvNet","eegconformer":"EEG Conformer"}[a.architecture],"model_initialization":"random","subjects":len(records),"tasks":list(tasks),"targets":list(TARGETS),"completed_folds":len(metrics),"expected_folds":a.outer_folds,"metrics_mean_by_target":mean_by_target,"participant_leakage_detected":False,"task_mask_used":True,"multiple_tasks_used":bool(max(len(r.tasks) for r in records)>=2),"encoder_updated":bool((metrics.encoder_parameter_max_delta>0).all()),"scientific_integrity_gate_passed":integrity,"prediction_quality_gate_passed_by_target":quality,"smoke_quality_gate_passed":integrity and all(quality.values()),"elapsed_seconds":time.time()-started,"finished_at_utc":single.utc_now()}
    single.put_json(client,a.output_bucket,f"{base}/summary.json",summary); single.put_json(client,a.output_bucket,f"{runtime}/status.json",summary)
    if not summary["smoke_quality_gate_passed"]: raise RuntimeError(f"{a.architecture} smoke quality gate failed")
    return 0


def main() -> int:
    started=time.time(); a=args()
    try: return main_work(a)
    except Exception as exc:
        try:
            client=single.s3_client(a.region); single.put_json(client,a.output_bucket,f"{a.output_prefix.strip('/')}/runtime/error.json",{"status":"failed","phase":PHASE,"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(limit=20),"elapsed_seconds":time.time()-started,"finished_at_utc":single.utc_now()})
        except Exception: pass
        raise

if __name__ == "__main__": raise SystemExit(main())




