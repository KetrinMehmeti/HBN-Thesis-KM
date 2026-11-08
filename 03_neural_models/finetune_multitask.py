#!/usr/bin/env python3
"""Masked all-task, subject-level fine-tuning smoke for E3.

Every participant contributes every eligible HBN task.  Missing tasks are
represented only through an attention mask, never through a numerical zero
embedding.  The encoder is shared across tasks and the fused participant
representation has four regression heads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from botocore.exceptions import ClientError
from sklearn.model_selection import KFold

import finetune_foundation_models as single


TASKS = tuple(single.TASK_CONDITIONS)
TARGETS = single.TARGETS
PHASE = "E3-multitask-v1"
REVISION = "v1-masked-subject-fusion"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-bucket", required=True)
    p.add_argument("--input-prefix", default="ica_qamad_ready")
    p.add_argument("--channel-info-key", required=True)
    p.add_argument("--targets-uri", required=True)
    p.add_argument("--output-bucket", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--backbone", choices=single.BACKBONES, required=True)
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--max-subjects", type=int, default=16)
    p.add_argument("--outer-folds", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--full-unfreeze-epoch", type=int, default=2)
    p.add_argument("--partial-block-count", type=int, default=4)
    p.add_argument("--train-epochs-per-task", type=int, default=2)
    p.add_argument("--eval-epochs-per-task", type=int, default=1)
    p.add_argument("--train-epoch-chunk-size", type=int, default=1)
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--head-learning-rate", type=float, default=1e-3)
    p.add_argument("--encoder-learning-rate", type=float, default=3e-5)
    p.add_argument("--full-encoder-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--minimum-tasks-per-subject", type=int, default=2)
    p.add_argument("--minimum-epochs-per-task", type=int, default=2)
    p.add_argument("--minimum-prediction-std-ratio", type=float, default=0.03)
    p.add_argument("--biot-normalization", choices=("official_volts", "q95_ablation"), default="official_volts")
    p.add_argument("--cbramod-weights", type=Path)
    p.add_argument("--biot-weights", type=Path)
    p.add_argument("--labram-weights", type=Path)
    p.add_argument("--eegpt-weights", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--work-dir", type=Path, required=True)
    return p.parse_args()


def validate_args(a: argparse.Namespace) -> tuple[str, ...]:
    tasks = tuple(value.strip() for value in a.tasks.split(",") if value.strip())
    if not tasks or any(value not in TASKS for value in tasks):
        raise ValueError("tasks must be a non-empty subset of the HBN task taxonomy")
    if a.max_subjects < a.outer_folds * 4 or a.outer_folds < 2:
        raise ValueError("smoke requires at least four subjects per outer fold")
    if not 0 <= a.warmup_epochs < a.full_unfreeze_epoch < a.max_epochs:
        raise ValueError("Expected warmup-epochs < full-unfreeze-epoch < max-epochs")
    if a.train_epochs_per_task < 2 or a.eval_epochs_per_task < 1:
        raise ValueError("Smoke requires at least two train epochs and one evaluation epoch per task")
    if a.minimum_tasks_per_subject < 2:
        raise ValueError("At least two tasks per subject are required to validate masking")
    if not 0 < a.minimum_prediction_std_ratio < 1:
        raise ValueError("minimum-prediction-std-ratio must be between zero and one")
    required = {
        "cbramod": a.cbramod_weights, "biot": a.biot_weights,
        "labram_128_spatial": a.labram_weights,
        "eegpt_62_canonical": a.eegpt_weights,
        "eegpt_128_spatial": a.eegpt_weights,
    }[a.backbone]
    if required is None or not required.is_file():
        raise ValueError(f"Missing checkpoint for {a.backbone}: {required}")
    if single.sha256_file(required) != single.EXPECTED_CHECKPOINT_SHA256[a.backbone]:
        raise ValueError("Pretrained checkpoint checksum mismatch")
    return tasks


def download_all_targets(client, uri: str, work_dir: Path) -> pd.DataFrame:
    bucket, key = single.parse_s3_uri(uri)
    path = work_dir / "multitask_targets.parquet"
    client.download_file(bucket, key, str(path), Config=single.TRANSFER_CONFIG)
    try:
        frame = pd.read_parquet(path, columns=["participant_id", *TARGETS])
    finally:
        path.unlink(missing_ok=True)
    frame["participant_id"] = frame.participant_id.astype(str)
    for target in TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=list(TARGETS)).drop_duplicates("participant_id")
    if len(frame) < 8 or any(frame[target].nunique() < 8 for target in TARGETS):
        raise RuntimeError("Insufficient non-constant psychopathology targets")
    return frame.reset_index(drop=True)


def list_task_keys_fast(client, bucket: str, prefix: str, task: str) -> dict[str, dict[str, Any]]:
    """List signals cheaply; epoch headers are read only for selected subjects."""
    found: dict[str, dict[str, Any]] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix.strip('/')}/{task}/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".npy"):
                participant_id = Path(key).stem
                found[participant_id] = {
                    "source_key": key, "source_size": int(item["Size"]),
                    "source_etag": str(item.get("ETag", "")).strip('"'),
                }
    return found


@dataclass(frozen=True)
class MultiSubjectRecord:
    participant_id: str
    targets: np.ndarray
    tasks: dict[str, single.SubjectRecord]


def build_records(
    a: argparse.Namespace, client, target_frame: pd.DataFrame, tasks: tuple[str, ...], store: single.SignalStore,
) -> tuple[list[MultiSubjectRecord], pd.DataFrame]:
    task_keys = {task: list_task_keys_fast(client, a.input_bucket, a.input_prefix, task) for task in tasks}
    availability = pd.DataFrame({
        "participant_id": target_frame.participant_id,
        **{task: target_frame.participant_id.map(lambda pid: pid in task_keys[task]) for task in tasks},
    })
    availability["listed_task_count"] = availability[list(tasks)].sum(axis=1)
    candidates = target_frame.merge(availability[["participant_id", "listed_task_count"]], on="participant_id")
    candidates = candidates[candidates.listed_task_count >= a.minimum_tasks_per_subject].copy()
    candidates = single.select_subjects(candidates, "attention", max(a.max_subjects * 4, a.max_subjects), a.seed)
    rows: list[dict[str, Any]] = []
    records: list[MultiSubjectRecord] = []
    for row in candidates.itertuples(index=False):
        subject_tasks: dict[str, single.SubjectRecord] = {}
        for task in tasks:
            meta = task_keys[task].get(str(row.participant_id))
            if meta is None:
                continue
            candidate = single.SubjectRecord(str(row.participant_id), 0.0, epoch_count=0, **meta)
            try:
                count = len(store.load(candidate))
            except Exception as exc:
                rows.append({"participant_id": row.participant_id, "task": task, "eligible": False, "reason": type(exc).__name__})
                continue
            eligible = count >= a.minimum_epochs_per_task
            rows.append({"participant_id": row.participant_id, "task": task, "eligible": eligible, "epoch_count": count, "reason": "" if eligible else "insufficient_epochs"})
            if eligible:
                subject_tasks[task] = single.SubjectRecord(str(row.participant_id), 0.0, epoch_count=count, **meta)
        if len(subject_tasks) >= a.minimum_tasks_per_subject:
            records.append(MultiSubjectRecord(
                participant_id=str(row.participant_id),
                targets=np.asarray([getattr(row, target) for target in TARGETS], dtype=np.float32),
                tasks=subject_tasks,
            ))
        if len(records) >= a.max_subjects:
            break
    if len(records) < a.outer_folds * 4:
        raise RuntimeError(f"Only {len(records)} subjects have {a.minimum_tasks_per_subject}+ valid tasks")
    return records, pd.DataFrame(rows)


class MaskedTaskFusionRegressor(torch.nn.Module):
    def __init__(self, adapter: single.EncoderAdapter, task_order: tuple[str, ...], fusion_dim: int):
        super().__init__()
        self.adapter = adapter
        self.task_order = task_order
        self.task_index = {task: index for index, task in enumerate(task_order)}
        self.project = torch.nn.Sequential(torch.nn.LayerNorm(adapter.feature_dim), torch.nn.Linear(adapter.feature_dim, fusion_dim))
        self.task_embedding = torch.nn.Embedding(len(task_order), fusion_dim)
        self.score = torch.nn.Sequential(torch.nn.Linear(fusion_dim, fusion_dim), torch.nn.Tanh(), torch.nn.Linear(fusion_dim, 1))
        self.norm = torch.nn.LayerNorm(fusion_dim)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(fusion_dim, len(TARGETS)))

    def forward_subject(self, inputs: list[tuple[str, torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not inputs:
            raise ValueError("A subject must contribute at least one valid task")
        slots = torch.zeros(len(self.task_order), self.project[-1].out_features, device=inputs[0][1].device)
        mask = torch.zeros(len(self.task_order), dtype=torch.bool, device=slots.device)
        for task, tensor, epochs in inputs:
            feature = self.adapter(tensor, 1, epochs).reshape(1, -1)
            position = self.task_index[task]
            token = torch.tensor([position], device=slots.device)
            slots[position] = (self.project(feature) + self.task_embedding(token)).reshape(-1)
            mask[position] = True
        weights = self.score(slots).reshape(-1).masked_fill(~mask, -torch.inf)
        attention = torch.softmax(weights, dim=0)
        fused = self.norm(torch.sum(attention[:, None] * slots, dim=0, keepdim=True))
        return self.head(fused).reshape(-1), mask


def subject_inputs(
    model: MaskedTaskFusionRegressor, record: MultiSubjectRecord, store: single.SignalStore,
    epochs_per_task: int, rng: np.random.Generator, deterministic: bool, device: torch.device,
    rotation_step: int | None = None,
) -> tuple[list[tuple[str, torch.Tensor, int]], dict[str, list[int]]]:
    inputs: list[tuple[str, torch.Tensor, int]] = []
    audit: dict[str, list[int]] = {}
    for task in model.task_order:
        task_record = record.tasks.get(task)
        if task_record is None:
            continue
        values, indices = store.sample(task_record, epochs_per_task, rng, deterministic, rotation_step=rotation_step, rotation_seed=97)
        tensor, counts = model.adapter.preprocess([values])
        if counts != [len(values)]:
            raise RuntimeError("Task epoch count changed during preprocessing")
        inputs.append((task, tensor.to(device, non_blocking=True), len(values)))
        audit[task] = indices
    return inputs, audit


def stage(epoch: int, a: argparse.Namespace) -> str:
    return "head_warmup" if epoch < a.warmup_epochs else ("partial_encoder" if epoch < a.full_unfreeze_epoch else "full_encoder")


def metrics_for_targets(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, target in enumerate(TARGETS):
        for name, value in single.metric_dict(y_true[:, index], y_pred[:, index]).items():
            result[f"{target}_{name}"] = value
    return result


def run_fold(a: argparse.Namespace, records: list[MultiSubjectRecord], train: np.ndarray, test: np.ndarray, fold: int, channel_info, store, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(a.seed + fold * 997)
    if set(train) & set(test):
        raise RuntimeError("Participant leakage detected")
    y = np.stack([record.targets for record in records])
    mean, std = y[train].mean(axis=0), y[train].std(axis=0)
    if np.any(std < 1e-8):
        raise RuntimeError("Near-constant train target")
    adapter = single.load_adapter(a, channel_info, device)
    model = MaskedTaskFusionRegressor(adapter, tuple(a.task_order), a.fusion_dim).to(device)
    reference = single.encoder_reference_state(model)
    current_stage = ""
    optimizer = None
    max_grad, grad_steps, nonfinite = 0.0, 0, 0
    history: list[dict[str, Any]] = []
    for epoch in range(a.max_epochs):
        now = stage(epoch, a)
        if now != current_stage:
            single.set_trainable_stage(model, now, a.partial_block_count)
            optimizer = single.make_optimizer(model, a, now)
            current_stage = now
        assert optimizer is not None
        model.train()
        losses: list[float] = []
        for raw_index in rng.permutation(train):
            index = int(raw_index)
            optimizer.zero_grad(set_to_none=True)
            inputs, _ = subject_inputs(model, records[index], store, a.train_epochs_per_task, rng, False, device, rotation_step=epoch)
            prediction, _ = model.forward_subject(inputs)
            target = torch.tensor((y[index] - mean) / std, dtype=torch.float32, device=device)
            loss = torch.nn.functional.smooth_l1_loss(prediction.float(), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite multi-task loss")
            loss.backward()
            encoder_norms = [p.grad.detach().float().norm(2) for p in model.adapter.encoder.parameters() if p.requires_grad and p.grad is not None]
            encoder_gradient = float(torch.linalg.vector_norm(torch.stack(encoder_norms)).detach().cpu()) if encoder_norms else 0.0
            total_gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
            if now != "head_warmup":
                max_grad, grad_steps, nonfinite = single.update_gradient_diagnostics(max_grad, grad_steps, nonfinite, total_gradient, encoder_gradient)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"fold": fold, "epoch": epoch + 1, "stage": now, "mean_train_loss": float(np.mean(losses)), "task_fusion": "masked_attention", "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_index in test:
            index = int(raw_index)
            inputs, selected = subject_inputs(model, records[index], store, a.eval_epochs_per_task, rng, True, device)
            standard, mask = model.forward_subject(inputs)
            pred = standard.float().cpu().numpy() * std + mean
            rows.append({"fold": fold, "participant_id": records[index].participant_id, **{f"y_true_{t}": float(y[index, n]) for n, t in enumerate(TARGETS)}, **{f"y_pred_{t}": float(pred[n]) for n, t in enumerate(TARGETS)}, "available_tasks": json.dumps([task for task in a.task_order if bool(mask[model.task_index[task]])]), "task_mask": json.dumps(mask.cpu().tolist()), "selected_epoch_indices": json.dumps(selected)})
    predictions = pd.DataFrame(rows)
    true = predictions[[f"y_true_{t}" for t in TARGETS]].to_numpy(float)
    pred = predictions[[f"y_pred_{t}" for t in TARGETS]].to_numpy(float)
    deltas = []
    state = model.adapter.encoder.state_dict()
    for name, initial in reference.items():
        current = state[name.removeprefix("adapter.encoder.")].detach().float().cpu()
        deltas.append(float(torch.max(torch.abs(current - initial))))
    result = {"fold": fold, "participant_overlap": 0, "encoder_parameter_max_delta": max(deltas, default=0.0), "maximum_trainable_gradient_norm": max_grad, "nonfinite_gradient_steps": nonfinite, "nonfinite_gradient_fraction": nonfinite / max(grad_steps, 1), "full_stage_reached": True, "tasks_in_fusion": len(a.task_order), "masked_task_fusion": True, **metrics_for_targets(true, pred)}
    checkpoint = {"format": "multitask-smoke-full-state-v1", "fold": fold, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "task_order": list(a.task_order), "targets": list(TARGETS)}
    return result, predictions, pd.DataFrame(history), checkpoint


def main_work(a: argparse.Namespace) -> int:
    tasks = validate_args(a)
    a.task_order = tasks
    a.work_dir.mkdir(parents=True, exist_ok=True)
    client = single.s3_client(a.region)
    base = a.output_prefix.strip("/")
    runtime = f"{base}/runtime"
    started = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Multi-task fine-tuning requires CUDA")
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    single.put_json(client, a.output_bucket, f"{runtime}/status.json", {"status": "running", "phase": PHASE, "pipeline_revision": REVISION, "mode": a.mode, "backbone": a.backbone, "tasks": list(tasks), "targets": list(TARGETS), "device": str(device), "started_at_utc": single.utc_now()})
    channel_info = single.load_channel_info(client, a.input_bucket, a.channel_info_key, a.work_dir)
    target_frame = download_all_targets(client, a.targets_uri, a.work_dir)
    store = single.SignalStore(client, a.input_bucket, a.work_dir / "signal-cache")
    records, input_audit = build_records(a, client, target_frame, tasks, store)
    single.upload_frame(client, input_audit, a.output_bucket, f"{base}/input_audit.csv", a.work_dir)
    split_rows: list[dict[str, Any]] = []
    splits = list(KFold(n_splits=a.outer_folds, shuffle=True, random_state=a.seed).split(np.arange(len(records))))
    for fold, (train, test) in enumerate(splits, 1):
        for split_name, values in (("outer_train", train), ("outer_test", test)):
            split_rows.extend({"fold": fold, "split": split_name, "participant_id": records[int(i)].participant_id} for i in values)
    single.upload_frame(client, pd.DataFrame(split_rows), a.output_bucket, f"{base}/split_manifest.csv", a.work_dir)
    single.put_json(client, a.output_bucket, f"{base}/run_config.json", {"phase": PHASE, "pipeline_revision": REVISION, "mode": a.mode, "backbone": a.backbone, "statistical_unit": "participant", "input": "all_available_preprocessed_tasks", "task_order": list(tasks), "task_fusion": "masked_attention", "missing_task_policy": "attention_mask_not_numerical_zero", "targets": list(TARGETS), "subjects": len(records), "outer_folds": a.outer_folds, "max_epochs": a.max_epochs, "train_epochs_per_task": a.train_epochs_per_task, "eval_epochs_per_task": a.eval_epochs_per_task, "minimum_tasks_per_subject": a.minimum_tasks_per_subject, "minimum_epochs_per_task": a.minimum_epochs_per_task, "metrics": ["r2", "mae", "rmse", "pearson", "spearman"]})
    results: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(splits, 1):
        done_key = f"{base}/done/fold-{fold:02d}.json"
        checkpoint_key = f"{base}/folds/fold-{fold:02d}/checkpoint.pt"
        existing = single.get_json_or_none(client, a.output_bucket, done_key)
        if existing and single.object_exists(client, a.output_bucket, checkpoint_key):
            results.append(existing); continue
        result, predictions, history, checkpoint = run_fold(a, records, train, test, fold, channel_info, store, device)
        results.append(result)
        single.upload_frame(client, predictions, a.output_bucket, f"{base}/folds/fold-{fold:02d}/predictions.csv", a.work_dir)
        single.upload_frame(client, history, a.output_bucket, f"{base}/folds/fold-{fold:02d}/training_history.csv", a.work_dir)
        single.upload_checkpoint(client, checkpoint, a.output_bucket, checkpoint_key, a.work_dir)
        single.put_json(client, a.output_bucket, f"{base}/folds/fold-{fold:02d}/metrics.json", result)
        single.put_json(client, a.output_bucket, done_key, result)
        single.put_json(client, a.output_bucket, f"{runtime}/progress.json", {"status": "running", "completed_folds": fold, "expected_folds": a.outer_folds, "elapsed_seconds": time.time() - started})
    metrics = pd.DataFrame(results).sort_values("fold")
    predictions = pd.concat([single.download_frame(client, a.output_bucket, f"{base}/folds/fold-{fold:02d}/predictions.csv", a.work_dir) for fold in range(1, a.outer_folds + 1)], ignore_index=True)
    if predictions.participant_id.duplicated().any() or len(predictions) != len(records):
        raise RuntimeError("Out-of-fold predictions must contain each participant exactly once")
    single.upload_frame(client, metrics, a.output_bucket, f"{base}/metrics_by_fold.csv", a.work_dir)
    single.upload_frame(client, predictions, a.output_bucket, f"{base}/predictions_all_folds.csv", a.work_dir)
    mean_by_target: dict[str, dict[str, float]] = {}
    quality: dict[str, bool] = {}
    for target in TARGETS:
        y_true = predictions[f"y_true_{target}"].to_numpy(float); y_pred = predictions[f"y_pred_{target}"].to_numpy(float)
        diag = single.prediction_diagnostics(y_true, y_pred)
        quality[target] = not single.prediction_collapse_detected(diag, len(predictions), a.minimum_prediction_std_ratio)
        mean_by_target[target] = {name: float(metrics[f"{target}_{name}"].mean()) for name in ("r2", "mae", "rmse", "pearson", "spearman")}
        mean_by_target[target]["prediction_std_ratio"] = float(diag["prediction_std_ratio"])
    integrity = bool(len(metrics) == a.outer_folds and (metrics.participant_overlap == 0).all() and (metrics.encoder_parameter_max_delta > 0).all() and (metrics.maximum_trainable_gradient_norm > 0).all() and np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all())
    summary = {"status": "succeeded" if integrity and all(quality.values()) else "failed_quality_gate", "phase": PHASE, "pipeline_revision": REVISION, "mode": a.mode, "backbone": a.backbone, "tasks": list(tasks), "targets": list(TARGETS), "subjects": len(records), "completed_folds": len(metrics), "expected_folds": a.outer_folds, "metrics_mean_by_target": mean_by_target, "participant_leakage_detected": False, "task_mask_used": True, "multiple_tasks_used": bool(max(len(record.tasks) for record in records) >= 2), "encoder_updated": bool((metrics.encoder_parameter_max_delta > 0).all()), "scientific_integrity_gate_passed": integrity, "prediction_quality_gate_passed_by_target": quality, "smoke_quality_gate_passed": integrity and all(quality.values()), "elapsed_seconds": time.time() - started, "finished_at_utc": single.utc_now()}
    single.put_json(client, a.output_bucket, f"{base}/summary.json", summary)
    single.put_json(client, a.output_bucket, f"{runtime}/status.json", summary)
    if not summary["smoke_quality_gate_passed"]:
        raise RuntimeError("Multi-task smoke quality gate failed")
    return 0


def main() -> int:
    started = time.time(); args = parse_args()
    try:
        return main_work(args)
    except Exception as exc:
        try:
            client = single.s3_client(args.region)
            single.put_json(client, args.output_bucket, f"{args.output_prefix.strip('/')}/runtime/error.json", {"status": "failed", "phase": PHASE, "backbone": args.backbone, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=20), "elapsed_seconds": time.time() - started, "finished_at_utc": single.utc_now()})
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
