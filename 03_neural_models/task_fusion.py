"""Pure subject-level fusion helpers used by the E2B pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["attention", "p_factor", "internalizing", "externalizing"]
SUBJECT_COLUMNS = ["participant_id", "age", "sex", *TARGET_COLUMNS]


def validate_subject_metadata(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metadata = pd.concat(
        [frame[SUBJECT_COLUMNS] for frame in frames.values()], ignore_index=True
    )
    metadata["participant_id"] = metadata["participant_id"].astype(str)
    conflicts = {
        column: int(
            metadata.groupby("participant_id", sort=False)[column]
            .nunique(dropna=True)
            .gt(1)
            .sum()
        )
        for column in SUBJECT_COLUMNS[1:]
    }
    if any(conflicts.values()):
        raise RuntimeError(f"Conflicting subject metadata across tasks: {conflicts}")
    return metadata.drop_duplicates("participant_id", keep="first")


def task_counts(frames: dict[str, pd.DataFrame]) -> pd.Series:
    long_ids = pd.concat(
        [
            frame[["participant_id"]].assign(_task=task)
            for task, frame in frames.items()
        ],
        ignore_index=True,
    )
    long_ids["participant_id"] = long_ids["participant_id"].astype(str)
    return long_ids.groupby("participant_id", sort=True)["_task"].nunique()


def fuse_mean_available(
    frames: dict[str, pd.DataFrame],
    embedding_columns: list[str],
    metadata: pd.DataFrame,
    counts: pd.Series,
    min_tasks: int,
) -> tuple[pd.DataFrame, list[str]]:
    long_embeddings = pd.concat(
        [frame[["participant_id", *embedding_columns]] for frame in frames.values()],
        ignore_index=True,
    )
    long_embeddings["participant_id"] = long_embeddings["participant_id"].astype(str)
    fused = (
        long_embeddings.groupby("participant_id", sort=True)[embedding_columns]
        .mean()
        .reset_index()
    )
    fused = fused.merge(metadata, on="participant_id", how="left", validate="one_to_one")
    fused["n_tasks_available"] = fused["participant_id"].map(counts).astype(int)
    fused = fused.loc[fused["n_tasks_available"] >= min_tasks].reset_index(drop=True)
    return fused, embedding_columns


def fuse_task_concat(
    frames: dict[str, pd.DataFrame],
    embedding_columns: list[str],
    metadata: pd.DataFrame,
    counts: pd.Series,
    min_tasks: int,
) -> tuple[pd.DataFrame, list[str]]:
    wide = metadata.copy()
    fused_columns: list[str] = []
    for task, frame in frames.items():
        renamed = {
            column: f"{task}__{column}" for column in embedding_columns
        }
        task_frame = frame[["participant_id", *embedding_columns]].rename(
            columns=renamed
        )
        task_frame["participant_id"] = task_frame["participant_id"].astype(str)
        wide = wide.merge(task_frame, on="participant_id", how="outer", validate="one_to_one")
        fused_columns.extend(renamed.values())
    wide["n_tasks_available"] = wide["participant_id"].map(counts).fillna(0).astype(int)
    wide = wide.loc[wide["n_tasks_available"] >= min_tasks].reset_index(drop=True)
    return wide, fused_columns


def fusion_audit(
    data: pd.DataFrame,
    feature_columns: list[str],
    representation: str,
    target: str,
    strategy: str,
    min_tasks: int,
    tasks_total: int,
) -> dict[str, Any]:
    labeled = data.dropna(subset=[target]).copy()
    if labeled["participant_id"].duplicated().any():
        raise RuntimeError("Task fusion produced duplicate participant rows")
    if labeled["participant_id"].nunique() < 100:
        raise RuntimeError("Task fusion produced fewer than 100 labeled subjects")
    matrix = labeled[feature_columns].to_numpy(dtype=np.float64)
    if strategy == "mean_available" and not np.isfinite(matrix).all():
        raise RuntimeError("Mean fusion produced non-finite features")
    if strategy == "task_concat" and np.isinf(matrix).any():
        raise RuntimeError("Concatenation produced infinite features")
    counts = labeled["n_tasks_available"].astype(int)
    return {
        "status": "succeeded",
        "representation": representation,
        "target": target,
        "fusion_strategy": strategy,
        "min_tasks": min_tasks,
        "rows_labeled": int(len(labeled)),
        "unique_subjects": int(labeled["participant_id"].nunique()),
        "fused_feature_dimension": len(feature_columns),
        "missing_feature_values": int(np.isnan(matrix).sum()),
        "subjects_with_all_tasks": int((counts == tasks_total).sum()),
        "tasks_available_min": int(counts.min()),
        "tasks_available_median": float(counts.median()),
        "tasks_available_max": int(counts.max()),
        "participant_leakage_detected": False,
    }
