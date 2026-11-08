#!/usr/bin/env python3
"""Freeze the all-subject EEGNet cohort and subject-level outer-CV splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

TARGETS = ("attention", "p_factor", "internalizing", "externalizing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    args = parser.parse_args()

    signal = pd.read_parquet(args.signal_manifest)
    labels = pd.read_parquet(args.targets, columns=["participant_id", *TARGETS])
    labels["participant_id"] = labels.participant_id.astype(str)
    labels = labels.dropna(subset=list(TARGETS)).drop_duplicates("participant_id").copy()
    signal["participant_id"] = signal.participant_id.astype(str)
    signal = signal.loc[signal.eligible_min_2_epochs].copy()
    signal = signal.loc[signal.participant_id.isin(labels.participant_id)].copy()
    counts = signal.groupby("participant_id").task.nunique()
    signal = signal.loc[signal.participant_id.isin(counts[counts >= 1].index)].copy()
    ordered_ids = sorted(signal.participant_id.unique(), key=lambda value: hashlib.sha256(f"{args.seed}|{value}".encode()).hexdigest())
    if len(ordered_ids) < args.outer_folds * 6:
        raise RuntimeError("Insufficient eligible subjects")
    fold_by_id = {}
    for fold, (_, test) in enumerate(KFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed).split(ordered_ids), 1):
        fold_by_id.update({ordered_ids[index]: fold for index in test})
    frame = signal.merge(labels, on="participant_id", how="inner", validate="many_to_one")
    frame["outer_fold"] = frame.participant_id.map(fold_by_id).astype(int)
    frame["source_etag"] = ""
    keep = ["participant_id", "task", "source_key", "source_size", "source_etag", "epoch_count", "outer_fold", *TARGETS]
    frame = frame.rename(columns={"s3_key": "source_key", "size_bytes": "source_size", "n_epochs": "epoch_count"})[keep].sort_values(["outer_fold", "participant_id", "task"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = args.output_dir / "frozen_cohort_records.csv"
    frame.to_csv(cohort, index=False)
    subject = frame.groupby("participant_id", as_index=False).agg(outer_fold=("outer_fold", "first"), available_tasks=("task", "nunique"), total_epochs=("epoch_count", "sum"))
    subject.to_csv(args.output_dir / "frozen_cohort_subjects.csv", index=False)
    digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
    summary = {
        "purpose": "EEGNet raw-signal full benchmark frozen cohort and 5-fold participant-level CV",
        "selection": "all participants with complete four-target labels and at least one structurally validated signal with at least two epochs",
        "selection_uses_target_values": "only completeness, never values or distribution",
        "subjects": int(len(subject)), "records": int(len(frame)), "outer_folds": args.outer_folds,
        "subjects_by_fold": {str(k): int(v) for k, v in subject.outer_fold.value_counts().sort_index().items()},
        "tasks_per_subject": {"mean": float(subject.available_tasks.mean()), "median": float(subject.available_tasks.median())},
        "record_csv_sha256": digest,
    }
    (args.output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()