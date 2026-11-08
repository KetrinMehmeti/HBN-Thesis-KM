#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the compact Muse four-channel model comparison (Phase B3).

The two aggregation representations retained for each psychopathology target
are fixed from the completed Phase B2 cross-task ranking. Each worker handles
one target and evaluates the same six regression families across all 13 tasks.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


MUSE_CHANNELS = ["E26", "E2", "E57", "E100"]
MUSE_LABELS = {"E26": "AF7", "E2": "AF8", "E57": "TP9", "E100": "TP10"}
TARGETS = ["attention", "p_factor", "internalizing", "externalizing"]
TASKS = [
    "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "ThePresent",
    "FunwithFractals", "symbolSearch", "contrastChangeDetection_1",
    "contrastChangeDetection_2", "contrastChangeDetection_3",
    "seqLearning6target", "seqLearning8target", "surroundSupp_1",
    "surroundSupp_2",
]
MODELS = [
    "ridge", "pls", "extratrees", "randomforest",
    "histgradientboosting", "svr",
]
REPRESENTATIONS = {
    "attention": {
        "aggregate_sets": ["iqr_only", "mean_only"],
        "correlation_thresholds": ["0.95"],
    },
    "p_factor": {
        "aggregate_sets": ["iqr_only", "std_only"],
        "correlation_thresholds": ["0.95"],
    },
    "internalizing": {
        "aggregate_sets": ["std_only"],
        "correlation_thresholds": ["none", "0.95"],
    },
    "externalizing": {
        "aggregate_sets": ["median_only", "iqr_only"],
        "correlation_thresholds": ["0.95"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or execute Muse four-channel Phase B3."
    )
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def build_command(target: str, output_dir: str, execute: bool) -> list[str]:
    runner = Path(__file__).with_name("02_run_benchmark_optuna.py")
    representation = REPRESENTATIONS[target]
    command = [
        sys.executable, str(runner),
        "--data-path", "s3://hbn-ml-benchmark-tesi-ketrin/benchmark_v1/data/df_ml.parquet",
        "--output-dir", output_dir,
        "--targets", target,
        "--tasks", *TASKS,
        "--feature-families", "all",
        "--aggregate-sets", *representation["aggregate_sets"],
        "--input-sets", "eeg_confounds",
        "--correlation-thresholds", *representation["correlation_thresholds"],
        "--models", *MODELS,
        "--selectors", "f_regression", "mutual_info",
        "--selector-k-max", "500",
        "--svr-kernels", "rbf",
        "--primary-metric", "neg_rmse",
        "--outer-folds", "5",
        "--inner-folds", "2",
        "--n-trials", "4",
        "--random-state", "42",
        "--min-subjects", "100",
        "--n-jobs", "1",
        "--region-name", "us-east-1",
        "--channel-ids", *MUSE_CHANNELS,
        "--strict-model-dependencies", "--resume",
    ]
    if not execute:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    representation = REPRESENTATIONS[args.target]
    representation_count = (
        len(representation["aggregate_sets"])
        * len(representation["correlation_thresholds"])
    )
    summary = {
        "phase": "phase_b3_compact_model_comparison_muse4",
        "target": args.target,
        "selection_rule": (
            "top two Phase B2 representations by mean R2 across all 13 tasks"
        ),
        "tasks": TASKS,
        "channel_ids": MUSE_CHANNELS,
        "channel_labels": MUSE_LABELS,
        "feature_family": "all",
        "aggregate_sets": representation["aggregate_sets"],
        "correlation_thresholds": representation["correlation_thresholds"],
        "input_set": "eeg_confounds",
        "models": MODELS,
        "svr_kernels": ["rbf"],
        "selectors": ["f_regression", "mutual_info"],
        "outer_folds": 5,
        "inner_folds": 2,
        "optuna_trials": 4,
        "expected_combinations": 13 * representation_count * len(MODELS),
        "expected_fold_markers": 13 * representation_count * len(MODELS) * 5,
        "output_dir": args.output_dir,
        "writes_enabled": bool(args.execute),
    }
    print(json.dumps(summary, indent=2))
    return subprocess.run(
        build_command(args.target, args.output_dir, args.execute), check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
