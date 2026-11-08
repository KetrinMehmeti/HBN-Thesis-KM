#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the first four-channel Muse replication phase.

The wrapper freezes the B1 input-ablation design while intentionally omitting
the covariates-only condition, which has already been evaluated in the
all-channel benchmark. Each invocation handles one target so independent Spot
workers can resume from durable S3 checkpoints.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or execute Muse four-channel Phase B1."
    )
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def build_command(target: str, output_dir: str, execute: bool) -> list[str]:
    runner = Path(__file__).with_name("02_run_benchmark_optuna.py")
    command = [
        sys.executable, str(runner),
        "--data-path", "s3://hbn-ml-benchmark-tesi-ketrin/benchmark_v1/data/df_ml.parquet",
        "--output-dir", output_dir,
        "--targets", target,
        "--tasks", *TASKS,
        "--feature-families", "all",
        "--aggregate-sets", "mean_only",
        "--input-sets", "eeg_only", "eeg_confounds",
        "--models", "dummy", "elasticnet", "extratrees",
        "--selectors", "f_regression", "mutual_info",
        "--selector-k-max", "1000",
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
    summary = {
        "phase": "phase_b1_input_ablation_muse4",
        "target": args.target,
        "tasks": TASKS,
        "channel_ids": MUSE_CHANNELS,
        "channel_labels": MUSE_LABELS,
        "input_sets": ["eeg_only", "eeg_confounds"],
        "excluded_input_set": "confounds_only",
        "models": ["dummy", "elasticnet", "extratrees"],
        "selectors_for_tuned_models": ["f_regression", "mutual_info"],
        "outer_folds": 5,
        "inner_folds": 2,
        "optuna_trials": 4,
        "expected_combinations": 78,
        "expected_fold_markers": 390,
        "output_dir": args.output_dir,
        "writes_enabled": bool(args.execute),
    }
    print(json.dumps(summary, indent=2))
    return subprocess.run(
        build_command(args.target, args.output_dir, args.execute), check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
