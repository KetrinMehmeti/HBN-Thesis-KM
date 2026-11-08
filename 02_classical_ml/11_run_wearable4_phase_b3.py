#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the wearable four-channel compact model comparison (Phase B3).

For each wearable montage and target, the two retained aggregation/filter
pairs are fixed from the completed Phase B2 cross-task ranking.  Each worker
handles one target and evaluates the same six model families across all 13
tasks.  Exact pairs are executed sequentially so that pairs such as
``std_only/none`` and ``iqr_only/0.95`` are not expanded into an unintended
Cartesian product.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONFIGURATIONS = {
    "frontal_headband": {
        "channel_ids": ["E22", "E9", "E33", "E122"],
        "channel_labels": {"E22": "Fp1", "E9": "Fp2", "E33": "F7", "E122": "F8"},
        "rationale": "Four frontal positions feasible in a forehead headband.",
    },
    "temporal_ear_adjacent": {
        "channel_ids": ["E45", "E108", "E57", "E100"],
        "channel_labels": {"E45": "T7", "E108": "T8", "E57": "TP9", "E100": "TP10"},
        "rationale": "Scalp approximation of a bilateral ear-adjacent wearable.",
    },
}

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

# Each item is one exact (aggregation, correlation-filter) pair.  The values
# are frozen from Phase B2 before Phase B3 model-family comparison.
REPRESENTATIONS = {
    "frontal_headband": {
        "attention": [
            {"aggregate_set": "std_only", "correlation_threshold": "none", "cross_task_mean_r2": 0.00933460},
            {"aggregate_set": "iqr_only", "correlation_threshold": "0.95", "cross_task_mean_r2": 0.00926036},
        ],
        "p_factor": [
            {"aggregate_set": "iqr_only", "correlation_threshold": "0.95", "cross_task_mean_r2": -0.00396113},
            {"aggregate_set": "variability", "correlation_threshold": "0.95", "cross_task_mean_r2": -0.00419134},
        ],
        "internalizing": [
            {"aggregate_set": "iqr_only", "correlation_threshold": "none", "cross_task_mean_r2": 0.04730338},
            {"aggregate_set": "iqr_only", "correlation_threshold": "0.95", "cross_task_mean_r2": 0.04690401},
        ],
        "externalizing": [
            {"aggregate_set": "iqr_only", "correlation_threshold": "0.95", "cross_task_mean_r2": 0.03775272},
            {"aggregate_set": "iqr_only", "correlation_threshold": "none", "cross_task_mean_r2": 0.03546659},
        ],
    },
    "temporal_ear_adjacent": {
        "attention": [
            {"aggregate_set": "std_only", "correlation_threshold": "none", "cross_task_mean_r2": 0.00995117},
            {"aggregate_set": "std_only", "correlation_threshold": "0.95", "cross_task_mean_r2": 0.00959452},
        ],
        "p_factor": [
            {"aggregate_set": "iqr_only", "correlation_threshold": "0.95", "cross_task_mean_r2": -0.00335987},
            {"aggregate_set": "iqr_only", "correlation_threshold": "none", "cross_task_mean_r2": -0.00414541},
        ],
        "internalizing": [
            {"aggregate_set": "median_only", "correlation_threshold": "0.95", "cross_task_mean_r2": 0.04471437},
            {"aggregate_set": "variability", "correlation_threshold": "0.95", "cross_task_mean_r2": 0.04394388},
        ],
        "externalizing": [
            {"aggregate_set": "iqr_only", "correlation_threshold": "0.95", "cross_task_mean_r2": 0.03211978},
            {"aggregate_set": "iqr_only", "correlation_threshold": "none", "cross_task_mean_r2": 0.03211898},
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or execute wearable four-channel Phase B3."
    )
    parser.add_argument("--configuration", required=True, choices=sorted(CONFIGURATIONS))
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def build_command(
    configuration: str,
    target: str,
    representation: dict,
    output_dir: str,
    execute: bool,
) -> list[str]:
    runner = Path(__file__).with_name("02_run_benchmark_optuna.py")
    command = [
        sys.executable, str(runner),
        "--data-path", "s3://hbn-ml-benchmark-tesi-ketrin/benchmark_v1/data/df_ml.parquet",
        "--output-dir", output_dir,
        "--targets", target,
        "--tasks", *TASKS,
        "--feature-families", "all",
        "--aggregate-sets", representation["aggregate_set"],
        "--input-sets", "eeg_confounds",
        "--correlation-thresholds", representation["correlation_threshold"],
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
        "--channel-ids", *CONFIGURATIONS[configuration]["channel_ids"],
        "--strict-model-dependencies", "--resume",
    ]
    if not execute:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    config = CONFIGURATIONS[args.configuration]
    representations = REPRESENTATIONS[args.configuration][args.target]
    summary = {
        "schema_version": 1,
        "phase": "phase_b3_compact_model_comparison_wearable4",
        "configuration": args.configuration,
        "target": args.target,
        "selection_rule": "top two exact Phase B2 representation pairs by mean outer-fold R2 across all 13 tasks",
        "representations": representations,
        "tasks": TASKS,
        "channel_ids": config["channel_ids"],
        "channel_labels": config["channel_labels"],
        "channel_filter": "exact '<channel>_' prefix only",
        "rationale": config["rationale"],
        "feature_family": "all",
        "input_set": "eeg_confounds",
        "models": MODELS,
        "svr_kernels": ["rbf"],
        "selectors": ["f_regression", "mutual_info"],
        "outer_folds": 5,
        "inner_folds": 2,
        "optuna_trials": 4,
        "expected_combinations": 13 * len(representations) * len(MODELS),
        "expected_fold_markers": 13 * len(representations) * len(MODELS) * 5,
        "output_dir": args.output_dir,
        "writes_enabled": bool(args.execute),
    }
    print(json.dumps(summary, indent=2))

    for representation in representations:
        result = subprocess.run(
            build_command(
                args.configuration,
                args.target,
                representation,
                args.output_dir,
                args.execute,
            ),
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
