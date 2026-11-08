#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the wearable four-channel aggregation ablation (Phase B2).

The full protocol mirrors the completed Muse four-channel B2 benchmark.  It
changes only the exact EGI channel prefixes and therefore isolates how the
epoch-level aggregation behaves for each prespecified wearable montage.
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
FULL_TASKS = [
    "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "ThePresent",
    "FunwithFractals", "symbolSearch", "contrastChangeDetection_1",
    "contrastChangeDetection_2", "contrastChangeDetection_3",
    "seqLearning6target", "seqLearning8target", "surroundSupp_1",
    "surroundSupp_2",
]
FULL_AGGREGATE_SETS = [
    "mean_only", "std_only", "median_only", "iqr_only", "mean_std",
    "median_iqr", "central_tendency", "variability", "all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or execute wearable four-channel Phase B2."
    )
    parser.add_argument("--configuration", required=True, choices=sorted(CONFIGURATIONS))
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def protocol(smoke: bool) -> dict:
    if smoke:
        return {
            "tasks": ["RestingState"],
            "aggregate_sets": ["mean_only", "std_only"],
            "outer_folds": 2,
            "inner_folds": 2,
            "n_trials": 1,
            "min_subjects": 30,
            "expected_combinations": 4,
            "expected_fold_markers": 8,
        }
    return {
        "tasks": FULL_TASKS,
        "aggregate_sets": FULL_AGGREGATE_SETS,
        "outer_folds": 5,
        "inner_folds": 2,
        "n_trials": 4,
        "min_subjects": 100,
        "expected_combinations": 234,
        "expected_fold_markers": 1170,
    }


def build_command(args: argparse.Namespace, selected: dict) -> list[str]:
    runner = Path(__file__).with_name("02_run_benchmark_optuna.py")
    command = [
        sys.executable, str(runner),
        "--data-path", "s3://hbn-ml-benchmark-tesi-ketrin/benchmark_v1/data/df_ml.parquet",
        "--output-dir", args.output_dir,
        "--targets", args.target,
        "--tasks", *selected["tasks"],
        "--feature-families", "all",
        "--aggregate-sets", *selected["aggregate_sets"],
        "--input-sets", "eeg_confounds",
        "--correlation-thresholds", "none", "0.95",
        "--models", "extratrees",
        "--selectors", "f_regression", "mutual_info",
        "--selector-k-max", "1000",
        "--primary-metric", "neg_rmse",
        "--outer-folds", str(selected["outer_folds"]),
        "--inner-folds", str(selected["inner_folds"]),
        "--n-trials", str(selected["n_trials"]),
        "--random-state", "42",
        "--min-subjects", str(selected["min_subjects"]),
        "--n-jobs", "1",
        "--region-name", "us-east-1",
        "--channel-ids", *CONFIGURATIONS[args.configuration]["channel_ids"],
        "--strict-model-dependencies", "--resume",
    ]
    if not args.execute:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    selected = protocol(args.smoke)
    config = CONFIGURATIONS[args.configuration]
    summary = {
        "schema_version": 1,
        "phase": "phase_b2_aggregation_ablation_wearable4",
        "mode": "smoke" if args.smoke else "full",
        "configuration": args.configuration,
        "target": args.target,
        "tasks": selected["tasks"],
        "channel_ids": config["channel_ids"],
        "channel_labels": config["channel_labels"],
        "channel_filter": "exact '<channel>_' prefix only",
        "rationale": config["rationale"],
        "feature_family": "all",
        "aggregate_sets": selected["aggregate_sets"],
        "input_set": "eeg_confounds",
        "model": "extratrees",
        "correlation_thresholds": ["none", 0.95],
        "selectors": ["f_regression", "mutual_info"],
        "outer_folds": selected["outer_folds"],
        "inner_folds": selected["inner_folds"],
        "optuna_trials": selected["n_trials"],
        "random_state": 42,
        "expected_combinations": selected["expected_combinations"],
        "expected_fold_markers": selected["expected_fold_markers"],
        "output_dir": args.output_dir,
        "writes_enabled": bool(args.execute),
    }
    print(json.dumps(summary, indent=2))
    return subprocess.run(build_command(args, selected), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
