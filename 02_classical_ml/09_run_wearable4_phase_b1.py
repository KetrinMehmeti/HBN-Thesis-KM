#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run Phase B1 for a prespecified four-channel wearable montage.

This wrapper mirrors the completed Muse four-channel B1 experiment while
changing only the exact EGI channel prefixes.  Each invocation handles one
psychopathology target and can resume from durable S3 fold checkpoints.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or execute wearable four-channel Phase B1."
    )
    parser.add_argument("--configuration", required=True, choices=sorted(CONFIGURATIONS))
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def build_command(
    configuration: str, target: str, output_dir: str, execute: bool
) -> list[str]:
    runner = Path(__file__).with_name("02_run_benchmark_optuna.py")
    channel_ids = CONFIGURATIONS[configuration]["channel_ids"]
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
        "--channel-ids", *channel_ids,
        "--strict-model-dependencies", "--resume",
    ]
    if not execute:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    config = CONFIGURATIONS[args.configuration]
    summary = {
        "schema_version": 1,
        "phase": "phase_b1_input_ablation_wearable4",
        "configuration": args.configuration,
        "target": args.target,
        "tasks": TASKS,
        "channel_ids": config["channel_ids"],
        "channel_labels": config["channel_labels"],
        "channel_filter": "exact '<channel>_' prefix only",
        "rationale": config["rationale"],
        "input_sets": ["eeg_only", "eeg_confounds"],
        "excluded_input_set": "confounds_only_already_completed",
        "models": ["dummy", "elasticnet", "extratrees"],
        "selectors_for_tuned_models": ["f_regression", "mutual_info"],
        "outer_folds": 5,
        "inner_folds": 2,
        "optuna_trials": 4,
        "random_state": 42,
        "expected_combinations": 78,
        "expected_fold_markers": 390,
        "output_dir": args.output_dir,
        "writes_enabled": bool(args.execute),
    }
    print(json.dumps(summary, indent=2))
    return subprocess.run(
        build_command(
            args.configuration, args.target, args.output_dir, args.execute
        ),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
