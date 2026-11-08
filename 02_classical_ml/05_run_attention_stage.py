#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_STAGE_FIELDS = {
    "tasks",
    "feature_families",
    "aggregate_sets",
    "models",
    "selectors",
    "outer_folds",
    "inner_folds",
    "n_trials",
    "include_confounds",
    "min_subjects",
    "n_jobs",
    "resume",
}


def stage_output_dir(plan: dict[str, Any], stage: dict[str, Any]) -> str:
    output_subdir = stage.get("output_subdir", stage["id"])
    output_root = stage.get("output_root", plan["output_root"])
    return f"{output_root.rstrip('/')}/{output_subdir}/"


def stage_targets(plan: dict[str, Any], stage: dict[str, Any]) -> list[str]:
    return list(stage.get("targets", [plan["target"]]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or execute one stage of the progressive attention benchmark."
    )
    parser.add_argument(
        "--plan",
        default=str(
            Path(__file__).with_name("configs") / "attention_progressive_plan.json"
        ),
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional target override used for independent AWS shards.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override used for independent AWS shards.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the benchmark. Without this flag, only the no-write dry-run is executed.",
    )
    return parser.parse_args()


def load_plan(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported plan schema_version")
    if plan.get("target") != "attention":
        raise ValueError("This runner only accepts an attention plan")
    return plan


def select_stage(plan: dict[str, Any], stage_id: str) -> dict[str, Any]:
    matches = [stage for stage in plan["stages"] if stage.get("id") == stage_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one stage named {stage_id!r}")
    stage = matches[0]
    if stage.get("status") != "ready":
        raise ValueError(
            f"Stage {stage_id!r} is a decision gate and cannot run before review"
        )
    missing = sorted(REQUIRED_STAGE_FIELDS - set(stage))
    if missing:
        raise ValueError(
            f"Stage {stage_id!r} is missing fields: {', '.join(missing)}"
        )
    return stage


def build_command(
    plan: dict[str, Any],
    stage: dict[str, Any],
    execute: bool,
    targets_override: list[str] | None = None,
    output_dir_override: str | None = None,
) -> list[str]:
    runner = Path(__file__).with_name("02_run_benchmark_optuna.py")
    output_dir = output_dir_override or stage_output_dir(plan, stage)
    command = [
        sys.executable,
        str(runner),
        "--data-path",
        plan["data_path"],
        "--output-dir",
        output_dir,
        "--targets",
        *(targets_override or stage_targets(plan, stage)),
        "--tasks",
        *stage["tasks"],
        "--feature-families",
        *stage["feature_families"],
        "--aggregate-sets",
        *stage["aggregate_sets"],
        "--models",
        *stage["models"],
        "--selectors",
        *stage["selectors"],
        "--primary-metric",
        stage.get("primary_metric", plan["primary_metric"]),
        "--outer-folds",
        str(stage["outer_folds"]),
        "--inner-folds",
        str(stage["inner_folds"]),
        "--n-trials",
        str(stage["n_trials"]),
        "--random-state",
        str(plan["random_state"]),
        "--min-subjects",
        str(stage["min_subjects"]),
        "--n-jobs",
        str(stage["n_jobs"]),
        "--region-name",
        plan["region_name"],
    ]
    if stage.get("input_sets"):
        command.extend(["--input-sets", *stage["input_sets"]])
    if stage["include_confounds"]:
        command.append("--include-confounds")
    if stage["resume"]:
        command.append("--resume")
    if stage.get("selector_k_max") is not None:
        command.extend(["--selector-k-max", str(stage["selector_k_max"])])
    if stage.get("correlation_thresholds"):
        command.extend(
            [
                "--correlation-thresholds",
                *[
                    str(value)
                    for value in stage["correlation_thresholds"]
                ],
            ]
        )
    if stage.get("literal_all_aggregate_set"):
        command.append("--literal-all-aggregate-set")
    if stage.get("svr_kernels"):
        command.extend(["--svr-kernels", *stage["svr_kernels"]])
    if stage.get("strict_model_dependencies", True):
        command.append("--strict-model-dependencies")
    if not execute:
        command.append("--dry-run")
    return command


def summarize_stage(plan: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    representation_combinations = (
        len(stage["tasks"])
        * len(stage["feature_families"])
        * len(stage["aggregate_sets"])
        * len(stage.get("input_sets", [None]))
        * len(stage.get("correlation_thresholds", ["none"]))
    )
    targets = stage_targets(plan, stage)
    n_combinations = (
        len(targets) * representation_combinations * len(stage["models"])
    )
    dummy_models = sum(model == "dummy" for model in stage["models"])
    tuned_models = len(stage["models"]) - dummy_models
    fits_per_dummy = stage["outer_folds"] * (stage["inner_folds"] + 1)
    fits_per_tuned_model = stage["outer_folds"] * (
        stage["n_trials"] * stage["inner_folds"] + 1
    )
    return {
        "targets": targets,
        "stage": stage["id"],
        "tasks": stage["tasks"],
        "feature_families": stage["feature_families"],
        "aggregate_sets": stage["aggregate_sets"],
        "correlation_thresholds": stage.get(
            "correlation_thresholds", ["none"]
        ),
        "input_sets": stage.get("input_sets"),
        "models": stage["models"],
        "selectors_searched_within_each_model": stage["selectors"],
        "svr_kernels": stage.get("svr_kernels", ["rbf", "linear"]),
        "n_combinations": n_combinations,
        "outer_folds": stage["outer_folds"],
        "inner_folds": stage["inner_folds"],
        "n_trials": stage["n_trials"],
        "estimated_model_fits": (
            len(targets)
            * representation_combinations
            * (
                dummy_models * fits_per_dummy
                + tuned_models * fits_per_tuned_model
            )
        ),
        "output_dir": stage_output_dir(plan, stage),
        "writes_enabled": False,
    }


def main() -> int:
    args = parse_args()
    try:
        plan = load_plan(args.plan)
        stage = select_stage(plan, args.stage)
        targets_override = (
            None if args.targets in (None, ["all"]) else args.targets
        )
        command = build_command(
            plan,
            stage,
            execute=args.execute,
            targets_override=targets_override,
            output_dir_override=args.output_dir,
        )
        mode = "EXECUTE" if args.execute else "DRY-RUN"
        print(f"Mode: {mode}")
        print(f"Stage: {stage['id']}")
        print(f"Purpose: {stage['purpose']}")
        if not args.execute:
            print(json.dumps(summarize_stage(plan, stage), indent=2))
            return 0
        completed = subprocess.run(command, check=False)
        return completed.returncode
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
