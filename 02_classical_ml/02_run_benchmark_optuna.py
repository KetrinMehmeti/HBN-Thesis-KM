#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import json
import logging
import platform
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from checkpointing import (
    drop_fold_rows,
    fold_checkpoint_key,
    should_skip_fold,
)
from ml_utils import (
    AGGREGATE_SETS,
    ALL_MODELS,
    ALL_SELECTORS,
    FEATURE_FAMILIES,
    TARGET_COLS,
    build_pipeline,
    dump_joblib,
    exists_path,
    get_selected_feature_rows,
    join_path,
    make_model,
    make_optuna_objective,
    metric_to_score,
    read_table,
    regression_metrics,
    select_feature_columns,
    suggest_model_params,
    suggest_selector_params,
    write_json,
    write_table,
)


def parse_args():
    p = argparse.ArgumentParser(description="Nested-CV EEG ML benchmark with Optuna.")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--targets", nargs="+", default=TARGET_COLS)
    p.add_argument(
        "--channel-ids",
        nargs="+",
        default=None,
        help=(
            "Optional exact EEG channel IDs to retain, for example E2 E26. "
            "Features are matched only by the strict '<channel>_' prefix."
        ),
    )
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--feature-families", nargs="+", default=["all"])
    p.add_argument(
        "--expand-all-feature-families",
        action="store_true",
        help=(
            "Expand a sole --feature-families all into every named family. "
            "Without this flag, all means the single combined representation."
        ),
    )
    p.add_argument("--aggregate-sets", nargs="+", default=["all"])
    p.add_argument(
        "--literal-all-aggregate-set",
        action="store_true",
        help=(
            "Interpret a sole --aggregate-sets all as the combined "
            "mean/std/median/IQR representation instead of expanding it."
        ),
    )
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument(
        "--strict-model-dependencies",
        action="store_true",
        help=(
            "Fail before loading data when a requested optional model backend "
            "is unavailable, instead of silently skipping that model."
        ),
    )
    p.add_argument("--selectors", nargs="+", default=["all"])
    p.add_argument(
        "--input-sets",
        nargs="+",
        choices=["eeg_only", "confounds_only", "eeg_confounds"],
        default=None,
        help=(
            "Predictor sets to compare. When omitted, preserve legacy behavior "
            "from --include-confounds."
        ),
    )
    p.add_argument(
        "--selector-k-max",
        type=int,
        default=None,
        help="Maximum number of EEG features retained by k-based selectors.",
    )
    p.add_argument(
        "--correlation-thresholds",
        nargs="+",
        default=["none"],
        help=(
            "EEG correlation filters evaluated as separate configurations. "
            "Use 'none' to disable the filter or values such as 0.95."
        ),
    )
    p.add_argument(
        "--svr-kernels",
        nargs="+",
        default=["rbf", "linear"],
        choices=["rbf", "linear"],
        help="Kernel candidates considered when the model is SVR.",
    )
    p.add_argument("--primary-metric", default="ccc", choices=["pearson", "spearman", "ccc", "neg_rmse", "neg_mae"])
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--include-confounds", action="store_true")
    p.add_argument("--min-subjects", type=int, default=50)
    p.add_argument("--limit-combinations", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1, help="Optuna trial parallelism within each outer fold.")
    p.add_argument("--region-name", default="us-east-1")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the experiment plan without reading data or writing outputs.",
    )
    return p.parse_args()


def expand_arg(values, all_values):
    if len(values) == 1 and values[0] == "all":
        return list(all_values)
    return values


def validate_choices(label, values, allowed):
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")


def build_plan(args):
    tasks = list(args.tasks)
    feature_families = (
        expand_arg(args.feature_families, FEATURE_FAMILIES)
        if args.expand_all_feature_families
        else list(args.feature_families)
    )
    aggregate_sets = (
        list(args.aggregate_sets)
        if args.literal_all_aggregate_set
        else expand_arg(args.aggregate_sets, AGGREGATE_SETS.keys())
    )
    requested_models = expand_arg(args.models, ALL_MODELS)
    selectors = expand_arg(args.selectors, ALL_SELECTORS)
    input_sets = list(
        args.input_sets
        or (["eeg_confounds"] if args.include_confounds else ["eeg_only"])
    )
    correlation_thresholds = []
    for value in args.correlation_thresholds:
        if str(value).lower() == "none":
            correlation_thresholds.append(None)
        else:
            threshold = float(value)
            if not 0.0 < threshold < 1.0:
                raise ValueError(
                    "correlation thresholds must be between 0 and 1"
                )
            correlation_thresholds.append(threshold)

    validate_choices("target", args.targets, TARGET_COLS)
    validate_choices("feature family", feature_families, FEATURE_FAMILIES)
    validate_choices("aggregate set", aggregate_sets, AGGREGATE_SETS.keys())
    validate_choices("model", requested_models, ALL_MODELS)
    validate_choices("selector", selectors, ALL_SELECTORS)

    if any(value < 2 for value in (args.outer_folds, args.inner_folds)):
        raise ValueError("outer-folds and inner-folds must both be at least 2")
    if args.n_trials < 1:
        raise ValueError("n-trials must be at least 1")
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be at least 1")
    if args.selector_k_max is not None and args.selector_k_max < 1:
        raise ValueError("selector-k-max must be at least 1")

    combinations = list(
        itertools.product(
            args.targets,
            tasks,
            feature_families,
            aggregate_sets,
            input_sets,
            correlation_thresholds,
            requested_models,
        )
    )
    if args.limit_combinations is not None:
        combinations = combinations[: args.limit_combinations]
    return {
        "tasks": tasks,
        "feature_families": feature_families,
        "aggregate_sets": aggregate_sets,
        "requested_models": requested_models,
        "selectors": selectors,
        "input_sets": input_sets,
        "correlation_thresholds": correlation_thresholds,
        "combinations": combinations,
    }


def run_id_for(
    target,
    task,
    feature_family,
    aggregate_set,
    model,
    input_set=None,
    correlation_threshold=None,
    include_correlation_id=False,
):
    parts = [target, task, feature_family, aggregate_set]
    if input_set is not None:
        parts.append(input_set)
    if include_correlation_id:
        parts.append(
            "corr_none"
            if correlation_threshold is None
            else f"corr_{correlation_threshold:g}"
        )
    parts.append(model)
    return "__".join(parts).replace("/", "_")


def init_logging(output_dir, region_name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("benchmark")


def write_outputs(output_dir, region_name, tables):
    for name, df in tables.items():
        if df is not None and len(df) > 0:
            suffix = ".parquet" if name == "fold_predictions" else ".csv"
            write_table(df, join_path(output_dir, f"{name}{suffix}"), region_name=region_name, index=False)


def read_existing_outputs(output_dir, region_name):
    specs = {
        "metrics_by_fold": "csv",
        "fold_predictions": "parquet",
        "best_trials": "csv",
        "optuna_trials": "csv",
        "selected_features": "csv",
        "errors": "csv",
        "benchmark_log": "csv",
        "input_audit": "csv",
    }
    out = {}
    for name, ext in specs.items():
        path = join_path(output_dir, f"{name}.{ext}")
        if exists_path(path, region_name=region_name):
            out[name] = read_table(
                path,
                region_name=region_name,
            ).to_dict("records")
        else:
            out[name] = []
    return out


def existing_complete(output_dir, run_id, region_name):
    return exists_path(join_path(output_dir, "done", f"{run_id}.json"), region_name=region_name)


def mark_complete(output_dir, run_id, region_name):
    write_json({"run_id": run_id, "completed_at_utc": datetime.now(timezone.utc).isoformat()}, join_path(output_dir, "done", f"{run_id}.json"), region_name=region_name)


def fold_complete_path(output_dir, run_id, outer_fold):
    return join_path(
        output_dir,
        "done_folds",
        fold_checkpoint_key(run_id, outer_fold),
    )


def existing_fold_complete(output_dir, run_id, outer_fold, region_name):
    return exists_path(
        fold_complete_path(output_dir, run_id, outer_fold),
        region_name=region_name,
    )


def mark_fold_complete(output_dir, run_id, outer_fold, region_name):
    write_json(
        {
            "run_id": run_id,
            "outer_fold": outer_fold,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        fold_complete_path(output_dir, run_id, outer_fold),
        region_name=region_name,
    )


def drop_run_rows(rows, run_id):
    return [row for row in rows if row.get("run_id") != run_id]


def output_tables(
    metrics_rows,
    prediction_rows,
    best_rows,
    trial_rows,
    selected_rows,
    error_rows,
    log_rows,
):
    return {
        "metrics_by_fold": pd.DataFrame(metrics_rows),
        "fold_predictions": pd.DataFrame(prediction_rows),
        "best_trials": pd.DataFrame(best_rows),
        "optuna_trials": pd.DataFrame(trial_rows),
        "selected_features": pd.DataFrame(selected_rows),
        "errors": pd.DataFrame(error_rows),
        "benchmark_log": pd.DataFrame(log_rows),
    }


def available_models(models, strict=False):
    out = []
    missing = []
    for model in models:
        if make_model(model, {}, random_state=42) is not None:
            out.append(model)
        else:
            missing.append(model)
    if strict and missing:
        raise RuntimeError(
            "Unavailable requested model dependencies: " + ", ".join(missing)
        )
    return out


def best_params_to_components(best_params, model_name, n_features, n_samples, random_state):
    selector = best_params.get("selector", "none")
    selector_params = {}
    if selector in {"f_regression", "mutual_info", "spearman"}:
        selector_params = {"k": min(int(best_params.get("selector_k", n_features)), n_features)}
    elif selector == "pca":
        selector_params = {"n_components": min(int(best_params.get("pca_n_components", 1)), n_features), "random_state": random_state}
    elif selector == "variance_threshold":
        selector_params = {"threshold": 0.0}

    class FrozenTrial:
        def __init__(self, params):
            self.params = params

        def suggest_float(self, name, *args, **kwargs):
            return self.params[name]

        def suggest_int(self, name, *args, **kwargs):
            return self.params[name]

        def suggest_categorical(self, name, choices):
            return self.params[name]

    model_params = suggest_model_params(FrozenTrial(best_params), model_name, n_features, n_samples, random_state=random_state)
    return selector, selector_params, model_params


def main():
    args = parse_args()
    plan = build_plan(args)
    if args.dry_run:
        representation_combinations = (
            len(args.targets)
            * len(plan["tasks"])
            * len(plan["feature_families"])
            * len(plan["aggregate_sets"])
            * len(plan["input_sets"])
            * len(plan["correlation_thresholds"])
        )
        dummy_models = sum(model == "dummy" for model in plan["requested_models"])
        tuned_models = len(plan["requested_models"]) - dummy_models
        print(
            json.dumps(
                {
                    "targets": args.targets,
                    "tasks": plan["tasks"],
                    "feature_families": plan["feature_families"],
                    "aggregate_sets": plan["aggregate_sets"],
                    "models": plan["requested_models"],
                    "selectors": plan["selectors"],
                    "input_sets": plan["input_sets"],
                    "correlation_thresholds": plan[
                        "correlation_thresholds"
                    ],
                    "n_combinations": len(plan["combinations"]),
                    "outer_folds": args.outer_folds,
                    "inner_folds": args.inner_folds,
                    "n_trials": args.n_trials,
                    "estimated_model_fits": (
                        representation_combinations
                        * (
                            dummy_models
                            * args.outer_folds
                            * (args.inner_folds + 1)
                            + tuned_models
                            * args.outer_folds
                            * (args.n_trials * args.inner_folds + 1)
                        )
                    ),
                    "writes_enabled": False,
                },
                indent=2,
            )
        )
        return

    log = init_logging(args.output_dir, args.region_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    models = available_models(
        plan["requested_models"], strict=args.strict_model_dependencies
    )

    df = read_table(args.data_path, region_name=args.region_name)
    df["participant_id"] = df["participant_id"].astype(str)
    df["task"] = df["task"].astype(str)

    tasks = args.tasks
    if len(tasks) == 1 and tasks[0] == "all":
        tasks = ["all_tasks"] + sorted(df["task"].dropna().unique().tolist())
    feature_families = plan["feature_families"]
    aggregate_sets = plan["aggregate_sets"]
    selectors = plan["selectors"]
    input_sets = plan["input_sets"]
    correlation_thresholds = plan["correlation_thresholds"]
    config = vars(args).copy()
    config.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "resolved_tasks": tasks,
            "resolved_feature_families": feature_families,
            "resolved_aggregate_sets": aggregate_sets,
            "resolved_models": models,
            "resolved_selectors": selectors,
            "resolved_input_sets": input_sets,
            "resolved_correlation_thresholds": correlation_thresholds,
            "resolved_channel_ids": args.channel_ids,
            "library_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "optuna": optuna.__version__,
            },
        }
    )
    write_json(config, join_path(args.output_dir, "run_config.json"), region_name=args.region_name)

    combinations = list(
        itertools.product(
            args.targets,
            tasks,
            feature_families,
            aggregate_sets,
            input_sets,
            correlation_thresholds,
            models,
        )
    )
    if args.limit_combinations is not None:
        combinations = combinations[: args.limit_combinations]

    existing = read_existing_outputs(args.output_dir, args.region_name) if args.resume else {}
    metrics_rows = existing.get("metrics_by_fold", [])
    prediction_rows = existing.get("fold_predictions", [])
    best_rows = existing.get("best_trials", [])
    trial_rows = existing.get("optuna_trials", [])
    selected_rows = existing.get("selected_features", [])
    error_rows = existing.get("errors", [])
    log_rows = existing.get("benchmark_log", [])
    input_audit_rows = existing.get("input_audit", [])

    for (
        target,
        task,
        feature_family,
        aggregate_set,
        input_set,
        correlation_threshold,
        model,
    ) in tqdm(combinations, desc="Benchmark combinations"):
        run_id = run_id_for(
            target,
            task,
            feature_family,
            aggregate_set,
            model,
            input_set=None if args.input_sets is None else input_set,
            correlation_threshold=correlation_threshold,
            include_correlation_id=(
                args.correlation_thresholds != ["none"]
            ),
        )
        if args.resume and existing_complete(args.output_dir, run_id, args.region_name):
            log.info("Skipping completed run %s", run_id)
            continue
        if not args.resume:
            metrics_rows = drop_run_rows(metrics_rows, run_id)
            prediction_rows = drop_run_rows(prediction_rows, run_id)
            best_rows = drop_run_rows(best_rows, run_id)
            trial_rows = drop_run_rows(trial_rows, run_id)
            selected_rows = drop_run_rows(selected_rows, run_id)
        error_rows = drop_run_rows(error_rows, run_id)
        log_rows = drop_run_rows(log_rows, run_id)
        try:
            data = df.copy() if task == "all_tasks" else df[df["task"] == task].copy()
            data = data.dropna(subset=[target, "participant_id"]).reset_index(drop=True)
            n_subjects = data["participant_id"].nunique()
            if n_subjects < args.min_subjects:
                log_rows.append({"run_id": run_id, "status": "skipped", "reason": f"n_subjects={n_subjects} < min_subjects"})
                continue

            eeg_cols = (
                []
                if input_set == "confounds_only"
                else select_feature_columns(
                    data,
                    aggregate_set=aggregate_set,
                    feature_family=feature_family,
                    channel_ids=args.channel_ids,
                )
            )
            if input_set != "confounds_only" and not eeg_cols:
                log_rows.append({"run_id": run_id, "status": "skipped", "reason": "no selected EEG feature columns"})
                continue

            use_confounds = input_set in {"confounds_only", "eeg_confounds"}
            numeric_extra = ["age"] if use_confounds and "age" in data.columns else []
            categorical_cols = []
            if use_confounds and "sex" in data.columns:
                categorical_cols.append("sex")
            if task == "all_tasks":
                categorical_cols.append("task")

            input_audit_rows = drop_run_rows(input_audit_rows, run_id)
            input_audit_rows.append(
                {
                    "run_id": run_id,
                    "target": target,
                    "task": task,
                    "feature_family": feature_family,
                    "aggregate_set": aggregate_set,
                    "input_set": input_set,
                    "model": model,
                    "channel_ids": ",".join(args.channel_ids or []),
                    "channel_filter_mode": (
                        "exact_channel_prefix" if args.channel_ids else "all_channels"
                    ),
                    "n_eeg_features": int(len(eeg_cols)),
                    "n_numeric_confounds": int(len(numeric_extra)),
                    "n_categorical_confounds": int(len(categorical_cols)),
                    "n_subjects": int(n_subjects),
                    "n_rows": int(len(data)),
                }
            )
            write_table(
                pd.DataFrame(input_audit_rows),
                join_path(args.output_dir, "input_audit.csv"),
                region_name=args.region_name,
                index=False,
            )

            X_cols = eeg_cols + numeric_extra + categorical_cols
            if not X_cols:
                log_rows.append(
                    {
                        "run_id": run_id,
                        "status": "skipped",
                        "reason": f"no columns available for input_set={input_set}",
                    }
                )
                continue
            X = data[X_cols].copy()
            y = data[target].astype(float).to_numpy()
            groups = data["participant_id"].astype(str).to_numpy()
            unique_groups = np.unique(groups)
            outer_splits = min(args.outer_folds, len(unique_groups))
            if outer_splits < 2:
                log_rows.append({"run_id": run_id, "status": "skipped", "reason": "not enough groups for outer CV"})
                continue

            cv_outer = GroupKFold(n_splits=outer_splits)
            for outer_fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y, groups), start=1):
                marker_exists = (
                    args.resume
                    and existing_fold_complete(
                        args.output_dir,
                        run_id,
                        outer_fold,
                        args.region_name,
                    )
                )
                if should_skip_fold(
                    marker_exists,
                    metrics_rows,
                    run_id,
                    outer_fold,
                ):
                    log.info(
                        "Skipping completed fold %s/%s for %s",
                        outer_fold,
                        outer_splits,
                        run_id,
                    )
                    continue

                metrics_rows = drop_fold_rows(
                    metrics_rows, run_id, outer_fold
                )
                prediction_rows = drop_fold_rows(
                    prediction_rows, run_id, outer_fold
                )
                best_rows = drop_fold_rows(best_rows, run_id, outer_fold)
                trial_rows = drop_fold_rows(
                    trial_rows, run_id, outer_fold
                )
                selected_rows = drop_fold_rows(
                    selected_rows, run_id, outer_fold
                )

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                g_train, g_test = groups[train_idx], groups[test_idx]

                objective = make_optuna_objective(
                    X_train,
                    y_train,
                    g_train,
                    model_name=model,
                    allowed_selectors=(
                        ["none"]
                        if model == "dummy" or not eeg_cols
                        else selectors
                    ),
                    primary_metric=args.primary_metric,
                    inner_folds=args.inner_folds,
                    random_state=args.random_state,
                    eeg_feature_cols=eeg_cols,
                    numeric_extra_cols=numeric_extra,
                    categorical_cols=categorical_cols,
                    correlation_threshold=correlation_threshold,
                    selector_k_max=args.selector_k_max,
                    svr_kernels=args.svr_kernels,
                )
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.random_state + outer_fold))
                study.optimize(
                    objective,
                    n_trials=1 if model == "dummy" else args.n_trials,
                    n_jobs=args.n_jobs,
                    show_progress_bar=False,
                    catch=(Exception,),
                )

                best_params = dict(study.best_trial.params)
                selector, selector_params, model_params = best_params_to_components(best_params, model, len(eeg_cols), len(X_train), args.random_state)
                pipe = build_pipeline(
                    eeg_feature_cols=eeg_cols,
                    model_name=model,
                    selector_name=selector,
                    selector_params=selector_params,
                    model_params=model_params,
                    numeric_extra_cols=numeric_extra,
                    categorical_cols=categorical_cols,
                    correlation_threshold=correlation_threshold,
                    random_state=args.random_state,
                )
                pipe.fit(X_train, y_train)
                y_pred = np.asarray(pipe.predict(X_test)).ravel()
                metrics = regression_metrics(y_test, y_pred)

                n_selected = (
                    selector_params.get("k")
                    or selector_params.get("n_components")
                    or len(eeg_cols)
                )
                fitted_eeg = pipe.named_steps[
                    "preprocessor"
                ].named_transformers_.get("eeg")
                if fitted_eeg is not None:
                    fitted_selector = fitted_eeg.named_steps.get(
                        "selector"
                    )
                    if hasattr(fitted_selector, "get_support"):
                        n_selected = int(
                            np.sum(fitted_selector.get_support())
                        )
                meta = {
                    "run_id": run_id,
                    "target": target,
                    "task": task,
                    "feature_family": feature_family,
                    "aggregate_set": aggregate_set,
                    "input_set": input_set,
                    "correlation_threshold": (
                        "none"
                        if correlation_threshold is None
                        else f"{correlation_threshold:g}"
                    ),
                    "model": model,
                    "outer_fold": outer_fold,
                    "selector": selector,
                    "n_features_selected": int(n_selected),
                    "n_eeg_features_available": int(len(eeg_cols)),
                }
                metrics_rows.append(
                    {
                        **meta,
                        "primary_metric": args.primary_metric,
                        "best_inner_score": float(study.best_value),
                        **metrics,
                        "n_train_subjects": int(len(np.unique(g_train))),
                        "n_test_subjects": int(len(np.unique(g_test))),
                        "n_train_rows": int(len(train_idx)),
                        "n_test_rows": int(len(test_idx)),
                        "best_params_json": json.dumps(best_params),
                    }
                )
                best_rows.append({**meta, "best_value": float(study.best_value), "best_params_json": json.dumps(best_params)})
                for t in study.trials:
                    trial_rows.append(
                        {
                            **{k: v for k, v in meta.items() if k != "selector" and k != "n_features_selected"},
                            "trial_number": t.number,
                            "trial_value": t.value,
                            "trial_state": str(t.state),
                            "params_json": json.dumps(t.params),
                        }
                    )
                for idx, pred in zip(test_idx, y_pred):
                    prediction_rows.append(
                        {
                            "run_id": run_id,
                            "participant_id": data.loc[idx, "participant_id"],
                            "task": task,
                            "source_task": data.loc[idx, "task"],
                            "target": target,
                            "y_true": float(data.loc[idx, target]),
                            "y_pred": float(pred),
                            "outer_fold": outer_fold,
                            "feature_family": feature_family,
                            "aggregate_set": aggregate_set,
                            "input_set": input_set,
                            "correlation_threshold": (
                                "none"
                                if correlation_threshold is None
                                else f"{correlation_threshold:g}"
                            ),
                            "model": model,
                            "selector": selector,
                        }
                    )
                selected_rows.extend(get_selected_feature_rows(pipe, eeg_cols, meta))
                if args.save_models:
                    dump_joblib(pipe, join_path(args.output_dir, "models", run_id, f"outer_fold={outer_fold}.joblib"), region_name=args.region_name)

                write_outputs(
                    args.output_dir,
                    args.region_name,
                    output_tables(
                        metrics_rows,
                        prediction_rows,
                        best_rows,
                        trial_rows,
                        selected_rows,
                        error_rows,
                        log_rows,
                    ),
                )
                mark_fold_complete(
                    args.output_dir,
                    run_id,
                    outer_fold,
                    args.region_name,
                )

            log_rows.append({"run_id": run_id, "status": "ok", "reason": ""})

            write_outputs(
                args.output_dir,
                args.region_name,
                output_tables(
                    metrics_rows,
                    prediction_rows,
                    best_rows,
                    trial_rows,
                    selected_rows,
                    error_rows,
                    log_rows,
                ),
            )
            mark_complete(args.output_dir, run_id, args.region_name)
        except Exception as exc:
            error_rows.append(
                {
                    "run_id": run_id,
                    "target": target,
                    "task": task,
                    "feature_family": feature_family,
                    "aggregate_set": aggregate_set,
                    "input_set": input_set,
                    "correlation_threshold": (
                        "none"
                        if correlation_threshold is None
                        else f"{correlation_threshold:g}"
                    ),
                    "model": model,
                    "error": str(exc)[:500],
                    "traceback": traceback.format_exc(limit=8)[:3000],
                }
            )
            log.exception("Combination failed: %s", run_id)
            write_outputs(args.output_dir, args.region_name, {"errors": pd.DataFrame(error_rows), "benchmark_log": pd.DataFrame(log_rows)})

    write_outputs(
        args.output_dir,
        args.region_name,
        output_tables(
            metrics_rows,
            prediction_rows,
            best_rows,
            trial_rows,
            selected_rows,
            error_rows,
            log_rows,
        ),
    )
    log.info("Benchmark completed")


if __name__ == "__main__":
    main()
