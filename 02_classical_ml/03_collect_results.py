#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import pandas as pd

from ml_utils import join_path, read_table, write_table


def parse_args():
    p = argparse.ArgumentParser(description="Collect and summarize EEG benchmark results.")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--primary-metric", default="ccc_mean")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--region-name", default="us-east-1")
    return p.parse_args()


def mode_or_nan(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else pd.NA


def main():
    args = parse_args()
    metrics = read_table(join_path(args.results_dir, "metrics_by_fold.csv"), region_name=args.region_name)
    selected = None
    try:
        selected = read_table(join_path(args.results_dir, "selected_features.csv"), region_name=args.region_name)
    except Exception:
        selected = pd.DataFrame()

    group_cols = ["target", "task", "feature_family", "aggregate_set", "model"]
    summary = (
        metrics.groupby(group_cols)
        .agg(
            selector_most_frequent=("selector", mode_or_nan),
            n_features_selected_mean=("n_features_selected", "mean"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            ccc_mean=("ccc", "mean"),
            ccc_std=("ccc", "std"),
            n_outer_folds=("outer_fold", "nunique"),
            n_subjects_mean=("n_test_subjects", "mean"),
            n_rows_mean=("n_test_rows", "mean"),
        )
        .reset_index()
    )
    write_table(summary, join_path(args.output_dir, "benchmark_summary.csv"), region_name=args.region_name, index=False)

    ranking = summary.sort_values(["target", "task", args.primary_metric], ascending=[True, True, False]).copy()
    ranking["rank"] = ranking.groupby(["target", "task"])[args.primary_metric].rank(method="first", ascending=False)
    write_table(ranking, join_path(args.output_dir, "model_ranking.csv"), region_name=args.region_name, index=False)

    if selected is not None and len(selected):
        stability_group = group_cols + ["feature_name"]
        folds = selected.groupby(group_cols)["outer_fold"].nunique().rename("n_folds").reset_index()
        counts = selected.groupby(stability_group).size().rename("selection_count").reset_index()
        stability = counts.merge(folds, on=group_cols, how="left")
        stability["selection_frequency"] = stability["selection_count"] / stability["n_folds"].replace(0, pd.NA)
    else:
        stability = pd.DataFrame(columns=group_cols + ["feature_name", "selection_count", "n_folds", "selection_frequency"])
    write_table(stability, join_path(args.output_dir, "feature_stability.csv"), region_name=args.region_name, index=False)

    best = (
        summary.sort_values(["target", args.primary_metric], ascending=[True, False])
        .groupby("target", group_keys=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )
    write_table(best, join_path(args.output_dir, "best_overall.csv"), region_name=args.region_name, index=False)


if __name__ == "__main__":
    main()

