#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from ml_utils import (
    BASE_META_COLS,
    TARGET_COLS,
    list_aggregated_feature_files,
    read_table,
    write_json,
    write_table,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build ML dataframe from aggregated EEG features on S3.")
    p.add_argument("--input-bucket", default="hbn-features-tesi-ketrin")
    p.add_argument("--input-prefix", default="aggregated_subject_task")
    p.add_argument("--targets-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--region-name", default="us-east-1")
    p.add_argument("--limit-files", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("build_ml_dataset")

    files = list_aggregated_feature_files(args.input_bucket, args.input_prefix, region_name=args.region_name)
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise RuntimeError(f"No features_aggregated.parquet found under s3://{args.input_bucket}/{args.input_prefix}/")

    rows = []
    for item in tqdm(files, desc="Reading aggregated feature files"):
        df = read_table(item["path"], region_name=args.region_name)
        if "participant_id" not in df.columns:
            df.insert(0, "participant_id", item["participant_id"])
        if "task" not in df.columns:
            df.insert(1, "task", item["task"])
        rows.append(df)
    features = pd.concat(rows, ignore_index=True)
    features["participant_id"] = features["participant_id"].astype(str)

    targets = read_table(args.targets_path, region_name=args.region_name)
    if "participant_id" not in targets.columns:
        raise RuntimeError("Target/demography file must contain participant_id.")
    missing_required = [c for c in ["age", "sex"] + TARGET_COLS if c not in targets.columns]
    if missing_required:
        raise RuntimeError(f"Target/demography file is missing required columns: {missing_required}")
    targets["participant_id"] = targets["participant_id"].astype(str)

    keep_target_cols = ["participant_id", "age", "sex"] + TARGET_COLS
    df_ml = features.merge(targets[keep_target_cols], on="participant_id", how="left", validate="many_to_one")
    ordered = BASE_META_COLS + [c for c in df_ml.columns if c not in BASE_META_COLS]
    df_ml = df_ml[[c for c in ordered if c in df_ml.columns]]
    write_table(df_ml, args.output_path, region_name=args.region_name, index=False)

    feature_cols = [c for c in df_ml.columns if c not in set(BASE_META_COLS + ["age", "sex"] + TARGET_COLS)]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_bucket": args.input_bucket,
        "input_prefix": args.input_prefix,
        "targets_path": args.targets_path,
        "output_path": args.output_path,
        "n_subjects": int(df_ml["participant_id"].nunique()),
        "n_rows": int(len(df_ml)),
        "tasks": sorted(df_ml["task"].dropna().astype(str).unique().tolist()),
        "n_feature_cols": int(len(feature_cols)),
        "target_columns": TARGET_COLS,
        "covariate_columns": ["age", "sex"],
        "missing_targets": {c: int(df_ml[c].isna().sum()) for c in TARGET_COLS},
        "n_aggregated_files_read": int(len(files)),
    }
    write_json(manifest, args.manifest_path, region_name=args.region_name)
    log.info("Saved dataframe to %s", args.output_path)
    log.info("Saved manifest to %s", args.manifest_path)


if __name__ == "__main__":
    main()

