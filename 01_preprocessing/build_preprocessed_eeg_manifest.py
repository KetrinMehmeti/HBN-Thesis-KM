"""Build a reproducible raw-EEG input manifest for subject-level modelling.

The script does not read or modify EEG signal contents.  It joins the canonical
feature-extraction inventory with the S3 object listing, validates the exact
NPY storage formula, and writes compact metadata artefacts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import mne
import pandas as pd


BUCKET = "hbn-ica-tesi-ketrin"
PREFIX = "ica_qamad_ready"
SFREQ_HZ = 200
N_CHANNELS = 128
N_SAMPLES = 2_000
DTYPE_BYTES = 4  # float32
TASK_CLASS = {
    "RestingState": "resting_state",
    "DespicableMe": "passive_movie_watching",
    "DiaryOfAWimpyKid": "passive_movie_watching",
    "FunwithFractals": "passive_movie_watching",
    "ThePresent": "passive_movie_watching",
    "contrastChangeDetection_1": "active_task",
    "contrastChangeDetection_2": "active_task",
    "contrastChangeDetection_3": "active_task",
    "seqLearning6target": "active_task",
    "seqLearning8target": "active_task",
    "surroundSupp_1": "active_task",
    "surroundSupp_2": "active_task",
    "symbolSearch": "active_task",
}


def parse_s3_listing(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) != 4 or not fields[3].endswith(".npy"):
            continue
        date, clock, size, key = fields
        rel = key.removeprefix(f"{PREFIX}/")
        task, filename = rel.split("/", maxsplit=1)
        rows.append(
            {
                "s3_key": key,
                "s3_uri": f"s3://{BUCKET}/{key}",
                "task": task,
                "participant_id": Path(filename).stem,
                "size_bytes": int(size),
                "s3_last_modified_utc": f"{date}T{clock}Z",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No NPY objects found in {path}")
    return result


def channel_order(channel_info: Path) -> dict[str, object]:
    with channel_info.open("rb") as handle:
        actual = [channel["ch_name"] for channel in pickle.load(handle)]
    expected = mne.channels.make_standard_montage("GSN-HydroCel-128").ch_names
    exact = actual == expected
    if not exact:
        raise ValueError("channel_info.pkl does not match GSN-HydroCel-128")
    sequence = "\n".join(actual)
    return {
        "montage": "GSN-HydroCel-128",
        "channel_count": len(actual),
        "channel_names_in_array_order": actual,
        "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        "matches_mne_standard_montage_exactly": exact,
        "coordinate_reference": "head",
        "validation_scope": (
            "The static channel-info artefact exactly matches the standard "
            "HydroCel 128 sequence. All arrays have 128 channels. NPY files "
            "cannot themselves retain labels, so this establishes the required "
            "array convention but does not replace a per-source-header audit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--s3-listing", type=Path, required=True)
    parser.add_argument("--channel-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_parquet(args.inventory, columns=["participant_id", "task", "n_epochs"])
    objects = parse_s3_listing(args.s3_listing)
    if inventory.duplicated(["participant_id", "task"]).any():
        raise ValueError("Canonical inventory contains duplicate participant-task rows")
    if objects.duplicated(["participant_id", "task"]).any():
        raise ValueError("S3 listing contains duplicate participant-task files")
    manifest = objects.merge(inventory, on=["participant_id", "task"], how="outer", indicator=True)
    if not (manifest["_merge"] == "both").all():
        counts = manifest["_merge"].value_counts().to_dict()
        raise ValueError(f"Inventory and S3 listing disagree: {counts}")
    manifest = manifest.drop(columns="_merge")
    expected_size = manifest["n_epochs"] * N_CHANNELS * N_SAMPLES * DTYPE_BYTES + 128
    manifest["n_channels"] = N_CHANNELS
    manifest["n_samples_per_epoch"] = N_SAMPLES
    manifest["sampling_hz"] = SFREQ_HZ
    manifest["dtype"] = "float32"
    manifest["epoch_duration_seconds"] = N_SAMPLES / SFREQ_HZ
    manifest["task_class"] = manifest["task"].map(TASK_CLASS)
    manifest["eligible_min_2_epochs"] = manifest["n_epochs"] >= 2
    manifest["eligible_min_8_epochs"] = manifest["n_epochs"] >= 8
    manifest["expected_npy_size_bytes"] = expected_size.astype("int64")
    manifest["structural_validation"] = manifest["size_bytes"].eq(expected_size).map(
        {True: "passed_npy_size_formula", False: "failed_npy_size_formula"}
    )
    order = channel_order(args.channel_info)
    manifest["channel_order_id"] = order["sequence_sha256"]
    manifest["record_id"] = manifest["s3_key"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    )
    if manifest["task_class"].isna().any():
        raise ValueError("One or more tasks lack a class mapping")
    if not (manifest["structural_validation"] == "passed_npy_size_formula").all():
        raise ValueError("One or more NPY objects fail the stored-size validation")

    columns = [
        "record_id", "participant_id", "task", "task_class", "s3_uri", "s3_key",
        "s3_last_modified_utc", "size_bytes", "n_epochs", "n_channels",
        "n_samples_per_epoch", "sampling_hz", "epoch_duration_seconds", "dtype",
        "eligible_min_2_epochs", "eligible_min_8_epochs", "structural_validation",
        "expected_npy_size_bytes", "channel_order_id",
    ]
    manifest = manifest[columns].sort_values(["participant_id", "task"]).reset_index(drop=True)
    manifest.to_csv(args.output_dir / "records.csv", index=False)
    manifest.to_parquet(args.output_dir / "records.parquet", index=False)

    task_summary = (
        manifest.groupby(["task", "task_class"], as_index=False)
        .agg(
            registrations=("record_id", "count"),
            unique_subjects=("participant_id", "nunique"),
            total_epochs=("n_epochs", "sum"),
            median_epochs=("n_epochs", "median"),
            min_epochs=("n_epochs", "min"),
            max_epochs=("n_epochs", "max"),
            eligible_min_2_epochs=("eligible_min_2_epochs", "sum"),
            eligible_min_8_epochs=("eligible_min_8_epochs", "sum"),
        )
        .sort_values("task")
    )
    task_summary.to_csv(args.output_dir / "task_summary.csv", index=False)
    (args.output_dir / "channel_order.json").write_text(json.dumps(order, indent=2), encoding="utf-8")
    manifest_sha = hashlib.sha256((args.output_dir / "records.csv").read_bytes()).hexdigest()
    metadata = {
        "manifest_version": "eegnet_raw_input_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_bucket": BUCKET,
        "source_prefix": PREFIX,
        "records": int(len(manifest)),
        "unique_subjects": int(manifest["participant_id"].nunique()),
        "tasks": int(manifest["task"].nunique()),
        "total_epochs": int(manifest["n_epochs"].sum()),
        "records_with_at_least_2_epochs": int(manifest["eligible_min_2_epochs"].sum()),
        "records_with_at_least_8_epochs": int(manifest["eligible_min_8_epochs"].sum()),
        "records_csv_sha256": manifest_sha,
        "channel_order": order,
        "storage_validation": "Every row passed the exact NPY size formula for float32 (epochs, 128, 2000).",
        "preprocessing": {
            "sampling_hz": SFREQ_HZ,
            "epoch_duration_seconds": 10,
            "array_shape": "(n_epochs, 128, 2000)",
            "normalization": "microvolts divided by 100, stored as float32",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    readme = """# Preprocessed EEG input manifest\n\nThis manifest indexes every `.npy` recording under `s3://hbn-ica-tesi-ketrin/ica_qamad_ready/`. It contains metadata only and never copies or changes EEG signals.\n\n- `records.csv` / `records.parquet`: one subject-task registration per row.\n- `task_summary.csv`: coverage and epoch counts by task.\n- `channel_order.json`: the canonical array-axis order, E1 through E128, matching MNE `GSN-HydroCel-128`.\n- `manifest.json`: reproducibility metadata and file checksums.\n\nAll records passed an exact storage-size check for a little-endian NumPy `float32` array with shape `(n_epochs, 128, 2000)`. Eligibility flags are deliberately descriptive: the downstream protocol may choose `min_2_epochs`, `min_8_epochs`, or another pre-registered threshold.\n\nImportant: NumPy arrays do not retain channel labels. The canonical order is bound to every row through `channel_order_id`; future raw-signal pipelines must use this sequence and must not reorder channels independently.\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
