"""Read-only availability audit for the masked multi-task E3 cohort."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import boto3
import pandas as pd
from botocore.config import Config

TARGETS = ("attention", "p_factor", "internalizing", "externalizing")
TASKS = (
    "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent",
    "contrastChangeDetection_1", "contrastChangeDetection_2", "contrastChangeDetection_3",
    "seqLearning6target", "seqLearning8target", "surroundSupp_1", "surroundSupp_2", "symbolSearch",
)

def s3_client(region: str):
    return boto3.Session(region_name=region).client("s3", config=Config(retries={"total_max_attempts": 10, "mode": "adaptive"}, read_timeout=300))

def load_targets(client, uri: str, output_dir: Path) -> pd.DataFrame:
    bucket, key = uri[5:].split("/", 1)
    columns = ", ".join(f"s.{name}" for name in ("participant_id", *TARGETS))
    response = client.select_object_content(
        Bucket=bucket, Key=key, ExpressionType="SQL",
        Expression=f"SELECT {columns} FROM S3Object s",
        InputSerialization={"Parquet": {}}, OutputSerialization={"CSV": {}},
    )
    payload = b"".join(event["Records"]["Payload"] for event in response["Payload"] if "Records" in event)
    frame = pd.read_csv(StringIO(payload.decode("utf-8")), header=None, names=["participant_id", *TARGETS])
    frame["participant_id"] = frame.participant_id.astype(str)
    for target in TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    return frame.dropna(subset=list(TARGETS)).drop_duplicates("participant_id").reset_index(drop=True)
def list_task_ids(client, bucket: str, prefix: str, task: str) -> set[str]:
    ids: set[str] = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix.strip('/')}/{task}/"):
        ids.update(Path(item["Key"]).stem for item in page.get("Contents", []) if item["Key"].endswith(".npy"))
    return ids

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--input-bucket", default="hbn-ica-tesi-ketrin")
    parser.add_argument("--input-prefix", default="ica_qamad_ready")
    parser.add_argument("--targets-uri", default="s3://hbn-ml-benchmark-tesi-ketrin/benchmark_v1/data/df_ml.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = s3_client(args.region)
    targets = load_targets(client, args.targets_uri, args.output_dir)
    task_ids = {task: list_task_ids(client, args.input_bucket, args.input_prefix, task) for task in TASKS}
    audit = pd.DataFrame({"participant_id": targets.participant_id, **{task: targets.participant_id.isin(task_ids[task]) for task in TASKS}})
    audit["listed_task_count"] = audit[list(TASKS)].sum(axis=1)
    audit["meets_minimum_two_listed_tasks"] = audit.listed_task_count >= 2
    audit.to_csv(args.output_dir / "availability_by_subject.csv", index=False)
    distribution = audit.listed_task_count.value_counts().sort_index().rename_axis("listed_task_count").reset_index(name="subjects")
    distribution.to_csv(args.output_dir / "availability_distribution.csv", index=False)
    pd.DataFrame({"task": list(TASKS), "subjects_with_listed_signal": [int(audit[t].sum()) for t in TASKS]}).to_csv(args.output_dir / "availability_by_task.csv", index=False)
    summary = {"target_complete_subjects": int(len(audit)), "subjects_with_at_least_two_listed_tasks": int(audit.meets_minimum_two_listed_tasks.sum()), "mean_listed_tasks_per_target_complete_subject": float(audit.listed_task_count.mean()), "median_listed_tasks_per_target_complete_subject": float(audit.listed_task_count.median()), "task_count_distribution": {str(int(row.listed_task_count)): int(row.subjects) for row in distribution.itertuples(index=False)}, "note": "Metadata-only audit. The final run additionally excludes any listed signal with fewer than two readable epochs."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()