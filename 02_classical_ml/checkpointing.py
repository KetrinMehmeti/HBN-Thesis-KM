from typing import Any, Iterable


def feature_names_after_imputation(imputer, feature_names):
    """Return feature names aligned to the imputer output columns."""
    names = list(feature_names)
    if imputer is not None and hasattr(imputer, "get_feature_names_out"):
        return list(imputer.get_feature_names_out(names))
    return names


def fold_checkpoint_key(run_id: str, outer_fold: int) -> str:
    return f"{run_id}__outer_fold={outer_fold}.json"


def row_matches_fold(
    row: dict[str, Any],
    run_id: str,
    outer_fold: int,
) -> bool:
    return (
        row.get("run_id") == run_id
        and str(row.get("outer_fold")) == str(outer_fold)
    )


def drop_fold_rows(
    rows: Iterable[dict[str, Any]],
    run_id: str,
    outer_fold: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row_matches_fold(row, run_id, outer_fold)
    ]


def has_fold_result(
    metric_rows: Iterable[dict[str, Any]],
    run_id: str,
    outer_fold: int,
) -> bool:
    return any(
        row_matches_fold(row, run_id, outer_fold)
        for row in metric_rows
    )


def should_skip_fold(
    marker_exists: bool,
    metric_rows: Iterable[dict[str, Any]],
    run_id: str,
    outer_fold: int,
) -> bool:
    return marker_exists and has_fold_result(
        metric_rows,
        run_id,
        outer_fold,
    )
