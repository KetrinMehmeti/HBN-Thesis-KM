from __future__ import annotations

import re
from collections.abc import Iterable


EMBEDDING_COLUMN_RULE = r"^embedding_\d{4}$"
EMBEDDING_COLUMN_PATTERN = re.compile(EMBEDDING_COLUMN_RULE)
EXPECTED_EMBEDDING_DIMENSIONS = {
    "cbramod": 200,
    "biot": 256,
    "labram_128_spatial": 200,
    "eegpt_62_canonical": 512,
    "eegpt_128_spatial": 512,
}


def select_embedding_columns(
    columns: Iterable[object], representation: str
) -> list[str]:
    embedding_columns = sorted(
        str(column)
        for column in columns
        if EMBEDDING_COLUMN_PATTERN.fullmatch(str(column))
    )
    if not embedding_columns:
        raise RuntimeError(f"{representation}: no numbered embedding columns")
    try:
        expected_dimension = EXPECTED_EMBEDDING_DIMENSIONS[representation]
    except KeyError as error:
        raise RuntimeError(
            f"{representation}: missing expected embedding dimension"
        ) from error
    if len(embedding_columns) != expected_dimension:
        raise RuntimeError(
            f"{representation}: expected {expected_dimension} numbered embedding "
            f"columns, observed {len(embedding_columns)}"
        )
    return embedding_columns