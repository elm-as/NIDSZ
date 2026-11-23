"""Official UNSW-NB15 parquet loader, schema validator, and shortcut auditor."""

from pathlib import Path

import pandas as pd

from src.core.config import (
    ATTACK_FAMILY_COLUMN,
    NORMAL_FAMILY_LABEL,
    TARGET_COLUMN,
    TESTBED_SHORTCUT_COLUMNS,
    TTL_SHORTCUT_COLUMNS,
)
from src.core.contracts import FlowSchemaReport, LoadedFlowPartitions


def _validate_partition_alignment(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Train and test parquet schemas do not match exactly.")
    required = {TARGET_COLUMN, ATTACK_FAMILY_COLUMN}
    if not required.issubset(set(train_df.columns)):
        raise ValueError(f"Missing required target columns: {required}")
    missing_total = int(train_df.isna().sum().sum() + test_df.isna().sum().sum())
    if missing_total > 0:
        raise ValueError(f"Unexpected missing values detected: {missing_total}")


def _validate_binary_label_consistency(frame: pd.DataFrame, split_name: str) -> None:
    is_normal_cat = frame[ATTACK_FAMILY_COLUMN].astype(str) == NORMAL_FAMILY_LABEL
    is_zero_label = frame[TARGET_COLUMN].astype(int) == 0
    mismatches = int((is_normal_cat != is_zero_label).sum())
    if mismatches > 0:
        raise ValueError(
            f"Label/attack_cat mismatch in {split_name}: {mismatches} rows."
        )


def _audit_feature_overlap(
    train_df: pd.DataFrame, test_df: pd.DataFrame, predictor_cols: list[str]
) -> tuple[int, int, int, int]:
    train_hashes = pd.util.hash_pandas_object(train_df[predictor_cols], index=False)
    test_hashes = pd.util.hash_pandas_object(test_df[predictor_cols], index=False)
    train_unique = set(train_hashes.tolist())
    test_unique = set(test_hashes.tolist())
    overlap_signatures = len(train_unique.intersection(test_unique))
    overlap_test_rows = int(test_hashes.isin(train_unique).sum())
    return (
        len(train_unique),
        len(test_unique),
        overlap_signatures,
        overlap_test_rows,
    )


def _count_families(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame[ATTACK_FAMILY_COLUMN].astype(str).value_counts()
    return {str(name): int(val) for name, val in counts.items()}


def load_and_validate_flows(
    train_path: Path, test_path: Path
) -> LoadedFlowPartitions:
    """Load official UNSW-NB15 train and test parquets without mixing partitions."""
    if not train_path.exists():
        raise FileNotFoundError(f"Official training parquet missing: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Official testing parquet missing: {test_path}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    _validate_partition_alignment(train_df, test_df)
    _validate_binary_label_consistency(train_df, "train")
    _validate_binary_label_consistency(test_df, "test")

    predictor_cols = [
        col
        for col in train_df.columns
        if col not in {TARGET_COLUMN, ATTACK_FAMILY_COLUMN}
    ]
    (
        train_unique,
        test_unique,
        overlap_signatures,
        overlap_test_rows,
    ) = _audit_feature_overlap(train_df, test_df, predictor_cols)

    ttl_present = tuple(c for c in TTL_SHORTCUT_COLUMNS if c in predictor_cols)
    testbed_present = tuple(
        c for c in TESTBED_SHORTCUT_COLUMNS if c in predictor_cols
    )
    report = FlowSchemaReport(
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        total_rows=int(len(train_df) + len(test_df)),
        column_count=int(train_df.shape[1]),
        predictor_count=len(predictor_cols),
        train_attack_rate=float(train_df[TARGET_COLUMN].mean()),
        test_attack_rate=float(test_df[TARGET_COLUMN].mean()),
        missing_values_total=0,
        train_exact_duplicates=int(train_df.duplicated().sum()),
        test_exact_duplicates=int(test_df.duplicated().sum()),
        train_unique_feature_vectors=train_unique,
        test_unique_feature_vectors=test_unique,
        cross_split_overlap_signatures=overlap_signatures,
        cross_split_overlap_test_rows=overlap_test_rows,
        cross_split_overlap_rate=float(overlap_test_rows / len(test_df)),
        ttl_columns_present=ttl_present,
        testbed_shortcuts_present=testbed_present,
        train_family_counts=_count_families(train_df),
        test_family_counts=_count_families(test_df),
    )
    return LoadedFlowPartitions(
        train_df=train_df, test_df=test_df, schema_report=report
    )
