"""Feature preparation, TTL/testbed shortcut ablation, and LOAO partitioning."""

import numpy as np
import pandas as pd

from src.core.config import (
    ATTACK_FAMILY_COLUMN,
    CATEGORICAL_FLOW_COLUMNS,
    NORMAL_FAMILY_LABEL,
    TARGET_COLUMN,
    TESTBED_SHORTCUT_COLUMNS,
)
from src.core.contracts import FlowFeatureMatrix

_UNSEEN_CATEGORY_TOKEN = "__UNSEEN__"


def _select_feature_columns(
    columns: list[str], ablate_shortcuts: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    excluded = {TARGET_COLUMN, ATTACK_FAMILY_COLUMN}
    shortcuts = set(TESTBED_SHORTCUT_COLUMNS) if ablate_shortcuts else set()
    kept = tuple(c for c in columns if c not in excluded and c not in shortcuts)
    dropped = tuple(c for c in columns if c in shortcuts)
    return kept, dropped


def _align_categorical_columns(
    x_train: pd.DataFrame, x_test: pd.DataFrame, cat_cols: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Fit vocabulary strictly on training split; unseen test values map to fallback token.
    train_copy = x_train.copy()
    test_copy = x_test.copy()
    for col in cat_cols:
        train_str = train_copy[col].astype(str)
        vocab = sorted(set(train_str.unique()).union({_UNSEEN_CATEGORY_TOKEN}))
        test_str = test_copy[col].astype(str).where(
            test_copy[col].astype(str).isin(vocab), _UNSEEN_CATEGORY_TOKEN
        )
        train_copy[col] = pd.Categorical(train_str, categories=vocab)
        test_copy[col] = pd.Categorical(test_str, categories=vocab)
    return train_copy, test_copy


def _cast_numeric_columns(
    frame: pd.DataFrame, numeric_cols: list[str]
) -> pd.DataFrame:
    result = frame.copy()
    for col in numeric_cols:
        result[col] = result[col].astype(np.float32)
    return result


def build_flow_feature_matrix(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_mode: str
) -> FlowFeatureMatrix:
    """Build leak-free train/test feature frames for full or shortcut-ablated mode."""
    if feature_mode not in {"full_flow", "shortcut_ablated"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    ablate_shortcuts = feature_mode == "shortcut_ablated"
    kept_cols, dropped_cols = _select_feature_columns(
        list(train_df.columns), ablate_shortcuts
    )
    cat_cols = tuple(c for c in CATEGORICAL_FLOW_COLUMNS if c in kept_cols)
    num_cols = [c for c in kept_cols if c not in cat_cols]

    x_train = _cast_numeric_columns(train_df[list(kept_cols)], num_cols)
    x_test = _cast_numeric_columns(test_df[list(kept_cols)], num_cols)
    x_train, x_test = _align_categorical_columns(x_train, x_test, cat_cols)

    return FlowFeatureMatrix(
        feature_mode=feature_mode,
        feature_names=kept_cols,
        categorical_columns=cat_cols,
        dropped_shortcut_columns=dropped_cols,
        x_train=x_train,
        x_test=x_test,
        y_train=train_df[TARGET_COLUMN].to_numpy(dtype=np.int32),
        y_test=test_df[TARGET_COLUMN].to_numpy(dtype=np.int32),
        attack_cat_train=train_df[ATTACK_FAMILY_COLUMN].astype(str).to_numpy(),
        attack_cat_test=test_df[ATTACK_FAMILY_COLUMN].astype(str).to_numpy(),
    )


def build_loao_partition(
    train_df: pd.DataFrame, test_df: pd.DataFrame, held_out_family: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude `held_out_family` from train and evaluate against holdout Normal + family."""
    if held_out_family == NORMAL_FAMILY_LABEL:
        raise ValueError("Cannot hold out Normal traffic in zero-day LOAO evaluation.")

    train_mask = train_df[ATTACK_FAMILY_COLUMN].astype(str) != held_out_family
    test_mask = test_df[ATTACK_FAMILY_COLUMN].astype(str).isin(
        {NORMAL_FAMILY_LABEL, held_out_family}
    )
    loao_train = train_df.loc[train_mask].reset_index(drop=True)
    loao_test = test_df.loc[test_mask].reset_index(drop=True)
    if (loao_test[ATTACK_FAMILY_COLUMN].astype(str) == held_out_family).sum() == 0:
        raise ValueError(f"Held-out family missing from test split: {held_out_family}")
    return loao_train, loao_test
