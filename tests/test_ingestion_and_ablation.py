"""Tests for UNSW-NB15 schema validation, shortcut ablation, and zero-day LOAO."""

import numpy as np
import pandas as pd
import pytest

from src.core.config import (
    ATTACK_FAMILY_COLUMN,
    TESTBED_SHORTCUT_COLUMNS,
    resolve_project_paths,
)
from src.features.flow_preprocessor import (
    build_flow_feature_matrix,
    build_loao_partition,
)
from src.ingestion.flow_schema_validator import load_and_validate_flows
from src.modeling.estimator_factory import (
    EstimatorSpec,
    build_unsupervised_detector,
)


def _make_synthetic_flow_frame(n_rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    families = np.array(["Normal", "Exploits", "Worms", "DoS"])
    chosen = families[np.arange(n_rows) % len(families)]
    labels = (chosen != "Normal").astype(np.int8)
    return pd.DataFrame(
        {
            "dur": rng.uniform(0.01, 2.0, n_rows).astype(np.float32),
            "proto": pd.Categorical(rng.choice(["tcp", "udp"], n_rows)),
            "service": pd.Categorical(rng.choice(["http", "-"], n_rows)),
            "state": pd.Categorical(rng.choice(["FIN", "INT"], n_rows)),
            "sbytes": rng.integers(60, 4000, n_rows, dtype=np.int32),
            "dbytes": rng.integers(0, 3000, n_rows, dtype=np.int32),
            "swin": rng.choice([0, 255], n_rows).astype(np.int16),
            "ct_dst_sport_ltm": rng.integers(1, 10, n_rows, dtype=np.int8),
            "sttl": rng.choice([31, 62, 254], n_rows).astype(np.int16),
            "attack_cat": pd.Categorical(chosen),
            "label": labels,
        }
    )


def test_official_parquet_partition_integrity() -> None:
    """Verify official UNSW-NB15 parquets load cleanly with exact expected shapes."""
    paths = resolve_project_paths()
    loaded = load_and_validate_flows(paths.train_parquet, paths.test_parquet)
    rep = loaded.schema_report
    assert rep.train_rows == 175_341
    assert rep.test_rows == 82_332
    assert rep.total_rows == 257_673
    assert rep.column_count == 36
    assert rep.missing_values_total == 0
    assert len(rep.train_family_counts) == 10
    assert len(rep.test_family_counts) == 10


def test_shortcut_ablation_removes_ttl_and_testbed_columns() -> None:
    """Shortcut-ablated feature matrix must drop TTL and testbed artifact columns."""
    df_train = _make_synthetic_flow_frame(40)
    df_test = _make_synthetic_flow_frame(20)
    full_mat = build_flow_feature_matrix(df_train, df_test, "full_flow")
    abl_mat = build_flow_feature_matrix(df_train, df_test, "shortcut_ablated")

    assert "sttl" in full_mat.feature_names
    assert "state" in full_mat.feature_names
    assert set(abl_mat.feature_names).isdisjoint(set(TESTBED_SHORTCUT_COLUMNS))
    assert "sttl" in abl_mat.dropped_shortcut_columns
    assert "swin" in abl_mat.dropped_shortcut_columns


def test_unseen_test_category_mapped_without_leakage() -> None:
    """Categories absent from train must map safely to __UNSEEN__ in test split."""
    df_train = _make_synthetic_flow_frame(24)
    df_test = _make_synthetic_flow_frame(12)
    df_test["proto"] = pd.Categorical(["sctp_novel"] * len(df_test))

    matrix = build_flow_feature_matrix(df_train, df_test, "full_flow")
    assert "sctp_novel" not in list(matrix.x_train["proto"].cat.categories)
    assert set(matrix.x_test["proto"].astype(str)) == {"__UNSEEN__"}


def test_loao_partition_enforces_zero_exposure_to_held_out_family() -> None:
    """LOAO training split must contain zero rows of the held-out zero-day family."""
    df_train = _make_synthetic_flow_frame(40)
    df_test = _make_synthetic_flow_frame(40)
    loao_train, loao_test = build_loao_partition(df_train, df_test, "Worms")

    assert "Worms" not in set(loao_train[ATTACK_FAMILY_COLUMN].astype(str))
    assert set(loao_test[ATTACK_FAMILY_COLUMN].astype(str)) == {"Normal", "Worms"}

    with pytest.raises(ValueError):
        build_loao_partition(df_train, df_test, "Normal")


def test_unsupervised_isolation_forest_trains_only_on_normal_flows() -> None:
    """NormalOnlyIsolationDetector must fail if no Normal rows exist and score [0, 1]."""
    df_train = _make_synthetic_flow_frame(40)
    df_test = _make_synthetic_flow_frame(20)
    matrix = build_flow_feature_matrix(df_train, df_test, "shortcut_ablated")
    spec = EstimatorSpec(
        model_name="isolation_forest",
        feature_names=matrix.feature_names,
        categorical_columns=matrix.categorical_columns,
        random_seed=42,
    )
    detector = build_unsupervised_detector(spec)
    with pytest.raises(ValueError):
        detector.fit(matrix.x_train, np.ones_like(matrix.y_train))

    detector.fit(matrix.x_train, matrix.y_train)
    scores = detector.predict_anomaly_score(matrix.x_test)
    assert scores.shape == (len(df_test),)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
