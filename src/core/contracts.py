"""Immutable domain dataclasses shared across NIDSZ layers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FlowSchemaReport:
    """Audit summary of official UNSW-NB15 train and holdout test partitions."""

    train_rows: int
    test_rows: int
    total_rows: int
    column_count: int
    predictor_count: int
    train_attack_rate: float
    test_attack_rate: float
    missing_values_total: int
    train_exact_duplicates: int
    test_exact_duplicates: int
    train_unique_feature_vectors: int
    test_unique_feature_vectors: int
    cross_split_overlap_signatures: int
    cross_split_overlap_test_rows: int
    cross_split_overlap_rate: float
    ttl_columns_present: tuple[str, ...]
    testbed_shortcuts_present: tuple[str, ...]
    train_family_counts: dict[str, int]
    test_family_counts: dict[str, int]


@dataclass(frozen=True)
class LoadedFlowPartitions:
    """Validated official UNSW-NB15 train/test frames paired with schema audit."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    schema_report: FlowSchemaReport


@dataclass(frozen=True)
class FlowFeatureMatrix:
    """Prepared feature frames for a specific ablation regime."""

    feature_mode: str
    feature_names: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    dropped_shortcut_columns: tuple[str, ...]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    attack_cat_train: np.ndarray
    attack_cat_test: np.ndarray


@dataclass(frozen=True)
class ReliabilityBin:
    """Single bin of a uniform reliability calibration curve."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_predicted: float
    empirical_rate: float


@dataclass(frozen=True)
class SocOperatingPoint:
    """Alert-fatigue operating point evaluated under a strict FPR ceiling."""

    target_fpr: float
    actual_fpr: float
    recall_tpr: float
    precision: float
    threshold: float
    alerts_per_100k_benign: float
    true_positives: int
    false_positives: int


@dataclass(frozen=True)
class HoldoutMetrics:
    """Discrimination and probability-calibration metrics on official holdout."""

    roc_auc: float
    pr_auc: float
    f1_default_050: float
    f1_optimal: float
    optimal_f1_threshold: float
    log_loss: float
    brier_score: float
    ece_uniform_10: float


@dataclass(frozen=True)
class LatencyBenchmark:
    """Single-batch inference speed benchmark for SOC line-rate feasibility."""

    model_name: str
    feature_mode: str
    batch_size: int
    median_ms: float
    throughput_flows_per_sec: float
    latency_us_per_flow: float


@dataclass(frozen=True)
class ModelHoldoutArtifact:
    """Complete evaluation artifact for one model and calibration setting."""

    feature_mode: str
    model_name: str
    calibration_method: str
    metrics: HoldoutMetrics
    soc_operating_points: tuple[SocOperatingPoint, ...]
    reliability_bins_10: tuple[ReliabilityBin, ...]
    latency: LatencyBenchmark
    roc_curve_fpr: tuple[float, ...]
    roc_curve_tpr: tuple[float, ...]


@dataclass(frozen=True)
class LoaoFamilyResult:
    """Zero-Day Leave-One-Attack-Family-Out generalization result for one family."""

    family_name: str
    test_attack_count: int
    test_normal_count: int
    full_roc_auc: float
    ablated_roc_auc: float
    iso_forest_roc_auc: float
    full_recall_at_001: float
    full_recall_at_005: float
    full_recall_at_010: float
    ablated_recall_at_001: float
    ablated_recall_at_005: float
    ablated_recall_at_010: float
    iso_recall_at_001: float
    iso_recall_at_005: float
    iso_recall_at_010: float


@dataclass(frozen=True)
class ShapFeatureImpact:
    """Global TreeSHAP magnitude and telemetry category for a network feature."""

    feature_name: str
    feature_group: str
    mean_abs_shap: float


@dataclass(frozen=True)
class ShapAuditSummary:
    """Comparison of TreeSHAP attributions between Full and Ablated models."""

    sample_size: int
    shortcut_shap_share: float
    full_top_features: tuple[ShapFeatureImpact, ...]
    ablated_top_features: tuple[ShapFeatureImpact, ...]


@dataclass(frozen=True)
class NidszReportPayload:
    """Top-level payload passed to exporters, plots, and HTML dashboard."""

    schema_report: FlowSchemaReport
    full_artifacts: tuple[ModelHoldoutArtifact, ...]
    ablated_artifacts: tuple[ModelHoldoutArtifact, ...]
    loao_results: tuple[LoaoFamilyResult, ...]
    shap_summary: ShapAuditSummary
    best_full_model: str
    best_full_calibration: str
    best_ablated_model: str
    best_ablated_calibration: str
    random_seed: int
