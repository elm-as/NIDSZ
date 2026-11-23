"""Tests for SOC low-FPR thresholds, probability calibration ECE, and latency."""

import numpy as np
import pandas as pd

from src.evaluation.calibration_and_metrics import (
    compute_ece_from_bins,
    compute_holdout_metrics,
    compute_reliability_bins,
)
from src.evaluation.latency_profiler import (
    LatencyProfileRequest,
    benchmark_batch_inference,
)
from src.evaluation.soc_operating_points import (
    compute_soc_operating_point,
    evaluate_soc_budgets,
    sample_log_spaced_roc_curve,
)


def test_low_fpr_thresholds_enforce_ceiling_and_monotonic_recall() -> None:
    """Actual FPR must not exceed target ceiling and recall must be non-decreasing."""
    rng = np.random.default_rng(42)
    n_neg, n_pos = 10_000, 2_000
    y_true = np.concatenate([np.zeros(n_neg, dtype=np.int32), np.ones(n_pos, dtype=np.int32)])
    neg_scores = rng.beta(1.5, 8.0, size=n_neg)
    pos_scores = rng.beta(6.0, 2.0, size=n_pos)
    scores = np.concatenate([neg_scores, pos_scores])

    budgets = (0.001, 0.005, 0.01)
    points = evaluate_soc_budgets(y_true, scores, budgets)
    assert len(points) == 3

    for point, ceiling in zip(points, budgets):
        assert point.actual_fpr <= ceiling + 1e-12
        assert 0.0 <= point.recall_tpr <= 1.0
        assert point.alerts_per_100k_benign <= ceiling * 100_000.0 + 1e-6

    assert points[0].recall_tpr <= points[1].recall_tpr <= points[2].recall_tpr


def test_perfect_separation_reaches_full_recall_at_strict_fpr() -> None:
    """Separable scores must achieve 100% recall even at FPR <= 0.1%."""
    y_true = np.array([0] * 1000 + [1] * 200, dtype=np.int32)
    scores = np.array([0.05] * 1000 + [0.95] * 200, dtype=np.float64)
    point = compute_soc_operating_point(y_true, scores, 0.001)
    assert point.actual_fpr == 0.0
    assert point.recall_tpr == 1.0
    assert point.false_positives == 0


def test_calibration_ece_and_holdout_metrics_bounds() -> None:
    """Perfectly calibrated probabilities yield zero ECE and valid metrics."""
    y_true = np.array([0] * 90 + [1] * 10 + [0] * 10 + [1] * 90, dtype=np.int32)
    probs = np.array([0.10] * 100 + [0.90] * 100, dtype=np.float64)
    bins = compute_reliability_bins(y_true, probs, num_bins=10)
    ece = compute_ece_from_bins(bins)
    assert ece < 1e-6

    metrics, _ = compute_holdout_metrics(y_true, probs, num_bins=10)
    assert 0.5 < metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.brier_score <= 1.0
    assert metrics.ece_uniform_10 < 1e-6


def test_log_spaced_roc_and_latency_profiler_contract() -> None:
    """Verify ROC grid monotonicity and positive throughput/latency measurements."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
    fpr, tpr = sample_log_spaced_roc_curve(y_true, scores, num_points=40)
    assert len(fpr) == len(tpr)
    assert all(a <= b for a, b in zip(fpr, fpr[1:]))
    assert all(a <= b for a, b in zip(tpr, tpr[1:]))

    frame = pd.DataFrame({"x": np.arange(200, dtype=np.float32)})
    bench = benchmark_batch_inference(
        lambda df: df["x"].to_numpy() * 0.01,
        frame,
        LatencyProfileRequest(
            model_name="dummy", feature_mode="full_flow", batch_size=100, repeats=3
        ),
    )
    assert bench.batch_size == 100
    assert bench.throughput_flows_per_sec > 0.0
    assert bench.latency_us_per_flow > 0.0
