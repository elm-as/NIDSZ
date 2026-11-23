"""Holdout discrimination metrics, Brier score, and 10-bin ECE calibration."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from src.core.contracts import HoldoutMetrics, ReliabilityBin


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)


def compute_reliability_bins(
    y_true: np.ndarray, probabilities: np.ndarray, num_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    """Partition predictions into `num_bins` uniform intervals and return bin stats."""
    probs = _clip_probabilities(probabilities)
    targets = np.asarray(y_true, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[ReliabilityBin] = []
    for idx in range(num_bins):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        mask = (probs >= low) & (probs <= high if idx == num_bins - 1 else probs < high)
        count = int(np.sum(mask))
        mean_pred = float(np.mean(probs[mask])) if count > 0 else 0.0
        emp_rate = float(np.mean(targets[mask])) if count > 0 else 0.0
        bins.append(
            ReliabilityBin(
                bin_index=idx,
                lower_bound=low,
                upper_bound=high,
                sample_count=count,
                mean_predicted=mean_pred,
                empirical_rate=emp_rate,
            )
        )
    return tuple(bins)


def compute_ece_from_bins(bins: tuple[ReliabilityBin, ...]) -> float:
    """Compute weighted Expected Calibration Error (ECE) from reliability bins."""
    total = sum(b.sample_count for b in bins)
    if total == 0:
        return 0.0
    weighted_gap = sum(
        b.sample_count * abs(b.empirical_rate - b.mean_predicted) for b in bins
    )
    return float(weighted_gap / total)


def _compute_optimal_f1(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.0, 0.5
    denom = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1_values))
    return float(f1_values[best_idx]), float(thresholds[best_idx])


def compute_holdout_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, num_bins: int = 10
) -> tuple[HoldoutMetrics, tuple[ReliabilityBin, ...]]:
    """Compute ROC-AUC, PR-AUC, F1, Log-Loss, Brier, ECE-10, and reliability bins."""
    probs = _clip_probabilities(probabilities)
    targets = np.asarray(y_true, dtype=np.int32)
    bins = compute_reliability_bins(targets, probs, num_bins=num_bins)
    ece_10 = compute_ece_from_bins(bins)
    f1_opt, opt_tau = _compute_optimal_f1(targets, probs)
    metrics = HoldoutMetrics(
        roc_auc=float(roc_auc_score(targets, probs)),
        pr_auc=float(average_precision_score(targets, probs)),
        f1_default_050=float(f1_score(targets, (probs >= 0.5).astype(np.int32))),
        f1_optimal=f1_opt,
        optimal_f1_threshold=opt_tau,
        log_loss=float(log_loss(targets, probs)),
        brier_score=float(brier_score_loss(targets, probs)),
        ece_uniform_10=ece_10,
    )
    return metrics, bins
