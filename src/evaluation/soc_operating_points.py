"""SOC alert-fatigue low-FPR operating points and ROC curve extraction."""

import numpy as np
from sklearn.metrics import roc_curve

from src.core.contracts import SocOperatingPoint


def _find_threshold_index_for_fpr(fpr: np.ndarray, target_fpr: float) -> int:
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return 0
    return int(valid_indices[-1])


def compute_soc_operating_point(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
) -> SocOperatingPoint:
    """Compute exact TPR/precision/threshold under strict FPR <= `target_fpr`."""
    if not 0.0 < target_fpr <= 1.0:
        raise ValueError(f"target_fpr must lie in (0, 1], got {target_fpr}")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = _find_threshold_index_for_fpr(fpr, target_fpr)
    threshold = float(thresholds[idx])
    predicted = (y_score >= threshold).astype(np.int32)

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    true_pos = int(np.sum((predicted == 1) & (y_true == 1)))
    false_pos = int(np.sum((predicted == 1) & (y_true == 0)))

    actual_fpr = float(false_pos / negatives) if negatives > 0 else 0.0
    recall_tpr = float(true_pos / positives) if positives > 0 else 0.0
    denom = true_pos + false_pos
    precision = float(true_pos / denom) if denom > 0 else 0.0
    return SocOperatingPoint(
        target_fpr=float(target_fpr),
        actual_fpr=actual_fpr,
        recall_tpr=recall_tpr,
        precision=precision,
        threshold=threshold,
        alerts_per_100k_benign=float(actual_fpr * 100_000.0),
        true_positives=true_pos,
        false_positives=false_pos,
    )


def evaluate_soc_budgets(
    y_true: np.ndarray, y_score: np.ndarray, fpr_budgets: tuple[float, ...]
) -> tuple[SocOperatingPoint, ...]:
    """Evaluate all requested SOC false-alarm budgets on `(y_true, y_score)`."""
    return tuple(
        compute_soc_operating_point(y_true, y_score, budget)
        for budget in fpr_budgets
    )


def sample_log_spaced_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray, num_points: int = 160
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Interpolate ROC curve on a log-spaced FPR grid focused on low-FPR SOC zones."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    grid = np.unique(
        np.concatenate(
            [
                np.array([0.0]),
                np.logspace(-4.5, -1.0, num_points // 2),
                np.linspace(0.1, 1.0, num_points // 2),
            ]
        )
    )
    interpolated_tpr = np.interp(grid, fpr, tpr)
    return (
        tuple(float(v) for v in grid),
        tuple(float(v) for v in interpolated_tpr),
    )
