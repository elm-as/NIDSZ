"""TreeSHAP behavioral feature attribution and shortcut-reliance auditing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src.core.config import TESTBED_SHORTCUT_COLUMNS
from src.core.contracts import ShapAuditSummary, ShapFeatureImpact

_PROTOCOL_FEATURES = {"proto", "service", "trans_depth", "is_ftp_login"}
_TIMING_FEATURES = {
    "dur",
    "rate",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "tcprtt",
    "synack",
    "ackdat",
}


@dataclass(frozen=True)
class ShapComparisonInput:
    """Fitted LightGBM estimators and test frames for Full vs Ablated TreeSHAP."""

    full_model: object
    full_x_test: pd.DataFrame
    ablated_model: object
    ablated_x_test: pd.DataFrame
    sample_size: int
    random_seed: int


def _classify_feature_group(feature_name: str) -> str:
    if feature_name in TESTBED_SHORTCUT_COLUMNS:
        return "TTL / Testbed Shortcut"
    if feature_name in _PROTOCOL_FEATURES:
        return "Protocol / Service"
    if feature_name in _TIMING_FEATURES:
        return "Timing / Jitter / RTT"
    return "Volumetric / Packet"


def _extract_positive_shap_matrix(raw_values: object) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.asarray(raw_values[-1], dtype=np.float64)
    arr = np.asarray(raw_values, dtype=np.float64)
    if arr.ndim == 3:
        return arr[:, :, -1]
    return arr


def _compute_single_model_shap(
    model: object, x_sample: pd.DataFrame
) -> tuple[ShapFeatureImpact, ...]:
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(x_sample)
    shap_matrix = _extract_positive_shap_matrix(raw_shap)
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    impacts = [
        ShapFeatureImpact(
            feature_name=str(col),
            feature_group=_classify_feature_group(str(col)),
            mean_abs_shap=float(val),
        )
        for col, val in zip(x_sample.columns, mean_abs)
    ]
    impacts.sort(key=lambda item: item.mean_abs_shap, reverse=True)
    return tuple(impacts)


def compute_shap_behavioral_audit(
    comparison: ShapComparisonInput,
) -> ShapAuditSummary:
    """Compute TreeSHAP rankings for Full vs Shortcut-Ablated LightGBM models."""
    rng = np.random.default_rng(comparison.random_seed)
    n_rows = min(comparison.sample_size, len(comparison.full_x_test))
    sample_idx = rng.choice(len(comparison.full_x_test), size=n_rows, replace=False)

    full_impacts = _compute_single_model_shap(
        comparison.full_model, comparison.full_x_test.iloc[sample_idx]
    )
    ablated_impacts = _compute_single_model_shap(
        comparison.ablated_model, comparison.ablated_x_test.iloc[sample_idx]
    )

    total_full = sum(item.mean_abs_shap for item in full_impacts)
    shortcut_sum = sum(
        item.mean_abs_shap
        for item in full_impacts
        if item.feature_group == "TTL / Testbed Shortcut"
    )
    shortcut_share = float(shortcut_sum / total_full) if total_full > 0 else 0.0
    return ShapAuditSummary(
        sample_size=int(n_rows),
        shortcut_shap_share=shortcut_share,
        full_top_features=full_impacts[:15],
        ablated_top_features=ablated_impacts[:15],
    )
