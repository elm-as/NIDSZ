"""Zero-Day Leave-One-Attack-Family-Out (LOAO) generalization benchmark."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.core.config import ATTACK_FAMILY_COLUMN, NORMAL_FAMILY_LABEL, NidszConfig
from src.core.contracts import LoaoFamilyResult
from src.evaluation.soc_operating_points import evaluate_soc_budgets
from src.features.flow_preprocessor import (
    build_flow_feature_matrix,
    build_loao_partition,
)
from src.modeling.estimator_factory import (
    EstimatorSpec,
    NormalOnlyIsolationDetector,
    build_supervised_estimator,
)


@dataclass(frozen=True)
class LoaoBenchmarkInput:
    """Inputs needed to execute the 7-family zero-day LOAO benchmark."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    iso_detector_full: NormalOnlyIsolationDetector
    config: NidszConfig


def _fit_and_score_loao_lgbm(
    loao_train: pd.DataFrame, loao_test: pd.DataFrame, feature_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    matrix = build_flow_feature_matrix(loao_train, loao_test, feature_mode)
    spec = EstimatorSpec(
        model_name="lightgbm",
        feature_names=matrix.feature_names,
        categorical_columns=matrix.categorical_columns,
        random_seed=42,
    )
    model = build_supervised_estimator(spec)
    model.fit(matrix.x_train, matrix.y_train)
    probs = np.asarray(model.predict_proba(matrix.x_test)[:, 1], dtype=np.float64)
    return matrix.y_test, probs


def _score_single_family(
    bundle: LoaoBenchmarkInput, family_name: str
) -> LoaoFamilyResult:
    loao_train, loao_test = build_loao_partition(
        bundle.train_df, bundle.test_df, family_name
    )
    budgets = bundle.config.soc_fpr_budgets

    y_test, full_probs = _fit_and_score_loao_lgbm(
        loao_train, loao_test, "full_flow"
    )
    _, ablated_probs = _fit_and_score_loao_lgbm(
        loao_train, loao_test, "shortcut_ablated"
    )
    full_matrix_for_iso = build_flow_feature_matrix(
        bundle.train_df, loao_test, "full_flow"
    )
    iso_scores = bundle.iso_detector_full.predict_anomaly_score(
        full_matrix_for_iso.x_test
    )

    full_soc = evaluate_soc_budgets(y_test, full_probs, budgets)
    abl_soc = evaluate_soc_budgets(y_test, ablated_probs, budgets)
    iso_soc = evaluate_soc_budgets(y_test, iso_scores, budgets)

    family_col = loao_test[ATTACK_FAMILY_COLUMN].astype(str)
    n_attack = int((family_col == family_name).sum())
    n_normal = int((family_col == NORMAL_FAMILY_LABEL).sum())
    return LoaoFamilyResult(
        family_name=family_name,
        test_attack_count=n_attack,
        test_normal_count=n_normal,
        full_roc_auc=float(roc_auc_score(y_test, full_probs)),
        ablated_roc_auc=float(roc_auc_score(y_test, ablated_probs)),
        iso_forest_roc_auc=float(roc_auc_score(y_test, iso_scores)),
        full_recall_at_001=full_soc[0].recall_tpr,
        full_recall_at_005=full_soc[1].recall_tpr,
        full_recall_at_010=full_soc[2].recall_tpr,
        ablated_recall_at_001=abl_soc[0].recall_tpr,
        ablated_recall_at_005=abl_soc[1].recall_tpr,
        ablated_recall_at_010=abl_soc[2].recall_tpr,
        iso_recall_at_001=iso_soc[0].recall_tpr,
        iso_recall_at_005=iso_soc[1].recall_tpr,
        iso_recall_at_010=iso_soc[2].recall_tpr,
    )


def run_zero_day_loao_benchmark(
    bundle: LoaoBenchmarkInput,
) -> tuple[LoaoFamilyResult, ...]:
    """Evaluate zero-day detection across all configured unseen attack families."""
    return tuple(
        _score_single_family(bundle, family)
        for family in bundle.config.loao_families
    )
