"""Official holdout benchmark runner comparing Full vs Shortcut-Ablated models."""

from dataclasses import dataclass

import numpy as np

from src.core.config import NidszConfig
from src.core.contracts import (
    FlowFeatureMatrix,
    LatencyBenchmark,
    ModelHoldoutArtifact,
)
from src.evaluation.calibration_and_metrics import compute_holdout_metrics
from src.evaluation.latency_profiler import (
    LatencyProfileRequest,
    benchmark_batch_inference,
)
from src.evaluation.soc_operating_points import (
    evaluate_soc_budgets,
    sample_log_spaced_roc_curve,
)
from src.modeling.estimator_factory import (
    EstimatorSpec,
    NormalOnlyIsolationDetector,
    build_unsupervised_detector,
)
from src.modeling.oof_calibrator import fit_and_calibrate_on_train_oof


@dataclass(frozen=True)
class HoldoutSuiteOutput:
    """Artifacts and fitted key models from one feature-mode holdout suite."""

    feature_mode: str
    artifacts: tuple[ModelHoldoutArtifact, ...]
    fitted_lightgbm: object
    fitted_isolation_forest: NormalOnlyIsolationDetector
    best_model_name: str
    best_calibration_method: str


@dataclass(frozen=True)
class _ArtifactAssemblyInput:
    matrix: FlowFeatureMatrix
    model_name: str
    calibration_method: str
    probabilities: np.ndarray
    latency: LatencyBenchmark
    config: NidszConfig


def _assemble_artifact(item: _ArtifactAssemblyInput) -> ModelHoldoutArtifact:
    metrics, bins = compute_holdout_metrics(
        item.matrix.y_test, item.probabilities, num_bins=item.config.ece_bins
    )
    soc_points = evaluate_soc_budgets(
        item.matrix.y_test, item.probabilities, item.config.soc_fpr_budgets
    )
    roc_fpr, roc_tpr = sample_log_spaced_roc_curve(
        item.matrix.y_test, item.probabilities
    )
    return ModelHoldoutArtifact(
        feature_mode=item.matrix.feature_mode,
        model_name=item.model_name,
        calibration_method=item.calibration_method,
        metrics=metrics,
        soc_operating_points=soc_points,
        reliability_bins_10=bins,
        latency=item.latency,
        roc_curve_fpr=roc_fpr,
        roc_curve_tpr=roc_tpr,
    )


def _evaluate_supervised_model(
    matrix: FlowFeatureMatrix, model_name: str, config: NidszConfig
) -> tuple[list[ModelHoldoutArtifact], object]:
    spec = EstimatorSpec(
        model_name=model_name,
        feature_names=matrix.feature_names,
        categorical_columns=matrix.categorical_columns,
        random_seed=config.random_seed,
    )
    bundle = fit_and_calibrate_on_train_oof(matrix, spec, config.calibration_folds)
    latency = benchmark_batch_inference(
        lambda frame: bundle.fitted_model.predict_proba(frame)[:, 1],
        matrix.x_test,
        LatencyProfileRequest(
            model_name=model_name,
            feature_mode=matrix.feature_mode,
            batch_size=config.latency_batch_size,
            repeats=config.latency_repeats,
        ),
    )
    variants = (
        ("raw", bundle.test_raw),
        ("platt", bundle.test_platt),
        ("isotonic", bundle.test_isotonic),
    )
    artifacts = [
        _assemble_artifact(
            _ArtifactAssemblyInput(
                matrix=matrix,
                model_name=model_name,
                calibration_method=cal_name,
                probabilities=probs,
                latency=latency,
                config=config,
            )
        )
        for cal_name, probs in variants
    ]
    return artifacts, bundle.fitted_model


def _evaluate_isolation_forest(
    matrix: FlowFeatureMatrix, config: NidszConfig
) -> tuple[ModelHoldoutArtifact, NormalOnlyIsolationDetector]:
    spec = EstimatorSpec(
        model_name="isolation_forest",
        feature_names=matrix.feature_names,
        categorical_columns=matrix.categorical_columns,
        random_seed=config.random_seed,
    )
    detector = build_unsupervised_detector(spec)
    detector.fit(matrix.x_train, matrix.y_train)
    scores = detector.predict_anomaly_score(matrix.x_test)
    latency = benchmark_batch_inference(
        detector.predict_anomaly_score,
        matrix.x_test,
        LatencyProfileRequest(
            model_name="isolation_forest",
            feature_mode=matrix.feature_mode,
            batch_size=config.latency_batch_size,
            repeats=config.latency_repeats,
        ),
    )
    artifact = _assemble_artifact(
        _ArtifactAssemblyInput(
            matrix=matrix,
            model_name="isolation_forest",
            calibration_method="unsupervised_normal_only",
            probabilities=scores,
            latency=latency,
            config=config,
        )
    )
    return artifact, detector


def _select_best_artifact(
    artifacts: list[ModelHoldoutArtifact],
) -> ModelHoldoutArtifact:
    # Prioritize Recall @ strict 0.1% FPR, breaking ties by ROC-AUC and low Brier.
    def _rank_key(art: ModelHoldoutArtifact) -> tuple[float, float, float]:
        strict_recall = art.soc_operating_points[0].recall_tpr
        return (strict_recall, art.metrics.roc_auc, -art.metrics.brier_score)

    return max(artifacts, key=_rank_key)


def run_holdout_benchmark_suite(
    matrix: FlowFeatureMatrix, config: NidszConfig
) -> HoldoutSuiteOutput:
    """Run all supervised models + Normal-only IsolationForest on official holdout."""
    all_artifacts: list[ModelHoldoutArtifact] = []
    fitted_lgb: object = None

    for model_name in config.supervised_models:
        model_artifacts, fitted = _evaluate_supervised_model(
            matrix, model_name, config
        )
        all_artifacts.extend(model_artifacts)
        if model_name == "lightgbm":
            fitted_lgb = fitted

    iso_artifact, iso_detector = _evaluate_isolation_forest(matrix, config)
    all_artifacts.append(iso_artifact)
    best_art = _select_best_artifact(all_artifacts)

    return HoldoutSuiteOutput(
        feature_mode=matrix.feature_mode,
        artifacts=tuple(all_artifacts),
        fitted_lightgbm=fitted_lgb,
        fitted_isolation_forest=iso_detector,
        best_model_name=best_art.model_name,
        best_calibration_method=best_art.calibration_method,
    )
