"""Leak-free out-of-fold probability calibration (Platt & Isotonic) on train split."""

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.core.contracts import FlowFeatureMatrix
from src.modeling.estimator_factory import EstimatorSpec, build_supervised_estimator


@dataclass(frozen=True)
class CalibratedPredictionBundle:
    """Fitted full-train model plus raw, Platt, and Isotonic holdout probabilities."""

    fitted_model: object
    oof_raw_train: np.ndarray
    test_raw: np.ndarray
    test_platt: np.ndarray
    test_isotonic: np.ndarray


def _clip_for_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _fit_platt_calibrator(
    oof_probs: np.ndarray, y_train: np.ndarray, test_probs: np.ndarray
) -> np.ndarray:
    calibrator = LogisticRegression(C=10.0, max_iter=200)
    calibrator.fit(_clip_for_logit(oof_probs), y_train)
    return np.asarray(
        calibrator.predict_proba(_clip_for_logit(test_probs))[:, 1], dtype=np.float64
    )


def _fit_isotonic_calibrator(
    oof_probs: np.ndarray, y_train: np.ndarray, test_probs: np.ndarray
) -> np.ndarray:
    calibrator = IsotonicRegression(
        y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip"
    )
    calibrator.fit(oof_probs, y_train)
    return np.asarray(calibrator.predict(test_probs), dtype=np.float64)


def fit_and_calibrate_on_train_oof(
    matrix: FlowFeatureMatrix, spec: EstimatorSpec, n_folds: int
) -> CalibratedPredictionBundle:
    """Fit OOF calibrators strictly on train split and score official holdout test."""
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=spec.random_seed
    )
    oof_raw = np.zeros(len(matrix.y_train), dtype=np.float64)

    for train_idx, val_idx in splitter.split(matrix.x_train, matrix.y_train):
        fold_model = build_supervised_estimator(spec)
        fold_model.fit(
            matrix.x_train.iloc[train_idx], matrix.y_train[train_idx]
        )
        oof_raw[val_idx] = fold_model.predict_proba(matrix.x_train.iloc[val_idx])[:, 1]

    final_model = build_supervised_estimator(spec)
    final_model.fit(matrix.x_train, matrix.y_train)
    test_raw = np.asarray(
        final_model.predict_proba(matrix.x_test)[:, 1], dtype=np.float64
    )
    test_platt = _fit_platt_calibrator(oof_raw, matrix.y_train, test_raw)
    test_isotonic = _fit_isotonic_calibrator(oof_raw, matrix.y_train, test_raw)

    return CalibratedPredictionBundle(
        fitted_model=final_model,
        oof_raw_train=oof_raw,
        test_raw=test_raw,
        test_platt=test_platt,
        test_isotonic=test_isotonic,
    )
