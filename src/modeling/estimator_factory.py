"""Estimator factory for supervised NIDS models and unsupervised IsolationForest."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class EstimatorSpec:
    """Parameters defining which model to instantiate and its column layout."""

    model_name: str
    feature_names: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    random_seed: int


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


def _build_tabular_preprocessor(spec: EstimatorSpec) -> ColumnTransformer:
    cat_cols = list(spec.categorical_columns)
    num_cols = [c for c in spec.feature_names if c not in spec.categorical_columns]
    num_pipe = Pipeline(
        [
            ("log1p", FunctionTransformer(_signed_log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = [("num", num_pipe, num_cols)]
    if cat_cols:
        cat_enc = OneHotEncoder(
            handle_unknown="ignore", min_frequency=50, sparse_output=False
        )
        transformers.append(("cat", cat_enc, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


class NormalOnlyIsolationDetector:
    """Unsupervised IsolationForest trained strictly on Normal (label == 0) flows."""

    def __init__(self, spec: EstimatorSpec) -> None:
        self._preprocessor = _build_tabular_preprocessor(spec)
        self._forest = IsolationForest(
            n_estimators=200,
            max_samples=512,
            contamination="auto",
            random_state=spec.random_seed,
            n_jobs=-1,
        )
        self._normal_median: float = 0.0
        self._normal_scale: float = 1.0

    def fit(self, x_train: pd.DataFrame, y_train: np.ndarray) -> "NormalOnlyIsolationDetector":
        """Fit preprocessing and IsolationForest exclusively on Normal training flows."""
        normal_mask = np.asarray(y_train, dtype=np.int32) == 0
        if int(np.sum(normal_mask)) == 0:
            raise ValueError("IsolationForest requires at least one Normal training flow.")

        x_normal = x_train.loc[normal_mask]
        transformed = self._preprocessor.fit_transform(x_normal)
        self._forest.fit(transformed)
        raw_scores = -self._forest.decision_function(transformed)
        q25, q50, q75 = np.percentile(raw_scores, [25.0, 50.0, 75.0])
        self._normal_median = float(q50)
        self._normal_scale = float(max(q75 - q25, 1e-4))
        return self

    def predict_anomaly_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        """Score flows into [0, 1] using only Normal-calibrated anomaly scaling."""
        transformed = self._preprocessor.transform(x_frame)
        raw_scores = -self._forest.decision_function(transformed)
        z_scores = (raw_scores - self._normal_median) / self._normal_scale
        return np.asarray(expit(z_scores), dtype=np.float64)


def build_supervised_estimator(spec: EstimatorSpec) -> object:
    """Instantiate a configured supervised NIDS estimator matching `spec`."""
    if spec.model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.85,
            subsample=0.85,
            subsample_freq=1,
            random_state=spec.random_seed,
            n_jobs=-1,
            verbose=-1,
        )
    if spec.model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            categorical_features="from_dtype",
            random_state=spec.random_seed,
        )
    if spec.model_name == "logistic_regression":
        preprocessor = _build_tabular_preprocessor(spec)
        return Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0, max_iter=500, random_state=spec.random_seed
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown supervised model name: {spec.model_name}")


def build_unsupervised_detector(spec: EstimatorSpec) -> NormalOnlyIsolationDetector:
    """Instantiate an unsupervised Normal-only IsolationForest detector."""
    return NormalOnlyIsolationDetector(spec)
