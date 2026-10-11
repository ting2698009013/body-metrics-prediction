"""Reusable preprocessing and uncertainty helpers for v0.1 models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RAW_MODEL_INPUTS = [
    "sex_code",
    "age_years",
    "height_cm",
    "weight_kg",
    "waist_circumference_cm",
]

MODEL_FEATURES = [
    "female_indicator",
    "age_years",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_circumference_cm",
]


class ProductFeatureBuilder(BaseEstimator, TransformerMixin):
    """Turn the five user-facing inputs into the locked F2 feature set."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "ProductFeatureBuilder":
        self._validate(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        result = pd.DataFrame(index=X.index)
        result["female_indicator"] = X["sex_code"].eq(2).astype(float)
        result["age_years"] = X["age_years"].astype(float)
        result["height_cm"] = X["height_cm"].astype(float)
        result["weight_kg"] = X["weight_kg"].astype(float)
        result["bmi"] = result["weight_kg"] / (result["height_cm"] / 100.0) ** 2
        result["waist_circumference_cm"] = X["waist_circumference_cm"].astype(float)
        return result[MODEL_FEATURES]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(MODEL_FEATURES, dtype=object)

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ProductFeatureBuilder requires a pandas DataFrame")
        missing = [column for column in RAW_MODEL_INPUTS if column not in X.columns]
        if missing:
            raise KeyError(f"Missing raw model inputs: {missing}")
        values = X[RAW_MODEL_INPUTS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Raw model inputs must be finite")
        if (X["height_cm"] <= 0).any():
            raise ValueError("height_cm must be positive")
        if not X["sex_code"].isin([1, 2]).all():
            raise ValueError("sex_code must be 1 or 2")


def finite_sample_absolute_quantile(
    absolute_residuals: np.ndarray | pd.Series,
    coverage: float = 0.90,
) -> float:
    """Return the finite-sample corrected absolute-residual quantile."""

    residuals = np.asarray(absolute_residuals, dtype=float)
    if residuals.ndim != 1 or len(residuals) == 0:
        raise ValueError("absolute_residuals must be a non-empty one-dimensional array")
    if not np.isfinite(residuals).all() or (residuals < 0).any():
        raise ValueError("absolute_residuals must be finite and non-negative")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    order = min(len(residuals), math.ceil((len(residuals) + 1) * coverage))
    return float(np.partition(residuals, order - 1)[order - 1])

