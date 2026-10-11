import numpy as np
import pandas as pd
import pytest

from body_soft_sensor.modeling import (
    MODEL_FEATURES,
    ProductFeatureBuilder,
    finite_sample_absolute_quantile,
)


def test_product_feature_builder_recomputes_bmi() -> None:
    raw = pd.DataFrame(
        {
            "sex_code": [1, 2],
            "age_years": [25.0, 30.0],
            "height_cm": [200.0, 160.0],
            "weight_kg": [100.0, 64.0],
            "waist_circumference_cm": [90.0, 75.0],
        }
    )
    transformed = ProductFeatureBuilder().fit_transform(raw)

    assert transformed.columns.tolist() == MODEL_FEATURES
    assert transformed["female_indicator"].tolist() == [0.0, 1.0]
    assert transformed["bmi"].tolist() == pytest.approx([25.0, 25.0])


def test_product_feature_builder_rejects_invalid_sex_code() -> None:
    raw = pd.DataFrame(
        {
            "sex_code": [3],
            "age_years": [25.0],
            "height_cm": [175.0],
            "weight_kg": [70.0],
            "waist_circumference_cm": [80.0],
        }
    )
    with pytest.raises(ValueError, match="sex_code"):
        ProductFeatureBuilder().fit_transform(raw)


def test_finite_sample_absolute_quantile_is_conservative() -> None:
    residuals = np.arange(1.0, 11.0)
    assert finite_sample_absolute_quantile(residuals, coverage=0.80) == 9.0

