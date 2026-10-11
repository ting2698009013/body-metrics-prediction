import pandas as pd
import pytest

from body_soft_sensor.features import FEATURE_SETS, add_derived_features


def test_engineered_ratios_and_interactions() -> None:
    data = pd.DataFrame(
        {
            "sex_code": [1.0, 2.0],
            "age_years": [20.0, 30.0],
            "height_cm": [200.0, 160.0],
            "weight_kg": [100.0, 64.0],
            "bmi": [25.0, 25.0],
            "waist_circumference_cm": [100.0, 80.0],
            "arm_circumference_cm": [40.0, 32.0],
            "upper_arm_length_cm": [40.0, 32.0],
        }
    )

    result = add_derived_features(data)

    assert result["female_indicator"].tolist() == [0.0, 1.0]
    assert result["waist_to_height_ratio"].tolist() == pytest.approx([0.5, 0.5])
    assert result["arm_to_height_ratio"].tolist() == pytest.approx([0.2, 0.2])
    assert result["bmi_female_interaction"].tolist() == pytest.approx([0.0, 25.0])


def test_feature_sets_are_incremental() -> None:
    names = list(FEATURE_SETS)
    for previous, current in zip(names, names[1:]):
        assert set(FEATURE_SETS[previous]).issubset(FEATURE_SETS[current])

